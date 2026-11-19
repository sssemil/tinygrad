"""Run a freshly compiled S30 shader with all GPU mappings in Chestnut SRAM. --cold cycles the downstream slot."""
import argparse, secrets, struct, time
from tinygrad import Context, UOp, dtypes
from tinygrad.codegen import do_to_program
from tinygrad.helpers import Target, ceildiv, round_up
from tinygrad.renderer import TinyELF
from tinygrad.renderer.usc import MTUSCRenderer
from tinygrad.runtime.support.mt.compute import FW_DEVVA
from tinygrad.runtime.support.mt.mmu import MTMMU4, MT_PTE_READ_ONLY, MT_PTE_READ_WRITE
from tinygrad.runtime.support.mt.mtdev import MTDev, MT_IPC_RING_SIZE, MT_S30_SRAM_OFFSET
from tinygrad.runtime.support.mt.s30 import DEFAULT_S30_FIRMWARE, DIRECT_CDM, PDS_BASE, USC_BASE, MTS30Firmware, s30_cdm_stream, s30_dispatch, s30_grid, s30_pds, s30_shader
from tinygrad.runtime.support.usb import HALF, SLOT
from tinygrad.uop.ops import KernelInfo

# GPU virtual addresses for streamed input, resident activations, and the completion word visible at USB XDATA 0xb800.
INPUT, ACTIVATIONS, RESULT = 0x7100000000, 0x7200000000, 0x7300000000
INPUT_SIZE, ACTIVATION_SIZE = 0x28000, 0x38000

class ChestnutBoot(MTS30Firmware):
  context_pa = (1<<39) | 0x240000  # upper half of the bridge SRAM on the Sudi outbound PCIe bus
  def _map_context(self, size:int):
    self.mem, self.code_end = bytearray(0x20000), 0
    mmu = MTMMU4(self.context_pa+0x10000, size=0x10000)
    # The first 160 KiB holds each kernel's inputs. Join the two free runs around the context into one activation heap.
    base = self.context_pa-HALF
    mmu.map_range(INPUT, base, INPUT_SIZE, MT_PTE_READ_ONLY)
    mmu.map_range(ACTIVATIONS, base+INPUT_SIZE, HALF-INPUT_SIZE, MT_PTE_READ_WRITE)
    mmu.map_range(ACTIVATIONS+HALF-INPUT_SIZE, self.context_pa+len(self.mem), HALF-len(self.mem), MT_PTE_READ_WRITE)
    mmu.map(RESULT, (1<<39) | 0x822000, MT_PTE_READ_WRITE)
    for va in (USC_BASE, PDS_BASE): mmu.map_range(va, self.context_pa+0x8000, 0x8000, MT_PTE_READ_ONLY)
    return memoryview(self.mem)[:0x8000], mmu

  def prepare(self, program:UOp, addresses:list[int]) -> tuple[bytearray, bytearray]:
    text, params, temps, attrs, local, layout = s30_shader(TinyELF.from_program(program))
    for (off, fmt), addr in zip(layout, addresses, strict=True): struct.pack_into(fmt, params, off, addr)
    usc = self.code_end
    pds = round_up(usc+len(text)+32, 64)  # include the USC prefetch guard
    loader = s30_pds(FW_DEVVA+0x3000, len(params)//4, usc, temps, bool(local))
    self.code_end = round_up(pds+len(loader), 64)
    if self.code_end > 0x8000 or len(params) > 0x1000: raise MemoryError("S30 program exceeds Chestnut SRAM")
    self.mem[0x8000+usc:0x8000+usc+len(text)] = text
    self.mem[0x8000+pds:0x8000+pds+len(loader)] = loader
    shared = ceildiv(len(params), 64)
    grid = s30_grid(program.arg.global_size, program.arg.local_size, shared+ceildiv(local, 16))
    return params, s30_cdm_stream(s30_dispatch(pds, shared, attrs)+grid)

  def launch(self, data:bytes|bytearray, params:bytearray, stream:bytearray):
    if len(data) > INPUT_SIZE: raise MemoryError("S30 input exceeds Chestnut SRAM")
    if data: self.dev.pci_dev.usb.scsi_write(data)
    self.mem[0x3000:0x3000+len(params)] = params
    self.mem[DIRECT_CDM:DIRECT_CDM+len(stream)] = stream
    self.upload_mmu()
    self.kick_direct(0.5)

  def result(self) -> int: return int.from_bytes(self.dev.pci_dev.usb.read(0xb800, 4), "little")

  def upload_mmu(self, full:bool=False):
    self.mem[0x10000:0x10000+self.mmu.next] = self.mmu.root[:self.mmu.next]
    self.dev.pci_dev.usb.scsi_write(self.mem, slot_start=HALF//SLOT)

def cold_link(mt:MTDev) -> float:
  pci, bridge = mt.pci_dev, mt.pci_dev.usb
  bar0 = pci.bar_info(0)[0]
  bridge.set_pcie_power(False)
  try:
    time.sleep(0.3)
    if bridge.read(0xb450, 1)[0] == 0x78: raise RuntimeError("downstream link did not drop")
    # Same lane order as ed4e39b7 pcie_power_on, without the three 100 ms stable-link delays.
    start = time.monotonic()
    bridge.write(0xc656, bytes([bridge.read(0xc656, 1)[0] | 0x20]))
    for reg,val in ((0xca81,0x0e), (0xca06,0x21), (0xb431,0x0c), (0xe764,0x1c)): bridge.write(reg, bytes([val]))
    bridge.write(0xc659, bytes([bridge.read(0xc659, 1)[0] | 1]))
    bridge.write(0xb480, b"\x00")
    while bridge.read(0xb450, 1)[0] != 0x78:
      if time.monotonic() - start > 2: raise TimeoutError("downstream PCIe link training")
    # Restore the register aperture and bus mastering. Leave GPU memory at its power-on BAR address.
    pci.write_config(0x10, bar0, 4)
    pci.write_config(4, pci.read_config(4, 2) | 6, 2)
    return start
  except BaseException:
    bridge.set_pcie_power(True)
    raise

def mailbox(mt:MTDev) -> bool:
  header = 0x14
  return all(struct.unpack("<II", mt.regs._read(base+8, 8)) == (MT_IPC_RING_SIZE, base+header)
             for base in (MT_S30_SRAM_OFFSET, MT_S30_SRAM_OFFSET+header+MT_IPC_RING_SIZE))

def run(mt:MTDev, cold:bool):
  fw = ChestnutBoot(mt, DEFAULT_S30_FIRMWARE)
  start = cold_link(mt) if cold else time.monotonic()
  out, inp = (UOp.placeholder((1,), dtypes.uint, slot=i) for i in range(2))
  shader = out[0].store(inp[0]*3+7).sink(arg=KernelInfo(name="s30_sram", opts_to_apply=()))
  renderer = MTUSCRenderer(Target("MT","USC","mp_10"))
  params, stream = fw.prepare(do_to_program(shader, renderer), [RESULT, INPUT])
  with fw.boot_direct():
    assert all(pa & (1<<39) for pa in fw.mmu.pages.values())
    for _ in range(3):
      x = secrets.randbits(28)
      fw.launch(struct.pack("<I", x), params, stream)
      got = fw.result()
      if got != 3*x+7: raise RuntimeError(f"S30 returned {got:#x}, expected {3*x+7:#x}")
      header = mailbox(mt)
      print(f"{1000*(time.monotonic()-start):7.2f} ms  {x:#010x} * 3 + 7 = {got:#010x}  mailbox ready {header}")
      if cold and header: raise RuntimeError("management ring initialized before challenge completed")

  if cold:
    while not mailbox(mt):
      if time.monotonic()-start > 3: raise TimeoutError("management rings never initialized after the challenges")
      time.sleep(0.005)
    print(f"{1000*(time.monotonic()-start):7.2f} ms  management rings ready")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--cold", action="store_true", help="cycle the S30 slot and check all three results before the SMC ring is ready")
  args = parser.parse_args()
  mt = MTDev.open_usb(manage_power=False)
  if not mt.s30: raise RuntimeError("requires a Chestnut S30")
  with Context(CACHELEVEL=0): run(mt, args.cold)
