"""Run a freshly compiled S30 shader with all GPU mappings in Chestnut SRAM. --cold cycles the downstream slot."""
import argparse, functools, secrets, struct, time
from tinygrad import Context, UOp, dtypes
from tinygrad.codegen import do_to_program
from tinygrad.helpers import Target, ceildiv, round_up
from tinygrad.renderer import TinyELF
from tinygrad.renderer.usc import MTUSCRenderer
from tinygrad.runtime.support.mt.compute import FW_DEVVA
from tinygrad.runtime.support.mt.mmu import MTMMU4, MT_PTE_READ_ONLY, MT_PTE_READ_WRITE
from tinygrad.runtime.support.mt.mtdev import MTDev, MT_DDR_BAR, MT_IPC_RING_SIZE, MT_S30_ID, MT_S30_SRAM_OFFSET, MT_SYS_BAR, MT_USB_GPU_BUS, MT_VENDOR_ID
from tinygrad.runtime.support.mt.s30 import DIRECT_CDM, PDS_BASE, USC_BASE, MTS30Firmware, s30_direct_stream, s30_dispatch, s30_pds, s30_shader
from tinygrad.runtime.support.mt.s30 import s30_read_signal
from tinygrad.runtime.support.system import USBPCIDevice
from tinygrad.runtime.support.usb import CustomASM24Controller, HALF, SLOT, USB3
from tinygrad.uop.ops import KernelInfo

# GPU virtual addresses for streamed input, resident activations, and the completion word visible at USB XDATA 0xb800.
INPUT, ACTIVATIONS, RESULT = 0x7100000000, 0x7200000000, 0x7300000000
INPUT_SIZE, ACTIVATION_SIZE = 0x28000, 0x38000
PARAMS = 0x5000  # after the dispatch ring and before the compute state at 0x6000

class ChestnutBoot(MTS30Firmware):
  context_pa = (1<<39) | 0x240000  # upper half of the bridge SRAM on the Sudi outbound PCIe bus
  mem_size, code_base, code_size, tables = 0x20000, 0x8000, 0x8000, 0x10000
  def _map_context(self, size:int):
    self.mem, self.code_end, self.dirty_code = bytearray(self.mem_size), 0, False
    mmu = MTMMU4(self.context_pa+self.tables, size=0x10000)
    # The first 160 KiB holds each kernel's inputs. Join the two free runs around the context into one activation heap.
    base = self.context_pa-HALF
    mmu.map_range(INPUT, base, INPUT_SIZE, MT_PTE_READ_ONLY)
    mmu.map_range(ACTIVATIONS, base+INPUT_SIZE, HALF-INPUT_SIZE, MT_PTE_READ_WRITE)
    mmu.map_range(ACTIVATIONS+HALF-INPUT_SIZE, self.context_pa+len(self.mem), HALF-len(self.mem), MT_PTE_READ_WRITE)
    mmu.map(RESULT, (1<<39) | 0x822000, MT_PTE_READ_WRITE)
    for va in (USC_BASE, PDS_BASE): mmu.map_range(va, self.context_pa+self.code_base, self.code_size, MT_PTE_READ_ONLY)
    return memoryview(self.mem)[:0x8000], mmu

  def prepare(self, program:UOp, addresses:list[int]) -> tuple[bytearray, bytearray]:
    text, params, temps, attrs, local, layout, split = s30_shader(TinyELF.from_program(program))
    for (off, fmt), addr in zip(layout, addresses, strict=True): struct.pack_into(fmt, params, off, addr)
    usc = self.code_end
    pds = round_up(usc+len(text)+32, 64)  # include the USC prefetch guard
    loader = s30_pds(FW_DEVVA+PARAMS, len(params)//4, usc, temps, bool(local))
    shared = ceildiv(len(params), 64)
    dispatches = (s30_dispatch(pds, shared, attrs),)
    if split:
      stride = round_up(len(params), 32)
      dispatches += (s30_dispatch(pds+len(loader), shared, attrs),)
      loader += s30_pds(FW_DEVVA+PARAMS+stride, len(params)//4, usc, temps, bool(local))
      params.extend(bytes(stride-len(params)))
    self.code_end = round_up(pds+len(loader), 64)
    if self.code_end > self.code_size or len(params)*(1+bool(split)) > 0x1000: raise MemoryError("S30 program exceeds Chestnut SRAM")
    self.mem[self.code_base+usc:self.code_base+usc+len(text)] = text
    self.mem[self.code_base+pds:self.code_base+pds+len(loader)] = loader
    self.dirty_code = True
    return s30_direct_stream(dispatches, program.arg.global_size, program.arg.local_size, shared+ceildiv(local, 16), split, params, 8*len(layout))

  def stage(self, params:bytearray, stream:bytearray):
    self.mem[PARAMS:PARAMS+len(params)] = params
    self.mem[DIRECT_CDM:DIRECT_CDM+len(stream)] = stream
    if self.dirty_code or self.mmu.dirty_ranges: self.upload_mmu()
    else: self.dev.pci_dev.usb.scsi_write(self.mem[DIRECT_CDM:PARAMS+len(params)], slot_start=(HALF+DIRECT_CDM)//SLOT)

  def launch(self, data:bytes|bytearray, params:bytearray, stream:bytearray):
    if len(data) > INPUT_SIZE: raise MemoryError("S30 input exceeds Chestnut SRAM")
    if data: self.dev.pci_dev.usb.scsi_write(data)
    self.stage(params, stream)
    self.kick_direct(0.5)

  @functools.cached_property
  def read_signal(self):
    return self.prepare(do_to_program(s30_read_signal(), MTUSCRenderer(Target("MT","USC","mp_10"))), [RESULT])

  def read_output(self, size:int) -> bytes:
    if not 0 < size <= HALF: raise ValueError("readback must fit in the lower SRAM half")
    self.stage(*self.read_signal)
    data = bytearray(size)
    self.dev.pci_dev.usb.scsi_read(memoryview(data), self.start_direct)
    # Drain the payload before polling core events, which also uses the bulk IN endpoint.
    self.wait_direct(0.5)
    return bytes(data)

  def result(self) -> int: return int.from_bytes(self.dev.pci_dev.usb.read(0xb800, 4), "little")

  def upload_mmu(self, full:bool=False):
    self.mem[self.tables:self.tables+self.mmu.next] = self.mmu.root[:self.mmu.next]
    self.dev.pci_dev.usb.scsi_write(self.mem, slot_start=HALF//SLOT)
    self.mmu.dirty_ranges.clear()
    self.dirty_code = False

class ColdSlot(USBPCIDevice):
  def _setup_pcie(self, usb:USB3, vram_bars:dict[int,int]|None):
    bridge = self.usb = CustomASM24Controller(usb, train_link=False)
    identity = MT_S30_ID<<16 | MT_VENDOR_ID
    up = bridge.read(0xb450, 1)[0] == 0x78
    if up and self.read_config(0, 4) != identity: raise RuntimeError("expected an S30 on Chestnut")
    bridge.set_pcie_power(False)
    try:
      time.sleep(0.3)
      if bridge.read(0xb450, 1)[0] == 0x78: raise RuntimeError("downstream link did not drop")
      # ed4e39b7 pcie_power_on, including the bank-1 PHY routing bit. Start as soon as the link trains.
      bridge.write(0xb403, b"\x01")
      bridge.write(0xb430, bytes([bridge.read(0xb430, 1)[0] & ~1]))
      route = usb.control_read(0xe4, 1, value=0x6025, index=0x100)[0]
      usb.control_write(0xe5, value=0x6025, index=0x180 | route)
      self.start = time.monotonic()
      bridge.write(0xc656, bytes([bridge.read(0xc656, 1)[0] | 0x20]))
      for reg,val in ((0xca81,0x0e), (0xca06,0x21), (0xb431,0x0c), (0xe764,0x1c)): bridge.write(reg, bytes([val]))
      bridge.write(0xc659, bytes([bridge.read(0xc659, 1)[0] | 1]))
      bridge.write(0xb298, bytes([bridge.read(0xb298, 1)[0] | 0x10]))
      bridge.write(0xb480, b"\x00")
      while bridge.read(0xb450, 1)[0] != 0x78:
        if time.monotonic()-self.start > 2: raise TimeoutError("downstream PCIe link training")
      if self.read_config(0, 4) != identity: raise RuntimeError("expected an S30 on Chestnut")
      # Size and assign BAR0 with memory decoding disabled. No VRAM aperture is needed.
      command = self.read_config(4, 2)
      self.write_config(4, command & ~2, 2)
      self.write_config(0x10, 0xffffffff, 4)
      mask = self.read_config(0x10, 4)
      if mask != 0xfe000000: raise RuntimeError(f"unexpected S30 BAR0 size mask {mask:#x}")
      bar0 = 0x10000000
      self.write_config(0x10, bar0, 4)
      self.write_config(4, command | 6, 2)
      self._bar_info = {MT_SYS_BAR:(bar0, 1<<25), MT_DDR_BAR:(0, 0)}
    except BaseException:
      bridge.set_pcie_power(up)
      raise

def open_cold() -> tuple[MTDev, float]:
  (dev, bus), = USB3.list_devices(0x3801, 1)
  slot = ColdSlot("MT", dev, bus, gpu_bus=MT_USB_GPU_BUS)
  return MTDev(slot, manage_power=False), slot.start

def mailbox(mt:MTDev) -> bool:
  header = 0x14
  return all(struct.unpack("<II", mt.regs._read(base+8, 8)) == (MT_IPC_RING_SIZE, base+header)
             for base in (MT_S30_SRAM_OFFSET, MT_S30_SRAM_OFFSET+header+MT_IPC_RING_SIZE))

def run(mt:MTDev, start:float|None):
  fw = ChestnutBoot(mt)
  cold = start is not None
  if start is None: start = time.monotonic()
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
  with Context(CACHELEVEL=0): run(*(open_cold() if args.cold else (MTDev.open_usb(manage_power=False), None)))
