"""Submit STREAM_TERMINATE to one S30 core using source-built META firmware."""
import struct, time
from pathlib import Path
from extra.usbgpu.s30.boot import MetaPort, assemble
from tinygrad import Device
from tinygrad.runtime.support.mt.compute import FW_BASE, FW_DEVVA, ROOT_BASE
from tinygrad.runtime.support.mt.mmu import MTMMU4, MT_PTE_READ_WRITE
from tinygrad.runtime.support.mt.s30 import CDM_RING_SIZE

if __name__ == "__main__":
  code = assemble(Path(__file__).with_suffix(".s"))
  image = bytearray(0x8000)
  assert 0 < len(code) <= 0x1000
  image[:len(code)] = code
  # Mailbox page, CDM ring, control pages and context storage are all within this mapping.
  ring, control, state, context = (0x4000, 0x5000, 0x6000, 0x7000)
  struct.pack_into('<6Q', image, 0x1008, FW_DEVVA+control, FW_DEVVA+ring, CDM_RING_SIZE, FW_DEVVA+context, FW_DEVVA+state, 1)
  struct.pack_into('<I', image, ring, 0x80000000)
  struct.pack_into('<I', image, control, 4+64)  # write pointer includes the CDM prefetch guard
  fw = Device["MT"].s30_fw
  fw.mmu = MTMMU4(ROOT_BASE)
  fw.mmu.map_range(FW_DEVVA, FW_BASE, len(image), MT_PTE_READ_WRITE)
  with fw._boot(image, master=False):
    MetaPort(fw).start()
    fw.v[0x1000:0x1004] = struct.pack('<I', 1)
    deadline = time.monotonic()+1
    while not fw._read(0x130, 4) & 4:
      if time.monotonic() >= deadline: raise TimeoutError(f"S30 CDM completion, MMU fault {fw._read(0xe160):#x}")
    ack, = struct.unpack('<I', fw.v[0x1004:0x1008])
    rd, = struct.unpack('<I', fw.v[control+8:control+12])
    if (ack, rd, fw._read(0xe160)) != (1, 4, 0): raise RuntimeError(f"Unexpected compute completion: {ack=} {rd=}")
    print(f"S30 CDM consumed {rd} bytes")
