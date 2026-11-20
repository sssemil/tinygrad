"""Boot an open META mailbox on the S30. META_AS and META_OBJCOPY select GNU binutils configured for metag-elf."""
import random, struct, time
from pathlib import Path
from tinygrad import Device
from tinygrad.runtime.support.mt.compute import FW_BASE, FW_DEVVA, ROOT_BASE
from tinygrad.runtime.support.mt.mmu import MTMMU4, MT_PTE_READ_WRITE
from tinygrad.runtime.support.mt.meta import MetaPort, assemble

if __name__ == "__main__":
  code = assemble(Path(__file__).with_name("heartbeat.s"))
  fw = Device["MT"].s30_fw
  # Expose just the generated reset code and one mailbox page to the processor.
  fw.mmu = MTMMU4(ROOT_BASE)
  fw.mmu.map_range(FW_DEVVA, FW_BASE, 0x2000, MT_PTE_READ_WRITE)
  if not 0 < len(code) <= 0x1000: raise ValueError("META program must fit in the boot page")
  with fw._boot(code.ljust(0x2000, b'\0'), master=False, reset_domains=False):
    port = MetaPort(fw)
    port.start()
    for challenge in (*random.sample(range(1<<32), 4), 0xffffffff):
      fw.v[0x1000:0x1004] = struct.pack("<I", challenge)
      deadline = time.monotonic()+1
      while (reply:=struct.unpack("<I", fw.v[0x1004:0x1008])[0]) != (challenge+1) & 0xffffffff:
        if time.monotonic() >= deadline: raise TimeoutError(f"META mailbox {challenge=:08x} {reply=:08x}")
      print(f"{challenge:08x} -> {reply:08x}")
    ids = list(struct.unpack("<2Q", fw.v[0x1008:0x1018]))
    if ids != fw.dev.core_ids: raise RuntimeError(f"META core IDs {ids!r} differ from the PCIe register reads {fw.dev.core_ids!r}")
    print("GPU core IDs", *(hex(i) for i in ids))
