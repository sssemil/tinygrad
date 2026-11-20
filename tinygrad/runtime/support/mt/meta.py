from __future__ import annotations
import subprocess, tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from tinygrad.helpers import getenv, wait_cond
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.support.mt.compute import FW_DEVVA
if TYPE_CHECKING: from tinygrad.runtime.support.mt.s30 import MTS30Firmware

META_AS = getenv("META_AS", "metag-elf-as")

class MetaPort:
  def __init__(self, fw:MTS30Firmware): self.fw = fw
  def ready(self):
    wait_cond(lambda: self.fw._read(0xa18, 4) & 0x05000000, value=0x05000000, timeout_ms=1000, msg="META port busy")
  def read(self, addr:int) -> int:
    self.ready()
    self.fw._write(0xa10, addr|1, 4)
    self.fw._read(0xa10, 4)
    self.ready()
    return self.fw._read(0xa00, 4)
  def write(self, addr:int, value:int):
    self.ready()
    self.fw._write(0xa10, addr, 4)
    self.fw._read(0xa10, 4)
    self.fw._write(0xa08, value, 4)
    self.fw._read(0xa08, 4)
  def start(self, mailbox_size:int=0x1000):
    self.write(0x04830030, 4)  # select the control port for thread register access
    # META segment 1 covers 0x20000000..0x3fffffff. Its global alias at 0xb0000000 bypasses META's data cache.
    # Set the SLC uncached bit in the output address so the USB BAR and META see each other's writes.
    mailbox = FW_DEVVA + 0x1000 | 1<<40
    for i, word in enumerate((0x30000f02, mailbox_size-0x1000, mailbox & 0xffffffff, mailbox >> 32)): self.write(0x04850010+i*4, word)
    self.write(0x0480fff0, 0xc0000000)  # boot segment in the global address range
    self.write(0x0480fff8, 5)  # transfer to thread 0 PC
    wait_cond(lambda: self.read(0x0480fff8) >> 31, timeout_ms=1000, msg="META PC transfer")
    self.write(0x048000e8, 0)  # full-width instructions
    self.write(0x04800000, 1)  # enable thread 0

def assemble(src:Path) -> bytes:
  with tempfile.NamedTemporaryFile() as obj:
    subprocess.run([META_AS, "-o", obj.name, src], check=True)
    _, sections, relocs = elf_loader(obj.read())
  if relocs: raise ValueError("META dispatch must be position-independent and fully assembled")
  return next(s.content for s in sections if s.name == ".text")
