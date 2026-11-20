from __future__ import annotations
import os, subprocess, tempfile, time
from pathlib import Path
from typing import TYPE_CHECKING
from tinygrad.runtime.support.mt.compute import FW_DEVVA
if TYPE_CHECKING: from tinygrad.runtime.support.mt.s30 import MTS30Firmware

class MetaPort:
  def __init__(self, fw:MTS30Firmware): self.fw = fw
  def ready(self):
    deadline = time.monotonic()+1
    while self.fw._read(0xa18, 4) & 0x05000000 != 0x05000000:
      if time.monotonic() >= deadline: raise TimeoutError("META port busy")
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
  def start(self):
    self.write(0x04830030, 4)  # select the control port for thread register access
    # META segment 1 covers 0x20000000..0x3fffffff. Its global alias at 0xb0000000 bypasses META's data cache.
    # Set the SLC uncached bit in the output address so the USB BAR and META see each other's writes.
    mailbox = FW_DEVVA + 0x1000 | 1<<40
    for i, word in enumerate((0x30000f02, 0, mailbox & 0xffffffff, mailbox >> 32)): self.write(0x04850010+i*4, word)
    self.write(0x0480fff0, 0xc0000000)  # boot segment in the global address range
    self.write(0x0480fff8, 5)  # transfer to thread 0 PC
    deadline = time.monotonic()+1
    while not self.read(0x0480fff8) & 1<<31:
      if time.monotonic() >= deadline: raise TimeoutError("META PC transfer")
    self.write(0x048000e8, 0)  # full-width instructions
    self.write(0x04800000, 1)  # enable thread 0

def assemble(src:Path) -> bytes:
  with tempfile.TemporaryDirectory() as tmp:
    obj, text = Path(tmp)/'meta.o', Path(tmp)/'meta.bin'
    subprocess.run([os.getenv("META_AS", "metag-elf-as"), "-o", obj, src], check=True)
    subprocess.run([os.getenv("META_OBJCOPY", "metag-elf-objcopy"), "-O", "binary", "-j", ".text", obj, text], check=True)
    return text.read_bytes()
