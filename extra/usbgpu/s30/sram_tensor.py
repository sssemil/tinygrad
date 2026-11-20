"""Stage one fixed-shape tensor kernel in the lower half of Chestnut SRAM, then return its output over USB."""
from collections.abc import Callable
import numpy as np
from extra.usbgpu.s30.sram import ChestnutBoot
from tinygrad import Tensor
from tinygrad.codegen import do_to_program
from tinygrad.helpers import Target, round_up
from tinygrad.renderer.usc import MTUSCRenderer
from tinygrad.runtime.support.mt.mmu import MT_PTE_READ_WRITE
from tinygrad.runtime.support.usb import HALF

class SRAMKernel:
  def __init__(self, fw:ChestnutBoot, fn:Callable[..., Tensor], *samples:np.ndarray):
    self.fw = fw
    va = 0x7400000000
    fw.mmu.map_range(va, fw.context_pa-HALF, HALF, MT_PTE_READ_WRITE)
    tensors = [Tensor(np.ascontiguousarray(v)).contiguous().realize() for v in samples]
    output = fn(*tensors)
    call, = output.schedule_linear().src
    prg = do_to_program(call.body, MTUSCRenderer(Target("MT","USC","mp_10")))
    self.shape, self.dtype, self.nbytes = output.shape, np.dtype(output.dtype.fmt), output.nbytes()
    assert len(prg.arg.outs) == 1 and self.dtype.itemsize == output.dtype.itemsize
    # Slot zero holds the bulk IN payload. The input tensors follow it in the same transfer.
    offset = round_up(self.nbytes, 64)
    offsets = {}
    for t,v in zip(tensors, samples, strict=True):
      offsets[t.uop.buf_uop] = offset
      offset += round_up(v.nbytes, 64)
    if offset > HALF: raise MemoryError("tensor kernel exceeds the lower half of Chestnut SRAM")
    self.payload = bytearray(offset)
    self.inputs = [np.ndarray(v.shape, v.dtype, self.payload, offsets[t.uop.buf_uop]) for t,v in zip(tensors, samples, strict=True)]
    self.cmd = fw.prepare(prg, [va if i in prg.arg.outs else va+offsets[call.src[i+1]] for i in prg.arg.globals])

  def __call__(self, *args:np.ndarray) -> np.ndarray:
    for buf,v in zip(self.inputs, args, strict=True):
      if buf.shape != v.shape: raise ValueError(f"expected input shape {buf.shape}, got {v.shape}")
      np.copyto(buf, v, casting="no")
    self.fw.dev.pci_dev.usb.scsi_write(self.payload)
    self.fw.stage(*self.cmd)
    self.fw.kick_direct(0.5)
    return np.frombuffer(self.fw.read_output(self.nbytes), self.dtype).reshape(self.shape)
