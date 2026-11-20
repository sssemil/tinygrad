"""Stage one fixed-shape tensor kernel in the lower half of Chestnut SRAM, then return its output over USB."""
import operator
from collections.abc import Callable
import numpy as np
from extra.usbgpu.s30.sram import ChestnutBoot
from tinygrad import Tensor
from tinygrad.codegen import do_to_program
from tinygrad.helpers import Target, round_up
from tinygrad.renderer.usc import MTUSCRenderer
from tinygrad.runtime.support.mt.mmu import MT_PTE_READ_WRITE
from tinygrad.runtime.support.usb import HALF

class SRAMTensorBoot(ChestnutBoot):
  # Tiled models keep activations on the host between launches. Use the upper SRAM half for context, page tables and programs.
  mem_size, code_base, code_size, tables = HALF, 0x18000, HALF-0x18000, 0x8000

class SRAMKernel:
  def __init__(self, fw:ChestnutBoot, fn:Callable[..., Tensor], *samples:np.ndarray):
    self.fw = fw
    va = 0x7400000000
    fw.mmu.map_range(va, fw.context_pa-HALF, HALF, MT_PTE_READ_WRITE)
    tensors = [Tensor(np.ascontiguousarray(v)).contiguous().realize() for v in samples]
    output = fn(*tensors).contiguous()
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


def tiles(fn:Callable, *xs:np.ndarray, size:int) -> np.ndarray:
  return np.concatenate([fn(*(x[i:i+size] for x in xs)) for i in range(0,len(xs[0]),size)])

def _project(x:Tensor, w:Tensor): return x @ w.T
def _peak(x:Tensor): return x.max(-1, keepdim=True)
def _exp(x:Tensor, m:Tensor): return (x-m).exp()
def _sum(x:Tensor): return x.sum(-1, keepdim=True)
def _argmax(x:Tensor): return x.argmax()

class SRAMModel:
  def __init__(self, fw:ChestnutBoot):
    self.fw = fw
    self.programs:dict[tuple, SRAMKernel] = {}

  def op(self, fn:Callable, *xs:np.ndarray) -> np.ndarray:
    key = fn, tuple((x.shape, x.dtype) for x in xs)
    if key not in self.programs: self.programs[key] = SRAMKernel(self.fw, fn, *xs)
    return self.programs[key](*xs)

  def projection(self, x:np.ndarray, w:np.ndarray, rows:int, batch:int, bias:np.ndarray|None=None) -> np.ndarray:
    def apply(v):
      out = np.concatenate([self.op(_project, v, w[j:j+rows]) for j in range(0,len(w),rows)], axis=1)
      return out if bias is None else self.op(operator.add, out, bias)
    return tiles(apply, x, size=batch)

  def softmax(self, scores:np.ndarray) -> np.ndarray:
    ex = self.op(_exp, scores, self.op(_peak, scores))
    return self.op(operator.truediv, ex, self.op(_sum, ex))

  def argmax(self, scores:np.ndarray) -> int:
    flat = scores.ravel()
    maxima = np.concatenate([self.op(_peak, flat[i:i+8192]) for i in range(0,len(flat),8192)])
    page = int(self.op(_argmax, maxima).item())
    return page*8192 + int(self.op(_argmax, flat[page*8192:(page+1)*8192]).item())
