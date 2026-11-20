import math
from functools import partial
from typing import TypeGuard
from tinygrad import Context, Device, Tensor, UOp, dtypes, nn
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import all_int, prod
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

def is_s30(device:str|tuple[str, ...]|None, renderer:str|None=None) -> TypeGuard[str]:
  if not isinstance(device, str) or device.partition(":")[0] != "MT": return False
  with Context(ALLOW_DEVICE_USAGE=1): return (dev:=Device[device]).arch == "mp_10" and (renderer is None or dev.renderer.target.renderer == renderer)

def _rmsnorm(out:UOp, x:UOp, weight:UOp, eps:float) -> UOp:
  rows, width = x.shape
  lanes = 64 if width % 64 == 0 else 32
  row, lane = UOp.special(rows, "gidx0"), UOp.special(lanes, "lidx0")
  r, j, k = (UOp.range(n, i, AxisType.LOOP) for i,n in enumerate((width//lanes, lanes, width//lanes)))
  shared = UOp.placeholder((lanes,), dtypes.float, slot=0, addrspace=AddrSpace.LOCAL)
  ready = shared[lane].store(x[row, r*lanes+lane].square().reduce(r, arg=Ops.ADD)).barrier()
  inv_rms = (shared.after(ready)[j].reduce(j, arg=Ops.ADD)/width + eps).rsqrt()
  i = k*lanes + lane
  return out[row, i].store(x[row, i]*inv_rms*weight[i].float()).end(k).sink(arg=KernelInfo(name="s30_rmsnorm", optimized=True))

def _rmsnorm_grad(grad:UOp, call:UOp, eps:float) -> tuple:
  x, w = (Tensor(u) for u in call.src[2:])
  dx, dw = (x.rmsnorm(eps=eps)*w).gradient(x, w, gradient=Tensor(grad))
  return None, dx.uop, dw.uop

class RMSNorm(nn.RMSNorm):
  def __call__(self, x:Tensor) -> Tensor:
    if x.dtype == dtypes.float and (w:=self.weight) is not None and w.dtype in (dtypes.float, dtypes.half) and all_int(x.shape) and \
       x.ndim and min(x.shape) > 0 and x.shape[-1] % 32 == 0 and w.shape == (x.shape[-1],) and x.device == w.device and is_s30(x.device, "USC"):
      flat = x.reshape(-1, x.shape[-1]).contiguous()
      result = flat.empty_like().custom_kernel(flat, w, fxn=partial(_rmsnorm, eps=self.eps), grad_fxn=partial(_rmsnorm_grad, eps=self.eps))[0]
      return result.reshape(x.shape)
    return super().__call__(x)

def s30_gemv(x:Tensor, weight:Tensor) -> Tensor|None:
  # The extra launch pays off on large FP16 projections. Limit the FP32 partials to 32 MiB.
  if x.dtype != dtypes.float or weight.dtype != dtypes.half or weight.ndim != 2 or not all_int(x.shape) or not all_int(weight.shape): return None
  rows, k = weight.shape
  if not x.ndim or x.shape[-1] != k or prod(x.shape[:-1]) != 1 or k % 32 or not (1<<21 <= rows*k <= 1<<28): return None
  if (device:=x.device) != weight.device or not is_s30(device): return None
  with Context(ALLOW_DEVICE_USAGE=1):
    if rows >= 128 and rows%8 == 0 and k >= 128 and Device[device].renderer.has_shared: return None  # native workgroup reduction
  # Adjacent work items can read short contiguous spans of a row before the second kernel adds the partial sums.
  partials = (weight.reshape(rows, k//32, 32) * x.reshape(k//32, 32)).sum(-1).contiguous()
  return partials.sum(-1).reshape(*x.shape[:-1], rows)

def _softmax_value(out:UOp, scores:UOp, cache:UOp, pos:UOp) -> UOp:
  b, h, d = (UOp.range(n, axis) for axis,n in enumerate((out.shape[0], out.shape[1], out.shape[3])))
  m, s, k = (UOp.range(pos+1, axis) for axis in (3, 4, 5))
  peak = scores[b, h, m].reduce(m, arg=Ops.MAX)
  def exp(i:UOp): return (scores[b, h, i] - peak).exp()
  inv_sum = exp(s).reduce(s, arg=Ops.ADD).reciprocal()
  value = cache[1, b, h//(out.shape[1]//cache.shape[2]), k, d].float()
  return out[b, h, 0, d].store((exp(k)*inv_sum*value).reduce(k, arg=Ops.ADD)).end(b, h, d).sink(arg=KernelInfo(name="s30_softmax_value"))

def s30_decode_attention(q:Tensor, cache:Tensor, pos:UOp) -> Tensor:
  b, h, _, d = q.shape
  _, _, kv_heads, capacity, _ = cache.shape
  # Score the short cache as a fixed-size matmul. Only the valid prefix participates in softmax and the value sum.
  scores = q.reshape(b, kv_heads, h//kv_heads, d).matmul(cache[0].transpose(-1, -2), dtype=dtypes.float) / math.sqrt(d)
  return q.empty_like().custom_kernel(scores.reshape(b, h, capacity).contiguous(), cache, Tensor(pos, device=q.device), fxn=_softmax_value)[0]
