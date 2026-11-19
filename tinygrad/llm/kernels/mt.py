import math
from typing import TypeGuard
from tinygrad import Context, Device, Tensor, UOp, dtypes
from tinygrad.helpers import all_int, prod
from tinygrad.uop.ops import KernelInfo, Ops

def is_s30(device:str|tuple[str, ...]|None, renderer:str|None=None) -> TypeGuard[str]:
  if not isinstance(device, str) or device.partition(":")[0] != "MT": return False
  with Context(ALLOW_DEVICE_USAGE=1): return (dev:=Device[device]).arch == "mp_10" and (renderer is None or dev.renderer.target.renderer == renderer)

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
