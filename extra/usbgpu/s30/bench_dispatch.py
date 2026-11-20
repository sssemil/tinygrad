"""Measure dependent S30 launches with DEV=MT:USC and MT_META=0 or 1."""
import random, statistics, struct, time
from collections import Counter
from contextlib import ExitStack
from unittest.mock import patch
from tinygrad import Device, UOp, dtypes
from tinygrad.codegen import to_program
from tinygrad.device import Buffer
from tinygrad.renderer import TinyELF
from tinygrad.runtime.support.mt.s30 import MTS30Program
from tinygrad.runtime.support.usb import USB3
from tinygrad.uop.ops import KernelInfo

def increment(dev):
  out = UOp.placeholder((9*32,), dtypes.int, slot=0)
  i = UOp.special(9, "gidx0")*32 + UOp.special(32, "lidx0")
  ast = out[i].store(out[i]+1).sink(arg=KernelInfo(name="increment_grid", optimized=True))
  return MTS30Program(dev, TinyELF.from_program(to_program(ast, dev.renderer)), {})

if __name__ == "__main__":
  dev = Device["MT"]
  assert dev.mt.s30 and dev.renderer.target.renderer == "USC"
  program = increment(dev)
  rng = random.Random(30)
  values = [rng.randrange(10000) for _ in range(9*32)]
  buf = Buffer("MT", 4*len(values), initial_value=struct.pack(f"<{len(values)}i", *values))
  def run(n):
    for _ in range(n): program(buf.meta.va, global_size=(9,1,1), local_size=(32,1,1))
    dev.s30_fw.flush_meta()
  def check(n):
    values[:] = [x+n for x in values]
    assert list(buf.as_memoryview().cast("i")) == values
  run(1)
  check(1)
  for n in (1, 8, 64):
    times = []
    for _ in range(11):
      start = time.perf_counter()
      run(n)
      times.append(time.perf_counter()-start)
      check(n)
    counts:Counter = Counter()
    def counted(fn):
      def wrapped(*args, **kwargs):
        counts[fn.__name__] += 1
        return fn(*args, **kwargs)
      return wrapped
    with ExitStack() as stack:
      for method in ("control_read", "control_write", "bulk_read", "bulk_write"):
        stack.enter_context(patch.object(USB3, method, counted(getattr(USB3, method))))
      run(n)
    check(n)
    print(f"{n:2d} dispatches {statistics.median(times)*1e3:7.3f} ms, {counts.total()} USB transfers {dict(counts)}")
