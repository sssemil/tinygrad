"""Compare three S30 shader compilers on stock MTFW and source-built META with identical scalar FP32 kernels."""
import argparse, hashlib, json, os, re, statistics, subprocess, sys, time
import numpy as np
from tinygrad import Device, UOp, dtypes
from tinygrad.codegen import do_linearize, do_to_program, full_rewrite_to_sink
from tinygrad.device import Buffer
from tinygrad.helpers import Target
from tinygrad.renderer import TinyELF
from tinygrad.renderer.usc import MTUSCRenderer
from tinygrad.runtime.ops_mt import MTLLVMRenderer, MUSARenderer
from tinygrad.runtime.support.mt.s30 import MTS30Program, s30_shader
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, ProgramInfo

def kernels():
  rng = np.random.default_rng(30)
  def data(*shape): return rng.integers(-4, 5, shape).astype(np.float32)
  def buf(a, slot): return UOp.placeholder((a.size,), dtypes.float, slot=slot)
  def axis(n): return UOp.special(n//32, "gidx0")*32 + UOp.special(32, "lidx0")
  for n in (32, 262144):
    a, b = data(n), data(n)
    i = axis(n)
    yield f"add_{n}", (a, b), a+b, buf(a, 0)[i].store(buf(a, 1)[i]+buf(b, 2)[i])
  a, b = data(2048, 512), data(512)
  i, k = axis(len(a)), UOp.range(len(b), 0, AxisType.LOOP)
  expected = a@b
  yield "matvec", (a, b), expected, buf(expected, 0)[i].store((buf(a, 1)[i*len(b)+k]*buf(b, 2)[k]).reduce(k, arg=Ops.ADD))
  a, b = data(128, 128), data(128, 128)
  i, k = axis(a.size), UOp.range(128, 0, AxisType.LOOP)
  dot = (buf(a, 1)[i//128*128+k]*buf(b, 2)[k*128+i%128]).reduce(k, arg=Ops.ADD)
  yield "matmul", (a, b), a@b, buf(a, 0)[i].store(dot)

def measure(compiler):
  dev = Device["MT"]
  assert dev.mt.s30
  # Firmware comparisons use the same board management request as the stock driver. Device finalization restores P12.
  dev.mt.power_on()
  ref = MTUSCRenderer(Target("MT", "USC", "mp_10"))
  ren = {"USC":MTUSCRenderer, "MUSA":MUSARenderer, "LLVM":MTLLVMRenderer}[compiler](Target("MT", compiler, "mp_10"))
  rows = {}
  for name,inputs,expected,store in kernels():
    sink = full_rewrite_to_sink(store.sink(arg=KernelInfo(name=name, optimized=True)), ref, optimize=False)
    info = ProgramInfo.from_sink(sink, ren.target)
    fixed = do_linearize(ref, UOp(Ops.PROGRAM, src=(sink,), arg=info), sink)
    prg = do_to_program(fixed, ren)
    assert prg.src[1] is fixed.src[1]
    obj = TinyELF.from_program(prg)
    run = MTS30Program(dev, obj, {}, direct=dev.renderer.target.renderer == "USC")
    buffers = [Buffer("MT", x.nbytes, initial_value=x.tobytes()) for x in (np.full_like(expected, np.nan), *inputs)]
    pointers = [buffers[j].meta.va for j in info.globals]
    groups, local = info.launch_dims({})
    times = []
    for _ in range(13):
      start = time.perf_counter_ns()
      run(*pointers, global_size=groups, local_size=local, wait=True)
      times.append((time.perf_counter_ns()-start)/1e6)
    np.testing.assert_array_equal(np.frombuffer(buffers[0].as_memoryview(), np.float32).reshape(expected.shape), expected)
    rows[name] = {"work":[fixed.src[1].key.hex(), groups, local], "shader":hashlib.sha256(s30_shader(obj)[0]).hexdigest(),
                  "ms":statistics.median(times[2:])}
  return rows

def model_benchmark(argv:list[str], vendor_schedule:bool):
  from collections import Counter
  from unittest.mock import patch
  from extra.usbgpu import bench_s30
  from tinygrad import codegen
  from tinygrad.llm.kernels import mt
  from tinygrad.codegen.opt import Opt, OptOps
  from tinygrad.codegen.opt.postrange import Scheduler
  original_opts, launch = codegen.apply_opts, MTS30Program.__call__
  reference = MUSARenderer(Target("MT", "MUSA", "mp_10"))
  schedules:set[str] = set()
  launches, transport = hashlib.sha256(), Counter()
  def fixed_opts(ast, ren, beam=0):
    if ren.target.device != "MT": return original_opts(ast, ren, beam)
    if vendor_schedule: out = original_opts(ast, reference, beam)
    else:
      k = Scheduler(ast, ren)
      k.convert_loop_to_global()
      if k.ast.arg.opts_to_apply is not None:
        for opt in k.ast.arg.opts_to_apply: k.apply_opt(opt)
      else:
        # Keep the output shape and reduction order. Distribute one output axis across at most 32 lanes.
        for i in reversed(k.axes_of(AxisType.GLOBAL)):
          r = k.rngs[i]
          if r.src[0].op is Ops.CONST and (n:=next((n for n in range(32, 1, -1) if (r.vmax+1)%n == 0), 1)) > 1:
            k.apply_opt(Opt(OptOps.SPLIT, i, (n, AxisType.LOCAL)))
            break
      out = k.get_optimized_ast()
    schedules.add(out.key.hex())
    return out
  def record(program, *args, **kwargs):
    if program.name in ("s30_copy", "s30_read_signal"): transport[program.name] += 1
    else: launches.update(repr((program.name, kwargs.get("global_size"), kwargs.get("local_size"))).encode())
    return launch(program, *args, **kwargs)
  # Use the generic model graph on every renderer. S30-only attention, RMSNorm and weight layouts change the kernel stream.
  with patch.object(mt, "is_s30", return_value=False), patch.object(codegen, "apply_opts", fixed_opts), \
       patch.object(MTS30Program, "__call__", record), patch.object(sys, "argv", [sys.argv[0], *argv]):
    bench_s30.main()
  digest = hashlib.sha256(str(sorted(schedules)).encode()).hexdigest()
  print(f"{len(schedules)} schedules {digest}, launches {launches.hexdigest()}, transport {dict(transport)}")

def main():
  p = argparse.ArgumentParser(description=__doc__)
  mode = p.add_mutually_exclusive_group()
  mode.add_argument("--model", nargs=argparse.REMAINDER, help="run the bench_s30 model and its arguments with a common schedule")
  mode.add_argument("--compare-model", nargs=argparse.REMAINDER, help="run the model on all three stacks and require matching work fingerprints")
  p.add_argument("--musa-schedule", action="store_true", help="use the MUSA scheduler for the selected model")
  mode.add_argument("--compiler", choices=["USC", "MUSA", "LLVM"], help="measure one compiler with the firmware selected by DEV")
  args = p.parse_args()
  if args.compare_model:
    model_args = list(args.compare_model)
    if model_args[0] != "train-mnist" and "--check-reference" not in model_args: model_args.append("--check-reference")
    cmd = [sys.executable, __file__, *(["--musa-schedule"] if args.musa_schedule else []), "--model", *model_args]
    fingerprints = {}
    for cc in ("USC", "MUSA", "LLVM"):
      env = dict(os.environ, DEV="MT:"+cc, MT_META=str(int(cc == "USC")), CACHELEVEL="0", PARALLEL="0")
      if cc == "USC": env.update(MUSA_PATH="/nonexistent", MT_FIRMWARE="/nonexistent")
      print(f"Running {model_args[0]} on {cc}", flush=True)
      output = subprocess.check_output(cmd, env=env, text=True)
      print(output, end="", flush=True)
      fingerprints[cc] = re.findall(r"^(\d+) schedules ([0-9a-f]{64}), launches ([0-9a-f]{64}),", output, re.M)
    if not all(len(v) == 1 for v in fingerprints.values()) or len({tuple(v) for v in fingerprints.values()}) != 1:
      raise RuntimeError(f"model schedules or launch sequences differ: {fingerprints}")
    return
  if args.model: return model_benchmark(args.model, args.musa_schedule)
  if args.musa_schedule or args.model == [] or args.compare_model == []: p.error("--model needs a bench_s30 workload and its arguments")
  if args.compiler:
    print(json.dumps(measure(args.compiler)))
    return
  # Run one compiler/firmware pair per process, in opposite orders and then rotated.
  pairs = [(fw,cc) for fw in ("MUSA", "USC") for cc in ("USC", "MUSA", "LLVM")]
  work, shaders, times = {}, {}, {}
  for order in (pairs, pairs[::-1], pairs[3:]+pairs[:3]):
    for fw,cc in order:
      env = dict(os.environ, DEV="MT:"+fw, MT_META=str(int(fw == "USC")), CACHELEVEL="0", PARALLEL="0")
      if fw == cc == "USC": env.update(MUSA_PATH="/nonexistent", MT_FIRMWARE="/nonexistent")
      results = json.loads(subprocess.check_output([sys.executable, __file__, "--compiler", cc], env=env))
      for name,row in results.items():
        if row["work"] != work.setdefault(name, row["work"]): raise RuntimeError(f"{name}: different kernel or launch dimensions")
        if row["shader"] != shaders.setdefault((cc,name), row["shader"]): raise RuntimeError(f"{cc} {name}: different shader text")
        times.setdefault((fw,cc,name), []).append(row["ms"])
  print("firmware   compiler  "+" ".join(f"{name:>12}" for name in work)+"  (ms)")
  for fw,cc in pairs:
    print(f"{'MTFW' if fw == 'MUSA' else 'open META':10} {cc:8}  "+" ".join(f"{statistics.median(times[fw,cc,name]):12.4f}" for name in work))

if __name__ == "__main__": main()
