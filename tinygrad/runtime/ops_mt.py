from __future__ import annotations
import dataclasses, os, pathlib, re, subprocess, tempfile, hashlib, functools
from typing import Any
from tinygrad.device import Compiled, Program, Allocator, BufferStorage, BufferSpec, Buffer
from tinygrad.renderer import Compiler, TinyELF
from tinygrad.renderer.cstyle import CUDARenderer, CStyleLanguage
from tinygrad.renderer.usc import MTUSCRenderer
from tinygrad.renderer.llvmir import LLVMRenderer, base_rewrite, ldt, llvm_intrinsics
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.helpers import round_up, diskcache_get, diskcache_put, DEV, Target, unwrap
from tinygrad.uop.ops import KernelInfo, Ops, PatternMatcher, UOp, UPat
from tinygrad.runtime.support.memory import TLSFAllocator
from tinygrad.runtime.support.usb import USBMMIOInterface, HALF, SLOT
from tinygrad.runtime.support.mt.mmu import MT_PTE_READ_ONLY, MT_PTE_READ_WRITE
from tinygrad.runtime.support.mt.mtdev import MTDev
from tinygrad.runtime.support.mt.launch import MTLaunch, parse_mt_bundle, compress_mt_bundle, decompress_mt_bundle
from tinygrad.runtime.support.mt.compiler import MTCompiler, MTLLVMCompiler
from tinygrad.runtime.support.mt.elf import MTKernelImage
from tinygrad.runtime.support.mt.s30 import DEFAULT_S30_FIRMWARE, MTS30Firmware, MTS30Program

MUSA_SOURCES:dict[str,str] = {}


def generate_mt_launch(source:str, name:str, args:list[str], global_size:tuple[int,int,int],
                       local_size:tuple[int,int,int]) -> dict[str,Any]:
  td = tempfile.mkdtemp(prefix='mt-launch-')
  src_path = pathlib.Path(td) / 'kernel.mu'
  arg_decls = 'void *out = reinterpret_cast<void *>(uintptr_t(0xe1c2003000ULL)); uint64_t scalar = 1;'
  arg_names = ','.join('&out' if x == 'out' else '&scalar' for x in args)
  main = (f'#include <musa.h>\n#include <cstdint>\n#include <cstdio>\n#include <cstdlib>\n#include <unistd.h>\nint main() {{\n'
          f'  musaFunction_t fn = nullptr;\n  musaGetFuncBySymbol(&fn, (const void *)&{name});\n'
          f'  {arg_decls}\n  void *args[] = {{{arg_names}}};\n'
          f'  musaLaunchKernel((const void *)&{name}, dim3{global_size}, dim3{local_size}, args, 0, nullptr);\n'
          f'  musaDeviceSynchronize();\n'
          f'  char done[512];\n  snprintf(done, sizeof(done), "%s/real-capture-done", getenv("MT_CAPTURE_DIR"));\n'
          f'  while (access(done, F_OK) != 0) usleep(1000);\n  _Exit(0);\n}}\n')
  src_path.write_text(source + '\n' + main)
  musa = os.environ.get('MUSA_PATH', '/tmp/mt-research/toolkit/musa_toolkits_4.3.0')
  driver = '/tmp/mt-research/musa-driver-3.3.0/usr/lib/x86_64-linux-gnu'
  host = str(pathlib.Path(td) / 'host')
  include = pathlib.Path(__file__).parent/'support/mt/include'
  cmd = [str(pathlib.Path(musa)/'bin/mcc'), f'--musa-path={musa}', '--offload-arch=mp_21', f'-I{include}', str(src_path), '-o', host,
         f'-L{musa}/lib', f'-L{driver}', '-lmusart', '-lmusa', '-lstdc++']
  result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
  if result.returncode: raise RuntimeError(f'MUSA launch host build failed:\n{result.stdout}{result.stderr}')
  env = os.environ.copy()
  env.update({'MT_CAPTURE_DIR':td, 'MT_MPX':'0xff', 'MT_UMD':'1', 'LD_PRELOAD':mtum_library(),
              'LD_LIBRARY_PATH':f'{musa}/lib:{driver}'})
  result = subprocess.run([host], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
  if 'JOB SUBMIT' not in result.stderr: raise RuntimeError(f'MUSA launch generation failed:\n{result.stdout}{result.stderr}')
  state, lib = parse_mt_bundle(pathlib.Path(td)), MTCompiler(musa).extract_raw(pathlib.Path(host),td)
  for path in pathlib.Path(td).iterdir():
    if path.name != 'real-submission.bin': path.unlink()
  return {'bundle':td,'packet':str(pathlib.Path(td)/'real-submission.bin'),'state':state,'lib':lib}


class MUSACompiler(Compiler):
  def __init__(self, arch:str="mp_21"):
    super().__init__({'mp_21':'mt', 'mp_10':'mt_mp_10_no_const_calc'}.get(arch, f'mt_{arch}'))
    self.mt = MTCompiler(arch=arch)
  def compile(self, src:str) -> bytes:
    if self.mt.arch == "mp_10": return self.mt.compile_raw(src)
    if os.environ.get('MT_GENERATE','1') == '0': lib = self.mt.compile_raw(src)
    else:
      match = re.search(r'extern "C" __global__ void\s+(\w+)\s*\(([^)]*)\)',src)
      if match is None: raise RuntimeError('MUSA kernel signature is missing')
      params = [p for p in match.group(2).split(',') if p.strip()]
      args = ['out' if '*' in p else 'scalar' for p in params]
      cached = generate_mt_launch(src,match.group(1),args,(1,1,1),(1,1,1))
      lib = cached['lib']
      cached['state'] = compress_mt_bundle(cached['state'])
      key = hashlib.sha256(lib).hexdigest()
      diskcache_put('mt_launch_v3',f'{key}_{(1,1,1)}_{(1,1,1)}',cached)
    key = hashlib.sha256(lib).hexdigest()
    MUSA_SOURCES[key] = src
    diskcache_put('mt_sources', key, src)
    return lib

pm_s30_half = PatternMatcher([
  (UPat((Ops.SQRT, Ops.RECIP, Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.TRUNC), dtypes.half, (UPat.var("x"),), name="op"),
   lambda op,x: x.cast(dtypes.float).alu(op.op).cast(dtypes.half)),
])

class MUSARenderer(CUDARenderer):
  compiler:MUSACompiler
  kernel_typedef = 'extern "C" __global__ void'
  def __init__(self, target):
    CStyleLanguage.__init__(self, target)
    self.compiler, self.tensor_cores = MUSACompiler("mp_10" if target.arch == "sudi" else target.arch or "mp_21"), []
    if self.compiler.mt.arch == "mp_10":
      self.has_shared, self.shared_max = False, 0
      self.type_map = {**self.type_map, dtypes.half:"mt_half", dtypes.float:"mt_float"}
      self.extra_matcher += pm_s30_half
  def render_vector_prefix(self, dt, count):
    if self.compiler.mt.arch == "mp_10" and dt in (dtypes.float, dtypes.half) and count in (2, 4):
      # Conditional loads of MUSA's vector structs spill to private memory on mp_10.
      scalar = self.type_map[dt]
      vec = f"{scalar}{count}"
      return f"typedef {scalar} {vec} __attribute__((ext_vector_type({count})));\n#define make_{vec}(...) (({vec}){{__VA_ARGS__}})"
    return super().render_vector_prefix(dt, count)
  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None):
    src = super().render_kernel(function_name, kernel, bufs, uops, ['#include <musa.h>'] + (prefix or []))
    if self.compiler.mt.arch != "mp_10": return src
    vectors = [self.render_vector_prefix(dt, count) for dt,count in ((dtypes.half, 2), (dtypes.float, 2), (dtypes.float, 4))]
    return "\n".join(["using mt_float = float; using mt_half = _Float16;", *vectors, src])
  def supported_dtypes(self): return set(dtypes.all) - {dtypes.double}

def mt_sreg(axis:str) -> str:
  return f"llvm.musa.read.ptx.sreg.{'ctaid' if axis[0] == 'g' else 'tid'}.{'xyz'[int(axis[-1])]}"

mt_intrinsics = {**llvm_intrinsics, Ops.SIN:"sin", Ops.TRUNC:"trunc"}

class MTLLVMRenderer(LLVMRenderer):
  abi = "mtgpu_kernel"
  has_shared, shared_max = False, 0
  global_max, local_max = CUDARenderer.global_max, CUDARenderer.local_max
  code_for_op = {**LLVMRenderer.code_for_op, **dict.fromkeys(mt_intrinsics, lambda: None)}
  string_rewrite = PatternMatcher([
    (UPat(Ops.SPECIAL, name="x"), lambda ctx,x: f"  {ctx[x]} = call i32 @{mt_sreg(x.arg)}()"),
    # mcc uses this instruction for float division. Generic LLVM fdiv pulls in a device-library routine on mp_10.
    (UPat(Ops.FDIV, dtypes.float, (UPat.var("a"), UPat.var("b")), name="x"), lambda ctx,x,a,b:
     f"  {ctx[x]} = call float @llvm.musa.fast.fdivide.f(float {ctx[a]}, float {ctx[b]})"),
    (UPat(tuple(mt_intrinsics), name="x"), lambda ctx,x:
     f"  {ctx[x]} = call {ldt(x.dtype)} @llvm.{mt_intrinsics[x.op]}.f{x.dtype.itemsize*8}({ldt(x.dtype)} {ctx[x.src[0]]})"),
  ]) + base_rewrite
  def __init__(self, target:Target):
    if target.arch != "mp_10": raise RuntimeError("MT LLVM requires the S30")
    super().__init__(target)
    self.extra_matcher = unwrap(self.extra_matcher) + pm_s30_half
    self.compiler = MTLLVMCompiler(target.arch)
  def render(self, uops:list[UOp]) -> str:
    decls = {f"declare i32 @{mt_sreg(u.arg)}()" for u in uops if u.op is Ops.SPECIAL}
    if any(u.op is Ops.FDIV and u.dtype == dtypes.float for u in uops): decls.add("declare float @llvm.musa.fast.fdivide.f(float, float)")
    decls |= {f"declare {ldt(u.dtype)} @llvm.{mt_intrinsics[u.op]}.f{u.dtype.itemsize*8}({ldt(u.dtype)})" for u in uops if u.op in mt_intrinsics}
    _, kernel = self._render_kernel(uops, sorted(decls))
    return kernel
  def supported_dtypes(self): return super().supported_dtypes() - {dtypes.double, *dtypes.fp8s}

@dataclasses.dataclass(frozen=True)
class MTBufferMeta:
  va: int
  pa: int
  size: int

S30_STAGE_VA = 0x7000000000

class S30VRAMView(USBMMIOInterface):
  def __init__(self, view:USBMMIOInterface, dev:MTDevice, va:int):
    super().__init__(view.usb, view.addr, view.nbytes, view.fmt)
    self.dev, self.va = dev, va

  def view(self, offset:int=0, size:int|None=None, fmt=None):
    return S30VRAMView(super().view(offset, size, fmt), self.dev, self.va+offset)

  def __setitem__(self, index, data):
    offset, _ = self._off_from_index(index)
    if (self.fmt == 'B' and isinstance(index, slice) and isinstance(data, (bytes, bytearray, memoryview)) and
        SLOT <= memoryview(data).nbytes <= self.nbytes-offset and offset >= 0 and
        (self.va+offset)%4 == 0 and self.dev.renderer.target.renderer == "USC"):
      src = bytes(data)  # a USB read can alias the buffer used by the next transfer
      n = len(src)//128*128
      # The upper SRAM half survives the bridge's ordinary F0 traffic for command packets and page tables.
      with self.dev.s30_fw.lock:
        copy = self.dev.s30_upload
        for off in range(0, n, HALF):
          chunk = src[off:min(off+HALF, n)]
          self.usb.scsi_write(chunk, slot_start=HALF//SLOT)
          copy(self.va+offset+off, S30_STAGE_VA, global_size=(len(chunk)//128,1,1), local_size=(32,1,1))
        if n < len(src): super().__setitem__(slice(offset+n, offset+len(src)), src[n:])
      return
    super().__setitem__(index, data)

class MTAllocator(Allocator['MTDevice']):
  def __init__(self, dev):
    super().__init__(dev)
    self.vram = TLSFAllocator(max(0, dev.mt.vram_size - 0x80000000), base=0x80000000, block_size=0x1000)

  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage:
    size = round_up(size, 0x1000)
    pa = self.vram.alloc(size)
    va = 0x8001000000 + (pa - self.vram.base)
    try:
      view = self.dev.mt.pci_dev.map_bar(2, off=pa, size=size, fmt='B')
      self.dev.map_buffer(va, pa, size)
      if self.dev.mt.s30 and size >= SLOT and isinstance(view, USBMMIOInterface) and view.usb.usb.product.startswith("custom"):
        view = S30VRAMView(view, self.dev, va)
    except Exception:
      self.vram.free(pa)
      raise
    meta = MTBufferMeta(va, pa, size)
    self.dev.views[va] = (view, meta)
    return BufferStorage(va, meta, view)

  def _free(self, storage:BufferStorage, options:BufferSpec):
    self.dev.synchronize()
    meta = storage.meta
    self.dev.views.pop(meta.va, None)
    self.vram.free(meta.pa)

  def _copyin(self, dest:int, src:memoryview):
    view, meta = self.dev._view(dest)
    off = dest-meta.va
    if off % 4 == 0 and src.nbytes % 4 == 0: view[off:off+src.nbytes] = src.cast('B')
    else:
      start, end = off & ~3, round_up(off+src.nbytes,4)
      data = bytearray(view[start:end])
      data[off-start:off-start+src.nbytes] = src.cast('B')
      view[start:end] = data

  def _copyout(self, dest:memoryview, src:int):
    view, meta = self.dev._view(src)
    off = src-meta.va
    start, end = off & ~3, round_up(off+dest.nbytes,4)
    data = view[start:end]
    dest[:] = data[off-start:off-start+dest.nbytes]

  def _map(self, buf:Buffer) -> BufferStorage: raise NotImplementedError('MT external mappings are not implemented')
  def _offset(self, buf:int, size:int, offset:int) -> int: return buf + offset

def mtum_library() -> str:
  path = os.environ.get('MT_UMD', str(pathlib.Path.home()/'.cache/tinygrad/libmtum.so'))
  source = pathlib.Path(__file__).parent/'support/mt/mtum.c'
  if not pathlib.Path(path).exists() or pathlib.Path(path).stat().st_mtime < source.stat().st_mtime:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = f'{path}.{os.getpid()}'
    result = subprocess.run(['cc','-shared','-fPIC','-O2','-o',tmp,str(source),'-ldl'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode: raise RuntimeError(f'libmtum build failed:\n{result.stdout}{result.stderr}')
    os.replace(tmp,path)
  return path

class MTLaunchGenerator:
  def __init__(self, dev:MTDevice):
    self.dev = dev

  def generate(self, obj:TinyELF, global_size:tuple[int,int,int], local_size:tuple[int,int,int]) -> MTLaunch:
    args = ['out' if addrspace != AddrSpace.ALU else 'scalar' for _,addrspace,_,_ in obj.signature]
    key = hashlib.sha256(obj.lib).hexdigest()
    source = MUSA_SOURCES.get(key) or diskcache_get('mt_sources', key)
    if source is None: raise RuntimeError('MUSA source is missing from the launch-generation cache')
    cached = generate_mt_launch(source,obj.name,args,global_size,local_size)
    return MTLaunch(self.dev.mt,cached['bundle'],cached['packet'],cached['state'])

class MTProgram(Program['MTDevice']):
  def __init__(self, dev:MTDevice, obj:TinyELF, links:dict[str,MTProgram]):
    self.dev, self.name, self.profile_key, self.obj = dev, obj.name, obj.profile_key, obj
    self.image = MTKernelImage.from_elf(obj.lib)
    self.launch:MTLaunch|None = None

  def __call__(self, *args:int, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1),
               wait:bool=False, timeout:int|None=None):
    self.launch = self.dev.get_launch(self.obj, global_size, local_size)
    if os.environ.get('MT_GENERATE','1') == '0' and not os.environ.get('MT_SKIP_LOAD'):
      self.launch.load_kernel(self.image, global_size, local_size)
    self.dev.ensure_started(self.launch)
    self.launch.set_args(tuple(args))
    self.launch.set_grid(global_size, local_size)
    result = self.launch.execute()
    return result.run_time / 25_000_000 if wait else None

class MTDevice(Compiled):
  renderers = [MTUSCRenderer, MUSARenderer, MTLLVMRenderer]
  def __init__(self, device:str=''):
    dev_id = int(device.split(':')[1]) if ':' in device else 0
    self.mt = MTDev.open_usb(dev_id, manage_power=DEV.target("MT").renderer in ("MUSA", "LLVM"))
    self.views:dict[int,tuple[Any,MTBufferMeta]] = {}
    super().__init__(device, MTAllocator(self), MTS30Program if self.mt.s30 else MTProgram, arch='mp_10' if self.mt.s30 else 'mp_21')
    if self.mt.s30:
      self.s30_fw = MTS30Firmware(self.mt, os.getenv('MT_FIRMWARE', DEFAULT_S30_FIRMWARE))
      return
    bundle = os.environ.get('MT_LAUNCH_BUNDLE', '/tmp/mt-research/argfresh')
    packet = os.environ.get('MT_LAUNCH_PACKET', '/tmp/mt-research/real-submission-direct.bin')
    self.launch = None if os.environ.get('MT_GENERATE','1') != '0' else MTLaunch(self.mt, bundle, packet)
    self.launches = [] if self.launch is None else [self.launch]
    self.launch_cache:dict[tuple[int,tuple[int,int,int],tuple[int,int,int]],MTLaunch] = {}
    self.generator = MTLaunchGenerator(self)
    self.buffer_maps:list[tuple[int,int,int]] = []
    self.current_launch:MTLaunch|None = None
    self.started = False

  @property
  def has_copy_queue(self) -> bool: return False

  @functools.cached_property
  def s30_upload(self) -> MTS30Program:
    from tinygrad.codegen import to_program
    dst, src = (UOp.placeholder((HALF//4,), dtypes.uint, slot=i) for i in range(2))
    idx = UOp.special(HALF//128, "gidx0")*32+UOp.special(32, "lidx0")
    copy = dst[idx].store(src[idx]).sink(arg=KernelInfo(name="s30_upload", optimized=True))
    program = MTS30Program(self, TinyELF.from_program(to_program(copy, self.renderer)), {})
    # Sudi physical address bit 39 selects outbound PCIe. Reserve the bridge SRAM and map its upper half.
    _, (pa,) = self.mt.pci_dev.alloc_sysmem(2*HALF)
    self.s30_fw.mmu.map_range(S30_STAGE_VA, (1<<39) | (pa+HALF), HALF, MT_PTE_READ_ONLY)
    return program

  def get_launch(self, obj:TinyELF, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1)) -> MTLaunch:
    if os.environ.get('MT_GENERATE','1') == '0':
      assert self.launch is not None
      return self.launch
    key = (hash(obj.lib), global_size, local_size)
    if key not in self.launch_cache:
      lib_key = hashlib.sha256(obj.lib).hexdigest()
      cache_key = f'{lib_key}_{global_size}_{local_size}'
      cached = diskcache_get('mt_launch_v3', cache_key)
      if cached is None and (global_size,local_size) != ((1,1,1),(1,1,1)):
        cached = diskcache_get('mt_launch_v3', f'{lib_key}_{(1,1,1)}_{(1,1,1)}')
      if cached is not None and pathlib.Path(cached['packet']).exists():
        state = cached.get('state')
        if isinstance(state,bytes): state = decompress_mt_bundle(state)
        launch = MTLaunch(self.mt, cached['bundle'], cached['packet'], state)
      else:
        launch = self.generator.generate(obj, global_size, local_size)
      self.launch_cache[key] = launch
      self.launches.append(launch)
      for va,pa,size in self.buffer_maps: launch.map_buffer(va,pa,size)
      assert launch.bundle_state is not None
      diskcache_put('mt_launch_v3', cache_key,
                    {'bundle':str(launch.bundle),'packet':str(launch.packet_path),'state':compress_mt_bundle(launch.bundle_state)})
    return self.launch_cache[key]

  def map_buffer(self, va:int, pa:int, size:int):
    if self.mt.s30:
      with self.s30_fw.lock: self.s30_fw.mmu.map_range(va, pa, size, MT_PTE_READ_WRITE)
      return
    self.buffer_maps.append((va,pa,size))
    for launch in self.launches: launch.map_buffer(va,pa,size)

  def ensure_started(self, launch:MTLaunch):
    if self.current_launch is not launch:
      launch.start(timeout=float(os.environ.get('MT_BOOT_TIMEOUT','10')))
      self.current_launch = launch
      self.started = True

  def _view(self, va:int):
    for base, (view, meta) in self.views.items():
      if base <= va < base + meta.size: return view, meta
    raise RuntimeError(f'Unknown MT buffer address {va:#x}')

  def finalize(self):
    try:
      super().finalize()
      if self.mt.s30: self.s30_fw.session.close()
    finally: self.mt.close()
