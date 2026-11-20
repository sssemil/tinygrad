from __future__ import annotations
import os, pathlib, subprocess, tempfile
from tinygrad.renderer import Compiler, CompileError
from tinygrad.runtime.support.mt.elf import MTKernelImage

DEFAULT_MUSA_PATH = '/tmp/mt-research/toolkit/musa_toolkits_4.3.0'

class MTLLVMCompiler(Compiler):
  def __init__(self, arch:str):
    super().__init__(f"mt_llvm_{arch}")
    self.arch = arch
  def compile(self, src:str) -> bytes:
    clang = pathlib.Path(os.getenv("MUSA_PATH", DEFAULT_MUSA_PATH))/"bin/clang-14"
    p = subprocess.run([clang, "-cc1", "-triple", "mtgpu-mt-musa", "-target-cpu", self.arch, "-mllvm", "-opaque-pointers",
                        "-mllvm", "-mtgpu-enable-const-calc=false", "-O2", "-emit-obj", "-x", "ir", "-", "-o", "-"],
                       input=src.encode(), capture_output=True)
    if p.returncode: raise CompileError(p.stderr.decode())
    return p.stdout

class MTCompiler:
  def __init__(self, musa_path:str|None=None, arch:str="mp_21"):
    self.arch = arch
    self.musa_path = musa_path or os.environ.get('MUSA_PATH', DEFAULT_MUSA_PATH)
    self.bin = pathlib.Path(self.musa_path) / 'bin'

  def _run(self, argv:list[str], cwd:str) -> None:
    result = subprocess.run(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode: raise RuntimeError(f"MT tool failed ({' '.join(argv)}):\n{result.stdout}{result.stderr}")

  def extract_raw(self, obj:pathlib.Path, cwd:str) -> bytes:
    fatbin = pathlib.Path(cwd) / 'fatbin.bin'
    self._run([str(self.bin/'llvm-objcopy'), f'--dump-section=.mt_fatbin={fatbin}', str(obj)], cwd)
    device = pathlib.Path(cwd) / 'device.out'
    self._run([str(self.bin/'clang-offload-bundler'), '-unbundle', '-type=o',
               f'-targets=host-x86_64-unknown-linux,musa-mtgpu-mt-musa--{self.arch}',
               f'-inputs={fatbin}', f'-outputs={cwd}/host.o,{device}'], cwd)
    return device.read_bytes()

  def compile_raw(self, source:str) -> bytes:
    with tempfile.TemporaryDirectory(prefix='mt-mcc-') as td:
      src, obj = pathlib.Path(td) / 'kernel.mu', pathlib.Path(td) / 'kernel.o'
      src.write_text(source)
      include = pathlib.Path(__file__).parent/'include'
      self._run([str(self.bin/'mcc'), f'--musa-path={self.musa_path}', f'--offload-arch={self.arch}', f'-I{include}',
                 *(['--cuda-device-only', '--no-gpu-bundle-output', '-mllvm', '-mtgpu-enable-const-calc=false'] if self.arch == 'mp_10' else []),
                 '-c', str(src), '-o', str(obj)], td)
      return obj.read_bytes() if self.arch == 'mp_10' else self.extract_raw(obj,td)

  def compile(self, source:str) -> MTKernelImage:
    return MTKernelImage.from_elf(self.compile_raw(source))
