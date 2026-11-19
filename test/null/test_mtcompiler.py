import unittest
from tinygrad.runtime.support.mt.compiler import MTCompiler, MTLLVMCompiler
from tinygrad.runtime.support.mt.elf import MTKernelImage

class TestMTCompiler(unittest.TestCase):
  @unittest.skipUnless((MTCompiler().bin/'mcc').is_file(), 'MUSA toolchain is not installed')
  def test_compile_and_metadata(self):
    image = MTCompiler().compile('#include <musa.h>\nextern "C" __global__ void mt_compute(int *out) { out[0] = 43; }\n')
    self.assertEqual(image.entry, 0x1700)
    self.assertEqual(image.text, bytes.fromhex('4e2ac341400040c01c070080'))
    self.assertEqual(image.metadata.immediates[-1].value, 43)
    self.assertEqual(image.metadata.kernel['.name'], 'mt_compute')

  @unittest.skipUnless((MTCompiler().bin/'clang-14').is_file(), 'MTGPU clang is not installed')
  def test_s30_llvm_store(self):
    lib = MTLLVMCompiler("mp_10").compile("define mtgpu_kernel void @store(i32* %out) { store i32 43, i32* %out\n ret void }")
    image = MTKernelImage.from_elf(lib)
    self.assertEqual(image.text, bytes.fromhex('4e2ac341400040c01c070080'))
    self.assertEqual((image.metadata.kernel['.arch_id'], image.metadata.immediates[-1].value), (110, 43))

if __name__ == '__main__': unittest.main()
