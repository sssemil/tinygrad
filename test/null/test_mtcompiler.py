import unittest
from tinygrad.runtime.support.mt.compiler import MTCompiler, MTLLVMCompiler
from tinygrad.runtime.support.mt.elf import MTKernelImage

class TestMTCompiler(unittest.TestCase):
  def test_usc_rematerialize_ids(self):
    from tinygrad.renderer.usc import assemble, Instruction
    # The store keeps both hardware IDs live across 63 loads. The max pins those loads before the sum.
    ins:list[Instruction] = [("group", (), 0), ("lane", (), 0), ("st", (0,1), 0)]
    loads = []
    for i in range(63):
      ins.extend([("imm", (), 4+i), ("ld", (len(ins),), 2)])
      loads.append(len(ins)-1)
    acc = len(ins)
    ins.append(("fmax", (loads[0], loads[0]), 0))
    for v in loads[1:]:
      ins.append(("fadd", (acc,v), 0))
      acc = len(ins)-1
    ins.extend([("st", (0,acc), 0), ("st", (1,acc), 0)])
    self.assertLessEqual(assemble(ins)[1], 64)

  def test_usc_slot_snapshot(self):
    from tinygrad.renderer.usc import assemble, _assemble, Instruction
    start:list[Instruction] = [("imm", (), 2), ("slot", (), 0), ("assign", (1,0), 0)]
    snapshot:Instruction = ("mov", (1,), 0)
    self.assertEqual(assemble(start+[snapshot, ("st", (0,3), 0)]), _assemble(start+[("st", (0,1), 0)]))
    # Keep snapshots across writes, loop edges and changes of active lanes.
    for body in [[("assign", (1,0), 0)], [("range", (0,), 0), ("ilt", (4,0), 0), ("back", (4,5), 0)],
                 [("ilt", (0,1), 0), ("cndst", (4,), 0), ("cndend", (), 0)]]:
      ins = start+[snapshot]+body+[("st", (0,3), 0)]
      self.assertEqual(assemble(ins), _assemble(ins))

  def test_default_renderer(self):
    from tinygrad.device import select_renderer
    from tinygrad.helpers import Target
    from tinygrad.runtime.ops_mt import MTUSCRenderer, MUSARenderer
    native = select_renderer(Target("MT", arch="mp_10"))
    self.assertIsInstance(native, MTUSCRenderer)
    self.assertEqual(native.target.renderer, "USC")
    self.assertIsInstance(select_renderer(Target("MT", arch="mp_21")), MUSARenderer)
    self.assertIsInstance(select_renderer(Target("MT", "MUSA", "mp_10")), MUSARenderer)

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
