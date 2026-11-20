import pathlib, unittest
from tinygrad.runtime.support.mt.compiler import MTCompiler, DEFAULT_MUSA_PATH

class TestUSC(unittest.TestCase):
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

  def test_usc_dot_schedule_preserves_memory_order(self):
    from tinygrad.renderer.usc import _schedule_fmas, Instruction
    # The second load can alias the first store. Both dot products must see their respective values.
    ins:list[Instruction] = [("imm", (), 2), ("ld", (0,), 0), ("fmul", (1,1), 0), ("fma", (1,1,2), 0), ("st", (0,3), 0),
                             ("ld", (0,), 0), ("fmul", (5,5), 0), ("fma", (5,5,6), 0), ("st", (0,7), 0)]
    self.assertEqual([op for op,_,_ in _schedule_fmas(ins) if op in ("ld","st")], ["ld","st","ld","st"])

  def test_usc_dot_burst_bounds(self):
    from tinygrad.renderer.usc import _burst_dots, Instruction
    # One output from two arrays, with a shared 4-byte stride and a register accumulator.
    constants = [0, 64, 1, 4]
    ins:list[Instruction] = [("imm", (), 6), ("slot", (), 0), ("assign", (1,0), 0), ("imm", (), 7), ("range", (0,), 64),
                             ("imul", (~9,4), 0), ("ld", (5,), 2), ("ld", (5,), 4), ("fma", (6,7,1), 0), ("assign", (1,8), 0),
                             ("iadd", (4,~8), 0), ("assign", (4,10), 0), ("ilt", (4,3), 0), ("back", (4,12), 0), ("st", (0,1), 0)]
    self.assertIsNotNone(_burst_dots(ins, constants.copy(), 6))
    for i,change in [(4, ("range", (0,), 65)), (5, ("imul", (~8,4), 0)), (14, ("st", (0,7), 0))]:
      self.assertIsNone(_burst_dots(ins[:i]+[change]+ins[i+1:], constants, 6))
    self.assertEqual(constants, [0,64,1,4])

  def test_register_limit_retries_only_automatic_opts(self):
    from unittest.mock import patch
    from tinygrad import UOp, dtypes
    from tinygrad.codegen import do_to_program
    from tinygrad.device import select_renderer
    from tinygrad.helpers import Target
    from tinygrad.renderer import RegisterAllocationError
    from tinygrad.uop.ops import KernelInfo
    ren = select_renderer(Target("MT", "USC", "mp_10"))
    out = UOp.placeholder((1,), dtypes.int, slot=0)
    def kernel(opts): return out[0].store(1).sink(arg=KernelInfo(opts_to_apply=opts))
    render = ren.render
    def fail_once(uops):
      if mocked.call_count == 1: raise RegisterAllocationError()
      return render(uops)
    with patch.object(ren, "render", side_effect=fail_once) as mocked:
      self.assertEqual(do_to_program(kernel(None), ren).src[-1].op.name, "BINARY")
      self.assertEqual(mocked.call_count, 2)
      mocked.reset_mock()
      with self.assertRaises(RegisterAllocationError): do_to_program(kernel(()), ren)
      self.assertEqual(mocked.call_count, 1)

  def test_default_renderer(self):
    from tinygrad.device import select_renderer
    from tinygrad.helpers import Target
    from tinygrad.runtime.ops_mt import MTUSCRenderer, MUSARenderer
    native = select_renderer(Target("MT", arch="mp_10"))
    self.assertIsInstance(native, MTUSCRenderer)
    self.assertEqual(native.target.renderer, "USC")
    self.assertIsInstance(select_renderer(Target("MT", arch="mp_21")), MUSARenderer)
    with self.assertRaises(RuntimeError): select_renderer(Target("MT", "MUSA", "mp_10"))

@unittest.skipUnless(pathlib.Path(DEFAULT_MUSA_PATH, 'bin/mcc').exists(), 'MUSA toolchain is not installed')
class TestMTCompiler(unittest.TestCase):
  def test_compile_and_metadata(self):
    image = MTCompiler().compile('#include <musa.h>\nextern "C" __global__ void mt_compute(int *out) { out[0] = 43; }\n')
    self.assertEqual(image.entry, 0x1700)
    self.assertEqual(image.text, bytes.fromhex('4e2ac341400040c01c070080'))
    self.assertEqual(image.metadata.immediates[-1].value, 43)
    self.assertEqual(image.metadata.kernel['.name'], 'mt_compute')

if __name__ == '__main__': unittest.main()
