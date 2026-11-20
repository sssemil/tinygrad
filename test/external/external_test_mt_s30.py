import os, random, struct, unittest
from pathlib import Path
from tinygrad import Context, Device, Tensor, TinyJit, Variable, dtypes
from tinygrad.device import Buffer
from tinygrad.helpers import DEV, ceildiv
from tinygrad.runtime.support.mt.compiler import MTCompiler
from tinygrad.runtime.support.mt.compute import FW_DEVVA, META_BASE
from tinygrad.runtime.support.mt.mmu import MT_PTE_READ_WRITE
from tinygrad.runtime.support.mt.s30 import DEFAULT_S30_FIRMWARE, PDS_BASE, USC_BASE, s30_pds

renderer = DEV.target("MT").renderer
reference = renderer in ("MUSA", "LLVM")
has_firmware = reference and Path(os.getenv("MT_FIRMWARE", DEFAULT_S30_FIRMWARE)).is_file()
requires_firmware = unittest.skipUnless(has_firmware, "select a reference renderer with mtfw-gen1.bin for the firmware tests")
compiler_bin = 'clang-14' if renderer == "LLVM" else 'mcc'
requires_renderer = unittest.skipUnless(not reference or (has_firmware and (MTCompiler().bin/compiler_bin).is_file()),
                                        f"selected reference renderer requires S30 firmware and {compiler_bin}")


@unittest.skipUnless(DEV.device == "MT", "run explicitly with DEV=MT on a USB Chestnut with an S30")
class TestS30USB(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    if not Device["MT"].mt.s30: raise unittest.SkipTest("requires a two-core MTT S30")

  def setUp(self): Device["MT"].s30_fw.session.close()

  def test_core_ids(self):
    cores = Device["MT"].mt.core_ids
    self.assertEqual(len(cores), 2)
    self.assertTrue(all(cores))

  @unittest.skipUnless(reference, "select a reference renderer to query the board management firmware")
  def test_firmware(self):
    mt = Device["MT"].mt
    for _ in range(12): self.assertEqual(mt.smc.rtos_version(), mt.fw_versions)

  @requires_firmware
  def test_compute_firmware_nop(self):
    fw = Device["MT"].s30_fw
    sentinel = b'\xa5' * 64
    out = Buffer("MT", len(sentinel), initial_value=sentinel)
    # META addresses the firmware heap directly. Map the next page to the tinygrad buffer.
    off = len(fw.image)
    fw.mmu.map(FW_DEVVA+off, out.meta.pa, MT_PTE_READ_WRITE)
    for _ in range(2):
      with fw.boot():
        self.assertEqual(fw.started & 0xff, 0x4d)
        self.assertEqual(fw.build, b'48ac908')
        for token,value in enumerate((0x1234567800000013, 43, 57), start=1):
          response = fw.nop(token, signal=(META_BASE+off+16, value))
          self.assertEqual((response.value, response.token), (0, token))
          self.assertEqual(bytes(out.as_memoryview()), sentinel[:16] + value.to_bytes(8, 'little') + sentinel[24:])

  @requires_firmware
  def test_compute_store(self):
    from tinygrad.renderer.usc import assemble
    fw = Device["MT"].s30_fw
    expected = bytearray(b'\xa5' * 64)
    out = Buffer("MT", len(expected), initial_value=expected)
    # Store the value in shared register 18 at the pointer in 14:15. Shared register 0 supplies a zero byte offset.
    code, temps = assemble([("imm", (), 0), ("imm", (), 18), ("st", (0,1), 14)])
    usc = Buffer("MT", len(code)+32, initial_value=code+bytes(32))
    fw.mmu.map(USC_BASE, usc.meta.pa, MT_PTE_READ_WRITE)
    data = s30_pds(PDS_BASE+96, 19, 0, ceildiv(temps, 2)) + bytes(19*4)
    pds = Buffer("MT", len(data), initial_value=data)
    fw.mmu.map(PDS_BASE, pds.meta.pa, MT_PTE_READ_WRITE)
    stream = (2, 2, 16, 0, 0, 0, 0x02000002, 0, 0x101085, 32, 64, 0, 0, 0, 0x02000002, 0)
    with fw.boot():
      for i in range(3):
        resp = fw.cdm(i+1, ())
        self.assertEqual((resp.value, resp.token), (0, i+1))
        self.assertEqual(bytes(out.as_memoryview()), expected)
      # Repeat the shared-register load so the second shader store executes near the end of the 2 KiB ring.
      for i,(off,value,commands) in enumerate(((16, 43, stream), (40, 0x89abcdef, stream[:8]*59 + stream))):
        pds.host[96+14*4:] = struct.pack('<Q8xI', out.meta.va+off, value)
        resp = fw.cdm(i+4, commands)
        self.assertEqual((resp.value, resp.token), (0, i+4))
        struct.pack_into('<I', expected, off, value)
        self.assertEqual(bytes(out.as_memoryview()), expected)

  def test_vram_roundtrip(self):
    with self.assertRaises(MemoryError): Buffer("MT", Device["MT"].mt.vram_size + 1, preallocate=True)
    rng = random.Random(30)
    for n in (1, 3, 16, 257, 4093, 4096, 4099, 16384):
      with self.subTest(n=n):
        left, right = rng.randbytes(n), rng.randbytes(n)
        a = Buffer("MT", n, initial_value=left)
        b = Buffer("MT", n, initial_value=right)
        self.assertEqual(bytes(a.as_memoryview()), left)
        self.assertEqual(bytes(b.as_memoryview()), right)

  def test_native_staged_upload(self):
    from tinygrad.runtime.support.usb import HALF, SLOT
    from tinygrad.runtime.support.mt.s30 import S30_UPLOAD_SIZE
    rng = random.Random(31)
    with Context(DEV="MT:USC", CACHELEVEL=0):
      buf = Buffer("MT", HALF*2+256, initial_value=bytes(HALF*2+256))
      expected = bytearray(buf.nbytes)
      for offset, n in ((4, SLOT-1), (128, SLOT), (128, S30_UPLOAD_SIZE), (4, S30_UPLOAD_SIZE+127), (128, HALF), (4, HALF+3),
                        (4, HALF*2+127), (5, SLOT+1)):
        with self.subTest(offset=offset, n=n):
          data = rng.randbytes(n)
          buf.host.view(offset-1)[1:1+n] = data
          expected[offset:offset+n] = data
          self.assertEqual(bytes(buf.as_memoryview()), expected)

  def test_tensor_copy(self):
    original = [13, 21, 34, 55]
    resident = Tensor(original, device="CPU", dtype=dtypes.int32).to("MT").realize()
    self.assertEqual(resident.to("CPU").tolist(), original)

  def test_native_staged_read(self):
    from concurrent.futures import ThreadPoolExecutor
    from tinygrad.runtime.support.usb import HALF, SLOT, USBMMIOInterface
    rng = random.Random(32)
    with Context(DEV="MT:USC", CACHELEVEL=0):
      buf = Buffer("MT", 2*HALF+512, preallocate=True)
      for _ in range(2):
        expected = rng.randbytes(buf.nbytes)
        USBMMIOInterface.__setitem__(buf.host, slice(0, buf.nbytes), expected)
        for offset, size in ((4, SLOT-1), (4, SLOT), (4, HALF), (4, HALF+127), (5, SLOT+1), (128, 2*HALF+129)):
          with self.subTest(offset=offset, size=size):
            self.assertEqual(buf.host.view(offset-1)[1:size+1], expected[offset:offset+size])
      small = Buffer("MT", 32, initial_value=expected[:32])
      with ThreadPoolExecutor(2) as pool:
        larger = pool.submit(lambda: bytes(buf.host[:HALF*2]))
        reads = [bytes(small.host[:32]) for _ in range(16)]
        self.assertEqual(larger.result(), expected[:HALF*2])
        self.assertEqual(reads, [expected[:32]]*16)

  def test_native_rmsnorm(self):
    import numpy as np
    from tinygrad.llm.kernels.mt import RMSNorm
    from tinygrad import nn
    rng = np.random.default_rng(30)
    for width, dtype in ((96, np.float32), (896, np.float16)):
      x, weight, grad = (rng.standard_normal(s).astype(d) for s,d in (((2,width),np.float32), ((width,),dtype), ((2,width),np.float32)))
      def run(device, layer):
        a, layer.weight, g = [Tensor(v, device=device).realize() for v in (x, weight, grad)]
        y = layer(a)
        return [t.numpy() for t in (y, *y.gradient(a, layer.weight, gradient=g))]
      with self.subTest(width=width), Context(DEV="MT:USC", CACHELEVEL=0):
        for got, ref in zip(run("MT", RMSNorm(width)), run("CPU", nn.RMSNorm(width)), strict=True):
          np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-5)

  def test_native_prefill_attention(self):
    import numpy as np
    from tinygrad.llm.kernels.mt import s30_prefill_attention
    rng = np.random.default_rng(31)
    def run(device, q, k, v, mask, g):
      a,b,c,m,grad = (Tensor(x, device=device).realize() for x in (q,k,v,mask,g))
      ret = s30_prefill_attention(a,b,c,m) if device == "MT" else a.scaled_dot_product_attention(b,c,m)
      self.assertIsNotNone(ret)
      return [t.numpy() for t in (ret, *ret.gradient(a,b,c,gradient=grad))]
    for length in (192, 1024):
      q,k,v,g = (rng.standard_normal((1,2,n,d)).astype(np.float32) for n,d in ((1024,8), (length,8), (length,16), (1024,16)))
      mask = rng.random((1024,length)) > 0.4
      with self.subTest(length=length), Context(DEV="MT:USC", CACHELEVEL=0):
        for actual, expected in zip(run("MT",q,k,v,mask,g), run("CPU",q,k,v,mask,g), strict=True):
          np.testing.assert_allclose(actual, expected, rtol=5e-4, atol=5e-5)

  def test_native_inplace_update(self):
    import numpy as np
    with Context(DEV="MT:USC", CACHELEVEL=0):
      for size, dtype in ((1, np.int64), (65, np.float32), (1024*1024, np.int32)):
        with self.subTest(size=size):
          values = np.arange(size, dtype=dtype)
          x = Tensor(values, device="MT").realize()
          for delta in (3, 7):
            x.assign(x+delta).realize()
            values = values+delta
            np.testing.assert_array_equal(x.numpy(), values)

  def test_native_vector_add(self):
    import numpy as np
    from unittest.mock import patch
    from tinygrad.runtime.support.mt.compiler import MTLLVMCompiler
    rng = np.random.default_rng(30)
    with Context(DEV="MT:USC", CACHELEVEL=0), patch.object(MTCompiler, "compile_raw", side_effect=AssertionError("called mcc")), \
         patch.object(MTLLVMCompiler, "compile", side_effect=AssertionError("called MTGPU clang")), \
         patch.object(Device["MT"].s30_fw, "boot", side_effect=AssertionError("loaded vendor compute firmware")):
      self.assertEqual(Device["MT"].renderer.target.renderer, "USC")
      for n in (1, 32, 256, 257, 4099):
        for dt in (np.int32, np.float32):
          with self.subTest(n=n, dtype=dt):
            a,b = (rng.integers(-(1<<31), 1<<31, n, dtype=dt) if dt == np.int32 else rng.normal(size=n).astype(dt) for _ in range(2))
            np.testing.assert_array_equal((Tensor(a, device="MT")+Tensor(b, device="MT")).numpy(), a+b)
      self.assertTrue(Device["MT"].s30_fw.direct)
      self.assertIsNone(Device["MT"].s30_fw.queue)

  def test_native_shared_constants(self):
    import numpy as np
    data = np.linspace(-2, 3, 257, dtype=np.float32)
    with Context(DEV="MT:USC", CACHELEVEL=0):
      x = Tensor(data, device="MT")
      y, expected = Tensor.zeros_like(x), np.zeros_like(data)
      for i in range(1, 74):
        threshold = np.float32(i/37)
        y = y + (x-threshold.item()).relu()
        expected += np.maximum(data-threshold, 0.)
      np.testing.assert_allclose(y.numpy(), expected, rtol=1e-5, atol=1e-5)

  def test_native_workgroup_coordinates(self):
    from tinygrad.uop.ops import KernelInfo, UOp
    def kernel(out):
      x,y,z = (UOp.special(n, f"lidx{i}") for i,n in enumerate((8,4,2)))
      return out.flatten().index(x+8*y+32*z).store(x.cast(dtypes.int)+y.cast(dtypes.int)*32+z.cast(dtypes.int)*1024).sink(
        arg=KernelInfo(name="s30_local_ids", opts_to_apply=()))
    with Context(DEV="MT:USC", CACHELEVEL=0):
      out = Tensor.empty(64, device="MT", dtype=dtypes.int).custom_kernel(fxn=kernel)[0]
      self.assertEqual(out.tolist(), [x+32*y+1024*z for z in range(2) for y in range(4) for x in range(8)])

  def test_native_split_grid(self):
    from tinygrad.codegen import to_program
    from tinygrad.renderer import TinyELF
    from tinygrad.runtime.support.mt.s30 import MTS30Program
    from tinygrad.uop.ops import KernelInfo, UOp
    with Context(DEV="MT:USC", CACHELEVEL=0):
      dev = Device["MT"]
      for axis in range(3):
        out = UOp.placeholder((9*32,), dtypes.int, slot=0)
        i = UOp.special(9, f"gidx{axis}")*32+UOp.special(32, "lidx0")
        ast = out[i].store(out[i]+1).sink(arg=KernelInfo(name="increment_grid", optimized=True))
        prg = MTS30Program(dev, TinyELF.from_program(to_program(ast, dev.renderer)), {})
        buf = Buffer("MT", 9*32*4, initial_value=bytes(9*32*4))
        expected = [0]*(9*32)
        for groups in (9, 1, 4):
          sizes = [1,1,1]
          sizes[axis] = groups
          prg(buf.meta.va, global_size=tuple(sizes), local_size=(32,1,1))
          for j in range(groups*32): expected[j] += 1
          self.assertEqual(list(buf.as_memoryview().cast("i")), expected)

  def test_meta_batch_and_restart(self):
    from unittest.mock import patch
    from extra.usbgpu.s30.bench_dispatch import increment
    dev = Device["MT"]
    fw = dev.s30_fw
    if not fw.meta_enabled: self.skipTest("requires native META dispatch")
    prg = increment(dev)
    rng = random.Random(31)
    for _ in range(2):
      fw.session.close()
      values = [rng.randrange(10000) for _ in range(9*32)]
      buf = Buffer("MT", 4*len(values), initial_value=struct.pack(f"<{len(values)}i", *values))
      for n in (1, 8, 64, 65):
        seq = fw.meta_sequence if fw.direct else 0
        for i in range(n):
          groups = 9 if i%2 == 0 else 1
          prg(buf.meta.va, global_size=(groups,1,1), local_size=(32,1,1))
          for j in range(groups*32): values[j] += 1
        self.assertEqual(list(buf.as_memoryview().cast("i")), values)
        self.assertEqual(fw.meta_sequence-seq, ceildiv(n,64))
      prg(buf.meta.va, global_size=(9,1,1), local_size=(32,1,1))
      buf.host[:4] = struct.pack("<i", 7)
      self.assertEqual(list(buf.as_memoryview().cast("i")), [7]+[v+1 for v in values[1:]])
      prg(buf.meta.va, global_size=(9,1,1), local_size=(32,1,1))
      def missed_doorbell(): fw.meta_sequence += 1
      with patch.object(fw, "start_direct", missed_doorbell), self.assertRaisesRegex(TimeoutError, "completed"):
        fw.flush_meta(0.01)
      self.assertFalse(fw.direct)
      self.assertFalse(fw.meta_pending)
      prg(buf.meta.va, global_size=(9,1,1), local_size=(32,1,1), wait=True)
      self.assertEqual(list(buf.as_memoryview().cast("i")), [8]+[v+2 for v in values[1:]])

  def test_native_workgroup_exchange(self):
    import numpy as np
    from tinygrad.dtype import AddrSpace
    from tinygrad.uop.ops import KernelInfo, UOp
    rng = np.random.default_rng(30)
    with Context(DEV="MT:USC", CACHELEVEL=0):
      for width in (32,64,128,256):
        def kernel(out, src):
          lane, group = UOp.special(width, "lidx0"), UOp.special(33, "gidx0")
          a,b = (UOp.placeholder((4*width,), dtypes.int, slot=i, addrspace=AddrSpace.LOCAL) for i in range(2))
          idx, start = group*width+lane, group%4*width
          fill = a[start+lane].store(src[idx]).barrier()
          swap = b[start+lane].store(a.after(fill)[start+(lane^(width//2))]+group.cast(dtypes.int)).barrier()
          return out[idx].store(b.after(swap)[start+(lane^(width//4))]).sink(arg=KernelInfo(name="s30_exchange", opts_to_apply=()))
        for _ in range(3):
          values = rng.integers(-10000, 10000, (33,width), dtype=np.int32)
          inputs = Tensor(values.flatten(), device="MT").realize()
          out = inputs.empty_like().custom_kernel(inputs, fxn=kernel)[0].numpy().reshape(33,width)
          np.testing.assert_array_equal(out, values[:,np.arange(width)^(width//2)^(width//4)]+np.arange(33)[:,None])

  def test_native_gated_store(self):
    import numpy as np
    from tinygrad.uop.ops import KernelInfo, UOp
    def kernel(out, flags):
      i = UOp.special(3, "gidx0")*64+UOp.special(64, "lidx0")
      return out[(i+2).valid(flags[i]>0)].store(i.cast(dtypes.int)+91).sink(arg=KernelInfo(name="s30_gated_store", opts_to_apply=()))
    rng = np.random.default_rng(30)
    with Context(DEV="MT:USC", CACHELEVEL=0):
      for gate in (np.zeros(192, np.int32), np.ones(192, np.int32), rng.integers(0, 2, 192, dtype=np.int32)):
        original = rng.integers(-500, 500, 196, dtype=np.int32)
        dest, flags = (Tensor(v, device="MT").realize() for v in (original, gate))
        expected = original.copy()
        active = np.flatnonzero(gate)
        expected[active+2] = active+91
        np.testing.assert_array_equal(dest.custom_kernel(flags, fxn=kernel)[0].numpy(), expected)

  def test_native_load_pressure(self):
    from tinygrad.dtype import AddrSpace
    from tinygrad.renderer import TinyELF
    from tinygrad.renderer.usc import assemble, Instruction
    from tinygrad.runtime.support.mt.s30 import MTS30Program
    # Load all inputs before the first add. The original instruction order needs more than 64 registers.
    count = 80
    ins:list[Instruction] = []
    for i in range(count): ins.extend([("imm", (), 4+i), ("ld", (2*i,), 2)])
    acc = 1
    for i in range(1, count):
      ins.append(("fadd", (acc, 2*i+1), 0))
      acc = len(ins)-1
    ins.append(("st", (0, acc), 0))
    code, temps = assemble(ins)
    self.assertLessEqual(temps, 8)
    lib = struct.pack(f"<HHI{count}I", temps, 0, count, *range(0, 4*count, 4)) + code
    with Context(DEV="MT:USC", CACHELEVEL=0):
      dev = Device["MT"]
      sig = ((None, AddrSpace.GLOBAL, dtypes.float, ()),)*2
      prg = MTS30Program(dev, TinyELF(lib, "sum_inputs", dev.renderer.target, sig), {})
      rng = random.Random(30)
      vals = rng.sample(range(-1000, 1000), count)
      src = Buffer("MT", 4*count, initial_value=struct.pack(f"<{count}f", *vals))
      dst = Buffer("MT", 4, initial_value=bytes(4))
      prg(dst.meta.va, src.meta.va, wait=True)
      self.assertEqual(struct.unpack("<f", dst.as_memoryview())[0], sum(vals))

  def test_native_interleaved_dots(self):
    import numpy as np
    from tinygrad.uop.ops import KernelInfo, UOp
    def kernel(out, a, b):
      lane = UOp.special(32, "lidx0")
      stores = [out[(lane*4+r)*4+c].store(sum(a[(lane*4+r)*16+k]*b[k*4+c] for k in range(16))) for r in range(4) for c in range(4)]
      return UOp.sink(*stores, arg=KernelInfo(name="dot_tile", optimized=True))
    rng = np.random.default_rng(30)
    a, b = rng.integers(-4, 5, (128,16)).astype(np.float32), rng.integers(-4, 5, (16,4)).astype(np.float32)
    with Context(DEV="MT:USC", CACHELEVEL=0):
      out = Tensor.empty(128*4, device="MT").custom_kernel(Tensor(a.flatten(), device="MT"), Tensor(b.flatten(), device="MT"), fxn=kernel)[0]
      np.testing.assert_array_equal(out.numpy().reshape(128,4), a@b)

  def test_native_accumulator_write(self):
    from tinygrad.dtype import AddrSpace
    from tinygrad.renderer import TinyELF
    from tinygrad.renderer.usc import assemble, Instruction
    from tinygrad.runtime.support.mt.s30 import MTS30Program
    # Both adds read the old accumulator before the first assignment. The final add can write the slot directly.
    ins:list[Instruction] = [("imm", (), 2), ("imm", (), 3), ("slot", (), 0), ("assign", (2,1), 0),
                             ("fadd", (2,1), 0), ("fadd", (2,1), 0), ("assign", (2,4), 0),
                             ("fadd", (2,5), 0), ("assign", (2,7), 0), ("st", (0,2), 0)]
    code, temps = assemble(ins)
    lib = struct.pack("<HHH2xIf", temps, 0, 2, 0, 3.)+code
    with Context(DEV="MT:USC", CACHELEVEL=0):
      dev, out = Device["MT"], Buffer("MT", 4, initial_value=bytes(4))
      prg = MTS30Program(dev, TinyELF(lib, "accumulator", dev.renderer.target, ((None, AddrSpace.GLOBAL, dtypes.float, ()),)), {})
      prg(out.meta.va, wait=True)
      self.assertEqual(struct.unpack("<f", out.as_memoryview())[0], 12.)

  def test_native_paired_projections(self):
    import numpy as np
    rng = np.random.default_rng(30)
    a,b = (rng.integers(-4, 5, (256, 192)).astype(np.float32) for _ in range(2))
    v = rng.integers(-9, 10, 192).astype(np.float32)
    with Context(DEV="MT:USC", CACHELEVEL=0):
      x, w, u = (Tensor(i, device="MT") for i in (v, a, b))
      np.testing.assert_array_equal(((x*w).sum(-1) * (x*u).sum(-1)).numpy(), (a@v)*(b@v))

  def test_native_conv2d(self):
    import numpy as np
    rng = np.random.default_rng(30)
    with Context(DEV="MT:USC", CACHELEVEL=0):
      weight = rng.integers(-3, 4, (8, 4, 3, 3)).astype(np.float32)
      for width in (3, 6, 7, 12, 15):
        x = rng.integers(-4, 5, (2, 4, 8, width+2)).astype(np.float32)
        windows = np.lib.stride_tricks.sliding_window_view(x, (3, 3), axis=(2, 3))
        expected = np.einsum("ncyxij,ocij->noyx", windows, weight)
        np.testing.assert_array_equal(Tensor(x, device="MT").conv2d(Tensor(weight, device="MT")).numpy(), expected)

  def test_native_projection(self):
    import numpy as np
    rng = np.random.default_rng(30)
    x = rng.integers(0, 256, (3, 48), dtype=np.uint8)
    w = rng.integers(-4, 5, (11, 48)).astype(np.float32)
    with Context(DEV="MT:USC", CACHELEVEL=0):
      inputs, weights = Tensor(x, device="MT"), Tensor(w, device="MT")
      out = (inputs.float().unsqueeze(1)*weights).sum(-1).relu()
      np.testing.assert_array_equal(out.numpy(), np.maximum(x@w.T, 0))
      np.testing.assert_array_equal((inputs.int() * inputs.int()).numpy(), x.astype(np.int32)**2)

  def test_native_mixed_precision(self):
    import numpy as np
    rng = np.random.default_rng(30)
    a,b,c = (rng.normal(size=257).astype(np.float16) for _ in range(3))
    with Context(DEV="MT:USC", CACHELEVEL=0):
      x,y,z = (Tensor(v, device="MT").float() for v in (a,b,c))
      af,bf,cf = (v.astype(np.float32) for v in (a,b,c))
      np.testing.assert_array_equal((x*y+z).numpy(), af*bf+cf)
      np.testing.assert_array_equal((x*y+0.25).numpy(), af*bf+0.25)
      np.testing.assert_array_equal((x*2.+z).numpy(), af*2.+cf)

  def test_native_half_and_mask(self):
    import numpy as np
    rng = np.random.default_rng(30)
    a,b = (rng.normal(size=65).astype(np.float16) for _ in range(2))
    with Context(DEV="MT:USC", CACHELEVEL=0):
      x,y = Tensor(a, device="MT"), Tensor(b, device="MT")
      np.testing.assert_array_equal((x*y).numpy(), a*b)
      np.testing.assert_array_equal((x.float()<y.float()).where(x,y).numpy(), np.minimum(a,b))
      np.testing.assert_array_equal(x.pad(((2,3),), value=2.).numpy(), np.pad(a, (2,3), constant_values=2.))
      np.testing.assert_array_equal(Tensor.full((37,37), -float("inf"), device="MT").triu(1).numpy(), np.triu(np.full((37,37), -np.inf), 1))

  def test_native_symbolic_math(self):
    import numpy as np
    data = np.arange(-32, 32, dtype=np.int32)
    with Context(DEV="MT:USC", CACHELEVEL=0):
      x, offset = Tensor(data, device="MT"), Variable("offset", -11, 17)
      run = TinyJit(lambda x,v: ((x+Tensor(v, device=x.device))>>2).float().exp2().log2().realize())
      for n in (-11,3,17,-2): np.testing.assert_allclose(run(x, offset.bind(n)).numpy(), (data+n)>>2, atol=1e-5)

  def test_native_variable_divisor(self):
    import numpy as np
    data = np.arange(4097, dtype=np.int32)
    with Context(DEV="MT:USC", CACHELEVEL=0):
      x, divisor = Tensor(data, device="MT"), Variable("divisor", 1, 448)
      @TinyJit
      def run(x, divisor):
        n, d = x & 8191, Tensor(divisor, device=x.device)
        return ((n//d)*1024 + n%d).realize()
      for d in (1,2,3,17,127,448): np.testing.assert_array_equal(run(x, divisor.bind(d)).numpy(), data//d*1024+data%d)

  def test_native_integer_lanes(self):
    import numpy as np
    rng = np.random.default_rng(30)
    with Context(DEV="MT:USC", CACHELEVEL=0):
      for dt in (np.int8,np.uint8,np.int16,np.uint16,np.uint32):
        a = rng.integers(np.iinfo(dt).min, np.iinfo(dt).max, 255, dtype=dt)
        b = rng.integers(0, 5, 255, dtype=dt)
        x,y = Tensor(a, device="MT"), Tensor(b, device="MT")
        np.testing.assert_array_equal((((x*y)>>y) ^ (x<<y)).numpy(), ((a*b)>>b) ^ (a<<b))
        np.testing.assert_array_equal(x.float().numpy(), a.astype(np.float32))

  @requires_renderer
  def test_tensor_compute(self):
    for n,dt in ((4, dtypes.int32), (257, dtypes.int32), (4099, dtypes.float32)):
      with self.subTest(n=n, dtype=dt):
        values = [(i % 31) - 15 for i in range(n)]
        cpu = Tensor([v * 0.25 for v in values] if dt == dtypes.float32 else values, device="CPU", dtype=dt)
        gpu = cpu.to("MT").realize()
        add = (gpu + 1).realize()
        self.assertEqual(add.to("CPU").tolist(), (cpu + 1).tolist())
        self.assertEqual((add * gpu).to("CPU").tolist(), ((cpu + 1) * cpu).tolist())

  @requires_renderer
  def test_batched_matmul(self):
    import numpy as np
    rng = np.random.default_rng(30)
    for rows in (64, 65):
      with self.subTest(rows=rows):
        a = rng.integers(-4, 5, (2, rows, 128)).astype(np.float32)
        b = rng.integers(-4, 5, (2, 96, 128)).astype(np.float16)
        result = Tensor(a, device="MT") @ Tensor(b, device="MT").float().transpose(-1, -2)
        np.testing.assert_array_equal(result.numpy(), a @ b.transpose(0, 2, 1))

  @requires_renderer
  def test_llm_decode_attention(self):
    import numpy as np
    from tinygrad.llm.kernels.mt import s30_decode_attention
    rng = np.random.default_rng(30)
    q = rng.normal(size=(2, 6, 1, 64)).astype(np.float32)
    kv = rng.normal(size=(2, 2, 2, 128, 64)).astype(np.float16)
    pos = Variable("pos", 0, 127)
    run = TinyJit(lambda q, kv, p: s30_decode_attention(q, kv, p).realize())
    for p in (0, 31, 127, 5):
      cache = kv.copy()
      cache[:, :, :, p+1:] = np.nan
      keys, values = (a[:, :, :p+1].astype(np.float32).repeat(3, axis=1) for a in cache)
      scores = q @ keys.swapaxes(-2, -1) / 8
      exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
      result = run(Tensor(q, device="MT"), Tensor(cache, device="MT"), pos.bind(p))
      np.testing.assert_allclose(result.numpy(), (exp / exp.sum(axis=-1, keepdims=True)) @ values, atol=2e-5, rtol=2e-5)

  @requires_renderer
  def test_llm_gemv(self):
    import numpy as np
    from tinygrad.llm.kernels.amd import Linear
    rng = np.random.default_rng(30)
    rows, k = 2048, 1024
    w = rng.integers(-4, 5, (rows, k)).astype(np.float16)
    bias = rng.integers(-8, 9, rows).astype(np.float32)
    layer = Linear(k, rows)
    layer.weight, layer.bias = Tensor(w, device="MT").realize(), Tensor(bias, device="MT").realize()
    run = TinyJit(lambda x: layer(x).realize())
    for _ in range(3):
      x = rng.integers(-4, 5, (1, 1, k)).astype(np.float32)
      np.testing.assert_array_equal(run(Tensor(x, device="MT")).numpy(), x @ w.astype(np.float32).T + bias)

  @requires_renderer
  def test_symbolic_matmul(self):
    import numpy as np
    rng = np.random.default_rng(30)
    a = rng.integers(-3, 4, (33, 256)).astype(np.float32)
    b = rng.integers(-3, 4, (128, 256)).astype(np.float16)
    x, w = Tensor(a, device="MT").realize(), Tensor(b, device="MT").realize()
    @TinyJit
    def run(x, w, n): return (x[:n] @ w.T).contiguous().realize().pad_to((33, 128)).contiguous().realize()
    rows = Variable("rows", 1, 33)
    for n in (33, 1, 8, 9, 31, 32, 7):
      np.testing.assert_array_equal(run(x, w, rows.bind(n)).numpy(), np.pad(a[:n] @ b.T, ((0, 33-n), (0, 0))))

  @requires_renderer
  def test_half_math(self):
    x = Tensor([0.25, 0.5, 1.0, 2.0], device="MT", dtype=dtypes.half)
    for op in (Tensor.sqrt, Tensor.reciprocal, Tensor.exp2, Tensor.log2, Tensor.sin, Tensor.trunc):
      with self.subTest(op=op.__name__):
        self.assertEqual(op(x).tolist(), op(x.to("CPU").float()).half().tolist())

  def test_native_trunc(self):
    import numpy as np
    with Context(DEV="MT:USC", CACHELEVEL=0):
      for dtype, uint in ((np.float16, np.uint16), (np.float32, np.uint32)):
        limit = 2**np.finfo(dtype).nmant
        values = np.array([0., -0., -0.5, 0.5, -3.75, limit-0.5, -(limit-0.5), limit, np.inf, -np.inf, np.nan], dtype=dtype)
        got = Tensor(values, device="MT").trunc().numpy()
        np.testing.assert_array_equal(got[:-1].view(uint), np.trunc(values[:-1]).view(uint))
        self.assertTrue(np.isnan(got[-1]))

  @requires_renderer
  def test_resident_programs_after_restart(self):
    @TinyJit
    def run(x, n): return (x[:n]+3).realize().pad(((0, 512-n),)).contiguous().realize()
    count = Variable("count", 1, 512)
    for i,n in enumerate((129, 129, 257, 17, 129)):
      if i == 4: Device["MT"].s30_fw.session.close()
      values = list(range(i, i+512))
      self.assertEqual(run(Tensor(values, device="MT"), count.bind(n)).tolist(), [x+3 for x in values[:n]] + [0]*(512-n))

  @requires_renderer
  def test_grid(self):
    def run(device):
      x = Tensor(list(range(7*16*12)), device=device).reshape(7,16,12)
      return (x.permute(2,0,1).contiguous().realize() * (x+1).permute(2,0,1)).sum(axis=2).tolist()
    self.assertEqual(run("MT"), run("CPU"))


if __name__ == "__main__": unittest.main()
