import random, struct, unittest
from tinygrad import Context, Device, Tensor, TinyJit, Variable, UOp, dtypes
from tinygrad.device import Buffer
from tinygrad.codegen import to_program
from tinygrad.helpers import DEV
from tinygrad.renderer import TinyELF
from tinygrad.runtime.support.mt.s30 import MTS30Program
from tinygrad.uop.ops import KernelInfo

def increment(dev):
  out = UOp.placeholder((9*32,), dtypes.int, slot=0)
  i = UOp.special(9, "gidx0")*32 + UOp.special(32, "lidx0")
  ast = out[i].store(out[i]+1).sink(arg=KernelInfo(name="increment_grid", optimized=True))
  return MTS30Program(dev, TinyELF.from_program(to_program(ast, dev.renderer)), {})

@unittest.skipUnless(DEV.device == "MT", "requires DEV=MT on Chestnut")
class TestS30USB(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    if not Device["MT"].mt.s30: raise unittest.SkipTest("requires an S30")

  def setUp(self): Device["MT"].s30_fw.session.close()

  def test_core_ids(self):
    cores = Device["MT"].mt.core_ids
    self.assertEqual(len(cores), 2)
    self.assertTrue(all(cores))

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

  def test_meta_queued_uploads(self):
    from tinygrad.runtime.support.usb import SLOT
    dev = Device["MT"]
    fw = dev.s30_fw
    if not fw.meta_enabled: self.skipTest("requires native META dispatch")
    prg = increment(dev)
    buf = Buffer("MT", SLOT, initial_value=bytes(SLOT))
    dev.synchronize()
    sequence = fw.meta_sequence
    source = bytearray(random.Random(30).randbytes(SLOT))
    expected = source.copy()
    buf.host[:] = source
    source[:] = bytes(SLOT)
    prg(buf.meta.va, global_size=(9,1,1), local_size=(32,1,1))
    for i in range(9*32): struct.pack_into("<I", expected, 4*i, (struct.unpack_from("<I", expected, 4*i)[0]+1) & 0xffffffff)
    buf.host[4:16] = expected[4:16] = bytes(range(12))
    self.assertEqual(fw.meta_sequence, sequence)
    fw.session.close()
    self.assertEqual(fw.meta_sequence, sequence+1)
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
    from tinygrad.runtime.support.mt.compiler import MTCompiler
    from tinygrad.runtime.support.mt.compute import MTFirmware
    rng = np.random.default_rng(30)
    with Context(DEV="MT:USC", CACHELEVEL=0), patch.object(MTCompiler, "compile_raw", side_effect=AssertionError("called mcc")), \
         patch.object(MTFirmware, "start", side_effect=AssertionError("loaded vendor compute firmware")):
      self.assertEqual(Device["MT"].renderer.target.renderer, "USC")
      for n in (1, 32, 256, 257, 4099):
        for dt in (np.int32, np.float32):
          with self.subTest(n=n, dtype=dt):
            a,b = (rng.integers(-(1<<31), 1<<31, n, dtype=dt) if dt == np.int32 else rng.normal(size=n).astype(dt) for _ in range(2))
            np.testing.assert_array_equal((Tensor(a, device="MT")+Tensor(b, device="MT")).numpy(), a+b)
      self.assertTrue(Device["MT"].s30_fw.direct)

  def test_native_workgroup_coordinates(self):
    from tinygrad.uop.ops import KernelInfo, UOp
    def kernel(out):
      x,y,z = (UOp.special(n, f"lidx{i}") for i,n in enumerate((8,4,2)))
      return out.flatten().index(x+8*y+32*z).store(x.cast(dtypes.int)+y.cast(dtypes.int)*32+z.cast(dtypes.int)*1024).sink(
        arg=KernelInfo(name="s30_local_ids", opts_to_apply=()))
    with Context(DEV="MT:USC", CACHELEVEL=0):
      out = Tensor.empty(64, device="MT", dtype=dtypes.int).custom_kernel(fxn=kernel)[0]
      self.assertEqual(out.tolist(), [x+32*y+1024*z for z in range(2) for y in range(4) for x in range(8)])

  def test_meta_batch_and_restart(self):
    from unittest.mock import patch
    dev = Device["MT"]
    fw = dev.s30_fw
    if not fw.meta_enabled: self.skipTest("requires native META dispatch")
    prg = increment(dev)
    rng = random.Random(31)
    for _ in range(2):
      fw.session.close()
      values = [rng.randrange(10000) for _ in range(9*32)]
      buf = Buffer("MT", 4*len(values), initial_value=struct.pack(f"<{len(values)}i", *values))
      for n in (1, 8, 64, 65, 256, 512):
        seq = fw.meta_sequence if fw.direct else 0
        for i in range(n):
          groups = 9 if i%2 == 0 else 1
          prg(buf.meta.va, global_size=(groups,1,1), local_size=(32,1,1))
          for j in range(groups*32): values[j] += 1
        self.assertEqual(list(buf.as_memoryview().cast("i")), values)
        if n <= 256: self.assertEqual(fw.meta_sequence-seq, 1)
        else: self.assertGreater(fw.meta_sequence-seq, 1)
      prg(buf.meta.va, global_size=(9,1,1), local_size=(32,1,1))
      buf.host[:4] = struct.pack("<i", 7)
      self.assertEqual(list(buf.as_memoryview().cast("i")), [7]+[v+1 for v in values[1:]])
      prg(buf.meta.va, global_size=(9,1,1), local_size=(32,1,1))
      def missed_doorbell(timestamps): fw.meta_sequence += 1
      with patch.object(fw, "start_direct", missed_doorbell), self.assertRaisesRegex(TimeoutError, "completed"):
        fw.flush_meta(0.01)
      self.assertFalse(fw.direct)
      self.assertFalse(fw.meta_pending)
      prg(buf.meta.va, global_size=(9,1,1), local_size=(32,1,1), wait=True)
      self.assertEqual(list(buf.as_memoryview().cast("i")), [8]+[v+2 for v in values[1:]])

  def test_meta_resident_replay(self):
    from unittest.mock import patch
    from tinygrad.runtime.support.usb import HALF, SLOT
    dev = Device["MT"]
    fw = dev.s30_fw
    if not fw.meta_enabled: self.skipTest("requires native META dispatch")
    prg = increment(dev)
    bufs = [Buffer("MT", 9*32*4, initial_value=bytes(9*32*4)) for _ in range(2)]
    expected = [[0]*(9*32) for _ in bufs]
    with patch.object(fw.bridge, "scsi_write", wraps=fw.bridge.scsi_write) as write:
      def run(last:int, groups:int):
        write.reset_mock()
        for which, count in [(0,9)]*23+[(last,groups)]:
          prg(bufs[which].meta.va, global_size=(count,1,1), local_size=(32,1,1))
          for j in range(count*32): expected[which][j] += 1
        dev.synchronize()
        for buf, vals in zip(bufs, expected): self.assertEqual(list(buf.as_memoryview().cast("i")), vals)
        return [(len(args[0]), kw["slot_start"]) for args,kw in write.call_args_list]

      cold, = run(0, 9)
      self.assertGreater(cold[0], SLOT)
      self.assertEqual(cold[1], HALF//SLOT)
      self.assertEqual(run(0, 9), [])
      for last, groups in ((1,9), (1,5), (0,1)):
        (patch_size, slot), = run(last, groups)
        self.assertLess(patch_size, cold[0])
        self.assertGreater(slot, cold[1])
        self.assertEqual(run(last, groups), [])
      fw.session.close()
      self.assertEqual(run(0, 1), [cold])

  def test_meta_timestamps(self):
    from unittest.mock import patch
    from tinygrad.helpers import perf_counter_us
    dev = Device["MT"]
    fw = dev.s30_fw
    if not fw.meta_enabled: self.skipTest("requires native META dispatch")
    prg = increment(dev)
    buf = Buffer("MT", 9*32*4, initial_value=bytes(9*32*4))
    for count in (4, 8):
      fw.session.close()
      with Context(PROFILE=1), patch.object(dev, "profile_events", []) as events:
        before = perf_counter_us()
        for _ in range(3): self.assertIsNone(prg(buf.meta.va, global_size=(9,1,1), local_size=(32,1,1)))
        elapsed = prg(buf.meta.va, global_size=(9,1,1), local_size=(32,1,1), wait=True)
        after = perf_counter_us()
        self.assertEqual([(e.name, e.device, e.profile_key) for e in events], [(prg.name, dev.device, prg.profile_key)]*4)
        for e in events:
          self.assertLessEqual(before, e.st)
          self.assertLess(e.st, e.en)
          before = e.en
        self.assertLessEqual(before, after)
        self.assertAlmostEqual(elapsed, float(events[-1].en-events[-1].st)*1e-6, places=12)
      self.assertEqual(list(buf.as_memoryview().cast("i")), [count]*(9*32))

  def test_meta_bulk_completion(self):
    from unittest.mock import patch
    from tinygrad.runtime.support.usb import SLOT
    dev = Device["MT"]
    fw = dev.s30_fw
    if not fw.meta_enabled: self.skipTest("requires native META dispatch")
    prg = increment(dev)
    buf = Buffer("MT", SLOT, initial_value=bytes(SLOT))
    dev.synchronize()
    fw.meta_sequence = 62
    with Context(PROFILE=1), patch.object(dev, "profile_events", []) as events:
      for _ in range(256): prg(buf.meta.va, global_size=(9,1,1), local_size=(32,1,1))
      dev.synchronize()
      self.assertEqual(fw.meta_sequence, 63)
      self.assertEqual(len(events), 256)  # the last timer pair crosses the first SRAM page
      self.assertTrue(all(e.st < e.en for e in events))
      self.assertTrue(all(a.en <= b.st for a,b in zip(events, events[1:])))
      expected = struct.pack("<I", 256)*(9*32)+bytes(SLOT-9*32*4)
      self.assertEqual(bytes(buf.as_memoryview()), expected)
      self.assertEqual(fw.meta_sequence, 64)  # staged data and its profile survive the six-bit sequence wrap
    read = fw.bridge.scsi_read
    def stale(dest, *args, **kwargs):
      read(dest, *args, **kwargs)
      dest[0] ^= 1
    with patch.object(fw.bridge, "scsi_read", stale), self.assertRaisesRegex(RuntimeError, "stale bulk completion"):
      prg(buf.meta.va, global_size=(9,1,1), local_size=(32,1,1), wait=True)
    self.assertFalse(fw.direct)
    prg(buf.meta.va, global_size=(9,1,1), local_size=(32,1,1), wait=True)
    self.assertEqual(bytes(buf.as_memoryview()), struct.pack("<I", 258)*(9*32)+bytes(SLOT-9*32*4))

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

  def test_native_dot_burst(self):
    import numpy as np
    from tinygrad.codegen import to_program
    from tinygrad.renderer import TinyELF
    from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp
    # Offset one input by a dword. Some bursts cross page boundaries. The values distinguish ordered and reassociated sums.
    rng = np.random.default_rng(30)
    dev = Device["MT"]
    with Context(DEV="MT:USC", CACHELEVEL=0):
      for width in (64, 65, 72):
        a = rng.integers(-16, 17, 64*width+1).astype(np.float32)
        a[1::8], a[3::8] = 2**26, -2**26
        b = rng.integers(-4, 5, width).astype(np.float32)
        expected = np.zeros(64, dtype=np.float32)
        for k in range(width): expected += a[1+k::width]*b[k]
        out, x, y = (UOp.placeholder((n,), dtypes.float, slot=i) for i,n in enumerate((64,64*width,width)))
        row, k = UOp.special(2, "gidx0")*32+UOp.special(32, "lidx0"), UOp.range(width, 0, AxisType.LOOP)
        dot = (x[row*width+k]*y[k]).reduce(k, arg=Ops.ADD)
        obj = TinyELF.from_program(to_program(out[row].store(dot).sink(arg=KernelInfo(name="dot_offset", optimized=True)), dev.renderer))
        dst, lhs, rhs = (Buffer("MT", v.nbytes, initial_value=v.tobytes()) for v in (np.full_like(expected, np.nan), a, b))
        dev.runtime(obj)(dst.meta.va, lhs.meta.va+4, rhs.meta.va, global_size=(2,1,1), local_size=(32,1,1), wait=True)
        np.testing.assert_array_equal(np.frombuffer(dst.as_memoryview(), np.float32), expected)

  def test_native_unaligned_burst_load(self):
    import numpy as np
    from dataclasses import replace
    from unittest.mock import patch
    from tinygrad.renderer.usc import _assemble
    from tinygrad.uop.ops import Ops
    dev = Device["MT"]
    rng = np.random.default_rng(31)
    for width in (4, 8):
      n = 256
      src = rng.integers(0, 2**32, n*(width+1), dtype=np.uint32)
      x, out = (UOp.placeholder((size,), dtypes.uint, slot=slot) for size,slot in ((src.size,1),(n*width,0)))
      i = UOp.special(n//32, "gidx0")*32+UOp.special(32, "lidx0")
      stores = [out[i*width+j].store(x[i*(width+1)+width-j]) for j in range(width)]
      prg = to_program(UOp.sink(*stores, arg=KernelInfo(name="reverse_burst", optimized=True)), dev.renderer)
      self.assertTrue(any(u.op is Ops.LOAD and u.shape == (width,) for u in prg.src[1].src))
      obj = TinyELF.from_program(prg)
      with patch("tinygrad.renderer.usc._assemble", side_effect=lambda ins: None if any(op.startswith("ldx") for op,_,_ in ins) else _assemble(ins)):
        scalar = bytes.fromhex(dev.renderer.render(list(prg.src[1].src)))
      self.assertNotEqual(obj.lib, scalar)
      buf = Buffer("MT", src.nbytes, initial_value=src.tobytes())
      for lib in (obj.lib, scalar):
        dst = Buffer("MT", n*width*4, initial_value=bytes(n*width*4))
        dev.runtime(replace(obj, lib=lib))(dst.meta.va, buf.meta.va, global_size=(n//32,1,1), local_size=(32,1,1), wait=True)
        np.testing.assert_array_equal(np.frombuffer(dst.as_memoryview(), np.uint32).reshape(n,width), src.reshape(n,width+1)[:,width:0:-1])

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

  def test_native_mixed_precision(self):
    import numpy as np
    rng = np.random.default_rng(30)
    a,b,c = (rng.normal(size=257).astype(np.float16) for _ in range(3))
    with Context(DEV="MT:USC", CACHELEVEL=0):
      x,y,z = (Tensor(v, device="MT") for v in (a,b,c))
      np.testing.assert_array_equal((x*y+z).numpy(), a*b+c)
      np.testing.assert_array_equal((x<y).numpy(), a<b)
      x,y,z = x.float(), y.float(), z.float()
      af,bf,cf = (v.astype(np.float32) for v in (a,b,c))
      np.testing.assert_array_equal((x*y+z).numpy(), af*bf+cf)
      np.testing.assert_array_equal((x*y+0.25).numpy(), af*bf+0.25)
      np.testing.assert_array_equal((x*2.+z).numpy(), af*2.+cf)

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
      # Long arithmetic decomposes into word operations before multiply-add fusion.
      a,b,c = (rng.integers(-(1<<40), 1<<40, 255, dtype=np.int64) for _ in range(3))
      x,y,z = (Tensor(v, device="MT") for v in (a,b,c))
      np.testing.assert_array_equal((x*y+z).numpy(), a*b+c)

  def test_batched_matmul(self):
    import numpy as np
    rng = np.random.default_rng(30)
    for rows in (64, 65):
      with self.subTest(rows=rows):
        a = rng.integers(-4, 5, (2, rows, 128)).astype(np.float32)
        b = rng.integers(-4, 5, (2, 96, 128)).astype(np.float16)
        result = Tensor(a, device="MT") @ Tensor(b, device="MT").float().transpose(-1, -2)
        np.testing.assert_array_equal(result.numpy(), a @ b.transpose(0, 2, 1))

  def test_resident_programs_after_restart(self):
    @TinyJit
    def run(x, n): return (x[:n]+3).realize().pad(((0, 512-n),)).contiguous().realize()
    count = Variable("count", 1, 512)
    for i,n in enumerate((129, 129, 257, 17, 129)):
      if i == 4: Device["MT"].s30_fw.session.close()
      values = list(range(i, i+512))
      self.assertEqual(run(Tensor(values, device="MT"), count.bind(n)).tolist(), [x+3 for x in values[:n]] + [0]*(512-n))

  def test_grid(self):
    def run(device):
      x = Tensor(list(range(7*16*12)), device=device).reshape(7,16,12)
      return (x.permute(2,0,1).contiguous().realize() * (x+1).permute(2,0,1)).sum(axis=2).tolist()
    self.assertEqual(run("MT"), run("CPU"))

  def test_integer_division(self):
    from tinygrad.uop import Ops
    from tinygrad.helpers import cdiv, cmod
    a = [-2147483648, -125, 13, 2147483647, 0, -3, 8, -2147483648]
    b = [-2147483648, 4, -4, 3, 1, 0, 0, -1]
    for dtype in (dtypes.int, dtypes.uint):
      if dtype == dtypes.uint: a,b = [[n & 0xffffffff for n in vals] for vals in (a,b)]
      x,y = (Tensor(vals, device="MT", dtype=dtype) for vals in (a,b))
      for op,ref in ((Ops.TRUNCDIV, cdiv), (Ops.TRUNCMOD, cmod)):
        got = Tensor(x.uop.alu(op,y.uop)).tolist()
        self.assertEqual([v & 0xffffffff for v in got], [ref(n,d) & 0xffffffff for n,d in zip(a,b)])
