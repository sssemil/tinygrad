import os, random, struct, unittest
from pathlib import Path
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.device import Buffer
from tinygrad.helpers import DEV
from tinygrad.runtime.support.mt.compiler import MTCompiler
from tinygrad.runtime.support.mt.compute import FW_DEVVA, META_BASE
from tinygrad.runtime.support.mt.mmu import MT_PTE_READ_WRITE
from tinygrad.runtime.support.mt.s30 import DEFAULT_S30_FIRMWARE, PDS_BASE, USC_BASE

has_firmware = Path(os.getenv("MT_FIRMWARE", DEFAULT_S30_FIRMWARE)).is_file()
requires_firmware = unittest.skipUnless(has_firmware, "requires mtgpu-s30 3.1.0-6 mtfw-gen1.bin")
requires_compiler = unittest.skipUnless(has_firmware and (MTCompiler().bin/'mcc').is_file(), "requires S30 firmware and the MUSA toolkit")


@unittest.skipUnless(DEV.device == "MT", "run explicitly with DEV=MT on a USB Chestnut with an S30")
class TestS30USB(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    if not Device["MT"].mt.s30: raise unittest.SkipTest("requires a two-core MTT S30")

  def setUp(self): Device["MT"].s30_fw.session.close()

  def test_firmware(self):
    mt = Device["MT"].mt
    self.assertEqual(len(mt.core_ids), 2)
    self.assertTrue(all(mt.core_ids))
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
    fw = Device["MT"].s30_fw
    expected = bytearray(b'\xa5' * 64)
    out = Buffer("MT", len(expected), initial_value=expected)
    # USC: st [sh14:15], sh18; wait; end.
    usc = Buffer("MT", 12, initial_value=bytes.fromhex("4e2ac341400040c01c070080"))
    fw.mmu.map(USC_BASE, usc.meta.pa, MT_PTE_READ_WRITE)
    data = bytearray(64+19*4)
    struct.pack_into('<QI', data, 0, PDS_BASE+64, 0x90000013)  # DMA 19 shared registers
    struct.pack_into('<I', data, 16, 0xf4001000)  # DOUTD
    struct.pack_into('<Q', data, 32, 0x1000000000)
    struct.pack_into('<I', data, 48, 0xf4000002)  # DOUTU
    pds = Buffer("MT", len(data), initial_value=data)
    fw.mmu.map(PDS_BASE, pds.meta.pa, MT_PTE_READ_WRITE)
    stream = (2, 2, 16, 0, 0, 0, 0x02000002, 0, 0x101003, 32, 48, 0, 0, 0, 0x02000002, 0)
    with fw.boot():
      for i in range(3):
        resp = fw.cdm(i+1, ())
        self.assertEqual((resp.value, resp.token), (0, i+1))
        self.assertEqual(bytes(out.as_memoryview()), expected)
      # Repeat the shared-register load so the second shader store executes near the end of the 2 KiB ring.
      for i,(off,value,commands) in enumerate(((16, 43, stream), (40, 0x89abcdef, stream[:8]*59 + stream))):
        pds.host[64+14*4:] = struct.pack('<Q8xI', out.meta.va+off, value)
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

  def test_tensor_copy(self):
    original = [13, 21, 34, 55]
    resident = Tensor(original, device="CPU", dtype=dtypes.int32).to("MT").realize()
    self.assertEqual(resident.to("CPU").tolist(), original)

  @requires_compiler
  def test_tensor_compute(self):
    for n,dt in ((4, dtypes.int32), (257, dtypes.int32), (4099, dtypes.float32)):
      with self.subTest(n=n, dtype=dt):
        values = [(i % 31) - 15 for i in range(n)]
        cpu = Tensor([v * 0.25 for v in values] if dt == dtypes.float32 else values, device="CPU", dtype=dt)
        gpu = cpu.to("MT").realize()
        add = (gpu + 1).realize()
        self.assertEqual(add.to("CPU").tolist(), (cpu + 1).tolist())
        self.assertEqual((add * gpu).to("CPU").tolist(), ((cpu + 1) * cpu).tolist())

  @requires_compiler
  def test_resident_programs_after_restart(self):
    @TinyJit
    def run(x): return ((x+3).realize() * 7).realize()
    for i in range(4):
      if i == 3: Device["MT"].s30_fw.session.close()
      values = list(range(i, i+257))
      self.assertEqual(run(Tensor(values, device="MT")).tolist(), [(x+3)*7 for x in values])

  @requires_compiler
  def test_grid(self):
    def run(device):
      x = Tensor(list(range(7*16*12)), device=device).reshape(7,16,12)
      return (x.permute(2,0,1).contiguous().realize() * (x+1).permute(2,0,1)).sum(axis=2).tolist()
    self.assertEqual(run("MT"), run("CPU"))


if __name__ == "__main__": unittest.main()
