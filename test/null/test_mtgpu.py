import ctypes, struct, unittest
from unittest.mock import Mock, patch
from tinygrad.device import BufferSpec
from tinygrad.runtime.ops_mt import MTAllocator
from tinygrad.runtime.support.memory import MMIOInterface
from tinygrad.runtime.support.mt.mtdev import MTSMC, MTDev, MT_IPC_EVENT_GET_FW_VERSION, MT_IPC_EVENT_SET_POWER_STATE, _Ring, _ipc_header


def make_mmio(s30=False):
  buf = (ctypes.c_ubyte * 0x84000)()
  if s30:
    for ctrl in (0x6f000, 0x6f214): struct.pack_into('<5I', buf, ctrl, 0, 0, 0x200, ctrl + 0x14, 0x7f)
  else:
    for ctrl in (0x30000, 0x30218): struct.pack_into('<6I', buf, ctrl, 0, 0, 0, 0x200, ctrl + 0x18, 0x7f)
  mmio = MMIOInterface(ctypes.addressof(buf), len(buf))
  mmio.owner = buf  # keep the backing allocation alive for the lifetime of the view
  return mmio


class MockSMC(MTSMC):
  def __init__(self, s30=False):
    super().__init__(make_mmio(s30), s30)
    self.requests = []
    self.response_error = 0

  def _kick(self, event_id, target, msg_id):
    header, payload, next_rd = self.tx.read_message()
    self._w32(self.tx.ctrl, next_rd)
    self.requests.append((event_id, target, msg_id, payload))
    if event_id == MT_IPC_EVENT_GET_FW_VERSION:
      response = b'rtos\0\0\0\0' + struct.pack('<I', 20107) + b'vbios\0\0\0' + struct.pack('<I', 10606)
    else: response = bytes(12)
    header = _ipc_header(event_id, len(response) // 4, msg_id=msg_id, msg_type=1) | (self.response_error << 53)
    self.rx.write(struct.pack('<Q', header) + response)


class TestMTSMC(unittest.TestCase):
  def test_version_and_pstate(self):
    smc = MockSMC()
    self.assertEqual(smc.rtos_version(), [('rtos', 20107), ('vbios', 10606)])
    smc.set_pstate(0)
    self.assertEqual([x[:3] for x in smc.requests], [(MT_IPC_EVENT_GET_FW_VERSION, 0, 0), (MT_IPC_EVENT_SET_POWER_STATE, 0, 1)])
    self.assertEqual(smc.requests[1][3], struct.pack('<III', 0, 0, 0))

  def test_ring_wrap(self):
    for s30 in (False, True):
      with self.subTest(s30=s30):
        smc = MockSMC(s30)
        for ring in (smc.tx, smc.rx):
          smc._w32(ring.ctrl, 0x1f8 * 2)
          smc._w32(ring.ctrl + 4, 0x1f8 * 2)
        self.assertEqual(smc.rtos_version(), [('rtos', 20107), ('vbios', 10606)])
        self.assertEqual(smc._r32(smc.tx.ctrl), (108 * 2) | 1)
        self.assertEqual(smc._r32(smc.tx.ctrl + 4), (108 * 2) | 1)
        self.assertEqual(smc._r32(smc.rx.ctrl), (24 * 2) | 1)

  def test_ipc_error(self):
    smc = MockSMC()
    smc.response_error = 1
    with self.assertRaisesRegex(RuntimeError, 'failed with error 1'): smc.set_pstate(0)

  def test_bad_ring(self):
    mmio = make_mmio()
    smc = MTSMC(mmio)
    smc._w32(0x30010, 0x3001c)
    with self.assertRaisesRegex(RuntimeError, 'data pointer'): _Ring(smc, 0x30000)


class TestMTDev(unittest.TestCase):
  def test_s30_direct_attach_without_smc(self):
    pci = Mock()
    pci.read_config.return_value = 0x01021ed5
    pci.bar_info.return_value = (0, 4<<30)
    regs = bytearray(0x301000)
    for base in (0x200000, 0x300000): struct.pack_into('<Q', regs, base+0x20, 0x0023000406600017)
    pci.map_bar.return_value = regs
    with patch.object(MTSMC, "__init__", side_effect=AssertionError("opened management mailbox")):
      dev = MTDev(pci, manage_power=False)
      self.assertEqual(dev.bvnc, (35, 4, 1632, 23))
      dev.close()

  def test_vram_reuse(self):
    dev = Mock(views={})
    dev.mt.vram_size = 0x80003000
    alloc, spec = MTAllocator(dev), BufferSpec(nolru=True)
    a, b = alloc.alloc(0x2000, spec), alloc.alloc(0x1000, spec)
    alloc.free(a, 0x2000, spec)
    alloc.free(b, 0x1000, spec)
    dev.map_buffer.side_effect = MemoryError
    with self.assertRaises(MemoryError): alloc.alloc(0x3000, spec)
    dev.map_buffer.side_effect = None
    self.assertEqual(alloc.alloc(0x3000, spec).meta.pa, 0x80000000)

  def test_bvnc(self):
    dev = MTDev.__new__(MTDev)
    dev.core_ids = [0x0023000406600017]
    self.assertEqual(dev.bvnc, (35, 4, 1632, 23))


if __name__ == '__main__': unittest.main()
