import struct, unittest
from unittest.mock import Mock
from tinygrad.runtime.support.system import System
from tinygrad.runtime.support.usb import CustomASM24Controller


class TestUSBPCI(unittest.TestCase):
  def test_memory_request_header(self):
    ctrl = CustomASM24Controller.__new__(CustomASM24Controller)
    ctrl.usb = Mock()
    for addr, read, write in ((0xfffffff8, 0x00, 0x40), (0x100000000, 0x20, 0x60)):
      payload = struct.pack('<III', addr & 0xffffffff, addr >> 32, 1)
      ctrl.pcie_mem_read(addr, 4)
      ctrl.usb.control_write.assert_called_with(0xf0, 0xf00 | read, 2, payload, 5000)
      ctrl.usb.reset_mock()
      ctrl.pcie_mem_write(addr, struct.pack('<I', 0x89abcdef))
      ctrl.usb.control_write.assert_called_once_with(0xf0, 0xf00 | write, 0, struct.pack('<III', addr & 0xffffffff, addr >> 32, 0x89abcdef), 5000)
      ctrl.usb.bulk_write.assert_not_called()
      ctrl.pcie_mem_write(addr, bytes(8))
      ctrl.usb.control_write.assert_called_with(0xf0, 0xf00 | write, 1, struct.pack('<III', addr & 0xffffffff, addr >> 32, 2), 5000)
      ctrl.usb.bulk_write.assert_called_once_with(bytes(8))

  def test_rebar_selection_and_rollback(self):
    for device_id, fail in ((0x01021ed5, False), (0x01021ed5, True), (0x02011ed5, False)):
      with self.subTest(device_id=device_id, fail=fail):
        regs = {0:device_id, 4:0x107, 0x100:0x18010001, 0x180:0x10015, 0x184:0x200, 0x188:0x540, 0x18c:0x1f000, 0x190:0x802}
        regs.update(dict.fromkeys(range(0x10, 0x28, 4), 0))
        regs[0x18] = 0x0c  # prefetchable, 64-bit BAR2
        original = regs.copy()
        def cfg(off, bus, dev=0, fn=0, value=None, size=4):
          if bus != 2: return
          if value is None: return regs[off]
          if off != 4: self.assertFalse(regs[4] & 2)
          if value == 0xffffffff:
            if fail and off == 0x18: raise OSError("BAR probe failed")
            value = {0x10:0xfe000000, 0x18:0x0c if regs[0x190] == 0xc02 else 0xf000000c, 0x1c:0xffffffff}.get(off, 0)
          regs[off] = value
        usb = CustomASM24Controller.__new__(CustomASM24Controller)
        usb.pcie_cfg_req = cfg
        args = (usb, 2, 0x10000000, 32 << 30, {0x01021ed5:2})
        if fail:
          with self.assertRaisesRegex(OSError, "probe failed"): System.pci_setup_usb_bars(*args)
          self.assertEqual(regs, original)
        else:
          self.assertEqual(System.pci_setup_usb_bars(*args)[2], (32 << 30, (4 << 30) if device_id == 0x01021ed5 else (256 << 20)))
          self.assertEqual(regs[4], original[4])
