import ctypes, struct, unittest
from unittest.mock import Mock, patch
from tinygrad.helpers import mv_address
from tinygrad.runtime.autogen import libusb
from tinygrad.runtime.support.system import System
from tinygrad.runtime.support.usb import USB3, CustomASM24Controller, USBMMIOInterface, alloc_cbuffer, STREAM

class TestUSBTransfers(unittest.TestCase):
  def test_bulk_write_memoryview_and_readonly_fallback(self):
    usb = USB3.__new__(USB3)
    usb.handle = None
    usb._bulk_buf, usb._bulk_mv = alloc_cbuffer(32)
    usb._transferred = ctypes.c_int(0)
    calls = []
    def transfer(handle, endpoint, data, length, transferred, timeout):
      calls.append((ctypes.addressof(data), length))
      transferred.value = length
      return 0
    with patch.object(libusb, 'libusb_bulk_transfer', transfer):
      payload = bytearray(b'abcd')
      usb.bulk_write(payload)
      self.assertEqual(calls[-1], (mv_address(memoryview(payload)), 4))
      usb.bulk_write(b'xy')
      self.assertEqual(calls[-1], (ctypes.addressof(usb._bulk_buf), 2))

  def test_pcie_reads_are_bounded_and_ordered(self):
    requests, reads = [], []
    class FakeUSB:
      def bulk_read(self, size, timeout=1000):
        reads.append(size)
        return bytes([len(reads)]) * size
    controller = CustomASM24Controller.__new__(CustomASM24Controller)
    controller.usb = FakeUSB()
    controller._f0_out = lambda fmt, byte_en, address, count, mode: requests.append((fmt,byte_en,address,count,mode))

    addr = (1<<32) + 0x1000
    result = controller.pcie_mem_read(addr,4 * STREAM + 4)

    self.assertEqual(requests,[(0x20,0x0f,addr,STREAM,2),(0x20,0x0f,addr + 4 * STREAM,1,2)])
    self.assertEqual(reads,[4 * STREAM,4])
    self.assertEqual(len(result),4 * STREAM + 4)
    self.assertEqual(result[:4],b'\x01' * 4)
    self.assertEqual(result[-4:],b'\x02' * 4)

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

  def test_unaligned_write_reads_only_edges(self):
    usb = Mock()
    base = 0x1001
    for offset, length, reads in ((0, 2, [0x1000]), (2, 9, [0x1000]), (3, 9, [0x100c]), (0, 10, [0x1000,0x1008]), (0, 0, [])):
      mem = bytearray(range(32))
      usb.reset_mock()
      usb.pcie_mem_read.side_effect = lambda addr, n: mem[addr-0x1000:addr-0x1000+n]
      def write(addr, data): mem[addr-0x1000:addr-0x1000+len(data)] = data
      usb.pcie_mem_write.side_effect = write
      USBMMIOInterface(usb, base, 16, 'B')[offset:offset+length] = bytes(length)
      self.assertEqual([call.args for call in usb.pcie_mem_read.call_args_list], [(addr, 4) for addr in reads])
      self.assertEqual(mem, bytes(range(1+offset))+bytes(length)+bytes(range(1+offset+length, 32)))

  def test_read_from_unaligned_view(self):
    usb = Mock()
    usb.pcie_mem_read.side_effect = lambda addr, size: bytes(range(addr-0x1000, addr-0x1000+size))
    view = USBMMIOInterface(usb, 0x1003, 64, 'B')
    self.assertEqual(bytes(view[1:6]), bytes(range(4,9)))
    usb.pcie_mem_read.assert_called_once_with(0x1004, 8)
    self.assertEqual(bytes(view[:0]), b'')

  def test_large_read_across_4g(self):
    limit = 4 << 20
    ctrl = CustomASM24Controller.__new__(CustomASM24Controller)
    ctrl.usb = Mock()
    backing = bytearray(limit)
    def read(size, timeout):
      if size == 4: backing[:4] = b'abcd'
      return memoryview(backing)[:size]
    ctrl.usb.bulk_read.side_effect = read
    for first in (limit, 8):
      backing[:4] = bytes(4)
      ctrl.usb.reset_mock()
      self.assertEqual(ctrl.pcie_mem_read((1<<32)-first, first+4), bytes(first)+b'abcd')
      self.assertEqual([call.args for call in ctrl.usb.control_write.call_args_list],
                       [(0xf0, 0xf00, 2, struct.pack('<III', (1<<32)-first, 0, first//4), 5000),
                        (0xf0, 0xf20, 2, struct.pack('<III', 0, 1, 1), 5000)])

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

if __name__ == '__main__': unittest.main()
