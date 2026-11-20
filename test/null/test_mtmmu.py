import unittest
from unittest.mock import Mock
from tinygrad.runtime.support.mt.compute import MTFirmwareBoot, ROOT_BASE
from tinygrad.runtime.support.mt.mmu import MTMMU4, MT_PTE_READ_ONLY, MT_PTE_READ_WRITE

class TestMTMMU4(unittest.TestCase):
  def test_map_and_share_tables(self):
    m = MTMMU4(0x30000000)
    va, pa = 0xe1c0000000, 0x20000000
    m.map(va, pa, MT_PTE_READ_WRITE)
    m.map(va + 0x1000, pa + 0x1000, MT_PTE_READ_WRITE)
    pci = (va >> 30) & 0x3ff
    pdi = (va >> 21) & 0x1ff
    self.assertEqual(int.from_bytes(m.root[pci*4:pci*4+4], 'little'), ((m.pd[pci] >> 12) << 4) | 1)
    pd_off = m.pd[pci] - m.root_pa + pdi * 8
    self.assertEqual(int.from_bytes(m.root[pd_off:pd_off+8], 'little'), m.pt[pci,pdi] | 1)
    for i in range(2):
      off = m.pt[pci,pdi] - m.root_pa + i * 8
      self.assertEqual(int.from_bytes(m.root[off:off+8], 'little'), pa + i * 0x1000 | MT_PTE_READ_WRITE)
    self.assertEqual(len(m.pd), 1)
    self.assertEqual(len(m.pt), 1)
    self.assertEqual(m.pages[va + 0x1000], pa + 0x1000)

  def test_read_only_and_range(self):
    m = MTMMU4(0x30000000)
    m.map_range(0x8000000000, 0x40000000, 0x3000)
    for i in range(3):
      va = 0x8000000000 + i * 0x1000
      pci, pdi = (va >> 30) & 0x3ff, (va >> 21) & 0x1ff
      off = m.pt[pci,pdi] - m.root_pa + i * 8
      self.assertEqual(int.from_bytes(m.root[off:off+8], 'little'), 0x40000000 + i * 0x1000 | MT_PTE_READ_ONLY)

  def test_upload_mmu_ranges(self):
    for s30 in (False, True):
      with self.subTest(s30=s30):
        dev = Mock(s30=s30)
        fw = MTFirmwareBoot(dev, 0x1000)
        ram = bytearray(b'\xff' * 0x4000)
        def map_bar(bar, off, size, fmt): return memoryview(ram)[off-ROOT_BASE:off-ROOT_BASE+size]
        dev.pci_dev.map_bar.side_effect = map_bar
        fw.mmu.map(0, 0)
        fw.upload_mmu()
        self.assertEqual(ram[:0x3000], fw.mmu.root[:0x3000])
        dev.pci_dev.map_bar.reset_mock()
        fw.mmu.map(0x3000, 0x7000)
        fw.mmu.map(0x1000, 0x9000)
        fw.upload_mmu()
        off, size = (0x2008, 24) if s30 else (0x2000, 0x1000)
        dev.pci_dev.map_bar.assert_called_once_with(2, off=ROOT_BASE+off, size=size, fmt='B')
        fw.mmu.map(1 << 21, 0)
        fw.upload_mmu()
        self.assertEqual(ram, fw.mmu.root[:len(ram)])
        ram[:] = b'\xff' * len(ram)
        fw.upload_mmu(full=True)
        self.assertEqual(ram, fw.mmu.root[:len(ram)])

  def test_invalid_and_exhaustion(self):
    m = MTMMU4(0x30000000, size=0x3000)
    with self.assertRaises(ValueError): m.map(1, 0)
    with self.assertRaises(ValueError): m.map(0, 1)
    with self.assertRaises(ValueError): m.map(0, 0, 0)
    m.map(0, 0)
    with self.assertRaises(MemoryError): m.map(1 << 30, 0)
    self.assertNotIn(1 << 30, m.pages)

if __name__ == '__main__': unittest.main()
