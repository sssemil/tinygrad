from __future__ import annotations
import functools, hashlib, struct, threading, time
from collections.abc import Iterator
from contextlib import contextmanager, ExitStack
from pathlib import Path
from typing import TYPE_CHECKING
from tinygrad.device import Buffer, Program
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import ceildiv, round_up, unwrap
from tinygrad.renderer import TinyELF
from tinygrad.runtime.autogen import mtfw
from tinygrad.runtime.support.mt.ccb import MTCCB, MTResponse
from tinygrad.runtime.support.mt.compute import MTFirmwareBoot, FW_BASE, FW_DEVVA, META_BASE, ROOT_BASE
from tinygrad.runtime.support.mt.elf import MTKernelImage
from tinygrad.runtime.support.mt.mmu import MT_PTE_READ_ONLY, MT_PTE_READ_WRITE
from tinygrad.runtime.support.mt.mtdev import MTDev
if TYPE_CHECKING: from tinygrad.runtime.ops_mt import MTDevice

# mtgpu-s30 3.1.0-6, mtfw-gen1.bin. The FWIF layout and META segments belong to this image.
DEFAULT_S30_FIRMWARE = '/lib/firmware/mthreads/mtfw-gen1.bin'
S30_FIRMWARE_SHA256 = '49d3e4dccbf79458071152bfa1a0f79b9b2354a8a6963fdce5b9bea09656496b'
# META segment id, address, page-aligned size in DDR (boot text, local data, global data).
SEGMENTS = ((2, 0x40000000, 0x13000), (1, 0x38880000, 0x2000), (3, 0x78880000, 0x1b000))
FWIF = sum(size for _,_,size in SEGMENTS)
COREMEM, QUEUES = FWIF + 0x2a960, FWIF + 0x42960
TRACE, PERF = FWIF + 0x56a50, FWIF + 0x86490
FAULT = PERF + 0xe6018
DUMP = FAULT + 0x8218
REG_CFG, REG_BUF = DUMP + 16, DUMP + 32
UTIL = REG_BUF + 64 * 48
WATCHDOG = UTIL + 0x4c0
ITEM = round_up(WATCHDOG + 16, 0x1000)
PACKET = ITEM + 0x1000
PDS_BASE, USC_BASE = 0xda00000000, 0xe000000000

def s30_boot_image(raw:bytes) -> bytearray:
  if hashlib.sha256(raw).hexdigest() != S30_FIRMWARE_SHA256: raise ValueError('Expected mtfw-gen1.bin from mtgpu-s30 3.1.0-6')
  mem = bytearray(PACKET + 0x1000)
  config = [(0x04830030, 4)]
  offset = 0
  for segment, addr, size in SEGMENTS:
    if segment != 2:  # the boot segment is already mapped on reset
      reg, out = 0x04850000 + segment * 16, 0x30000000000000 | (FW_DEVVA + offset)
      config.extend(((reg, addr | 0xf02), (reg+4, size-0x1000), (reg+8, out & 0xffffffff), (reg+12, out >> 32)))
    offset += size

  # The META LDR file is a linked list. LOAD records target DDR or the on-core RAM staging area.
  record = struct.unpack_from('<I', raw, 8)[0]
  while record != 0xffffffff:
    op, _, record, addr, arg = struct.unpack_from('<HHIII', raw, record)
    if op == 5:  # CONFIG contains register/value writes for the bootloader
      size = struct.unpack_from('<H', raw, addr+2)[0] - 6
      config.extend((reg, val) for _,reg,val in struct.iter_unpack('<III', raw[addr+4:addr+4+size]))
    elif op == 0:
      size = struct.unpack_from('<H', raw, arg+2)[0] - 6
      if 0x80000000 <= (addr & 0xfdffffff) < 0x80018000: dest = COREMEM + (addr & 0x1ffff)
      else:
        dest = 0
        for _,base,length in SEGMENTS:
          if 0 <= (rel:=(addr & 0x7fffffff) - base) < length:
            dest += rel
            break
          dest += length
        else: raise ValueError(f'Unmapped META address {addr:#x}')
      mem[dest:dest+size] = raw[arg+4:arg+4+size]
    # ZERO records already have zero backing. START and comments carry no host writes.
  for i,(reg,val) in enumerate(config): struct.pack_into('<II', mem, 0x200+8*i, reg, val)
  struct.pack_into('<QI', mem, 0x200+8*len(config), 0, META_BASE + FWIF)

  # 3.1.0 MTFW_FWIF, with the five queues and trace, performance and fault buffers addressed in META space.
  for field,off in ((0x10, WATCHDOG), (0x18, QUEUES), (0x26e68, TRACE), (0x26e78, PERF), (0x26e90, FAULT), (0x26e98, UTIL)):
    struct.pack_into('<Q', mem, FWIF+field, META_BASE+off)
  struct.pack_into('<I', mem, FWIF+0x40, 5)
  struct.pack_into('<I', mem, FWIF+0x26e88, 1)
  struct.pack_into('<72Q', mem, FWIF+0x26ea0, *(0xd9bfff0000+i*0x100 for i in range(72)))
  struct.pack_into('<7Q2I', mem, FWIF+0x288f8, 0xd9ffff0000, PDS_BASE, USC_BASE, 0xec00000000, 0xec40000000,
                   0xf000000000, 0xeb00000000, 0xfff80000, 25_000_000)
  struct.pack_into('<QI', mem, FAULT+0x8208, META_BASE+DUMP, 1)
  struct.pack_into('<QI', mem, DUMP, META_BASE+REG_CFG, 1)
  struct.pack_into('<IxIxH', mem, REG_CFG, 0x208e150, 0x1000000, 4)
  for i in range(64): struct.pack_into('<QI', mem, FAULT+i*0x208+0x1f0, META_BASE+REG_BUF+i*48, 6)
  return mem

class MTS30Firmware(MTFirmwareBoot):
  def __init__(self, dev:MTDev, path:str|Path):
    if not dev.s30: raise ValueError('Expected an MTT S30')
    self.path = Path(path)
    super().__init__(dev, PACKET+0x1000)
    self.mmu.map_range(FW_DEVVA, FW_BASE, PACKET+0x1000, MT_PTE_READ_WRITE)
    self.queue:MTCCB|None = None
    self.lock, self.session = threading.RLock(), ExitStack()

  @functools.cached_property
  def image(self) -> bytearray: return s30_boot_image(self.path.read_bytes())

  def _reset_domains(self) -> None:
    # Sudi gen1_gpu_reset resets all five GPU domains through the SMC.
    for domain in range(8, 13):
      ack = self.dev.smc.call(9, struct.pack('<II', domain, 0))
      if struct.unpack('<II', ack) != (domain, 0): raise RuntimeError(f"S30 GPU reset failed for domain {domain}")

  @contextmanager
  def boot(self, timeout:float=5.0) -> Iterator[MTS30Firmware]:
    if self.queue is not None: raise RuntimeError("S30 firmware is already running")
    image = self.image
    timers = (0x400008, 0x40000c)  # Sudi GPU subsystem SOC and USC timer enables
    saved = [self.dev.smc._r32(reg) for reg in timers]
    try:
      self._reset_domains()
      for reg in timers: self.dev.smc._w32(reg, 1)
      self._reset()
      self.v[:len(image)] = image
      self.upload_mmu(full=True)
      self._setup_mmu()
      time.sleep(3e-6)
      self._write(0x100, 0)
      self._read(0x100)
      self._wait_started(FWIF, timeout)
      self.queue = MTCCB(self.dev, self.v, ITEM, QUEUES + 3 * mtfw.MTFW_CCB_QUEUE.SIZE)
      yield self
    finally:
      try:
        with self.lock:
          queue, self.queue = self.queue, None
          if queue is not None and queue.wr_idx != queue.response_rd: self._reset_domains()
        self._write(0x100, 0x2000000000)  # hold META in reset before releasing its memory
        self._read(0x100)
      finally:
        for reg,old in zip(timers, saved): self.dev.smc._w32(reg, old)

  def _submit(self, token:int|None, command:int, region:int, packet:bytearray, region_size:int, timeout:float,
              signal:tuple[int,int]|None=None) -> MTResponse:
    # 3.1.0 submission header and single region descriptor.
    struct.pack_into('<I', packet, 28, command)
    struct.pack_into('<II', packet, 40, 72, 1)
    struct.pack_into('<I4xQI', packet, 72, region, 72+24, region_size)
    with self.lock:
      if self.queue is None: raise RuntimeError("S30 firmware is not running")
      if self.queue.wr_idx != self.queue.response_rd: raise RuntimeError("Previous S30 submission is still pending")
      self.v[PACKET:PACKET+len(packet)] = packet
      self.queue.submit(FW_DEVVA+PACKET, 96+region_size, ROOT_BASE, generation=self.queue.seq+1, command=command, token=token, signal=signal)
      return self.queue.wait(timeout)

  def nop(self, token:int, timeout:float=5.0, *, signal:tuple[int,int]|None=None) -> MTResponse:
    packet = bytearray(96+160)
    struct.pack_into('<I', packet, 96+8, 160)
    return self._submit(token, mtfw.MTFW_SUBMISSION_CMD_GPU_NOP, mtfw.MTFW_SUB_REGION_NOP, packet, 160, timeout, signal)

  def cdm(self, token:int|None, words:tuple[int,...], timeout:float=5.0, *, regions:int=8) -> MTResponse:
    # One page holds the submission, context state, two 128-byte queue controls, two 256-byte stacks and the command ring.
    context, controls, stacks, ring = 0x200, 0x400, 0x500, 0x800
    ring_size, write_offset = 0x1000-ring, (len(words)+1)*4 + 64  # include the CDM prefetch guard after STREAM_TERMINATE
    if write_offset >= ring_size: raise ValueError("S30 compute stream exceeds its ring")
    base, kick, kick_size = FW_DEVVA+PACKET, 96+16, 0x158
    packet = bytearray(ring+write_offset)
    struct.pack_into('<II', packet, 96+8, 16, 1)
    struct.pack_into('<I', packet, kick+8, kick_size)
    struct.pack_into('<II', packet, kick+0xc0, regions, regions)  # attribute and temporary regions
    struct.pack_into('<5Q', packet, kick+0x118, base+context, base+ring, base+controls, ring_size, base+stacks)
    for core in range(2): struct.pack_into('<Q', packet, controls+core*128, write_offset)
    struct.pack_into(f'<{len(words)+1}I', packet, ring, *words, 0x80000000)
    return self._submit(token, mtfw.MTFW_SUBMISSION_CMD_GPU_CDM, mtfw.MTFW_SUB_REGION_CDM, packet, 16+kick_size, timeout)

class MTS30Program(Program['MTDevice']):
  def __init__(self, dev:MTDevice, obj:TinyELF, links:dict):
    self.dev, self.name, self.profile_key = dev, obj.name, obj.profile_key
    image = MTKernelImage.from_elf(obj.lib)
    k = image.metadata.kernel
    if k['.arch_id'] != 110 or k['.private_memory_size'] or k['.coef_reg_count'] or k['.const_calc_kernel'] or k['.has_barrier']:
      raise NotImplementedError("Unsupported S30 shader resources")
    shared, temps = k['.shared_reg_count'], ceildiv(k['.temp_reg_count'], 2)
    attrs = max(1, ceildiv(k['.attr_reg_count'], 2))
    if not 0 < shared <= 0xfff or max(attrs, temps) > 0x7f: raise ValueError("S30 shader register allocation exceeds dispatch fields")
    self.regions = max(8, temps, attrs)
    self.params = bytearray(4*shared)
    pds = bytearray(96)
    # The first PDS program loads the shader arguments into shared registers (DOUTD).
    struct.pack_into('<QI', pds, 0, PDS_BASE+96, 0x90000000 | shared)
    struct.pack_into('<I', pds, 16, 0xf4001000)
    # The workgroup task writes group IDs to LM0:2. The instance task writes local IDs to AR0:1 and launches USC (DOUTU).
    struct.pack_into('<6I', pds, 32, 0, 0, 0xb0000347, 0x80000043, 0, temps<<3)
    struct.pack_into('<5I', pds, 64, 0xdcb40003, 0xf0001301, 0xd2800000, 0xf0001b01, 0xfc000012)
    for imm in image.metadata.immediates: struct.pack_into('<I', self.params, 4*imm.offset, imm.value)
    self.args = []
    if any(a['.kind'] not in (1, 2, 3, 5, 6, 7, 18, 19, 23) for a in k['.args']): raise NotImplementedError("Unsupported S30 kernel argument")
    params = (a['.argBase'] for a in k['.args'] if a['.kind'] == 18)
    for param,(_,space,dt,_) in zip(params, obj.signature, strict=True):
      if space is not AddrSpace.ALU: dt = dtypes.uint64
      if param['.reg_bank'] != 3 or param['.size'] != dt.itemsize: raise NotImplementedError("Unsupported S30 kernel argument layout")
      self.args.append((4*param['.offset'], '<'+unwrap(dt.fmt)))
    self.pds = Buffer(dev.device, len(pds)+len(self.params), initial_value=bytes(pds+self.params))
    self.last_params:bytearray|None = self.params.copy()
    self.code = Buffer(dev.device, len(image.text), initial_value=image.text)
    self.shared = ceildiv(shared, 16)
    self.dispatch = (2, 2, 16, 0, 0, 0, 0x02000000 | self.shared, 0, 0x100085 | attrs<<12, 32, 64)

  def __call__(self, *args, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), wait:bool=False, timeout:int|None=None):
    lx,ly,lz = local_size
    lanes = lx*ly*lz
    stream = self.dispatch + tuple(g-1 for g in global_size) + ((lanes if lanes < 32 else 0)<<25 | 2<<11 | self.shared,
                                                                (lx-1)<<20 | (ly-1)<<10 | (lz-1))
    fw = self.dev.s30_fw
    with fw.lock:
      for value,(off,fmt) in zip(args, self.args, strict=True): struct.pack_into(fmt, self.params, off, value)
      if self.params != self.last_params:
        self.last_params = None
        self.pds.host[96:] = self.params
        self.last_params = self.params.copy()
      fw.mmu.map_range(PDS_BASE, self.pds.meta.pa, self.pds.meta.size, MT_PTE_READ_ONLY)
      fw.mmu.map_range(USC_BASE, self.code.meta.pa, self.code.meta.size, MT_PTE_READ_ONLY)
      if fw.queue is None: fw.session.enter_context(fw.boot())
      else: fw.upload_mmu()
      try:
        start = time.perf_counter()
        resp = fw.cdm(None, stream, timeout=5.0 if timeout is None else timeout/1000, regions=self.regions)
        if resp.value: raise RuntimeError(f"S30 kernel {self.name} failed with firmware status {resp.value:#x}")
        return time.perf_counter()-start if wait else None
      except BaseException:
        fw.session.close()
        raise
