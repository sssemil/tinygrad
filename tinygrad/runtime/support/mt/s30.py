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
from tinygrad.runtime.support.mt.mtdev import MT_CORE_STRIDE, MTDev
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
CDM_CONTROL_SIZE, CDM_RING_SIZE, CDM_KICK_SIZE = 128, 0x800, 0x158
DIRECT_CDM = 0x4000  # shared ring controls in the direct context, below the compute state at 0x6000
CDM_SUBMISSION_SIZE = 96+16+CDM_KICK_SIZE
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
  context_pa:int = FW_BASE
  def __init__(self, dev:MTDev, path:str|Path):
    if not dev.s30: raise ValueError('Expected an MTT S30')
    self.path = Path(path)
    super().__init__(dev, PACKET+0x1000)
    self.direct = False
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
  def _boot(self, image:bytes|bytearray, *, master:bool=True, reset_domains:bool=True) -> Iterator[None]:
    if self.direct or self.queue is not None: raise RuntimeError("S30 compute is already running")
    timers = (0x400008, 0x40000c)  # Sudi GPU subsystem SOC and USC timer enables
    if reset_domains: self.dev.power_on()
    saved = [self.dev.regs._r32(reg) for reg in timers]
    try:
      if reset_domains: self._reset_domains()
      for reg in timers: self.dev.regs._w32(reg, 1)
      self._reset()
      self.mmu.map_range(FW_DEVVA, self.context_pa, round_up(len(image), 0x1000), MT_PTE_READ_WRITE)
      self.v[:len(image)] = image
      self.upload_mmu(full=True)
      self._setup_mmu()
      if not master: self._write(0xbf8, 0, 4)  # host starts META through its control port
      time.sleep(3e-6)
      self._write(0x100, 0)
      self._read(0x100)
      yield
    finally:
      try:
        self._write(0x100, 0x2000000000)  # hold META in reset before releasing its memory
        self._read(0x100)
      finally:
        for reg,old in zip(timers, saved): self.dev.regs._w32(reg, old)

  @contextmanager
  def boot(self, timeout:float=5.0) -> Iterator[MTS30Firmware]:
    with self._boot(self.image):
      self._wait_started(FWIF, timeout)
      self.queue = MTCCB(self.dev, self.v, ITEM, QUEUES + 3 * mtfw.MTFW_CCB_QUEUE.SIZE, batch_responses=True)
      try: yield self
      finally:
        with self.lock:
          queue, self.queue = self.queue, None
          if queue is not None and queue.wr_idx != queue.response_rd: self._reset_domains()

  @contextmanager
  def boot_direct(self) -> Iterator[None]:
    # Enable both compute cores and route the workgroups across them. META stays stopped.
    with self._boot(bytes(0x8000), master=False, reset_domains=False):
      route = self._read(0x24e0, 4)
      self._reset(MT_CORE_STRIDE)
      self._write(MT_CORE_STRIDE+0xbf8, 0, 4)
      self._write(MT_CORE_STRIDE+0x100, 0)
      self._read(MT_CORE_STRIDE+0x100)
      self._write(0x24e0, (route & ~0xff) | 3, 4)
      try:
        self._setup_mmu(MT_CORE_STRIDE)
        for core in (0, MT_CORE_STRIDE):
          self._write(core+0x4000, 1, 4)  # compute scheduler
          self._write(core+0xf920, 0, 4)  # CDM MMU context zero
          self._write(core+0xcd8, 1, 4)
          for reg in (0x638, 0x658, 0x6e8, 0x6f8): self._write(core+reg, PDS_BASE)
          for reg in (0x4008, 0x4010, 0xf268, 0x4028): self._write(core+reg, USC_BASE)
          self._write(core+0x4e8, FW_DEVVA+DIRECT_CDM+2*CDM_CONTROL_SIZE)
          self._write(core+0x4f0, CDM_RING_SIZE)
          self._write(core+0x4f8, FW_DEVVA+DIRECT_CDM)
          self._write(core+0x598, FW_DEVVA+0x7000)  # context state
          self._write(core+0x5b8, 0, 4)
          self._write(core+0x4d0, 1, 4)
          self._write(core+0x498, FW_DEVVA+0x6000)  # compute state
          self._write(core+0x68, 3)
          self._write(core+0x24c0, 3, 4)
        self.direct = True
        yield
      finally:
        self.direct = False
        self._write(0x24e0, route, 4)

  def kick_direct(self, timeout:float) -> None:
    for reg,value in ((0x138, 4), (0x478, 1)):
      for core in (MT_CORE_STRIDE, 0): self._write(core+reg, value, 4)
    deadline = time.monotonic()+timeout
    for core in (0, MT_CORE_STRIDE):
      while not self._read(core+0x130, 4) & 4:
        if time.monotonic() >= deadline:
          fault = self._read(core+0xe160)
          for offset in (MT_CORE_STRIDE, 0): self._reset(offset)
          raise TimeoutError(f"S30 compute timed out on core {core//MT_CORE_STRIDE}, MMU fault {fault:#x}")

  def submit(self, token:int|None, command:int, size:int, data:bytearray, timeout:float, signal:tuple[int,int]|None=None,
             resident:Buffer|None=None, offset:int=0) -> MTResponse:
    with self.lock:
      if self.queue is None: raise RuntimeError("S30 firmware is not running")
      if self.queue.wr_idx != self.queue.response_rd: raise RuntimeError("Previous S30 submission is still pending")
      addr, mem = (FW_DEVVA+PACKET, self.v.view(PACKET)) if resident is None else (resident.meta.va, resident.host)
      mem[offset:offset+len(data)] = data
      self.queue.submit(addr, size, ROOT_BASE, generation=self.queue.seq+1, command=command, token=token, signal=signal)
      return self.queue.wait(timeout, spin_for=0.001)

  def nop(self, token:int, timeout:float=5.0, *, signal:tuple[int,int]|None=None) -> MTResponse:
    packet = s30_submission(mtfw.MTFW_SUBMISSION_CMD_GPU_NOP, mtfw.MTFW_SUB_REGION_NOP, 160)
    struct.pack_into('<I', packet, 96+8, 160)
    return self.submit(token, mtfw.MTFW_SUBMISSION_CMD_GPU_NOP, len(packet), packet, timeout, signal)

  def cdm(self, token:int|None, words:tuple[int,...], timeout:float=5.0, *, regions:int=8) -> MTResponse:
    packet = s30_cdm_packet(words, regions, FW_DEVVA+PACKET)
    return self.submit(token, mtfw.MTFW_SUBMISSION_CMD_GPU_CDM, CDM_SUBMISSION_SIZE, packet, timeout)

def s30_submission(command:int, region:int, region_size:int) -> bytearray:
  # 3.1.0 submission header and single region descriptor.
  packet = bytearray(96+region_size)
  struct.pack_into('<I', packet, 28, command)
  struct.pack_into('<II', packet, 40, 72, 1)
  struct.pack_into('<I4xQI', packet, 72, region, 96, region_size)
  return packet

def s30_dispatch(pds_off:int, shared:int, attrs:int) -> tuple[int, ...]:
  return (2, pds_off | 2, pds_off+16, 0, 0, 0, 0x02000000 | shared, 0, 0x100085 | attrs<<12, pds_off+32, pds_off+64)

def s30_cdm_stream(words:tuple[int,...]) -> bytearray:
  write_offset = (len(words)+1)*4 + 64  # include the CDM prefetch guard after STREAM_TERMINATE
  if write_offset >= CDM_RING_SIZE: raise ValueError("S30 compute stream exceeds its ring")
  stream = bytearray(2*CDM_CONTROL_SIZE+write_offset)
  for core in range(2): struct.pack_into('<Q', stream, core*CDM_CONTROL_SIZE, write_offset)
  struct.pack_into(f'<{len(words)+1}I', stream, 2*CDM_CONTROL_SIZE, *words, 0x80000000)
  return stream

def s30_cdm_packet(words:tuple[int,...], regions:int, addr:int, controls:int=0x200) -> bytearray:
  packet = s30_submission(mtfw.MTFW_SUBMISSION_CMD_GPU_CDM, mtfw.MTFW_SUB_REGION_CDM, 16+CDM_KICK_SIZE)
  ring = controls + 2*CDM_CONTROL_SIZE
  packet.extend(bytes(controls-len(packet))+s30_cdm_stream(words))
  struct.pack_into('<II', packet, 96+8, 16, 1)
  kick, scratch = 96+16, FW_DEVVA+PACKET
  struct.pack_into('<I', packet, kick+8, CDM_KICK_SIZE)
  struct.pack_into('<II', packet, kick+0xc0, regions, regions)
  struct.pack_into('<5Q', packet, kick+0x118, scratch+0xc00, addr+ring, addr+controls, CDM_RING_SIZE, scratch+0xe00)
  return packet

def s30_shader(obj:TinyELF) -> tuple[bytes, bytearray, int, int, int, list[tuple[int,str]]]:
  if obj.target.renderer == "USC":
    ntemps, local_words, nconst = struct.unpack_from('<HHI', obj.lib)
    native_args = [(8*i, '<i' if space is AddrSpace.ALU else '<Q') for i,(_,space,_,_) in enumerate(obj.signature)]
    off = 8+4*nconst
    return obj.lib[off:], bytearray(8*len(native_args))+obj.lib[8:off], ceildiv(ntemps, 2), 1, local_words, native_args
  image = MTKernelImage.from_elf(obj.lib)
  k = image.metadata.kernel
  if k['.arch_id'] != 110 or k['.private_memory_size'] or k['.coef_reg_count'] or k['.const_calc_kernel'] or k['.has_barrier']:
    raise NotImplementedError("Unsupported S30 shader resources")
  shared, temps = k['.shared_reg_count'], ceildiv(k['.temp_reg_count'], 2)
  attrs = max(1, ceildiv(k['.attr_reg_count'], 2))
  if not 0 < shared <= 0xfff or max(attrs, temps) > 0x7f: raise ValueError("S30 shader register allocation exceeds dispatch fields")
  params = bytearray(4*shared)
  for imm in image.metadata.immediates: struct.pack_into('<I', params, 4*imm.offset, imm.value)
  args:list[tuple[int,str]] = []
  if any(a['.kind'] not in (1, 2, 3, 5, 6, 7, 18, 19, 23) for a in k['.args']): raise NotImplementedError("Unsupported S30 kernel argument")
  arg_layouts = (a['.argBase'] for a in k['.args'] if a['.kind'] == 18)
  for param,(_,space,dt,_) in zip(arg_layouts, obj.signature, strict=True):
    if space is not AddrSpace.ALU: dt = dtypes.uint64
    if param['.reg_bank'] != 3 or param['.size'] != dt.itemsize: raise NotImplementedError("Unsupported S30 kernel argument layout")
    args.append((4*param['.offset'], '<'+unwrap(dt.fmt)))
  return image.text, params, temps, attrs, 0, args

def s30_pds(params:int, shared:int, usc:int, temps:int, local:bool=False) -> bytearray:
  # DOUT encodes a 64-bit source pair, a 32-bit control word and a destination (DMA, register write, USC).
  def dout(dest:int, pair:int, ctrl:int=0, *, end:bool=False, pred:bool=False) -> int:
    return 0xf<<28 | pred<<27 | end<<26 | ctrl<<11 | pair<<3 | dest

  pds = bytearray(96)
  struct.pack_into('<QI', pds, 0, params, 1<<31 | 1<<28 | shared)
  struct.pack_into('<I', pds, 16, dout(0, 0, 2, end=True))
  # The same PDS program runs once for the workgroup and once per instance. Its input IDs arrive in T0:T1.
  # Select source components for LM0:2 and AR0:1 and enable each corresponding write lane.
  group_ids = 1<<31 | 3<<28 | 3<<8 | 1<<6 | 0b111
  local_ids = 1<<31 | 1<<6 | 0b11
  struct.pack_into('<6I', pds, 32, 0, 0, group_ids, local_ids, usc, temps<<3)
  branch_instance = 0xd4800000 | 1<<27 | 3<<20 | 2<<17 | 3  # branch three instructions to the instance path
  halt = 0xd2800000
  temp_pair = 0xc0//2
  # DOUTW addresses common-store groups of four. Clear LM4 before the first task enters the barrier.
  if local: struct.pack_into('<I', pds, 56, 3<<28 | 1<<13 | 1<<6 | 3)
  program = [branch_instance+int(local)] + ([dout(1, 0, 6)] if local else []) + [dout(1, temp_pair, 2), halt, dout(1, temp_pair, 3),
                                                                                  dout(2, 2, end=True, pred=True)]
  struct.pack_into(f'<{len(program)}I', pds, 64, *program)
  return pds

class MTS30Program(Program['MTDevice']):
  def __init__(self, dev:MTDevice, obj:TinyELF, links:dict):
    self.dev, self.name, self.profile_key = dev, obj.name, obj.profile_key
    self.direct = obj.target.renderer == "USC"
    text, params, temps, attrs, local_words, self.args = s30_shader(obj)
    shared = len(params)//4
    # Rogue USC prefetches one more instruction (up to 32 bytes), even when the shader ends at a page boundary.
    code = text + bytes(32)
    self.code = Buffer(dev.device, len(code), initial_value=code)
    # One transfer updates the parameters immediately before the ring controls and the dispatch stream.
    pds_start = 0 if self.direct else round_up(CDM_SUBMISSION_SIZE, 32)
    param_size = round_up(len(params), 32)
    controls = DIRECT_CDM if self.direct else round_up(pds_start + 96 + len(params), 256)
    if param_size > controls: raise ValueError("S30 parameters exceed the direct ring allocation")
    self.param_offset = controls - param_size
    packet_size = 96 if self.direct else controls + 2*CDM_CONTROL_SIZE + CDM_RING_SIZE
    self.packet = Buffer(dev.device, packet_size, preallocate=True)
    packet_off, usc_off = self.packet.meta.va & 0xffffffff, self.code.meta.va & 0xffffffff
    pds_off = packet_off + pds_start
    params_base = FW_DEVVA if self.direct else PDS_BASE+packet_off
    pds = s30_pds(params_base+self.param_offset, shared, usc_off, temps, bool(local_words))
    with dev.s30_fw.lock:
      dev.s30_fw.mmu.map_range(PDS_BASE+packet_off, self.packet.meta.pa, self.packet.meta.size, MT_PTE_READ_ONLY)
      dev.s30_fw.mmu.map_range(USC_BASE+usc_off, self.code.meta.pa, self.code.meta.size, MT_PTE_READ_ONLY)
    self.shared = ceildiv(shared, 16)
    dispatch = s30_dispatch(pds_off, self.shared, attrs)
    words = dispatch + (0,)*5
    self.shared += ceildiv(local_words, 16)
    self.grid_offset = param_size+2*CDM_CONTROL_SIZE+4*len(dispatch)
    if self.direct:
      self.packet.host[:len(pds)] = pds
      self.launch = bytearray(param_size)+s30_cdm_stream(words)
    else:
      packet = s30_cdm_packet(words, max(8, temps, attrs), self.packet.meta.va, controls)
      packet[pds_start:pds_start+len(pds)] = pds
      self.packet.host[:len(packet)] = packet
      self.launch = packet[self.param_offset:self.param_offset+self.grid_offset+20]
    self.launch[:len(params)] = params

  def __call__(self, *args, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), wait:bool=False, timeout:int|None=None):
    lx,ly,lz = local_size
    lanes = lx*ly*lz
    fw = self.dev.s30_fw
    with fw.lock:
      for value,(off,fmt) in zip(args, self.args, strict=True): struct.pack_into(fmt, self.launch, off, value)
      if (fw.direct if self.direct else fw.queue is not None): fw.upload_mmu()
      else:
        fw.session.close()
        fw.session.enter_context(fw.boot_direct() if self.direct else fw.boot())
      try:
        start = time.perf_counter()
        struct.pack_into('<5I', self.launch, self.grid_offset, *(g-1 for g in global_size),
                         (lanes if lanes < 32 else 0)<<25 | 2<<11 | self.shared, (lx-1)<<20 | (ly-1)<<10 | (lz-1))
        wait_seconds = 5.0 if timeout is None else timeout/1000
        if self.direct:
          fw.v[self.param_offset:self.param_offset+len(self.launch)] = self.launch
          fw.kick_direct(wait_seconds)
        else:
          resp = fw.submit(None, mtfw.MTFW_SUBMISSION_CMD_GPU_CDM, CDM_SUBMISSION_SIZE, self.launch, wait_seconds,
                           resident=self.packet, offset=self.param_offset)
          if resp.value: raise RuntimeError(f"S30 kernel {self.name} failed with firmware status {resp.value:#x}")
        return time.perf_counter()-start if wait else None
      except BaseException:
        fw.session.close()
        raise
