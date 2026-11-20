from __future__ import annotations
import functools, struct, threading, time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, ExitStack
from pathlib import Path
from typing import TYPE_CHECKING
from tinygrad.device import Buffer, Program
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import ceildiv, getenv, round_up, unwrap
from tinygrad.renderer import TinyELF
from tinygrad.uop.ops import UOp, KernelInfo
from tinygrad.runtime.support.mt.compute import MTComputeRegisters, FW_BASE, FW_DEVVA, ROOT_BASE
from tinygrad.runtime.support.mt.meta import MetaPort, assemble, use_meta
from tinygrad.runtime.support.mt.mmu import MTMMU4, MT_PAGE_SIZE, MT_PTE_READ_ONLY, MT_PTE_READ_WRITE
from tinygrad.runtime.support.mt.mtdev import MT_BUFFER_PA, MT_BUFFER_VA, MT_CORE_STRIDE, MTDev
from tinygrad.runtime.support.system import USBPCIDevice
from tinygrad.runtime.support.usb import HALF, SLOT, USBMMIOInterface
if TYPE_CHECKING: from tinygrad.runtime.ops_mt import MTDevice

CDM_CONTROL_SIZE, CDM_RING_SIZE = 128, 0x800
DIRECT_CDM = 0x4000  # shared ring controls in the direct context, below the compute state at 0x6000
META_BATCH = 0x8000
META_BATCH_PDS = 0xf0000000  # above the S30 allocator's 2 GiB range
S30_STAGE_VA, S30_UPLOAD_SIZE = 0x7000000000, HALF-2*SLOT
S30_READ_VA, S30_COMPLETION_VA = S30_STAGE_VA-HALF, S30_STAGE_VA+HALF
S30_RING_OFFSET = 2*HALF-SLOT+1024  # leave room before the controls for parameters
PDS_BASE, USC_BASE = 0xda00000000, 0xe000000000

class MTS30Firmware(MTComputeRegisters):
  def __init__(self, dev:MTDev):
    if not dev.s30: raise ValueError('Expected an MTT S30')
    self.dev, self.mmu = dev, MTMMU4(ROOT_BASE)
    self.v = dev.pci_dev.map_bar(2, off=FW_BASE, size=META_BATCH, fmt="B")
    self.root_view = dev.pci_dev.map_bar(2, off=ROOT_BASE, size=len(self.mmu.root), fmt="B")
    self.bridge = pci.usb if isinstance(pci:=dev.pci_dev, USBPCIDevice) and pci.usb.usb.product.startswith("custom") else None
    self.ring_va = S30_STAGE_VA+S30_RING_OFFSET-HALF if self.bridge is not None else FW_DEVVA+DIRECT_CDM
    self.meta_enabled = self.bridge is not None and use_meta()
    if getenv("MT_META") == 1 and not self.meta_enabled:
      raise RuntimeError("MT_META=1 requires native USC and the Chestnut SRAM transport")
    if self.meta_enabled:
      # The S30 can retain an absent page catalogue entry across META submissions. Install the directories before either engine runs.
      for va in (MT_BUFFER_VA, PDS_BASE+(MT_BUFFER_VA & 0xffffffff), USC_BASE+(MT_BUFFER_VA & 0xffffffff)):
        self.mmu.reserve_directories(va, dev.vram_size-MT_BUFFER_PA)
    self.meta_args:set[int] = set()
    self.meta_data = bytearray()
    self.meta_pending:list[tuple[int, MTS30Program]] = []
    self.meta_sequence = 0
    self.meta_flushing = False
    self.direct = False
    self.lock, self.session = threading.RLock(), ExitStack()

  @contextmanager
  def _boot(self, image:bytes) -> Iterator[None]:
    if self.direct: raise RuntimeError("S30 compute is already running")
    timers = (0x400008, 0x40000c)  # Sudi GPU subsystem SOC and USC timer enables
    saved = [self.dev.regs._r32(reg) for reg in timers]
    try:
      for reg in timers: self.dev.regs._w32(reg, 1)
      self._reset()
      self.mmu.map_range(FW_DEVVA, FW_BASE, round_up(len(image), 0x1000), MT_PTE_READ_WRITE)
      self.v[:len(image)] = image
      self.upload_mmu(full=True)
      self._setup_mmu()
      self._write(0xbf8, 0, 4)  # host starts META through its control port
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

  @functools.cached_property
  def meta_code(self) -> bytes:
    code = assemble(Path(__file__).with_name("dispatch.s"))
    if not 0 < len(code) <= 0x1000: raise ValueError("S30 META scheduler exceeds its code page")
    return code

  @contextmanager
  def boot_direct(self) -> Iterator[None]:
    # Enable both compute cores and route the workgroups across them.
    if self.bridge is not None and S30_STAGE_VA not in self.mmu.pages:
      _, (pa,) = self.dev.pci_dev.alloc_sysmem(2*HALF)
      # F0 traffic shares the lower half. META uses the upper half for either a dispatch list or a staged upload.
      base = (1<<39) | (pa+HALF)
      self.mmu.map_range(S30_READ_VA, base-HALF, HALF, MT_PTE_READ_WRITE)
      self.mmu.map(S30_COMPLETION_VA, (1<<39) | 0x822000, MT_PTE_READ_WRITE)
      self.mmu.map_range(S30_STAGE_VA, base, S30_UPLOAD_SIZE, MT_PTE_READ_ONLY)
      self.mmu.map_range(S30_STAGE_VA+S30_UPLOAD_SIZE, base+S30_UPLOAD_SIZE, HALF-S30_UPLOAD_SIZE, MT_PTE_READ_WRITE)
    if self.meta_enabled:
      for va in (FW_DEVVA+META_BATCH, PDS_BASE+META_BATCH_PDS): self.mmu.map_range(va, self.mmu.pages[S30_STAGE_VA], HALF, MT_PTE_READ_WRITE)
    with self._boot((self.meta_code if self.meta_enabled else b"").ljust(0x8000, b"\0")):
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
          self._write(core+0x4e8, self.ring_va+2*CDM_CONTROL_SIZE)
          self._write(core+0x4f0, CDM_RING_SIZE)
          self._write(core+0x4f8, self.ring_va)
          self._write(core+0x598, FW_DEVVA+0x7000)  # context state
          self._write(core+0x5b8, 0, 4)
          self._write(core+0x4d0, 1, 4)
          self._write(core+0x498, FW_DEVVA+0x6000)  # compute state
          self._write(core+0x68, 3)
          self._write(core+0x24c0, 3, 4)
        if self.meta_enabled:
          self.meta_sequence = 0
          MetaPort(self).start(META_BATCH+HALF-0x1000)
        self.direct = True
        try: yield
        finally: self.flush_meta()
      finally:
        self.meta_args.clear()
        self.meta_pending.clear()
        self.meta_data.clear()
        self.direct = False
        self._write(0x24e0, route, 4)

  def start_direct(self) -> None:
    if self.meta_enabled:
      self.meta_sequence = self.meta_sequence % 0xffffffff + 1
      self.v[0x1000:0x1004] = struct.pack("<I", self.meta_sequence)
      return
    for reg,value in ((0x138, 4), (0x478, 1)):
      for core in (MT_CORE_STRIDE, 0): self._write(core+reg, value, 4)

  def wait_direct(self, timeout:float) -> None:
    deadline = time.monotonic()+timeout
    if self.meta_enabled:
      while (completed:=int.from_bytes(self.v[0x1004:0x1008], "little")) != self.meta_sequence:
        if time.monotonic() >= deadline:
          events = [self._read(core+0x130, 4) for core in (0, MT_CORE_STRIDE)]
          faults = [hex(self._read(core+0xe150)) for core in (0, MT_CORE_STRIDE)]
          kernels = [(off, program.name) for off,program in self.meta_pending]
          for offset in (MT_CORE_STRIDE, 0): self._reset(offset)
          raise TimeoutError(f"S30 META sequence {self.meta_sequence}, completed {completed}, events {events}, faults {faults}, pending {kernels}")
      return
    for core in (0, MT_CORE_STRIDE):
      while not self._read(core+0x130, 4) & 4:
        if time.monotonic() >= deadline:
          fault = self._read(core+0xe160)
          for offset in (MT_CORE_STRIDE, 0): self._reset(offset)
          raise TimeoutError(f"S30 compute timed out on core {core//MT_CORE_STRIDE}, MMU fault {fault:#x}")

  def flush_meta(self, timeout:float|None=None, readback:memoryview|None=None):
    with self.lock:
      if self.meta_flushing or not self.meta_pending: return
      self.meta_flushing = True  # buffer finalizers may read the timeline during submission
      try:
        self.upload_mmu()
        struct.pack_into('<I', self.meta_data, self.meta_pending[-1][0], 0)
        unwrap(self.bridge).scsi_write(bytes(self.meta_data), slot_start=HALF//SLOT)
        self.kick_direct(5.0*len(self.meta_pending) if timeout is None else timeout, readback)
      except BaseException:
        self.session.close()
        raise
      finally:
        self.meta_args.clear()
        self.meta_data.clear()
        self.meta_pending.clear()
        self.meta_flushing = False

  def kick_direct(self, timeout:float, readback:memoryview|None=None) -> None:
    if readback is None: self.start_direct()
    else: unwrap(self.bridge).scsi_read(readback, self.start_direct)
    self.wait_direct(timeout)

def s30_dispatch(pds_off:int, shared:int, attrs:int) -> tuple[int, ...]:
  return (2, pds_off | 2, pds_off+16, 0, 0, 0, 0x02000000 | shared, 0, 0x100085 | attrs<<12, pds_off+32, pds_off+64)

def s30_read_signal() -> UOp:
  cq = UOp.placeholder((4,), dtypes.uint, slot=0)
  return cq[3].store(1<<16).sink(arg=KernelInfo(name="release_sram", opts_to_apply=()))

def s30_grid(groups:Sequence[int], local:tuple[int,int,int], shared:int) -> tuple[int,...]:
  lx,ly,lz = local
  lanes = lx*ly*lz
  return (*(g-1 for g in groups), (lanes if lanes < 32 else 0)<<25 | 2<<11 | shared, (lx-1)<<20 | (ly-1)<<10 | (lz-1))

def s30_cdm_stream(words:tuple[int,...], secondary:tuple[int,...]|None=None) -> bytearray:
  # A direct secondary stream either starts at TERMINATE or after it with its own disjoint grid.
  secondary_offset = 4*(len(words)+int(bool(secondary))) if secondary is not None else 0
  if secondary: words += (0x80000000,)+secondary
  write_offset = (len(words)+1)*4 + 64  # include the CDM prefetch guard after STREAM_TERMINATE
  if write_offset >= CDM_RING_SIZE: raise ValueError("S30 compute stream exceeds its ring")
  stream = bytearray(2*CDM_CONTROL_SIZE+write_offset)
  for core in range(2): struct.pack_into('<Q', stream, core*CDM_CONTROL_SIZE, write_offset)
  struct.pack_into('<Q', stream, CDM_CONTROL_SIZE+8, secondary_offset)
  struct.pack_into(f'<{len(words)+1}I', stream, 2*CDM_CONTROL_SIZE, *words, 0x80000000)
  return stream

def s30_direct_stream(dispatches:tuple[tuple[int,...],...], groups:tuple[int,int,int], local:tuple[int,int,int], shared:int,
                      split:int, params:bytearray, offset_arg:int) -> tuple[bytearray, bytearray]:
  primary = list(groups)
  secondary:tuple[int,...] = ()
  second_params = params.copy() if split else bytearray()
  if split and (count:=groups[split-1]) > 1:
    other = primary.copy()
    first = ceildiv(count, 2)
    primary[split-1], other[split-1] = first, count-first
    struct.pack_into('<I', second_params, offset_arg, first)
    secondary = dispatches[1]+s30_grid(other, local, shared)
  return params+second_params, s30_cdm_stream(dispatches[0]+s30_grid(primary, local, shared), secondary)

def s30_shader(obj:TinyELF) -> tuple[bytes, bytearray, int, int, int, list[tuple[int,str]], int]:
  if obj.target.renderer != "USC": raise ValueError("S30 requires native USC code")
  ntemps, local_words, nconst, split = struct.unpack_from('<HHHBx', obj.lib)
  native_args = [(8*i, '<i' if space is AddrSpace.ALU else '<Q') for i,(_,space,_,_) in enumerate(obj.signature)]
  off = 8+4*nconst
  return obj.lib[off:], bytearray(8*len(native_args))+obj.lib[8:off], ceildiv(ntemps, 2), 1, local_words, native_args, split

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
    text, params, temps, attrs, local_words, self.args, self.split = s30_shader(obj)
    shared = len(params)//4
    # Rogue USC prefetches one more instruction (up to 32 bytes), even when the shader ends at a page boundary.
    code = text + bytes(32)
    # One transfer updates the parameters immediately before the ring controls and the dispatch stream.
    stride = round_up(len(params), 32)
    param_size = stride*(1+bool(self.split))
    if param_size > DIRECT_CDM: raise ValueError("S30 parameters exceed the direct ring allocation")
    self.param_offset = DIRECT_CDM - param_size
    fw = dev.s30_fw
    if fw.meta_enabled:
      # USC and its immutable PDS loader share a single upload and allocation.
      pds_offset = round_up(len(code), 32)
      self.code = Buffer(dev.device, pds_offset+96, preallocate=True)
      usc_off = self.code.meta.va & 0xffffffff
      pds = s30_pds(0, shared, usc_off, temps, bool(local_words))
      self.code.host[:] = code.ljust(pds_offset, b'\0')+pds
      with fw.lock:
        fw.mmu.map_range(USC_BASE+usc_off, self.code.meta.pa, self.code.meta.size, MT_PTE_READ_ONLY)
        page = pds_offset & -MT_PAGE_SIZE
        fw.mmu.map_range(PDS_BASE+usc_off+page, self.code.meta.pa+page, self.code.meta.size-page, MT_PTE_READ_ONLY)
      self.meta_dma = pds[:16]
      self.meta_dispatch = s30_dispatch(usc_off+pds_offset, ceildiv(shared, 16), attrs)
      self.shared = ceildiv(shared, 16)+ceildiv(local_words, 16)
      self.launch = params.ljust(stride, b'\0')
      return
    self.code = Buffer(dev.device, len(code), initial_value=code)
    usc_off = self.code.meta.va & 0xffffffff
    with fw.lock: fw.mmu.map_range(USC_BASE+usc_off, self.code.meta.pa, self.code.meta.size, MT_PTE_READ_ONLY)
    packet_size = 96*(1+bool(self.split))
    self.packet = Buffer(dev.device, packet_size, preallocate=True)
    pds_off = self.packet.meta.va & 0xffffffff
    params_base = fw.ring_va-DIRECT_CDM
    pds = s30_pds(params_base+self.param_offset, shared, usc_off, temps, bool(local_words))
    with fw.lock: fw.mmu.map_range(PDS_BASE+pds_off, self.packet.meta.pa, self.packet.meta.size, MT_PTE_READ_ONLY)
    self.shared = ceildiv(shared, 16)
    dispatch = s30_dispatch(pds_off, self.shared, attrs)
    self.dispatches = (dispatch, s30_dispatch(pds_off+96, self.shared, attrs)) if self.split else (dispatch,)
    self.shared += ceildiv(local_words, 16)
    self.packet.host[:len(pds)] = pds
    if self.split:
      self.packet.host[96:192] = s30_pds(params_base+self.param_offset+stride, shared, usc_off, temps, bool(local_words))
    self.launch = bytearray(stride)
    self.launch[:len(params)] = params

  def enqueue_meta(self, fw:MTS30Firmware, groups:tuple[int,int,int], local:tuple[int,int,int], upload:bytes=b"") -> None:
    nprog = 1+bool(self.split)
    param_offset = 32+16*nprog
    controls = round_up(param_offset+len(self.launch)*nprog, 256)
    off = len(fw.meta_data)
    base = FW_DEVVA+META_BATCH+off
    pds = META_BATCH_PDS+off+32
    dispatches = tuple((2, (pds+16*i)|2, *self.meta_dispatch[2:]) for i in range(nprog))
    params, stream = s30_direct_stream(dispatches, groups, local, self.shared, self.split, self.launch, 8*len(self.args))
    size = round_up(controls+len(stream), 256)
    stride = round_up(size+len(upload), 256)
    if stride > HALF: raise ValueError("S30 META dispatch exceeds batch storage")
    if off+stride > HALF:
      fw.flush_meta()
      return self.enqueue_meta(fw, groups, local, upload)
    if upload:
      # Both CDMs read the inline payload following this copy's control stream.
      src, fmt = self.args[1]
      for i in range(nprog): struct.pack_into(fmt, params, i*len(self.launch)+src, S30_STAGE_VA+off+size)
    entry = bytearray(stride)
    struct.pack_into('<I4xQQ', entry, 0, stride, base+controls, base+controls+2*CDM_CONTROL_SIZE)
    for i in range(nprog):
      entry[32+i*16:32+(i+1)*16] = self.meta_dma
      struct.pack_into('<Q', entry, 32+i*16, base+param_offset+i*len(self.launch))
    entry[param_offset:param_offset+len(params)] = params
    entry[controls:controls+len(stream)] = stream
    entry[size:size+len(upload)] = upload
    resident = (off, self)  # keep USC code resident until both CDMs finish
    if off != len(fw.meta_data): return self.enqueue_meta(fw, groups, local, upload)  # a buffer finalizer drained the batch
    fw.meta_pending.append(resident)
    fw.meta_data.extend(entry)

  def __call__(self, *args, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), wait:bool=False, timeout:int|None=None,
               readback:memoryview|None=None, upload:bytes=b""):
    fw = self.dev.s30_fw
    with fw.lock:
      for value,(off,fmt) in zip(args, self.args, strict=True): struct.pack_into(fmt, self.launch, off, value)
      if not fw.direct:
        fw.flush_meta()
        fw.session.close()
        fw.session.enter_context(fw.boot_direct())
      elif not fw.meta_enabled: fw.upload_mmu()
      try:
        if fw.meta_enabled and (wait or timeout is not None): fw.flush_meta()
        start = time.perf_counter()
        wait_seconds = 5.0 if timeout is None else timeout/1000
        if fw.meta_enabled:
          self.enqueue_meta(fw, global_size, local_size, upload)
          fw.meta_args.update(args)
          if wait or readback is not None or timeout is not None: fw.flush_meta(wait_seconds, readback)
        else:
          params, stream = s30_direct_stream(self.dispatches, global_size, local_size, self.shared, self.split, self.launch, 8*len(self.args))
          data = params+stream
          if fw.bridge is not None:
            slot, pad = divmod(S30_RING_OFFSET+self.param_offset-DIRECT_CDM, SLOT)
            fw.bridge.scsi_write(bytes(pad)+data, slot_start=slot)
          else: fw.v[self.param_offset:self.param_offset+len(data)] = data
          fw.kick_direct(wait_seconds, readback)
        return time.perf_counter()-start if wait else None
      except BaseException:
        fw.session.close()
        raise

class S30VRAMView(USBMMIOInterface):
  def __init__(self, view:USBMMIOInterface, dev:MTDevice, va:int, bounds:tuple[int,int]|None=None):
    super().__init__(view.usb, view.addr, view.nbytes, view.fmt)
    self.dev, self.va = dev, va
    self.bounds = bounds or (va, va+view.nbytes)

  def view(self, offset:int=0, size:int|None=None, fmt=None):
    return S30VRAMView(super().view(offset, size, fmt), self.dev, self.va+offset, self.bounds)

  def _can_stage(self, index, offset:int, size:int, minimum:int=SLOT) -> bool:
    return (self.fmt == 'B' and isinstance(index, slice) and index.step in (None, 1) and offset >= 0 and minimum <= size <= self.nbytes-offset and
            (self.va+offset)%4 == 0)

  def __getitem__(self, index):
    offset, size = self._off_from_index(index)
    with self.dev.s30_fw.lock:
      self.dev.s30_fw.flush_meta()
      if not self._can_stage(index, offset, size):
        data = super().__getitem__(index)
        return memoryview(bytes(data)) if isinstance(data, memoryview) else data  # the next USB transfer reuses the bulk buffer
      dest, n = memoryview(bytearray(size)), size//128*128
      copy, signal = self.dev.s30_copy, self.dev.s30_read_signal
      for off in range(0, n, HALF):
        chunk = dest[off:min(off+HALF, n)]
        copy(S30_READ_VA, self.va+offset+off, global_size=(len(chunk)//128,1,1), local_size=(32,1,1))
        signal(S30_COMPLETION_VA, readback=chunk)
      if n < size: dest[n:] = super().__getitem__(slice(offset+n, offset+size))
      return dest

  def __setitem__(self, index, data):
    offset, _ = self._off_from_index(index)
    with self.dev.s30_fw.lock:
      fw = self.dev.s30_fw
      # Small aligned writes can join the pending list once the device's copy shader is resident.
      minimum = 4 if fw.meta_pending and "s30_copy" in self.dev.__dict__ else SLOT
      if not isinstance(data, (bytes, bytearray, memoryview)) or not self._can_stage(index, offset, memoryview(data).nbytes, minimum):
        if any(self.bounds[0] <= addr < self.bounds[1] for addr in fw.meta_args): fw.flush_meta()
        return super().__setitem__(index, data)
      src = bytes(data)
      copy = self.dev.s30_copy
      n = len(src)//128*128
      step = HALF-MT_PAGE_SIZE if fw.meta_enabled else S30_UPLOAD_SIZE
      for off in range(0, n, step):
        chunk = src[off:min(off+step, n)]
        if not fw.meta_enabled: self.usb.scsi_write(chunk, slot_start=HALF//SLOT)
        copy(self.va+offset+off, S30_STAGE_VA, global_size=(len(chunk)//128,1,1), local_size=(32,1,1), upload=chunk if fw.meta_enabled else b"")
      if fw.meta_enabled and (words:=(len(src)-n)//4):
        copy(self.va+offset+n, S30_STAGE_VA, local_size=(words,1,1), upload=src[n:n+words*4])
        n += words*4
      if n < len(src):
        fw.flush_meta()
        super().__setitem__(slice(offset+n, offset+len(src)), src[n:])
