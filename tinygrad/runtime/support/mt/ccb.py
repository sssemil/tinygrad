from __future__ import annotations
import ctypes, dataclasses, struct, time
from tinygrad.runtime.autogen import mtfw
from tinygrad.runtime.support.mt.mtdev import MTDev

@dataclasses.dataclass(frozen=True)
class MTResponse:
  flags: int
  value: int
  token: int
  private: int
  start_timestamp: int
  run_time: int
  info: int
  ext_info: int

  @classmethod
  def from_bytes(cls, data:bytes) -> MTResponse:
    if len(data) != 0x30: raise ValueError("Invalid MT CCB response size")
    return cls(*struct.unpack('<IIQQQIIQ', data))

class MTCCB:
  def __init__(self, dev:MTDev, mem, item_off:int, queue_off:int, ccb_type:int=3, meta_base:int=0x90000000):
    self.dev, self.mem, self.item_off, self.queue_off = dev, mem, item_off, queue_off
    self.ccb_type, self.wr_idx = ccb_type, 0
    self.response_rd = 0
    self.pending_token = 0
    self.seq = 0
    # Synchronous submissions reuse one item for every slot in the request ring.
    self.mem[queue_off:queue_off+256*8] = struct.pack('<Q', meta_base+item_off) * 256

  def _publish(self, ci, skip_semaphores:bool=False) -> None:
    item = bytes(ci)
    if skip_semaphores:
      # Both valid masks are zero. The firmware only needs the header and the fields after the semaphore arrays.
      sp = ci.ccbItemPara.submissionPara
      start = ctypes.addressof(sp.dependQSem) - ctypes.addressof(ci)
      end = start + ctypes.sizeof(sp.dependQSem) + ctypes.sizeof(sp.finishQSem)
      self.mem[self.item_off:self.item_off+start] = item[:start]
      self.mem[self.item_off+end:self.item_off+len(item)] = item[end:]
    else: self.mem[self.item_off:self.item_off+len(item)] = item
    self.pending_token = ci.ccbItemToken
    self.seq += 1
    self.wr_idx = (self.wr_idx + 1) & 0xff
    self.mem[self.queue_off+0x4000:self.queue_off+0x4004] = struct.pack('<I', self.wr_idx)
    self.dev.smc._write(self.dev.socif + 0xb00, struct.pack('<I', self.ccb_type))

  def submit(self, submission_addr:int, submission_size:int, root_pa:int, generation:int=1, command:int=104,
             token:int|None=None, private:int|None=None, queue_id:int=0, signal:tuple[int,int]|None=None) -> None:
    if token is None: token = 0x103 + self.seq
    if private is None: private = 0x4d544e4f50
    ci = mtfw.MTFW_CCB_ITEM()
    ci.submissionCmd, ci.ccbItemToken, ci.hostPrivateData, ci.processId = command, token, private, 0
    ci.cbId = 0
    ci.frameId = 0
    sp = ci.ccbItemPara.submissionPara
    sp.pageTableRootAddr, sp.pageTableGeneration = root_pa, generation
    sp.submissionQueueId, sp.submissionAddr, sp.submissionSize = queue_id, submission_addr, submission_size
    if signal is not None:
      addr, value = signal
      if addr & 7: raise ValueError("MT queue semaphore must be 8-byte aligned")
      sp.finishQSemValidMask = 1
      sp.finishQSem[0].addr, sp.finishQSem[0].value = addr | 1, value  # META address, binary finish semaphore
    sp.qSempOutofOrder = 1
    self._publish(ci, skip_semaphores=signal is None)

  def flush_mmu(self, root_pa:int, generation:int=1) -> None:
    ci = mtfw.MTFW_CCB_ITEM()
    ci.submissionCmd, ci.ccbItemToken, ci.hostPrivateData, ci.processId = 13, 0x103, 0x4d544e4f50, 1
    sp = ci.ccbItemPara.invalidTlbCfg
    sp.pageTableRootAddr, sp.pageTableGeneration = root_pa, generation
    self._publish(ci)

  def response_index(self) -> int:
    return struct.unpack('<I', self.mem[self.queue_off+0x4020:self.queue_off+0x4024])[0]

  def wait(self, timeout:float=5.0) -> MTResponse:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
      if (wr := self.response_index()) != self.response_rd:
        off = self.queue_off + 0x1000 + ((wr - 1) % 256) * 0x30
        response = MTResponse.from_bytes(bytes(self.mem[off:off+0x30]))
        if response.token != self.pending_token:
          time.sleep(0.001)
          continue
        self.response_rd = wr
        self.mem[self.queue_off+0x4028:self.queue_off+0x402c] = struct.pack('<I', wr)
        return response
      time.sleep(0.001)
    raise TimeoutError(f"MT CCB {self.ccb_type} response timed out")
