from __future__ import annotations
import functools, struct, time
from tinygrad.helpers import DEBUG, pluralize
from tinygrad.runtime.support.memory import MMIOInterface
from tinygrad.runtime.support.system import PCIDevice, USBPCIDevice, System, filter_visible_devices
from tinygrad.runtime.support.usb import USB3

MT_VENDOR_ID = 0x1ed5
MT_USB_GPU_BUS = 2
MT_SYS_BAR, MT_DDR_BAR = 0, 2
MT_S30_ID = 0x0102
MT_SRAM_OFFSET, MT_S30_SRAM_OFFSET = 0x30000, 0x6f000
MT_DISTRIBUTOR_OFFSET = 0x83000
MT_SOCIF_OFFSET, MT_CORE_STRIDE, MT_CORE_COUNT = 0xc00000, 0x100000, 8
MT_S30_SOCIF_OFFSET = 0x200000
MT_CORE_ID = 0x20

MT_IPC_RING_SIZE = 0x200
MT_IPC_IRQ = 2
MT_IPC_NODE_SMC, MT_IPC_NODE_PCIE = 0, 2
MT_IPC_EVENT_GET_FW_VERSION = 11
MT_IPC_EVENT_SET_POWER_STATE = 91
MT_PSTATE_P0, MT_PSTATE_P12 = 0, 12


def _ipc_header(event_id:int, data_size:int, msg_id:int=0, msg_type:int=0, event_type:int=0,
                target:int=MT_IPC_NODE_SMC, response:bool=True) -> int:
  return (event_type | (event_id << 5) | (msg_id << 14) | (msg_type << 22) | (int(response) << 25) |
          (MT_IPC_NODE_PCIE << 29) | (target << 37) | (data_size << 45))


class _Ring:
  def __init__(self, smc:MTSMC, ctrl:int, header_size:int=0x18):
    self.smc, self.ctrl = smc, ctrl
    self.size = smc._r32(ctrl + header_size - 12)
    self.data = smc._r32(ctrl + header_size - 8)
    if self.size == 0 or self.size & 3: raise RuntimeError(f"Invalid MT IPC ring size {self.size:#x} at {ctrl:#x}")
    if (self.data & 0xfff) != ((ctrl + header_size) & 0xfff): raise RuntimeError(f"Invalid MT IPC ring data pointer {self.data:#x}")

  def _data_len(self) -> int:
    rd, wr = self.smc._r32(self.ctrl), self.smc._r32(self.ctrl + 4)
    rp, wp = rd >> 1, wr >> 1
    if ((wr ^ rd) & ~1) == 0: return self.size if (rd ^ wr) & 1 else 0
    return wp - rp if rp < wp else self.size - (rp - wp)

  def _advance(self, ptr:int, nbytes:int) -> int:
    pos, parity = ptr >> 1, ptr & 1
    if nbytes >= self.size - pos: return ((nbytes - (self.size - pos)) * 2) | (parity ^ 1)
    return ((pos + nbytes) * 2) | parity

  def read(self, pos:int, nbytes:int) -> bytes:
    first = min(nbytes, self.size - pos)
    return self.smc._read(self.data + pos, first) + (self.smc._read(self.data, nbytes - first) if first != nbytes else b'')

  def write(self, payload:bytes):
    if len(payload) > self.size - self._data_len(): raise RuntimeError("MT IPC ring is full")
    wr = self.smc._r32(self.ctrl + 4)
    pos = wr >> 1
    first = min(len(payload), self.size - pos)
    self.smc._write(self.data + pos, payload[:first])
    if first != len(payload): self.smc._write(self.data, payload[first:])
    self.smc._w32(self.ctrl + 4, self._advance(wr, len(payload)))

  def read_message(self) -> tuple[int, bytes, int]:
    rd = self.smc._r32(self.ctrl)
    header = struct.unpack('<Q', self.read(rd >> 1, 8))[0]
    data_size = (header >> 45) & 0xff
    if not 1 <= data_size <= 64: raise RuntimeError(f"Invalid MT IPC response size {data_size} dwords")
    nbytes = 8 + data_size * 4
    payload = self.read(rd >> 1, nbytes)[8:]
    return header, payload, self._advance(rd, nbytes)


class MTRegisters:
  def __init__(self, mmio:MMIOInterface): self.mmio = mmio
  def _read(self, offset:int, nbytes:int) -> bytes:
    # The ASM24 bridge returns short reads for some MT register windows above 64 bytes.
    return b''.join(bytes(self.mmio[off:off + min(64, offset + nbytes - off)]) for off in range(offset, offset + nbytes, 64))

  def _write(self, offset:int, data:bytes):
    assert offset % 4 == 0 and len(data) % 4 == 0
    self.mmio[offset:offset + len(data)] = data

  def _r32(self, offset:int) -> int: return struct.unpack('<I', self._read(offset, 4))[0]
  def _w32(self, offset:int, value:int): self._write(offset, struct.pack('<I', value))

class MTSMC(MTRegisters):
  def __init__(self, mmio:MMIOInterface, s30:bool=False):
    super().__init__(mmio)
    base, header = (MT_S30_SRAM_OFFSET, 0x14) if s30 else (MT_SRAM_OFFSET, 0x18)
    self.tx = _Ring(self, base, header)
    self.rx = _Ring(self, base + header + MT_IPC_RING_SIZE, header)
    if s30 and any(r.size != MT_IPC_RING_SIZE or r.data != r.ctrl + header for r in (self.tx, self.rx)):
      raise RuntimeError("Invalid S30 IPC ring")
    self.msg_id = 0

  def _kick(self, event_id:int, target:int, msg_id:int):
    # GIC SGIR: write-1-to-set, target list, event ID and SGI interrupt source.
    value = 0x40000 | 1 | (0x100 << target) | ((event_id * 2) & 0xff)
    self._w32(MT_DISTRIBUTOR_OFFSET + 0x1f8 + MT_IPC_IRQ * 4, value)

  def _receive(self, old_wr:int, msg_id:int, event_id:int, timeout:float) -> bytes:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
      if (wr:=self._r32(self.rx.ctrl + 4)) != old_wr:
        header, payload, next_rd = self.rx.read_message()
        self._w32(self.rx.ctrl, next_rd)
        old_wr = wr
        if (header >> 14) & 0xff != msg_id or (header >> 5) & 0x7f != event_id: continue  # asynchronous message
        if not (header >> 22) & 1: raise RuntimeError("MT IPC response has the request bit clear")
        if (error := (header >> 53) & 3): raise RuntimeError(f"MT IPC event {event_id} failed with error {error}")
        return payload
      time.sleep(0.001)
    raise TimeoutError(f"MT IPC event {event_id} timed out")

  def call(self, event_id:int, payload:bytes, timeout=5.0) -> bytes:
    assert len(payload) % 4 == 0 and 4 <= len(payload) <= 256
    msg_id, self.msg_id = self.msg_id, (self.msg_id + 1) & 0xff
    header = _ipc_header(event_id, len(payload) // 4, msg_id=msg_id)
    old_wr = self._r32(self.rx.ctrl + 4)
    self.tx.write(struct.pack('<Q', header) + payload)
    self._kick(event_id, MT_IPC_NODE_SMC, msg_id)
    return self._receive(old_wr, msg_id, event_id, timeout)

  def rtos_version(self) -> list[tuple[str, int]]:
    payload = self.call(MT_IPC_EVENT_GET_FW_VERSION, bytes(108))
    if len(payload) % 12: raise RuntimeError("Invalid MT RTOS version response")
    return [(payload[off:off + 8].rstrip(b'\0').decode('ascii'), struct.unpack_from('<I', payload, off + 8)[0])
            for off in range(0, len(payload), 12)]

  def set_pstate(self, level:int):
    # OS (Linux), requested P-state, reserved.
    return self.call(MT_IPC_EVENT_SET_POWER_STATE, struct.pack('<III', 0, level, 0))


class MTDev:
  def __init__(self, pci_dev:PCIDevice, *, manage_power:bool=True):
    self.pci_dev, self.devfmt = pci_dev, pci_dev.pcibus
    identity = pci_dev.read_config(0, 4)
    if identity & 0xffff != MT_VENDOR_ID: raise RuntimeError(f"Expected Moore Threads GPU, got PCI ID {identity:#010x}")
    device_id = identity >> 16
    self.s30 = device_id == MT_S30_ID
    self.socif = MT_S30_SOCIF_OFFSET if self.s30 else MT_SOCIF_OFFSET
    if device_id >> 8 == 0x01 and not self.s30: raise RuntimeError(f"Unsupported Sudi device {device_id:04x}")
    self.mmio = pci_dev.map_bar(MT_SYS_BAR, fmt='B')
    self.vram_size = pci_dev.bar_info(MT_DDR_BAR)[1]
    self.regs = MTRegisters(self.mmio)
    self.managed_power = False
    if manage_power or not self.s30: self.power_on()
    try:
      self.core_ids = self._read_core_ids()
      if not any(self.core_ids): raise RuntimeError("MT GPU cores did not power on")
    except Exception:
      self.close()
      raise
    if DEBUG >= 1: print(f"mt {self.devfmt}: {len(self.core_ids)} cores, BVNC {self.bvnc}, BAR2 aperture {self.vram_size >> 20} MiB")

  @functools.cached_property
  def smc(self): return MTSMC(self.mmio, self.s30)

  @functools.cached_property
  def fw_versions(self): return self.smc.rtos_version()

  def power_on(self):
    if self.managed_power: return
    versions = self.fw_versions
    if DEBUG >= 1: print(f"mt {self.devfmt}: firmware {versions}")
    self.smc.set_pstate(MT_PSTATE_P0)
    self.managed_power = True

  def _read_core_ids(self) -> list[int]:
    return [struct.unpack('<Q', self.regs._read(self.socif + core * MT_CORE_STRIDE + MT_CORE_ID, 8))[0]
            for core in range(2 if self.s30 else MT_CORE_COUNT)]

  @property
  def bvnc(self) -> tuple[int, int, int, int]:
    value = self.core_ids[0]
    return ((value >> 48) & 0xffff, (value >> 32) & 0xffff, (value >> 16) & 0xffff, value & 0xffff)

  def close(self):
    if self.managed_power:
      self.smc.set_pstate(MT_PSTATE_P12)
      self.managed_power = False

  @classmethod
  def open_usb(cls, dev_id=0, *, manage_power:bool=True) -> MTDev:
    devices = filter_visible_devices(USB3.list_devices(0xADD1, 0x0001) + USB3.list_devices(0x3801, 0x0001), "MT")
    if dev_id >= len(devices): raise RuntimeError(f"MT:{dev_id} does not exist ({pluralize('device', len(devices))} available)")
    dev, pcibus = devices[dev_id]
    return cls(USBPCIDevice("MT", dev, pcibus, gpu_bus=MT_USB_GPU_BUS, vram_bars={(MT_S30_ID << 16) | MT_VENDOR_ID: MT_DDR_BAR}),
               manage_power=manage_power)

  @classmethod
  def open_pci(cls, dev_id=0) -> MTDev:
    pci_dev = System.pci_probe_device("MT", dev_id, MT_VENDOR_ID, ((0xff00, (0x0200, 0x0300)),), base_class=0x03)
    return cls(pci_dev)
