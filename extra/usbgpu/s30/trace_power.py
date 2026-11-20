"""Cycle the Chestnut S30 slot in stages so an external probe can capture board startup before PCIe link training."""
import argparse, struct, time
from tinygrad.runtime.support.mt.mtdev import MT_S30_ID, MT_USB_GPU_BUS, MT_VENDOR_ID
from tinygrad.runtime.support.system import System
from tinygrad.runtime.support.usb import CustomASM24Controller, USB3

def trace_power(bridge:CustomASM24Controller, hold:float):
  start = time.monotonic()
  fault = 0
  def sample(stage:str, trained:bool=False):
    nonlocal fault
    time.sleep(hold)
    mv, ma, fault = struct.unpack("<HhB", bridge.usb.control_read(0xc0, 5))
    perst, ltssm = (bridge.read(reg, 1)[0] for reg in (0xb480, 0xb450))
    pins = int.from_bytes(bridge.read(0xc650, 4), "little") & ((1<<28)-1)
    print(f"{time.monotonic()-start:7.3f} s  {stage:24s}  PERST control {perst:#04x}  LTSSM {ltssm:#04x}  GPIO {pins:#09x}  {mv} mV  {ma} mA", flush=True)
    if fault:
      bridge.set_pcie_power(False)
      raise OSError("Chestnut reports a power fault")
    if (ltssm == 0x78) != trained: raise RuntimeError(f"unexpected link state during {stage}")

  try:
    bridge.set_pcie_power(False)
    sample("slot off")
    bridge.write(0xc656, bytes([bridge.read(0xc656, 1)[0] | 0x20]))
    sample("rail on, lane disabled")
    bridge.write(0xb480, b"\x00")
    sample("PERST control released")
  finally:
    if not fault: bridge.set_pcie_power(True)
  sample("normal power on", trained=True)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--hold", type=float, default=1.0, help="seconds to hold each power state")
  args = parser.parse_args()
  if not 0 < args.hold < float("inf"): parser.error("--hold must be a finite positive duration")
  (device, bus), = USB3.list_devices(0x3801, 1)
  lock = System.flock_acquire(f"mt_{bus.lower()}.lock")
  bridge = CustomASM24Controller(USB3(device))
  if bridge.pcie_cfg_req(0, bus=MT_USB_GPU_BUS) != (MT_S30_ID<<16 | MT_VENDOR_ID): raise RuntimeError("expected an S30 on Chestnut")
  trace_power(bridge, args.hold)
