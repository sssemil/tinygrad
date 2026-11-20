"""Compare the complete external flash on the Chestnut used for S30 bring-up. Called by build_chestnut.sh --verify."""
import ast, hashlib, sys
from pathlib import Path
from tinygrad.runtime.support.system import System
from tinygrad.runtime.support.usb import USB3

if __name__ == "__main__":
  from e4_flash import flash_jedec_id, flash_read
  build = Path(sys.argv[1])
  # Read the two configuration literals without executing the flasher's top-level USB commands.
  records = {n.targets[0].id:ast.literal_eval(n.value.args[0]) for n in ast.parse((build/"flash.py").read_text()).body
             if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("config1", "config2")}
  if any(len(records[name]) != 128 for name in ("config1", "config2")): raise ValueError("unexpected bridge config size")
  config = bytearray(records["config1"]+records["config2"])
  jedec = bytes.fromhex("c2 28 15")
  config[:3], config[128:131] = b"\0"+jedec[1:], jedec  # ID fields on this board
  expected = (config+(build/"firmware_wrapped.bin").read_bytes()).ljust(1<<jedec[2], b"\xff")

  (device, bus), = USB3.list_devices(0x3801, 1)
  lock = System.flock_acquire(f"mt_{bus.lower()}.lock")
  usb = USB3(device)
  if flash_jedec_id(usb.handle) != jedec: raise ValueError("expected Chestnut with c2 28 15 flash")
  actual = flash_read(usb.handle, 0, len(expected))
  if actual != expected: raise ValueError("bridge flash differs from the source build and board configuration")
  print(f"Verified all {len(actual):,} external flash bytes on {bus}. SHA-256 {hashlib.sha256(actual).hexdigest()}")
