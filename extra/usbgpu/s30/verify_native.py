"""Run the S30 model benchmark and check the origin of each USC program loaded in this process."""
import runpy, sys
from pathlib import Path
from unittest.mock import patch
from tinygrad.renderer.usc import MTUSCRenderer
from tinygrad.runtime.support.mt.mtdev import MTSMC
from tinygrad.runtime.support.mt.s30 import MTS30Firmware, MTS30Program
from tinygrad.runtime.support.system import System, USBPCIDevice
from tinygrad.runtime.support.usb import CustomASM24Controller

if __name__ == "__main__":
  render, load, boot = MTUSCRenderer.render, MTS30Program.__init__, MTS30Firmware._boot
  emitted:set[bytes] = set()
  loaded:set[bytes] = set()

  def render_shader(self, uops):
    source = render(self, uops)
    emitted.add(bytes.fromhex(source))
    return source

  def check_shader(obj):
    if obj.target.renderer != "USC" or obj.lib not in emitted: raise RuntimeError(f"S30 shader {obj.name} was not generated in this run")
    loaded.add(obj.lib)

  def load_shader(self, dev, obj, links):
    check_shader(obj)
    load(self, dev, obj, links)

  def direct_boot(self, image, *, master=True, reset_domains=True):
    if master or reset_domains or any(image): raise RuntimeError("expected zeroed GPU context with META master boot disabled")
    return boot(self, image, master=master, reset_domains=reset_domains)

  with patch.object(MTUSCRenderer, "render", render_shader), patch.object(MTS30Program, "__init__", load_shader), \
       patch.object(MTS30Firmware, "_boot", direct_boot), patch.object(MTS30Firmware, "boot", side_effect=RuntimeError("vendor firmware boot")), \
       patch.object(MTSMC, "__init__", side_effect=RuntimeError("management firmware IPC")):
    if sys.argv[1] in ("sram-mnist", "sram-whisper"):
      from extra.usbgpu.s30 import sram
      module = sys.argv.pop(1).replace("-", "_")
      map_bar, f0_out, shader = USBPCIDevice.map_bar, CustomASM24Controller._f0_out, sram.s30_shader

      def register_bar(self, bar, *args, **kwargs):
        if bar != 0: raise RuntimeError(f"unexpected mapping of BAR{bar}")
        return map_bar(self, bar, *args, **kwargs)

      def register_request(self, fmt, byte_en, address, value, mode=0):
        if fmt in (0, 0x20, 0x40, 0x60):
          if not 0x10000000 <= address <= 0x12000000-(4*value if mode else 4): raise RuntimeError(f"memory TLP outside BAR0 at {address:#x}")
        elif fmt in (0x44, 0x45) and (address & 0xfff, byte_en) not in ((4, 3), (0x10, 15)):
          raise RuntimeError(f"unexpected configuration write at {address:#x}")
        return f0_out(self, fmt, byte_en, address, value, mode)

      upload = sram.ChestnutBoot.upload_mmu
      def bridge_tables(self, full=False):
        base = (1<<39) | 0x200000
        if not base <= self.mmu.root_pa <= base+0x80000-self.mmu.next: raise RuntimeError("page tables outside bridge SRAM")
        if any(not base <= pa < base+0x80000 and pa != (1<<39 | 0x822000) for pa in self.mmu.pages.values()):
          raise RuntimeError("GPU mapping outside Chestnut SRAM and completion queue")
        return upload(self, full)

      power = CustomASM24Controller.set_pcie_power
      def slot_power(self, enabled, *args, **kwargs):
        if enabled: raise RuntimeError("normal bridge power-on in cold SRAM diagnostic")
        return power(self, enabled, *args, **kwargs)

      def sram_shader(obj):
        check_shader(obj)
        return shader(obj)

      with patch.object(USBPCIDevice, "map_bar", register_bar), patch.object(CustomASM24Controller, "_f0_out", register_request), \
           patch.object(sram.ChestnutBoot, "upload_mmu", bridge_tables), patch.object(CustomASM24Controller, "set_pcie_power", slot_power), \
           patch.object(System, "pci_setup_usb_bars", side_effect=RuntimeError("generic PCI enumeration")), patch.object(sram, "s30_shader", sram_shader):
        runpy.run_module(f"extra.usbgpu.s30.{module}", run_name="__main__")
    else: runpy.run_path(str(Path(__file__).resolve().parents[1]/"bench_s30.py"), run_name="__main__")
  if not loaded: raise RuntimeError("benchmark loaded zero S30 shaders")
  print(f"Verified {len(loaded)} distinct S30 shader binaries generated from UOps in this process.")
