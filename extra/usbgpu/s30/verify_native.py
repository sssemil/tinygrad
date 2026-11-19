"""Run the S30 model benchmark and check the origin of each USC program loaded in this process."""
import runpy
from pathlib import Path
from unittest.mock import patch
from tinygrad.renderer.usc import MTUSCRenderer
from tinygrad.runtime.support.mt.mtdev import MTSMC
from tinygrad.runtime.support.mt.s30 import MTS30Firmware, MTS30Program

if __name__ == "__main__":
  render, load = MTUSCRenderer.render, MTS30Program.__init__
  emitted:set[bytes] = set()
  loaded:set[bytes] = set()

  def render_shader(self, uops):
    source = render(self, uops)
    emitted.add(bytes.fromhex(source))
    return source

  def load_shader(self, dev, obj, links):
    if obj.target.renderer != "USC" or obj.lib not in emitted: raise RuntimeError(f"S30 shader {obj.name} was not generated in this run")
    loaded.add(obj.lib)
    load(self, dev, obj, links)

  with patch.object(MTUSCRenderer, "render", render_shader), patch.object(MTS30Program, "__init__", load_shader), \
       patch.object(MTS30Firmware, "boot", side_effect=RuntimeError("vendor firmware boot")), \
       patch.object(MTSMC, "__init__", side_effect=RuntimeError("management firmware IPC")):
    runpy.run_path(str(Path(__file__).resolve().parents[1]/"bench_s30.py"), run_name="__main__")
  if not loaded: raise RuntimeError("benchmark loaded zero S30 shaders")
  print(f"Verified {len(loaded)} distinct S30 shader binaries generated from UOps in this process.")
