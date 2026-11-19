"""Classify one MNIST image on the S30 using only Chestnut SRAM, with all model kernels compiled after slot power-on."""
import argparse, time
from pathlib import Path
from examples.beautiful_mnist import Model
from extra.usbgpu.s30.sram import ACTIVATIONS, ACTIVATION_SIZE, INPUT, RESULT, ChestnutBoot, cold_link, mailbox
from tinygrad import Context, UOp
from tinygrad.codegen import do_to_program
from tinygrad.helpers import Target, prod, round_up
from tinygrad.nn.datasets import mnist
from tinygrad.nn.state import load_state_dict, safe_load
from tinygrad.renderer.usc import MTUSCRenderer
from tinygrad.runtime.support.mt.mtdev import MTDev
from tinygrad.runtime.support.mt.s30 import DEFAULT_S30_FIRMWARE
from tinygrad.uop.ops import Ops

def classify(weights:Path, index:int):
  # Prepare the inputs and obtain the ordinary model schedule. Only its S30 programs execute the inference graph.
  model = Model()
  load_state_dict(model, safe_load(weights))
  _, _, images, labels = mnist(device="CPU")
  if not 0 <= index < images.shape[0]: raise IndexError(index)
  image, label = images[index:index+1].contiguous().realize(), labels[index].item()
  linear = model(image).argmax(-1).schedule_linear()
  mt = MTDev.open_usb(manage_power=False)
  fw = ChestnutBoot(mt, DEFAULT_S30_FIRMWARE)
  start = cold_link(mt)
  renderer = MTUSCRenderer(Target("MT","USC","mp_10"))
  resident:dict[UOp, int] = {}
  used = 0
  with fw.boot_direct():
    for i,call in enumerate(linear.src):
      if call.op is not Ops.CALL or call.body.op is not Ops.SINK: raise ValueError("expected a model kernel")
      prg = do_to_program(call.body, renderer)
      inputs, addresses = bytearray(), []
      for slot,param in zip(prg.arg.globals, prg.arg.params, strict=True):
        arg = call.src[slot+1]
        if slot in prg.arg.outs:
          size = prod(param.shape)*param.dtype.itemsize
          if i == len(linear.src)-1:
            assert size == 4
            resident[arg] = RESULT
          else:
            resident[arg] = ACTIVATIONS+used
            used += round_up(size, 64)
            if used > ACTIVATION_SIZE: raise MemoryError("MNIST activations exceed Chestnut SRAM")
        if arg in resident: addresses.append(resident[arg])
        else:
          if not arg.buffer.is_allocated(): raise ValueError("expected a resident input tensor")
          addresses.append(INPUT+len(inputs))
          inputs += arg.buffer.as_memoryview()
          inputs += bytes(-len(inputs)%64)
      fw.launch(inputs, *fw.prepare(prg, addresses))
    prediction = fw.result()
    elapsed, ready = time.monotonic()-start, mailbox(mt)
  print(f"MNIST[{index}] predicted {prediction}, label {label}, {len(linear.src)} S30 kernels, {elapsed*1000:.2f} ms from power-on, mailbox ready {ready}")
  if ready: raise RuntimeError("management rings initialized before inference completed")
  if prediction != label: raise ValueError(f"prediction differs from dataset label {label}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("weights", type=Path)
  parser.add_argument("--index", type=int, default=0)
  args = parser.parse_args()
  with Context(DEV="CPU", CACHELEVEL=0): classify(args.weights, args.index)
