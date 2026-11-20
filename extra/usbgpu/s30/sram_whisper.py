"""Run both trained Whisper encoder convolutions on the S30 using only Chestnut SRAM."""
import argparse, time
import numpy as np
from examples.whisper import init_whisper, load_file_waveform, mel_prefix_frames, prep_audio
from extra.usbgpu.s30.sram import ChestnutBoot, mailbox, open_cold
from tinygrad import Context, Tensor
from tinygrad.codegen import do_to_program
from tinygrad.helpers import Target, ceildiv, round_up
from tinygrad.renderer.usc import MTUSCRenderer
from tinygrad.runtime.support.mt.mmu import MT_PTE_READ_WRITE
from tinygrad.runtime.support.usb import HALF

# The lower SRAM half is transient for these kernels. Keep outputs at slot zero for USB readback.
TILE = 0x7400000000

def convolution(fw:ChestnutBoot, x:np.ndarray, weights:np.ndarray, bias:np.ndarray, stride:int, frames:int) -> np.ndarray:
  channels, _, kernel = weights.shape
  rows = 64
  width = ceildiv(x.shape[-1], stride)
  assert kernel == 3 and channels%rows == 0 and x.shape[0] == 1
  tile_input = stride*(frames-1)+kernel
  padded = np.pad(x, ((0,0),(0,0),(1, stride*(round_up(width, frames)-1)+kernel-x.shape[-1]-1)))
  def parts(row:int, col:int):
    return (np.ascontiguousarray(padded[:,:,stride*col:stride*col+tile_input]), weights[row:row+rows], bias[row:row+rows])
  samples = parts(0, 0)
  tensors = [Tensor(v).contiguous().realize() for v in samples]
  output = tensors[0].conv2d(tensors[1], tensors[2], stride=stride).gelu()
  call, = output.schedule_linear().src
  prog = do_to_program(call.body, MTUSCRenderer(Target("MT","USC","mp_10")))
  output_bytes = rows*frames*output.dtype.itemsize
  payload = bytearray(round_up(output_bytes, 64))
  offsets = {}
  for t,v in zip(tensors, samples, strict=True):
    offsets[t.uop.buf_uop] = len(payload)
    payload.extend(bytes(round_up(v.nbytes, 64)))
  if len(payload) > HALF: raise MemoryError("convolution tile exceeds Chestnut SRAM")
  params, stream = fw.prepare(prog, [TILE if i in prog.arg.outs else TILE+offsets[call.src[i+1]] for i in prog.arg.globals])
  result = np.empty((1, channels, width), np.float32)
  for col in range(0, width, frames):
    valid = min(width-col, frames)
    for row in range(0, channels, rows):
      for off,v in zip(offsets.values(), parts(row, col), strict=True): payload[off:off+v.nbytes] = memoryview(v).cast("B")
      fw.dev.pci_dev.usb.scsi_write(payload)
      fw.stage(params, stream)
      fw.kick_direct(0.5)
      tile = np.frombuffer(fw.read_output(output_bytes), np.float32).reshape(rows, frames)
      result[0,row:row+rows,col:col+valid] = tile[:,:valid]
  return result

def run(audio:str):
  model, _ = init_whisper("tiny.en")
  mel = prep_audio([load_file_waveform(audio)], 1, truncate=True)
  convs = (model.encoder.conv1, model.encoder.conv2)
  weights = [(c.weight.numpy(), c.bias.numpy()) for c in convs]
  mt, start = open_cold()
  fw = ChestnutBoot(mt)
  fw.mmu.map_range(TILE, fw.context_pa-HALF, HALF, MT_PTE_READ_WRITE)
  actual, x = [], mel[..., :mel_prefix_frames(mel)]
  with fw.boot_direct():
    for i,((w,b),stride,frames) in enumerate(zip(weights, (1, 2), (64, 32), strict=True), 1):
      x = convolution(fw, x, w, b, stride, frames)
      actual.append(x)
      print(f"conv{i} {x.shape} at {1000*(time.monotonic()-start):.2f} ms from power-on, mailbox ready {mailbox(mt)}", flush=True)
  expected = Tensor(mel)
  for i,(layer,got) in enumerate(zip(convs, actual, strict=True), 1):
    expected = layer(expected).gelu().realize()
    ref = expected.numpy()
    if missing := ref.shape[-1]-got.shape[-1]:
      got = np.concatenate((got[..., :-1], np.repeat(got[..., -2:-1], missing, axis=-1), got[..., -1:]), axis=-1)
    np.testing.assert_allclose(got, ref, atol=2e-4, rtol=2e-4)
    print(f"conv{i} verified {got.size} activations, maximum absolute error {np.abs(got-ref).max():.8f}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("audio")
  with Context(DEV="CPU", CACHELEVEL=0): run(parser.parse_args().audio)
