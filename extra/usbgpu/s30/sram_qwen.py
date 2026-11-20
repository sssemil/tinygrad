"""Run Qwen2.5-0.5B's first feed-forward block for the final prompt position on the S30 using only Chestnut SRAM."""
import argparse, time
from pathlib import Path
import numpy as np
from extra.usbgpu.bench_s30 import qwen_prompt
from extra.usbgpu.s30.sram import ChestnutBoot, mailbox, open_cold
from extra.usbgpu.s30.sram_tensor import SRAMKernel
from tinygrad import Context, Tensor
from tinygrad.llm.model import Transformer

def projections(fw:ChestnutBoot, x:np.ndarray, rows:int, *weights:np.ndarray) -> tuple[np.ndarray,...]:
  kernel = SRAMKernel(fw, lambda v,w: w@v, x, weights[0][:rows])
  return tuple(np.concatenate([kernel(x, w[i:i+rows]) for i in range(0, len(w), rows)]) for w in weights)

def run(weights:Path):
  model, kv = Transformer.from_gguf(weights, 128, realize=False)
  tokens = qwen_prompt(kv)
  x = model.token_embd(Tensor([tokens])).float().realize()
  block = model.blk[0]
  block._init_state(x)
  # CPU fixture ends at the input to the feed-forward module, after the real prompt attention and residual.
  residual = x + block._attention(block.attn_norm(x), (0,))
  normalized = block.ffn_norm(residual[:,-1:]).realize()
  vector = normalized.numpy().ravel()
  gate_w, up_w, down_w = (layer.weight.numpy() for layer in (block.ffn_gate, block.ffn_up, block.ffn_down))
  mt, start = open_cold()
  fw = ChestnutBoot(mt)
  with fw.boot_direct():
    gate, up = projections(fw, vector, 128, gate_w, up_w)
    hidden = SRAMKernel(fw, lambda g,u: g.silu()*u, gate, up)(gate, up)
    out, = projections(fw, hidden, 16, down_w)
    elapsed, ready = time.monotonic()-start, mailbox(mt)
  ref = block._feed_forward(normalized).numpy().ravel()
  np.testing.assert_allclose(out, ref, atol=5e-4, rtol=5e-4)
  print(f"Qwen block 0 FFN, final position of {len(tokens)} tokens, {1000*elapsed:.2f} ms from power-on, mailbox ready {ready}")
  print(f"Verified {out.size} activations, maximum absolute error {np.abs(out-ref).max():.8f}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("weights", type=Path)
  with Context(DEV="CPU", CACHELEVEL=0): run(parser.parse_args().weights)
