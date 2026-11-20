"""Generate the Qwen benchmark response with every transformer block running in Chestnut SRAM."""
import argparse, functools, itertools, math, operator, time
from pathlib import Path
import numpy as np
from tinygrad import Context, Tensor
from tinygrad.llm.model import Transformer, apply_rope, precompute_freqs_cis
from extra.usbgpu.bench_s30 import qwen_prompt
from extra.usbgpu.s30.sram import open_cold
from extra.usbgpu.s30.sram_tensor import SRAMModel, SRAMTensorBoot, tiles

def _mean_square(x:Tensor): return x.square().mean(-1, keepdim=True)
def _normalize(x:Tensor, s:Tensor, w:Tensor, eps:float): return x*(s+eps).rsqrt()*w
def _score(q:Tensor, k:Tensor, mask:Tensor): return q @ k.T / math.sqrt(q.shape[-1]) + mask
def _gate(g:Tensor, u:Tensor): return g.silu()*u
def _embed(w:Tensor, i:Tensor): return w[i].float()

class StreamedQwen(SRAMModel):
  def __init__(self, fw:SRAMTensorBoot, model:Transformer, capacity:int):
    super().__init__(fw)
    self.model = model
    cfg = model.config
    self.normalize = functools.partial(_normalize, eps=cfg.norm_eps)
    self.cache = np.zeros((len(model.blk), 2, cfg.n_kv_heads, capacity, cfg.head_dim), np.float16)
    self.freqs = precompute_freqs_cis(cfg.rope_dim, capacity, cfg.rope_theta).numpy()

  def norm(self, x:np.ndarray, w:np.ndarray) -> np.ndarray: return self.op(self.normalize, x, self.op(_mean_square, x), w)

  def project(self, x:np.ndarray, w:np.ndarray, bias:np.ndarray|None=None) -> np.ndarray:
    # Four prompt positions and one weight tile fit beside their output in the 256 KiB staging area.
    rows = 16 if w.shape[1] > 1024 else 128
    return self.projection(x, w, rows, 4, bias)

  def rotate(self, x:np.ndarray, heads:int, start:int) -> np.ndarray:
    x = x.reshape(len(x), heads, -1)
    return tiles(lambda a,f: self.op(apply_rope, a[None].transpose(0,2,1,3), f[None])[0].transpose(1,0,2), x, self.freqs[start:start+len(x)], size=4)

  def attention(self, x:np.ndarray, layer:int, start:int) -> np.ndarray:
    b, cfg = self.model.blk[layer], self.model.config
    q, k, v = (self.project(x, l.weight.numpy(), l.bias.numpy()) for l in (b.attn_q,b.attn_k,b.attn_v))
    q, k = self.rotate(q, cfg.n_heads, start), self.rotate(k, cfg.n_kv_heads, start)
    cache = self.cache[layer]
    for dst, a in zip(cache, (k,v.reshape(k.shape)), strict=True):
      dst[:,start:start+len(x)] = self.op(Tensor.half, a).transpose(1,0,2)
    group = cfg.n_heads//cfg.n_kv_heads
    out = np.empty_like(q)
    for h,(keys,values) in enumerate(zip(*cache, strict=True)):
      for i in range(0,len(x),4):
        qs = q[i:i+4,h*group:(h+1)*group]
        pos = (start+i+np.arange(len(qs))).repeat(group)
        mask = np.where(np.arange(len(keys)) <= pos[:,None], 0, -np.inf).astype(np.float32)
        scores = self.op(_score, qs.reshape(-1, cfg.head_dim), keys, mask)
        probs = self.softmax(scores)
        out[i:i+4,h*group:(h+1)*group] = self.op(operator.matmul, probs, values).reshape(qs.shape)
    return self.project(out.reshape(len(x),-1), b.attn_output.weight.numpy())

  def next_token(self, tokens:list[int], start:int) -> int:
    weights = self.model.token_embd.weight
    # Select the embedding page for transport. The GPU indexes the row within it.
    x = np.stack([self.op(_embed, weights[t//128*128:t//128*128+128].numpy(), np.array([t%128], np.int32))[0] for t in tokens])
    for layer,b in enumerate(self.model.blk):
      att = self.attention(tiles(functools.partial(self.norm, w=b.attn_norm.weight.numpy()), x, size=4), layer, start)
      x = tiles(functools.partial(self.op, operator.add), x, att, size=4)
      if layer == len(self.model.blk)-1: x = x[-1:]  # only the last position supplies logits
      n = tiles(functools.partial(self.norm, w=b.ffn_norm.weight.numpy()), x, size=4)
      gate, up = (self.project(n, l.weight.numpy()) for l in (b.ffn_gate, b.ffn_up))
      hidden = tiles(functools.partial(self.op, _gate), gate, up, size=4)
      ffn = self.project(hidden, b.ffn_down.weight.numpy())
      x = tiles(functools.partial(self.op, operator.add), x, ffn, size=4)
    z = self.norm(x, self.model.output_norm.weight.numpy())
    return self.argmax(self.project(z, self.model.output.weight.numpy()))

def run(weights:Path, question:str|None):
  model, kv = Transformer.from_gguf(weights, 128, realize=True)
  if kv["general.architecture"] != "qwen2" or (model.config.dim, model.config.hidden_dim, model.config.n_heads) != (896,4864,14):
    raise ValueError("expected the Qwen2.5-0.5B benchmark checkpoint")
  prompt = qwen_prompt(kv) if question is None else qwen_prompt(kv, question)
  if len(prompt)+8 > model.max_context: raise ValueError("prompt and eight new tokens must fit the 128-token context")
  tokens, start = prompt.copy(), 0
  mt, begin = open_cold()
  with (fw:=SRAMTensorBoot(mt)).boot_direct():
    runner = StreamedQwen(fw, model, 1 << (len(prompt)+7).bit_length())
    for _ in range(8):
      token = runner.next_token(tokens[start:], start)
      print(f"token {token} at {time.monotonic()-begin:.3f} s from slot power-on", flush=True)
      start = len(tokens)
      tokens.append(token)
      if token == kv["tokenizer.ggml.eos_token_id"]: break
  generated = tokens[len(prompt):]
  reference = list(itertools.islice(model.generate(prompt.copy()), len(generated)))
  if generated != reference: raise ValueError(f"S30 returned {generated}, CPU returned {reference}")
  print(f"Verified {len(generated)} tokens against the CPU transformer")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("weights", type=Path)
  parser.add_argument("--prompt", help="user message for the model chat template")
  args = parser.parse_args()
  with Context(DEV="CPU", CACHELEVEL=0): run(args.weights, args.prompt)
