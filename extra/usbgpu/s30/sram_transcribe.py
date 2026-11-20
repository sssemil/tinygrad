"""Transcribe one 30-second Whisper tiny.en segment on the S30 using only Chestnut SRAM for GPU-visible memory."""
import argparse, functools, math, operator, time
import numpy as np
from tinygrad import Context, Tensor, dtypes
from examples.whisper import init_whisper, load_file_waveform, mel_prefix_frames, prep_audio, transcribe_waveform
from extra.usbgpu.s30.sram import open_cold
from extra.usbgpu.s30.sram_tensor import SRAMModel, SRAMTensorBoot, tiles
from extra.usbgpu.s30.sram_whisper import convolution

def _mean(x:Tensor): return x.mean(-1, keepdim=True)
def _mean_square(x:Tensor): return x.square().mean(-1, keepdim=True)
def _scale(x:Tensor, var:Tensor, w:Tensor, b:Tensor, eps:float): return x*(var+eps).rsqrt()*w+b
def _scores(x:Tensor, k:Tensor): return x.matmul(k.T, dtype=dtypes.float)/math.sqrt(x.shape[-1])
def _embedding(w:Tensor, idx:Tensor): return w[idx]

class StreamedWhisper(SRAMModel):
  def __init__(self, fw:SRAMTensorBoot, eps:float):
    super().__init__(fw)
    self.eps, self.scale = eps, functools.partial(_scale, eps=eps)

  def norm(self, x:np.ndarray, layer) -> np.ndarray:
    assert layer.eps == self.eps
    w, b = layer.weight.numpy(), layer.bias.numpy()
    def apply(v):
      centered = self.op(operator.sub, v, self.op(_mean, v))
      return self.op(self.scale, centered, self.op(_mean_square, centered), w, b)
    return tiles(apply, x, size=16)

  def linear(self, x:np.ndarray, layer) -> np.ndarray:
    w = layer.weight.numpy()
    return self.projection(x, w, 32 if w.shape[1] > 384 else 128, 16, layer.bias.numpy() if getattr(layer, "bias", None) is not None else None)

  def attention(self, x:np.ndarray, normalized:np.ndarray, attn, k:np.ndarray, v:np.ndarray, pos:int|None) -> np.ndarray:
    q = self.linear(normalized, attn.query)
    def heads(a): return a.reshape(len(a), attn.n_head, -1).transpose(1,0,2)
    out = np.empty((len(q), attn.n_head, q.shape[-1]//attn.n_head), dtype=np.result_type(q.dtype, v.dtype))
    for h,(query,key,value) in enumerate(zip(heads(q), heads(k), heads(v), strict=True)):
      for i in range(0,len(q),16):
        qt = query[i:i+16]
        scores = np.concatenate([self.op(_scores, qt, key[j:j+384]) for j in range(0,len(key),384)], axis=1)
        if pos is not None:
          mask = np.where(np.arange(len(key)) <= (pos+i+np.arange(len(qt)))[:,None], 0, -np.inf).astype(q.dtype)
          scores = self.op(operator.add, scores, mask)
        probs = self.softmax(scores if scores.dtype == q.dtype else self.op(Tensor.half, scores))
        out[i:i+16,h] = np.concatenate([self.op(operator.matmul, probs, value[:,j:j+16]) for j in range(0,value.shape[1],16)], axis=1)
    return tiles(functools.partial(self.op, operator.add), x, self.linear(out.reshape(len(x),-1), attn.out), size=16)

  def block(self, x:np.ndarray, b, cache=None, cross=None, pos:int|None=None) -> np.ndarray:
    n = self.norm(x, b.attn_ln)
    k, v = (self.linear(n, layer) for layer in (b.attn.key,b.attn.value))
    if cache is not None:
      for dst,a in zip(cache, (k,v), strict=True): dst[pos:pos+len(x)] = a if a.dtype == np.float32 else self.op(Tensor.float, a)
      k, v = cache
    x = self.attention(x, n, b.attn, k, v, pos)
    if cross is not None: x = self.attention(x, self.norm(x,b.cross_attn_ln), b.cross_attn, *cross, None)
    mlp = self.linear(self.norm(x,b.mlp_ln), b.mlp[0])
    mlp = tiles(functools.partial(self.op, Tensor.gelu), mlp, size=8)
    return tiles(functools.partial(self.op, operator.add), x, self.linear(mlp,b.mlp[2]), size=16)

  def encode(self, mel:np.ndarray, encoder) -> np.ndarray:
    x = mel[...,:mel_prefix_frames(mel)]
    for layer,stride,frames in ((encoder.conv1,1,64),(encoder.conv2,2,32)):
      x = convolution(self.fw, x, layer.weight.numpy(), layer.bias.numpy(), stride, frames)
    pos, x = encoder.positional_embedding.numpy(), x[0].T
    if missing:=len(pos)-len(x): x = np.concatenate([x[:-1], np.repeat(x[-2:-1], missing, axis=0), x[-1:]])
    x = tiles(functools.partial(self.op, operator.add), x, pos, size=16)
    for i,b in enumerate(encoder.blocks):
      x = self.block(x,b)
      print(f"encoder block {i+1}/{len(encoder.blocks)}", flush=True)
    return self.norm(x, encoder.ln_post)

  def decode(self, xa:np.ndarray, decoder, enc) -> str:
    cross = [[self.linear(xa,l) for l in (b.cross_attn.key,b.cross_attn.value)] for b in decoder.blocks]
    cache = np.zeros((len(cross),2,*decoder.positional_embedding.shape), np.float32)
    ids, words, pos = [enc._special_tokens[k] for k in ("<|startoftranscript|>","<|notimestamps|>")], [], 0
    for _ in range(decoder.max_tokens_to_sample):
      w = decoder.token_embedding.weight
      emb = np.stack([self.op(_embedding, w[t//128*128:t//128*128+128].numpy(), np.array([t%128],np.int32))[0] for t in ids])
      x = self.op(operator.add, emb, decoder.positional_embedding[pos:pos+len(ids)].numpy())
      for i,b in enumerate(decoder.blocks): x = self.block(x,b,cache[i],cross[i],pos)
      pos += len(ids)
      token = self.argmax(self.linear(self.norm(x[-1:],decoder.ln),decoder.token_embedding))
      if token == enc._special_tokens["<|endoftext|>"]: break
      words.append(token)
      ids = [token]
    return enc.decode(words).strip()

def run(audio:str):
  m, enc = init_whisper("tiny.en", audio_ctx_alignment=64)
  waveform = load_file_waveform(audio)
  mel = prep_audio([waveform], 1, truncate=True)
  mt, start = open_cold()
  with (fw:=SRAMTensorBoot(mt)).boot_direct():
    runner = StreamedWhisper(fw, m.encoder.ln_post.eps)
    transcript = runner.decode(runner.encode(mel,m.encoder),m.decoder,enc)
    elapsed = time.monotonic()-start
  reference = transcribe_waveform(m, enc, [waveform], truncate=True)
  if transcript != reference: raise ValueError(f"S30 returned {transcript!r}, CPU returned {reference!r}")
  print(f"Verified transcript in {elapsed:.3f} s from slot power-on: {transcript}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("audio")
  with Context(DEV="CPU", CACHELEVEL=0): run(parser.parse_args().audio)
