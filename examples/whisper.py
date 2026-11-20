# thanks to https://github.com/openai/whisper for a good chunk of MIT licensed code

import sys, base64, multiprocessing, itertools, collections, functools, math, operator
from typing import Optional, Union, Literal, List

from tinygrad import Device, Tensor, TinyJit, UOp, Variable, nn, dtypes
from tinygrad.nn.state import torch_load, load_state_dict
from tinygrad.dtype import least_upper_dtype, sum_acc_dtype
from tinygrad.uop.ops import KernelInfo, Ops
from tinygrad.helpers import getenv, fetch, round_up

from examples.audio_helpers import mel
import numpy as np
import librosa

def _token_norm_kernel(out:UOp, x:UOp, weight:UOp, bias:UOp, eps:float) -> UOp:
  width = x.shape[-1]
  row = UOp.range(x.shape[0], 0)
  i, j, k = (UOp.range(width, axis) for axis in (1, 2, 3))
  def mean(v, axis): return (v.float().reduce(axis, arg=Ops.ADD) / width).cast(x.dtype)
  mu = mean(x[row, j], j)
  var = mean((x[row, k] - mu).square(), k)
  y = (x[row, i] - mu) * (var + eps).rsqrt()
  return out[row, i].store(y * weight[i] + bias[i]).end(row, i).sink(arg=KernelInfo(name="whisper_token_norm"))

def _token_norm(layer, x):
  flat = x.flatten(end_dim=1)
  out = flat.empty_like(dtype=least_upper_dtype(x.dtype, layer.weight.dtype, layer.bias.dtype))
  return out.custom_kernel(flat.contiguous(), layer.weight, layer.bias, fxn=functools.partial(_token_norm_kernel, eps=layer.eps))[0].reshape(x.shape)

def _token_qkv(q, keys, values, x, qw, kw, vw, qb, vb, pos):
  b, t, i = (UOp.range(n, axis) for axis, n in enumerate(q.shape))
  j = UOp.range(x.shape[-1], 3)
  def project(w):
    product = x[b, t, j] * w[i, j]
    return product.float().reduce(j, arg=Ops.ADD).cast(product.dtype)
  stores = (q[b, t, i].store(project(qw) + qb[i]), keys[b, pos+t, i].store(project(kw).cast(keys.dtype)),
            values[b, pos+t, i].store((project(vw) + vb[i]).cast(values.dtype)))
  return UOp.group(*stores).end(b, t, i).sink(arg=KernelInfo(name="whisper_token_qkv"))

def _token_weighted_sum(out, score, values, pos):
  b, h, t, i = (UOp.range(n, axis) for axis, n in enumerate(out.shape))
  m, s, k = (UOp.range(pos+1, axis) for axis in (4, 5, 6))
  peak = score[b, h, t, m].reduce(m, arg=Ops.MAX)
  def exponential(j): return (score[b, h, t, j] - peak).exp()
  inv_sum = exponential(s).float().reduce(s, arg=Ops.ADD).cast(score.dtype).reciprocal()
  weighted = (exponential(k) * inv_sum) * values[b, k, h*out.shape[-1]+i]
  return out[b, h, t, i].store(weighted.float().reduce(k, arg=Ops.ADD).cast(out.dtype)).end(b, h, t, i).sink(arg=KernelInfo(name="whisper_token_attn"))

class MultiHeadAttention:
  def __init__(self, n_state, n_head, kv_caching: Literal['cross', 'self']=None, max_self_attn_cache_len=None):
    self.n_head = n_head
    self.query = nn.Linear(n_state, n_state)
    self.key = nn.Linear(n_state, n_state, bias=False)
    self.value = nn.Linear(n_state, n_state)
    self.out = nn.Linear(n_state, n_state)

    self.kv_caching = kv_caching
    self.max_self_attn_cache_len = max_self_attn_cache_len

  def decode_token(self, x, pos):
    step = Tensor(pos, device=x.device)
    q = x.empty_like(dtype=least_upper_dtype(x.dtype, self.query.weight.dtype, self.query.bias.dtype))
    q, keys, values, *_ = q.custom_kernel(self.cache_k, self.cache_v, x.contiguous(), self.query.weight, self.key.weight, self.value.weight,
                                          self.query.bias, self.value.bias, step, fxn=_token_qkv)
    def heads(a): return a.reshape(*a.shape[:2], self.n_head, a.shape[-1]//self.n_head).transpose(1, 2)
    q.realize(keys, values)
    q, k = heads(q), heads(keys[:, :pos+1])
    # Only the cached prefix participates in softmax. The custom kernel takes a statically shaped score buffer.
    score = (q.matmul(k.transpose(-2,-1), dtype=dtypes.float)/math.sqrt(q.shape[-1])).cast(q.dtype).pad((0, values.shape[1]-pos-1)).contiguous()
    out = q.empty_like(dtype=least_upper_dtype(q.dtype, values.dtype)).custom_kernel(score, values, step, fxn=_token_weighted_sum)[0]
    return self.out(out.transpose(1, 2).flatten(start_dim=2))

  def cache_audio(self, xa:Tensor, frames:int):
    k, v = self.key(xa), self.value(xa)
    if frames != xa.shape[1]: k, v = k.realize()[:, :frames], v.realize()[:, :frames]
    if not hasattr(self, 'cache_k'):
      self.cache_k, self.cache_v = k.empty_like().realize(), v.empty_like().realize()
    self.cache_k.assign(k).realize()
    self.cache_v.assign(v).realize()

  def __call__(self, x:Tensor, xa:Optional[Tensor]=None, mask:Optional[Tensor]=None, len: Union[Variable,int]=None):
    if self.kv_caching == 'cross':
      if xa is not None: self.cache_audio(xa, xa.shape[1])
      k, v = self.cache_k, self.cache_v
    else:
      if self.kv_caching == 'self': x = x.realize()  # shared by the separate key and value cache writes
      k, v = self.key(x), self.value(x)
      if self.kv_caching == 'self':
        if not hasattr(self, 'cache_k'):
          self.cache_k = Tensor.empty(x.shape[0], self.max_self_attn_cache_len, x.shape[2], device=x.device)
          self.cache_v = self.cache_k.empty_like()
        end = len + x.shape[1]
        self.cache_k[:, len:end].assign(k.cast(self.cache_k.dtype)).realize()
        self.cache_v[:, len:end].assign(v.cast(self.cache_v.dtype)).realize()
        k, v = self.cache_k[:, :end], self.cache_v[:, :end]

    q = self.query(x)
    n_ctx = q.shape[1]
    assert(q.shape[-1] == k.shape[-1] == v.shape[-1])
    head_dim = q.shape[-1] // self.n_head
    q = q.reshape(*q.shape[:2], self.n_head, head_dim).permute(0, 2, 1, 3)
    k = k.reshape(*k.shape[:2], self.n_head, head_dim).permute(0, 2, 1, 3)
    v = v.reshape(*v.shape[:2], self.n_head, head_dim).permute(0, 2, 1, 3)
    attn = Tensor.scaled_dot_product_attention(q, k, v, mask[:n_ctx,:n_ctx] if mask is not None else None)
    wv = attn.permute(0, 2, 1, 3).flatten(start_dim=2)
    return self.out(wv)


class ResidualAttentionBlock:
  def __init__(self, n_state, n_head, is_decoder_block=False, max_self_attn_cache_len=None):
    self.attn = MultiHeadAttention(n_state, n_head, kv_caching='self' if is_decoder_block else None, max_self_attn_cache_len=max_self_attn_cache_len)
    self.attn_ln = nn.LayerNorm(n_state)

    self.cross_attn = MultiHeadAttention(n_state, n_head, kv_caching='cross') if is_decoder_block else None
    self.cross_attn_ln = nn.LayerNorm(n_state) if is_decoder_block else None

    self.mlp = [nn.Linear(n_state, n_state*4), Tensor.gelu, nn.Linear(n_state*4, n_state)]
    self.mlp_ln = nn.LayerNorm(n_state)

  def __call__(self, x, xa=None, mask=None, len: Union[Variable, int]=None, token_kernels=False):
    norm = _token_norm if token_kernels else operator.call
    normalized = norm(self.attn_ln, x)
    if token_kernels and x.shape[1] == 1 and isinstance(len, UOp): x = x + self.attn.decode_token(normalized, len)
    else: x = x + self.attn(normalized, mask=mask, len=len)
    if self.cross_attn: x = x + self.cross_attn(norm(self.cross_attn_ln, x), xa)
    x = x + norm(self.mlp_ln, x).sequential(self.mlp)
    return x.realize()

def mel_prefix_frames(mel:np.ndarray) -> int:
  # Keep six identical frames for both convolution boundaries and bucket the JIT input length.
  n = mel.shape[-1]
  if n < 6 or n%2: return n
  changes = np.flatnonzero(np.any(mel[..., :-1] != mel[..., -1:], axis=(0, 1)))
  return min(n, round_up((int(changes[-1])+1 if changes.size else 0)+6, 64))

class AudioEncoder:
  def __init__(self, n_mels, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer, audio_ctx_alignment=1, **_):
    self.audio_ctx_alignment = audio_ctx_alignment
    self.conv1 = nn.Conv1d(n_mels, n_audio_state, kernel_size=3, padding=1)
    self.conv2 = nn.Conv1d(n_audio_state, n_audio_state, kernel_size=3, stride=2, padding=1)
    self.blocks = [ResidualAttentionBlock(n_audio_state, n_audio_head) for _ in range(n_audio_layer)]
    self.ln_post = nn.LayerNorm(n_audio_state)
    self.positional_embedding = Tensor.empty(n_audio_ctx, n_audio_state)
    self.encode = TinyJit(self.__call__)
    self._prefix_encode = TinyJit(functools.partial(self.__call__, pad_silence=True))
    self._prefix_shape:tuple|None = None

  def encode_silence_tail(self, x:Tensor) -> Tensor:
    if x.shape != self._prefix_shape:
      self._prefix_encode.reset()
      self._prefix_shape = x.shape
    return self._prefix_encode(x)

  def __call__(self, x, pad_silence=False):
    if pad_silence:
      assert 6 <= x.shape[-1] <= 2*self.positional_embedding.shape[0] and x.shape[-1]%2 == 0, "expected an even mel prefix with six silent frames"
    x = self.conv1(x).gelu()
    x = self.conv2(x).gelu()
    x = x.permute(0, 2, 1)
    if pad_silence and (missing := self.positional_embedding.shape[0] - x.shape[1]):
      # The caller supplies an even mel prefix ending in six constant frames. Only the last output sees the right boundary.
      x = x.realize()
      x = x[:, :-1].cat(x[:, -2:-1].expand(-1, missing, -1), x[:, -1:], dim=1)
    x = x + self.positional_embedding[:x.shape[1]]
    n = x.shape[1]
    mask = None
    if pad := -n % self.audio_ctx_alignment:
      x = x.pad((0, 0, 0, pad))
      mask = (Tensor.arange(n+pad).to(x.device) < n).reshape(1, n+pad)
    for block in self.blocks: x = block(x, mask=mask)
    x = self.ln_post(x[:, :n])
    return x.realize()

class TextDecoder:
  def __init__(self, n_vocab, n_text_ctx, n_text_state, n_text_head, n_text_layer, audio_ctx_alignment=1, **_):
    self.audio_ctx_alignment = audio_ctx_alignment
    self.max_tokens_to_sample = n_text_ctx // 2
    self.max_self_attn_cache_len = n_text_ctx

    self.token_embedding = nn.Embedding(n_vocab, n_text_state)
    self.positional_embedding = Tensor.empty(n_text_ctx, n_text_state)
    self.blocks = [ResidualAttentionBlock(n_text_state, n_text_head, is_decoder_block=True, max_self_attn_cache_len=self.max_self_attn_cache_len) for _ in range(n_text_layer)]
    self.ln = nn.LayerNorm(n_text_state)
    self.mask = Tensor.full((n_text_ctx, n_text_ctx), -np.inf).triu(1).realize()
    self.getjitted = collections.defaultdict(lambda: TinyJit(self.forward))

  def __call__(self, x: Tensor, pos: int, encoded_audio: Optional[Tensor], *, greedy=False):
    pos = Variable("self_attn_cache_len", 1, self.max_self_attn_cache_len-1).bind(pos) if pos else 0
    return self.getjitted[x.shape, encoded_audio is None, greedy](x, pos, encoded_audio, greedy)

  def forward(self, x:Tensor, pos:Union[Variable, Literal[0]], encoded_audio:Optional[Tensor], greedy=False):
    seqlen = x.shape[-1]
    x = self.token_embedding(x) + self.positional_embedding.shrink(((pos, pos+seqlen), None))
    # A short decoder step on the USB S30 spends more time launching the separate LayerNorm reductions than recomputing them per lane.
    token_kernels = greedy and x.device == "MT" and Device["MT"].renderer.target.arch == "mp_10" and x.shape[0]*seqlen <= 2 and x.shape[-1] <= 384
    frames = encoded_audio.shape[1] if encoded_audio is not None else 0
    if encoded_audio is not None and (pad := -frames % self.audio_ctx_alignment):
      encoded_audio = encoded_audio.pad((0, 0, 0, pad)).contiguous().realize()
    for block in self.blocks:
      if encoded_audio is not None: block.cross_attn.cache_audio(encoded_audio, frames)
      x = block(x, mask=self.mask, len=pos, token_kernels=token_kernels)
    if greedy:
      # Score only the last position, splitting each vocabulary dot product into parallel reductions.
      w = self.token_embedding.weight
      tile = (-1, math.gcd(32, w.shape[-1]))
      norm = _token_norm if token_kernels else operator.call
      products = norm(self.ln, x[:, -1:]).unflatten(-1, tile) * w.unflatten(-1, tile)
      partials = products.sum(-1, dtype=sum_acc_dtype(products.dtype)).contiguous().realize()
      return partials.sum(-1).cast(products.dtype).argmax(-1).realize()
    return self.output_tok(x)

  def output_tok(self, x):
    return (self.ln(x) @ self.token_embedding.weight.T).realize()

class Whisper:
  def __init__(self, dims, batch_size=1, audio_ctx_alignment=1):
    self.encoder = AudioEncoder(**dims, audio_ctx_alignment=audio_ctx_alignment)
    self.decoder = TextDecoder(**dims, audio_ctx_alignment=audio_ctx_alignment)
    self.is_multilingual = dims["n_vocab"] == 51865
    self.batch_size = batch_size


RATE = 16000
SEGMENT_SECONDS=30
SAMPLES_PER_SEGMENT = RATE * SEGMENT_SECONDS # 480000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
FRAMES_PER_SEGMENT = SAMPLES_PER_SEGMENT // HOP_LENGTH # 3000

@functools.cache
def mel_filters():
  return mel(sr=RATE, n_fft=N_FFT, n_mels=N_MELS).numpy()

def prep_audio(waveforms: List[np.ndarray], batch_size: int, truncate=False) -> np.ndarray:
  """
  :param waveforms: A list of possibly variable length 16000Hz audio samples
  :param batch_size: The batch_size associated with the Whisper model being used to transcribe the audio.
                     Used to prevent JIT mismatch errors since the encoder does not accept symbolic shapes
  :param truncate: If true, truncates (or pads) audio to exactly 30s for a single encoder pass
  :return: mel spectrogram of the given waveforms
  """
  def pad_or_trim(arr, target_len):
    curr_len = len(arr)
    if curr_len == target_len:
      return arr
    elif curr_len < target_len:
      return np.pad(arr, (0, target_len - curr_len), 'constant')
    else:
      return arr[:target_len]

  max_len = SAMPLES_PER_SEGMENT if truncate else max(len(wav) for wav in waveforms)
  if (r := max_len % SAMPLES_PER_SEGMENT) > 0: max_len += SAMPLES_PER_SEGMENT - r
  waveforms = np.array(list(map(lambda w: pad_or_trim(w, max_len), waveforms)))
  assert waveforms.shape[0] <= batch_size
  if waveforms.shape[0] < batch_size:
    # we could have a symbolic batch_size dim instead of manually padding here if conv/layernorm supported symbolic shapes
    waveforms = np.pad(waveforms, pad_width=((0, batch_size - waveforms.shape[0]), (0, 0)))

  return log_mel_spectrogram(waveforms)

def log_mel_spectrogram(waveforms:np.ndarray) -> np.ndarray:
  stft = librosa.stft(waveforms, n_fft=N_FFT, hop_length=HOP_LENGTH, window='hann', dtype=np.csingle)
  magnitudes = np.absolute(stft[..., :-1]) ** 2
  mel_spec = mel_filters() @ magnitudes

  log_spec = np.log10(np.clip(mel_spec, 1e-10, None))
  log_spec = np.maximum(log_spec, log_spec.max((1,2), keepdims=True) - 8.0)
  log_spec = (log_spec + 4.0) / 4.0

  return log_spec

LANGUAGES = {
  "en": "english", "zh": "chinese", "de": "german", "es": "spanish", "ru": "russian", "ko": "korean", "fr": "french", "ja": "japanese", "pt": "portuguese", "tr": "turkish",
  "pl": "polish", "ca": "catalan", "nl": "dutch", "ar": "arabic", "sv": "swedish", "it": "italian", "id": "indonesian", "hi": "hindi", "fi": "finnish", "vi": "vietnamese",
  "he": "hebrew", "uk": "ukrainian", "el": "greek", "ms": "malay", "cs": "czech", "ro": "romanian", "da": "danish", "hu": "hungarian", "ta": "tamil", "no": "norwegian",
  "th": "thai", "ur": "urdu", "hr": "croatian", "bg": "bulgarian", "lt": "lithuanian", "la": "latin", "mi": "maori", "ml": "malayalam", "cy": "welsh", "sk": "slovak", "te": "telugu",
  "fa": "persian", "lv": "latvian", "bn": "bengali", "sr": "serbian", "az": "azerbaijani", "sl": "slovenian", "kn": "kannada", "et": "estonian", "mk": "macedonian",
  "br": "breton", "eu": "basque", "is": "icelandic", "hy": "armenian", "ne": "nepali", "mn": "mongolian", "bs": "bosnian", "kk": "kazakh", "sq": "albanian", "sw": "swahili",
  "gl": "galician", "mr": "marathi", "pa": "punjabi", "si": "sinhala", "km": "khmer", "sn": "shona", "yo": "yoruba", "so": "somali", "af": "afrikaans", "oc": "occitan", "ka": "georgian",
  "be": "belarusian", "tg": "tajik", "sd": "sindhi", "gu": "gujarati", "am": "amharic", "yi": "yiddish", "lo": "lao", "uz": "uzbek", "fo": "faroese", "ht": "haitian creole",
  "ps": "pashto", "tk": "turkmen", "nn": "nynorsk", "mt": "maltese", "sa": "sanskrit", "lb": "luxembourgish", "my": "myanmar", "bo": "tibetan", "tl": "tagalog", "mg": "malagasy",
  "as": "assamese", "tt": "tatar", "haw": "hawaiian", "ln": "lingala", "ha": "hausa", "ba": "bashkir", "jw": "javanese", "su": "sundanese",
}

def get_encoding(encoding_name):
  with fetch(f"https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/{encoding_name}.tiktoken").open() as f:
    ranks = {base64.b64decode(token): int(rank) for token, rank in (line.split() for line in f if line)}
  n_vocab = len(ranks)
  specials = [
    "<|endoftext|>",
    "<|startoftranscript|>",
    *[f"<|{lang}|>" for lang in LANGUAGES.keys()],
    "<|translate|>",
    "<|transcribe|>",
    "<|startoflm|>",
    "<|startofprev|>",
    "<|nospeech|>",
    "<|notimestamps|>",
    *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
  ]
  special_tokens = dict(zip(specials, itertools.count(n_vocab)))
  n_vocab += len(specials)
  import tiktoken
  return tiktoken.Encoding(
    name=encoding_name,
    explicit_n_vocab=n_vocab,
    pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    mergeable_ranks=ranks,
    special_tokens=special_tokens)

MODEL_URLS = {
  "tiny.en": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
  "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
  "base.en": "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
  "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
  "small.en": "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
  "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
  "medium.en": "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
  "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
  "large-v1": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt",
  "large-v2": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
  "large": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
}
def init_whisper(model_name="tiny.en", batch_size=1, *, audio_ctx_alignment=1):
  assert MODEL_URLS[model_name] is not None

  filename = fetch(MODEL_URLS[model_name])
  state = torch_load(filename)
  model = Whisper(state['dims'], batch_size, audio_ctx_alignment)
  load_state_dict(model, state['model_state_dict'], strict=False)
  enc = get_encoding("multilingual" if model.is_multilingual else "gpt2")
  return model, enc

def load_file_waveform(filename):
  waveform, _ = librosa.load(filename, sr=RATE)
  return waveform

def transcribe_file(model, enc, filename):
  return transcribe_waveform(model, enc, [load_file_waveform(filename)])

def transcribe_waveform(model: Whisper, enc, waveforms, truncate=False):
  """
  Expects an array of shape (N,S) where N is the number waveforms to transcribe in parallel and S is number of 16000Hz samples
  Returns the transcribed text if a single waveform is provided, or an array of transcriptions if multiple are provided
  """

  log_spec = prep_audio(waveforms, model.batch_size, truncate)
  nsample = model.decoder.max_tokens_to_sample
  nctx = model.decoder.max_self_attn_cache_len

  def inferloop(ctx: Union[np.ndarray, List[np.ndarray]], encoded_audio):
    pos, next_tokens = 0, ctx
    for i in range(nsample):
      logits = model.decoder(Tensor(next_tokens, dtype=dtypes.int32), pos, encoded_audio if pos == 0 else None)
      next_tokens = logits[:, -1].argmax(axis=-1).numpy().astype(np.int32).reshape(-1, 1)
      next_tokens[ctx[:, -1] == eot] = eot
      ctx = np.concatenate((ctx, next_tokens), axis=1)
      pos = ctx.shape[-1] - 1
      if (next_tokens == eot).all() or pos == nctx: break
    return ctx

  def gettexttoks(line): return [tok for tok in line if tok < eot or tok > enc._special_tokens["<|notimestamps|>"]][-nsample+len(start_tokens):]
  start_tokens = [enc._special_tokens["<|startoftranscript|>"]]
  if model.is_multilingual:
    # TODO detect language
    language_token = enc._special_tokens["<|startoftranscript|>"] + 1 + tuple(LANGUAGES.keys()).index("en")
    start_tokens.append(language_token)
    start_tokens.append(enc._special_tokens["<|transcribe|>"])
  start_tokens.append(enc._special_tokens["<|notimestamps|>"])

  eot = enc._special_tokens["<|endoftext|>"]

  ctx = np.tile(start_tokens, (model.batch_size,1))
  transcriptions = [[] for _ in waveforms]

  for curr_frame in range(0, log_spec.shape[-1], FRAMES_PER_SEGMENT):
    segment = log_spec[:, :, curr_frame:curr_frame + FRAMES_PER_SEGMENT]
    if (frames:=mel_prefix_frames(segment)) < segment.shape[-1] == 2*model.encoder.positional_embedding.shape[0]:
      encoded_audio = model.encoder.encode_silence_tail(Tensor(np.ascontiguousarray(segment[..., :frames])))
    else: encoded_audio = model.encoder.encode(Tensor(segment))

    if all(len(c) == len(ctx[0]) for c in ctx): ctx = inferloop(np.array(ctx), encoded_audio)
    else: ctx = [inferloop((np.array([c]*model.batch_size)), encoded_audio)[i] for i,c in enumerate(ctx)]

    for i, (res, arr) in enumerate(zip(transcriptions, ctx)):
      if curr_frame*HOP_LENGTH <= len(waveforms[i]):res.extend(arr[np.where(arr == start_tokens[-1])[0][0]+1:eoti[0] if len (eoti:=np.where(arr == eot)[0]) else None])
    ctx = [[enc._special_tokens['<|startofprev|>']]+gettexttoks(cs)+start_tokens for cs in ctx]

  transcriptions = list(map(lambda tokens: enc.decode(tokens).strip(), transcriptions))
  return transcriptions if len(transcriptions) > 1 else transcriptions[0]

CHUNK = 1600
RECORD_SECONDS = 10

def listener(q):
  import pyaudio
  p = pyaudio.PyAudio()
  stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
  print("listening")
  for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    waveform = ((np.frombuffer(data, np.int16)/32768).astype(np.float32)*3)
    q.put(waveform)
  print("done listening")

if __name__ == "__main__":
  model, enc = init_whisper("small.en" if getenv("SMALL") else "tiny.en", batch_size=1)

  if len(sys.argv) > 1:
    print(transcribe_file(model, enc, sys.argv[1]))
  else:
    # online
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=listener, args=(q,))
    p.daemon = True
    p.start()

    lst = [enc._special_tokens["<|startoftranscript|>"], enc._special_tokens["<|notimestamps|>"]]
    total = None
    did_read = False
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
      while not q.empty() or total is None:
        waveform = q.get()
        if total is None: total = waveform
        else: total = np.concatenate([total, waveform])
        did_read = True
      if did_read:
        log_spec = prep_audio(total.reshape(1, -1), model.batch_size, truncate=True)
        encoded_audio = model.encoder.encode(Tensor(log_spec))
      # pass the previously inferred tokens as 'prefix' - https://github.com/openai/whisper/discussions/117#discussioncomment-3727051
      out = model.decoder(Tensor([lst]), 0, encoded_audio).realize()
      idx = int(out[0,-1].argmax().numpy().item())
      lst.append(idx)
      dec = enc.decode(lst)
      print(dec) # DO NOT REMOVE PRINT. IT'S VERY IMPORTANT
      if dec.endswith("<|endoftext|>"):
        lst.pop()
