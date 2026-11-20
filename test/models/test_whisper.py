from copy import copy
import unittest
import pathlib
from examples.whisper import _token_norm, MultiHeadAttention
from examples.whisper import AudioEncoder, TextDecoder, init_whisper, mel_prefix_frames, load_file_waveform, transcribe_file, transcribe_waveform
from examples.audio_helpers import mel
import examples.mlperf.metrics as metrics
from tinygrad.helpers import fetch
from test.helpers import slow
from tinygrad import Tensor, TinyJit, Variable, Device, dtypes, nn
import numpy as np

# Audio generated with the command on MacOS:
# say "Could you please let me out of the box?" --file-format=WAVE  --data-format=LEUI8@16000 -o test
# We use the WAVE type because it's easier to decode in CI test environments
TEST_FILE_1 = str(pathlib.Path(__file__).parent / "whisper/test.wav")
TRANSCRIPTION_1 = "Could you please let me out of the box?"
TEST_FILE_2 = str(pathlib.Path(__file__).parent / "whisper/test2.wav")
TRANSCRIPTION_2 = "a slightly longer audio file so that we can test batch transcriptions of varying length."
# TODO this file will possibly not survive long. find another 1-2 minute sound file online to transcribe
TEST_FILE_3_URL = 'https://homepage.ntu.edu.tw/~karchung/miniconversations/mc45.mp3'
TRANSCRIPTION_3 = """Just lie back and relax.
Is the level of pressure about right?
Yes, it's fine. And I'd like conditioner, please.
Sure. I'm going to start the second lathering now.
Would you like some Q-tips?
How'd you like it cut?
I'd like my bangs and the back trimmed,
and I'd like the rest thinned out a bit and layered.
Where would you like the part?
On the left, right about here.
Here, have a look. What do you think?
It's fine. Here's thousand NT dollars.
It's 30 NT extra for the rinse. Here's your change and receipt.
Thank you, and please come again!
So, how do you like it?
It could have been worse. But you'll notice that I didn't ask her for her card.
Hmm, yeah.
Mm, maybe you can try that place over there next time."""

TRANSCRIPTION_3_ALT = "Just lie back and relax. Is the level of pressure about right? Yes, it's fine. And I'd like conditioner please. Sure. I'm going to start the second lathering now. Would you like some Q-tips? How'd you like it cut? I'd like my bangs on the back trimmed, and I'd like the rest to stand out a bit and layered. Where would you like the part? On the left, right about here. Here. Have a look. What do you think? It's fine. Here's a thousand and eighty dollars. It's thirty and t extra for the rants. Here's your change and receipt. Thank you, and please come again. So how do you like it? It could have been worse, but you'll notice that I didn't ask her for her card. Hmm, yeah. Maybe you can try that place over there next time." #noqa: E501
# NOTE: same as TRANSCRIPTION_3 but with minor changes that should only amount to ~0.079 WER difference (see test_wer_same)
# 'and'     --> 'on'
# 'thinned' --> 'to stand'
# 'nt'      --> 'and eighty'
# '30 nt'   --> 'thirty and t'
# 'rinse'   --> 'rants'
# 'mm'      --> ''

def wer_helper(result: str, reference: str)->float:
  result = metrics.normalize_string(result)
  reference = metrics.normalize_string(reference)
  wer, _, _ = metrics.word_error_rate([result], [reference])
  return wer

class TestWhisperDecoder(unittest.TestCase):
  def test_token_norm(self):
    rng = np.random.default_rng(30)
    layer = nn.LayerNorm(384)
    layer.weight = Tensor(rng.uniform(0, 2, 384).astype(np.float16))
    layer.bias = Tensor(rng.uniform(-1, 1, 384).astype(np.float16))
    for tokens, dtype in ((1, dtypes.float), (2, dtypes.half)):
      values = rng.normal(size=(1, tokens, 384)).astype(np.float32)
      for samples in (values, np.ones_like(values), 32 + 0.25*values):
        with self.subTest(tokens=tokens, dtype=dtype):
          x = Tensor(samples).cast(dtype)
          expected, actual = layer(x), _token_norm(layer, x)
          self.assertEqual(actual.dtype, expected.dtype)
          np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=8e-3 if dtype == dtypes.half else 1e-3, rtol=2e-3)

  def test_token_self_attention(self):
    rng = np.random.default_rng(30)
    ref = MultiHeadAttention(32, 4, "self", max_self_attn_cache_len=32)
    for w in nn.state.get_state_dict(ref).values(): w.replace(Tensor(rng.normal(0, 0.2, w.shape).astype(np.float16)))
    fast = copy(ref)
    for name in ("cache_k", "cache_v"):
      data = rng.normal(size=(2, 32, 32)).astype(np.float32)
      for attn in (ref, fast): setattr(attn, name, Tensor(data.copy()).realize())
    step = TinyJit(lambda x, pos: fast.decode_token(x, pos).realize())
    for index in (1, 2, 17, 31, 1):
      pos = Variable("pos", 1, 31).bind(index)
      x = Tensor(rng.normal(size=(2, 1, 32)).astype(np.float32))
      np.testing.assert_allclose(step(x, pos).numpy(), ref(x, len=pos).numpy(), atol=1e-5, rtol=1e-4)
      for name in ("cache_k", "cache_v"):
        np.testing.assert_allclose(getattr(fast, name).numpy(), getattr(ref, name).numpy(), atol=1e-5, rtol=1e-4)

  def test_aligned_audio_cache(self):
    dims = dict(n_vocab=47, n_text_ctx=12, n_text_state=16, n_text_head=2, n_text_layer=1)
    ref, aligned = TextDecoder(**dims), TextDecoder(**dims, audio_ctx_alignment=8)
    ref.positional_embedding = Tensor.randn(12, 16)
    nn.state.load_state_dict(aligned, nn.state.get_state_dict(ref), verbose=False)
    for _ in range(2):
      audio = Tensor.randn(1, 11, 16)
      for pos, tokens in ((0, [[1, 3]]), (2, [[5]])):
        args = Tensor(tokens), pos, audio if pos == 0 else None
        np.testing.assert_allclose(aligned(*args).numpy(), ref(*args).numpy(), rtol=1e-4, atol=1e-5)

  def test_greedy_matches_last_position(self):
    decoder = TextDecoder(n_vocab=47, n_text_ctx=12, n_text_state=48, n_text_head=3, n_text_layer=1)
    decoder.positional_embedding = Tensor.randn(12, 48)
    audio = Tensor.randn(2, 8, 48)
    for pos, tokens in ((0, [[1, 7], [8, 3]]), (2, [[2], [5]]), (3, [[6], [4]])):
      x = Tensor(tokens)
      encoded = audio if pos == 0 else None
      expected = decoder(x, pos, encoded)[:, -1].argmax(-1).tolist()
      self.assertEqual(decoder(x, pos, encoded, greedy=True).tolist(), expected)

class TestWhisperEncoder(unittest.TestCase):
  def test_silence_tail(self):
    encoder = AudioEncoder(n_mels=4, n_audio_ctx=32, n_audio_state=8, n_audio_head=2, n_audio_layer=1, audio_ctx_alignment=8)
    encoder.positional_embedding = Tensor.randn(32, 8)
    for frames in (16, 16, 16, 32, 64, 16):
      silence = Tensor.randn(1, 4, 1)
      prefix = Tensor.randn(1, 4, frames-6).cat(silence.expand(1, 4, 6), dim=-1)
      full = prefix.pad((0, 64-frames), mode="replicate")
      np.testing.assert_allclose(encoder.encode_silence_tail(prefix.contiguous()).numpy(), encoder(full).numpy(), rtol=1e-4, atol=1e-5)

  def test_mel_prefix(self):
    mel = np.zeros((2, 4, 256), np.float32)
    self.assertEqual(mel_prefix_frames(mel), 64)
    mel[1, 0, 63] = 1
    self.assertEqual(mel_prefix_frames(mel), 128)
    mel[0, 1, -5] = 1
    self.assertEqual(mel_prefix_frames(mel), 256)
    self.assertEqual(mel_prefix_frames(mel[..., :127]), 127)

  def test_aligned_attention(self):
    encoder = AudioEncoder(n_mels=4, n_audio_ctx=16, n_audio_state=8, n_audio_head=2, n_audio_layer=2)
    encoder.positional_embedding = Tensor.randn(16, 8)
    audio = Tensor.randn(2, 4, 20)
    expected = encoder(audio).numpy()
    encoder.audio_ctx_alignment = 8
    np.testing.assert_allclose(encoder(audio).numpy(), expected, rtol=1e-4, atol=1e-5)

# TODO: WEBGPU GPU dispatch dimensions limit
@unittest.skipIf(Device.DEFAULT == "WEBGPU", "WEBGPU GPU dispatch dimensions limit")
class TestWhisper(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    model, enc = init_whisper("tiny.en", batch_size=2)
    cls.model = model
    cls.enc = enc

  @classmethod
  def tearDownClass(cls):
    del cls.model
    del cls.enc

  def assertWER(self, actual: str, expected: str, threshold: float):
    __tracebackhide__ = True  # Hide traceback for py.test
    wer = wer_helper(actual, expected)
    if wer > threshold:
      err = f"WER={wer:.3f} > {threshold}"
      raise AssertionError(
        err
      )

  @slow
  def test_transcribe_file1(self):
    self.assertEqual(transcribe_file(self.model, self.enc, TEST_FILE_1),  TRANSCRIPTION_1)

  @slow
  def test_transcribe_file2(self):
    self.assertEqual(transcribe_file(self.model, self.enc, TEST_FILE_2),  TRANSCRIPTION_2)

  @slow
  def test_transcribe_batch12(self):
    waveforms = [load_file_waveform(TEST_FILE_1), load_file_waveform(TEST_FILE_2)]
    transcriptions = transcribe_waveform(self.model, self.enc, waveforms)
    self.assertEqual(2, len(transcriptions))
    self.assertEqual(TRANSCRIPTION_1,  transcriptions[0])
    self.assertEqual(TRANSCRIPTION_2,  transcriptions[1])

  @slow
  def test_transcribe_batch21(self):
    waveforms = [load_file_waveform(TEST_FILE_2), load_file_waveform(TEST_FILE_1)]
    transcriptions = transcribe_waveform(self.model, self.enc, waveforms)
    self.assertEqual(2, len(transcriptions))
    self.assertEqual(TRANSCRIPTION_2,  transcriptions[0])
    self.assertEqual(TRANSCRIPTION_1,  transcriptions[1])

  @unittest.skip("file 3 url is broken")
  @slow
  def test_transcribe_long(self):
    waveform = [load_file_waveform(fetch(TEST_FILE_3_URL))]
    transcription = transcribe_waveform(self.model, self.enc, waveform)
    self.assertWER(transcription, TRANSCRIPTION_3, 0.085)

  @unittest.skip("file 3 url is broken")
  @slow
  def test_transcribe_long_no_batch(self):
    waveforms = [load_file_waveform(fetch(TEST_FILE_3_URL)), load_file_waveform(TEST_FILE_1)]

    trancriptions = transcribe_waveform(self.model, self.enc, waveforms)
    self.assertEqual(2, len(trancriptions))
    self.assertWER(trancriptions[0], TRANSCRIPTION_3, 0.085)
    self.assertEqual(TRANSCRIPTION_1, trancriptions[1])

  def test_wer_same(self):
    reference = TRANSCRIPTION_3
    self.assertWER(TRANSCRIPTION_3_ALT, reference, 0.079)

  def test_wer_different(self):
    reference = TRANSCRIPTION_3
    self.assertWER("[no speech]", reference, 1.0)

  def test_wer_different_2(self):
    reference = TRANSCRIPTION_3
    self.assertWER("", reference, 1.0)

  def test_wer_different_3(self):
    reference = TRANSCRIPTION_3
    self.assertWER(reference[:len(reference)//2], reference, 0.524)

  def test_mel_filters(self):
    # reference = librosa.filters.mel(sr=16000, n_fft=16, n_mels=16)
    reference = Tensor([[-0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0021111054811626673, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.003133024089038372, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0017568661132827401, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0009823603322729468, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0007768510840833187, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0010490329004824162, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0011341988574713469, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.000231665835599415, 0.0006950111710466444, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.00040073052514344454, 0.0005822855746373534, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00033081238507293165, 0.0006097797304391861, 0.0]])
    np.testing.assert_allclose(mel(sr=16000, n_fft=16, n_mels=16, dtype=dtypes.float32).numpy(), reference.numpy(), atol=1e-6)

if __name__ == '__main__':
  unittest.main()
