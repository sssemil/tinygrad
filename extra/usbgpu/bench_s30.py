"""Benchmark the trained Chestnut models. Run with DEV=MT:USC or DEV=MT:MUSA and PYTHONPATH=."""
import argparse, itertools, statistics, time
from pathlib import Path
from tinygrad import Device, TinyJit

def mnist_inference(weights:Path):
  from examples.beautiful_mnist import Model
  from tinygrad.nn.datasets import mnist
  from tinygrad.nn.state import load_state_dict, safe_load
  model = Model()
  load_state_dict(model, safe_load(weights))
  _, _, x, _ = mnist(device="CPU")
  batch = x[:16].to("MT").realize()
  classify = TinyJit(lambda x: model(x).argmax(axis=-1).realize())
  return lambda: classify(batch).tolist()

def whisper_inference(audio:Path):
  from examples.whisper import init_whisper, load_file_waveform, transcribe_waveform
  model, enc = init_whisper("tiny.en", audio_ctx_alignment=64)
  waveform = load_file_waveform(str(audio))
  return lambda: transcribe_waveform(model, enc, [waveform], truncate=True)

def qwen_prompt(kv, question:str="What is the capital of France?") -> list[int]:
  from jinja2 import Template
  from tinygrad.llm.tokenizer import SimpleTokenizer
  tok = SimpleTokenizer.from_gguf_kv(kv)
  prompt = Template(tok.chat_template).render(messages=[dict(role="user", content=question)], add_generation_prompt=True)
  return tok.encode(prompt)

def qwen_inference(weights:Path):
  from tinygrad.llm.model import Transformer
  model, kv = Transformer.from_gguf(weights, 128, realize=True)
  ids = qwen_prompt(kv)
  def run():
    model._cached_tokens = []
    return list(itertools.islice(model.generate(ids.copy()), 8))
  return run

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("model", choices=("mnist", "whisper", "qwen"))
  parser.add_argument("input", type=Path, help="MNIST safetensors, Whisper WAV or Qwen GGUF")
  parser.add_argument("--check-reference", action="store_true", help="require the recorded result for the trained S30 benchmark fixtures on every call")
  args = parser.parse_args()
  start_model = time.perf_counter()
  if Device.DEFAULT != "MT" or not Device["MT"].mt.s30: raise RuntimeError("Select the S30 with DEV=MT:USC or DEV=MT:MUSA")
  build = {"mnist":mnist_inference, "whisper":whisper_inference, "qwen":qwen_inference}[args.model]
  run = build(args.input)
  print(f"device and model setup {(time.perf_counter()-start_model):.3f} s", flush=True)
  expected = {"mnist":[7, 2, 1, 0, 4, 1, 4, 9, 5, 9, 0, 6, 9, 0, 1, 5], "whisper":"Could you please let me out of the box?",
              "qwen":[785, 6722, 315, 9625, 374, 12095, 13, 151645]}[args.model]
  times = []
  for i in range(5):
    start = time.perf_counter()
    prediction = run()
    times.append(time.perf_counter()-start)
    if args.check_reference and prediction != expected: raise RuntimeError(f"{args.model} call {i} returned {prediction!r}, expected {expected!r}")
    if i == 0: print(f"time to first output {(time.perf_counter()-start_model):.3f} s", flush=True)
    print(f"{args.model} {Device['MT'].renderer.target.renderer} {i}: {times[-1]*1000:.2f} ms  {prediction}", flush=True)
  print(f"warm median {statistics.median(times[2:])*1000:.2f} ms")
