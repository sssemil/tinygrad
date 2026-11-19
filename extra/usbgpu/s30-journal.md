# S30 bring-up journal

Hardware used for these measurements is the 4 GiB MTT S30 on Chestnut over USB. The host runs tinygrad. Model checkpoints are inputs. The shader cache starts empty for native acceptance runs.

## November 20, 2026 — proprietary reference

The existing `MT:MUSA` path uses mcc to build GPU code. The alternate `MT:LLVM` path uses the vendor MTGPU clang. Both use tinygrad's USB transport, MMU, queues and shader dispatch. The S30 boot path uploads the vendor's `mtfw-gen1.bin`. The board management controller also runs firmware.

The following are medians of three consecutive warm calls in one process after two initial calls. Timings include output readback and exclude checkpoint loading. The MNIST checkpoint was trained previously on this card.

| Workload on the S30 | MUSA warm latency | Reference output |
| --- | ---: | --- |
| MNIST, first 16 test images, batch of 16 | 14.31 ms | 7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 |
| Whisper tiny.en, repository test.wav, trimmed audio with context alignment 64 | 2.642 s | Could you please let me out of the box? |
| Qwen2.5-0.5B-Instruct Q8_0, France capital prompt, eight greedy tokens including EOS | 9.694 s | 785 6722 315 9625 374 12095 13 151645 |

The Qwen measurement includes processing the prompt on every call. Clearing the cached token prefix between calls is necessary to measure that workload.
