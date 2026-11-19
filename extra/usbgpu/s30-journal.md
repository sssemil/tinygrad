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

## November 20, 2026 — MNIST with a native shader compiler

Tinygrad emitted scalar USC instructions for the three-dimensional grid, byte loads, addressing, loop counters, accumulator updates, comparisons, square roots and reciprocals. For the first test image, a scalar workgroup produced the correct class in 2.02 s warm. The same schedule exceeded the command timeout at batch size 16. Grouping adjacent output coordinates in one workgroup brought the batch within the normal timeout. All sixteen predictions matched the MUSA reference on five consecutive runs.

| MNIST batch of 16 | First inference after loading weights | Warm median |
| --- | ---: | ---: |
| USC, empty cache, default scheduler | 3.281 s | 2.456 s |
| MUSA reference | 1.357 s (existing compiler cache) | 14.31 ms |

The first-inference measurements use different compiler-cache conditions. The warm native path is about 172× slower. The current compiler issues scalar loads with a wait after each and limits live temporaries to 16. Aggressively moving address arithmetic outside the inner loops exceeded that register limit in four of ten captured kernels. Neither the vendor compiler nor prebuilt shaders are needed for native MNIST. The vendor compute firmware and onboard management firmware still run.

Reproduce from the repository root with `MT_FIRMWARE` pointing to the S30 boot image and `PYTHONPATH=.`:

```sh
DEV=MT:USC CACHELEVEL=0 MUSA_PATH=/nonexistent python extra/usbgpu/bench_s30.py mnist /path/to/mnist-s30.safetensors
DEV=MT:MUSA python extra/usbgpu/bench_s30.py mnist /path/to/mnist-s30.safetensors
```

## November 20, 2026 — first open compute-control firmware

GNU binutils 2.41, built from source with `--target=metag-elf`, assembled six META instructions for each test invocation. Only two pages were mapped for the processor. With both the MUSA toolkit and vendor compute image unavailable, thread 0 read fresh 32-bit challenges and returned each value plus one, including wraparound from `ffffffff` to zero. This establishes execution of our own firmware on the processor that normally schedules GPU work. USC shader dispatch still uses the vendor firmware.

The master boot used by the vendor image did not execute our full-width code successfully. Starting thread 0 through the documented META control port with full-width instruction decoding worked. Reading the mailbox directly across USB BAR2 returned stale data. Reading and writing it through the META port worked repeatedly. A separate attempt to map an uncached mailbox generated an MMU fault, so that mapping is excluded from the working path. Tests use volatile memory and the board's existing SMC for power-domain reset.

From the tinygrad checkout, with GNU's metag tools on PATH:

```sh
DEV=MT:USC MT_FIRMWARE=/nonexistent MUSA_PATH=/nonexistent PYTHONPATH=. python extra/usbgpu/s30/boot.py
```

For a local binutils build, set `META_AS` to `build/gas/as-new` and `META_OBJCOPY` to `build/binutils/objcopy`. The next firmware acceptance test is a command stream that writes a known value from a USC shader after an open boot. The mailbox result alone exercises META.

Register and boot references are the [Imagination host boot implementation](https://chromium.googlesource.com/chromiumos/third_party/kernel/+/f66be639e83bf9b6fe6e57d5fc84029958022f3c/drivers/gpu/drm/img-rogue/1.17/rgxstartstop.c) and the [upstream Linux META driver](https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/imagination/pvr_fw_meta.c).
