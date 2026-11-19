# S30 bring-up journal

Hardware used for these measurements is the 4 GiB MTT S30 on Chestnut over USB. The host runs tinygrad. Model checkpoints are inputs. The shader cache starts empty for native acceptance runs.

## Latest end-to-end cold and warm measurements

Measured on the same Chestnut S30 with compiler caching disabled. The cold column starts before device and checkpoint setup and stops when the first result is available. Warm latency is the median of calls 3–5 in the same process. Each Qwen call reprocesses the prompt and returns eight tokens. The input files and output comparisons are described below.

| Workload | Native cold | MUSA cold | Native warm | MUSA warm |
| --- | ---: | ---: | ---: | ---: |
| MNIST, batch of 16 | 0.932 s | 3.506 s | 35.63 ms | 13.82 ms |
| Whisper tiny.en, test.wav | 30.525 s | 50.836 s | 6.300 s | 2.595 s |
| Qwen2.5-0.5B Q8_0 | 159.952 s | 171.460 s | 14.433 s | 9.622 s |

All five native outputs per model matched the reference. The native measurements ran offline in a fresh filesystem namespace (details below). Separate hardware runs disabled vendor compilation, vendor firmware boot and management mailbox construction. The card powered up with its factory firmware still installed. The host did not upload proprietary code during these model runs. The board-management mailbox was unused.

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

## November 20, 2026 — coherent USB mailbox

The earlier uncached mapping failed because the selected virtual address belonged to a different META segment. Mapping the second physical page through segment 1 at 0x30000000 and using its global alias at 0xb0000000 bypassed the META data cache. Setting bit 40 on the output address selected uncached SLC access. Three independent source builds and boots returned all fifteen challenge responses through ordinary USB VRAM reads and writes, including the unsigned wraparound case on each boot. The control port now handles startup only. This provides a shared-memory command channel for further firmware work.

## November 20, 2026 — FP16 and the Whisper causal mask

Native USC now loads and stores half precision, rounds FP32 results to FP16, compares scalars and selects values per lane. For a masked global load, disabled lanes use in-bounds element zero and receive the specified fallback value. Padding checks cover indices outside both ends of a real allocation. The hardware tests also cover FP16 multiplication, a data-dependent choice and an upper-triangular mask containing negative infinity.

Whisper's first cold failure was constructing the decoder mask. That operation now passes. The full Whisper run and Qwen Q8_0 import subsequently stopped at unsupported bitcasts. Both errors occurred before producing a transcript or token, so there is no native warm model latency to report for either. The S30 vendor-stack baselines above still apply.

## November 20, 2026 — symbolic kernel arguments and special functions

Added signed 32-bit scalar arguments to the native launch ABI. One TinyJit program accepted positive and negative bound values across replays. On the S30, the native exponential, logarithm, integer conversions, signed right shift and FP32 bit reinterpretation matched NumPy on small arrays. Whisper now reaches the mel filter calculation and exhausts the compiler's 16 temporary registers. Offline compilation of 92 captured Whisper kernels leaves three failures, two with runtime divisors and one with excess register pressure. That count measures code generation only.

## November 20, 2026 — native scheduling and 32-register shaders

Extended physical-register selection to 32 temporaries. The 80×201 Whisper mel filter now executes directly on the S30, with a maximum absolute difference of 5.22e-8 against tinygrad CPU. A full Whisper encoder attempt timed out. The card's PCIe configuration subsequently stopped responding. Chestnut reported no power fault, and its normal downstream power control restored the card. A subsequent SMC test and native model run passed.

Placing address calculations where tinygrad scheduled them, before the inner reduction, reduced the warm MNIST batch time from 2.456 s to 1.581 s. Unrolling three- and five-tap convolution reductions and groups of four in longer reductions reduced it to 1.060 s. All 16 predictions still matched the reference on every run. This is a 2.32× native speedup and leaves a 74× gap to the 14.31 ms MUSA reference.

## November 20, 2026 — Whisper from an empty shader cache

Whisper tiny.en completed on the S30 with the MUSA path unavailable and the compiler cache disabled. The complete audio-to-transcript call returned “Could you please let me out of the box?” in 67.804 s after checkpoint loading. This run included per-kernel diagnostic logging. It used the vendor compute firmware for dispatch.

Symbolic attention shapes required quotient and remainder by a runtime divisor. The native renderer emits integer shifts, comparisons and subtraction for this case. On the card, one replayable kernel matched NumPy on 4,097 input values and six divisors covering 1 through 448. All 92 captured Whisper graphs compiled offline. The matching transcript is the separate hardware check. Warm timing is recorded separately from this first call.

## November 20, 2026 — narrow integer kernels for quantized weights

The Qwen cold loader reads raw GGUF bytes and combines them into FP32 and FP16 values. Added signed byte and halfword extension, modulo-width arithmetic, unsigned comparisons and integer-to-float conversion, bitwise OR/XOR, and variable shifts. S30 runs matched NumPy for signed and unsigned 8- and 16-bit data and unsigned 32-bit data. Offline compilation of the 63 captured Qwen graphs now succeeds for 61. Two exceed the current 32-register limit. This is compiler coverage, not yet an end-to-end token result.

## November 20, 2026 — Qwen rotary embeddings

Qwen's 623-operation rotary kernel exhausted the 32-register allocator. The scalar encodings now address up to 64 temporaries. Reloading shared constants near use avoids keeping every coefficient live for the whole kernel. The 159×64 rotary embedding test ran on hardware with maximum absolute error 7.60e-6 relative to tinygrad CPU. All 63 captured Qwen graphs now compile offline. Token generation is a separate acceptance check.

## November 20, 2026 — Qwen token parity and firmware GPU register access

Qwen2.5-0.5B-Instruct Q8_0 generated the same eight token IDs as MUSA, including EOS, from a fresh process with the compiler cache disabled and the toolkit unavailable. The first prompt-plus-generation call took 53.276 s after checkpoint loading and included diagnostic logging. This completes native shader compilation for the three test workloads on this particular checkpoint, input and context configuration. Each inference still uses the vendor compute firmware.

In a separate boot, our META program read both physical GPU cores' 64-bit identification registers at offset 0x20 in the 0x02000000 and 0x02100000 META MMIO windows. It returned 0x0023000406600017 for each core through coherent VRAM, matching the independent PCIe reads. This establishes access to GPU registers from open firmware. It does not dispatch a shader.

## November 20, 2026 — source-generated PDS and compute ring completion

The native shader path also needs short PDS programs to load parameters, write workgroup or instance IDs and start USC. tinygrad now describes those programs in terms of branch, halt and DOUT instructions and encodes their operands when it builds a launch. The generated words matched the previous working launch for three different shared-register sizes.

An independent META program assembled from source configured the hardware compute ring and issued a kick with both toolkit and vendor compute firmware paths unavailable. An empty stream raised event bit 2, advanced the first core's read pointer from zero to four, and left the MMU fault register zero. A vendor-firmware control run on the same hardware also advanced the pointer to four. This check contains no USC work. The SMC remains active for power and reset.

The five-call native Whisper run returned the reference transcript each time. Warm median after the first two calls was 65.028 s, compared with 2.642 s under MUSA, using vendor compute firmware for both.

The repeatable firmware test is `PYTHONPATH=. DEV=MT:USC MT_FIRMWARE=/nonexistent MUSA_PATH=/nonexistent python extra/usbgpu/s30/cdm.py`, with the GNU META assembler and objcopy selected as for the mailbox test. The program checks the completion interrupt, firmware sequence acknowledgement, hardware read pointer and MMU fault register. A separate one-lane experiment sent a generated USC store instruction through this firmware and read the value 0x12345678 back from VRAM.

The host can also issue the kick directly through PCIe. With META thread zero stopped, host-issued launches returned the correct results for three different input vectors and all 16 trained MNIST predictions over five consecutive calls. The native warm median was 455.76 ms with the experimental multiply-add emitter. No compute firmware image was uploaded. The cold first inference call took 743.03 ms after weights loaded. The board management controller still supplied power and reset. The previous five Qwen runs using native shaders and vendor dispatch completed with 48.890 s warm median and exactly the reference token IDs each time.

## November 20, 2026 — direct hardware dispatch for MT:USC

Native programs now select the host-controlled compute ring on core 0 as part of normal tinygrad execution. The host programs the PDS and USC heaps, clears the previous completion event, writes the two ring pointers and issues a kick. META is stopped throughout. The MUSA and MTGPU LLVM renderers retain their reference path. The direct ring contains shader dispatch words and its terminator, independent of the vendor firmware submission format.

With `DEV=MT:USC CACHELEVEL=0 PARALLEL=0 MT_FIRMWARE=/nonexistent MUSA_PATH=/nonexistent`, the direct queue produced the same 16 MNIST predictions, Whisper sentence and eight Qwen tokens as the vendor stack. Five Qwen calls, clearing the prompt cache each time, took 25.537, 22.256, 22.245, 22.260 and 22.264 s after loading weights. Its 22.260 s warm median is 2.20× faster than native shaders dispatched by the vendor firmware and 2.30× slower than the 9.694 s MUSA reference. The initial direct Whisper experiment took 25.999, 23.837 and 23.691 s. These comparisons used the experimental multiply-add emitter. The direct hardware suite passed all nine applicable tests with the vendor paths unavailable. Ten tests restricted to the vendor firmware or compiler were skipped.

The board still runs its own SMC RTOS. The driver asks it to power the GPU up, reset five domains and return to idle. The experiments have removed the host-supplied GPU firmware and vendor compiler. They have not replaced the board's management firmware.

## November 20, 2026 — extended shared register indices

Giving each thread two independent outputs exposed a USC instruction encoding limit. Two of 92 Whisper kernels needed shared register 64 or higher. Added the high index bits used by the long integer instruction. All 92 Whisper, 63 Qwen and 10 MNIST trial graphs now compile with this schedule. A 73-threshold clipping kernel on the card agreed with NumPy. Tiling is a separate performance experiment.

The normal direct Whisper entry point returned the reference transcript five times, with a 24.687 s warm median and 48.610 s to first transcript including 21.538 s for device and model setup, using the scalar unfused compiler. An experimental attempt to kick both cores separately timed out without either completion event. The single-core path immediately ran MNIST again. Dual-core scheduling is unresolved.

## November 20, 2026 — direct attach after a Chestnut power cycle

Chestnut switched downstream power off and on. The GPU then returned both core IDs over PCIe before any host management messages. With opening the SMC mailbox and sending IPC deliberately disabled, three vectors of 7, 32 and 4,097 elements matched NumPy and five MNIST calls matched the reference predictions. The unfused MNIST warm median was 525.46 ms. Disabling host management messages did not require programming nonvolatile storage.

The native path now opens the register aperture directly, uses the GPU's register reset sequence and leaves the board's power-state policy alone. Reference runs retain the existing version query, P0/P12 requests and domain resets. A regression test attaches to a mock S30 with no management rings and verifies the physical core ID. This only establishes which software the host needs. The factory controller and VBIOS were present during the power-cycle test, and the experiment does not establish their role in pre-PCIe initialization.

## November 20, 2026 — overlapping independent loads and native multiply-add

The S30 assembler can issue two independent global loads before waiting on both data fences. It keeps the first address live and checks that no intervening instruction reads the pending result. The renderer also contracts single-use 32-bit multiply-add expressions within the same loop block. Half-precision operations retain their intermediate conversion. In an isolated same-board comparison, warm MNIST fell from 524.67 to 454.97 ms with multiply-add alone. The final paired-load build reached 427.02 ms. An experiment issuing independent ALU operations without the usual write-fence wait timed out and was excluded.

All three fixed workloads ran five times with compiler caching disabled, nonexistent toolkit and firmware paths, and exceptions installed on vendor compilation, vendor boot and SMC construction. Outputs matched every time. The warm medians were 427.02 ms for MNIST, 20.229 s for Whisper, and 19.253 s for Qwen. Time to the first result including device setup and loading was 1.087 s, 44.034 s and 164.915 s respectively. The S30 MUSA rerun of MNIST took 3.506 s to first output and 13.82 ms warm. The new native path wins that cold start but still trails the vendor stack warm. Ten applicable direct hardware tests passed with the vendor paths absent. Ten alternating USC programs at the same GPU code address and physical address produced the expected readback from every launch.

## November 20, 2026 — dispatch on both S30 cores

The earlier two-core timeout consumed four bytes on core 0 while core 1's ring register stayed zero. Configuring the second register aperture and setting the primary's active-core routing mask to three let both cores accept the same queue configuration. Under the open META test both hardware read pointers advanced by four for an empty stream and by 68 for a real USC store. Each core raised the CDM completion event, the MMU reported no fault and the value 0x12345678 reached VRAM.

The native host path now programs each core's context-zero translation and compute heaps, kicks both cores, and waits for both hardware events. The production hardware suite passed its ten applicable tests. Fresh-cache MNIST returned the same 16 predictions five times and improved from 427.02 to 219.97 ms warm. The parked management controller received no requests in the strict workload test.

Native kernels can also use all three local coordinates. AR0 supplies X, while AR1 packs Y in the low 16 bits and Z in the high 16. A tinygrad custom kernel running a single 8×4×2 workgroup stored 64 distinct expected coordinate values.


## November 20, 2026 — read shader constants directly in arithmetic

USC integer multiply-add and floating-point arithmetic can take one operand directly from a shared register. This removes the temporary register and move used to copy a constant first. For dot products, placing two outputs with one common operand in the same thread allows that operand to be loaded once. Multi-axis reductions also benefited from wave groups of up to 64. The latest table above records the combined change with both GPU cores active. All 165 captured graphs compiled from source and the direct hardware suite passed 11 applicable tests.

## November 20, 2026 — recover both compute cores after a timeout

An experimental workgroup barrier stalled the secondary core. A fresh process running known-good MNIST then timed out with no MMU fault. Applying the existing register reset sequence to the secondary core as well as the primary restored the card without a management message or Chestnut power cycle. The same MNIST checkpoint then passed five times (171.66 ms warm), and all eleven applicable direct hardware tests passed. Both direct initialization and timeout recovery now reset the second core.

## November 20, 2026 — spatial convolution scheduling

The native scheduler grouped adjacent filter outputs across up to three workgroup axes and assigned four outputs per lane when the shape allowed. Five fresh-cache passes on the trained MNIST model returned the same 16 classes as MUSA. Median warm batch latency fell from 171.51 ms to 35.21 ms. Independent tile trials ranged from 33.45 to 35.92 ms. Cold time to the first batch was 0.772 s. All 165 captured kernels from the three models still compiled from source. The direct S30 hardware suite passed its eleven applicable checks.

A separate Qwen timing attributed 4.22 s and 2.33 s of a 14.59 s generation to two prompt-processing matrix multiplications. This profile was the second generation in its process. It is separate from the five-call benchmark above.

## November 20, 2026 — FP16 inputs to FP32 arithmetic and isolated end-to-end runs

The USC floating-point unit can widen an individual operand from the low 16 bits of a register. The renderer now encodes that operand format on add, multiply and fused multiply-add instructions when a widening conversion feeds them. Accumulation stays in FP32. An explicit FP16 result still rounds at the same point. Mixed-precision projections passed full-output checks against NumPy. All 165 captured kernels compiled from source.

Each model ran in its own fresh, network-isolated Linux mount namespace. We mounted a Git source export with the renderer change, the open-source host interpreter and packages, the input checkpoint or audio, and the cached dataset, tokenizer and model downloads. The previous tinygrad database, firmware download directory, workspace toolkit, system firmware directory, /usr/local and host /tmp were inaccessible. The only physical USB device passed through was the Chestnut. The device lock shared its inode with the host. Compiler caching was disabled. The host had no system-installed Moore Threads compiler or libraries. These restrictions apply to host software. They cannot inspect or substitute for firmware resident on the card.

All fifteen outputs matched MUSA. The medians of the three warm calls were 35.63 ms for MNIST, 6.300 s for Whisper and 14.433 s for Qwen. The previous Qwen median was 17.283 s. A separate hardware run passed all twelve applicable checks and the CPU-side MT suite passed all 25 tests.

## November 20, 2026 — reuse one direct ring on both cores

Direct initialization now sets the ring address and queue-control address once per core. Each launch updates the parameter block, both read positions and the dispatch words with one USB write while holding the device lock. The PDS and USC programs remain private to each kernel. Reusing the ring saved four register writes per launch. For two alternating three-element kernels, five batches of 200 dispatches measured 560 μs per launch warm, down from 869 μs. Output readback checked both results after every batch. All 22 tests passed on the vendor reference, including the native tests, and all twelve applicable direct tests passed with the vendor paths absent.

## November 20, 2026 — native guarded stores

The renderer emits conditional-start and conditional-end instructions around a store. Address and value calculations run before the lane mask changes. A hardware test wrote a mix of active and inactive lanes into a buffer with sentinel elements on both ends. All-inactive, all-active and mixed masks matched the expected whole buffer across three workgroups. This allows padded symbolic matrix dimensions to discard stores from the last partial tile.
