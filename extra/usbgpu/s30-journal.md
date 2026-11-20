# S30 bring-up journal

Hardware used for these measurements is the 4 GiB MTT S30 on Chestnut over USB. The host runs tinygrad. Model checkpoints are inputs. The shader cache starts empty for native acceptance runs.

## Latest end-to-end cold and warm measurements

Measured on the same Chestnut S30 with compiler caching disabled. The cold column starts before device and checkpoint setup and stops when the first result is available. Warm latency is the median of calls 3–5 in the same process. Each Qwen call reprocesses the prompt and returns eight tokens. The input files and output comparisons are described below.

| Workload | Native cold | MUSA cold | Native warm | MUSA warm |
| --- | ---: | ---: | ---: | ---: |
| MNIST, batch of 16 | 0.850 s | 3.745 s | 27.44 ms | 14.02 ms |
| Whisper tiny.en, test.wav | 10.788 s | 51.649 s | 4.858 s | 2.282 s |
| Qwen2.5-0.5B Q8_0 | 16.869 s | 172.335 s | 5.031 s | 4.815 s |

The benchmark checked all five outputs against the recorded reference for each model on each backend. The native column uses the offline, isolated benchmark. It checked every loaded USC program against compiler output generated in the same process. The check covered 10 distinct binaries for MNIST, 92 for Whisper and 77 for Qwen. The benchmark also checked that compute startup received only zero-filled context memory, with META master boot disabled. It blocked vendor firmware boot and management mailbox construction. The card powered up with its factory firmware still installed. The host did not upload proprietary code during these model runs. The board-management mailbox was unused.

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

## November 20, 2026 — tile both matrix dimensions

The native scheduler now groups independent output rows and columns into small register tiles. A symbolic row dimension is rounded up to an even number and guarded stores protect the last row. A NumPy comparison checked every output of both Qwen feed-forward projections for 1, 4, 17 and 32 input rows. With 32 rows, two-by-two tiles reduced one projection from 87 ms to 50 ms and the other from 96 ms to 48 ms. One-row inputs to the symbolic kernel were slower. Qwen's normal one-token decode uses a separate matrix-vector graph.

With the resident ring and the new scheduler, all fifteen model outputs matched MUSA. Warm medians were 32.52 ms for MNIST, 6.517 s for Whisper and 12.135 s for Qwen. Whisper regressed from 6.300 s, while MNIST and Qwen improved. The native and vendor Whisper runs both used the example's normal 30-second encoder context for the short fixture with context alignment 64. The model's silence handling preserves that context. The two slowest Qwen prompt projections and Whisper's encoder projections remain candidates for better memory access scheduling.

## November 20, 2026 — reproducible offline acceptance

Each invocation of the isolated benchmark starts a new offline process with its own home, temporary directory and Python bytecode cache path. The script mounts the selected input and allowlisted dataset or model files. Each of the 15 outputs matched the vendor reference. The updated measurements at the top use this command. MUSA ran on the same hardware with its compiler cache disabled and the same inputs and warm-up policy.

The S30 reported VBIOS 10520 and management RTOS 10206 in the earlier board-identification test. The published driver headers describe management RPCs after PCIe attach. They do not specify a volatile boot path for replacing the onboard controller before enumeration. The open META diagnostic controls a different processor. No onboard flash was modified. [Moore Threads management software and SUDI firmware release matrix](https://docs.mthreads.com/gmc/gmc-doc-online/gmi/releasenotes/gmi_releasenote/)

## November 20, 2026 — workgroup memory and barriers

A PDS workgroup launch wrote two distinctive constants to local memory. Destination field 1 placed them at LM4 and LM5. Field 4 placed them at LM16 and LM17. The earlier barrier timeout had initialized the wrong address. Clearing LM4 before starting the workgroup lets the USC atomic barrier count all participating tasks and reset for the next barrier.

Tinygrad now lowers local buffer loads, stores and barriers for the native renderer. An exchange test used two local arrays and two barriers to permute random inputs across lanes. It compared every element in 33 independent workgroups, including sizes 32, 64, 128 and 256. The largest case addressed 2,048 local dwords. Narrow elements occupy one dword each, so the scheduler advertises 2 KiB for every supported dtype. Fourteen applicable direct S30 hardware tests passed, including these exchanges. The generic tinygrad local-storage test also passed.

The unmodified beautiful_mnist training example executed three batch-size-two Adam updates with the native renderer, disabled compiler cache and nonexistent vendor paths. The final reported loss was 1.74. A CPU run captured 118 training kernels that also compiled with the native renderer. Full training convergence was not measured.

A separate clock diagnostic requested P0 via the management controller solely for comparison. The same native 128×1,536×384 projection took 14.76 ms beforehand and 14.83 ms in P0. The driver restored the board's idle policy. Native model runs continue to use direct attach.

## November 20, 2026 — source-built device diagnostics

The low-level store test now assembles its USC program and builds its PDS launch code on each run. Both store values reached the expected offsets, including a dispatch near the end of the ring. All 24 hardware tests passed on the MUSA reference. The optional META boot and CDM diagnostics also ran with management mailbox construction disabled. Fresh assembly returned all five challenge responses, including wraparound, and the open CDM program consumed four bytes. Model inference reaches the compute queues directly and does not invoke these diagnostics.

## November 20, 2026 — native S30 default

Plain `DEV=MT` on the S30 now selects USC and opens the card without management IPC. With calls to both vendor compilers, the vendor compute boot path and the management mailbox set to raise, the trained MNIST checkpoint returned all 16 reference classes in five runs. Median warm latency was 32.36 ms. An offline selection test also checked the S80's MUSA default and an explicit S30 MUSA request. The normal native MNIST training launcher completed three batch-size-two Adam updates in a separate process.

## November 20, 2026 — reproduce the installed Chestnut bridge firmware

The bridge identified itself as `custom ed4e39b7-CLEAN`. A clean checkout of the public bridge firmware at `ed4e39b7e0794e19ba193477067c48757a5cf9ef` built with Ubuntu's SDCC `4.2.0+dfsg-1` produced exactly the 9,207 bytes installed at offset 0x100 in this bridge's SPI flash. The wrapped image SHA-256 was `9520fde0bf43d499c07abd0a09b74e94d8a7cc3d610f577b7a4e218ab8a378e9`. Builds from the same source with SDCC 4.4, 4.5 and 4.6 differed. The normal SDCC link used seven objects from its packaged open-source 8051 runtime. The source-only build below replaces them all. [Bridge source and build rules at the verified commit](https://github.com/tinygrad/asm2464pd-firmware/tree/ed4e39b7e0794e19ba193477067c48757a5cf9ef/handmade)

From the tinygrad checkout, use SDCC 4.2.0 with its runtime sources installed. The build script extracts the pinned bridge revision from the source repository and compiles each target object. The comparison uses the bridge's existing SPI setup and issues JEDEC identification and flash reads. It acquires tinygrad's normal S30 device lock. The comparison covers the executable image and its length, checksum and CRC wrapper.

```sh
extra/usbgpu/s30/build_chestnut.sh "$BRIDGE_SRC" /usr/share/sdcc/lib/src /tmp/chestnut-from-source.bin
DEV=MT:USC CACHELEVEL=0 PYTHONPATH=".:$BRIDGE_SRC/handmade" python - /tmp/chestnut-from-source.bin <<'PY'
import hashlib, sys
from pathlib import Path
from tinygrad import Device
from e4_flash import flash_jedec_id, flash_read
built = Path(sys.argv[1]).read_bytes()
handle = Device["MT"].mt.pci_dev.usb.usb.handle
jedec = flash_jedec_id(handle)
assert jedec == bytes.fromhex("c2 28 15"), f"unexpected SPI flash {jedec.hex()}"
flash = flash_read(handle, 0, 1 << jedec[2])
end = 0x100 + len(built)
assert flash[0x100:end] == built, "bridge image differs from source build"
assert flash[end:].count(0xff) == len(flash) - end, "found additional data after the application"
print(len(built), hashlib.sha256(built).hexdigest())
PY
```

This verifies the installed USB bridge application. The downstream S30's expansion ROM BAR read 0x2 with its enable bit clear after attach. That register reports the host mapping state. Establishing what the GPU board controller needs for power, clocks and DDR before PCIe enumeration still requires independent evidence.

## November 20, 2026 — one workgroup reduction per matrix-vector projection

On eligible native matrix-vector products an 8×8 workgroup reads eight adjacent reduction elements for each of eight rows. The lanes combine partial sums in local memory, avoiding the global temporary and second dispatch used by the earlier S30 model path. For a 4,864×896 projection this reduced 2.705 ms to 1.712 ms. All 165 captured model graphs compiled from source. Five isolated Qwen runs produced the same eight greedy tokens as MUSA. Warm latency fell from 12.147 s to 10.622 s. MUSA took 9.622 s on the same prompt and tokens. The table at the top includes the corresponding isolated Whisper and MNIST reruns. Rebuilding all three workloads in separate empty namespaces preserved all 15 outputs.

A Whisper experiment materialized attention probabilities before the value product. It took 5.999 s warm compared with 6.434 s for the normal graph. An 8×8 workgroup layout for static matrix products alone took 6.444 s. Neither experiment changed the shipped attention graph.

For the dominant Whisper attention product, the existing 4×2 per-thread output tile took 469 ms. A 2×4 tile took 821 ms with equal outputs, and larger tiles frequently exhausted the 64 temporary registers. Another native prototype fetched two adjacent FP32 values in a single eight-byte USC transfer. It returned exact random bytes at six offsets, including a burst across a 4 KiB boundary. Warm Whisper took 6.438 s versus 6.434 s in the existing implementation. The production renderer keeps its scalar load and register allocation path.

## November 20, 2026 — identify the downstream PCI option ROM

A diagnostic acquired the S30 lock and temporarily mapped the downstream GPU's PCI expansion ROM into an unused portion of the Chestnut bus address space. The ROM aperture measured 2 MiB. The PCI image chain at offsets 0, 0xe400 and 0x1d000 contains three compressed EFI boot-service drivers. Their machine fields identify ARM64, LoongArch64 and x86-64. The images measure 58,368, 60,416 and 63,488 bytes respectively. The diagnostic restored expansion ROM BAR 0x2 and PCI command register 0x7 in a finally block. A fresh native shader returned the expected vector [10, 11] after restoration. The test read the mapped aperture without programming flash. [Linux PCI ROM mapping and image headers](https://github.com/torvalds/linux/blob/master/drivers/pci/rom.c) · [EDK II machine and subsystem identifiers](https://github.com/tianocore/edk2/blob/master/MdePkg/Include/IndustryStandard/PeImage.h)

These images target the host CPU's UEFI environment. The native Chestnut driver attaches with expansion ROM decoding disabled. This narrows the pre-PCIe investigation to the S30's own boot and management processors. The ROM inventory does not establish their reset vector or DDR initialization sequence.

## November 20, 2026 — fixed-size Qwen prompt chunks

Full prompt chunks on the S30 now have their actual length at compile time. This lets the renderer tile the prompt matrix dimensions. A separate JIT handles the variable final chunk. Changing chunk size recreates the appropriate captures. Sixteen LLM server tests passed, including a new test that switches between four- and eight-token chunks with a partial tail and compares generated tokens.

On the physical S30, five native and five MUSA runs returned the same eight tokens. The native warm median improved from 10.622 s to 8.471 s. Applying the same optimization to MUSA improved its warm median from 9.622 s to 4.811 s. The updated proprietary reference took 143.941 s for setup and 172.244 s to first output. Each timed call starts with the same prompt and clears the prefix cache.

## November 20, 2026 — upload weights through Chestnut SRAM

A GPU load from Sudi physical address bit 39 can reach Chestnut's PCIe SRAM. The native allocator reserves that SRAM, maps its upper 256 KiB read-only into the GPU and submits a tinygrad copy kernel. Each group of 32 lanes copies 128 bytes. The bridge's normal command traffic uses the lower half. Short writes and final partial groups use the original PCIe path. The copy kernel itself is generated from UOps at first use. This path is enabled on the tested custom bridge firmware.

In four alternating-payload trials per method on one card, uploading 2 MiB took a median 548.0 ms with the existing direct BAR write and 9.39 ms with the staged copy, from 3.83 to 223.32 MB/s. Every byte matched an independent BAR readback. The first staged transfer took 73 ms including kernel setup. Hardware regression cases covered guard bytes, unaligned nested views, the 16 KiB threshold and writes spanning two SRAM windows plus a tail. An initial experiment with a reverse copy timed out waiting for USB bulk IN. Ordinary VRAM reads still use the BAR.

The isolated native Whisper setup fell from 22 s to 2.94 s. Qwen setup fell from 142 s to about seven seconds. The normal S30 hardware suite passed all 25 tests with vendor tools available and all 15 applicable tests with vendor paths absent.

## November 20, 2026 — reuse convolution inputs across output channels

The native scheduler tiles a convolution across two output channels and up to four adjacent spatial positions when the input and filter have distinct output axes. The isolated MNIST batch improved from 32.48 ms to 29.12 ms warm on the final transport run. All 16 labels matched on all five calls. Wider tiles exhausted the 64 temporary registers on some kernels. All 165 captured inference graphs and 118 captured training graphs compiled from source with the new schedule. The physical card also completed three Adam steps with a batch of two.

## November 20, 2026 — check the origin of every loaded native shader

An empty shader cache excludes a previous run's compiler output. To catch a precompiled shader embedded elsewhere, the offline runner now records the bytes emitted by the native USC renderer. Before any S30 program loads, it checks its target and looks up its entire binary in that set. This includes the copy kernel used to upload the checkpoint. The runner fails if it loads zero native programs, enters vendor compute firmware boot, or constructs the management IPC interface. A deliberately untracked program was rejected before accessing the device.

The final isolated runs verified 10, 88 and 68 distinct loaded binaries respectively for MNIST, Whisper and Qwen. These are counts of unique binary contents per process. All fifteen outputs matched the corresponding proprietary run. The table at the top uses the instrumented native measurements. Qwen includes the full-chunk optimization on both paths. The board's own pre-PCIe startup remains outside this check. The driver exposes the Sudi management IPC rings in a 4 KiB shared SRAM window starting at BAR0 offset 0x6f000. That window supplies the post-boot IPC interface. A pre-PCIe controller bootstrap and a recovery path for this board are still unverified.

## November 20, 2026 — reverse transfer proof and recovery

A diagnostic generated both a VRAM-to-SRAM copy and a completion store from tinygrad UOps. The bridge released USB bulk IN after the GPU wrote an NVMe completion entry with its phase bit set. The lower SRAM half produced byte-exact readbacks at 512 B, 16 KiB, 256 KiB, 256 KiB + 128 B and 2 MiB, with two different random payloads at each size. The 2 MiB case measured 94 MB/s including diagnostic synchronization. This was a prototype. The driver uses the lower half for normal bridge traffic and still reads model outputs through its existing BAR path.

The equivalent test starting at SRAM slot 16 timed out. Cycling downstream PCIe power restored register access, but the bridge's bulk OUT channel still timed out. Resetting the USB bridge using libusb restored normal transfers. A fresh isolated MNIST run generated all ten shader binaries and returned all 16 expected digits on all five calls. The production uploader uses slot 16 in the tested outbound direction. [Chestnut application source and its PCIe-initiated completion protocol](https://github.com/tinygrad/asm2464pd-firmware/tree/ed4e39b7)

## November 20, 2026 — first-stage bootloader identified in the board's inventory

In a separate diagnostic, the S30 management controller reported seven partition names. The device returned RTOS 10206, VBIOS 10520, VAR 0, MGBL 20713, ENTRY 10000, ERRLOG 0 and MT-BIOS 10315. The driver's `struct mtgpu_fw_versions` in `inc/mtgpu/mtgpu.h` labels MGBL as the first-stage bootloader. This gives the pre-PCIe investigation a specific component to locate. The version report establishes installation. It does not trace execution at reset or identify the code that initializes DDR.

A read-only capability walk found standard capabilities 01, 05 and 10, and extended capabilities 01, 03, 04, 15, 18, 16, 19, 02, 1e, 25, 27 and 26 (hexadecimal). The endpoint did not advertise PCI Vital Product Data, a vendor-specific capability or PCI Data Object Exchange. A bootstrap method exposed outside standard PCI configuration space remains possible. The diagnostic issued the normal version query only after PCIe enumeration. The isolated model benchmark continues to prohibit management IPC.

## November 20, 2026 — fewer USC instructions per memory access

The native renderer now multiplies an element index by its byte width with one integer instruction, using a shared literal for the element size. Byte loads and stores use the index directly. The earlier implementation doubled the index twice for a four-byte element. Replacing it removed six lines from the renderer. All 165 saved inference graphs and 118 training graphs compiled. The physical suite passed all 15 applicable tests with the vendor paths absent. On the same inputs the change alone took warm MNIST from 28.51 to about 27 ms, Whisper from 6.210 to 5.717 s and Qwen from 8.466 to 8.129 s. Each output matched the reference on all five calls.

## November 20, 2026 — contiguous output channels for native Qwen prompt projection

For a 32×896 input times 4,864 projection rows, storing the weights with adjacent output channels contiguous reduced the warm product from 35.32 to 23.33 ms. The 32×4,864 input times 896 rows improved from 35.54 to 23.02 ms. Fourteen layouts per shape were tested with random FP16 weights and FP32 activations. Every output element matched NumPy. Splitting the format into eight-row tiles slowed the first projection to 172 ms. For single-token decoding the full transpose changed those two products from 1.69 to 2.08 ms and 2.04 to 1.98 ms respectively.

The native GGUF loader now materializes eligible linear block weights once in this layout. The logical tensor shape and values are preserved. The output vocabulary projection and tied token embedding keep their original storage. Five isolated full Qwen calls verified 75 unique source-generated shaders and the same eight output IDs as MUSA each time. Warm latency fell from 8.129 s with the address optimization alone to 6.801 s. Model setup took 8.006 s and time to first output was 18.514 s. The table at the top uses the isolated timings for the current branch.

## November 20, 2026 — build every bridge target object from source

SDCC's link map identified five C runtime helpers for arithmetic, generic pointer stores and startup, plus two assembly startup objects. A fresh offline build compiled the five C files, assembled the two assembly files and linked them explicitly with the bridge's C and assembly objects using `--nostdlib`. The build directory started empty and the isolated toolchain contained only the host executables and headers. The eight-character source revision and clean flag reproduce the version string embedded by the upstream Makefile.

All 9,207 bytes, including the wrapper, matched a new read from this Chestnut's SPI flash at offset 0x100. A subsequent native GPU calculation returned [12, 14]. The build script above reproduces this check's input without a packaged target library or prebuilt firmware image. The comparison covers the bridge application. Startup inside the bridge silicon and the S30 board's first-stage firmware remain outside that byte comparison.

## November 20, 2026 — narrow the search for an independent S30 bootstrap

Two read-only observations rule out specific leads. The 256 KiB range at offset 0x40000 in the 2 MiB expansion ROM aperture exactly repeated the first 256 KiB, including the three host EFI images. The next 1 MiB was zero. At 0x180000 there were 32 pages with an eight-byte header beginning with `LGDB`; the remaining bytes in these pages were fill. The last 384 KiB was zero. This aperture exposed neither a distinguishable management boot image nor a reset trace. The diagnostic restored both PCI configuration registers and a native shader returned [10, 11].

In the S30 3.1.0 driver, `mtgpu_smc_fw_update` queries the PCI identity and returns `-EINVAL` without transferring firmware. The driver's PCI configuration window at BAR0 + 0x800000 matched the downstream PCI configuration registers at all 11 offsets compared, including vendor/device ID, BARs and capability headers. The word at +0x900 was the L1 substates capability header on both paths. Neither observation supplies a way to start an open management image before enumeration. No board flash writes were issued.

Rechecking the transposed Qwen block projections on MUSA produced the same eight tokens on all five calls, but took 5.569 s warm. The original MUSA weight layout measured 4.811 s and remains the comparison. A separate native matrix-vector sweep checked every element for 22 workgroup and unroll combinations at each of three shapes. The fastest choice depended on the shape. The shipped single-token schedule is unchanged.

## November 20, 2026 — wider workgroups for tiled matrix products

The native scheduler now allows tiled matrix products to use up to 128 work items along an output axis. This keeps the existing per-thread tile and reduction order. For a 32×896 input times 4,864 packed projection rows, a 128-wide limit took 15.46 ms, compared with 23.73 ms at 64 and 21.10 ms at 256. All matrix elements matched NumPy. A full Qwen run took 6.001 s warm at 128 and 6.592 s at 256, compared with 6.801 s previously. Whisper took 5.172 s at 128, compared with 5.688 s previously. The filter schedule still distributes at most 64 lanes along each convolution axis. All 165 captured inference graphs and 118 training graphs compiled. All 15 applicable hardware tests passed without vendor tools.

## November 20, 2026 — verify the direct GPU boot input

The isolated runner now checks the bytes passed into the common GPU boot routine. It rejects nonzero image data, enabling META master boot and requesting a management domain reset. Each of those three negative cases failed before accessing hardware. All three end-to-end model runs passed that check and the shader provenance check on this S30. All fifteen predictions matched the references. The table above reports cold latency and the median of calls three through five with both checks enabled.

A separate read-only management query identified this card's Sudi board type as 10. Its 104-byte capability record had the expected 0xee marker, 4 GiB DDR capacity and two enabled memory channels. The capacity and PCI IDs agreed with the independently read BAR size and PCI configuration. These observations constrain a future open board initializer. The record describes the running configuration and supplies no DDR training sequence or code for power-on.

## November 20, 2026 — isolated model runs after downstream power cycling

Each model also ran immediately after its own Chestnut PCIe off/on cycle. The bridge reported LTSSM 0x00 while off and 0x78 after retraining each time. The isolated runner checked all target shaders and the zero-filled compute boot input. All fifteen outputs still matched the references. Cold and warm times respectively were 0.853 s and 27.36 ms for MNIST, 11.018 s and 5.186 s for Whisper, and 17.736 s and 6.019 s for Qwen. These times start after link retraining. The normal table at the top uses the separate process-cold measurements.

Three paired controls compared a fresh random 4 KiB page in allocated GPU memory at physical offset 0x80000000. The diagnostic checked all bytes against a direct BAR read before releasing the device. Simply reopening the driver preserved the page in all three controls. Cycling downstream power changed it in all three trials. The diagnostic read the page on reopening, before any shader dispatch. This establishes that the sampled VRAM contents did not survive the cycle. The independent pre-PCIe boot sequence still needs a trace. [Bridge power-control implementation](https://github.com/tinygrad/asm2464pd-firmware/blob/ed4e39b7e0794e19ba193477067c48757a5cf9ef/handmade/src/main.c)

The bridge application proof covers the replaceable program in external SPI flash. ASMedia documents the external SPI interface for custom RAM code and depicts a separate 8051 program ROM inside the chip. The ROM's behavior at reset is unverified here. [ASMedia ASM2464PD product description](https://www.asmedia.com.tw/product/802zX91Yw3tsFgm4/C64ZX59yu4sY1GW5.html) · [ASMedia datasheet block diagram](https://egpu.io/wp-content/uploads/wpforo/attachments/160145/15125-ASM2464PDData-SheetR02-1.pdf)

## November 20, 2026 — separate downstream power from link training

Chestnut's source-built bridge application exposes PCIe power control and bridge register access. After the normal downstream off command, enabling the rail at C656 bit 5 while keeping the lane disabled at C659 held the link in LTSSM 0 or 1. Changing the bridge PERST control at B480 from 1 to 0 still left the link untrained. The normal downstream on command then trained it to 0x78. Three passes measured median bridge input currents of 9.5–12 mA off, 195–211 mA with rail on, 196–208 mA after releasing the bridge PERST control and 727–759 mA with a trained link. Each median used 16 samples. This provides a measurement window for an external boot trace. Current alone cannot identify what the board ran.

Three more trials per condition marked both management IPC rings with version requests and seeded a 4 KiB VRAM page with random data. A fresh process inspected the cursors and page before issuing a management command. Reopening preserved both. Toggling the bridge PERST control for half a second with power maintained also preserved both, and LTSSM still read 0x78. A downstream power cycle changed the page and returned both ring cursor pairs to (0,0) in every trial. This distinguishes the two controls on this fixture without assuming that changing B480 electrically resets the S30. The factory board boot path remains present.

## November 20, 2026 — inventory the entire external bridge flash

The Chestnut SPI device returned JEDEC ID c2 28 15. A read-only sweep of its entire 2 MiB found the 256-byte configuration region at 0, the source-built 9,207-byte wrapped application at 0x100, and exclusively 0xff bytes from 0x24f7 through the end. The application matched the independently rebuilt file byte for byte. A GPU shader returned [14, 15] after the read. There is no second image stored elsewhere on this external bridge flash.

The configuration contains enclosure identification strings and board parameters. Comparing with the two 128-byte configuration records in the pinned upstream flasher found differences at offsets 0, 1, 2, 128, 129 and 130. The second triple matched the live JEDEC ID and the first was 00 28 15. The comparison treated the records as data without executing the flasher. These reads cover the bridge's external storage. The on-chip bridge ROM and the downstream S30 boot path require separate evidence. [Bridge configuration records](https://github.com/tinygrad/asm2464pd-firmware/blob/ed4e39b7e0794e19ba193477067c48757a5cf9ef/flash.py)

## November 20, 2026 — compile a shader that exceeded the register file

A fused Qwen projection produced a 483-instruction kernel with 78 simultaneously live values for the S30's 64 registers. Postponing input loads and their integer address calculations until their first use reduced the allocation to 42 registers. The new pass retries only when allocation exhausts the register file. It keeps loads on the original side of stores, branches, mutable assignments and barriers. Applying delayed loads to every shader slowed the normal Qwen run to 6.211 s and Whisper to 5.501 s.

A regression shader on the physical S30 loaded 80 floats before their first sum, compiled them in at most eight registers, and returned their exact sum. The experimental fused model that previously failed compilation ran and produced the reference eight tokens on all five calls. Its weight packing stays outside the model loader.

## November 20, 2026 — combine two independent matrix-vector reductions

Qwen's gate and up projections each reduced a row of 896 elements. Profiling measured 168 launches of the combined kernel at 1.188 s per eight-token response. The compiler now gives two independent row-wise dot products with identical reduction bounds the same loop index before splitting it across the workgroup. A dedicated device test compared every output of two 256×192 projections against NumPy. The cold-cache Qwen run fell from 17.814 to 17.327 s and its warm median from 6.012 to 5.587 s. All fifteen model outputs matched, 17 applicable hardware tests passed, and the native compiler accepted all 165 saved inference graphs and 118 MNIST training graphs. The MUSA reference uses the previous measurement on this card.

## November 20, 2026 — reduce repeated work in native projections

Materializing the input to native S30 linear layers once lets sibling projections reuse the activation and keeps normalization out of the inner dot product. Warm Qwen took 5.232 s with that change alone, down from 5.587 s. A separate experiment also transposed the vocabulary projection and increased warm latency to 5.294 s even with the wider paired schedule.

For the captured paired projection, distributing a group over 32 rows and four reduction lanes took 3.372 ms, compared with 4.579 ms for eight rows and eight reduction lanes. All 4,864 outputs matched a NumPy reference at both widths. The 32-row schedule applies to paired FP32 sums when the row count is divisible by 32. The resulting isolated Qwen benchmark measured 5.131 and 5.146 s warm in two fresh processes. Each produced the same eight tokens on all five calls.

## November 20, 2026 — check the vendor's Sudi controller startup hook

The generic 3.1.0 driver exposes a FEC startup hook taking a program counter. The Sudi-specific operation table points to `sudi_startup`. Its entire implementation returns `-EOPNOTSUPP`; the two neighboring clock-gating and force-stop hooks do the same. The relocated operand reads entry 173 of the driver's 64-bit `os_value` table, and both the matching debug enumeration and kernel wrapper identify that entry as EOPNOTSUPP. None of these three implementations writes a device register. The generic FEC API therefore supplies no S30 bootstrap path in this driver. This check was offline and concerned the FEC API, separate from the board's MGBL inventory.

## November 20, 2026 — check every benchmark prediction on both paths

Both benchmark commands accept `--check-reference` for the three recorded fixtures. Each rejects a different prediction on the first mismatched call. The isolated native runs verified 10, 88 and 77 distinct source-generated programs and all 15 predictions. Fresh processes using MUSA with compiler caching disabled independently passed the same 15 checks on the same S30. The table at the top reports this pair of runs.

As a negative control, the isolated MNIST runner received a copy of the checkpoint with every tensor set to zero. It returned sixteen zeros and exited with a mismatch on call zero. Repeating the run with the trained weights passed all five calls. For example, from the checkout with the model virtualenv active:

```sh
extra/usbgpu/s30/bench_isolated.sh mnist ../.s30-tools/mnist-s30.safetensors --check-reference
```

## November 20, 2026 — reproduce the pre-link capture window

The staged trace command takes the same exclusive lock as the S30 runtime, identifies the S30 and controls downstream slot power. It provides distinct edges for an external board probe. Run from the checkout with the card idle:

```sh
PYTHONPATH=. python extra/usbgpu/s30/trace_power.py --hold 0.5
```

This run read 13 mA with slot power off, 203 mA with the rail enabled, 192 mA with the bridge PERST control released and 729 mA after normal link training. LTSSM stayed at 0 until training completed at 0x78. The next isolated process built all 10 MNIST shaders from source and classified all 16 digits correctly on five calls. In simulated interrupted and power-fault runs, the command restored normal power after an interruption and turned it off following a reported power fault. Capturing the downstream board's own boot transactions still requires a probe or an independently accessible trace.

## November 20, 2026 — regenerate hardware IDs under register pressure

Joining Q, K and V as an ordinary tensor expression produced a 702-instruction shader for which delaying loads alone still exhausted the register file. The compiler can now retry by reading group and lane IDs directly from hardware again at each use. This brought that kernel within 64 registers. A regression case keeps both IDs alive across 63 independent input values and now assembles. On the physical card, the previously failing graph generated the exact reference tokens on all five calls with vendor code unavailable. The experimental QKV expression took about 5.13 s warm and adds no demonstrated model-level gain over the normal projections. The model continues to use its existing Q, K and V parameters.

## November 20, 2026 — reach the GPU during Sudi board startup

The bridge's power-on command waits for three stable link samples. Replaying its downstream lane-enable sequence without that final wait exposed the endpoint to the host much earlier. In three cold cycles with immediate training, PCI ID 1ed5:0102 was readable at 51 ms after rail enable and the compute core ID at 52 ms. Both management ring headers first acquired their normal capacities and pointers at 732–738 ms. In three other cycles, holding the lane disabled for 100 ms moved those observations to 147–148 ms for the PCI ID and 825–836 ms for the initialized rings. The host issued no management requests.

On independent cold cycles, a single PCIe dword read of an allocated VRAM page at S30 physical offset 0x80000000 failed to complete after a write at 100, 150, 230, 340 and 451 ms. A 64-byte bulk read at 64 ms also failed to return the pattern and took over three seconds. Reopening normally restored correct transfers. A fresh dword round trip at 561 ms succeeded in four trials. A source-compiled vector kernel then returned all 65 random integer results by 676–679 ms in each, while both mailbox headers still lacked their operating values. These observations locate a working DDR-backed compute path before the management mailbox is ready. They do not establish which board code trained the memory.

## November 20, 2026 — execute from Chestnut SRAM before card memory is available

The S30 can fetch USC, PDS, command and page-table data across PCIe from Chestnut's SRAM. Its MMU root and all mapped pages pointed into the bridge. The output went to the bridge completion word, readable through USB XDATA. After cycling slot power the host restored only the register BAR and bus mastering. Tinygrad compiled the unsigned multiply and add from UOps after link-up. Each of three fresh random inputs per cycle returned the exact 32-bit result. Across three guarded cycles all nine results arrived 94.77–96.52 ms after rail enable. Both management rings acquired valid headers at 727–737 ms. A warm-slot control returned three correct results with the rings already valid.

The guard rejected any host mapping outside BAR0, any PCIe request at or above the host's 32 GiB VRAM aperture, management IPC and access to the vendor compute image. An early test using the generic device wrapper attempted to allocate its VRAM timeline on exit. The diagnostic now opens the PCI device directly. This establishes compute independent of local DDR for the challenge. The factory boot storage remained installed throughout.

From the checkout with the card idle, run `PYTHONPATH=. python extra/usbgpu/s30/sram.py --cold`.

## November 20, 2026 — run the complete trained MNIST graph without local DDR

All 16 leading MNIST test images classified correctly on the S30, each in a fresh process and a separate slot power cycle. Tinygrad scheduled the trained beautiful_mnist model on the host, compiled all nine kernels to USC after rail enable and dispatched each on the card. The CPU prepared the checkpoint and a single input image. The GPU performed every convolution, activation, normalization, pooling, linear and class selection operation. Input weights streamed through a 160 KiB bridge window. Intermediate tensors stayed in a separate 224 KiB GPU-visible activation region, and the GPU stored its class index directly in Chestnut's USB-readable completion word.

The 16 image results arrived in 337–371 ms from rail enable. Neither management ring pair was ready. The diagnostic rejected all non-register BAR mappings, any TLP in the VRAM aperture, use of management IPC and access to the vendor compute image in these runs. Zeroing every tensor in a copy of the checkpoint changed the first prediction from 7 to 0 and failed the label check. In a separate placement experiment, compiling fresh just before rail enable produced a prediction at 70 ms. The checked command compiles after power-on. These single-image measurements include compilation, exclude loading the checkpoint on the host, and use different timing boundaries from the normal batch benchmark.

Run from the tinygrad checkout with the card idle and the model environment active:

```sh
PYTHONPATH=. python extra/usbgpu/s30/sram_mnist.py ../.s30-tools/mnist-s30.safetensors --index 0
```

The factory board boot storage remained present. The SRAM run removes GPU DDR initialization from MNIST's compute and data path. It cannot establish which board code executed before the PCIe link came up.

A separate launch-latency experiment read 0xff03 from the multicore routing register. After clearing the primary completion event, the primary event word read 0 and the secondary still read 4. The normal launcher continues to acknowledge each core.

The offline sandbox also accepts the SRAM model as a target. Its verifier rejects VRAM BAR mappings and TLPs, ties each loaded shader to freshly emitted UOps, and retains the normal checks on zeroed compute startup and blocked management IPC. All 16 test images passed in 16 independent sandboxes. Each verified nine distinct generated shaders and returned its class by 341–367 ms. Reproduce with the model virtualenv active:

```sh
extra/usbgpu/s30/bench_isolated.sh sram-mnist ../.s30-tools/mnist-s30.safetensors --index 0
```

## November 20, 2026 — distribute long dot products across more output rows

A 896-row, 4,864-element FP16 weight projection took 1.897 ms with eight rows and eight reduction lanes per workgroup, and 1.611 ms with 64 rows and two reduction lanes. All outputs matched NumPy. The compiler applies the latter shape to single row reductions at least twice as long as the output row count when that count is divisible by 64. For fixed matrix products with a long reduction, four output columns and two unrolled reduction terms also reuse the input across more outputs. A broader heuristic exhausted the register file on Qwen. The accepted rule requires three matrix axes and a single reduction.

Warm Qwen took 5.020 s in the tuning process and 5.031 s in the offline sandbox, down from 5.144 s immediately before the change. All five calls returned the reference tokens. Whisper measured 5.162 s and batch-16 MNIST 27.44 ms in the same isolated suite. A subsequent fresh MUSA run on this card supplies the comparison column at the top. All 165 archived inference graphs and 118 MNIST training graphs compiled, and all 17 applicable native hardware tests passed.

## November 20, 2026 — separate the early rail and lane-enable intervals

Across six slot cycles, the bridge held the S30 rail on for 1.2 s with the downstream lane disabled. Alternating trials kept the bridge PERST control asserted or released it at the beginning of that interval. LTSSM stayed at zero in both cases. The endpoint answered at 1.247 s and the management rings became ready at 1.926–1.936 s from rail enable. On this setup, releasing the PERST control alone with the lane disabled did not advance the observed mailbox transition. This identifies the lane-enable interval for an independent board trace. It does not determine the electrical state of the card's reset pin or tell us which code ran while PCIe was inaccessible.

The bridge-SRAM commands construct the direct compute context without a firmware path. The optional reference boot requires an explicitly supplied image.

## November 20, 2026 — compare complete SRAM scores and rerun the proprietary baseline

A separate early-slot run kept all ten pre-argmax scores in Chestnut SRAM and used a freshly compiled GPU copy kernel to return each score through the same completion word. For the first 16 images, each in a separate slot cycle, the largest absolute difference across all 160 logits versus an independent FP32 CPU process was 0.00000620. All 16 classifications matched. The normal SRAM diagnostic also runs argmax on the GPU.

Fresh MUSA processes with compiler caching disabled passed all 15 reference checks across the three workloads. The table at the top uses these measurements. For a possible launch-latency improvement, three trials comparing 200 four-byte reads of the compute ID returned a median of 80.46–80.47 µs with bulk input and 81.93–82.41 µs with a control transfer. Both returned the same ID. The runtime retains bulk reads.

## November 20, 2026 — stream Whisper's convolutional frontend through Chestnut SRAM

The trained tiny.en encoder ran both convolutions and both GELUs on the S30. The first layer used 64 output channels and 64 frames per tile, the second used 64 channels and 32 frames per tile. Adjacent tiles included the input halo for the three-tap filter. The host assembled each result for the next layer. Both compute shaders and a third shader that releases the USB bulk read came from tinygrad UOps compiled after rail enable. A host XDATA write to the completion phase byte timed out. A GPU store to the completion entry released a byte-exact bulk read from slot zero. This lets the host read a full activation tile.

Three isolated runs compared all 1,152,000 first-layer activations and all 576,000 second-layer activations for test.wav with the CPU model. Maximum absolute errors were 0.00000334 and 0.00000834 respectively. The first layer finished at 563–568 ms with the management rings uninitialized. The second finished at about 1.4 s with the rings ready. The verifier prohibited management IPC and card VRAM transactions for the entire run. This check ends before the encoder's positional embedding and attention. The full transcription benchmark at the top uses card VRAM.

```sh
extra/usbgpu/s30/bench_isolated.sh sram-whisper test/models/whisper/test.wav
```

## November 20, 2026 — finish both Whisper convolutions before management readiness

The file inference path now uses the encoder's existing silence-tail reconstruction when all batch members have an exactly constant mel suffix. It retains six equal frames for the convolution boundaries, then rounds the input up to a 64-frame JIT bucket. A late change in any mel band or batch member extends the prefix. A different bucket resets the prefix JIT.

For test.wav the prefix was 256 of 3,000 mel frames. In three isolated rail cycles the second convolution completed by 254–269 ms with the management mailbox still uninitialized. Reconstructing the repeated interior columns from the computed values gave the same maximum errors across all 1,728,000 activations as the full convolution experiment. Using the same optimization for both full transcriptions gave 4.858 s native and 2.282 s MUSA warm. All five transcriptions on each path matched the reference. The native sandbox verified 92 fresh programs.
