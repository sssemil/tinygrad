# Runtimes

tinygrad supports various runtimes, enabling your code to scale across a wide range of devices. The default runtime can be automatically selected based on the available hardware, or you can force a specific runtime to be default using environment variables (e.g., `DEV=CPU`).

| Runtime | Description | Compiler Options | Requirements |
|---------|-------------|------------------|--------------|
| [NV](https://github.com/tinygrad/tinygrad/tree/master/tinygrad/runtime/ops_nv.py) | Provides acceleration for NVIDIA GPUs | nvrtc (default)<br>PTX (`DEV=NV:PTX`) | Ampere/Ada/Blackwell series GPUs.<br>You can select an interface via [the `DEV` variable](env_vars.md#dev-variable). See [NV interfaces](#nv-interfaces) for details. |
| [AMD](https://github.com/tinygrad/tinygrad/tree/master/tinygrad/runtime/ops_amd.py) | Provides acceleration for AMD GPUs | LLVM (`DEV=AMD:LLVM`)<br>HIP/COMGR (`DEV=AMD:HIP`) | CDNA3, CDNA4, RDNA3 or RDNA4 GPUs.<br>You can select an interface via [the `DEV` variable](env_vars.md#dev-variable). See [AMD interfaces](#amd-interfaces) for details. |
| [QCOM](https://github.com/tinygrad/tinygrad/tree/master/tinygrad/runtime/ops_qcom.py) | Provides acceleration for QCOM GPUs | - | 6xx series GPUs |
| [METAL](https://github.com/tinygrad/tinygrad/tree/master/tinygrad/runtime/ops_metal.py) | Utilizes Metal for acceleration on Apple devices | - | M1+ Macs; Metal 3.0+ for `bfloat` support |
| [CUDA](https://github.com/tinygrad/tinygrad/tree/master/tinygrad/runtime/ops_cuda.py) | Utilizes CUDA for acceleration on NVIDIA GPUs | nvrtc (default)<br> PTX (`DEV=CUDA:PTX`) | NVIDIA GPU with CUDA support |
| [CL](https://github.com/tinygrad/tinygrad/tree/master/tinygrad/runtime/ops_cl.py) | Accelerates computations using OpenCL on GPUs | - | OpenCL 2.0 compatible device |
| [CPU](https://github.com/tinygrad/tinygrad/tree/master/tinygrad/runtime/ops_cpu.py) | Runs on CPU using the clang or llvm compiler | Clang JIT (default)<br>LLVM IR (`DEV=CPU:LLVM`) | `clang` compiler in system `PATH`<br>You can specify additional arch parameters via [the `DEV` variable](env_vars.md#dev-variable). See [CPU arch](#cpu-arch) for details. |
| [WEBGPU](https://github.com/tinygrad/tinygrad/tree/master/tinygrad/runtime/ops_webgpu.py) | Runs on GPU using the Dawn WebGPU engine (used in Google Chrome) | - | Dawn library installed and discoverable. Binaries: [pydawn v0.3.0](https://github.com/wpmed92/pydawn/releases/tag/v0.3.0) |


## MTT S30 on USB Chestnut

On a two-core S30, `DEV=MT` selects tinygrad’s native USC renderer. Use `DEV=MT:USC` to require it explicitly. Tinygrad generates USC shader instructions and the PDS launch program, maps them into VRAM, and submits work directly to both compute cores. This path requires the Chestnut USB bridge and access to its USB device. The S30 can still run factory firmware at power-on. Direct host dispatch uses neither the management mailbox nor a host-supplied firmware image.

Tinygrad uses the S30's META processor for dispatch when the GNU META toolchain is available. Set `MT_META=1` to require this path. At startup tinygrad assembles `tinygrad/runtime/support/mt/dispatch.s` and uploads the result to volatile GPU memory. META starts both compute engines and acknowledges each batch after both finish. This path uses GNU binutils built with `--target=metag-elf`. Put `metag-elf-as` and `metag-elf-objcopy` on PATH, or set `META_AS` and `META_OBJCOPY` to their paths. Use `MT_META=0` for direct host dispatch.

Run a model five times from the checkout with the normal Python dependencies for that example. Set `CACHELEVEL=0` to compile from source in a fresh process. The benchmark reports time to the first output and median warm latency. Whisper uses the repository audio fixture. Qwen processes the same prompt each time.

```sh
export DEV=MT:USC CACHELEVEL=0 PYTHONPATH=.
python extra/usbgpu/bench_s30.py mnist /path/to/mnist-s30.safetensors
python extra/usbgpu/bench_s30.py whisper test/models/whisper/test.wav
python extra/usbgpu/bench_s30.py qwen /path/to/Qwen2.5-0.5B-Instruct-Q8_0.gguf
```

On a Linux host with the toolkit outside the system libraries, the offline benchmark can also isolate the filesystem. Activate the Python virtualenv for the example, prefetch its model and dataset, and install Bubblewrap and usbutils. The script exposes the selected input and cached model data, creates a fresh home and temporary directory, and opens the Chestnut through the same device lock as the host.

```sh
extra/usbgpu/s30/bench_isolated.sh mnist /path/to/mnist-s30.safetensors
extra/usbgpu/s30/bench_isolated.sh whisper test/models/whisper/test.wav
extra/usbgpu/s30/bench_isolated.sh qwen /path/to/Qwen2.5-0.5B-Instruct-Q8_0.gguf
```

For the vendor reference, select `DEV=MT:MUSA`. Point `MUSA_PATH` at the MUSA 4.3.0 toolkit containing `bin/mcc`. Both that renderer and `DEV=MT:LLVM`, which uses the toolkit's MTGPU `bin/clang-14`, load `mtfw-gen1.bin` from the `mtgpu-s30` 3.1.0-6 package. Set `MT_FIRMWARE` if the file is outside `/lib/firmware/mthreads`. [Hardware measurements and bring-up journal](../extra/usbgpu/s30-journal.md)

For a short native training check, run three Adam updates on the S30. The ten-step evaluation interval in this example exceeds the length of the check.

```sh
DEV=MT:USC CACHELEVEL=0 PYTHONPATH=. BS=2 STEPS=3 python3 examples/beautiful_mnist.py
```

Run the full MNIST training and evaluation demo on the vendor reference from the checkout.

```sh
DEV=MT:MUSA MUSA_PATH=/path/to/musa MT_FIRMWARE=/path/to/mtfw-gen1.bin \
  PYTHONPATH=. TARGET_EVAL_ACC_PCT=96 python3 examples/beautiful_mnist.py
```

## Interoperability

tinygrad provides interoperability with OpenCL and PyTorch, allowing efficient tensor data sharing between frameworks through the `Tensor.from_blob` API. This enables zero-copy operations by working directly with external memory pointers.

**Important**: When using external memory pointers with tinygrad tensors, you must ensure these pointers remain valid throughout the entire lifetime of the tinygrad tensor to prevent memory corruption.

### `CUDA`/`METAL` PyTorch Interoperability

You can seamlessly work with CUDA/MPS tensors between PyTorch and tinygrad without data copying:
```python
from tinygrad.dtype import _from_torch_dtype
tensor1 = torch.tensor([1.0, 2.0, 3.0], device=torch.device("cuda"))
tiny_tensor1 = Tensor.from_blob(tensor1.data_ptr(), tensor1.shape, dtype=_from_torch_dtype(tensor1.dtype), device='CUDA')

# Before tinygrad calculations, mps needs to be synchronized to make sure data is valid.
if data.device.type == "mps": torch.mps.synchronize()
else: torch.cuda.synchronize()

x = (tiny_tensor1 + 1).realize()
```

### `QCOM` OpenCL Interoperability

tinygrad supports OpenCL interoperability on `QCOM` backend.

Buffer interop allows direct access to OpenCL memory buffers:
```python
# create raw opencl buffer.
cl_buf = cl.clCreateBuffer(cl_context, cl.CL_MEM_READ_WRITE, 0x100, None, status := ctypes.c_int32())

# extract pointers
cl_buf_desc_ptr = to_mv(ctypes.addressof(cl_buf), 8).cast('Q')[0]
rawbuf_ptr = to_mv(cl_buf_desc_ptr, 0x100).cast('Q')[20] # offset 0xA0 is a raw gpu pointer.

# create tiny tensor
tiny = Tensor.from_blob(rawbuf_ptr, (8, 8), dtype=dtypes.int, device='QCOM')
```

And the same for the images:
```python
# create cl image.
cl_img = cl.clCreateImage2D(cl_context, cl.CL_MEM_READ_WRITE, cl.cl_image_format(cl.CL_RGBA, cl.CL_FLOAT), w, h, 0, None, status := ctypes.c_int32())

# extract pointers
cl_buf_desc_ptr = to_mv(ctypes.addressof(cl_img), 8).cast('Q')[0]
rawbuf_ptr = to_mv(cl_buf_desc_ptr, 0x100).cast('Q')[20] # offset 0xA0 is a raw gpu pointer.

# create tiny tensor
tiny = Tensor.from_blob(rawbuf_ptr, (h*w*4,), dtype=dtypes.imagef((h,w)), device='QCOM')
```

## AMD Interfaces
AMD backend supports several interfaces for communicating with devices:

* `KFD`: uses the amdgpu driver
* `PCI`: uses the [AM driver](developer/am.md)
* `USB`: USB3 interface for asm24xx chips.

You can force an interface by setting the interface component of [the `DEV` environment variable](env_vars.md#dev-variable) to one of these values. When set to `PCI`, this may unbind your GPU from the amdgpu driver.

## NV Interfaces
NV backend supports several interfaces for communicating with devices:

* `NVK`: uses the nvidia driver
* `PCI`: uses the [NV driver](https://github.com/tinygrad/tinygrad/tree/master/tinygrad/runtime/support/nv/nvdev.py)

## CPU Arch
The CPU renderers may be additionally configured using the arch component of [the `DEV` environment variable](env_vars.md#dev-variable).
CPU arch should be specified as a comma-separated list of parameters, and must contain at least two values: the architecture family (ie. x86_64, arm64, or riscv64) and the cpu type (as accepted by `clang`'s `-march`).
If native is specified as the cpu type, tinygrad (or delegate compiler) will query the host cpu type. Additional comma-separated values are interpreted as cpu feature flags. When a value is preceded by a `-` character, the corresponding feature flag will be disabled, otherwise the flag will be enabled.
Note that enabled feature flags should not be preceded by a `+`.
