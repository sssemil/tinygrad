#!/usr/bin/env bash
# Offline S30 inference on a merged-/usr Linux host with open-source system libraries.
set -eu
if [[ ${1:-} == sram-mnist ]]; then
  if [[ $# != 2 && ( $# != 4 || $3 != --index ) ]]; then echo "usage: $0 sram-mnist WEIGHTS [--index N]" >&2; exit 2; fi
elif [[ ${1:-} == sram-whisper || ${1:-} == sram-generate || ${1:-} == sram-qwen ]]; then
  if [[ $# != 2 ]]; then echo "usage: $0 $1 INPUT" >&2; exit 2; fi
elif [[ $# != 2 && ( $# != 3 || ${3:-} != --check-reference ) ]]; then
  echo "usage: $0 {mnist|whisper|qwen} INPUT [--check-reference]" >&2; exit 2
fi
repo=$(cd "$(dirname "$0")/../../.." && pwd)
input=$(realpath "$2")
python=${PYTHON:-python}
venv=$("$python" -c 'import sys; assert sys.prefix != sys.base_prefix, "activate the model virtualenv"; print(sys.prefix)')
base=$("$python" -c 'import sys; print(sys.base_prefix)')
downloads=${XDG_CACHE_HOME:-$HOME/.cache}/tinygrad/downloads
read -r bus dev < <(lsusb -d 3801:0001 | awk 'NR==1 {print $2, substr($4,1,3)}')
node=/dev/bus/usb/$bus/$dev
name=mt_usb:$((10#$bus))-$((10#$dev)).lock
lock=$("$python" -c 'import os,sys,tempfile; print(os.path.join(tempfile.gettempdir(), sys.argv[1]))' "$name")
# The process inside the namespace must acquire the same lock as applications outside it.
(umask 000; set -C; : > "$lock") 2>/dev/null || test -e "$lock"

opts=(--die-with-parent --unshare-net --unshare-pid --clearenv
  --ro-bind /usr /usr --tmpfs /usr/lib/firmware --tmpfs /usr/local
  --symlink usr/lib /lib --symlink usr/lib64 /lib64 --symlink usr/bin /bin
  --ro-bind /etc/passwd /etc/passwd --ro-bind /etc/group /etc/group --ro-bind /etc/ld.so.cache /etc/ld.so.cache
  --proc /proc --dev /dev --ro-bind /sys /sys --dev-bind "$node" "$node" --tmpfs /tmp --bind "$lock" "/tmp/$name"
  --ro-bind "$venv" /venv --ro-bind "$repo/tinygrad" /src/tinygrad --ro-bind "$repo/examples" /src/examples
  --ro-bind "$repo/extra/usbgpu/bench_s30.py" /src/extra/usbgpu/bench_s30.py --ro-bind "$repo/extra/usbgpu/s30/verify_native.py" /src/extra/usbgpu/s30/verify_native.py
  --ro-bind "$input" /input
  --setenv HOME /home/bench --setenv PATH /venv/bin:/usr/bin --setenv PYTHONPATH /src
  --setenv PYTHONPYCACHEPREFIX /tmp/pycache --setenv PYTHONDONTWRITEBYTECODE 1 --setenv NUMBA_CACHE_DIR /tmp/numba --setenv CACHELEVEL 0 --setenv DEV MT:USC
  --setenv PARALLEL 0 --setenv MUSA_PATH /nonexistent --setenv MT_FIRMWARE /nonexistent --chdir /src)
if [[ $base == /usr/local ]]; then
  stdlib=$("$python" -c 'import sysconfig; print(sysconfig.get_path("stdlib"))')
  opts+=(--ro-bind "$base/bin" "$base/bin" --ro-bind "$stdlib" "$stdlib")
elif [[ $base != /usr ]]; then opts+=(--ro-bind "$base" "$base"); fi
# tinygrad names downloads by the MD5 of the upstream URL. These are the four IDX files, tiny.en weights and GPT-2 tokenizer.
case "$1" in
  mnist|sram-mnist) assets=(e8e6ed0857e9bb000985bfea8283b0b3.gunzip 684fe191a76d9107981692e6f0d3842f.gunzip 40a9ec2164c7887e8a5f1225de927907.gunzip 7b1e157ab43d54d0946f76d1bb6df574.gunzip) ;;
  whisper|sram-whisper) assets=(3a2cae277fcec4464349df535729caaf 5386a662f5b1245493601b29a5e61f40) ;;
  qwen|sram-qwen|sram-generate) assets=() ;;
  *) echo "unknown model $1" >&2; exit 2 ;;
esac
for asset in "${assets[@]}"; do
  opts+=(--ro-bind "$downloads/$asset" "/home/bench/.cache/tinygrad/downloads/$asset")
done
if [[ $1 == sram-* ]]; then
  for file in sram.py sram_tensor.py "${1/-/_}.py"; do opts+=(--ro-bind "$repo/extra/usbgpu/s30/$file" "/src/extra/usbgpu/s30/$file"); done
fi
exec bwrap "${opts[@]}" -- /venv/bin/python -m extra.usbgpu.s30.verify_native "$1" /input "${@:3}"
