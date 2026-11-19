#!/usr/bin/env bash
# Reproduce the installed ed4e39b7 bridge application with SDCC 4.2.0.
set -euo pipefail
if [[ $# != 3 ]]; then echo "usage: $0 BRIDGE_GIT_DIR SDCC_RUNTIME_SOURCES OUTPUT" >&2; exit 2; fi
bridge=$(realpath "$1")
runtime=$(realpath "$2")
build=$(mktemp -d)
trap 'rm -rf "$build"' EXIT
(
  cd "$build"
  git -C "$bridge" archive ed4e39b7e0794e19ba193477067c48757a5cf9ef handmade/src handmade/Makefile | tar -x --strip-components=1
  sdas8051 -los src/crt0.s
  sdcc -mmcs51 --model-small --opt-code-speed --allow-unsafe-read --std-c99 -Isrc --no-xinit-opt \
    '-DGIT_VERSION="ed4e39b7-CLEAN"' -c src/main.c -o main.rel
  for name in _divulong _gptrput _mullong _divslong _startup; do
    sdcc -mmcs51 --model-small -c "$runtime/$name.c" -o "$name.rel"
  done
  for name in crtclear crtstart; do
    cp "$runtime/mcs51/$name.asm" .
    sdas8051 -losg "$name.asm"
  done
  # Preserve the installed image's section order while explicitly linking every target object built above.
  sdcc -mmcs51 --model-small --nostdlib --code-loc 0 --code-size 0x10000 --xram-loc 0 --xram-size 0x10000 --iram-size 0x100 \
    src/crt0.rel main.rel _divulong.rel crtclear.rel _gptrput.rel _mullong.rel _divslong.rel crtstart.rel _startup.rel -o firmware.ihx
  make BUILD_DIR=. -o firmware.ihx wrapped
  echo "9520fde0bf43d499c07abd0a09b74e94d8a7cc3d610f577b7a4e218ab8a378e9  firmware_wrapped.bin" | sha256sum -c -
)
cp "$build/firmware_wrapped.bin" "$3"
