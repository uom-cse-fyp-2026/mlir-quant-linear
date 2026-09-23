#!/usr/bin/env bash
# Full pipeline:
#   PyTorch -> torch-mlir (linalg) -> quantize-linear pass -> LLVM -> run -> compare
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh

QOPT=./build-pass/bin/quant-opt
[ -x "$QOPT" ] || { echo "quant-opt not found; run ./scripts/build_pass.sh first"; exit 1; }
RUNNER=$(command -v mlir-runner || command -v mlir-cpu-runner || true)
[ -n "$RUNNER" ] || { echo "mlir-runner / mlir-cpu-runner not found in $TORCH_MLIR_BUILD/bin"; exit 1; }
LIBS=$TORCH_MLIR_BUILD/lib/libmlir_runner_utils.so,$TORCH_MLIR_BUILD/lib/libmlir_c_runner_utils.so

export PYTHONPATH=$PWD/python:$PYTHONPATH

echo "== 1. PyTorch model + calibration"
python python/calibrate.py

echo "== 2. Export to MLIR (torch dialect + linalg-on-tensors)"
python python/export.py

ACT=$(python -c "import json;print(repr(json.load(open('build/scales.json'))['act_scale']))")
W=$(python -c "import json;print(repr(json.load(open('build/scales.json'))['w_scale']))")

echo "== 3. quantize-linear pass (act-scale=$ACT, w-scale=$W)"
$QOPT build/2_linalg.mlir \
  --pass-pipeline="builtin.module(func.func(quantize-linear{act-scale=$ACT w-scale=$W}))" \
  -o build/3_quantized.mlir
grep -n "quant\.\(qcast\|dcast\)" build/3_quantized.mlir | cut -c1-200

echo "== 4. Lower to LLVM and execute"
LOWER=(--lower-quant-ops --canonicalize --strip-func-quant-types
       --convert-elementwise-to-linalg
       --one-shot-bufferize="bufferize-function-boundaries"
       --convert-linalg-to-loops --lower-affine --convert-scf-to-cf
       --expand-strided-metadata --finalize-memref-to-llvm
       --convert-math-to-llvm --convert-arith-to-llvm --convert-index-to-llvm
       --convert-cf-to-llvm --convert-func-to-llvm
       --reconcile-unrealized-casts)

for v in fp32:2_linalg q:3_quantized; do
  name=${v%%:*}; src=${v##*:}
  $QOPT build/$src.mlir "${LOWER[@]}" -o build/4_llvm_$name.mlir
  if grep -q "quant\." build/4_llvm_$name.mlir; then
    echo "ERROR: quant ops left in build/4_llvm_$name.mlir:"; grep -n "quant\." build/4_llvm_$name.mlir | head
    exit 1
  fi
  $RUNNER build/4_llvm_$name.mlir -e main -entry-point-result=void \
    -shared-libs=$LIBS > build/out_$name.txt
  echo "-- $name output:"; cat build/out_$name.txt
done

echo "== 5. Compare with PyTorch"
python python/compare.py
