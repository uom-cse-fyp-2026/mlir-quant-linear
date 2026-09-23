#!/usr/bin/env bash
# Build quant-opt (mlir-opt + the quantize-linear pass) against torch-mlir's MLIR.
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh

cmake -S pass -B build-pass -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR="$TORCH_MLIR_BUILD/lib/cmake/mlir" \
  -DLLVM_DIR="$TORCH_MLIR_BUILD/lib/cmake/llvm" \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_LLD=ON
cmake --build build-pass

./build-pass/bin/quant-opt --help | grep -q "quantize-linear" \
  && echo "OK: quant-opt built with --quantize-linear"

# Sanity check on the hand-written test IR
./build-pass/bin/quant-opt pass/test/matmul.mlir \
  --pass-pipeline="builtin.module(func.func(quantize-linear{act-scale=0.02 w-scale=0.01}))"
