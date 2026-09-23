#!/usr/bin/env bash
# One-time setup on a fresh Ubuntu 22.04/24.04 machine (e.g. a GCP VM).
# Builds torch-mlir from source together with a matching MLIR/LLVM tree.
# Needs ~16 GB RAM and ~40 GB disk; takes 1-2 hours.
#   JOBS=4 ./scripts/setup_toolchain.sh     # lower JOBS if the build runs out of RAM
set -euo pipefail

TORCH_MLIR_SRC=${TORCH_MLIR_SRC:-$HOME/tools/torch-mlir}
JOBS=${JOBS:-$(nproc)}

sudo apt-get update
sudo apt-get install -y git cmake ninja-build clang lld python3-venv python3-dev build-essential

if [ ! -d "$TORCH_MLIR_SRC" ]; then
  mkdir -p "$(dirname "$TORCH_MLIR_SRC")"
  git clone https://github.com/llvm/torch-mlir.git "$TORCH_MLIR_SRC"
fi
cd "$TORCH_MLIR_SRC"
git submodule update --init --progress --depth=1

if [ ! -d mlir_venv ]; then
  python3 -m venv mlir_venv
fi
source mlir_venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r torchvision-requirements.txt

# Flags follow torch-mlir docs/development.md; if they change, follow that file.
# lld is selected with LLVM_USE_LINKER (LLVM_ENABLE_LLD conflicts with it inside
# torch-mlir's sub-projects). StableHLO is not needed for the linalg path.
cmake -GNinja -Bbuild \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_LLD=OFF -DLLVM_USE_LINKER=lld \
  -DTORCH_MLIR_ENABLE_STABLEHLO=OFF \
  -DPython3_FIND_VIRTUALENV=ONLY \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS="torch-mlir" \
  -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="$PWD" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_TARGETS_TO_BUILD=host \
  externals/llvm-project/llvm
cmake --build build -j "$JOBS"

echo
echo "torch-mlir commit: $(git rev-parse HEAD)"
echo "llvm-project commit: $(git -C externals/llvm-project rev-parse HEAD)"
echo "Done. Now: cd <this repo> && source env.sh && ./scripts/build_pass.sh"
