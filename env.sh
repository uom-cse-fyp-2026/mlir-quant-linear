# Source this file (". env.sh") before running anything in this repo.
# Override TORCH_MLIR_SRC if you cloned torch-mlir somewhere else.
export TORCH_MLIR_SRC=${TORCH_MLIR_SRC:-$HOME/tools/torch-mlir}
export TORCH_MLIR_BUILD=${TORCH_MLIR_BUILD:-$TORCH_MLIR_SRC/build}
source "$TORCH_MLIR_SRC/mlir_venv/bin/activate"
export PATH=$TORCH_MLIR_BUILD/bin:$PATH
export PYTHONPATH=$TORCH_MLIR_BUILD/tools/torch-mlir/python_packages/torch_mlir:$TORCH_MLIR_SRC/test/python/fx_importer
