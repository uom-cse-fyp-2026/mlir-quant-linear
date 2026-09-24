# mlir-quant-linear

Quantize a PyTorch `Linear + ReLU` layer with a custom MLIR pass that uses the
`quant` dialect.

```
PyTorch (Linear + ReLU) -> torch-mlir -> linalg IR -> quantize-linear pass (quant dialect)
                        -> lower to LLVM -> mlir-runner -> compare with PyTorch
```

* **Linear:** `a = Wx + b` (`nn.Linear(64, 32)`)
* **Non-linear:** `ReLU(a)`
* **Quantized:** the weight `W` and the input activation `x` of the matmul, to
  symmetric per-tensor INT8 (`!quant.uniform<i8:f32, scale>`, zero-point 0), i.e. W8A8.

## Layout

```
env.sh                     environment (torch-mlir venv, PATH, PYTHONPATH)
python/model.py            LinearReLU model + fake_quant reference
python/calibrate.py        fixed weights, INT8 scales, reference outputs
python/export.py           torch-mlir export -> build/1_torch.mlir, build/2_linalg.mlir
python/compare.py          checks MLIR outputs against PyTorch
pass/lib/QuantizeLinear.cpp  the transformation pass (quantize-linear)
pass/tools/quant-opt.cpp     mlir-opt driver with the pass registered
pass/test/matmul.mlir        small hand-written input for the pass
scripts/setup_toolchain.sh   one-time build of torch-mlir + MLIR (Ubuntu / GCP VM)
scripts/build_pass.sh        builds build-pass/bin/quant-opt
scripts/run.sh               full pipeline end to end
```

## Running (Ubuntu 22.04/24.04, e.g. a GCP VM)

The VM needs about 16 GB RAM (e2-standard-8 or larger) and 40+ GB disk.

```bash
git clone https://github.com/uom-cse-fyp-2026/mlir-quant-linear.git
cd mlir-quant-linear

# 1. One time: build torch-mlir and the matching MLIR (1-2 h). Use JOBS=4 if RAM is short.
./scripts/setup_toolchain.sh

# 2. Build the pass
./scripts/build_pass.sh

# 3. Run everything
./scripts/run.sh
```

torch-mlir exports the IR and `quant-opt` parses it, so both must come from the
same MLIR version. Building torch-mlir from source (step 1) provides that MLIR
tree, and `build_pass.sh` links against it. If torch-mlir lives elsewhere, set
`TORCH_MLIR_SRC` before sourcing `env.sh`.

### Expected output of `run.sh`

```
MLIR fp32  vs PyTorch fp32                : True
MLIR quant vs PyTorch fake-quant (round)  : ...
MLIR quant vs PyTorch fake-quant (trunc)  : ...
quantization error (max |q - fp32|)       : <small, non-zero>
RESULT: PASS
```

* fp32 MLIR matches PyTorch, so the export and lowering are correct.
* Quantized MLIR matches the PyTorch fake-quant reference. MLIR's
  `--lower-quant-ops` converts float to int with `arith.fptosi`, which rounds
  toward zero, so both rounding modes are checked.
* A small non-zero error against fp32 shows that quantization really happened.

Every stage is kept in `build/`: `1_torch.mlir` -> `2_linalg.mlir` ->
`3_quantized.mlir` -> `4_llvm_q.mlir`.

## The transformation

`quantize-linear` runs on each `func.func` and rewrites every f32 `linalg.matmul`
(`C = init + X*W`) into INT8 form:

```
Xq  = quant.scast(quant.qcast(X))   tensor<i8>   activation quantized
Wq  = quant.scast(quant.qcast(W))   tensor<i8>   weight quantized
Acc = linalg.matmul(Xq, Wq)         tensor<i32>  integer matmul
C   = init + sitofp(Acc) * (act_scale * w_scale)
```

Before (from `2_linalg.mlir`):

```mlir
%t  = linalg.transpose ins(%W : tensor<32x64xf32>) outs(...) permutation = [1, 0]
%mm = linalg.matmul ins(%x, %t : tensor<1x64xf32>, tensor<64x32xf32>) outs(%acc ...)
```

After (`3_quantized.mlir`):

```mlir
%0 = quant.qcast %x : tensor<1x64xf32> to tensor<1x64x!quant.uniform<i8:f32, <act_scale>>>
%1 = quant.scast %0 : tensor<1x64x!quant.uniform<i8:f32, <act_scale>>> to tensor<1x64xi8>
%2 = quant.qcast %t : tensor<64x32xf32> to tensor<64x32x!quant.uniform<i8:f32, <w_scale>>>
%3 = quant.scast %2 : tensor<64x32x!quant.uniform<i8:f32, <w_scale>>> to tensor<64x32xi8>
%4 = linalg.matmul ins(%1, %3 : tensor<1x64xi8>, tensor<64x32xi8>) outs(%acc_i32 : tensor<1x32xi32>)
%5 = arith.sitofp %4 : tensor<1x32xi32> to tensor<1x32xf32>
%6 = arith.mulf %5, %scale : tensor<1x32xf32>        // act_scale * w_scale
%7 = arith.addf %6, %init : tensor<1x32xf32>         // then + b, ReLU unchanged
```

Why not just `quant.dcast(quant.qcast(v))` in front of an f32 matmul? The quant
dialect folds that pair back to `v`, so after lowering the quantization is gone
and the output equals fp32. `compare.py` checks for that: the quantization
error must be non-zero.

Scales are computed in `calibrate.py` (`max|t| / 127`) and passed to the pass
as options (`act-scale`, `w-scale`).

To run only the pass:

```bash
source env.sh
./build-pass/bin/quant-opt pass/test/matmul.mlir \
  --pass-pipeline="builtin.module(func.func(quantize-linear{act-scale=0.02 w-scale=0.01}))"
```

## Troubleshooting

* **Link errors in `build_pass.sh`:** compare the library list with
  `$TORCH_MLIR_SRC/externals/llvm-project/mlir/examples/standalone/standalone-opt/CMakeLists.txt`,
  which matches your LLVM version.
* **Unknown pass name while lowering:** pass names change between LLVM versions.
  `quant-opt --help` lists the passes your build has.
* **`linalg.batch_matmul` instead of `linalg.matmul`:** the input must stay 2-D (`1x64`).
* **Leftover ops at run time:** see which dialect remains in `build/4_llvm_*.mlir`
  and add the matching `--convert-<dialect>-to-llvm` pass in `scripts/run.sh`.

## Possible extensions

* Per-channel weight scales (`quant::UniformQuantizedPerAxisType`).
* Compute the weight scale inside the pass from the constant weight.
* Also quantize the layer output `a` (a QDQ pair after the matmul).

## Toolchain version

Record the commits you built with (printed by `setup_toolchain.sh`):

* torch-mlir: `git -C ~/tools/torch-mlir rev-parse HEAD`
* llvm-project: `git -C ~/tools/torch-mlir/externals/llvm-project rev-parse HEAD`
