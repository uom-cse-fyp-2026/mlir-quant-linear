// Stand-alone input for the pass (no torch-mlir needed):
//   ./build-pass/bin/quant-opt pass/test/matmul.mlir \
//     --pass-pipeline="builtin.module(func.func(quantize-linear{act-scale=0.02 w-scale=0.01}))"
//
// Expected: activation and weight quantized to i8, integer matmul, dequantize:
//   %q0 = quant.qcast %x : tensor<1x64xf32> to tensor<1x64x!quant.uniform<i8:f32, 2.000000e-02>>
//   %x8 = quant.scast %q0 : tensor<1x64x!quant.uniform<i8:f32, 2.000000e-02>> to tensor<1x64xi8>
//   %q1 = quant.qcast %w : tensor<64x32xf32> to tensor<64x32x!quant.uniform<i8:f32, 1.000000e-02>>
//   %w8 = quant.scast %q1 : ... to tensor<64x32xi8>
//   %acc = linalg.matmul ins(%x8, %w8 : tensor<1x64xi8>, tensor<64x32xi8>) outs(... : tensor<1x32xi32>)
//   %f = arith.sitofp %acc : tensor<1x32xi32> to tensor<1x32xf32>
//   %y = arith.mulf %f, <act_scale * w_scale>  ;  arith.addf %y, <matmul init>

func.func(quantize-linear{act-scale=0.02 w-scale=0.01}))"
//
// Expected: two quant.qcast / quant.dcast pairs feeding linalg.matmul, e.g.
//   %0 = quant.qcast %x : tensor<1x64xf32> to tensor<1x64x!quant.uniform<i8:f32, 2.000000e-02>>
//   %1 = quant.dcast %0 : ... to tensor<1x64xf32>
//   %2 = quant.qcast %w : tensor<64x32xf32> to tensor<64x32x!quant.uniform<i8:f32, 1.000000e-02>>
//   %3 = quant.dcast %2 : ... to tensor<64x32xf32>
//   linalg.matmul ins(%1, %3 : ...)

func.func @linear_relu(%x: tensor<1x64xf32>, %w: tensor<64x32xf32>,
                       %b: tensor<1x32xf32>) -> tensor<1x32xf32> {
  %zero = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<1x32xf32>
  %acc = linalg.fill ins(%zero : f32) outs(%init : tensor<1x32xf32>) -> tensor<1x32xf32>
  %mm = linalg.matmul ins(%x, %w : tensor<1x64xf32>, tensor<64x32xf32>)
                      outs(%acc : tensor<1x32xf32>) -> tensor<1x32xf32>
  %a = arith.addf %mm, %b : tensor<1x32xf32>
  %zeros = arith.constant dense<0.0> : tensor<1x32xf32>
  %y = arith.maximumf %a, %zeros : tensor<1x32xf32>
  return %y : tensor<1x32xf32>
}
