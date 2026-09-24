//===- QuantizeLinear.cpp - INT8 (W8A8) quantization of linalg.matmul -----===//
//
// For every f32 linalg.matmul  C = init + X * W  the pass emits
//
//   Xq  = quant.scast(quant.qcast(X))   : tensor<MxKxi8>   (activation, INT8)
//   Wq  = quant.scast(quant.qcast(W))   : tensor<KxNxi8>   (weight, INT8)
//   Acc = linalg.matmul(Xq, Wq)         : tensor<MxNxi32>  (integer matmul)
//   C   = init + sitofp(Acc) * (act_scale * w_scale)
//
// Quantization is symmetric per-tensor: !quant.uniform<i8:f32, scale>,
// zero-point 0, storage range [-128, 127].
//
// Note: a plain quant.dcast(quant.qcast(v)) pair is NOT used, because the quant
// dialect folds that pair back to v, which would remove the quantization.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
struct QuantizeLinearPass
    : PassWrapper<QuantizeLinearPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QuantizeLinearPass)

  QuantizeLinearPass() = default;
  QuantizeLinearPass(const QuantizeLinearPass &p) : PassWrapper(p) {}

  StringRef getArgument() const final { return "quantize-linear"; }
  StringRef getDescription() const final {
    return "Quantize activation and weight of linalg.matmul to INT8 and run "
           "the matmul in integer arithmetic (i8 x i8 -> i32)";
  }
  void getDependentDialects(DialectRegistry &r) const override {
    r.insert<quant::QuantDialect, arith::ArithDialect, tensor::TensorDialect,
             linalg::LinalgDialect>();
  }

  Option<double> actScale{*this, "act-scale",
                          llvm::cl::desc("activation (x) scale"),
                          llvm::cl::init(1.0 / 127)};
  Option<double> wScale{*this, "w-scale", llvm::cl::desc("weight (W) scale"),
                        llvm::cl::init(1.0 / 127)};

  // f32 tensor -> quant.qcast -> quant.scast -> i8 tensor
  Value quantizeToI8(OpBuilder &b, Location loc, Value v, double scale) {
    auto ty = cast<RankedTensorType>(v.getType());
    auto i8 = b.getIntegerType(8);
    auto qElemTy = quant::UniformQuantizedType::get(
        quant::QuantizationFlags::Signed, i8, ty.getElementType(), scale,
        /*zeroPoint=*/0, /*storageTypeMin=*/-128, /*storageTypeMax=*/127);
    Value q = quant::QuantizeCastOp::create(
        b, loc, RankedTensorType::get(ty.getShape(), qElemTy), v);
    return quant::StorageCastOp::create(
        b, loc, RankedTensorType::get(ty.getShape(), i8), q);
  }

  // Only plain (non-transposed) f32 matmuls on static tensors are rewritten.
  bool isCandidate(linalg::MatmulOp op) {
    auto xTy = dyn_cast<RankedTensorType>(op.getDpsInputOperand(0)->get().getType());
    auto wTy = dyn_cast<RankedTensorType>(op.getDpsInputOperand(1)->get().getType());
    if (!xTy || !wTy || op->getNumResults() != 1)
      return false;
    auto outTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!outTy || !xTy.hasStaticShape() || !wTy.hasStaticShape() ||
        !outTy.hasStaticShape())
      return false;
    if (!xTy.getElementType().isF32() || !wTy.getElementType().isF32() ||
        !outTy.getElementType().isF32())
      return false;

    MLIRContext *ctx = op.getContext();
    AffineExpr d0, d1, d2;
    bindDims(ctx, d0, d1, d2);
    SmallVector<AffineMap> defaultMaps = {
        AffineMap::get(3, 0, {d0, d2}, ctx), AffineMap::get(3, 0, {d2, d1}, ctx),
        AffineMap::get(3, 0, {d0, d1}, ctx)};
    return op.getIndexingMapsArray() == defaultMaps;
  }

  void rewrite(linalg::MatmulOp op) {
    OpBuilder b(op); // insert right before the matmul
    Location loc = op.getLoc();
    Value x = op.getDpsInputOperand(0)->get(); // activation
    Value w = op.getDpsInputOperand(1)->get(); // weight (transposed)
    Value init = op.getDpsInitOperand(0)->get();
    auto outTy = cast<RankedTensorType>(op->getResult(0).getType());

    // 1. Quantize activation and weight to INT8.
    Value xq = quantizeToI8(b, loc, x, actScale);
    Value wq = quantizeToI8(b, loc, w, wScale);

    // 2. Integer matmul: i8 x i8 -> i32 (linalg.matmul sign-extends inputs).
    auto accTy = RankedTensorType::get(outTy.getShape(), b.getI32Type());
    Value empty =
        tensor::EmptyOp::create(b, loc, outTy.getShape(), b.getI32Type());
    Value zero = arith::ConstantOp::create(b, loc, b.getI32IntegerAttr(0));
    Value acc = linalg::FillOp::create(b, loc, zero, empty).getResult(0);
    Value qmm = linalg::MatmulOp::create(b, loc, TypeRange{accTy},
                                         ValueRange{xq, wq}, ValueRange{acc})
                    .getResult(0);

    // 3. Dequantize the i32 result: f32(acc) * act_scale * w_scale, + init.
    Value accF = arith::SIToFPOp::create(b, loc, outTy, qmm);
    float s = static_cast<float>(actScale) * static_cast<float>(wScale);
    Value scale = arith::ConstantOp::create(
        b, loc, DenseElementsAttr::get(outTy, ArrayRef<float>{s}));
    Value y = arith::MulFOp::create(b, loc, accF, scale);
    y = arith::AddFOp::create(b, loc, y, init); // matmul accumulates into init

    op->getResult(0).replaceAllUsesWith(y);
    op->erase();
  }

  void runOnOperation() override {
    SmallVector<linalg::MatmulOp> matmuls;
    getOperation().walk([&](linalg::MatmulOp op) {
      if (isCandidate(op))
        matmuls.push_back(op);
      else
        op.emitWarning("quantize-linear: matmul skipped (not a static f32 "
                       "matmul with default indexing maps)");
    });
    for (linalg::MatmulOp op : matmuls)
      rewrite(op);
  }
};
} // namespace

void registerQuantizeLinearPass() { PassRegistration<QuantizeLinearPass>(); }
