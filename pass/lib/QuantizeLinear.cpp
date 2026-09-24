//===- QuantizeLinear.cpp - Quantize linalg.matmul inputs to INT8 ---------===//
//
// For every linalg.matmul, rewrite both inputs (activation x and weight W) as
//
//   v  ->  quant.dcast(quant.qcast(v))
//
// using a symmetric per-tensor type !quant.uniform<i8:f32, scale> (zero-point 0,
// storage range [-128, 127]). After this pass the IR explicitly carries INT8
// activation and weight tensors (W8A8, quantize-dequantize form).
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/Builders.h"
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
    return "Quantize activation and weight inputs of linalg.matmul to INT8";
  }
  void getDependentDialects(DialectRegistry &r) const override {
    r.insert<quant::QuantDialect>();
  }

  Option<double> actScale{*this, "act-scale",
                          llvm::cl::desc("activation (x) scale"),
                          llvm::cl::init(1.0 / 127)};
  Option<double> wScale{*this, "w-scale", llvm::cl::desc("weight (W) scale"),
                        llvm::cl::init(1.0 / 127)};

  // v  ->  quant.dcast(quant.qcast(v))  with a symmetric int8 type.
  Value insertQDQ(OpBuilder &b, Location loc, Value v, double scale) {
    auto ty = cast<RankedTensorType>(v.getType());
    auto qElemTy = quant::UniformQuantizedType::get(
        quant::QuantizationFlags::Signed, b.getIntegerType(8),
        ty.getElementType(), scale, /*zeroPoint=*/0,
        /*storageTypeMin=*/-128, /*storageTypeMax=*/127);
    auto qTy = RankedTensorType::get(ty.getShape(), qElemTy);
    Value q = quant::QuantizeCastOp::create(b, loc, qTy, v);
    return quant::DequantizeCastOp::create(b, loc, ty, q);
  }

  void runOnOperation() override {
    getOperation().walk([&](linalg::MatmulOp op) {
      OpOperand *xOpd = op.getDpsInputOperand(0); // activation
      OpOperand *wOpd = op.getDpsInputOperand(1); // weight (transposed)
      if (xOpd->get().getDefiningOp<quant::DequantizeCastOp>())
        return; // already quantized
      if (!isa<RankedTensorType>(xOpd->get().getType()) ||
          !isa<RankedTensorType>(wOpd->get().getType())) {
        op.emitWarning("quantize-linear: skipping matmul on non-tensor operands");
        return;
      }
      OpBuilder b(op); // insert right before the matmul
      xOpd->set(insertQDQ(b, op.getLoc(), xOpd->get(), actScale));
      wOpd->set(insertQDQ(b, op.getLoc(), wOpd->get(), wScale));
    });
  }
};
} // namespace

void registerQuantizeLinearPass() { PassRegistration<QuantizeLinearPass>(); }
