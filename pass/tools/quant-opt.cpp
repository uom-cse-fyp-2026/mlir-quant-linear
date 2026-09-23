//===- quant-opt.cpp - mlir-opt with the quantize-linear pass -------------===//

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

void registerQuantizeLinearPass();

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  registerQuantizeLinearPass();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "quant-opt driver\n", registry));
}
