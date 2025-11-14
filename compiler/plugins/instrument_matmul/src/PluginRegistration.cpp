#include "iree/compiler/PluginAPI/Client.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

// Factory from InstrumentMatmul.cpp
std::unique_ptr<mlir::Pass> createInstrumentMatmulPass();

namespace mlir::iree_compiler::plugins::instrument_matmul {

struct InstrumentMatmulSession
    : public mlir::iree_compiler::PluginSession<InstrumentMatmulSession> {
  void extendPreprocessingPassPipeline(OpPassManager &pm) override {
    // Run our pass per-function at preprocessing time (when func.func exists).
    auto &funcPM = pm.nest<func::FuncOp>();
    funcPM.addPass(createInstrumentMatmulPass());
  }
};

extern "C" bool iree_register_compiler_plugin_instrument_matmul(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<InstrumentMatmulSession>("instrument_matmul");
  return true;
}

}  // namespace mlir::iree_compiler::plugins::instrument_matmul