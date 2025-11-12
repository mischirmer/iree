#include "iree/compiler/PluginAPI/Client.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

// Factory from ReplaceMatmulWithCall.cpp
std::unique_ptr<mlir::Pass> createReplaceMatmulWithCallPass();

namespace mlir::iree_compiler::plugins::hwacc {

struct ReplaceMatmulWithCallSession
    : public mlir::iree_compiler::PluginSession<ReplaceMatmulWithCallSession> {
  // Insert the pass after GlobalOptimization so that earlier high-level
  // canonicalization/generalization has completed and we are past phases that
  // consider certain bufferized ops (e.g. memref.dim) illegal. This lets the
  // pass safely introduce memref-based helper calls without tripping early
  // legality checks.
  void extendGlobalOptimizationPassPipeline(OpPassManager &pm) override {
    auto &funcPM = pm.nest<func::FuncOp>();
    funcPM.addPass(createReplaceMatmulWithCallPass());
  }
};

extern "C" bool iree_register_compiler_plugin_hwacc(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<ReplaceMatmulWithCallSession>("replace_matmul_with_call");
  return true;
}

}  // namespace mlir::iree_compiler::plugins::hwacc