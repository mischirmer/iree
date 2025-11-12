// ReplaceMatmulWithCall.cpp
#include "llvm/Support/Casting.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
// for tensor::DimOp
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

namespace {

// Create (or find) a declaration for:
//   func private @hwacc_gemm_f32(
//     memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>,
//     index, index, index, f32, f32)
// Create (or find) a declaration for hwacc gemm. If `forTensors` is true
// create a tensor-typed version that returns a tensor result. Otherwise
// create the memref-typed version that returns void (buffer-based).
static func::FuncOp ensureHwaccDecl(ModuleOp module, bool forTensors = false) {
  // Emit a module-qualified runtime-visible name so VM imports are scoped.
  // The VM bytecode writer requires import names to contain a '.' (module.func).
  // We use the module name 'hwacc' and function 'hwacc_gemm_f32'.
  StringRef name = "hwacc.hwacc_gemm_f32";
  if (auto f = module.lookupSymbol<func::FuncOp>(name)) return f;

  MLIRContext *ctx = module.getContext();
  OpBuilder b(module.getBodyRegion());
  auto loc = b.getUnknownLoc();

  auto f32 = b.getF32Type();
  auto idx = b.getIndexType();
  if (!forTensors) {
    auto mem2D = MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, f32);

    auto fnType = FunctionType::get(
        ctx,
        TypeRange{mem2D, mem2D, mem2D, idx, idx, idx, f32, f32},
        TypeRange{});

    auto fn = b.create<func::FuncOp>(loc, name, fnType);
    fn.setPrivate();
    return fn;
  }

  // tensor version: takes (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>,
  // index, index, index, f32, f32) -> tensor<?x?xf32>
  auto tensor2D = RankedTensorType::get({ShapedType::kDynamic, ShapedType::kDynamic}, f32);
  auto fnType = FunctionType::get(ctx,
                                  TypeRange{tensor2D, tensor2D, tensor2D, idx, idx, idx, f32, f32},
                                  TypeRange{tensor2D});

  auto fn = b.create<func::FuncOp>(loc, name, fnType);
  fn.setPrivate();
  return fn;
}

struct ReplaceMatmulWithCallPass
    : public PassWrapper<ReplaceMatmulWithCallPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceMatmulWithCallPass)

  StringRef getArgument() const final { return "hwacc-replace-matmul-with-call"; }
  StringRef getDescription() const final {
    return "Replace linalg.matmul (memref,f32) with a call to @hwacc_gemm_f32";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect,
                    func::FuncDialect,
                    linalg::LinalgDialect,
                    memref::MemRefDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
  ModuleOp module = func->getParentOfType<ModuleOp>();

    SmallVector<linalg::MatmulOp> mats;
    func.walk([&](linalg::MatmulOp mm) { mats.push_back(mm); });

    int replaced = 0;
    for (auto mm : mats) {
      // Expect ins(%A,%B) outs(%C)
      Value A = mm.getInputs()[0];
      Value B = mm.getInputs()[1];
      Value C = mm.getOutputs()[0];

      OpBuilder b(mm);
      Location loc = mm.getLoc();
      auto f32 = b.getF32Type();

      // First try memref path (bufferized matmul)
      if (auto aTy = dyn_cast<MemRefType>(A.getType())) {
        auto bTy = dyn_cast<MemRefType>(B.getType());
        auto cTy = dyn_cast<MemRefType>(C.getType());
        if (bTy && cTy && aTy.getRank() == 2 && bTy.getRank() == 2 && cTy.getRank() == 2 &&
            aTy.getElementType().isF32() && bTy.getElementType().isF32() && cTy.getElementType().isF32()) {

          // Dimensions as index (MemRef::DimOp returns index)
          auto dim = [&](Value mem, int64_t d) -> Value {
            return b.create<memref::DimOp>(loc, mem, d);
          };
          Value M = dim(A, 0);
          Value K = dim(A, 1);
          Value N = dim(B, 1);

          // alpha=1.0, beta=0.0 initially
          Value alpha = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(1.0f));
          Value beta  = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.0f));

          // Ensure callee exists (may have been DCE'd otherwise).
          auto callee = module.lookupSymbol<func::FuncOp>("hwacc_gemm_f32");
          if (!callee) callee = ensureHwaccDecl(module, /*forTensors=*/false);

          // Ensure operand types match the callee param types. If the
          // memref operands have static dimensions we insert casts to the
          // expected dynamic memref type so the call type-checks.
          MemRefType expectedMemTy = MemRefType::get(
              {ShapedType::kDynamic, ShapedType::kDynamic}, f32);
          Value A_arg = A;
          Value B_arg = B;
          Value C_arg = C;
          if (A.getType() != expectedMemTy) {
            A_arg = b.create<memref::CastOp>(loc, expectedMemTy, A);
          }
          if (B.getType() != expectedMemTy) {
            B_arg = b.create<memref::CastOp>(loc, expectedMemTy, B);
          }
          if (C.getType() != expectedMemTy) {
            C_arg = b.create<memref::CastOp>(loc, expectedMemTy, C);
          }

          b.create<func::CallOp>(loc, callee.getSymName(), TypeRange{},
                                 ValueRange{A_arg, B_arg, C_arg, M, N, K, alpha, beta});

          // In buffer world, matmul has no results; just erase it.
          mm.erase();
          ++replaced;
          continue;
        }
      }

      // Then try tensor path
      if (auto aTy = dyn_cast<RankedTensorType>(A.getType())) {
        auto bTy = dyn_cast<RankedTensorType>(B.getType());
        auto cTy = dyn_cast<RankedTensorType>(C.getType());
        if (bTy && cTy && aTy.getRank() == 2 && bTy.getRank() == 2 && cTy.getRank() == 2 &&
            aTy.getElementType().isF32() && bTy.getElementType().isF32() && cTy.getElementType().isF32()) {

          // Dimensions as index (tensor::DimOp returns index)
          auto dim = [&](Value mem, int64_t d) -> Value {
            return b.create<tensor::DimOp>(loc, mem, d);
          };
          Value M = dim(A, 0);
          Value K = dim(A, 1);
          Value N = dim(B, 1);

          // alpha=1.0, beta=0.0 initially
          Value alpha = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(1.0f));
          Value beta  = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.0f));

          // Ensure tensor callee exists (we unify the name to hwacc_gemm_f32).
          auto callee = module.lookupSymbol<func::FuncOp>("hwacc_gemm_f32");
          if (!callee) callee = ensureHwaccDecl(module, /*forTensors=*/true);

          // The callee expects tensor<?x?xf32> params; insert tensor.cast
          // ops when callers have more specific static shapes so the call
          // type-checks.
          RankedTensorType expectedTensorTy =
              RankedTensorType::get({ShapedType::kDynamic, ShapedType::kDynamic}, f32);
          Value A_arg = A;
          Value B_arg = B;
          Value C_arg = C;
          if (A.getType() != expectedTensorTy) {
            A_arg = b.create<tensor::CastOp>(loc, expectedTensorTy, A);
          }
          if (B.getType() != expectedTensorTy) {
            B_arg = b.create<tensor::CastOp>(loc, expectedTensorTy, B);
          }
          if (C.getType() != expectedTensorTy) {
            C_arg = b.create<tensor::CastOp>(loc, expectedTensorTy, C);
          }

          // Call must use the callee-declared result type (e.g. tensor<?x?xf32>).
          // After the call, cast the dynamic result back to the concrete
          // cTy expected by original users and replace uses.
          auto calleeFuncTy = callee.getFunctionType();
          TypeRange callResultTypes = calleeFuncTy.getResults();
          auto call = b.create<func::CallOp>(loc, callee.getSymName(), callResultTypes,
                                             ValueRange{A_arg, B_arg, C_arg, M, N, K, alpha, beta});
          // Replace uses of the matmul result with a cast of the call result
          // to the original concrete result type, then erase the matmul.
          if (!mm->getResults().empty()) {
            Value callRes = call.getResult(0);
            if (callRes.getType() != cTy) {
              auto castRes = b.create<tensor::CastOp>(loc, cTy, callRes);
              mm->getResult(0).replaceAllUsesWith(castRes.getResult());
            } else {
              mm->getResult(0).replaceAllUsesWith(callRes);
            }
          }
          mm.erase();
          ++replaced;
          continue;
        }
      }
      // Otherwise not a supported matmul type; skip.
    }

    if (replaced)
      mlir::emitRemark(func.getLoc())
          << "[hwacc-replace-matmul-with-call] replaced=" << replaced;
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createReplaceMatmulWithCallPass() {
  return std::make_unique<ReplaceMatmulWithCallPass>();
}

// Register using the zero-arg form (name/desc come from the pass methods).
static mlir::PassRegistration<ReplaceMatmulWithCallPass> reg;