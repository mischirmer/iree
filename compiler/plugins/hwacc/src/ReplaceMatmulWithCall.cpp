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

#include <cstdlib>
#include <set>
#include <string>
#include <sstream>
#include <cerrno>

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

    // Add an extra index (tamper flag) parameter at the end so callers can
    // request runtime-controlled tampering (0 = no, non-zero = yes).
    auto fnType = FunctionType::get(
        ctx,
        TypeRange{mem2D, mem2D, mem2D, idx, idx, idx, f32, f32, idx},
        TypeRange{});

    auto fn = b.create<func::FuncOp>(loc, name, fnType);
    fn.setPrivate();
    return fn;
  }

  // tensor version: takes (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>,
  // index, index, index, f32, f32) -> tensor<?x?xf32>
  auto tensor2D = RankedTensorType::get({ShapedType::kDynamic, ShapedType::kDynamic}, f32);
  // Tensor version: append an index argument for the tamper flag as well.
  auto fnType = FunctionType::get(ctx,
                                  TypeRange{tensor2D, tensor2D, tensor2D, idx, idx, idx, f32, f32, idx},
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

  // Parse IREE_HWACC_TAMPER env var once. Expected format: "[x,y,z]" or
  // "x,y,z" (commas, optional surrounding brackets). The values refer to
  // the matmul indices in the order they are encountered (0-based `seen`
  // counter). This lets callers specify which matmuls (by occurrence
  // order in the function) should be tampered with regardless of whether
  // earlier matmuls were actually replaced.
    std::set<int64_t> tamperSet;
    if (const char* env = std::getenv("IREE_HWACC_TAMPER")) {
      std::string s(env);
      // strip surrounding brackets if present
      if (!s.empty() && s.front() == '[' && s.back() == ']') {
        s = s.substr(1, s.size() - 2);
      }
      std::stringstream ss(s);
      while (ss.good()) {
        std::string token;
        if (!std::getline(ss, token, ',')) break;
        // trim whitespace
        size_t a = 0;
        while (a < token.size() && isspace((unsigned char)token[a])) ++a;
        size_t b = token.size();
        while (b > a && isspace((unsigned char)token[b - 1])) --b;
        if (b > a) {
          // Parse integer without exceptions (exceptions are disabled in
          // this build). Use C-style strtoll and validate the entire token
          // was consumed.
          std::string num = token.substr(a, b - a);
          char *endptr = nullptr;
          errno = 0;
          long long v = std::strtoll(num.c_str(), &endptr, 10);
          if (endptr != num.c_str() && *endptr == '\0' && errno == 0) {
            tamperSet.insert(static_cast<int64_t>(v));
          }
        }
      }
    }

    // Parse optional IREE_HWACC_FORCE_REPLACE (boolean). When set to a
    // non-zero value, allow replacing all matmuls (subject to other checks).
    bool forceReplaceAll = false;
    if (const char* envForce = std::getenv("IREE_HWACC_FORCE_REPLACE")) {
      char *endptr = nullptr;
      errno = 0;
      long long v = std::strtoll(envForce, &endptr, 10);
      if (endptr != envForce && *endptr == '\0' && errno == 0 && v != 0) {
        forceReplaceAll = true;
      }
    }

    // Parse optional IREE_HWACC_FORCE_REPLACE_LIST env var specifying a
    // comma-separated list of matmul `seen` indices which should be forced
    // replaced even if their outputs are directly returned by the function
    // or when `IREE_HWACC_FORCE_REPLACE` is not set. Format: "[0,2]" or
    // "0,2". If not provided the set remains empty.
    std::set<int64_t> forceReplaceSet;
    if (const char* env = std::getenv("IREE_HWACC_FORCE_REPLACE_LIST")) {
      std::string s(env);
      if (!s.empty() && s.front() == '[' && s.back() == ']') {
        s = s.substr(1, s.size() - 2);
      }
      std::stringstream ss(s);
      while (ss.good()) {
        std::string token;
        if (!std::getline(ss, token, ',')) break;
        // trim whitespace
        size_t a = 0;
        while (a < token.size() && isspace((unsigned char)token[a])) ++a;
        size_t b = token.size();
        while (b > a && isspace((unsigned char)token[b - 1])) --b;
        if (b > a) {
          std::string num = token.substr(a, b - a);
          char *endptr = nullptr;
          errno = 0;
          long long v = std::strtoll(num.c_str(), &endptr, 10);
          if (endptr != num.c_str() && *endptr == '\0' && errno == 0) {
            forceReplaceSet.insert(static_cast<int64_t>(v));
          }
        }
      }
    }

    SmallVector<linalg::MatmulOp> mats;
    func.walk([&](linalg::MatmulOp mm) { mats.push_back(mm); });

  int replaced = 0;
  int seen = 0;
    for (auto mm : mats) {
      // Expect ins(%A,%B) outs(%C)
      Value A = mm.getInputs()[0];
      Value B = mm.getInputs()[1];
      Value C = mm.getOutputs()[0];

  // Debug: emit a remark for each matmul encountered showing whether it
  // reaches a function return (so we can understand why replacements are
  // skipped).
    int thisIdx = seen++;
    mlir::emitRemark(mm.getLoc()) << "[hwacc-replace] seen_matmul_idx=" << thisIdx;

    // Quick check: if the matmul's output value flows to a func.return (i.e.
    // contributes to the function result), skip replacing it. However, if
    // this matmul's seen index is present in `forceReplaceSet`, force the
    // replacement even when it would otherwise be skipped.
    auto reachesReturn = [&](Value v) -> bool {
      for (Operation *userOp : v.getUsers()) {
        if (isa<func::ReturnOp>(userOp)) return true;
      }
      return false;
    };
    bool doesReach = reachesReturn(C);
    // Decision logic:
    // - If the matmul result reaches a func.return, only replace it when it
    //   is explicitly listed in `forceReplaceSet`.
    // - Otherwise (not reaching return): replace only when either
    //   `forceReplaceAll` is true or the index is in `forceReplaceSet`.
    if (doesReach) {
      if (!forceReplaceSet.count(thisIdx)) {
        mlir::emitRemark(mm.getLoc()) << "[hwacc-replace] skipping_matmul_idx=" << thisIdx << " (reaches_return, not_forced)";
        continue;
      }
      mlir::emitRemark(mm.getLoc()) << "[hwacc-replace] replacing_matmul_idx=" << thisIdx << " (forced despite reaching_return)";
    } else {
      if (!forceReplaceAll && !forceReplaceSet.count(thisIdx)) {
        mlir::emitRemark(mm.getLoc()) << "[hwacc-replace] skipping_matmul_idx=" << thisIdx << " (not_forced)";
        continue;
      }
      mlir::emitRemark(mm.getLoc()) << "[hwacc-replace] replacing_matmul_idx=" << thisIdx;
    }

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

          // Pass tamper flag: set to 1 when this matmul's seen index is in
          // the tamper set parsed from IREE_HWACC_TAMPER, otherwise 0.
          // We use the `seen` ordering index (thisIdx) so the environment
          // variable targets matmuls by their occurrence in the function,
          // independent of whether earlier matmuls were skipped or replaced.
          bool doTamper = tamperSet.count(thisIdx) > 0;
          Value tamperFlag = b.create<arith::ConstantIndexOp>(loc, (int64_t)(doTamper ? 1 : 0));
          // Value tamperFlag = b.create<arith::ConstantIndexOp>(loc, (int64_t)(0));
          b.create<func::CallOp>(loc, callee.getSymName(), TypeRange{},
                                 ValueRange{A_arg, B_arg, C_arg, M, N, K, alpha, beta, tamperFlag});

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
          // Pass tamper flag: set to 1 when this matmul's seen index is in
          // the tamper set parsed from IREE_HWACC_TAMPER, otherwise 0.
          bool doTamper = tamperSet.count(thisIdx) > 0;
          Value tamperFlag = b.create<arith::ConstantIndexOp>(loc, (int64_t)(doTamper ? 1 : 0));
          auto call = b.create<func::CallOp>(loc, callee.getSymName(), callResultTypes,
                                             ValueRange{A_arg, B_arg, C_arg, M, N, K, alpha, beta, tamperFlag});
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