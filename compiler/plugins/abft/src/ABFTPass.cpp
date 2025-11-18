// ABFTPass.cpp
// Insert a computation of a ones-vector multiplied by the LHS matrix (ones * A)
// for each matched linalg.matmul operation. The ones*A for a matrix A (m x k)
// produces a row-vector of length k (tensor<?xf32>) and is implemented by
// calling a helper `column_checksum` function that sums rows per column.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
// Command line option parsing for the standalone flag used by the pass.
#include "llvm/Support/CommandLine.h"

using namespace mlir;

// Global CLI flag to enable Full-Checksum elementwise comparisons. Using a
// static command-line option avoids putting non-copyable llvm::cl::opt into
// the pass object (PassWrapper needs to be copyable).
static llvm::cl::opt<bool> abftEnableFuC(
    "abft-enable-fuc",
    llvm::cl::desc("Enable full elementwise checksum (t1/t2) comparisons"),
    llvm::cl::init(false));

namespace {

struct ABFTPass : public PassWrapper<ABFTPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ABFTPass)

  // NOTE: the CLI flag is defined as a static llvm::cl::opt above. We avoid
  // storing an Option<> member inside the pass because llvm::cl::opt is
  // non-copyable which breaks PassWrapper's cloning. Read the flag at
  // run-time from `abftEnableFuC`.

  StringRef getArgument() const final { return "abft-insert-ones"; }
  StringRef getDescription() const final {
    return "For each linalg.matmul insert a computation of ones * A (row sums)";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    ModuleOp module = func->getParentOfType<ModuleOp>();
    MLIRContext *ctx = func.getContext();

    // Ensure necessary dialects are loaded into the context so the textual
    // MLIR parser can construct ops like linalg.reduce, tensor.dim and
    // math.absf. We will try to insert full helper function bodies (so they
    // are available in the module and not turned into vm.imports). If parsing
    // fails we fall back to declaration-only functions so call sites still
    // type-check.
    ctx->getOrLoadDialect<func::FuncDialect>();
    ctx->getOrLoadDialect<linalg::LinalgDialect>();
    ctx->getOrLoadDialect<tensor::TensorDialect>();
    ctx->getOrLoadDialect<arith::ArithDialect>();
    ctx->getOrLoadDialect<math::MathDialect>();
    ctx->getOrLoadDialect<cf::ControlFlowDialect>();

    // Helper: parse and insert a helper func.func with body into the module
    // (clone). Returns the inserted func or nullptr on failure.
    auto ensureFunctionWithBody = [&](StringRef name,
                                      StringRef body) -> func::FuncOp {
      if (auto existing = module.lookupSymbol<func::FuncOp>(name))
        return existing;
      OwningOpRef<ModuleOp> tmp = parseSourceString<ModuleOp>(body, ctx);
      if (!tmp) {
        module.emitRemark() << "abft: failed to parse helper body for " << name;
        return {};
      }
      func::FuncOp srcFunc;
      for (auto f : tmp->getOps<func::FuncOp>()) {
        if (f.getSymName() == name) {
          srcFunc = f;
          break;
        }
      }
      if (!srcFunc) {
        module.emitRemark()
            << "abft: helper " << name << " not present in parsed body";
        return {};
      }
      Operation *cloned = srcFunc->clone();
      module.getBody()->getOperations().push_back(cloned);
      return cast<func::FuncOp>(cloned);
    };

    // Try to ensure helpers exist with bodies using small per-function MLIR.
    func::FuncOp parsedCol = ensureFunctionWithBody("column_checksum", R"mlir(
module {
  func.func @column_checksum(%matrix: tensor<?x?xf32>) -> tensor<?xf32> {
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32
    %n = tensor.dim %matrix, %c1 : tensor<?x?xf32>
    %empty = tensor.empty(%n) : tensor<?xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
    %sum = linalg.reduce ins(%matrix : tensor<?x?xf32>) outs(%init : tensor<?xf32>) dimensions = [0]
      (%in: f32, %acc: f32) {
        %r = arith.addf %in, %acc : f32
        linalg.yield %r : f32
      }
    return %sum : tensor<?xf32>
  }
}
)mlir");
    (void)parsedCol;

    func::FuncOp parsedRow = ensureFunctionWithBody("row_checksum", R"mlir(
module {
  func.func @row_checksum(%matrix: tensor<?x?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %m = tensor.dim %matrix, %c0 : tensor<?x?xf32>
    %empty = tensor.empty(%m) : tensor<?xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
    %sum = linalg.reduce ins(%matrix : tensor<?x?xf32>) outs(%init : tensor<?xf32>) dimensions = [1]
      (%in: f32, %acc: f32) {
        %r = arith.addf %in, %acc : f32
        linalg.yield %r : f32
      }
    return %sum : tensor<?xf32>
  }
}
)mlir");
    (void)parsedRow;

    func::FuncOp parsedSum = ensureFunctionWithBody("matrix_sum", R"mlir(
module {
  func.func @matrix_sum(%matrix: tensor<?x?xf32>) -> tensor<f32> {
    %zero = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<f32>
    %filled = linalg.fill ins(%zero : f32) outs(%init : tensor<f32>) -> tensor<f32>
    %sum = linalg.reduce ins(%matrix : tensor<?x?xf32>) outs(%filled : tensor<f32>) dimensions = [0, 1]
      (%in: f32, %acc: f32) {
        %r = arith.addf %in, %acc : f32
        linalg.yield %r : f32
      }
    return %sum : tensor<f32>
  }
}
)mlir");
    (void)parsedSum;

    func::FuncOp parsedDot =
        ensureFunctionWithBody("vector_dot_product", R"mlir(
module {
  func.func @vector_dot_product(%a: tensor<?xf32>, %b: tensor<?xf32>) -> tensor<f32> {
    %zero = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<f32>
    %filled = linalg.fill ins(%zero : f32) outs(%init : tensor<f32>) -> tensor<f32>
    %dot = linalg.dot ins(%a, %b : tensor<?xf32>, tensor<?xf32>) outs(%filled : tensor<f32>) -> tensor<f32>
    return %dot : tensor<f32>
  }
}
)mlir");
    (void)parsedDot;

    func::FuncOp parsedEps = ensureFunctionWithBody("epsilon_compare", R"mlir(
module {
  func.func @epsilon_compare(%x: tensor<f32>, %y: tensor<f32>, %eps: f32) {
    %vx = tensor.extract %x[] : tensor<f32>
    %vy = tensor.extract %y[] : tensor<f32>
    %d = arith.subf %vx, %vy : f32
    %ad = math.absf %d : f32
    %ok = arith.cmpf olt, %ad, %eps : f32
    cf.assert %ok, "FIC: values differ more than epsilon"
    return
  }
}
)mlir");
    (void)parsedEps;

    // Full-checksum helper implementations (so the pass can emit full-checksum
    // calls unconditionally). These compute:
    //  - rowvec_mul_mat(rv, mat) -> vector: for each column j, sum_k
    //  rv[k]*mat[k,j]
    //  - mat_mul_colvec(mat, cv) -> vector: for each row i, sum_j
    //  mat[i,j]*cv[j]
    //  - vector_epsilon_compare(v1, v2, eps) -> () : elementwise assert(|v1-v2|
    //  < eps)
    func::FuncOp parsedRowVec = ensureFunctionWithBody("rowvec_mul_mat", R"mlir(
module {
  func.func @rowvec_mul_mat(%rv: tensor<?xf32>, %mat: tensor<?x?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n = tensor.dim %mat, %c1 : tensor<?x?xf32>
    %empty = tensor.empty(%n) : tensor<?xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
    %res = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0)>, affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d1)>], iterator_types = ["reduction","parallel"]}
      ins(%rv, %mat : tensor<?xf32>, tensor<?x?xf32>) outs(%init : tensor<?xf32>) {
      ^bb0(%a: f32, %b: f32, %acc: f32):
        %prod = arith.mulf %a, %b : f32
        %sum = arith.addf %prod, %acc : f32
        linalg.yield %sum : f32
    } -> tensor<?xf32>
    return %res : tensor<?xf32>
  }
}
)mlir");
    (void)parsedRowVec;

    func::FuncOp parsedMatCol = ensureFunctionWithBody("mat_mul_colvec", R"mlir(
module {
  func.func @mat_mul_colvec(%mat: tensor<?x?xf32>, %cv: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %m = tensor.dim %mat, %c0 : tensor<?x?xf32>
    %empty = tensor.empty(%m) : tensor<?xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
    %res = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d1)>, affine_map<(d0,d1)->(d0)>], iterator_types = ["parallel","reduction"]}
      ins(%mat, %cv : tensor<?x?xf32>, tensor<?xf32>) outs(%init : tensor<?xf32>) {
      ^bb0(%a: f32, %b: f32, %acc: f32):
        %prod = arith.mulf %a, %b : f32
        %sum = arith.addf %prod, %acc : f32
        linalg.yield %sum : f32
    } -> tensor<?xf32>
    return %res : tensor<?xf32>
  }
}
)mlir");
    (void)parsedMatCol;

    func::FuncOp parsedVecEps =
        ensureFunctionWithBody("vector_epsilon_compare", R"mlir(
module {
  func.func @vector_epsilon_compare(%v1: tensor<?xf32>, %v2: tensor<?xf32>, %eps: f32) {
    %c0 = arith.constant 0 : index
    %n = tensor.dim %v1, %c0 : tensor<?xf32>
    %empty = tensor.empty(%n) : tensor<?xf32>
    %zero = arith.constant 0.0 : f32
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
    // create an epsilon vector of length n so it can be passed into the generic
    %eps_vec = tensor.empty(%n) : tensor<?xf32>
    %eps_filled = linalg.fill ins(%eps : f32) outs(%eps_vec : tensor<?xf32>) -> tensor<?xf32>
    linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>], iterator_types = ["parallel"]}
      ins(%v1, %v2, %eps_filled : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%init : tensor<?xf32>) {
      ^bb0(%a: f32, %b: f32, %e: f32, %acc: f32):
        %d = arith.subf %a, %b : f32
        %ad = math.absf %d : f32
        %ok = arith.cmpf olt, %ad, %e : f32
        cf.assert %ok, "FIC: vector elements differ more than epsilon"
        linalg.yield %acc : f32
    } -> tensor<?xf32>
    return
  }
}
)mlir");
    (void)parsedVecEps;

    // If any parse failed, fall back to inserting declaration-only functions
    // to keep the rest of the instrumentation working.
    OpBuilder modBuilder(module.getBodyRegion());
    Location modLoc = module.getLoc();
    auto f32 = modBuilder.getF32Type();
    auto dyn2d = RankedTensorType::get(
        {ShapedType::kDynamic, ShapedType::kDynamic}, f32);
    auto vecDyn = RankedTensorType::get({ShapedType::kDynamic}, f32);

    auto maybeInsertDecl = [&](StringRef name, FunctionType fnTy) {
      if (!module.lookupSymbol<func::FuncOp>(name)) {
        auto fn = modBuilder.create<func::FuncOp>(modLoc, name, fnTy);
        fn.setPrivate();
      }
    };

    if (!module.lookupSymbol<func::FuncOp>("column_checksum")) {
      auto ft = FunctionType::get(ctx, TypeRange{dyn2d}, TypeRange{vecDyn});
      maybeInsertDecl("column_checksum", ft);
    }
    if (!module.lookupSymbol<func::FuncOp>("row_checksum")) {
      auto ft = FunctionType::get(ctx, TypeRange{dyn2d}, TypeRange{vecDyn});
      maybeInsertDecl("row_checksum", ft);
    }
    if (!module.lookupSymbol<func::FuncOp>("matrix_sum")) {
      auto scalarTy = f32;
      auto ft = FunctionType::get(ctx, TypeRange{dyn2d}, TypeRange{scalarTy});
      maybeInsertDecl("matrix_sum", ft);
    }
    if (!module.lookupSymbol<func::FuncOp>("vector_dot_product")) {
      auto ft =
          FunctionType::get(ctx, TypeRange{vecDyn, vecDyn}, TypeRange{f32});
      maybeInsertDecl("vector_dot_product", ft);
    }
    if (!module.lookupSymbol<func::FuncOp>("epsilon_compare")) {
      auto ft = FunctionType::get(ctx, TypeRange{f32, f32, f32}, TypeRange{});
      maybeInsertDecl("epsilon_compare", ft);
    }

    // Collect matmul ops inside this function only to avoid mutating while
    // walking.
    SmallVector<Operation *, 8> targets;
    func.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "linalg.matmul")
        targets.push_back(op);
    });

    auto colFn =
        module.lookupSymbol<func::FuncOp>(StringRef("column_checksum"));
    auto rowFn = module.lookupSymbol<func::FuncOp>(StringRef("row_checksum"));
    auto sumFn = module.lookupSymbol<func::FuncOp>(StringRef("matrix_sum"));
    auto dotFn =
        module.lookupSymbol<func::FuncOp>(StringRef("vector_dot_product"));
    auto epsFn =
        module.lookupSymbol<func::FuncOp>(StringRef("epsilon_compare"));
    auto rowvecFn =
        module.lookupSymbol<func::FuncOp>(StringRef("rowvec_mul_mat"));
    auto matcolFn =
        module.lookupSymbol<func::FuncOp>(StringRef("mat_mul_colvec"));
    auto vecEpsFn =
        module.lookupSymbol<func::FuncOp>(StringRef("vector_epsilon_compare"));

    for (Operation *op : targets) {
      op->emitRemark() << "abft-ones: matched matmul for ones*A insertion";

      // Assume canonical linalg.matmul signature: inputs (A,B) outs(C)
      Value A = op->getOperand(0);
      Value B = op->getOperand(1);

      // Only handle tensor-typed A here (tensor<?x?xf32>) to keep
      // implementation simple and robust across bufferization states. For
      // memref-typed LHS we skip insertion (a future enhancement could
      // bufferize/convert).
      if (!isa<RankedTensorType>(A.getType())) {
        op->emitRemark() << "abft-ones: skipping non-tensor LHS";
        continue;
      }

      if (!colFn) {
        op->emitRemark() << "abft-ones: helper column_checksum not found; "
                            "skipping instrumentation";
        continue;
      }

      // Insert the call just before the matmul.
      OpBuilder b(op);
      Location loc = op->getLoc();

      // Ensure the A argument matches the helper's expected input type
      Type expectedArg = colFn.getFunctionType().getInput(0);
      Value callArg = A;
      if (callArg.getType() != expectedArg) {
        callArg =
            b.create<tensor::CastOp>(loc, expectedArg, callArg).getResult();
      }

      SmallVector<Type, 1> resultTypes;
      for (Type t : colFn.getFunctionType().getResults())
        resultTypes.push_back(t);

      auto call =
          b.create<func::CallOp>(loc, StringRef("column_checksum"),
                                 TypeRange(resultTypes), ValueRange{callArg});

      // Keep the call result in a local value (tmpA). We also compute the
      // row-wise checksum for B (row_checksum) which yields the per-row sums
      // of B (i.e. a k-length vector when B is k x n).
      Value tmpA = call.getResult(0);
      (void)tmpA;

      // Insert row checksum for B when it is a tensor.
      Value tmpB;
      if (rowFn && isa<RankedTensorType>(B.getType())) {
        Type expectedRowArg = rowFn.getFunctionType().getInput(0);
        Value callArgB = B;
        if (callArgB.getType() != expectedRowArg) {
          callArgB = b.create<tensor::CastOp>(loc, expectedRowArg, callArgB)
                         .getResult();
        }
        SmallVector<Type, 1> rowResultTypes;
        for (Type t : rowFn.getFunctionType().getResults())
          rowResultTypes.push_back(t);
        auto rowCall = b.create<func::CallOp>(loc, StringRef("row_checksum"),
                                              TypeRange(rowResultTypes),
                                              ValueRange{callArgB});
        tmpB = rowCall.getResult(0);
      }

      // Insert matrix sum AFTER the matmul op to compute ones * C * ones.
      // Place the insertion point just after the matmul operation.
      if (sumFn) {
        // The matmul result is the op's first result.
        if (!op->getResults().empty()) {
          Value C = op->getResult(0);
          // Build after the op.
          Block::iterator nextIt = std::next(Block::iterator(op));
          OpBuilder bAfter(op->getBlock(), nextIt);
          // Ensure C matches expected arg type.
          if (isa<RankedTensorType>(C.getType())) {
            Type expectedSumArg = sumFn.getFunctionType().getInput(0);
            Value sumArg = C;
            if (sumArg.getType() != expectedSumArg) {
              sumArg =
                  bAfter.create<tensor::CastOp>(loc, expectedSumArg, sumArg)
                      .getResult();
            }
            SmallVector<Type, 1> sumResultTypes;
            for (Type t : sumFn.getFunctionType().getResults())
              sumResultTypes.push_back(t);
            auto sumCall = bAfter.create<func::CallOp>(
                loc, StringRef("matrix_sum"), TypeRange(sumResultTypes),
                ValueRange{sumArg});
            Value totalSum = sumCall.getResult(0);
            (void)totalSum; // left in IR for later instrumentation

            // If we have both checksum vectors, compute their dot product and
            // compare with the summed up value using epsilon tolerance.
            if (dotFn && epsFn && tmpB) {
              // Ensure tmpA/tmpB match dot expected input types.
              SmallVector<Value, 2> dotArgs;
              auto dotTy = dotFn.getFunctionType();
              if (dotTy.getNumInputs() >= 1)
                dotArgs.push_back(tmpA.getType() == dotTy.getInput(0)
                                      ? tmpA
                                      : bAfter
                                            .create<tensor::CastOp>(
                                                loc, dotTy.getInput(0), tmpA)
                                            .getResult());
              if (dotTy.getNumInputs() >= 2)
                dotArgs.push_back(tmpB.getType() == dotTy.getInput(1)
                                      ? tmpB
                                      : bAfter
                                            .create<tensor::CastOp>(
                                                loc, dotTy.getInput(1), tmpB)
                                            .getResult());
              SmallVector<Type, 1> dotResults;
              for (Type t : dotTy.getResults())
                dotResults.push_back(t);
              auto dotCall = bAfter.create<func::CallOp>(
                  loc, StringRef("vector_dot_product"), TypeRange(dotResults),
                  ValueRange{dotArgs});
              Value dot_product_tensor = dotCall.getResult(0);

              // Now call epsilon_compare(totalSum, dot_product_tensor, epsilon)
              auto epsAttr = bAfter.getF32FloatAttr(12.0f);
              auto epsConst = bAfter.create<arith::ConstantOp>(
                  loc, bAfter.getF32Type(), epsAttr);
              // Ensure totalSum and dot_product_tensor match expected types.
              SmallVector<Value, 2> epsArgs;
              auto epsTy = epsFn.getFunctionType();
              if (epsTy.getNumInputs() >= 1)
                epsArgs.push_back(
                    totalSum.getType() == epsTy.getInput(0)
                        ? totalSum
                        : bAfter
                              .create<tensor::CastOp>(loc, epsTy.getInput(0),
                                                      totalSum)
                              .getResult());
              if (epsTy.getNumInputs() >= 2)
                epsArgs.push_back(
                    dot_product_tensor.getType() == epsTy.getInput(1)
                        ? dot_product_tensor
                        : bAfter
                              .create<tensor::CastOp>(loc, epsTy.getInput(1),
                                                      dot_product_tensor)
                              .getResult());
              // The epsilon scalar
              epsArgs.push_back(epsConst);
              bAfter.create<func::CallOp>(loc, StringRef("epsilon_compare"),
                                          TypeRange{}, ValueRange{epsArgs});

              // Full-checksum additional checks (controlled by
              // --abft-enable-fuc):
              if (abftEnableFuC) {
                auto epsAttrFuC = bAfter.getF32FloatAttr(1.0f);
                auto epsConstFuC = bAfter.create<arith::ConstantOp>(
                    loc, bAfter.getF32Type(), epsAttrFuC);
                // t1 = A_checksum * B  (1xK * KxN -> 1xN)
                if (rowvecFn && !rowvecFn.isExternal() && colFn &&
                    !colFn.isExternal()) {
                  SmallVector<Type, 1> t1Results;
                  for (Type t : rowvecFn.getFunctionType().getResults())
                    t1Results.push_back(t);
                  // Ensure arguments match rowvec_mul_mat expected input types.
                  SmallVector<Value, 2> t1Args;
                  auto rowvecTy = rowvecFn.getFunctionType();
                  // arg0: tmpA (vector)
                  if (rowvecTy.getNumInputs() >= 1)
                    t1Args.push_back(
                        tmpA.getType() == rowvecTy.getInput(0)
                            ? tmpA
                            : bAfter
                                  .create<tensor::CastOp>(
                                      loc, rowvecTy.getInput(0), tmpA)
                                  .getResult());
                  // arg1: B (matrix)
                  if (rowvecTy.getNumInputs() >= 2)
                    t1Args.push_back(B.getType() == rowvecTy.getInput(1)
                                         ? B
                                         : bAfter
                                               .create<tensor::CastOp>(
                                                   loc, rowvecTy.getInput(1), B)
                                               .getResult());
                  auto t1Call = bAfter.create<func::CallOp>(
                      loc, StringRef("rowvec_mul_mat"), TypeRange(t1Results),
                      ValueRange{t1Args});
                  Value t1 = t1Call.getResult(0);
                  // c1 = sum over rows of C -> column_checksum(C)
                  SmallVector<Type, 1> c1Results;
                  for (Type t : colFn.getFunctionType().getResults())
                    c1Results.push_back(t);
                  // Ensure C matches column_checksum expected input type.
                  Value c1Arg = C;
                  if (colFn) {
                    Type expectedColArg = colFn.getFunctionType().getInput(0);
                    if (c1Arg.getType() != expectedColArg)
                      c1Arg = bAfter
                                  .create<tensor::CastOp>(loc, expectedColArg,
                                                          c1Arg)
                                  .getResult();
                  }
                  auto c1Call = bAfter.create<func::CallOp>(
                      loc, StringRef("column_checksum"), TypeRange(c1Results),
                      ValueRange{c1Arg});
                  Value c1 = c1Call.getResult(0);
                  // compare elementwise t1 and c1
                  if (vecEpsFn && !vecEpsFn.isExternal()) {
                    bAfter.create<func::CallOp>(
                        loc, StringRef("vector_epsilon_compare"), TypeRange{},
                        ValueRange{t1, c1, epsConstFuC});
                  }

                  // t2 = A * B_checksum (MxK * Kx1 -> Mx1)
                  if (matcolFn && !matcolFn.isExternal() && rowFn &&
                      !rowFn.isExternal()) {
                    SmallVector<Type, 1> t2Results;
                    for (Type t : matcolFn.getFunctionType().getResults())
                      t2Results.push_back(t);
                    // Ensure arguments match mat_mul_colvec expected input
                    // types.
                    SmallVector<Value, 2> t2Args;
                    auto matcolTy = matcolFn.getFunctionType();
                    if (matcolTy.getNumInputs() >= 1)
                      t2Args.push_back(
                          A.getType() == matcolTy.getInput(0)
                              ? A
                              : bAfter
                                    .create<tensor::CastOp>(
                                        loc, matcolTy.getInput(0), A)
                                    .getResult());
                    if (matcolTy.getNumInputs() >= 2)
                      t2Args.push_back(
                          tmpB.getType() == matcolTy.getInput(1)
                              ? tmpB
                              : bAfter
                                    .create<tensor::CastOp>(
                                        loc, matcolTy.getInput(1), tmpB)
                                    .getResult());
                    auto t2Call = bAfter.create<func::CallOp>(
                        loc, StringRef("mat_mul_colvec"), TypeRange(t2Results),
                        ValueRange{t2Args});
                    Value t2 = t2Call.getResult(0);
                    // c2 = sum over columns of C -> row_checksum(C)
                    SmallVector<Type, 1> c2Results;
                    for (Type t : rowFn.getFunctionType().getResults())
                      c2Results.push_back(t);
                    // Ensure C matches row_checksum expected input type.
                    Value c2Arg = C;
                    if (rowFn) {
                      Type expectedRowArg = rowFn.getFunctionType().getInput(0);
                      if (c2Arg.getType() != expectedRowArg)
                        c2Arg = bAfter
                                    .create<tensor::CastOp>(loc, expectedRowArg,
                                                            c2Arg)
                                    .getResult();
                    }
                    auto c2Call = bAfter.create<func::CallOp>(
                        loc, StringRef("row_checksum"), TypeRange(c2Results),
                        ValueRange{c2Arg});
                    Value c2 = c2Call.getResult(0);
                    if (vecEpsFn && !vecEpsFn.isExternal()) {
                      // Use the elementwise epsilon (epsConstFuC) for vector
                      // comparison.
                      bAfter.create<func::CallOp>(
                          loc, StringRef("vector_epsilon_compare"), TypeRange{},
                          ValueRange{t2, c2, epsConstFuC});
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createABFTPass() {
  return std::make_unique<ABFTPass>();
}
static mlir::PassRegistration<ABFTPass> reg;