func.func @simple_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "arith.mulf"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %1 = "arith.mulf"(%0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  %cst = arith.constant 3.2 : f32
  %fill = tensor.splat %cst : tensor<4xf32>
  %2 = "arith.addf"(%1, %fill) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %2 : tensor<4xf32>
}

// Calculate column-wise checksum: sum all rows per column -> tensor<?xf32>
func.func @column_checksum(%matrix: tensor<?x?xf32>) -> tensor<?xf32> {
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  
  %n = tensor.dim %matrix, %c1 : tensor<?x?xf32>
  %empty_n = tensor.empty(%n) : tensor<?xf32>
  %checksum_init = linalg.fill ins(%zero : f32)
                               outs(%empty_n : tensor<?xf32>)
                   -> tensor<?xf32>

  %checksum = linalg.reduce 
                ins(%matrix : tensor<?x?xf32>)
                outs(%checksum_init : tensor<?xf32>)
                dimensions = [0]
                (%in: f32, %out: f32) {
                  %sum = arith.addf %in, %out : f32
                  linalg.yield %sum : f32
                }
  
  return %checksum : tensor<?xf32>
}

// Calculate row-wise checksum: sum all columns per row -> tensor<?xf32>
func.func @row_checksum(%matrix: tensor<?x?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  
  %m = tensor.dim %matrix, %c0 : tensor<?x?xf32>
  %empty_m = tensor.empty(%m) : tensor<?xf32>
  %checksum_init = linalg.fill ins(%zero : f32)
                               outs(%empty_m : tensor<?xf32>)
                   -> tensor<?xf32>

  %checksum = linalg.reduce 
                ins(%matrix : tensor<?x?xf32>)
                outs(%checksum_init : tensor<?xf32>)
                dimensions = [1]
                (%in: f32, %out: f32) {
                  %sum = arith.addf %in, %out : f32
                  linalg.yield %sum : f32
                }
  
  return %checksum : tensor<?xf32>
}

// Calculate sum of all elements in a matrix -> tensor<f32>
func.func @matrix_sum(%matrix: tensor<?x?xf32>) -> tensor<f32> {
  %zero = arith.constant 0.0 : f32
  
  %sum_init = tensor.empty() : tensor<f32>
  %sum_init_filled = linalg.fill ins(%zero : f32)
                                 outs(%sum_init : tensor<f32>)
                     -> tensor<f32>

  %sum = linalg.reduce 
           ins(%matrix : tensor<?x?xf32>)
           outs(%sum_init_filled : tensor<f32>)
           dimensions = [0, 1]
           (%in: f32, %out: f32) {
             %add = arith.addf %in, %out : f32
             linalg.yield %add : f32
           }
  
  return %sum : tensor<f32>
}

// Compare two scalar values within epsilon tolerance
func.func @epsilon_compare(%val1: tensor<f32>, %val2: tensor<f32>, %epsilon: f32) {
  %val1_scalar = tensor.extract %val1[] : tensor<f32>
  %val2_scalar = tensor.extract %val2[] : tensor<f32>

  %diff = arith.subf %val1_scalar, %val2_scalar : f32
  %abs_diff = math.absf %diff : f32

  %cmp = arith.cmpf olt, %abs_diff, %epsilon : f32
  cf.assert %cmp, "FIC Values differ more than epsilon"
  
  return
}

// Calculate dot product between two vectors -> tensor<f32>
func.func @vector_dot_product(%vec1: tensor<?xf32>, %vec2: tensor<?xf32>) -> tensor<f32> {
  %zero = arith.constant 0.0 : f32
  
  %dot_init = tensor.empty() : tensor<f32>
  %dot_init_filled = linalg.fill ins(%zero : f32)
                                  outs(%dot_init : tensor<f32>)
                     -> tensor<f32>

  %dot_product = linalg.dot 
                   ins(%vec1, %vec2 : tensor<?xf32>, tensor<?xf32>)
                   outs(%dot_init_filled : tensor<f32>)
                   -> tensor<f32>
  
  return %dot_product : tensor<f32>
}

// A: tensor<MxKxf32>, B: tensor<KxNxf32>  ==>  C: tensor<MxNxf32>
func.func @matmul_dynamic(%A: tensor<?x?xf32>,
                          %B: tensor<?x?xf32>)
    -> (tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %one = arith.constant 1 : index

  // A is m x n, B is n x k
  %m   = tensor.dim %A, %c0 : tensor<?x?xf32>
  %n   = tensor.dim %A, %c1 : tensor<?x?xf32>
  %n_b = tensor.dim %B, %c0 : tensor<?x?xf32>
  %k   = tensor.dim %B, %c1 : tensor<?x?xf32>

  // Check inner dim n matches
  %same = arith.cmpi eq, %n, %n_b : index
  cf.assert %same, "matmul inner dims (n) must match"

  %zero = arith.constant 0.0 : f32

  // ---- C = A (m×n) * B (n×k) -> (m×k) ----
  %empty_mk = tensor.empty(%m, %k) : tensor<?x?xf32>
  %Cinit    = linalg.fill ins(%zero : f32)
                       outs(%empty_mk : tensor<?x?xf32>)
              -> tensor<?x?xf32>

  %C = linalg.matmul
         ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
         outs(%Cinit : tensor<?x?xf32>)
         -> tensor<?x?xf32>

  // ---- Calculate checksums using helper functions ----
  %A_checksum = func.call @column_checksum(%A) : (tensor<?x?xf32>) -> tensor<?xf32>
  %row_sums_B = func.call @row_checksum(%B) : (tensor<?x?xf32>) -> tensor<?xf32>

  // ---- Dot product between checksum vectors using helper function ----
  %dot_product_tensor = func.call @vector_dot_product(%A_checksum, %row_sums_B) : (tensor<?xf32>, tensor<?xf32>) -> tensor<f32>


  // ---- Sum all elements of C using helper function ----
  %C_sum = func.call @matrix_sum(%C) : (tensor<?x?xf32>) -> tensor<f32>


  %epsilon = arith.constant 1.0e-5 : f32

  // ---- Validate that C_sum and dot_product are within epsilon tolerance ----
  func.call @epsilon_compare(%C_sum, %dot_product_tensor, %epsilon) : (tensor<f32>, tensor<f32>, f32) -> ()

  return %C : tensor<?x?xf32>
}
