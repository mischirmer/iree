// Minimal export list for hwacc module.
// Keep the functions sorted by name.

// clang-format off

// Return a ref (r) containing the result buffer/view. The module imports
// expect the function to return a ref (tensor result), so ensure our export
// signature matches: rrrIIIffI -> r
EXPORT_FN("hwacc_gemm_f32", iree_hwacc_module_hwacc_gemm_f32, rrrIIIffI, r)

// clang-format on
