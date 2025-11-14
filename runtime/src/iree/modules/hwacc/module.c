// Minimal HWACC native module that exposes a single function hwacc_gemm_f32.
// It marshals VM buffer arguments into host pointers and calls the existing
// runtime function hwacc_gemm_f32 (provided in runtime/hwacc_gemm.cpp).

#include "iree/modules/hwacc/module.h"

#include <string.h>

#include "iree/base/api.h"
#include "iree/vm/api.h"
#include "iree/hal/allocator.h"
#include "iree/hal/buffer.h"
#include "iree/hal/buffer_view.h"
// Pull in the HAL module helpers (including move-ref helpers that convert
// iree_hal_buffer_view_t* -> iree_vm_ref_t). Other callers use
// iree/modules/hal/module.h to get these helpers; include it here so
// iree_hal_buffer_view_move_ref is declared.
#include "iree/modules/hal/module.h"

// Declare external implementation we already have in the runtime tree.
// Signature: void hwacc_gemm_f32(const float* A, const float* B, float* C,
//                                 int64_t M, int64_t N, int64_t K,
//                                 float alpha, float beta,
//                                 int64_t tamper_flag);
extern void hwacc_gemm_f32(const float* A, const float* B, float* C,
                           int64_t M, int64_t N, int64_t K,
                           float alpha, float beta, int64_t tamper_flag);

#define IREE_HWACC_MODULE_VERSION_0_0 0x00000000u
#define IREE_HWACC_MODULE_VERSION_LATEST IREE_HWACC_MODULE_VERSION_0_0

// Define a custom ABI struct for (rrr, I I I, f f) so we can use the shim
// generation utilities for our specific signature: r0,r1,r2,i3,i4,i5,f6,f7.
IREE_VM_ABI_FIXED_STRUCT(rrrIIIffI, {
  iree_vm_ref_t r0;
  iree_vm_ref_t r1;
  iree_vm_ref_t r2;
  int64_t i3;
  int64_t i4;
  int64_t i5;
  float f6;
  float f7;
  int64_t i8;
});

// Define the shim that will marshal VM args/results into the target call.
// We return a ref (r) to match the module's import signature.
IREE_VM_ABI_DEFINE_SHIM(rrrIIIffI, r);

typedef struct iree_hwacc_module_t {
  iree_allocator_t host_allocator;
} iree_hwacc_module_t;

#define IREE_HWACC_MODULE_CAST(module) \
  (iree_hwacc_module_t*)((uint8_t*)(module) + iree_vm_native_module_size())

typedef struct iree_hwacc_module_state_t {
  iree_allocator_t host_allocator;
} iree_hwacc_module_state_t;

static void IREE_API_PTR iree_hwacc_module_destroy(void* base_module) {
  iree_hwacc_module_t* module = IREE_HWACC_MODULE_CAST(base_module);
  (void)module;
}

static iree_status_t IREE_API_PTR iree_hwacc_module_alloc_state(
    void* self, iree_allocator_t host_allocator,
    iree_vm_module_state_t** out_module_state) {
  iree_hwacc_module_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*state),
                                            (void**)&state));
  memset(state, 0, sizeof(*state));
  state->host_allocator = host_allocator;
  *out_module_state = (iree_vm_module_state_t*)state;
  return iree_ok_status();
}

static void IREE_API_PTR iree_hwacc_module_free_state(
    void* self, iree_vm_module_state_t* module_state) {
  iree_hwacc_module_state_t* state = (iree_hwacc_module_state_t*)module_state;
  iree_allocator_free(state->host_allocator, state);
}

static iree_status_t IREE_API_PTR iree_hwacc_module_fork_state(
    void* self, iree_vm_module_state_t* parent_state,
    iree_allocator_t host_allocator, iree_vm_module_state_t** out_child_state) {
  // For this simple module just allocate new empty state.
  return iree_hwacc_module_alloc_state(self, host_allocator, out_child_state);
}

static iree_status_t IREE_API_PTR iree_hwacc_module_notify(
    void* self, iree_vm_module_state_t* module_state, iree_vm_signal_t signal) {
  (void)self;
  (void)module_state;
  (void)signal;
  return iree_ok_status();
}

// Exported function shim. Calling convention declared in exports.inl as
// rrrIIIff -> (r0,r1,r2,i3,i4,i5,f6,f7) -> r
// We support two variants:
//  - Caller supplies an output buffer in r2: we write into it and return it
//    (rets->r0 is a retained reference to the provided buffer).
//  - Caller provides a null r2: we allocate a new VM buffer, write into it,
//    and return that new buffer.
IREE_VM_ABI_EXPORT(iree_hwacc_module_hwacc_gemm_f32, iree_hwacc_module_state_t,
                   rrrIIIffI, r) {
  // Args:
  //  r0: A buffer (input)
  //  r1: B buffer (input)
  //  r2: C buffer (output) or null
  //  i3: M
  //  i4: N
  //  i5: K
  //  f6: alpha
  //  f7: beta
  //  i8: tamper_flag

  // --------------------------------------------------------------------------
  // Fetch buffer_views for A, B, C (C may be null).
  // --------------------------------------------------------------------------
  iree_hal_buffer_view_t* a_view = NULL;
  iree_hal_buffer_view_t* b_view = NULL;
  iree_hal_buffer_view_t* c_view = NULL;  // may be NULL if caller didn't provide

  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_check_deref(args->r0, &a_view));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_check_deref(args->r1, &b_view));
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_view_check_deref_or_null(args->r2, &c_view));

  int64_t M = args->i3;
  int64_t N = args->i4;
  int64_t K = args->i5;
  float alpha = args->f6;
  float beta = args->f7;

  // --------------------------------------------------------------------------
  // Map A and B over their entire underlying buffers.
  // (Matches IREE's own custom-module examples.)
  // --------------------------------------------------------------------------
  iree_hal_buffer_t* a_buf = iree_hal_buffer_view_buffer(a_view);
  iree_hal_buffer_t* b_buf = iree_hal_buffer_view_buffer(b_view);
  // Debug: print buffer/view metadata to help diagnose mismatches.
  {
    // iree_device_size_t a_blen = iree_hal_buffer_view_byte_length(a_view);
    // iree_device_size_t b_blen = iree_hal_buffer_view_byte_length(b_view);
    // iree_hal_element_type_t a_et = iree_hal_buffer_view_element_type(a_view);
    // iree_hal_element_type_t b_et = iree_hal_buffer_view_element_type(b_view);
    // iree_device_size_t a_buf_off = iree_hal_buffer_byte_offset(a_buf);
    // iree_device_size_t b_buf_off = iree_hal_buffer_byte_offset(b_buf);
    iree_host_size_t a_rank = iree_hal_buffer_view_shape_rank(a_view);
    iree_host_size_t b_rank = iree_hal_buffer_view_shape_rank(b_view);
    /*printf("[hwacc_debug] A: elem_type=%u byte_len=%lld rank=%lld\n",
           (unsigned)a_et, (long long)a_blen, (long long)a_rank);
    printf("[hwacc_debug] B: elem_type=%u byte_len=%lld rank=%lld\n",
           (unsigned)b_et, (long long)b_blen, (long long)b_rank);
    printf("[hwacc_debug] A buffer byte_offset=%lld B buffer byte_offset=%lld\n",
           (long long)a_buf_off, (long long)b_buf_off);*/
    if (a_rank > 0) {
      for (iree_host_size_t i = 0; i < a_rank; ++i) {
        printf("[hwacc_debug] A.shape[%lld]=%lld\n", (long long)i,
               (long long)iree_hal_buffer_view_shape_dim(a_view, i));
      }
    }
    if (b_rank > 0) {
      for (iree_host_size_t i = 0; i < b_rank; ++i) {
        printf("[hwacc_debug] B.shape[%lld]=%lld\n", (long long)i,
               (long long)iree_hal_buffer_view_shape_dim(b_view, i));
      }
    }
  }

  // Map only the exact byte ranges backing the buffer_views so we get a
  // pointer directly to the view data (handles cases where the buffer view
  // wraps a larger buffer allocation).
  iree_hal_buffer_mapping_t a_mapping = {{0}};
  iree_hal_buffer_mapping_t b_mapping = {{0}};

  iree_device_size_t a_byte_length = iree_hal_buffer_view_byte_length(a_view);
  iree_device_size_t b_byte_length = iree_hal_buffer_view_byte_length(b_view);

  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
    a_buf, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
    /*byte_offset=*/0, /*byte_length=*/a_byte_length, &a_mapping));

  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
    b_buf, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
    /*byte_offset=*/0, /*byte_length=*/b_byte_length, &b_mapping));

  /*printf("[hwacc_debug] mapped A ptr=%p len=%lld mapped B ptr=%p len=%lld\n",
         a_mapping.contents.data, (long long)a_mapping.contents.data_length,
         b_mapping.contents.data, (long long)b_mapping.contents.data_length);*/

  const float* A = (const float*)a_mapping.contents.data;
  const float* B = (const float*)b_mapping.contents.data;

  // Expected element count for the result.
  iree_host_size_t element_count = (iree_host_size_t)M * (iree_host_size_t)N;
  iree_host_size_t byte_length = element_count * sizeof(float);

  // --------------------------------------------------------------------------
  // Case 1: Caller provided an output buffer_view (C).
  // --------------------------------------------------------------------------
  if (c_view) {
    iree_hal_buffer_t* out_hal_buf = iree_hal_buffer_view_buffer(c_view);

    iree_hal_buffer_mapping_t out_mapping = {{0}};
    iree_device_size_t c_byte_length = iree_hal_buffer_view_byte_length(c_view);
    // iree_device_size_t c_buf_off = iree_hal_buffer_byte_offset(out_hal_buf);
    IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
        out_hal_buf, IREE_HAL_MAPPING_MODE_SCOPED,
        IREE_HAL_MEMORY_ACCESS_ALL,
        /*byte_offset=*/0, /*byte_length=*/c_byte_length, &out_mapping));

    float* C = (float*)out_mapping.contents.data;

    // GEMM: C(MxN) = alpha * A(MxK) * B(KxN) + beta * C(MxN)
    hwacc_gemm_f32(A, B, C, M, N, K, alpha, beta, args->i8);

    IREE_RETURN_IF_ERROR(iree_hal_buffer_unmap_range(&out_mapping));

    // Return the same buffer_view back to the caller.
    iree_hal_buffer_view_retain(c_view);
    rets->r0 = iree_hal_buffer_view_move_ref(c_view);

  } else {
    // ------------------------------------------------------------------------
    // Case 2: No output buffer_view provided.
    //         Allocate a host-local buffer + buffer_view and return that.
    // ------------------------------------------------------------------------
    iree_hal_allocator_t* heap_allocator = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_allocator_create_heap(
        iree_make_cstring_view("hwacc_heap"), state->host_allocator,
        state->host_allocator, &heap_allocator));

    iree_hal_buffer_params_t params = {0};
    params.usage = IREE_HAL_BUFFER_USAGE_MAPPING |
                   IREE_HAL_BUFFER_USAGE_TRANSFER |
                   IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
    params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
    iree_hal_buffer_params_canonicalize(&params);

    iree_hal_buffer_t* hal_buffer = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
        heap_allocator, params, (iree_device_size_t)byte_length, &hal_buffer));

    iree_hal_buffer_mapping_t mapping = {{0}};
    IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
        hal_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
        IREE_HAL_MEMORY_ACCESS_ALL,
        0, IREE_HAL_WHOLE_BUFFER, &mapping));

    float* C = (float*)mapping.contents.data;

    // NOTE: if you ever set beta != 0 here, you must initialize C first.
    hwacc_gemm_f32(A, B, C, M, N, K, alpha, beta, args->i8);

    IREE_RETURN_IF_ERROR(iree_hal_buffer_unmap_range(&mapping));

    // Wrap in a buffer_view with [M, N] row-major shape.
    iree_hal_buffer_view_t* buffer_view = NULL;
    iree_hal_dim_t shape[2];
    shape[0] = (iree_hal_dim_t)M;
    shape[1] = (iree_hal_dim_t)N;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_view_create(
        hal_buffer,
        /*shape_rank=*/2, shape,
        /*element_type=*/IREE_HAL_ELEMENT_TYPE_FLOAT_32,
        /*encoding_type=*/IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        state->host_allocator, &buffer_view));

    iree_hal_buffer_release(hal_buffer);
    iree_hal_allocator_release(heap_allocator);

    rets->r0 = iree_hal_buffer_view_move_ref(buffer_view);
  }

  // --------------------------------------------------------------------------
  // Unmap inputs and return.
  // --------------------------------------------------------------------------
  iree_status_t status_unmap_a = iree_hal_buffer_unmap_range(&a_mapping);
  iree_status_t status_unmap_b = iree_hal_buffer_unmap_range(&b_mapping);
  if (!iree_status_is_ok(status_unmap_a)) return status_unmap_a;
  if (!iree_status_is_ok(status_unmap_b)) return status_unmap_b;

  return iree_ok_status();
}

// NOTE: order of exports must match exports.inl.
static const iree_vm_native_function_ptr_t iree_hwacc_module_funcs_[] = {
#define EXPORT_FN(name, target_fn, arg_types, ret_types)               \
  {                                                                     \
      .shim = (iree_vm_native_function_shim_t)                          \
          iree_vm_shim_##arg_types##_##ret_types,                       \
      .target = (iree_vm_native_function_target_t)(target_fn),          \
  },
#include "iree/modules/hwacc/exports.inl"
#undef EXPORT_FN
};

static const iree_vm_native_import_descriptor_t iree_hwacc_module_imports_[1];

static const iree_vm_native_export_descriptor_t iree_hwacc_module_exports_[] = {
#define EXPORT_FN(name, target_fn, arg_types, ret_types)               \
  {                                                                     \
      .local_name = iree_string_view_literal(name),                     \
      .calling_convention =                                            \
          iree_string_view_literal("0" #arg_types "_" #ret_types),  \
      .attr_count = 0,                                                  \
      .attrs = NULL,                                                    \
  },
#include "iree/modules/hwacc/exports.inl"
#undef EXPORT_FN
};

static const iree_vm_native_module_descriptor_t iree_hwacc_module_descriptor_ = {
    .name = iree_string_view_literal("hwacc"),
    .version = IREE_HWACC_MODULE_VERSION_LATEST,
    .attr_count = 0,
    .attrs = NULL,
    .dependency_count = 0,
    .dependencies = NULL,
    .import_count = 0,
    .imports = iree_hwacc_module_imports_,
    .export_count = IREE_ARRAYSIZE(iree_hwacc_module_exports_),
    .exports = iree_hwacc_module_exports_,
    .function_count = IREE_ARRAYSIZE(iree_hwacc_module_funcs_),
    .functions = iree_hwacc_module_funcs_,
};

IREE_API_EXPORT iree_status_t iree_hwacc_module_create(
    iree_vm_instance_t* instance, iree_allocator_t host_allocator,
    iree_vm_module_t** IREE_RESTRICT out_module) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;

  static const iree_vm_module_t interface = {
      .destroy = iree_hwacc_module_destroy,
      .alloc_state = iree_hwacc_module_alloc_state,
      .free_state = iree_hwacc_module_free_state,
      .fork_state = iree_hwacc_module_fork_state,
      .notify = iree_hwacc_module_notify,
  };

  iree_vm_module_t* base_module = NULL;
  iree_host_size_t total_size = iree_vm_native_module_size() + sizeof(iree_hwacc_module_t);
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, total_size, (void**)&base_module));
  memset(base_module, 0, total_size);

  iree_status_t status = iree_vm_native_module_initialize(&interface,
                                                         &iree_hwacc_module_descriptor_,
                                                         instance, host_allocator,
                                                         base_module);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, base_module);
    return status;
  }

  iree_hwacc_module_t* module = IREE_HWACC_MODULE_CAST(base_module);
  module->host_allocator = host_allocator;

  *out_module = base_module;
  return iree_ok_status();
}
