#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#define TAMPERING false

// Row-major, contiguous for simplicity.
// C[M,N] := alpha*A[M,K] * B[K,N] + beta*C[M,N]
void hwacc_gemm_f32(const float* A, const float* B, float* C, int64_t M,
                    int64_t N, int64_t K, float alpha, float beta) {
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      float acc = 0.0f;
      const float* Ai = A + i * K;
      const float* Bj = B + j;
      for (int64_t k = 0; k < K; ++k) {
        acc += Ai[k] * Bj[k * N];
      }
      C[i * N + j] = alpha * acc + beta * C[i * N + j];
    }
  }

#if TAMPERING
  float offset = 10.0f;
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      if ((i == 0 && j == 0) || (i == 1 && j == 1)) {
        C[i * N + j] += offset;
      } else if ((i == 0 && j == 1) || (i == 1 && j == 0)) {
        C[i * N + j] -= offset;
      }
    }
  }
#endif  // TAMPERING
}

#ifdef __cplusplus
}  // extern "C"
#endif