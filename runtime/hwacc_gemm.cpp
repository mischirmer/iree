#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <time.h>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdio.h>

/* Global storage for scaling factors (C linkage) */
float* g_scaling_factors_A = NULL;
float* g_scaling_factors_B = NULL;
int64_t g_scaling_factors_count_A = 0;
int64_t g_scaling_factors_count_B = 0;
int g_seeded = 0;

void scale_A(const float* A, float* A_modified, int64_t M, int64_t K) {
  for (int64_t i = 0; i < M; ++i) {
    float scaling_factor = ((float)rand() / (float)RAND_MAX) * 4.0f - 2.0f;
    g_scaling_factors_A[i] = scaling_factor;
    const float* Ai = A + i * K;
    for (int64_t k = 0; k < K; ++k) {
      A_modified[i * K + k] = Ai[k] * scaling_factor;
    }
  }
}

void scale_B(const float* B, float* B_modified, int64_t K, int64_t N) {
  for (int64_t i = 0; i < N; ++i) {
    float scaling_factor = ((float)rand() / (float)RAND_MAX) * 4.0f - 2.0f;
    g_scaling_factors_B[i] = scaling_factor;
    const float* Bi = B + i;
    for (int64_t k = 0; k < K; ++k) {
      B_modified[k * N + i] = Bi[k * N] * scaling_factor;
    }
  }
}

void descale_C(float* C, int64_t M, int64_t N) {
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      C[i * N + j] =
          C[i * N + j] / g_scaling_factors_A[i] / g_scaling_factors_B[j];
    }
  }
}

float dot_product(const std::vector<float>& A_col_checksums,
                  const std::vector<float>& B_row_checksums, int64_t K) {
  float dot = 0.0f;
  for (int64_t k = 0; k < K; ++k) {
    dot += A_col_checksums[k] * B_row_checksums[k];
  }
  return dot;
}

void init_random(int64_t M, int64_t N) {
  if (!g_seeded) {
    srand((unsigned)time(NULL));
    g_seeded = 1;
  }

  if (g_scaling_factors_count_A < M) {
    free(g_scaling_factors_A);
    g_scaling_factors_A = (float*)malloc(sizeof(float) * (size_t)M);
    g_scaling_factors_count_A = M;
  }

  if (g_scaling_factors_count_B < N) {
    free(g_scaling_factors_B);
    g_scaling_factors_B = (float*)malloc(sizeof(float) * (size_t)N);
    g_scaling_factors_count_B = N;
  }
}

// Row-major, contiguous for simplicity.
// C[M,N] := alpha*A[M,K] * B[K,N] + beta*C[M,N]
void hwacc_gemm_f32(const float* A, const float* B, float* C, int64_t M,
                    int64_t N, int64_t K, float alpha, float beta,
                    int64_t tamper_flag /* 0 = false, non-zero = true */) {
  // Setup random scaling factors for debuggin here
  extern float* g_scaling_factors_A;
  extern float* g_scaling_factors_B;

  init_random(M, N);

  // Happens on caller side
  // Calcaulate pre-checksums here
  std::vector<float> A_col_checksums(K, 0.f);
  std::vector<float> B_row_checksums(K, 0.f);

  for (int64_t k = 0; k < K; ++k) {
    for (int64_t m = 0; m < M; ++m) {
      A_col_checksums[k] += A[m * K + k];
    }
  }

  for (int64_t k = 0; k < K; ++k) {
    for (int64_t n = 0; n < N; ++n) {
      B_row_checksums[k] += B[k * N + n];
    }
  }

  // Scale the values here
  float* A_modified = (float*)malloc(sizeof(float) * (size_t)(M * K));
  scale_A(A, A_modified, M, K);

  float* B_modified = (float*)malloc(sizeof(float) * (size_t)(K * N));
  scale_B(B, B_modified, K, N);

  // Acutal GEMM computation on client side
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      float acc = 0.0f;
      const float* Ai = A_modified + i * K;
      const float* Bj = B_modified + j;
      for (int64_t k = 0; k < K; ++k) {
        acc += (Ai[k]) * Bj[k * N];
      }
      // Apply alpha and beta as in the GEMM contract:
      // C[i * N + j] = alpha * acc + beta * C[i * N + j];
      C[i * N + j] = alpha * acc + beta * C[i * N + j];
    }
  }

  // Potential Tampering on Client Side
  if (tamper_flag) {
    float offset = 1000.0f;
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        if ((i == 10 && j == 10) || (i == 20 && j == 20)) {
          C[i * N + j] += offset;
        } else if ((i == 10 && j == 20) || (i == 20 && j == 10)) {
          C[i * N + j] -= offset;
        }
      }
    }
  }

  // Happens on caller side
  // descale_C(C, M, N);

  // Calculate post-checksums here (+ descaling)
  std::vector<float> C_col_checksums(N, 0.f);
  std::vector<float> C_row_checksums(M, 0.f);
  float oc = 0.0f;
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      float val =
          C[i * N + j] / g_scaling_factors_A[i] / g_scaling_factors_B[j];
      C_row_checksums[i] += val;
      C_col_checksums[j] += val;
      C[i * N + j] = val;
      oc += val;
    }
  }

  // Calculate Row/Col for FullChecksum scheme
  std::vector<float> row_checksum(M, 0.f);
  std::vector<float> col_checksum(N, 0.f);

  for (int64_t j = 0; j < N; ++j) {
    float acc = 0.0f;
    /* Compute (1 x K) A_col_checksums times (K x N) original B -> (1 x N) */
    for (int64_t k = 0; k < K; ++k) {
      acc += A_col_checksums[k] * B[k * N + j];
    }
    col_checksum[j] = acc;
  }

  for (int64_t i = 0; i < M; ++i) {
    float acc = 0.0f;
    /* Compute (M x K) original A times (K x 1) B_row_checksums -> (M x 1) */
    const float* Ai = A + i * K;
    for (int64_t k = 0; k < K; ++k) {
      acc += Ai[k] * B_row_checksums[k];
    }
    row_checksum[i] = acc;
  }

  bool passed = true;
  float dot = dot_product(A_col_checksums, B_row_checksums, K);

  if (fabs(oc - dot) > 0.5f) {
    passed = false;
  }

  for (int64_t i = 0; i < N; i++) {
    if (fabs(C_col_checksums[i] - col_checksum[i]) > 1e-1) {
      passed = false;
      printf("Failed at column %lld\n", (long long)i);
    }
  }

  for (int64_t i = 0; i < M; i++) {
    if (fabs(C_row_checksums[i] - row_checksum[i]) > 1e-3) {
      passed = false;
      printf("Failed at row %lld\n", (long long)i);
    }
  }

  printf("OC: %f, Dot: %f (diff: %f)\n", oc, dot, oc - dot);

  printf("Row Checksums:\n");
  for (int64_t i = 0; i < M && i < 10; i++) {
    printf("%f ", C_row_checksums[i]);
  }
  printf("\n");

  for (int64_t i = 0; i < M && i < 10; i++) {
    printf("%f ", row_checksum[i]);
  }
  printf("\n");

  printf("Column Checksums:\n");
  for (int64_t i = 0; i < N && i < 10; i++) {
    printf("%f ", C_col_checksums[i]);
  }
  printf("\n");

  for (int64_t i = 0; i < N && i < 10; i++) {
    printf("%f ", col_checksum[i]);
  }
  printf("\n");

  if (!passed) {
    printf("GEMM result verification FAILED!\n");
  } else {
    printf("GEMM result verification PASSED!\n");
  }

  // Optional logging: write oc-dot and per-element checksum differences
  // Enable by setting either:
  // - env IREE_HWACC_LOGFILE to a path to append logs to, or
  // - env IREE_HWACC_ENABLE_LOG (non-empty) to use default /tmp/hwacc_gemm.log
  const char* hwacc_logfile = getenv("IREE_HWACC_LOGFILE");
  const char* hwacc_enable = getenv("IREE_HWACC_ENABLE_LOG");
  FILE* logf = NULL;
  if (hwacc_logfile != NULL && hwacc_logfile[0] != '\0') {
    logf = fopen(hwacc_logfile, "a");
  } else if (hwacc_enable != NULL && hwacc_enable[0] != '\0') {
    logf = fopen("/tmp/hwacc_gemm.log", "a");
  }

  if (logf) {
    fprintf(logf, "--- hwacc_gemm_f32 verification log ---\n");
    fprintf(logf, "OC: %f, Dot: %f, OC-Dot: %f\n", oc, dot, oc - dot);
    fprintf(logf, "Row diffs (C_row_checksums - row_checksum):\n");
    for (int64_t i = 0; i < M; ++i) {
      float diff = C_row_checksums[i] - row_checksum[i];
      fprintf(logf, "%lld,%f\n", (long long)i, diff);
    }
    fprintf(logf, "Col diffs (C_col_checksums - col_checksum):\n");
    for (int64_t i = 0; i < N; ++i) {
      float diff = C_col_checksums[i] - col_checksum[i];
      fprintf(logf, "%lld,%f\n", (long long)i, diff);
    }
    fprintf(logf, "---------------------------------------\n");
    fclose(logf);
  }
}

#ifdef __cplusplus
}  // extern "C"
#endif