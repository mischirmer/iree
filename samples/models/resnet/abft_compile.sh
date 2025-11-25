#!/usr/bin/env bash
set -euo pipefail

IREE_BASE_DIR=/data/iree/src
IREE_BUILD_DIR=${IREE_BASE_DIR}/../iree-build-3.1.0
IREE_TMP_DIR=/tmp

# For Docker, uncomment and adjust these
# IREE_BASE_DIR=/workspace/iree
# IREE_BUILD_DIR=${IREE_BASE_DIR}/build
# IREE_TMP_DIR=/tmp

# Single `iree-compile` invocations that run preprocessing passes (conv2d->img2col)
# and then the ABFT insertion pass pipeline before the normal IREE compilation
# pipeline. These avoid separate `iree-opt` calls.

# Common backend flags
COMMON_FLAGS=(
  --iree-hal-target-backends=llvm-cpu
  --iree-hal-target-device=host
  --iree-llvmcpu-link-embedded=true
  --iree-llvmcpu-target-cpu=host
  --mlir-disable-threading
  --iree-opt-level=O2
  --iree-plugin=abft_pass
)

INPUT=${IREE_BASE_DIR}/samples/models/resnet/iree_artifacts/iree_input.mlir

if [ ! -f "${INPUT}" ]; then
  echo "ERROR: input MLIR not found: ${INPUT}" >&2
  exit 2
fi

# ABFT FUC (no scaling)
"${IREE_BUILD_DIR}/tools/iree-compile" \
  "${INPUT}" \
  --iree-preprocessing-pass-pipeline='builtin.module(iree-preprocessing-convert-conv2d-to-img2col)' \
  --iree-preprocessing-pass-pipeline='builtin.module(iree-flow-transformation-pipeline, func.func(abft-insert-ones))' \
  --abft-enable-fuc \
  "${COMMON_FLAGS[@]}" \
  -o "${IREE_TMP_DIR}/out_abft_fuc.vmfb"

# ABFT FUC + scaling
"${IREE_BUILD_DIR}/tools/iree-compile" \
  "${INPUT}" \
  --iree-preprocessing-pass-pipeline='builtin.module(iree-preprocessing-convert-conv2d-to-img2col)' \
  --iree-preprocessing-pass-pipeline='builtin.module(iree-flow-transformation-pipeline, func.func(abft-insert-ones))' \
  --abft-enable-fuc \
  --abft-enable-scaling \
  "${COMMON_FLAGS[@]}" \
  -o "${IREE_TMP_DIR}/out_abft_fuc_scale.vmfb"

# ABFT FIC variant (no FUC)
"${IREE_BUILD_DIR}/tools/iree-compile" \
  "${INPUT}" \
  --iree-preprocessing-pass-pipeline='builtin.module(iree-preprocessing-convert-conv2d-to-img2col)' \
  --iree-preprocessing-pass-pipeline='builtin.module(iree-flow-transformation-pipeline, func.func(abft-insert-ones))' \
  "${COMMON_FLAGS[@]}" \
  -o "${IREE_TMP_DIR}/out_abft_fic.vmfb"

echo "Wrote outputs to ${IREE_TMP_DIR}: out_abft_fuc.vmfb, out_abft_fuc_scale.vmfb, out_abft_fic.vmfb"

echo "Benchmarking ABFT with FuC:"
${IREE_BUILD_DIR}/tools/iree-benchmark-module   --device=local-sync   --function=predict   --input=1x224x224x3xf32=@${IREE_BASE_DIR}/samples/models/resnet/labrador_input.bin --benchmark_repetitions=1 --module=${IREE_TMP_DIR}/out_abft_fuc.vmfb
echo "Benchmarking ABFT with FuC and Scaling:"
${IREE_BUILD_DIR}/tools/iree-benchmark-module   --device=local-sync   --function=predict   --input=1x224x224x3xf32=@${IREE_BASE_DIR}/samples/models/resnet/labrador_input.bin --benchmark_repetitions=1 --module=${IREE_TMP_DIR}/out_abft_fuc_scale.vmfb
echo "Benchmarking ABFT with FIC:"
${IREE_BUILD_DIR}/tools/iree-benchmark-module   --device=local-sync   --function=predict   --input=1x224x224x3xf32=@${IREE_BASE_DIR}/samples/models/resnet/labrador_input.bin --benchmark_repetitions=1 --module=${IREE_TMP_DIR}/out_abft_fic.vmfb