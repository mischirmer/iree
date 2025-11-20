IREE_BASE_DIR=/data/iree/src
IREE_BUILD_DIR=${IREE_BASE_DIR}/../iree-build-3.1.0
IREE_TMP_DIR=/tmp

# For Docker
#IREE_BASE_DIR=/workspace/iree
#IREE_BUILD_DIR=${IREE_BASE_DIR}/build
#IREE_TMP_DIR=/tmp

rm ${IREE_TMP_DIR}/out_linalg_abft.mlir ${IREE_TMP_DIR}/out_linalg.mlir ${IREE_BASE_DIR}/samples/models/resnet/classification_abft.bin ${IREE_TMP_DIR}/out_abft.vmfb

${IREE_BUILD_DIR}/tools/iree-opt \
  --iree-stablehlo-to-iree-input \
  --iree-preprocessing-convert-conv2d-to-img2col \
  ${IREE_BASE_DIR}/samples/models/resnet/iree_artifacts/iree_input.mlir \
  -o=${IREE_TMP_DIR}/out_linalg.mlir 
  
# Run both ABFT and hwacc replacement in a single flow pipeline invocation so
# that shape cleanup happens after both transforms and tensor.dim ops are
# resolved/removed as expected.
env IREE_HWACC_TAMPER='[5]' IREE_HWACC_FORCE_REPLACE_LIST='[5]' \
  ${IREE_BUILD_DIR}/tools/iree-opt ${IREE_TMP_DIR}/out_linalg.mlir \
  --pass-pipeline='builtin.module(iree-flow-transformation-pipeline, func.func(abft-insert-ones), func.func(hwacc-replace-matmul-with-call))' \
  --abft-enable-fuc --abft-enable-scaling --mlir-disable-threading \
  -o=${IREE_TMP_DIR}/out_linalg_abft_fuc_scale_call.mlir

# Compile the fully-transformed MLIR (both ABFT and hwacc applied).
${IREE_BUILD_DIR}/tools/iree-compile ${IREE_TMP_DIR}/out_linalg_abft_fuc_scale_call.mlir \
  --iree-opt-level=O2 --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-link-embedded=true \
  --mlir-disable-threading --iree-llvmcpu-target-cpu=host --iree-plugin=replace_matmul_with_call --iree-plugin=abft_pass \
  -o ${IREE_TMP_DIR}/out_abft_fuc_scale.vmfb
#${IREE_BUILD_DIR}/tools/iree-compile ${IREE_TMP_DIR}/out_linalg_abft_fic.mlir --iree-opt-level=O2 --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-link-embedded=true --mlir-disable-threading --iree-llvmcpu-target-cpu=host --iree-plugin=abft_pass -o ${IREE_TMP_DIR}/out_abft_fic.vmfb 
#${IREE_BUILD_DIR}/tools/iree-run-module   --device=local-sync   --function=predict   --input=1x224x224x3xf32=@${IREE_BASE_DIR}/samples/models/resnet/labrador_input.bin   --module=${IREE_TMP_DIR}/out_abft_fuc.vmfb --output=+${IREE_BASE_DIR}/samples/models/resnet/classification_abft_fuc.bin
#${IREE_BUILD_DIR}/tools/iree-run-module   --device=local-sync   --function=predict   --input=1x224x224x3xf32=@${IREE_BASE_DIR}/samples/models/resnet/labrador_input.bin   --module=${IREE_TMP_DIR}/out_abft_fic.vmfb --output=+${IREE_BASE_DIR}/samples/models/resnet/classification_abft_fic.bin

#echo "Benchmarking ABFT with FuC:"
#${IREE_BUILD_DIR}/tools/iree-benchmark-module   --device=local-sync   --function=predict   --input=1x224x224x3xf32=@${IREE_BASE_DIR}/samples/models/resnet/labrador_input.bin --benchmark_repetitions=10 --module=${IREE_TMP_DIR}/out_abft_fuc.vmfb
#echo "Benchmarking ABFT with FuC and Scaling:"
#${IREE_BUILD_DIR}/tools/iree-benchmark-module   --device=local-sync   --function=predict   --input=1x224x224x3xf32=@${IREE_BASE_DIR}/samples/models/resnet/labrador_input.bin --benchmark_repetitions=10 --module=${IREE_TMP_DIR}/out_abft_fuc_scale.vmfb
#echo "Benchmarking ABFT with FIC:"
#${IREE_BUILD_DIR}/tools/iree-benchmark-module   --device=local-sync   --function=predict   --input=1x224x224x3xf32=@${IREE_BASE_DIR}/samples/models/resnet/labrador_input.bin --benchmark_repetitions=10 --module=${IREE_TMP_DIR}/out_abft_fic.vmfb
#echo "Benchmarking Baseline:"
#${IREE_BUILD_DIR}/tools/iree-benchmark-module   --device=local-sync   --function=predict   --input=1x224x224x3xf32=@${IREE_BASE_DIR}/samples/models/resnet/labrador_input.bin --benchmark_repetitions=10 --module=${IREE_TMP_DIR}/out.vmfb

# python3 ${IREE_BASE_DIR}/src/samples/models/resnet/analyze_output.py ${IREE_BASE_DIR}/src/samples/models/resnet/classification_abft_fuc.bin
# python3 ${IREE_BASE_DIR}/src/samples/models/resnet/analyze_output.py ${IREE_BASE_DIR}/src/samples/models/resnet/classification_abft_fic.bin
${IREE_BUILD_DIR}/tools/iree-run-module \
  --device=local-sync --function=predict \
  --input=1x224x224x3xf32=@${IREE_BASE_DIR}/samples/models/resnet/labrador_input.bin \
  --module=${IREE_TMP_DIR}/out_abft_fuc_scale.vmfb \
  --output=+${IREE_BASE_DIR}/samples/models/resnet/classification_abft_fuc_scale.bin

# Only run analysis if the run-module step succeeded.
rc=$?
if [ $rc -eq 0 ]; then
  python3 ${IREE_BASE_DIR}/samples/models/resnet/analyze_output.py ${IREE_BASE_DIR}/samples/models/resnet/classification_abft_fuc_scale.bin
else
  echo "iree-run-module failed with exit code $rc; skipping analysis"
  exit $rc
fi

# Benchmarks (built with Debug):
# ABFT with FuC: 6347 ms / 
# ABFT with FuC and Scaling: 6493 ms / 
# ABFT with FIC: 6488 ms / 
# Baseline: 7153 ms / 
