IREE_BASE_DIR=/data/iree/src
IREE_BUILD_DIR=${IREE_BASE_DIR}/../iree-build-3.1.0
IREE_TMP_DIR=/tmp
IREE_HWACC_LOGFILE=/tmp/abft_hwacc.log
IREE_HWACC_ENABLE_LOG=1

export IREE_HWACC_LOGFILE=${IREE_HWACC_LOGFILE}
export IREE_HWACC_ENABLE_LOG=${IREE_HWACC_ENABLE_LOG}

# For Docker
#IREE_BASE_DIR=/workspace/iree
#IREE_BUILD_DIR=${IREE_BASE_DIR}/build
#IREE_TMP_DIR=/tmp

# Remove any previous log so each run starts clean.
rm -f "${IREE_HWACC_LOGFILE}"

rm ${IREE_TMP_DIR}/out_linalg_abft.mlir ${IREE_TMP_DIR}/out_linalg.mlir ${IREE_BASE_DIR}/samples/models/resnet/classification_abft.bin ${IREE_TMP_DIR}/out_abft.vmfb

${IREE_BUILD_DIR}/tools/iree-opt \
  --iree-stablehlo-to-iree-input \
  --iree-preprocessing-convert-conv2d-to-img2col \
  ${IREE_BASE_DIR}/samples/models/resnet/iree_artifacts/iree_input.mlir \
  -o=${IREE_TMP_DIR}/out_linalg.mlir 
  
# Run both ABFT and hwacc replacement in a single flow pipeline invocation so
# that shape cleanup happens after both transforms and tensor.dim ops are
# resolved/removed as expected.
# NOTE: Instead of compiling once for a single replacement list, iterate
# over node ids and compile a module per-node where the hwacc replace list
# contains the single node id. This allows running the module with one
# matmul replaced at a time and saving per-node logs for analysis.

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

# Run the compiled module for every binary input found in the sample `bins/`
# directory. Save logs and outputs per run in a dedicated directory so
# each run can be inspected independently. Use an indexed filename format.
LOG_ROOT="/tmp/logfiles"
# Where the sample binaries live
BINS_DIR="${IREE_BASE_DIR}/samples/models/resnet/bins"

# Loop over node ids (0..51), compile a module with that single node replaced,
# then run the compiled module over all bin inputs and save per-node logs.
for node_id in $(seq 0 51); do
  node_padded=$(printf "%04d" "${node_id}")
  NODE_LOG_DIR="${LOG_ROOT}/node_${node_padded}"
  mkdir -p "${NODE_LOG_DIR}"

  echo "\n=== NODE ${node_id} (logs: ${NODE_LOG_DIR}) ==="

  # Run transformation pipeline with this node forced for replacement
  echo "Compiling MLIR with IREE_HWACC_FORCE_REPLACE_LIST=[${node_id}]"
  env IREE_HWACC_TAMPER='[]' IREE_HWACC_FORCE_REPLACE_LIST="[${node_id}]" \
    ${IREE_BUILD_DIR}/tools/iree-opt ${IREE_TMP_DIR}/out_linalg.mlir \
    --pass-pipeline='builtin.module(iree-flow-transformation-pipeline, func.func(abft-insert-ones), func.func(hwacc-replace-matmul-with-call))' \
    --abft-enable-fuc --abft-enable-scaling --mlir-disable-threading \
    -o=${IREE_TMP_DIR}/out_linalg_abft_fuc_scale_call_node_${node_id}.mlir

  # Compile the fully-transformed MLIR (both ABFT and hwacc applied) for this node
  env IREE_HWACC_LOGFILE=${IREE_HWACC_LOGFILE} IREE_HWACC_ENABLE_LOG=1 ${IREE_BUILD_DIR}/tools/iree-compile ${IREE_TMP_DIR}/out_linalg_abft_fuc_scale_call_node_${node_id}.mlir \
    --iree-opt-level=O2 --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-link-embedded=true \
    --mlir-disable-threading --iree-llvmcpu-target-cpu=host --iree-plugin=replace_matmul_with_call --iree-plugin=abft_pass \
    -o ${IREE_TMP_DIR}/out_abft_fuc_scale_node_${node_id}.vmfb

  if [ $? -ne 0 ]; then
    echo "Compilation failed for node ${node_id}, skipping runs for this node."
    continue
  fi

  idx=0
  for binfile in "${BINS_DIR}"/*.bin; do
    if [ ! -f "${binfile}" ]; then
      continue
    fi
    idx=$((idx+1))
    idx_padded=$(printf "%03d" "${idx}")

    RUN_LOG="${NODE_LOG_DIR}/run_${idx_padded}.log"
    HWACC_LOG="${NODE_LOG_DIR}/hwacc_${idx_padded}.log"
    OUT_BIN="${IREE_BASE_DIR}/samples/models/resnet/classifications/classification_abft_fuc_scale_node${node_padded}_${idx_padded}.bin"

    echo "Running input ${binfile} -> output ${OUT_BIN}; logs: ${RUN_LOG}, ${HWACC_LOG}"

    # Run the compiled module for this node, point the hwacc logger to a per-run file
    env IREE_HWACC_LOGFILE="${HWACC_LOG}" IREE_HWACC_ENABLE_LOG=1 \
      ${IREE_BUILD_DIR}/tools/iree-run-module \
        --device=local-sync --function=predict \
        --input=1x224x224x3xf32=@"${binfile}" \
        --module=${IREE_TMP_DIR}/out_abft_fuc_scale_node_${node_id}.vmfb \
        --output=+"${OUT_BIN}" \
        > "${RUN_LOG}" 2>&1

    rc=$?
    if [ $rc -ne 0 ]; then
      echo "Run ${idx_padded} failed with exit code $rc (log: ${RUN_LOG})"
    else
      echo "Run ${idx_padded} succeeded (output: ${OUT_BIN})"
    fi
  done
done