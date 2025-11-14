rm -f /tmp/out_linalg.mlir /tmp/out_linalg_plain.mlir /tmp/out_linalg_call.mlir /tmp/out_plain.vmfb /tmp/out_call.vmfb /data/iree/src/samples/models/resnet/classification_call.bin /data/iree/src/samples/models/resnet/classification_plain.bin

# Generate initial linalg IR from stablehlo input
/data/iree/iree-build-3.1.0/tools/iree-opt \
  --iree-stablehlo-to-iree-input \
  --iree-preprocessing-convert-conv2d-to-img2col \
  /data/iree/src/samples/models/resnet/iree_artifacts/iree_input.mlir \
  -o=/tmp/out_linalg.mlir

# Produce the plain (reference) transformed module, compile and run it
/data/iree/iree-build-3.1.0/tools/iree-opt /tmp/out_linalg.mlir --pass-pipeline='builtin.module(iree-flow-transformation-pipeline)' --mlir-disable-threading -o=/tmp/out_linalg_plain.mlir
/data/iree/iree-build-3.1.0/tools/iree-compile /tmp/out_linalg_plain.mlir --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-link-embedded=true --mlir-disable-threading --iree-llvmcpu-target-cpu=host -o /tmp/out_plain.vmfb
/data/iree/iree-build-3.1.0/tools/iree-run-module --device=local-sync --function=predict --input=1x224x224x3xf32=@/data/iree/src/samples/models/resnet/labrador_input.bin --module=/tmp/out_plain.vmfb --output=+/data/iree/src/samples/models/resnet/classification_plain.bin

# Save plain predictions for comparison
python3 /data/iree/src/samples/models/resnet/analyze_output.py /data/iree/src/samples/models/resnet/classification_plain.bin > /tmp/plain_predictions.txt

echo "Plain predictions (saved to /tmp/plain_predictions.txt):"
cat /tmp/plain_predictions.txt

# Extract top-3 labels from plain predictions (one label per line)
# robustly handle two formats:
# 1) lines like: n02091244       Ibizan_hound    prob=0.001136
# 2) or just: Ibizan_hound
# skip empty lines and any header line starting with 'predictions'
# Prefer lines that start with an ImageNet id like 'n02091244'.
# Fallback to the generic extractor if we don't get 3 labels.
grep -E '^n[0-9]+' /tmp/plain_predictions.txt | awk '{print $2}' | head -n 3 > /tmp/plain_top3.txt
if [ $(wc -l < /tmp/plain_top3.txt) -lt 3 ]; then
  awk 'BEGIN{c=0} /^[[:space:]]*$/ {next} /^predictions/ {next} {lbl=($2!=""?$2:$1); print lbl; c++; if(c==3) exit}' /tmp/plain_predictions.txt > /tmp/plain_top3.txt
fi
echo
echo "Plain top-3 labels (saved to /tmp/plain_top3.txt):"
cat /tmp/plain_top3.txt

echo
echo "Running call-replace pipeline for IREE_HWACC_FORCE_REPLACE_LIST indices 0..50 and comparing to plain predictions"

# Collect indices where the call output matches the plain output
matched_indices=()

for i in $(seq 0 5); do
  echo
  echo "=== iteration: $i ==="
  # run replace pass with a single-element replace list [i]
  env IREE_HWACC_TAMPER="[$i]" IREE_HWACC_FORCE_REPLACE_LIST="[$i]" /data/iree/iree-build-3.1.0/tools/iree-opt /tmp/out_linalg.mlir --pass-pipeline='builtin.module(iree-flow-transformation-pipeline, func.func(hwacc-replace-matmul-with-call))' --mlir-disable-threading -o=/tmp/out_linalg_call.mlir

  /data/iree/iree-build-3.1.0/tools/iree-compile /tmp/out_linalg_call.mlir --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-link-embedded=true --mlir-disable-threading --iree-llvmcpu-target-cpu=host --iree-plugin=replace_matmul_with_call -o /tmp/out_call.vmfb || { echo "compile failed for index $i"; continue; }
  /data/iree/iree-build-3.1.0/tools/iree-run-module --device=local-sync --function=predict --input=1x224x224x3xf32=@/data/iree/src/samples/models/resnet/labrador_input.bin --module=/tmp/out_call.vmfb --output=+/data/iree/src/samples/models/resnet/classification_call.bin || { echo "run failed for index $i"; continue; }

  # analyze call result
  python3 /data/iree/src/samples/models/resnet/analyze_output.py /data/iree/src/samples/models/resnet/classification_call.bin > /tmp/tampered/call_predictions_${i}.txt

  echo "Call predictions for index $i:"
  cat /tmp/tampered/call_predictions_${i}.txt

  # Extract top-3 labels from call predictions and compare to plain top-3
  # Prefer lines that start with an ImageNet id like 'n02091244'.
  # Fallback to the generic extractor if we don't get 3 labels.
  grep -E '^n[0-9]+' /tmp/tampered/call_predictions_${i}.txt | awk '{print $2}' | head -n 3 > /tmp/tampered/call_top3_${i}.txt
  if [ $(wc -l < /tmp/tampered/call_top3_${i}.txt) -lt 3 ]; then
    awk 'BEGIN{c=0} /^[[:space:]]*$/ {next} /^predictions/ {next} {lbl=($2!=""?$2:$1); print lbl; c++; if(c==3) exit}' /tmp/tampered/call_predictions_${i}.txt > /tmp/tampered/call_top3_${i}.txt
  fi

  echo "Call top-3 labels for index $i (saved to /tmp/tampered/call_top3_${i}.txt):"
  cat /tmp/tampered/call_top3_${i}.txt

  # Compare top-3 label lists
  if cmp -s /tmp/plain_top3.txt /tmp/tampered/call_top3_${i}.txt; then
    echo "Result for index $i: SAME top-3 labels as plain"
    matched_indices+=("$i")
  else
    echo "Result for index $i: DIFFERENT top-3 labels from plain"
    echo "--- top-3 diff ---"
    diff -u /tmp/plain_top3.txt /tmp/call_top3_${i}.txt || true
    echo "--- end top-3 diff ---"
  fi
  rm /data/iree/src/samples/models/resnet/classification_call.bin
done
# Print summary of matching indices
echo
if [ ${#matched_indices[@]} -eq 0 ]; then
  echo "No indices produced the same output as plain."
else
  echo "Indices that matched the plain output (count=${#matched_indices[@]}):"
  printf '%s\n' "${matched_indices[@]}"
  # Also print as a comma-separated list
  savedIFS="$IFS"
  IFS=,
  echo "Matched list: [${matched_indices[*]}]"
  IFS="$savedIFS"
fi

# End
#/tmp/matmul_plain.vmfb
#/tmp/matmul_call.vmfb
