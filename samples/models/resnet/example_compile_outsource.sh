/data/iree/iree-build-3.1.0/tools/iree-opt \
  --iree-stablehlo-to-iree-input \
  /data/iree/src/samples/models/resnet/iree_artifacts/iree_input.mlir \
  -o=/tmp/out_linalg.mlir 

/data/iree/iree-build-3.1.0/tools/iree-opt /tmp/out_linalg.mlir --pass-pipeline='builtin.module(iree-flow-transformation-pipeline, func.func(hwacc-replace-matmul-with-call))' --mlir-disable-threading -o=/tmp/out_linalg_call.mlir
/data/iree/iree-build-3.1.0/tools/iree-compile /tmp/out_linalg_call.mlir --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-link-embedded=true --mlir-disable-threading --iree-llvmcpu-target-cpu=host --iree-plugin=replace_matmul_with_call -o /tmp/out.vmfb 
rm classification.bin
/data/iree/iree-build-3.1.0/tools/iree-run-module   --device=local-sync   --function=predict   --input=1x224x224x3xf32=@labrador_input.bin   --module=/tmp/out.vmfb --output=+classification.bin
python3 analyze_output.py classification.bin