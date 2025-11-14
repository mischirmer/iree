rm /tmp/out_linalg.mlir /tmp/out_linalg_call.mlir /data/iree/src/samples/models/resnet/classification_call.bin /data/iree/src/samples/models/resnet/classification_plain.bin

/data/iree/iree-build-3.1.0/tools/iree-opt \
  --iree-stablehlo-to-iree-input \
  --iree-preprocessing-convert-conv2d-to-img2col \
  /data/iree/src/samples/models/resnet/iree_artifacts/iree_input.mlir \
  -o=/tmp/out_linalg.mlir 
  # 
  

/data/iree/iree-build-3.1.0/tools/iree-opt /tmp/out_linalg.mlir --pass-pipeline='builtin.module(iree-flow-transformation-pipeline)' --mlir-disable-threading -o=/tmp/out_linalg_plain.mlir
/data/iree/iree-build-3.1.0/tools/iree-compile /tmp/out_linalg_plain.mlir --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-link-embedded=true --mlir-disable-threading --iree-llvmcpu-target-cpu=host -o /tmp/out_plain.vmfb 
/data/iree/iree-build-3.1.0/tools/iree-run-module   --device=local-sync   --function=predict   --input=1x224x224x3xf32=@/data/iree/src/samples/models/resnet/labrador_input.bin   --module=/tmp/out_plain.vmfb --output=+/data/iree/src/samples/models/resnet/classification_plain.bin


env IREE_HWACC_TAMPER='[]' IREE_HWACC_FORCE_REPLACE_LIST='[5]' /data/iree/iree-build-3.1.0/tools/iree-opt /tmp/out_linalg.mlir --pass-pipeline='builtin.module(iree-flow-transformation-pipeline, func.func(hwacc-replace-matmul-with-call))' --mlir-disable-threading -o=/tmp/out_linalg_call.mlir
/data/iree/iree-build-3.1.0/tools/iree-compile /tmp/out_linalg_call.mlir --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-link-embedded=true --mlir-disable-threading --iree-llvmcpu-target-cpu=host --iree-plugin=replace_matmul_with_call -o /tmp/out_call.vmfb 
/data/iree/iree-build-3.1.0/tools/iree-run-module   --device=local-sync   --function=predict   --input=1x224x224x3xf32=@/data/iree/src/samples/models/resnet/labrador_input.bin   --module=/tmp/out_call.vmfb --output=+/data/iree/src/samples/models/resnet/classification_call.bin

python3 /data/iree/src/samples/models/resnet/analyze_output.py /data/iree/src/samples/models/resnet/classification_call.bin
python3 /data/iree/src/samples/models/resnet/analyze_output.py /data/iree/src/samples/models/resnet/classification_plain.bin



#/tmp/matmul_plain.vmfb
#/tmp/matmul_call.vmfb
