rm /tmp/out_linalg_abft.mlir /tmp/out_linalg.mlir /data/iree/src/samples/models/resnet/classification_abft.bin /tmp/out_abft.vmfb

/data/iree/iree-build-3.1.0/tools/iree-opt \
  --iree-stablehlo-to-iree-input \
  --iree-preprocessing-convert-conv2d-to-img2col \
  /data/iree/src/samples/models/resnet/iree_artifacts/iree_input.mlir \
  -o=/tmp/out_linalg.mlir 
  
# ABFT Plugin
# /data/iree/iree-build-3.1.0/tools/iree-opt /tmp/out_linalg.mlir --pass-pipeline='builtin.module(iree-flow-transformation-pipeline, func.func(abft-insert-ones))' --abft-enable-fuc --mlir-disable-threading -o=/tmp/out_linalg_abft.mlir
/data/iree/iree-build-3.1.0/tools/iree-opt /tmp/out_linalg.mlir --pass-pipeline='builtin.module(iree-flow-transformation-pipeline, func.func(abft-insert-ones))' --mlir-disable-threading -o=/tmp/out_linalg_abft.mlir
/data/iree/iree-build-3.1.0/tools/iree-compile /tmp/out_linalg_abft.mlir --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-link-embedded=true --mlir-disable-threading --iree-llvmcpu-target-cpu=host --iree-plugin=abft_pass -o /tmp/out_abft.vmfb 
/data/iree/iree-build-3.1.0/tools/iree-run-module   --device=local-sync   --function=predict   --input=1x224x224x3xf32=@/data/iree/src/samples/models/resnet/labrador_input.bin   --module=/tmp/out_abft.vmfb --output=+/data/iree/src/samples/models/resnet/classification_abft.bin

python3 /data/iree/src/samples/models/resnet/analyze_output.py /data/iree/src/samples/models/resnet/classification_abft.bin