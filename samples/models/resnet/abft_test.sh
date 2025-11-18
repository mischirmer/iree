rm /tmp/out_linalg_abft.mlir /tmp/out_linalg.mlir /data/iree/src/samples/models/resnet/classification_abft.bin /tmp/out_abft.vmfb

/data/iree/iree-build-3.1.0/tools/iree-opt \
  --iree-stablehlo-to-iree-input \
  --iree-preprocessing-convert-conv2d-to-img2col \
  /data/iree/src/samples/models/resnet/iree_artifacts/iree_input.mlir \
  -o=/tmp/out_linalg.mlir 
  
# ABFT Plugin
/data/iree/iree-build-3.1.0/tools/iree-opt /tmp/out_linalg.mlir --pass-pipeline='builtin.module(iree-flow-transformation-pipeline, func.func(abft-insert-ones))' --abft-enable-fuc --mlir-disable-threading -o=/tmp/out_linalg_abft_fuc.mlir
/data/iree/iree-build-3.1.0/tools/iree-opt /tmp/out_linalg.mlir --pass-pipeline='builtin.module(iree-flow-transformation-pipeline, func.func(abft-insert-ones))' --mlir-disable-threading -o=/tmp/out_linalg_abft_fic.mlir
/data/iree/iree-build-3.1.0/tools/iree-opt /tmp/out_linalg.mlir --pass-pipeline='builtin.module(iree-flow-transformation-pipeline)' --mlir-disable-threading -o=/tmp/out_linalg_opt.mlir
/data/iree/iree-build-3.1.0/tools/iree-compile /tmp/out_linalg_abft_fuc.mlir --iree-opt-level=O2 --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-link-embedded=true --mlir-disable-threading --iree-llvmcpu-target-cpu=host --iree-plugin=abft_pass -o /tmp/out_abft_fuc.vmfb 
/data/iree/iree-build-3.1.0/tools/iree-compile /tmp/out_linalg_abft_fic.mlir --iree-opt-level=O2 --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-link-embedded=true --mlir-disable-threading --iree-llvmcpu-target-cpu=host --iree-plugin=abft_pass -o /tmp/out_abft_fic.vmfb 
/data/iree/iree-build-3.1.0/tools/iree-compile /tmp/out_linalg_opt.mlir --iree-opt-level=O2 --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-link-embedded=true --mlir-disable-threading --iree-llvmcpu-target-cpu=host -o /tmp/out.vmfb 
#/data/iree/iree-build-3.1.0/tools/iree-run-module   --device=local-sync   --function=predict   --input=1x224x224x3xf32=@/data/iree/src/samples/models/resnet/labrador_input.bin   --module=/tmp/out_abft_fuc.vmfb --output=+/data/iree/src/samples/models/resnet/classification_abft_fuc.bin
#/data/iree/iree-build-3.1.0/tools/iree-run-module   --device=local-sync   --function=predict   --input=1x224x224x3xf32=@/data/iree/src/samples/models/resnet/labrador_input.bin   --module=/tmp/out_abft_fic.vmfb --output=+/data/iree/src/samples/models/resnet/classification_abft_fic.bin

echo "Benchmarking ABFT with FuC:"
/data/iree/iree-build-3.1.0/tools/iree-benchmark-module   --device=local-sync   --function=predict   --input=1x224x224x3xf32=@/data/iree/src/samples/models/resnet/labrador_input.bin --benchmark_repetitions=5 --module=/tmp/out_abft_fuc.vmfb
echo "Benchmarking ABFT with FIC:"
/data/iree/iree-build-3.1.0/tools/iree-benchmark-module   --device=local-sync   --function=predict   --input=1x224x224x3xf32=@/data/iree/src/samples/models/resnet/labrador_input.bin --benchmark_repetitions=5 --module=/tmp/out_abft_fic.vmfb
echo "Benchmarking Baseline:"
/data/iree/iree-build-3.1.0/tools/iree-benchmark-module   --device=local-sync   --function=predict   --input=1x224x224x3xf32=@/data/iree/src/samples/models/resnet/labrador_input.bin --benchmark_repetitions=5 --module=/tmp/out.vmfb

# python3 /data/iree/src/samples/models/resnet/analyze_output.py /data/iree/src/samples/models/resnet/classification_abft_fuc.bin
# python3 /data/iree/src/samples/models/resnet/analyze_output.py /data/iree/src/samples/models/resnet/classification_abft_fic.bin

# Benchmarks (built with Debug):
# ABFT with FuC: 6837 ms
# ABFT with FIC: 6809 ms
# Baseline: 7473 ms