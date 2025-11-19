/data/iree/iree-build-3.1.0/tools/iree-compile /data/iree/src/samples/models/resnet/iree_artifacts/iree_input_small_conv2.mlir --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-link-embedded=true --mlir-disable-threading --iree-llvmcpu-target-cpu=host -o=/tmp/matmul_plain.vmfb
/data/iree/iree-build-3.1.0/tools/iree-opt  /data/iree/src/samples/models/resnet/iree_artifacts/iree_input_small_conv2.mlir   --pass-pipeline='builtin.module(func.func(hwacc-replace-matmul-with-call))'   --mlir-disable-threading   -o=/tmp/out_linalg_call.mlir
/data/iree/iree-build-3.1.0/tools/iree-compile /tmp/out_linalg_call.mlir --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-link-embedded=true --mlir-disable-threading --iree-llvmcpu-target-cpu=host --iree-plugin=replace_matmul_with_call -o /tmp/matmul_call.vmfb
rm /tmp/*.bin
/data/iree/iree-build-3.1.0/tools/iree-run-module --device=local-sync --function=predict --input=1x224x224x3xf32=@/data/iree/src/samples/models/resnet/labrador_input.bin --module=/tmp/matmul_plain.vmfb --output=+/tmp/out_plain.bin
/data/iree/iree-build-3.1.0/tools/iree-run-module --device=local-sync --function=predict --input=1x224x224x3xf32=@/data/iree/src/samples/models/resnet/labrador_input.bin --module=/tmp/matmul_call.vmfb --output=+/tmp/out_call.bin
/data/iree/src/samples/models/resnet/analyze_output.py /tmp/out_plain.bin > /tmp/an_plain.txt 2>&1 || true; python3 /data/iree/src/samples/models/resnet/analyze_output.py /tmp/out_call.bin > /tmp/an_call.txt 2>&1 || true; echo '--- plain analysis ---'; sed -n '1,120p' /tmp/an_plain.txt || true; echo '\n--- call analysis ---'; sed -n '1,120p' /tmp/an_call.txt || true; echo '\n--- md5sums ---'; md5sum /tmp/out_plain.bin /tmp/out_call.bin || true; echo '\n--- identical? (cmp -s exit -> 0 means identical) ---'; if cmp -s /tmp/out_plain.bin /tmp/out_call.bin; then echo IDENTICAL; else echo DIFFER; fi; echo '\n--- first 50 differing bytes (cmp -l | head) ---'; cmp -l /tmp/out_plain.bin /tmp/out_call.bin | head -n 50"

exit(0)

"""
python3 - <<'PY'
import numpy as np
p = '/tmp/out_plain.bin'
c = '/tmp/out_call.bin'
try:
    a = np.fromfile(p, dtype=np.float32)
except Exception as e:
    print(f"failed reading {p}: {e}")
    a = np.array([], dtype=np.float32)
try:
    b = np.fromfile(c, dtype=np.float32)
except Exception as e:
    print(f"failed reading {c}: {e}")
    b = np.array([], dtype=np.float32)

print('len a, b:', a.size, b.size)
print('dtype:', a.dtype)
minlen = min(a.size, b.size)
print('first 10 a:', a[:10])
print('first 10 b:', b[:10])
d = a[:minlen] - b[:minlen] if minlen > 0 else np.array([], dtype=np.float32)
print('\nstats over first', minlen, 'elements')
print('L2 norm:', np.linalg.norm(d))
print('max abs:', np.max(np.abs(d)) if d.size > 0 else 0)
print('mean abs:', np.mean(np.abs(d)) if d.size > 0 else 0)
print('num abs>1e-6:', int(np.sum(np.abs(d) > 1e-6)) if d.size > 0 else 0)
print('argmax a,b:', int(np.argmax(a)) if a.size > 0 else -1, int(np.argmax(b)) if b.size > 0 else -1)
idxs = np.argsort(a)[-10:][::-1] if a.size > 0 else np.array([], dtype=int)
print('top10 indices a:', idxs)
print('top10 values a:', a[idxs])
idxs2 = np.argsort(b)[-10:][::-1] if b.size > 0 else np.array([], dtype=int)
print('top10 indices b:', idxs2)
print('top10 values b:', b[idxs2])

print('\nfirst 20 differences (index,a,b,diff):')
cnt = 0
for i in range(minlen):
    if a[i] != b[i]:
        print(i, a[i], b[i], a[i] - b[i])
        cnt += 1
        if cnt >= 20:
            break
if cnt == 0:
    print('no differences found')
PY

"""