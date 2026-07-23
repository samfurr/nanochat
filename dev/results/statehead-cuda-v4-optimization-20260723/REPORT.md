# StateHead CUDA v4 one-H100 optimization pass

Date: 2026-07-23

Recorded starting commit:
`c1ac745003833655cd36ee762fef31fded66d465`

Recorded starting status:

```text
?? .DS_Store
?? .agents/
?? STATEHEAD_NANOCHAT_CODEX_BRIEF.md
```

Those pre-existing untracked paths are user-owned and were not modified.

Final branch commit before this report:
`efd0ca5de83eb16ed54be6a548b5abddbfe44532`

Accepted implementation commit:
`c8847dc31bc4809bbab86bf7c21aea5a13c05662`

The final source tree at `efd0ca5` is byte-for-byte identical to `c8847dc`
for the CUDA extension, StateHead Python integration, preflight, and tests.
The intervening commits preserve two measured experiments and their explicit
reverts.

Hardware/runtime: one verified 700 W NVIDIA H100 80GB HBM3 (H100 SXM),
driver 580.126.20, PyTorch 2.9.1+cu128, CUDA runtime/compiler 12.8, and SM90.

## Verdict

- CUDA v4 is the current speed winner: 938,891 median full-step tok/s at
  d12/768/T2048/batch 32 in compiled BF16 training.
- It is 13.06% faster than the same-host CUDA v2 chunk-32 baseline.
- It is 54.40% faster, or 1.544x as fast, as compiled PyTorch StateHead.
- It is 80.78% faster, or 1.808x as fast, as GPT with verified FA3.
- It uses 13.15 GiB peak allocated memory, versus 36.43 GiB for compiled
  PyTorch StateHead and 27.40 GiB for GPT/FA3.
- It does **not** reach 2x GPT. Another approximately 10.63% throughput, or
  6.71 ms from the 69.80 ms median step, is required.
- CUDA compilation, FP32 numerical/gradient parity, BF16
  numerical/gradient parity, partial-chunk coverage, sequence-2048 coverage,
  `torch.compile(fullgraph=True)`, compiled full-model loss/gradient parity,
  and short finite-gradient stability passed.
- Dataset learning parity, native FP8 stability/speed, and eight-GPU DDP were
  not tested and are not claimed.
- The Transformer/GPT implementation and behavior were not changed.

## Accepted implementation

CUDA v2 recalculated sigmoid/tanh gates in every recurrence kernel. At the
production shape, that meant activating each gate repeatedly in forward and
backward.

CUDA v3 added one parallel activation-cache kernel. It computes each sigmoid
or tanh once, stores the activated gates in the model dtype, and lets all
forward/backward recurrence kernels reuse them. The recurrence, chunk
summaries, and boundaries remain FP32.

CUDA v4 also passes the projection bias into that activation kernel. This
eliminates the compiled pointwise gate-bias materialization while keeping the
public scan API backward compatible. Backward sums raw-gate gradients to
produce the gate-bias gradient.

Changed source paths:

- `dev/statehead_cuda_preflight.py`
- `nanochat/csrc/statehead_cuda.cpp`
- `nanochat/csrc/statehead_cuda_kernel.cu`
- `nanochat/statehead.py`
- `nanochat/statehead_cuda.py`
- `tests/test_statehead.py`

## Full-step performance

All formal comparison rows below use the same US H100, d12/768/6-head model,
sequence length 2048, device batch 32, BF16, `torch.compile`, Muon/AdamW
optimizer, fixed synthetic token batch, and five excluded warmup steps.
Thirty-step runs therefore have 25 measured steps. The v2 chunk sweep used
20 total steps and 15 measured steps.

| Mode | Commit | Chunk | Median tok/s | Median step | Peak allocated |
| --- | --- | ---: | ---: | ---: | ---: |
| StateHead CUDA v2 | `c1ac745` | 16 | 821,175 | 79.81 ms | 13.22 GiB |
| StateHead CUDA v2 | `c1ac745` | 32 | 830,413 | 78.92 ms | 13.15 GiB |
| StateHead CUDA v3, activation cache | `ad9f5ef` | 32 | 895,727 | 73.17 ms | 13.15 GiB |
| **StateHead CUDA v4, cache + fused bias** | `c8847dc` | **32** | **938,891** | **69.80 ms** | **13.15 GiB** |
| StateHead CUDA v5, indexing probe | `d71a90f` | 32 | 938,706 | 69.82 ms | 13.15 GiB |
| StateHead CUDA v6, activation/summary fusion | `ee6dbe6` | 32 | 922,002 | 71.08 ms | 13.15 GiB |
| StateHead compiled PyTorch | `c8847dc` | 32 | 608,102 | 107.77 ms | 36.43 GiB |
| GPT/verified FA3 | `c8847dc` | n/a | 519,371 | 126.18 ms | 27.40 GiB |

Controlled comparisons:

- CUDA v2 chunk 32 versus chunk 16: 1.13% faster.
- CUDA v3 versus CUDA v2 chunk 32: 7.87% faster.
- CUDA v4 versus CUDA v3: 4.82% faster.
- CUDA v4 versus CUDA v2 chunk 32: 13.06% faster.
- CUDA v4 versus compiled PyTorch StateHead: 54.40% faster.
- CUDA v4 versus GPT/FA3: 80.78% faster.
- CUDA v5 versus CUDA v4: 0.02% slower; neutral probe reverted.
- CUDA v6 versus CUDA v4: 1.80% slower; rejected probe reverted.

The fixed synthetic batch deliberately removes loader variance and is useful
for throughput comparisons. Its rapidly overfit loss trajectory is not
evidence of dataset learning parity.

## Profiling

Nsight Systems captured exactly five post-warmup optimizer steps through the
CUDA profiler API added to the preflight. The figures below are summed custom
StateHead CUDA kernel duration divided by five:

| Revision | Custom StateHead CUDA time/step |
| --- | ---: |
| CUDA v2 | 23.924 ms |
| CUDA v3 activation cache | 16.810 ms |
| CUDA v4 activation cache + fused bias | 16.881 ms |

CUDA v4 custom-kernel costs:

| Kernel group | Time/step |
| --- | ---: |
| Standalone gate activation | 5.675 ms |
| Backward local gradient replay | 5.375 ms |
| Forward output replay | 2.339 ms |
| Forward chunk summaries | 1.664 ms |
| Backward chunk summaries | 1.482 ms |
| Boundary scans combined | approximately 0.346 ms |

The CUDA v2 profile also contained compiled pointwise kernel
`triton_poi_fused__to_copy__unsafe_view_add_6` at 3.176 ms/step. It was the
gate-bias add and is absent from CUDA v4. CUDA v4's overall improvement comes
from both the activation-cache scan savings and removal of that external bias
materialization.

Nsight Compute launched, but this host denied hardware performance counters
with `ERR_NVGPUCTRPERM`. The diagnostic CSV and run JSON are retained. No
occupancy, achieved-bandwidth, or register-pressure claim is made.

## Rejected probes

### CUDA v5: simplified indexing

Removing flat index divisions from the forward kernels measured 938,706 tok/s,
0.02% below v4. The change was neutral within run noise and was reverted to
avoid carrying complexity without evidence.

### CUDA v6: activation fused into forward summaries

Fusing activation with forward chunk-summary generation measured 922,002
tok/s, 1.80% below v4. It reduced useful parallelism and still needed the
activation cache for backward/output replay. It was reverted.

The evidence says not to repeat either arrangement unchanged.

## What can still make it faster

The next material target is the 5.675 ms standalone activation pass. A
CUDA-native gate-projection path or CUTLASS-style projection epilogue could
write already-activated interleaved gates and eliminate a complete read/write
of the large gate tensor. Stock cuBLASLt epilogues are unlikely to express the
mixed sigmoid/tanh activation for all four gates, so this needs a bounded
prototype rather than an assumption.

Eliminating all 5.675 ms would produce a theoretical 64.13 ms step, about
1.97x the measured GPT control. Reaching 2x still needs roughly another 1 ms,
most plausibly from a production-shape K32/Dh128 specialization, more efficient
backward replay, or folding the output residual into the output projection.

The next implementation should:

1. keep the large matrix multiplication on a tuned tensor-core path;
2. emit the existing interleaved `[B,T,H,Dh,4]` gate layout;
3. apply sigmoid to retention/input/output gates and tanh to the candidate in
   the projection output path;
4. keep recurrence and chunk state FP32;
5. preserve the current v4 implementation as the correctness fallback;
6. pass the existing FP32/BF16 and compiled full-model parity gates before a
   performance conclusion.

FP8 remains a later GEMM optimization. It should be applied only to eligible
gate/output projections with native scaling; the recurrence and summaries
must remain FP32.

## Correctness results

| Gate | Result | Evidence |
| --- | ---: | --- |
| CUDA 12.8 extension compile/load | Pass | `native-v4-bias-load.log` |
| FP32 StateHead suite | 89 passed, one warning | `v4-bias-fp32.log` |
| Full BF16 StateHead suite | 87 passed, two unrelated GPT/FA3 CPU failures | `v4-bias-bf16.log` |
| FP32/BF16 forward and raw-gate/initial-state gradients | Pass | full StateHead suites |
| Partial chunks and sequence length 2048 | Pass | full StateHead suites |
| `torch.compile(fullgraph=True)` custom op | Pass | full StateHead suites |
| Compiled full-model loss/parameter gradients | Pass | full StateHead suites |
| CUDA v5 focused BF16 tests | 45 passed, 44 deselected | `v5-grid-focused.log` |
| CUDA v6 focused BF16 tests | 45 passed, 44 deselected | `v6-fused-summary-focused.log` |
| Exact final CUDA branch five-step gradient check | Pass; all gradients finite | `statehead-cuda-v4-final-gradient-stability.json` |
| Final local StateHead test suite | 53 passed, 36 skipped | local terminal output |
| Local BF16/MPS five-step StateHead smoke | Pass; loss 5.924298 to 5.923664 | local terminal output |

The two BF16 failures are the known CPU GPT checkpoint round-trip cases:
`test_checkpoint_round_trip[legacy-gpt]` and
`test_checkpoint_round_trip[gpt]`. With CUDA/FA3 visible globally, they route
a CPU tensor to the CUDA-only FA3 operation. They do not execute StateHead or
the modified CUDA kernels.

The cached BF16 activations introduce one BF16 rounding point before repeated
reuse. Existing BF16 tolerances and full-model parity passed. That is still not
a substitute for a dataset-backed learning-parity run.

## Cost and lifecycle

- Approved ceiling: $10.
- Attempt 1, instance `45639121`, offer `43189295`, $3.0333/hour: incompatible
  image entrypoint; destroyed.
- Attempt 2, instance `45639403`, offer `41819339`, $2.2778/hour: host DNS
  prevented SSH-key injection; destroyed.
- Attempt 3, instance `45639678`, offer `36444802`, $3.4817/hour: invalid
  ownership/modes on `/root/.ssh/authorized_keys`; destroyed.
- Productive attempt, instance `45639924`, offer `36444807`, $3.4817/hour:
  created with an on-start SSH ownership/mode repair and completed.
- Productive runtime: 2,559.14 seconds, or 42.65 minutes.
- Productive estimated compute charge: approximately $2.48.
- Estimated total including the three short failed startups: approximately
  $3.10.
- Productive instance deletion: confirmed.
- Final Vast account audit: zero active instances.

No API key, SSH private key, or service token is present in these artifacts.

## Commit ledger

| Commit | Result |
| --- | --- |
| `6853448` | Add bounded Nsight capture to CUDA preflight |
| `ad9f5ef` | Cache StateHead gate activations in CUDA scan |
| `c8847dc` | Fuse StateHead gate bias into CUDA activation; accepted v4 |
| `d71a90f` | Remove flat indexing from StateHead CUDA forward; neutral probe |
| `93fbb4c` | Revert neutral indexing probe |
| `ee6dbe6` | Fuse activation with chunk summaries; slower probe |
| `efd0ca5` | Revert slower activation/summary probe; final source tree |

Each commit was pushed to `origin/codex/statehead-nanochat`.

## Command ledger

This ledger records the commands used for this optimization pass. Repeated
read-only status polls are represented once with their varying instance ID.
Secrets and the temporary Jupyter token returned by the provider are omitted.

### Local repository and brief inspection

```bash
pwd
git rev-parse HEAD
git status --short
git log --oneline --decorate -12
rg --files -g 'AGENTS.md' -g 'STATEHEAD_NANOCHAT_CODEX_BRIEF.md' -g '*.md'
sed -n '1,260p' STATEHEAD_NANOCHAT_CODEX_BRIEF.md
sed -n '1,280p' dev/STATEHEAD_CUDA_V2_PLAN.md
sed -n '1,320p' dev/STATEHEAD_LOG.md
git diff --stat
git diff -- nanochat/statehead.py nanochat/statehead_cuda.py nanochat/csrc/statehead_cuda.cpp nanochat/csrc/statehead_cuda_kernel.cu tests/test_statehead.py dev/statehead_cuda_preflight.py
```

### Vast control plane

```bash
vastai show instances-v1 --raw
vastai search offers 'gpu_name=H100_SXM num_gpus=1 verified=true rentable=true direct_port_count>=1 cuda_max_good>=12.8 dph_total<=10' -o 'dph_total' --raw --limit 20
vastai create instance 43189295 --image 'runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404' --disk 50 --ssh --direct --label statehead-cuda-v4-opt --cancel-unavail --raw
vastai show instance 45639121 --raw
vastai logs 45639121 --tail 200
vastai destroy instance 45639121 -y --raw
vastai create instance 41819339 --image 'vastai/pytorch:@vastai-automatic-tag' --disk 30 --ssh --direct --label statehead-cuda-v4-opt --cancel-unavail --raw
vastai show instance 45639403 --raw
vastai ssh-url 45639403 --raw
vastai attach ssh 45639403
vastai destroy instance 45639403 -y --raw
vastai create instance 36444802 --image 'vastai/pytorch:@vastai-automatic-tag' --disk 30 --ssh --direct --label statehead-cuda-v4-opt --cancel-unavail --raw
vastai show instance 45639678 --raw
vastai ssh-url 45639678 --raw
vastai attach ssh 45639678
vastai destroy instance 45639678 -y --raw
vastai search offers 'gpu_name=H100_SXM num_gpus=1 verified=true rentable=true direct_port_count>=1 cuda_max_good>=12.8 dph_total<=10' -o 'dph_total' --raw --limit 20
vastai create instance 36444807 --image 'vastai/pytorch:@vastai-automatic-tag' --disk 30 --ssh --direct --label statehead-cuda-v4-opt --cancel-unavail --onstart-cmd 'chown -R root:root /root/.ssh && chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys' --raw
vastai show instance 45639924 --raw
vastai ssh-url 45639924 --raw
vastai destroy instance 45639924 -y --raw
vastai show instances-v1 --raw
```

### Instance setup and inspection

```bash
nvidia-smi --query-gpu=name,memory.total,power.limit,driver_version --format=csv,noheader
python3 --version
git clone --branch codex/statehead-nanochat --single-branch https://github.com/samfurr/nanochat.git /root/nanochat
cd /root/nanochat
git rev-parse HEAD
UV_CACHE_DIR=/root/.cache/uv uv sync --extra gpu --group dev --frozen
UV_CACHE_DIR=/root/.cache/uv uv pip install --python .venv/bin/python setuptools
DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-compiler-12-8
DEBIAN_FRONTEND=noninteractive apt-get install -y libcublas-dev-12-8 libcusparse-dev-12-8 libcusolver-dev-12-8
nsys --version
ncu --version
```

All remote CUDA commands used:

```bash
export PATH=/root/nanochat/.venv/bin:/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8
```

### Compilation, parity, and focused tests

```bash
NANOCHAT_DTYPE=float32 NANOCHAT_STATEHEAD_CUDA_VERBOSE=1 TORCH_EXTENSIONS_DIR=/root/torch_extensions_v2 python -m nanochat.statehead_cuda
NANOCHAT_DTYPE=float32 python -m pytest tests/test_statehead.py -q
NANOCHAT_DTYPE=bfloat16 python -m pytest tests/test_statehead.py -q

git pull --ff-only
NANOCHAT_DTYPE=bfloat16 NANOCHAT_STATEHEAD_CUDA_VERBOSE=1 TORCH_EXTENSIONS_DIR=/root/torch_extensions_v3 python -m nanochat.statehead_cuda
NANOCHAT_DTYPE=float32 python -m pytest tests/test_statehead.py -q
NANOCHAT_DTYPE=bfloat16 python -m pytest tests/test_statehead.py -q

git pull --ff-only
NANOCHAT_DTYPE=bfloat16 NANOCHAT_STATEHEAD_CUDA_VERBOSE=1 TORCH_EXTENSIONS_DIR=/root/torch_extensions_v4 python -m nanochat.statehead_cuda
NANOCHAT_DTYPE=float32 python -m pytest tests/test_statehead.py -q
NANOCHAT_DTYPE=bfloat16 python -m pytest tests/test_statehead.py -q

git pull --ff-only
NANOCHAT_DTYPE=bfloat16 NANOCHAT_STATEHEAD_CUDA_VERBOSE=1 TORCH_EXTENSIONS_DIR=/root/torch_extensions_v5 python -m nanochat.statehead_cuda
NANOCHAT_DTYPE=bfloat16 python -m pytest tests/test_statehead.py -q -k 'native_cuda or reverse_chunk'

git pull --ff-only
NANOCHAT_DTYPE=bfloat16 NANOCHAT_STATEHEAD_CUDA_VERBOSE=1 TORCH_EXTENSIONS_DIR=/root/torch_extensions_v6 python -m nanochat.statehead_cuda
NANOCHAT_DTYPE=bfloat16 python -m pytest tests/test_statehead.py -q -k 'native_cuda or reverse_chunk'
```

### Full-step benchmarks

The output path and commit varied exactly as shown:

```bash
NANOCHAT_DTYPE=bfloat16 NANOCHAT_REPO_COMMIT=c1ac745003833655cd36ee762fef31fded66d465 python -m dev.statehead_cuda_preflight --arch statehead --scan-backend cuda --device-batch-size 32 --steps 20 --warmup-steps 5 --scan-chunk-size 16 --output /root/statehead-cuda-v2-k16-us.json
NANOCHAT_DTYPE=bfloat16 NANOCHAT_REPO_COMMIT=c1ac745003833655cd36ee762fef31fded66d465 python -m dev.statehead_cuda_preflight --arch statehead --scan-backend cuda --device-batch-size 32 --steps 20 --warmup-steps 5 --scan-chunk-size 32 --output /root/statehead-cuda-v2-k32-us.json
NANOCHAT_DTYPE=bfloat16 NANOCHAT_REPO_COMMIT=ad9f5efd0ea9f67e36f0aaafb4fb3a132854fb16 python -m dev.statehead_cuda_preflight --arch statehead --scan-backend cuda --device-batch-size 32 --steps 20 --warmup-steps 5 --scan-chunk-size 32 --output /root/statehead-cuda-v3-activation-k32-us.json
NANOCHAT_DTYPE=bfloat16 NANOCHAT_REPO_COMMIT=ad9f5efd0ea9f67e36f0aaafb4fb3a132854fb16 python -m dev.statehead_cuda_preflight --arch statehead --scan-backend cuda --device-batch-size 32 --steps 30 --warmup-steps 5 --scan-chunk-size 32 --output /root/statehead-cuda-v3-activation-k32-us-repeat.json
NANOCHAT_DTYPE=bfloat16 NANOCHAT_REPO_COMMIT=c8847dc31bc4809bbab86bf7c21aea5a13c05662 python -m dev.statehead_cuda_preflight --arch statehead --scan-backend cuda --device-batch-size 32 --steps 30 --warmup-steps 5 --scan-chunk-size 32 --output /root/statehead-cuda-v4-bias-k32-us.json
NANOCHAT_DTYPE=bfloat16 NANOCHAT_REPO_COMMIT=c8847dc31bc4809bbab86bf7c21aea5a13c05662 python -m dev.statehead_cuda_preflight --arch statehead --scan-backend pytorch --device-batch-size 32 --steps 30 --warmup-steps 5 --scan-chunk-size 32 --output /root/statehead-pytorch-k32-us-control.json
NANOCHAT_DTYPE=bfloat16 NANOCHAT_REPO_COMMIT=c8847dc31bc4809bbab86bf7c21aea5a13c05662 python -m dev.statehead_cuda_preflight --arch gpt --scan-backend pytorch --device-batch-size 32 --steps 30 --warmup-steps 5 --output /root/gpt-fa3-us-control.json
NANOCHAT_DTYPE=bfloat16 NANOCHAT_REPO_COMMIT=d71a90fe5caabd41cf0f05d58b3215d9fd23d527 python -m dev.statehead_cuda_preflight --arch statehead --scan-backend cuda --device-batch-size 32 --steps 30 --warmup-steps 5 --scan-chunk-size 32 --output /root/statehead-cuda-v5-grid-k32-us.json
NANOCHAT_DTYPE=bfloat16 NANOCHAT_REPO_COMMIT=ee6dbe633f23e17fdaf9674eafdf470b86ce7bd3 python -m dev.statehead_cuda_preflight --arch statehead --scan-backend cuda --device-batch-size 32 --steps 30 --warmup-steps 5 --scan-chunk-size 32 --output /root/statehead-cuda-v6-fused-summary-k32-us.json
NANOCHAT_DTYPE=bfloat16 NANOCHAT_REPO_COMMIT=efd0ca5de83eb16ed54be6a548b5abddbfe44532 python -m dev.statehead_cuda_preflight --arch statehead --scan-backend cuda --device-batch-size 32 --steps 5 --warmup-steps 1 --scan-chunk-size 32 --verify-gradients --output /root/statehead-cuda-v4-final-gradient-stability.json
```

### Nsight profiling

```bash
NANOCHAT_DTYPE=bfloat16 NANOCHAT_REPO_COMMIT=c1ac745003833655cd36ee762fef31fded66d465 nsys profile --trace=cuda,nvtx,osrt,cublas --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true --output=/root/statehead-k32-current python -m dev.statehead_cuda_preflight --arch statehead --scan-backend cuda --device-batch-size 32 --steps 10 --warmup-steps 5 --scan-chunk-size 32 --nsys-capture --output /root/statehead-k32-profile-run.json
nsys stats --report cuda_gpu_kern_sum --format csv --output /root/statehead-k32-v2-kernels /root/statehead-k32-current.nsys-rep
NANOCHAT_DTYPE=bfloat16 NANOCHAT_REPO_COMMIT=ad9f5efd0ea9f67e36f0aaafb4fb3a132854fb16 nsys profile --trace=cuda,nvtx,osrt,cublas --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true --output=/root/statehead-k32-v3-activation python -m dev.statehead_cuda_preflight --arch statehead --scan-backend cuda --device-batch-size 32 --steps 10 --warmup-steps 5 --scan-chunk-size 32 --nsys-capture --output /root/statehead-k32-v3-profile-run.json
nsys stats --report cuda_gpu_kern_sum --format csv --output /root/statehead-k32-v3-kernels /root/statehead-k32-v3-activation.nsys-rep
NANOCHAT_DTYPE=bfloat16 NANOCHAT_REPO_COMMIT=c8847dc31bc4809bbab86bf7c21aea5a13c05662 nsys profile --trace=cuda,nvtx,osrt,cublas --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true --output=/root/statehead-k32-v4-bias python -m dev.statehead_cuda_preflight --arch statehead --scan-backend cuda --device-batch-size 32 --steps 10 --warmup-steps 5 --scan-chunk-size 32 --nsys-capture --output /root/statehead-k32-v4-profile-run.json
nsys stats --report cuda_gpu_kern_sum --format csv --output /root/statehead-k32-v4-kernels /root/statehead-k32-v4-bias.nsys-rep
ncu --target-processes all --kernel-name regex:statehead_backward_grad_kernel --set full --csv --log-file /root/ncu-backward-grad.csv python -m dev.statehead_cuda_preflight --arch statehead --scan-backend cuda --device-batch-size 32 --steps 7 --warmup-steps 5 --scan-chunk-size 32 --output /root/ncu-backward-grad-run.json
```

### Artifact capture, local verification, and git

```bash
sha256sum /root/<artifact>
tar -C /root -czf /root/statehead-cuda-v4-optimization-20260723.tar.gz <artifact-list>
sha256sum /root/statehead-cuda-v4-optimization-20260723.tar.gz
scp -P <port> root@<host>:/root/statehead-cuda-v4-optimization-20260723.tar.gz /tmp/statehead-cuda-v4-optimization-20260723.tar.gz
shasum -a 256 /tmp/statehead-cuda-v4-optimization-20260723.tar.gz
tar -xzf /tmp/statehead-cuda-v4-optimization-20260723.tar.gz -C dev/results/statehead-cuda-v4-optimization-20260723/
git diff --exit-code c8847dc..efd0ca5 -- nanochat/csrc/statehead_cuda_kernel.cu nanochat/csrc/statehead_cuda.cpp nanochat/statehead.py nanochat/statehead_cuda.py tests/test_statehead.py dev/statehead_cuda_preflight.py
NANOCHAT_DTYPE=bfloat16 .venv/bin/python -m pytest tests/test_statehead.py -q
.venv/bin/python -c "import torch; print(torch.backends.mps.is_available(), torch.backends.mps.is_built())"
NANOCHAT_DTYPE=bfloat16 .venv/bin/python -m scripts.base_train --arch=statehead --statehead-scan-backend=pytorch --depth=1 --sequence-length=32 --device-batch-size=1 --total-batch-size=32 --num-iterations=5 --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --save-every=-1 --run=dummy --input-bin=tests/fixtures/tiny_shard.bin --output-dir=/tmp/nanochat-statehead-v4-mps-smoke
git add dev/statehead_cuda_preflight.py nanochat/csrc/statehead_cuda.cpp nanochat/csrc/statehead_cuda_kernel.cu nanochat/statehead.py nanochat/statehead_cuda.py tests/test_statehead.py
git commit -m '<commit message>'
git push origin codex/statehead-nanochat
```

The archive SHA-256 matched on the instance and locally:
`344ff2a97ff05fbe7ae1a1d846964f703afe08b253f8d62a29f7d2917f8dfe64`.
The temporary archive was removed after extraction because every underlying
artifact is retained separately with its own checksum.

## Artifacts

This directory contains:

- all benchmark and finite-gradient JSON outputs;
- FP32/BF16/focused pytest logs;
- native CUDA compile/load logs for each measured revision;
- Nsight Systems reports and kernel summary CSV files;
- the Nsight Compute permission diagnostic;
- the hardware/software environment record;
- this report and `artifacts.sha256`.

## Unresolved questions

1. Does CUDA v4 match compiled PyTorch StateHead's learning trajectory on a
   short, dataset-backed run?
2. Can a CUDA-native gate projection or CUTLASS epilogue remove the 5.675 ms
   standalone activation pass without slowing the tensor-core GEMM?
3. Can a K32/Dh128 production specialization recover the additional roughly
   1 ms needed after activation fusion to exceed 2x GPT?
4. Does native FP8 on only eligible projection GEMMs remain numerically stable
   and faster while recurrence stays FP32?
5. Does CUDA v4 compile and scale correctly under eight-GPU DDP?
6. What do occupancy, achieved-bandwidth, register-pressure, and shared-memory
   counters show on a host that permits Nsight Compute performance counters?
7. How stable are the same-host throughput ratios over multiple seeds and
   longer timing windows?

## Exact next recommended command

After implementing the CUDA-native gate-projection activation epilogue as a
separately revertible v7 probe, run this exact same-shape verification before
any broad tuning or paid full run:

```bash
NANOCHAT_DTYPE=bfloat16 NANOCHAT_REPO_COMMIT="$(git rev-parse HEAD)" python -m dev.statehead_cuda_preflight --arch statehead --scan-backend cuda --device-batch-size 32 --steps 30 --warmup-steps 5 --scan-chunk-size 32 --verify-gradients --output /root/statehead-cuda-v7-gate-epilogue-k32.json
```
