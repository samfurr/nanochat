# StateHead CUDA projection/activation optimization pass

Date: 2026-07-23

Recorded starting commit:
`ffef4251339104c1257c934ef2cbc593fc62581e`

Recorded starting status:

```text
?? .DS_Store
?? .agents/
?? STATEHEAD_NANOCHAT_CODEX_BRIEF.md
```

Those pre-existing untracked paths are user-owned and were not modified.

Final source commit before this report:
`a681dbb3869784cef44f12e9fae76546f57cb443`

Branch: `codex/statehead-nanochat`

Hardware/runtime: one verified 700 W NVIDIA H100 80GB HBM3 (H100
SXM), driver 580.126.20, PyTorch 2.9.1+cu128, CUDA runtime/compiler
12.8, and SM90.

## Verdict

- No new implementation beat CUDA v4. CUDA v4 remains the accepted default.
- The final active CUDA, Python, CLI, preflight, and test source is
  byte-for-byte identical to the starting v4 source at `ffef425`.
- Row tiling, output-feature/gate splitting, and fast transcendental
  approximations were all measured and rejected.
- The best row-tiled result was 940,540 tok/s, 0.87% below the same-session
  short v4 control at 948,838 tok/s.
- Four independent gate projections measured 884,133 tok/s, 6.82% below v4.
- Two `[2D,D]` gate groups measured 855,941 tok/s, 9.79% below v4.
- Direct `__expf` plus `__tanhf` appeared 0.20% faster in a short probe, but
  paired 30-step runs measured 941,874 tok/s versus 942,686 tok/s for exact
  v4. It was neutral-to-slower at -0.09% and was reverted.
- Nsight Systems confirms the 32,768-row pipeline still executes 5.72 ms of
  activation kernels per optimizer step. Overlap does not remove the work, and
  smaller GEMMs lose more efficiency than the overlap recovers.
- The next plausible material optimization is a true custom tensor-core GEMM
  epilogue that applies the mixed sigmoid/tanh activation while the accumulator
  tile is still resident. Another multistream overlap arrangement is not
  justified by these results.
- The Transformer/GPT source and behavior were not changed.

No parity claim is made beyond the exact tests listed below.

## Performance

All comparison rows use the same H100, d12/768/6-head StateHead,
sequence length 2048, device batch 32, BF16, `torch.compile`,
Muon/AdamW, and a fixed synthetic token batch.

Short probes have seven measured steps after five warmup steps. Formal runs
have 25 measured steps after five warmup steps.

| Variant | Commit | Measured steps | Median tok/s | Median step | Peak GiB | Relative control |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CUDA v4 short control | `db1c556` | 7 | 948,838 | 69.070 ms | 13.15 | control |
| v7 row tiles, 16,384 | `db1c556` | 7 | 934,513 | 70.129 ms | 13.65 | -1.51% |
| v7 row tiles, 32,768 | `db1c556` | 7 | 940,540 | 69.679 ms | 13.65 | -0.87% |
| v8 four gate projections | `b4b3652` | 7 | 884,133 | 74.125 ms | 14.15 | -6.82% |
| v9 two `[2D,D]` groups | `f9a4b7c` | 7 | 855,941 | 76.566 ms | 14.15 | -9.79% |
| v10 algebraic fast tanh | `8e02099` | 7 | 940,096 | 69.712 ms | 13.15 | -0.92% |
| v10b direct fast tanh, short | `669d63f` | 7 | 950,764 | 68.930 ms | 13.15 | +0.20% |
| CUDA v4 formal control | `db1c556` | 25 | 942,686 | 69.520 ms | 13.15 | control |
| v10b direct fast tanh, formal | `669d63f` | 25 | 941,874 | 69.580 ms | 13.15 | -0.09% |

The five-step final finite-gradient smoke is intentionally excluded from the
timing table because gradient inspection changes the timed workload.

## What the memory layout experiments established

The current `[B*T,4D]` full gate projection is large enough to use H100 tensor
cores efficiently. The existing weight and output layouts are already
gate-major inside each token:

```text
weight: [4D,D] = [retention | input | candidate | output]
output: [B,T,4,H,Dh]
```

Two attempted overlap dimensions both failed:

1. Token-row tiling preserved the full `4D` output width, but replacing one
   large GEMM with two or four smaller GEMMs cost more than overlapping the
   activation saved.
2. Gate/output-feature splitting matched the checkpoint layout and avoided a
   transpose, but `[D,D]` and `[2D,D]` GEMMs lost even more tensor-core
   efficiency. Concurrent execution also increased peak allocation.

The layout should therefore remain unchanged for the next attempt. The
activation must be fused into the large GEMM's output epilogue rather than
hidden behind a split GEMM.

## Profiling

Nsight Systems captured three post-warmup optimizer steps for the 32,768-row
v7 pipeline.

The activation kernel ran 72 times:

```text
2 row tiles * 12 layers * 3 measured steps = 72 launches
```

It consumed 17.164 ms across the capture, or 5.721 ms per optimizer step.
That is effectively the same activation cost as CUDA v4's earlier 5.675
ms/step profile. The trace and kernel summary are included in this directory.

The result rejects the premise that multistream overlap alone is enough. A
real epilogue fusion must remove the activation read/write pass.

## Correctness and stability

| Gate | Result | Scope |
| --- | --- | --- |
| Final CUDA 12.8 extension compile/load | Pass | exact final commit `a681dbb` |
| Final focused FP32 CUDA suite | 45 passed, 44 deselected | numerical, raw-gate/initial-state gradients, partial chunks, sequence 2048, fullgraph, compiled full-model parity |
| Final focused BF16 CUDA suite | 45 passed, 44 deselected | same scope as FP32 |
| Final complete BF16 StateHead suite | 87 passed, 2 unrelated failures | failures are GPT CPU checkpoint tests routed to CUDA-only FA3 |
| Final production-shape finite-gradient smoke | Pass | five compiled BF16 optimizer steps, all parameter gradients finite |
| Final local StateHead suite | 53 passed, 36 CUDA-skipped | macOS local environment |
| Local BF16/MPS five-step smoke | Pass | loss 4.158032 to 2.876626, all finite |
| v7 focused BF16 suite | 48 passed, 44 deselected | row-tiled projected backend plus v4 |
| v7 focused FP32 suite | 48 passed, 44 deselected | row-tiled projected backend plus v4 |
| v8/v9 focused BF16 suite | 49 passed, 44 deselected | gate-group backend plus v7/v4 |
| v10 focused FP32 suite | 49 passed, 44 deselected | fast activation plus all experimental backends |
| v10 focused BF16 suite | 49 passed, 44 deselected | fast activation plus all experimental backends |
| v10b focused FP32 suite | 49 passed, 44 deselected | direct fast tanh |
| v10b focused BF16 suite | 49 passed, 44 deselected | direct fast tanh |

The complete BF16 failures are:

```text
tests/test_statehead.py::test_checkpoint_round_trip[legacy-gpt]
tests/test_statehead.py::test_checkpoint_round_trip[gpt]
```

With FA3 available globally, those two CPU GPT tests dispatch a CPU tensor to
the CUDA-only FA3 operation. They do not execute StateHead or any changed
kernel.

The fixed synthetic loss spike at step 6 appears in both experimental and
v4-control runs at the same magnitude. It is part of this short fixed-batch
Muon trajectory, not evidence of an experimental-path-only instability.

Dataset-backed learning parity, native FP8 stability/speed, and eight-GPU DDP
were not run in this pass and are not claimed.

## Changed files

Files modified during experimental commits:

- `dev/STATEHEAD_CUDA_V2_PLAN.md`
- `dev/statehead_cuda_preflight.py`
- `nanochat/csrc/statehead_cuda.cpp`
- `nanochat/csrc/statehead_cuda_kernel.cu`
- `nanochat/statehead.py`
- `nanochat/statehead_cuda.py`
- `scripts/base_train.py`
- `tests/test_statehead.py`

At final commit `a681dbb`, all except
`dev/STATEHEAD_CUDA_V2_PLAN.md` are byte-for-byte restored to starting commit
`ffef425`.

New evidence files:

- `dev/results/statehead-cuda-v7-v10-optimization-20260723/REPORT.md`
- `dev/results/statehead-cuda-v7-v10-optimization-20260723/environment.txt`
- `dev/results/statehead-cuda-v7-v10-optimization-20260723/artifacts.sha256`
- `dev/results/statehead-cuda-v7-v10-optimization-20260723/statehead-cuda-v4-control-v7host-probe.json`
- `dev/results/statehead-cuda-v7-v10-optimization-20260723/statehead-cuda-v4-control-v7host-formal.json`
- `dev/results/statehead-cuda-v7-v10-optimization-20260723/statehead-cuda-v4-final-v7pass-gradient-stability.json`
- `dev/results/statehead-cuda-v7-v10-optimization-20260723/statehead-cuda-v7-projected-k32-tile16384-probe.json`
- `dev/results/statehead-cuda-v7-v10-optimization-20260723/statehead-cuda-v7-projected-k32-tile32768-probe.json`
- `dev/results/statehead-cuda-v7-v10-optimization-20260723/statehead-v7-projected-tile32768-profile-run.json`
- `dev/results/statehead-cuda-v7-v10-optimization-20260723/statehead-v7-projected-tile32768-profile.nsys-rep`
- `dev/results/statehead-cuda-v7-v10-optimization-20260723/statehead-v7-projected-tile32768-kernels.csv`
- `dev/results/statehead-cuda-v7-v10-optimization-20260723/statehead-cuda-v8-gate-parallel-k32-probe.json`
- `dev/results/statehead-cuda-v7-v10-optimization-20260723/statehead-cuda-v9-two-groups-k32-probe.json`
- `dev/results/statehead-cuda-v7-v10-optimization-20260723/statehead-cuda-v10-fast-activation-k32-probe.json`
- `dev/results/statehead-cuda-v7-v10-optimization-20260723/statehead-cuda-v10b-direct-fast-tanh-k32-probe.json`
- `dev/results/statehead-cuda-v7-v10-optimization-20260723/statehead-cuda-v10b-direct-fast-tanh-k32-formal.json`

`nanochat/gpt.py` was not changed.

## Commit ledger

| Commit | Result |
| --- | --- |
| `a2d8ad5` | Prototype row-tiled projection/activation pipeline |
| `db1c556` | Replace unavailable PyTorch-internal GEMM symbol with linked cuBLAS C API |
| `b4b3652` | Four gate-parallel projection probe |
| `f9a4b7c` | Two gate-group projection probe |
| `8e02099` | Fast sigmoid plus algebraic fast-tanh probe |
| `669d63f` | Direct CUDA fast-tanh probe |
| `a681dbb` | Remove all slower probes and restore active v4 source |

Every commit was pushed to `origin/codex/statehead-nanochat`.

## Cost and lifecycle

- Approved ceiling for this pass: $10.
- Vast offer: `36444802`.
- Vast instance: `45643523`.
- Rate: $3.415/hour including disk.
- Runtime: 2,473.13 seconds, or 41.22 minutes.
- Estimated charge: $2.35.
- Estimated cumulative charge including the preceding $3.10 CUDA-v4
  optimization pass: approximately $5.45.
- Instance destruction: confirmed.
- Final Vast audit: zero active instances and zero volumes.

No API key, SSH private key, Jupyter token, or other secret is present in this
evidence bundle.

## Unresolved questions

1. Can CUTLASS 3.x EVT or a small custom epilogue visitor express one tanh gate
   and three sigmoid gates while writing the current `[B,T,4,H,Dh]` layout,
   without reducing the full `[B*T,D] x [D,4D]` tensor-core GEMM efficiency?
2. Does eliminating the measured 5.7 ms activation pass translate into the
   predicted full-step improvement once epilogue register pressure and
   occupancy are included?
3. After a speed winner exists, does it preserve dataset-backed learning
   behavior over a meaningful training window?
4. FP8 projection/output GEMMs and eight-H100 DDP remain later gates; neither
   should begin until the BF16 epilogue path is correct and faster.
5. The two GPT/FA3 CPU checkpoint-test failures should eventually be isolated
   from global FA3 selection, but they are unrelated to StateHead CUDA.

## Exact next recommended command

Begin the epilogue work on a clean branch rooted at the verified final commit:

```bash
git switch -c codex/statehead-cutlass-epilogue a681dbb3869784cef44f12e9fae76546f57cb443
```

Do not provision another H100 until that branch has a compilable CUTLASS/custom
epilogue prototype and CPU/fake-dispatch tests pass locally.

## Command ledger

The list below records every shell command used for this optimization pass.
Repeated `write_stdin` polling of already-running commands is not a shell
command and is not listed. Secrets returned by provider APIs are omitted.

### Initial repository, brief, skill, memory, and source inspection

```bash
pwd
git rev-parse HEAD
git status --short
wc -l STATEHEAD_NANOCHAT_CODEX_BRIEF.md
sed -n '1,780p' STATEHEAD_NANOCHAT_CODEX_BRIEF.md
sed -n '781,1200p' STATEHEAD_NANOCHAT_CODEX_BRIEF.md
cat /Users/haybales/.codex/skills/vastai/SKILL.md
rg -n "statehead-speed|full_step|CUDA|nanochat" /Users/haybales/.codex/memories/MEMORY.md
rg --files
rg -n "projection|activation|scan_backend|statehead_cuda" nanochat dev scripts tests
git log --oneline --decorate -20
git diff --exit-code HEAD -- nanochat/gpt.py
sed -n '1,240p' nanochat/csrc/statehead_cuda_kernel.cu
sed -n '240,520p' nanochat/csrc/statehead_cuda_kernel.cu
sed -n '520,860p' nanochat/csrc/statehead_cuda_kernel.cu
sed -n '1,380p' nanochat/statehead_cuda.py
sed -n '1,220p' nanochat/csrc/statehead_cuda.cpp
sed -n '1,205p' nanochat/statehead.py
sed -n '300,420p' tests/test_statehead.py
sed -n '35,145p' dev/statehead_cuda_preflight.py
sed -n '45,140p;160,190p;268,290p' scripts/base_train.py
tail -180 dev/STATEHEAD_CUDA_V2_PLAN.md
```

### Local verification and git operations

```bash
NANOCHAT_DTYPE=bfloat16 .venv/bin/python -m pytest tests/test_statehead.py -q
.venv/bin/python -m compileall -q nanochat scripts dev tests
.venv/bin/python -m dev.statehead_cuda_preflight --help
git diff --check
git diff --exit-code HEAD -- nanochat/gpt.py
git status --short
git diff --stat
git add nanochat/csrc/statehead_cuda_kernel.cu nanochat/statehead_cuda.py
git commit -m 'Link projected gate path through cuBLAS'
git push origin codex/statehead-nanochat
git add dev/STATEHEAD_CUDA_V2_PLAN.md dev/statehead_cuda_preflight.py nanochat/csrc/statehead_cuda.cpp nanochat/csrc/statehead_cuda_kernel.cu nanochat/statehead.py nanochat/statehead_cuda.py scripts/base_train.py tests/test_statehead.py
git commit -m 'Probe gate-parallel StateHead projection'
git push origin codex/statehead-nanochat
git add dev/STATEHEAD_CUDA_V2_PLAN.md nanochat/csrc/statehead_cuda.cpp nanochat/csrc/statehead_cuda_kernel.cu nanochat/statehead_cuda.py
git commit -m 'Try two-group StateHead projection overlap'
git push origin codex/statehead-nanochat
git add dev/STATEHEAD_CUDA_V2_PLAN.md nanochat/csrc/statehead_cuda_kernel.cu
git commit -m 'Probe fast StateHead gate activations'
git push origin codex/statehead-nanochat
git add dev/STATEHEAD_CUDA_V2_PLAN.md nanochat/csrc/statehead_cuda_kernel.cu
git commit -m 'Use direct fast tanh in StateHead activation'
git push origin codex/statehead-nanochat
git diff --exit-code ffef4251339104c1257c934ef2cbc593fc62581e -- dev/statehead_cuda_preflight.py nanochat/csrc/statehead_cuda.cpp nanochat/csrc/statehead_cuda_kernel.cu nanochat/statehead.py nanochat/statehead_cuda.py scripts/base_train.py tests/test_statehead.py
git add dev/STATEHEAD_CUDA_V2_PLAN.md dev/statehead_cuda_preflight.py nanochat/csrc/statehead_cuda.cpp nanochat/csrc/statehead_cuda_kernel.cu nanochat/statehead.py nanochat/statehead_cuda.py scripts/base_train.py tests/test_statehead.py
git commit -m 'Revert slower StateHead projection probes'
git push origin codex/statehead-nanochat
```

### Local MPS smoke

The sandbox device query failed because the sandbox reports a synthetic macOS
version. The same command was rerun on the host:

```bash
NANOCHAT_DTYPE=bfloat16 .venv/bin/python - <<'PY'
import torch
from nanochat.statehead import StateHead, StateHeadConfig
print("mps_available", torch.backends.mps.is_available())
torch.manual_seed(123)
config = StateHeadConfig(sequence_len=16, vocab_size=64, n_layer=1, n_head=4, n_embd=32)
model = StateHead(config).to("mps")
model.init_weights()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
tokens = torch.randint(0, 64, (2, 16), device="mps")
targets = torch.roll(tokens, -1, 1)
for step in range(5):
    optimizer.zero_grad(set_to_none=True)
    loss = model(tokens, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    print(step, float(loss))
torch.mps.synchronize()
PY
```

### Vast provisioning and setup

```bash
vastai show instances --raw
vastai show volumes --raw
vastai search offers 'gpu_name=H100_SXM num_gpus=1 verified=true rentable=true direct_port_count>=1 cuda_max_good>=12.8 dph_total<=4' --order dph_total --raw
vastai create instance 36444802 --image 'vastai/pytorch:@vastai-automatic-tag' --disk 30 --ssh --direct --label statehead-cuda-v7-probe --cancel-unavail --onstart-cmd 'chown -R root:root /root/.ssh && chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys' --raw
vastai show instances --raw
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 7008 root@192.222.55.173 'nvidia-smi --query-gpu=name,memory.total,power.limit,driver_version --format=csv,noheader'
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 7008 root@192.222.55.173 'git clone --branch codex/statehead-nanochat https://github.com/samfurr/nanochat.git /root/nanochat'
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 7008 root@192.222.55.173 'cd /root/nanochat && UV_CACHE_DIR=/root/.cache/uv uv sync --extra gpu --group dev --frozen'
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 7008 root@192.222.55.173 'cd /root/nanochat && uv pip install setuptools'
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 7008 root@192.222.55.173 'DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-compiler-12-8 libcublas-dev-12-8 libcusparse-dev-12-8 libcusolver-dev-12-8'
```

### CUDA compilation and linkage diagnosis

All compilation commands used:

```bash
cd /root/nanochat
export PATH=/root/nanochat/.venv/bin:/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_EXTENSIONS_DIR=/root/torch_extensions_v7
NANOCHAT_DTYPE=bfloat16 NANOCHAT_STATEHEAD_CUDA_VERBOSE=1 python -m nanochat.statehead_cuda
```

The first link loaded with an undefined PyTorch-internal
`at::cuda::blas::gemm<double>` symbol. The diagnostic commands were:

```bash
for f in /root/nanochat/.venv/lib/python3.10/site-packages/torch/lib/*.so; do if nm -D "$f" 2>/dev/null | grep -q "_ZN2at4cuda4blas4gemmId"; then echo "$f"; nm -D "$f" | grep "_ZN2at4cuda4blas4gemmId" | head; fi; done
nm -D -C /root/nanochat/.venv/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so 2>/dev/null | grep "at::cuda::blas::gemm" | head -30
nm -D -C /root/nanochat/.venv/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so 2>/dev/null | grep -E "getCurrentCUDABlasHandle|getCurrentCUDAStream|getStreamFromPool" | head -30
```

After switching the experiment to the cuBLAS C API, the following exact
extension caches were compiled:

```bash
TORCH_EXTENSIONS_DIR=/root/torch_extensions_v7 NANOCHAT_DTYPE=bfloat16 NANOCHAT_STATEHEAD_CUDA_VERBOSE=1 python -m nanochat.statehead_cuda
TORCH_EXTENSIONS_DIR=/root/torch_extensions_v8 NANOCHAT_DTYPE=bfloat16 NANOCHAT_STATEHEAD_CUDA_VERBOSE=1 python -m nanochat.statehead_cuda
TORCH_EXTENSIONS_DIR=/root/torch_extensions_v9 NANOCHAT_DTYPE=bfloat16 NANOCHAT_STATEHEAD_CUDA_VERBOSE=1 python -m nanochat.statehead_cuda
TORCH_EXTENSIONS_DIR=/root/torch_extensions_v10 NANOCHAT_DTYPE=bfloat16 NANOCHAT_STATEHEAD_CUDA_VERBOSE=1 python -m nanochat.statehead_cuda
TORCH_EXTENSIONS_DIR=/root/torch_extensions_v10b NANOCHAT_DTYPE=bfloat16 NANOCHAT_STATEHEAD_CUDA_VERBOSE=1 python -m nanochat.statehead_cuda
TORCH_EXTENSIONS_DIR=/root/torch_extensions_final NANOCHAT_DTYPE=bfloat16 NANOCHAT_STATEHEAD_CUDA_VERBOSE=1 python -m nanochat.statehead_cuda
```

Each remote revision change used:

```bash
git pull --ff-only
git rev-parse HEAD
git status --short
```

The paired v4 control temporarily used:

```bash
git checkout --detach db1c5569b9b14dbe881bb7934a2f25414d646ab3
git checkout codex/statehead-nanochat
git pull --ff-only
```

### CUDA correctness commands

```bash
NANOCHAT_DTYPE=bfloat16 python -m pytest tests/test_statehead.py -q -k "projected or native_cuda or reverse_chunk"
NANOCHAT_DTYPE=float32 python -m pytest tests/test_statehead.py -q -k "projected or native_cuda or reverse_chunk"
NANOCHAT_DTYPE=bfloat16 python -m pytest tests/test_statehead.py -q
NANOCHAT_DTYPE=float32 python -m pytest tests/test_statehead.py -q -k "projected or native_cuda or reverse_chunk"
NANOCHAT_DTYPE=bfloat16 python -m pytest tests/test_statehead.py -q -k "projected or native_cuda or reverse_chunk"
NANOCHAT_DTYPE=bfloat16 python -m pytest tests/test_statehead.py -q -k "projected or native_cuda or reverse_chunk"
NANOCHAT_DTYPE=float32 python -m pytest tests/test_statehead.py -q -k "projected or native_cuda or reverse_chunk"
NANOCHAT_DTYPE=bfloat16 python -m pytest tests/test_statehead.py -q -k "projected or native_cuda or reverse_chunk"
NANOCHAT_DTYPE=float32 python -m pytest tests/test_statehead.py -q -k "projected or native_cuda or reverse_chunk"
NANOCHAT_DTYPE=bfloat16 python -m pytest tests/test_statehead.py -q -k "projected or native_cuda or reverse_chunk"
NANOCHAT_DTYPE=float32 python -m pytest tests/test_statehead.py -q -k "native_cuda or reverse_chunk"
NANOCHAT_DTYPE=bfloat16 python -m pytest tests/test_statehead.py -q -k "native_cuda or reverse_chunk"
NANOCHAT_DTYPE=bfloat16 python -m pytest tests/test_statehead.py -q
```

### H100 benchmark and profile commands

Every benchmark used this common prefix:

```bash
cd /root/nanochat
export PATH=/root/nanochat/.venv/bin:/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8
```

The full benchmark commands were:

```bash
NANOCHAT_DTYPE=bfloat16 NANOCHAT_REPO_COMMIT=db1c5569b9b14dbe881bb7934a2f25414d646ab3 python -m dev.statehead_cuda_preflight --arch statehead --scan-backend cuda_projected --projection-tile-rows 16384 --device-batch-size 32 --steps 12 --warmup-steps 5 --scan-chunk-size 32 --output /root/statehead-cuda-v7-projected-k32-tile16384-probe.json
NANOCHAT_DTYPE=bfloat16 NANOCHAT_REPO_COMMIT=db1c5569b9b14dbe881bb7934a2f25414d646ab3 python -m dev.statehead_cuda_preflight --arch statehead --scan-backend cuda --device-batch-size 32 --steps 12 --warmup-steps 5 --scan-chunk-size 32 --output /root/statehead-cuda-v4-control-v7host-probe.json
NANOCHAT_DTYPE=bfloat16 NANOCHAT_REPO_COMMIT=db1c5569b9b14dbe881bb7934a2f25414d646ab3 python -m dev.statehead_cuda_preflight --arch statehead --scan-backend cuda_projected --projection-tile-rows 32768 --device-batch-size 32 --steps 12 --warmup-steps 5 --scan-chunk-size 32 --output /root/statehead-cuda-v7-projected-k32-tile32768-probe.json
NANOCHAT_DTYPE=bfloat16 NANOCHAT_REPO_COMMIT=b4b3652feb9d4101ea59849603528b03dcbb769c python -m dev.statehead_cuda_preflight --arch statehead --scan-backend cuda_projected_gates --device-batch-size 32 --steps 12 --warmup-steps 5 --scan-chunk-size 32 --output /root/statehead-cuda-v8-gate-parallel-k32-probe.json
NANOCHAT_DTYPE=bfloat16 NANOCHAT_REPO_COMMIT=f9a4b7cd0f69e522a7aedfae9d3cb840f400d3d8 python -m dev.statehead_cuda_preflight --arch statehead --scan-backend cuda_projected_gates --device-batch-size 32 --steps 12 --warmup-steps 5 --scan-chunk-size 32 --output /root/statehead-cuda-v9-two-groups-k32-probe.json
NANOCHAT_DTYPE=bfloat16 NANOCHAT_REPO_COMMIT=8e02099f6ac04b6d001dbabecc7d20b42a1b9d0e python -m dev.statehead_cuda_preflight --arch statehead --scan-backend cuda --device-batch-size 32 --steps 12 --warmup-steps 5 --scan-chunk-size 32 --output /root/statehead-cuda-v10-fast-activation-k32-probe.json
NANOCHAT_DTYPE=bfloat16 NANOCHAT_REPO_COMMIT=669d63f5592aa3cc197e6dcb4af611c59fd20f61 python -m dev.statehead_cuda_preflight --arch statehead --scan-backend cuda --device-batch-size 32 --steps 12 --warmup-steps 5 --scan-chunk-size 32 --output /root/statehead-cuda-v10b-direct-fast-tanh-k32-probe.json
NANOCHAT_DTYPE=bfloat16 NANOCHAT_REPO_COMMIT=669d63f5592aa3cc197e6dcb4af611c59fd20f61 python -m dev.statehead_cuda_preflight --arch statehead --scan-backend cuda --device-batch-size 32 --steps 30 --warmup-steps 5 --scan-chunk-size 32 --output /root/statehead-cuda-v10b-direct-fast-tanh-k32-formal.json
NANOCHAT_DTYPE=bfloat16 NANOCHAT_REPO_COMMIT=db1c5569b9b14dbe881bb7934a2f25414d646ab3 python -m dev.statehead_cuda_preflight --arch statehead --scan-backend cuda --device-batch-size 32 --steps 30 --warmup-steps 5 --scan-chunk-size 32 --output /root/statehead-cuda-v4-control-v7host-formal.json
NANOCHAT_DTYPE=bfloat16 NANOCHAT_REPO_COMMIT=a681dbb3869784cef44f12e9fae76546f57cb443 python -m dev.statehead_cuda_preflight --arch statehead --scan-backend cuda --device-batch-size 32 --steps 5 --warmup-steps 1 --scan-chunk-size 32 --verify-gradients --output /root/statehead-cuda-v4-final-v7pass-gradient-stability.json
```

The profile and report commands were:

```bash
nsys --version
NANOCHAT_DTYPE=bfloat16 NANOCHAT_REPO_COMMIT=db1c5569b9b14dbe881bb7934a2f25414d646ab3 nsys profile --trace=cuda,nvtx,osrt,cublas --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true --output=/root/statehead-v7-projected-tile32768-profile python -m dev.statehead_cuda_preflight --arch statehead --scan-backend cuda_projected --projection-tile-rows 32768 --device-batch-size 32 --steps 8 --warmup-steps 5 --scan-chunk-size 32 --nsys-capture --output /root/statehead-v7-projected-tile32768-profile-run.json
nsys stats --report cuda_gpu_kern_sum --format csv /root/statehead-v7-projected-tile32768-profile.nsys-rep
```

### Artifact retrieval, calculations, and lifecycle

```bash
mkdir -p dev/results/statehead-cuda-v7-v10-optimization-20260723
scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P 7008 root@192.222.55.173:/root/statehead-cuda-v7-projected-k32-tile16384-probe.json root@192.222.55.173:/root/statehead-cuda-v4-control-v7host-probe.json root@192.222.55.173:/root/statehead-cuda-v7-projected-k32-tile32768-probe.json root@192.222.55.173:/root/statehead-v7-projected-tile32768-profile-run.json root@192.222.55.173:/root/statehead-v7-projected-tile32768-profile.nsys-rep root@192.222.55.173:/root/statehead-cuda-v8-gate-parallel-k32-probe.json root@192.222.55.173:/root/statehead-cuda-v9-two-groups-k32-probe.json root@192.222.55.173:/root/statehead-cuda-v10-fast-activation-k32-probe.json root@192.222.55.173:/root/statehead-cuda-v10b-direct-fast-tanh-k32-probe.json root@192.222.55.173:/root/statehead-cuda-v10b-direct-fast-tanh-k32-formal.json root@192.222.55.173:/root/statehead-cuda-v4-control-v7host-formal.json root@192.222.55.173:/root/statehead-cuda-v4-final-v7pass-gradient-stability.json dev/results/statehead-cuda-v7-v10-optimization-20260723/
for f in dev/results/statehead-cuda-v7-v10-optimization-20260723/*.json; do jq -r '[input_filename, .repo_commit, .config.scan_backend, (.config.projection_tile_rows // "n/a"), .steady_state.global_tokens_per_second_median, .steady_state.seconds_median_max_rank, .peak_allocated_bytes_max_rank, .gradient_finiteness_checked] | @tsv' "$f"; done | sort
jq -n '{v7_16_pct:((934512.5698588036/948837.7625893548-1)*100),v7_32_pct:((940539.5341893024/948837.7625893548-1)*100),v8_pct:((884133.3240265113/948837.7625893548-1)*100),v9_pct:((855941.357466745/948837.7625893548-1)*100),v10_pct:((940096.1207345967/948837.7625893548-1)*100),v10b_formal_pct:((941873.9015089365/942686.4688434418-1)*100),v4_gib:(14117856256/1073741824),v7_gib:(14656300032/1073741824),v8_gib:(15193170944/1073741824),cost:((2473.125077009201/3600)*3.415)}'
vastai show instances --raw
vastai destroy instance 45643523 -y --raw
vastai show instances-v1 --raw
vastai show volumes --raw
shasum -a 256 dev/results/statehead-cuda-v7-v10-optimization-20260723/REPORT.md dev/results/statehead-cuda-v7-v10-optimization-20260723/environment.txt dev/results/statehead-cuda-v7-v10-optimization-20260723/*.json dev/results/statehead-cuda-v7-v10-optimization-20260723/*.csv dev/results/statehead-cuda-v7-v10-optimization-20260723/*.nsys-rep
shasum -a 256 -c dev/results/statehead-cuda-v7-v10-optimization-20260723/artifacts.sha256
git diff --check
git status --short
git rev-parse HEAD
git diff --exit-code ffef4251339104c1257c934ef2cbc593fc62581e -- dev/statehead_cuda_preflight.py nanochat/csrc/statehead_cuda.cpp nanochat/csrc/statehead_cuda_kernel.cu nanochat/statehead.py nanochat/statehead_cuda.py scripts/base_train.py tests/test_statehead.py
git add dev/STATEHEAD_CUDA_V2_PLAN.md dev/results/statehead-cuda-v7-v10-optimization-20260723
git commit -m 'Document rejected StateHead CUDA projection probes'
git push origin codex/statehead-nanochat
git check-ignore -v dev/results/statehead-cuda-v7-v10-optimization-20260723/REPORT.md
git add -f dev/results/statehead-cuda-v7-v10-optimization-20260723/REPORT.md
git add dev/results/statehead-cuda-v7-v10-optimization-20260723/artifacts.sha256
git commit -m 'Add StateHead CUDA optimization report'
git push origin codex/statehead-nanochat
```

One preliminary `jq -n` calculation omitted parentheses around multiplication
inside object values and exited with a syntax error; the corrected command is
the one recorded above.
