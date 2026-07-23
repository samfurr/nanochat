# StateHead CUDA v2 one-H100 verification gate

Date: 2026-07-23

Tested commit: `52dd72e184bf1459009e9d99a4819a2634f872aa`

Hardware/runtime: one verified Vast H100 SXM 80GB, driver 580.95.05,
PyTorch 2.9.1+cu128, CUDA runtime/toolkit 12.8, and SM90.

## Verdict

- Native CUDA 12.8/SM90 compilation and load: pass.
- FP32 StateHead suite: 89 passed.
- Focused BF16 native-CUDA/reverse-summary gate: 45 passed.
- Full BF16 StateHead suite: 87 passed, with two unrelated CPU GPT checkpoint
  failures caused by FA3 dispatching a CPU tensor when BF16/CUDA capability was
  globally visible.
- Compiled full-model StateHead CUDA/PyTorch loss and gradient parity: pass
  through the corresponding tests in the FP32 and focused BF16 gates.
- Full-step speed versus compiled PyTorch StateHead: pass for this synthetic
  one-H100 probe. CUDA v2 chunk 32 was 33.96% faster.
- Full-step speed versus GPT/FA3: pass for this synthetic one-H100 probe. CUDA
  v2 chunk 32 was 55.42% faster.
- Dataset learning parity: not tested and not claimed.
- FP8 stability or speed: not tested in this gate.
- Nsight profiling: not collected because environment setup consumed most of
  the bounded rental window.

## Correctness results

| Gate | Result | Evidence |
| --- | ---: | --- |
| CUDA 12.8 extension compile/load | Pass | `native-cuda-load.log`, compiled extension used by subsequent tests |
| FP32 StateHead suite | 89 passed | `fp32.log`, `fp32.exit` |
| Full BF16 StateHead suite | 87 passed, 2 unrelated failures | `bf16.log`, `bf16.exit` |
| Focused BF16 CUDA/reverse-summary tests | 45 passed, 44 deselected | Command output recorded during the gate |
| FP32/BF16 partial chunks | Pass | CUDA cases include lengths 31, 33, 63, 65, and 129 |
| FP32/BF16 2048-token gradients | Pass | CUDA cases include chunk sizes 32 and 64 |
| `torch.compile(fullgraph=True)` custom op | Pass | `tests/test_statehead.py` native compile test |
| Compiled full-model loss/gradients | Pass | `tests/test_statehead.py` native full-model parity test |

The two full-BF16 failures were
`test_checkpoint_round_trip[legacy-gpt]` and
`test_checkpoint_round_trip[gpt]`. Both instantiate GPT on CPU and then call
the CUDA-only `flash_attn_3::_flash_attn_forward`. They do not execute
StateHead, the native scan, or the changed backward kernels.

## Full-step performance

All rows use the same H100, commit, d12/768/2048 shape, batch size 32,
PyTorch 2.9.1+cu128, BF16, compiled model, Muon/AdamW optimizer, 12 total
steps, three excluded warmup steps, and a fixed synthetic token batch.

| Mode | Chunk | Median tok/s | Median step | Peak allocated |
| --- | ---: | ---: | ---: | ---: |
| StateHead CUDA v2 | 32 | 824,653 | 79.47 ms | 13.15 GiB |
| StateHead CUDA v2 | 64 | 772,758 | 84.81 ms | 13.12 GiB |
| StateHead compiled PyTorch | 32 | 615,599 | 106.46 ms | 36.42 GiB |
| GPT verified FA3 | n/a | 530,591 | 123.52 ms | 27.40 GiB |

Controlled comparisons:

- Chunk 32 versus chunk 64: 6.72% faster.
- CUDA v2 chunk 32 versus compiled PyTorch: 33.96% faster.
- CUDA v2 chunk 32 versus GPT/FA3: 55.42% faster.
- CUDA v2 chunk 32 peak allocation versus compiled PyTorch: 63.90% lower.
- CUDA v2 chunk 32 peak allocation versus GPT/FA3: 52.01% lower.

The earlier CUDA v1 report measured 548,939 tok/s on a different H100 host.
CUDA v2's 824,653 tok/s is consistent with the hierarchical-backward hypothesis,
but that cross-host delta is not used as the formal speed claim. The formal
claims above use only same-host controls.

## Interpretation

The original backward assigned one thread to a state element and made it walk
all 2048 timesteps serially. CUDA v2 computes reverse affine summaries for all
chunks in parallel, scans only the chunk boundaries, and then calculates each
chunk's gradients independently. The measured 34% full-step advantage over the
compiled PyTorch reference shows that this change addresses a material
end-to-end bottleneck rather than merely improving an isolated scan.

Chunk 32 is the current BF16 CUDA candidate. It is faster than chunk 64 while
using essentially the same peak memory. Chunk 16 remains unmeasured.

The fixed synthetic batch produces an aggressive short optimization trajectory,
and small numerical-order differences are amplified across steps. These timing
runs establish finite execution and throughput, not learning equivalence.

## Cost and lifecycle

- Initially approved ceiling: $1.75.
- First approved offer `41819339`: unavailable at creation; no resource or
  charge was created.
- Created replacement offer `43189292` as instance `45636045`.
- Actual all-in observed rate: $3.02/hour.
- Runtime before deletion: approximately 23.17 minutes.
- Estimated charge at deletion: approximately $1.17.
- Instance deletion: confirmed.
- Post-gate Vast instances: zero.

## Artifacts

- `statehead-cuda-v2-k32.json`
- `statehead-cuda-v2-k64.json`
- `statehead-pytorch-k32-control.json`
- `gpt-fa3-control.json`
- `fp32.log` and `fp32.exit`
- `bf16.log` and `bf16.exit`
- `environment.txt`
- `native-cuda-load.log`
- `artifacts.sha256`

The retrieved files match the SHA-256 values generated on the instance.

## Commands run for this gate

Control plane:

```bash
vastai create instance 41819339 --image 'vastai/pytorch:@vastai-automatic-tag' --disk 30 --ssh --direct --label statehead-cuda-v2-gate --cancel-unavail --raw
vastai search offers 'gpu_name=H100_SXM num_gpus=1 verified=true rentable=true direct_port_count>=1 cuda_max_good>=12.8 dph_total<=2.32' -o 'dph_total' --raw --limit 20
vastai search offers 'gpu_name=H100_SXM num_gpus=1 verified=true rentable=true direct_port_count>=1 cuda_max_good>=12.8 dph_total<=3.20' -o 'dph_total' --raw --limit 10
vastai create instance 43189292 --image 'vastai/pytorch:@vastai-automatic-tag' --disk 30 --ssh --direct --label statehead-cuda-v2-gate --cancel-unavail --raw
vastai show instance 45636045 --raw
vastai ssh-url 45636045 --raw
vastai destroy instance 45636045 -y --raw
vastai show instances-v1 --raw
```

Instance inspection and setup:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python3 --version
/venv/main/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_capability())"
git clone --branch codex/statehead-nanochat --single-branch https://github.com/samfurr/nanochat.git /root/nanochat
cd /root/nanochat
git rev-parse HEAD
UV_CACHE_DIR=/root/.cache/uv uv sync --extra gpu --group dev --frozen
UV_CACHE_DIR=/root/.cache/uv uv pip install --python .venv/bin/python setuptools
apt-cache policy cuda-compiler-12-8 cuda-toolkit-12-8
apt-cache depends cuda-compiler-12-8
DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-compiler-12-8
apt-cache depends cuda-libraries-dev-12-8
DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-libraries-dev-12-8
sed -n '1,100p' .venv/lib/python3.10/site-packages/torch/include/ATen/cuda/CUDAContextLight.h
kill $(pgrep -x apt-get)
DEBIAN_FRONTEND=noninteractive apt-get install -y libcublas-dev-12-8 libcusparse-dev-12-8 libcusolver-dev-12-8
```

Compilation and correctness:

```bash
export PATH=/root/nanochat/.venv/bin:/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_EXTENSIONS_DIR=/root/torch_extensions_v2
NANOCHAT_DTYPE=float32 NANOCHAT_STATEHEAD_CUDA_VERBOSE=1 python -m nanochat.statehead_cuda
NANOCHAT_DTYPE=float32 python -m pytest tests/test_statehead.py -q
NANOCHAT_DTYPE=bfloat16 python -m pytest tests/test_statehead.py -q
NANOCHAT_DTYPE=bfloat16 python -m pytest tests/test_statehead.py -q -k "native_cuda or reverse_chunk"
```

The first compile attempt failed before compiling the source because
`setuptools` was absent. The second reached `nvcc` but lacked `cusparse.h`.
After installing the targeted CUDA 12.8 development headers, compilation and
load passed.

Performance:

```bash
NANOCHAT_DTYPE=bfloat16 NANOCHAT_REPO_COMMIT=52dd72e184bf1459009e9d99a4819a2634f872aa python -m dev.statehead_cuda_preflight --arch statehead --scan-backend cuda --device-batch-size 32 --steps 12 --warmup-steps 3 --scan-chunk-size 64 --output /root/statehead-cuda-v2-k64.json
NANOCHAT_DTYPE=bfloat16 NANOCHAT_REPO_COMMIT=52dd72e184bf1459009e9d99a4819a2634f872aa python -m dev.statehead_cuda_preflight --arch statehead --scan-backend cuda --device-batch-size 32 --steps 12 --warmup-steps 3 --scan-chunk-size 32 --output /root/statehead-cuda-v2-k32.json
NANOCHAT_DTYPE=bfloat16 NANOCHAT_REPO_COMMIT=52dd72e184bf1459009e9d99a4819a2634f872aa python -m dev.statehead_cuda_preflight --arch statehead --scan-backend pytorch --device-batch-size 32 --steps 12 --warmup-steps 3 --scan-chunk-size 32 --output /root/statehead-pytorch-k32-control.json
NANOCHAT_DTYPE=bfloat16 python -m nanochat.flash_attention
NANOCHAT_DTYPE=bfloat16 NANOCHAT_REPO_COMMIT=52dd72e184bf1459009e9d99a4819a2634f872aa python -m dev.statehead_cuda_preflight --arch gpt --scan-backend pytorch --device-batch-size 32 --steps 12 --warmup-steps 3 --output /root/gpt-fa3-control.json
```

Artifact capture:

```bash
sha256sum /root/statehead-cuda-v2-k64.json /root/statehead-cuda-v2-k32.json /root/statehead-pytorch-k32-control.json /root/gpt-fa3-control.json /root/fp32.log /root/bf16.log /root/environment.txt /root/native-cuda-load.log
scp -P 50113 root@219.86.90.203:/root/<artifact> dev/results/statehead-cuda-v2-gate-20260723/
shasum -a 256 dev/results/statehead-cuda-v2-gate-20260723/*
```

## Unresolved questions

1. Does chunk 16 outperform chunk 32 at the production shape?
2. Which kernels and HBM transactions dominate CUDA v2 after the hierarchical
   backward change? Nsight Systems/Compute was not collected.
3. Does a short dataset-backed run preserve the compiled-PyTorch BPB/CORE
   trajectory closely enough to authorize a full CUDA training run?
4. Can interleaved `[B,T,H,Dh,4]` gates and vector loads improve speed without
   introducing a runtime transpose or checkpoint ambiguity?
5. Can gate bias and output residual be folded into GEMM epilogues by the
   current compiler/library stack?
6. Does a native FP8 GEMM path improve throughput while keeping recurrence FP32
   and preserving learning stability?

## Exact next recommended command

On the next approved H100, measure chunk 16 on the same controlled probe before
profiling the winning chunk size:

```bash
NANOCHAT_DTYPE=bfloat16 NANOCHAT_REPO_COMMIT=52dd72e184bf1459009e9d99a4819a2634f872aa python -m dev.statehead_cuda_preflight --arch statehead --scan-backend cuda --device-batch-size 32 --steps 20 --warmup-steps 5 --scan-chunk-size 16 --output /workspace/statehead-cuda-v2-k16.json
```
