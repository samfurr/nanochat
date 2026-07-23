# StateHead fused CUDA verification gate

Date: 2026-07-23 UTC

Tested commit: `ada94afd6c526158331bc3a25785e9c48894d08b`

Hardware/runtime: one secure NVIDIA H100 80GB HBM3, PyTorch 2.9.1+cu128,
CUDA toolkit 12.8, driver 570.211.01, `kernels==0.11.7`, and verified
FlashAttention-3 for GPT. The pod was deleted after artifact retrieval. The
actual Runpod account charge was $0.6708058195.

## Verdict

- Native CUDA compilation: pass.
- FP32 forward and gradient parity: pass.
- BF16 forward and gradient parity with FP32 recurrence accumulation: pass.
- Compiled full-model StateHead CUDA versus compiled PyTorch loss/gradient
  parity: pass within the test's documented dtype-specific tolerances.
- StateHead FP8 50-step fixed-batch finiteness check: pass; every parameter
  gradient was checked on every step.
- Speed versus compiled PyTorch StateHead: fail. Native BF16 was 3.23% slower;
  native FP8 was 6.89% slower.
- Speed versus verified FA3 GPT at the same d12/768/2048/batch-32 shape: pass
  for this probe. Native StateHead BF16 was 7.42% faster than GPT BF16; native
  StateHead FP8 was 2.84% faster than GPT FP8.
- Dataset learning parity: not tested and not claimed.

## Correctness results

| Gate | Result | Artifact |
| --- | ---: | --- |
| Native C++/SM90 CUDA compile, link, load | Pass | `native-cuda-build-v2.log` |
| FP32 StateHead suite | 61 passed | `pytest-fp32-v2.log` |
| BF16 StateHead suite | 61 passed | `pytest-bf16-v2.log` |
| StateHead FP8 fixed-batch stability | 50 finite steps and finite gradients | `statehead-cuda-fp8-stability.json` |
| GPT FP8/FA3 stability control | 50 finite steps and finite gradients | `gpt-fa3-fp8-stability.json` |

The earlier `native-cuda-build.log`, `pytest-fp32.log`, and `pytest-bf16.log`
are retained as failure provenance. They record the rejected `--lineinfo`
compiler flag, the initial missing venv `PATH`, and the BF16 test-fixture dtype
asymmetry that were fixed before the passing reruns.

## Full-step speed

Median full optimizer-step throughput after excluding five compile/warmup steps:

| Mode | Median tok/s | Peak allocated GiB |
| --- | ---: | ---: |
| StateHead compiled PyTorch BF16 | 567,259 | 40.44 |
| StateHead native CUDA BF16 | 548,939 | 13.11 |
| StateHead native CUDA FP8, 25 linears | 528,153 | 23.17 |
| GPT FlashAttention-3 BF16 | 511,024 | 27.40 |
| GPT FlashAttention-3 FP8, 73 linears | 513,550 | 33.14 |

The PyTorch StateHead reference is still the fastest implementation. The native
kernel's clear current benefit is memory: BF16 peak allocation is about 68%
lower than the PyTorch parallel scan, but this does not establish a speed win.

## FP8 scope

Both models used the winner-matching tensorwise dynamic recipe: FP32 master
weights, BF16 activations, E4M3 forward operands, E5M2 gradient output, and the
same large/aligned-linear eligibility filter. StateHead recurrence accumulation
remained FP32. The 50-step fixed synthetic batch verifies short-run numerical
stability only; it does not demonstrate dataset learning parity.

## Commands run for this gate

Control-plane and provisioning:

```bash
runpodctl version
runpodctl user
runpodctl pod list --all
date -u -v+29M '+%Y-%m-%dT%H:%M:%SZ'
runpodctl pod create --name statehead-cuda-verification-gate --image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 --gpu-id "NVIDIA H100 80GB HBM3" --gpu-count 1 --cloud-type SECURE --container-disk-in-gb 30 --ports "22/tcp" --ssh --terminate-after "2026-07-23T03:27:33Z" --output json
runpodctl pod get p32xzueiy58c9g
runpodctl ssh info p32xzueiy58c9g
runpodctl pod delete p32xzueiy58c9g
runpodctl pod list --all
runpodctl user
```

Pod setup and environment checks:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
/usr/local/cuda-12.8/bin/nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_capability())"
git clone --branch codex/statehead-nanochat --single-branch https://github.com/samfurr/nanochat.git /workspace/nanochat
python -m venv --system-site-packages /workspace/venv
/workspace/venv/bin/python -m pip install numpy pytest ninja tiktoken rustbpe kernels psutil
/workspace/venv/bin/python -m pip install --force-reinstall --no-deps kernels==0.11.7
NANOCHAT_DTYPE=bfloat16 /workspace/venv/bin/python -c "from nanochat.flash_attention import HAS_FA3, USE_FA3; print(HAS_FA3, USE_FA3)"
```

Compilation and correctness:

```bash
NANOCHAT_DTYPE=float32 NANOCHAT_STATEHEAD_CUDA_VERBOSE=1 python -m nanochat.statehead_cuda
NANOCHAT_DTYPE=float32 python -m pytest tests/test_statehead.py -q
NANOCHAT_DTYPE=bfloat16 python -m pytest tests/test_statehead.py -q
```

The compiler and test commands were rerun after each recorded build/test-fixture
fix. The passing runs used `PATH=/workspace/venv/bin:/usr/local/cuda-12.8/bin`,
`CUDA_HOME=/usr/local/cuda-12.8`, and
`TORCH_EXTENSIONS_DIR=/workspace/torch_extensions_v3`.

Speed and stability:

```bash
ARCH=statehead SCAN_BACKEND=pytorch FP8=0 DEVICE_BATCH_SIZE=32 PREFLIGHT_STEPS=20 WARMUP_STEPS=5 VERIFY_GRADIENTS=0 PREFLIGHT_TAG=statehead-pytorch-bf16-speed bash runs/statehead_cuda_preflight.sh
ARCH=statehead SCAN_BACKEND=cuda FP8=0 DEVICE_BATCH_SIZE=32 PREFLIGHT_STEPS=20 WARMUP_STEPS=5 VERIFY_GRADIENTS=0 PREFLIGHT_TAG=statehead-cuda-bf16-speed bash runs/statehead_cuda_preflight.sh
ARCH=statehead SCAN_BACKEND=cuda FP8=1 DEVICE_BATCH_SIZE=32 PREFLIGHT_STEPS=20 WARMUP_STEPS=5 VERIFY_GRADIENTS=0 PREFLIGHT_TAG=statehead-cuda-fp8-speed bash runs/statehead_cuda_preflight.sh
ARCH=gpt SCAN_BACKEND=pytorch FP8=0 DEVICE_BATCH_SIZE=32 PREFLIGHT_STEPS=20 WARMUP_STEPS=5 VERIFY_GRADIENTS=0 PREFLIGHT_TAG=gpt-fa3-bf16-speed bash runs/statehead_cuda_preflight.sh
ARCH=gpt SCAN_BACKEND=pytorch FP8=1 DEVICE_BATCH_SIZE=32 PREFLIGHT_STEPS=20 WARMUP_STEPS=5 VERIFY_GRADIENTS=0 PREFLIGHT_TAG=gpt-fa3-fp8-speed bash runs/statehead_cuda_preflight.sh
ARCH=statehead SCAN_BACKEND=cuda FP8=1 DEVICE_BATCH_SIZE=32 PREFLIGHT_STEPS=50 WARMUP_STEPS=5 VERIFY_GRADIENTS=1 PREFLIGHT_TAG=statehead-cuda-fp8-stability bash runs/statehead_cuda_preflight.sh
ARCH=gpt SCAN_BACKEND=pytorch FP8=1 DEVICE_BATCH_SIZE=32 PREFLIGHT_STEPS=50 WARMUP_STEPS=5 VERIFY_GRADIENTS=1 PREFLIGHT_TAG=gpt-fa3-fp8-stability bash runs/statehead_cuda_preflight.sh
```

Artifact retrieval and local checks:

```bash
scp -r -i /Users/haybales/.runpod/ssh/runpodctl-ssh-key -P 22100 root@69.30.85.161:/workspace/statehead-preflight-results/. dev/results/statehead-fused-cuda-preflight-20260723/
jq -s ... dev/results/statehead-fused-cuda-preflight-20260723/*.json
rg -n "passed|failed" dev/results/statehead-fused-cuda-preflight-20260723/pytest-*.log
shasum -a 256 dev/results/statehead-fused-cuda-preflight-20260723/*
```

## Unresolved questions

1. Why does the native BF16 scan save substantial memory but trail the compiled
   PyTorch scan by 3.23%? A synchronized CUDA timeline is needed to separate
   scan launch overhead, reverse-scan cost, GEMM time, and optimizer time.
2. Why does the current custom tensorwise FP8 wrapper slow both architectures
   at d12 despite using H100 FP8 GEMMs? Likely candidates are quantization
   reductions, layout conversions, and opaque autograd boundaries.
3. Does StateHead FP8 preserve dataset-backed validation BPB and CORE learning
   curves? This gate used one fixed synthetic batch and cannot answer that.
4. Does the 2.84% StateHead-over-GPT FP8 result persist across repeated timing
   samples, eight-GPU DDP, and the official data loader?

## Exact next recommended command

On the next approved H100, collect a CUDA Systems timeline for the already
verified BF16 native candidate before changing the kernel:

```bash
nsys profile --trace=cuda,nvtx,cublas,osrt --sample=none --force-overwrite=true --output=/workspace/statehead-cuda-bf16 env ARCH=statehead SCAN_BACKEND=cuda FP8=0 DEVICE_BATCH_SIZE=32 PREFLIGHT_STEPS=20 WARMUP_STEPS=5 VERIFY_GRADIENTS=0 PREFLIGHT_TAG=statehead-cuda-bf16-profile bash runs/statehead_cuda_preflight.sh
```
