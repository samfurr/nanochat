# Vast.ai eight-H100 StateHead preflight

Date: 2026-07-23 UTC

Tested checkout: `df6e4161b2def519d7bf9061c15250f31b307abc`

This was a paid infrastructure/compiler/DDP probe on Vast.ai offer `40228016`,
instance `45629291`. The host was an unverified Japan marketplace host with
advertised reliability `0.9702319`. It supplied one node with eight
`NVIDIA H100 80GB HBM3` GPUs, 81,559 MiB per GPU, driver `595.71.05`, and
all-to-all `NV18` GPU topology. The offer price was `$16.00/hour` for GPUs plus
approximately `$0.0278/hour` for 100 GB local disk.

## Outcome

- Exact eight-GPU H100 SXM shape: pass.
- All-to-all NVLink (`NV18` for every GPU pair): pass.
- Frozen environment: pass, PyTorch `2.9.1+cu128`, CUDA runtime `12.8`.
- Compiled d12 StateHead full forward/backward/optimizer step on eight ranks:
  pass.
- Two optimizer steps: pass.
- Every parameter gradient finite on every rank: pass.
- Cross-rank parameter checksum spread: exactly `0.0`.
- First step including compilation: `53.1073154178448` seconds.
- Steady step: `0.12498901505023241` seconds.
- Steady aggregate throughput: `4,194,672.626144717` tokens/second.
- Peak allocated/reserved VRAM per rank:
  `43,207,041,024` / `47,355,789,312` bytes.

This is not a learning-quality result and does not establish GPT parity,
dataset-learning parity, validation BPB, CORE, or sustained-run stability.

## Incomplete optional check

The planned non-native StateHead parity suite did not run on this host. The
initial test command failed because the production-only frozen environment did
not include `pytest`. Installing the locked development group began, but the
five-minute destruction guard fired before installation finished and closed
the SSH session. Existing local and prior H100 parity evidence remains valid
for its recorded environments, but host-specific sequential/parallel and
eager/compiled parity is not claimed here.

The remote JSON and log were destroyed with the instance before `scp`
retrieval. `captured-result.json` was reconstructed from the complete JSON
printed by the successful probe to the captured command output; it is not
claimed to be a byte-for-byte retrieved remote artifact.

Captured result SHA-256:
`cde52bb02668615eb9a22c8ad177298690731f0ee459935342071a537c11cd4a`.

## Cost and cleanup

Vast.ai posted a `$1.749` instance charge:

- GPU: `$1.734` for `0.1084047` hours at `$16.00/hour`.
- Disk: `$0.004` for `0.1609991` hours at `$0.028/hour`.
- Download: `$0.011` for 4.31 GB.
- Upload: `$0.000` for 0.03496094 GB.

The observed credit delta was `$1.7503081587`, from `$95.34167914014` to
`$93.59137098144`. This exceeded the `$1.34` estimate because the posted GPU
duration was 6.504 minutes, longer than the five-minute guard interval after
readiness was detected.

The guard destroyed instance `45629291`. Post-cleanup queries returned zero
instances and zero volumes.

## Workload command

```bash
NANOCHAT_REPO_COMMIT=df6e4161b2def519d7bf9061c15250f31b307abc \
NPROC_PER_NODE=8 \
ARCH=statehead \
DEVICE_BATCH_SIZE=32 \
PREFLIGHT_STEPS=2 \
WARMUP_STEPS=1 \
SCAN_BACKEND=pytorch \
FP8=0 \
VERIFY_GRADIENTS=1 \
RESULTS_DIR=/workspace/statehead-vast-preflight-results \
PREFLIGHT_TAG=statehead-d12-b32-w8-vast \
bash runs/statehead_cuda_preflight.sh
```

## Command ledger

Commands containing authentication material are intentionally represented
without credentials. Read-only source-inspection commands are included where
they affected the gate.

```bash
sed -n '1,520p' /Users/haybales/.codex/skills/vastai/SKILL.md
git rev-parse HEAD && git status --short && git branch --show-current && git remote -v
rg -n -i 'nanochat|phase 3|vast|H100|statehead-speed' /Users/haybales/.codex/memories/MEMORY.md
sed -n '1,1240p' STATEHEAD_NANOCHAT_CODEX_BRIEF.md
rg -n -i 'preflight|torch.compile|compile|ddp|cuda|H100|statehead_d12' runs scripts tests pyproject.toml STATEHEAD_NANOCHAT_CODEX_BRIEF.md
sed -n '1,240p' runs/statehead_cuda_preflight.sh
sed -n '1,320p' dev/statehead_cuda_preflight.py
find .. -name AGENTS.md -print
git log -5 --oneline --decorate
git ls-files runs/statehead_cuda_preflight.sh dev/statehead_cuda_preflight.py tests/test_statehead.py nanochat/statehead_cuda.py
rg -n -C 4 'preflight|PREFLIGHT|batch|H100|8x|NPROC|SCAN_BACKEND|VERIFY_GRADIENTS' dev/STATEHEAD_LOG.md dev runs README.md
git show --stat --oneline 28f425e
git show --format=fuller --no-ext-diff 28f425e
git show --stat --oneline df6e416
git show --format=fuller --no-ext-diff df6e416
rg -n '^def test_|^@pytest' tests/test_statehead.py
sed -n '1,180p' dev/experiments/statehead-nanochat-cuda-8gpu-preflight-v1.yaml
jq '.' dev/results/statehead-nanochat-cuda-8gpu-preflight-v1.json
sed -n '1,260p' runs/statehead_d12_controlled.sh
sed -n '1,180p' dev/experiments/statehead-nanochat-d12-controlled-v1.yaml
sed -n '1,100p' dev/experiments/gpt-d12-controlled-v1.yaml
git diff --quiet b952753243ea14ac391d617244f2fbd52ba0a487 -- nanochat/gpt.py nanochat/statehead.py nanochat/optim.py nanochat/dataloader.py nanochat/checkpoint_manager.py scripts/base_train.py scripts/base_eval.py
vastai search offers --no-default 'id=40228016 gpu_name=H100_SXM num_gpus=8 rentable=true cuda_vers>=12.8 bw_nvlink>0 dph<24' --type on-demand --storage 100 --raw
vastai show instances --raw
vastai show user --raw
vastai search offers --no-default 'gpu_name=H100_SXM num_gpus=8 rentable=true external=false cuda_vers>=12.8 bw_nvlink>0 dph<24' --type on-demand --storage 100 --order dph_total --limit 50 --raw
vastai create instance 40228016 --image 'vastai/pytorch:@vastai-automatic-tag' --disk 100 --ssh --label statehead-vast-8xh100-preflight --cancel-unavail --raw
nohup sh -c 'sleep 300; vastai destroy instance 45629291 -y --raw' >/private/tmp/vastai-45629291-watchdog.log 2>&1 &
vastai show instance 45629291 --raw  # run nine times while polling
sleep 15  # run four times
sleep 20
sleep 10  # run twice
kill 36028  # failed: the nohup child had not persisted
sleep 300 && vastai destroy instance 45629291 -y --raw
vastai ssh-url 45629291 --raw
ssh -p 29290 root@ssh1.vast.ai 'nvidia-smi -L; nvidia-smi topo -m; python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.device_count())"'
ssh -p 29290 root@ssh1.vast.ai 'command -v python3 || true; command -v uv || true; ls -d /opt/conda /venv /workspace 2>/dev/null || true; env | sort | grep -E "^(PATH|CONDA|VIRTUAL_ENV)="'
ssh -p 29290 root@ssh1.vast.ai 'git clone --branch codex/statehead-nanochat --depth 1 https://github.com/samfurr/nanochat.git /workspace/nanochat && cd /workspace/nanochat && git rev-parse HEAD && uv sync --extra gpu --frozen'
ssh -p 29290 root@ssh1.vast.ai 'cd /workspace/nanochat && source .venv/bin/activate && NPROC_PER_NODE=8 ARCH=statehead DEVICE_BATCH_SIZE=32 PREFLIGHT_STEPS=2 WARMUP_STEPS=1 SCAN_BACKEND=pytorch FP8=0 VERIFY_GRADIENTS=1 RESULTS_DIR=/workspace/statehead-vast-preflight-results PREFLIGHT_TAG=statehead-d12-b32-w8-vast NANOCHAT_REPO_COMMIT=$(git rev-parse HEAD) bash runs/statehead_cuda_preflight.sh'
ssh -p 29290 root@ssh1.vast.ai 'cd /workspace/nanochat && source .venv/bin/activate && NANOCHAT_DTYPE=float32 python -m pytest tests/test_statehead.py -q -k "not native_cuda" 2>&1 | tee /workspace/statehead-vast-preflight-results/reference-parity-pytest.log'
ssh -p 29290 root@ssh1.vast.ai 'cd /workspace/nanochat && uv sync --extra gpu --group dev --frozen && source .venv/bin/activate && NANOCHAT_DTYPE=float32 python -m pytest tests/test_statehead.py -q -k "not native_cuda" 2>&1 | tee /workspace/statehead-vast-preflight-results/reference-parity-pytest.log'
vastai show instances --raw
vastai show volumes --raw
vastai show user --raw
vastai show invoices-v1 --help
date -j -f '%Y-%m-%d %H:%M:%S' '2026-07-23 00:00:00' '+%s'
vastai show invoices-v1 --charges --charge-type instance --start-date 2026-07-23 --latest-first --limit 20 --raw
vastai show invoices-v1 --charges --charge-type instance --start-date 1784779200 --end-date 1784865600 --latest-first --limit 20 --raw
```

The date-string invoice command failed in Vast CLI `1.4.3` with a `TypeError`;
the numeric-timestamp retry succeeded.
