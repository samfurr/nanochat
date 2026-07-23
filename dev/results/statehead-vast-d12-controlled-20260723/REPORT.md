# StateHead and GPT d12 controlled run

Date: 2026-07-23

## Outcome

The predetermined-step, fixed-token Phase 3 comparison completed on one Vast.ai
node with eight NVIDIA H100 80GB HBM3 GPUs. StateHead and GPT used the same
tokenizer, exact 170 training shards plus validation shard, seed, data order,
2,048-token rows, global batch, 2,520 optimizer steps, 1,321,205,760 tokens,
BF16 precision, and evaluation schedule.

The same-shape StateHead run did **not** demonstrate learning parity with GPT.
It finished with held-out validation BPB `0.980587` and CORE `0.067406`,
compared with GPT at `0.845578` and `0.146506`. StateHead's BPB was `0.135009`
or 15.97% higher, and its CORE was `0.079100` or 53.99% lower relative to GPT.
This is one predetermined seed, so it is not a variance estimate, but the gap
is not a parity result.

| Model | L×D | Params | Scaling params | Tokens | Precision | Trainer time | Training-command wall | Median tok/s | Peak VRAM | Held-out val BPB | CORE |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| nanochat GPT | 12×768 | 286,261,730 | 110,100,912 | 1,321,205,760 | BF16 | 324.950 s | 489 s | 4,053,035 | 28,014.59 MiB | 0.845578 | 0.146506 |
| MLP-free StateHead, same shape | 12×768 | 85,767,218 | 60,555,264 | 1,321,205,760 | BF16 | 304.807 s | 463 s | 4,321,656 | 41,432.27 MiB | 0.980587 | 0.067406 |

`Trainer time` is nanochat's accumulated training-step time. The
`training-command wall` measurement also includes scheduled 40M-token
validation passes and final checkpoint writes. Median step throughput is
computed from all 2,520 logged training steps. StateHead was 6.63% faster by
that measure and 6.20% faster by trainer time, despite an analytical training
FLOP estimate of `4.801456e17`, 47.84% of GPT's `1.003715e18`. Its peak VRAM
was 47.90% higher. This is measured evidence that the PyTorch associative scan
does not turn StateHead's lower analytical FLOPs into proportional wall-clock
savings.

GPT used the checked-in Flash Attention 3 path. StateHead used the explicit
PyTorch parallel scan backend under the same `torch.compile(dynamic=False)`
training path. No graph-break, compiler-fallback, NaN, Inf, or runtime-error
message appeared in either retrieved training log. Neither run used FP8 or
another reduced-precision quantization mode beyond BF16.

## Validation curve

| Step | Tokens seen | StateHead BPB | GPT BPB |
|---:|---:|---:|---:|
| 0 | 0 | 3.168812 | 3.168812 |
| 250 | 131,072,000 | 1.297860 | 1.096327 |
| 500 | 262,144,000 | 1.179565 | 1.004868 |
| 750 | 393,216,000 | 1.132789 | 0.969751 |
| 1,000 | 524,288,000 | 1.098211 | 0.946200 |
| 1,250 | 655,360,000 | 1.069454 | 0.922318 |
| 1,500 | 786,432,000 | 1.045303 | 0.902652 |
| 1,750 | 917,504,000 | 1.024929 | 0.884771 |
| 2,000 | 1,048,576,000 | 1.005989 | 0.868967 |
| 2,250 | 1,179,648,000 | 0.991906 | 0.855628 |
| 2,500 | 1,310,720,000 | 0.981668 | 0.846014 |
| 2,520 | 1,321,205,760 | 0.981185 | 0.845554 |

The final separately loaded checkpoint evaluation reported StateHead
train/validation BPB `0.981390 / 0.980587` and GPT `0.846397 / 0.845578`.

## Per-task CORE

| Task | StateHead centered | GPT centered | StateHead - GPT |
|---|---:|---:|---:|
| hellaswag_zeroshot | 0.074155 | 0.151431 | -0.077276 |
| jeopardy | 0.000000 | 0.007085 | -0.007085 |
| bigbench_qa_wikidata | 0.085035 | 0.254712 | -0.169677 |
| arc_easy | 0.313692 | 0.425365 | -0.111673 |
| arc_challenge | 0.014790 | 0.031854 | -0.017064 |
| copa | 0.100000 | 0.180000 | -0.080000 |
| commonsense_qa | 0.003890 | 0.067363 | -0.063473 |
| piqa | 0.284004 | 0.351469 | -0.067465 |
| openbook_qa | 0.040000 | 0.088000 | -0.048000 |
| lambada_openai | 0.093732 | 0.308752 | -0.215020 |
| hellaswag | 0.063931 | 0.147846 | -0.083915 |
| winograd | 0.076923 | 0.157509 | -0.080586 |
| winogrande | 0.024467 | -0.005525 | +0.029992 |
| bigbench_dyck_languages | 0.000000 | 0.088000 | -0.088000 |
| agi_eval_lsat_ar | 0.086956 | 0.076087 | +0.010869 |
| bigbench_cs_algorithms | 0.019697 | 0.428030 | -0.408333 |
| bigbench_operators | 0.080952 | 0.080952 | +0.000000 |
| bigbench_repeat_copy_logic | 0.000000 | 0.031250 | -0.031250 |
| squad | 0.008704 | 0.195364 | -0.186660 |
| coqa | 0.045096 | 0.205938 | -0.160842 |
| boolq | -0.107356 | -0.231289 | +0.123933 |
| bigbench_language_identification | 0.174257 | 0.182948 | -0.008691 |
| **CORE** | **0.067406** | **0.146506** | **-0.079100** |

StateHead was higher on three tasks, tied one, and lower on eighteen. The
aggregate and task table do not support a broad-capability parity claim.

## Fairness and provenance

- Local branch head before the paid full-run preparation:
  `8007640ec97cb5afb8cef2d21cc16d24be14ec6f`.
- Model/training source commit:
  `b952753243ea14ac391d617244f2fbd52ba0a487`.
- Remote checkout head:
  `07d3766f98ea1563b2034477f48e8de5fb12a746`.
  Later commits `53ad3e9` and `65a0196` changed only the experiment manifest.
- Runner SHA-256:
  `21ef5b3022c49c279561e433aa3ee19797cfe9ed5622fd5c1f79618e9e1b5564`.
- StateHead final checkpoint:
  `62e688b8b9b700deb0a14a6028beb8684c6bf2b4f1699d2a9e6e59225bb7da44`.
- GPT final checkpoint:
  `5db2a510188479cd039beb6912b7a62360c72146d5b7ca90f1ab51bdf7c44258`.
- Both final model, metadata, and CORE files matched their remote SHA-256
  manifests after download.
- All eight optimizer shards for each architecture passed their remote
  SHA-256 checks before instance deletion. The optimizer payloads were not
  downloaded; their hashes remain in the two tracked `.sha256` manifests.
- Both run-exit files report `exit_status=0`, and separate completion markers
  were retrieved.
- The exact shard check passed with 171 files: train shards `00000` through
  `00169` plus validation shard `06542`.

The experiment is the brief's same-depth/same-width comparison, not a
parameter-matched comparison. StateHead had only 55.00% of GPT's scaling
parameters and 29.96% of its total parameters. GPT's total includes
150,994,944 checked-in value-embedding parameters. No GPT behavior was changed.

## Infrastructure and cost

- Provider/instance: Vast.ai `45630532`, offer `40228016`.
- Hardware: one node, 8× NVIDIA H100 80GB HBM3, NVLink bandwidth field
  `478.1160888671875 GB/s`.
- Host: unverified Japan listing, reliability `0.9702994` in the final
  snapshot.
- Image request: `vastai/pytorch:@vastai-automatic-tag`.
- Runtime software: Python 3.10.20, PyTorch 2.9.1+cu128, CUDA runtime 12.8,
  driver 595.71.05.
- Price: `$16.00/hour` GPU plus `$0.0277778/hour` disk.
- Billed runtime: `0.6364842` hours.
- Full-run invoice: `$10.261`, comprising `$10.184` GPU, `$0.018` disk,
  `$0.054` download, and `$0.005` upload.
- Observed balance delta: `$10.260359`.
- Separate earlier Vast preflight invoice: `$1.749`.
- Approved full-run ceiling: `$50.00`.
- Post-run audit: zero active instances and zero volumes.

The initial remote checkout was shallow and did not contain the pinned model
commit, so the runner failed before environment setup, data work, or training.
The repository was unshallowed, the model-code diff was reverified as empty,
and the controlled runner was relaunched. This caused no partial checkpoint or
training-data ambiguity.

## Artifacts

Tracked lightweight evidence is in this directory:

- `COMPLETED`
- `dataset-all.log`
- `dataset-first-eight.log`
- `dataset-verification.txt`
- `environment.txt`
- `gpt-d12-controlled-v1-core.csv`
- `gpt-d12-controlled-v1-eval.log`
- `gpt-d12-controlled-v1-meta_002520.json`
- `gpt-d12-controlled-v1-train-wall-seconds.txt`
- `gpt-d12-controlled-v1-train.log`
- `gpt-d12-controlled-v1.sha256`
- `gpt-dataset-verification.txt`
- `gpt-environment.txt`
- `gpt-full-driver.log`
- `gpt-run-exit.txt`
- `nvidia-smi.csv`
- `run-exit.txt`
- `statehead-COMPLETED`
- `statehead-dataset-verification.txt`
- `statehead-environment.txt`
- `statehead-full-driver.log`
- `statehead-nanochat-d12-controlled-v1-core.csv`
- `statehead-nanochat-d12-controlled-v1-eval.log`
- `statehead-nanochat-d12-controlled-v1-meta_002520.json`
- `statehead-nanochat-d12-controlled-v1-train-wall-seconds.txt`
- `statehead-nanochat-d12-controlled-v1-train.log`
- `statehead-nanochat-d12-controlled-v1.sha256`
- `statehead-run-exit.txt`
- `tokenizer-train.log`
- `tokenizer.sha256`
- `uv-sync.log`

The complete 1.0 GiB downloaded bundle, including the two final models, is
kept locally outside Git under:

```text
dev-ignore/statehead-vast-d12-controlled-20260723/controlled-results/
```

The ignored model payloads are:

```text
statehead-nanochat-d12-controlled-v1-model_002520.pt  292,758,639 bytes
gpt-d12-controlled-v1-model_002520.pt                 792,761,399 bytes
```

They are intentionally not committed because GitHub rejects ordinary Git
objects larger than 100 MiB.

## Command ledger

Prelaunch inspection, approval recording, and launch:

```bash
sed -n '1,260p' /Users/haybales/.codex/skills/vastai/SKILL.md
git rev-parse HEAD
git status --short
shasum -a 256 runs/statehead_d12_controlled.sh
DRY_RUN=1 RUN_ARCHES=statehead NPROC_PER_NODE=8 DOWNLOAD_WORKERS=16 NANOCHAT_BASE_DIR=/workspace/nanochat-controlled WANDB_RUN_PREFIX=dummy SETUP_ENV=1 PREPARE_DATA=1 bash runs/statehead_d12_controlled.sh
vastai show user --raw
vastai show instances --raw
vastai show volumes --raw
vastai search offers --no-default 'id=40228016 gpu_name=H100_SXM num_gpus=8 rentable=true cuda_vers>=12.8 bw_nvlink>0 dph<24' --type on-demand --storage 100 --raw
git add dev/experiments/statehead-nanochat-d12-controlled-v1.yaml
git diff --cached --check
git commit -m 'Approve Vast StateHead-only controlled run'
git push origin codex/statehead-nanochat
vastai create instance 40228016 --image 'vastai/pytorch:@vastai-automatic-tag' --disk 100 --ssh --label statehead-vast-d12-full-v1 --cancel-unavail --raw
sleep 10417 && vastai destroy instance 45630532 -y --raw
git add dev/experiments/statehead-nanochat-d12-controlled-v1.yaml
git diff --cached --check
git commit -m 'Record live Vast StateHead full run'
git push origin codex/statehead-nanochat
git add dev/experiments/statehead-nanochat-d12-controlled-v1.yaml
git diff --cached --check
git commit -m 'Add matched GPT control to Vast run'
git push origin codex/statehead-nanochat
```

Remote preparation and StateHead run:

```bash
ssh -p 30532 root@ssh2.vast.ai 'nvidia-smi -L; python3 --version; df -h /workspace'
ssh -p 30532 root@ssh2.vast.ai 'git clone --branch codex/statehead-nanochat --depth 1 https://github.com/samfurr/nanochat.git /workspace/nanochat'
ssh -p 30532 root@ssh2.vast.ai 'cd /workspace/nanochat && git cat-file -e b952753243ea14ac391d617244f2fbd52ba0a487^{commit}'
ssh -p 30532 root@ssh2.vast.ai 'cd /workspace/nanochat && git fetch --unshallow && git diff --exit-code b952753243ea14ac391d617244f2fbd52ba0a487 -- nanochat/gpt.py nanochat/statehead.py nanochat/optim.py nanochat/dataloader.py nanochat/checkpoint_manager.py scripts/base_train.py scripts/base_eval.py'
ssh -p 30532 root@ssh2.vast.ai 'tmux new-session -d -s statehead-full "cd /workspace/nanochat && env PATH=/venv/main/bin:$PATH RUN_ARCHES=statehead NPROC_PER_NODE=8 DOWNLOAD_WORKERS=16 NANOCHAT_BASE_DIR=/workspace/nanochat-controlled WANDB_RUN_PREFIX=dummy SETUP_ENV=1 PREPARE_DATA=1 bash runs/statehead_d12_controlled.sh > /workspace/statehead-full-driver.log 2>&1"'
```

The first `git cat-file`/runner attempt failed because the shallow clone did
not contain the pinned commit. The unshallow command corrected that before any
paid training work.

StateHead verification and GPT run:

```bash
ssh -p 30532 root@ssh2.vast.ai 'cd /workspace/nanochat && sha256sum -c /workspace/nanochat-controlled/controlled-results/statehead-nanochat-d12-controlled-v1.sha256'
ssh -p 30532 root@ssh2.vast.ai 'test "$(find /workspace/nanochat-controlled/base_checkpoints/statehead-nanochat-d12-controlled-v1 -maxdepth 1 -name "optim_002520_rank*.pt" | wc -l)" -eq 8'
ssh -p 30532 root@ssh2.vast.ai 'cp /workspace/nanochat-controlled/controlled-results/environment.txt /workspace/nanochat-controlled/controlled-results/statehead-environment.txt; cp /workspace/nanochat-controlled/controlled-results/dataset-verification.txt /workspace/nanochat-controlled/controlled-results/statehead-dataset-verification.txt; cp /workspace/nanochat-controlled/controlled-results/run-exit.txt /workspace/nanochat-controlled/controlled-results/statehead-run-exit.txt; mv /workspace/nanochat-controlled/controlled-results/COMPLETED /workspace/nanochat-controlled/controlled-results/statehead-COMPLETED'
ssh -p 30532 root@ssh2.vast.ai 'tmux new-session -d -s gpt-full "cd /workspace/nanochat && env PATH=/venv/main/bin:$PATH RUN_ARCHES=gpt NPROC_PER_NODE=8 DOWNLOAD_WORKERS=16 NANOCHAT_BASE_DIR=/workspace/nanochat-controlled WANDB_RUN_PREFIX=dummy SETUP_ENV=0 PREPARE_DATA=0 bash runs/statehead_d12_controlled.sh > /workspace/gpt-full-driver.log 2>&1"'
ssh -p 30532 root@ssh2.vast.ai 'cd /workspace/nanochat && sha256sum -c /workspace/nanochat-controlled/controlled-results/gpt-d12-controlled-v1.sha256'
ssh -p 30532 root@ssh2.vast.ai 'test "$(find /workspace/nanochat-controlled/base_checkpoints/gpt-d12-controlled-v1 -maxdepth 1 -name "optim_002520_rank*.pt" | wc -l)" -eq 8'
```

During both runs, read-only `ssh` commands using `tmux list-sessions`,
`tail`, `grep`, `find`, `df`, and `nvidia-smi` were repeated approximately
every 30–60 seconds to detect completion, NaNs, errors, or guard risk.

Retrieval, verification, and teardown:

```bash
mktemp -d /private/tmp/nanochat-vast-45630532.XXXXXX
scp -C -r -P 30532 root@ssh2.vast.ai:/workspace/nanochat-controlled/controlled-results /private/tmp/nanochat-vast-45630532.9h6fcF/
cd /private/tmp/nanochat-vast-45630532.9h6fcF/controlled-results && shasum -a 256 -c <(sed -n '1,3s#  /workspace/nanochat-controlled/controlled-results/#  #p' statehead-nanochat-d12-controlled-v1.sha256)
cd /private/tmp/nanochat-vast-45630532.9h6fcF/controlled-results && shasum -a 256 -c <(sed -n '1,3s#  /workspace/nanochat-controlled/controlled-results/#  #p' gpt-d12-controlled-v1.sha256)
scp -C -P 30532 root@ssh2.vast.ai:/workspace/statehead-full-driver.log root@ssh2.vast.ai:/workspace/gpt-full-driver.log /private/tmp/nanochat-vast-45630532.9h6fcF/controlled-results/
vastai show instances --raw
vastai show volumes --raw
vastai show user --raw
vastai destroy instance 45630532 -y --raw
vastai show instances --raw
vastai show volumes --raw
vastai show user --raw
vastai show invoices-v1 --charges --charge-type instance --start-date 1784826000 --end-date 1784832000 --latest-first --limit 20 --raw
mkdir -p dev-ignore/statehead-vast-d12-controlled-20260723 dev/results/statehead-vast-d12-controlled-20260723
cp -R /private/tmp/nanochat-vast-45630532.9h6fcF/controlled-results dev-ignore/statehead-vast-d12-controlled-20260723/
rsync -a --exclude='*-model_002520.pt' /private/tmp/nanochat-vast-45630532.9h6fcF/controlled-results/ dev/results/statehead-vast-d12-controlled-20260723/
```

Local analysis and final validation:

```bash
git rev-parse HEAD
git status --short
rg -n 'Validation bpb|Peak memory|Total training time|Minimum validation bpb|train bpb|val bpb|CORE metric' dev/results/statehead-vast-d12-controlled-20260723/*-{train,eval}.log
.venv/bin/python -c '<parse all logged step throughputs and report mean and median>'
.venv/bin/python -c '<parse both CORE CSVs and report per-task deltas>'
jq '.' dev/results/statehead-vast-d12-controlled-20260723/statehead-nanochat-d12-controlled-v1-meta_002520.json
jq '.' dev/results/statehead-vast-d12-controlled-20260723/gpt-d12-controlled-v1-meta_002520.json
NANOCHAT_DTYPE=float32 .venv/bin/python -c 'import torch; torch.set_default_device("meta"); from nanochat.statehead import StateHead, StateHeadConfig; m=StateHead(StateHeadConfig(n_layer=12,n_embd=1152,n_head=9)); print(m.num_scaling_params()); print("scaling_params", m.num_scaling_params()["transformer_matrices"] + m.num_scaling_params()["lm_head"]); print("flops_per_token", m.estimate_flops()); print("state_bytes_per_row_bf16", m.recurrent_state_bytes(dtype=torch.bfloat16))'
NANOCHAT_DTYPE=float32 .venv/bin/python -m pytest tests/test_statehead.py -q
NANOCHAT_DTYPE=float32 .venv/bin/python -m pytest -q -k 'not test_memory_limit'
test -f /tmp/nanochat-statehead-smoke.W8Wrbv/tokenizer/tokenizer.pkl
.venv/bin/python -c 'import torch; print(torch.__version__, torch.backends.mps.is_built(), torch.backends.mps.is_available())'
NANOCHAT_BASE_DIR=/tmp/nanochat-statehead-smoke.W8Wrbv NANOCHAT_DTYPE=bfloat16 .venv/bin/python -m scripts.base_train --arch=statehead --seed=1337 --depth=1 --aspect-ratio=32 --head-dim=32 --max-seq-len=8 --device-batch-size=1 --total-batch-size=8 --num-iterations=5 --eval-every=-1 --eval-tokens=16 --core-metric-every=-1 --sample-every=-1 --save-every=-1 --device-type=mps --model-tag=phase3-final-statehead-mps
NANOCHAT_BASE_DIR=/tmp/nanochat-statehead-smoke.W8Wrbv NANOCHAT_DTYPE=bfloat16 .venv/bin/python -m scripts.base_train --seed=1337 --depth=1 --aspect-ratio=32 --head-dim=32 --max-seq-len=8 --window-pattern=L --device-batch-size=1 --total-batch-size=8 --num-iterations=5 --eval-every=-1 --eval-tokens=16 --core-metric-every=-1 --sample-every=-1 --save-every=-1 --device-type=mps --model-tag=phase3-final-default-gpt-mps
bash -n runs/statehead_d12_controlled.sh
DRY_RUN=1 RUN_ARCHES=gpt,statehead NPROC_PER_NODE=8 DOWNLOAD_WORKERS=16 NANOCHAT_BASE_DIR=/workspace/nanochat-controlled WANDB_RUN_PREFIX=dummy SETUP_ENV=1 PREPARE_DATA=1 bash runs/statehead_d12_controlled.sh
.venv/bin/python -m compileall -q nanochat scripts tests dev/statehead_cuda_preflight.py
git diff --check
git diff --quiet b952753243ea14ac391d617244f2fbd52ba0a487 -- nanochat/gpt.py nanochat/statehead.py nanochat/statehead_cuda.py nanochat/optim.py nanochat/dataloader.py nanochat/checkpoint_manager.py scripts/base_train.py scripts/base_eval.py
.venv/bin/python -c '<validate both YAML manifests, run exits, completion markers, exact shards, metadata, all 5,040 finite logged losses, and CORE aggregates>'
cd dev-ignore/statehead-vast-d12-controlled-20260723/controlled-results && shasum -a 256 -c <(sed -n '1,3s#  /workspace/nanochat-controlled/controlled-results/#  #p' statehead-nanochat-d12-controlled-v1.sha256)
cd dev-ignore/statehead-vast-d12-controlled-20260723/controlled-results && shasum -a 256 -c <(sed -n '1,3s#  /workspace/nanochat-controlled/controlled-results/#  #p' gpt-d12-controlled-v1.sha256)
```

Results:

```text
Focused StateHead suite: 46 passed, 15 skipped in 2.18s
Full suite excluding test_memory_limit: 89 passed, 29 skipped,
  1 deselected in 4.62s
Host MPS: PyTorch 2.9.1, built=True, available=True
StateHead BF16/MPS five-step smoke: loss 5.924298 -> 5.923664,
  finite checkpoint/metadata/optimizer save passed
Default GPT BF16/MPS five-step smoke: loss 5.924309 -> 5.923669,
  finite checkpoint/metadata/optimizer save passed
Runner shell syntax and paired dry run: passed
compileall and git diff --check: passed
Model/training source unchanged from b952753: passed
YAML, run exits, completion markers, exact shards, metadata,
  5,040 finite logged losses, and CORE aggregates: passed
Downloaded StateHead and GPT model/metadata/CORE SHA-256 checks: passed
```

Two validation-harness commands were corrected during final QA:

1. An initial combined source-diff check included the runner, which is
   intentionally newer than model commit `b952753`; the corrected model and
   training-source-only check passed.
2. An initial log glob also selected `tokenizer-train.log`, which has no model
   losses; restricting the finite-loss assertion to the two architecture
   training logs passed with exactly 2,520 values each.

These were harness-selection failures, not training or model-test failures.

The authored manifests, log entry, and report pass `git diff --check`. A
whole-staged-tree check intentionally reports source-emitted trailing spaces in
the raw CORE CSVs and driver/training logs. Those retrieved evidence files were
not normalized because preserving their remote bytes is more important than
restyling them.

## Changed files

Final tracked changes are:

- `dev/STATEHEAD_LOG.md`
- `dev/experiments/gpt-d12-controlled-v1.yaml`
- `dev/experiments/statehead-nanochat-d12-controlled-v1.yaml`
- this `REPORT.md`
- every lightweight evidence file listed in the Artifacts section

The two model files listed in Artifacts were added only to the ignored local
`dev-ignore/` bundle. Pre-existing untracked `.DS_Store`, `.agents/`, and
`STATEHEAD_NANOCHAT_CODEX_BRIEF.md` were not modified.

## Unresolved questions

1. This run answers only the same-shape question. It does not answer the
   parameter-matched question. A width-1,152, depth-12 StateHead has
   117,374,976 scaling parameters, 6.61% above GPT's 110,100,912, and is a
   practical first parameter-matched candidate.
2. Width 1,152 is estimated at 704,374,416 FLOPs/token and 27,648 persistent
   BF16 state bytes per row. Its batch-32 memory fit has not been tested; based
   on the width scaling and the 41.4 GiB d768 peak, a batch-16 capacity probe
   with gradient accumulation is the conservative next gate.
3. One seed is not a confidence interval. The brief recommends multiple seeds
   only after a promising run; this same-shape result is not promising enough
   to justify that spend.
4. Phase 4 inference/SFT work remains unimplemented. Given the base-model gap,
   it should not precede the cheaper parameter-matched sizing/capacity decision.
5. The remote optimizer shards are no longer recoverable after instance
   deletion. Their hashes were verified, but only final inference/evaluation
   model checkpoints were downloaded.
6. The Vast host was unverified and had 0.9703 reliability. No fault was
   observed, but the single-run result should not be treated as a
   high-confidence host-independent estimate.

## Exact next recommended command

Do not launch another paid run yet. Reproduce the width-1,152
parameter-matched sizing candidate locally:

```bash
NANOCHAT_DTYPE=float32 .venv/bin/python -c 'import torch; torch.set_default_device("meta"); from nanochat.statehead import StateHead, StateHeadConfig; m=StateHead(StateHeadConfig(n_layer=12,n_embd=1152,n_head=9)); print(m.num_scaling_params()); print("scaling_params", m.num_scaling_params()["transformer_matrices"] + m.num_scaling_params()["lm_head"]); print("flops_per_token", m.estimate_flops()); print("state_bytes_per_row_bf16", m.recurrent_state_bytes(dtype=torch.bfloat16))'
```

Expected scaling parameters: `117374976`. The next implementation task after
that read-only check is a separately named parameter-matched manifest/runner
with an approved one-H100 batch-16 capacity preflight; it should not overwrite
the completed same-shape experiment.
