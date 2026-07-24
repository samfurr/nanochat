# StateHead d32 CUDA v4 FP8 99-minute result

## Outcome

The StateHead run completed successfully, but it did **not** reach GPT-2
capability:

| Metric | StateHead d32 | Comparison |
|---|---:|---:|
| CORE | **0.137938** | GPT-2 threshold: `0.256525` |
| CORE delta to threshold | **-0.118587** | must be positive for parity |
| Validation BPB | **0.824316** | GPT Run 6 reference: `0.71800` |
| Measured trainer time | **5,970.262 s (99.504 min)** | target: `5,940 s (99 min)` |
| Training tokens | **5,419,040,768** | 5,168 optimizer steps |
| Scaling parameters | **738,197,504** | GPT d24: `729,810,624` |
| Median logged throughput after warmup | **905,565 tok/s** | eight H100s |

This is a completed negative result, not a parity result. The measured trainer
time also exceeded the nominal 99-minute target by 30.262 seconds, so it is not
a strict sub-99-minute run. The published GPT Run 6 comparison was not rerun on
this Vast host; its average CORE of `0.262634` and 99-minute duration are
reference values from `dev/LEADERBOARD.md`.

## Fixed experiment

- Model/training source pin:
  `f68feae68a6be965c8f76c40026cb134aba2c216`
- Launch checkout:
  `ede67fb538fe8e947c6b1525002d465bf39bb076`
- Architecture: StateHead d32, width 2,048, 16 state heads, head dimension
  128, no attention, no MLP
- Sequence length: 2,048
- Scan: accepted CUDA v4, chunk size 32, FP32 recurrent accumulation
- Precision: tensorwise FP8 eligible projections, BF16 activations, FP32
  master weights
- Training: 8 ranks, device batch 8/rank, global batch 1,048,576 tokens,
  seed 1337
- Data: exact ClimbMix train shards 00000 through 00169 and validation shard
  06542
- Schedule: 40 warmup steps, 0.65 warmdown ratio, final LR fraction 0.05
- Final selection: the predetermined calibrated final step; no intermediate
  validation or checkpoint selection

The GPT implementation and its Flash Attention path were not modified for
this run. The later experimental CUDA v7-v10 probes were not used; the pinned
model paths match the accepted CUDA v4 source.

## Production-shape gate and calibration

The exact d32 FP8/DDP CUDA configuration passed before the full run:

- CUDA extension compiled with PyTorch 2.9.1+cu128 and CUDA 12.8.
- All eight devices were NVIDIA H100 80GB HBM3.
- 65 linears were converted to tensorwise FP8.
- All populated parameter gradients were finite.
- Cross-rank parameter checksum spread was exactly `0.0`.
- Three post-warmup probe steps had median global throughput
  `736,199.495 tok/s`.
- Maximum per-rank allocated/reserved memory was
  `23,782,129,152 / 25,736,249,344` bytes.

The 125-step real-data calibration also passed:

| Calibration measure | Initial | Final |
|---|---:|---:|
| Validation BPB | 3.160622 | 1.279521 |
| Logged training loss | 10.400154 | 4.319742 |

The 114 timed calibration steps took 131.307 seconds, or
1.151819867 seconds/step. Applying the fixed 5,940-second trainer budget
produced a predetermined horizon of 5,168 steps. The full run sustained a
slightly slower 1.1577-second mean logged step, causing the 30.262-second
overrun.

## Full training

- Total tokens: `5,419,040,768`
- Tokens per scaling parameter: `7.34`
- Estimated training FLOPs: `2.400513e19`
- Smoothed training loss: `10.400154` initially, `2.784258` finally
- Minimum logged loss: `2.623209` at step 5,126
- Median/mean logged throughput after warmup:
  `905,565 / 905,751.748 tok/s`
- Median/mean logged step:
  `1.15792 / 1.15770 s`
- Peak allocated memory reported by the trainer: `25,625.78 MiB`
- Trainer time: `5,970.262 s`
- Whole training command wall time, including compile and final checkpoint:
  `6,020 s`
- Final checkpoint: step 5,168

The full training run started from newly initialized seed-1337 weights. The
calibration checkpoint was not used as initialization.

## Final evaluation

The final checkpoint was evaluated with full CORE (`max_per_task=-1`) and
40 Mi validation/train BPB splits.

| Task | Accuracy | Centered |
|---|---:|---:|
| hellaswag_zeroshot | 0.423123 | 0.230831 |
| jeopardy | 0.005196 | 0.005196 |
| bigbench_qa_wikidata | 0.270754 | 0.270754 |
| arc_easy | 0.608586 | 0.478114 |
| arc_challenge | 0.337884 | 0.117179 |
| copa | 0.610000 | 0.220000 |
| commonsense_qa | 0.286650 | 0.108313 |
| piqa | 0.712731 | 0.425462 |
| openbook_qa | 0.362000 | 0.149333 |
| lambada_openai | 0.226858 | 0.226858 |
| hellaswag | 0.419837 | 0.226449 |
| winograd | 0.575092 | 0.150183 |
| winogrande | 0.507498 | 0.014996 |
| bigbench_dyck_languages | 0.000000 | 0.000000 |
| agi_eval_lsat_ar | 0.226087 | 0.032609 |
| bigbench_cs_algorithms | 0.002273 | 0.002273 |
| bigbench_operators | 0.104762 | 0.104762 |
| bigbench_repeat_copy_logic | 0.000000 | 0.000000 |
| squad | 0.025166 | 0.025166 |
| coqa | 0.087812 | 0.087812 |
| boolq | 0.613456 | -0.017222 |
| bigbench_language_identification | 0.250600 | 0.175578 |
| **CORE** |  | **0.137938** |

Train BPB was `0.825806`; validation BPB was `0.824316`. The evaluation
completed with process status zero and wrote the `COMPLETED` marker.

## Failure and recovery record

1. The first approved Vast instance, `45647137`, failed during startup and was
   replaced without producing training evidence.
2. The replacement environment's frozen uv sync did not provide
   `setuptools`, so the first runner invocation failed before CUDA compilation
   or training. `setuptools==83.0.0` was installed into the remote virtual
   environment only; no result was reused from this failed invocation.
3. The production gate and calibration then passed, but the runner stopped
   before full training with:

   ```text
   runs/statehead_d32_fp8_99m.sh: line 253:
   TARGET_TRAINING_SECONDS: readonly variable
   ```

4. Commit `ede67fb` renamed the parser-only environment variable to
   `CALIBRATION_TARGET_SECONDS` and added an explicit, validated
   `REUSE_CALIBRATION=1` path. Default behavior remains a fresh gate. The
   remote checkout was updated, the already-completed gate/calibration
   evidence was validated, and the full run began from fresh weights.

## Artifacts and integrity

Tracked lightweight evidence is this directory. The complete ignored local
bundle, including the 2.9 GiB final model, is:

```text
dev-ignore/statehead-vast-d32-fp8-99m-20260724/results/
```

Verified SHA-256 values:

```text
4283abbfb8456da5dc46f3281c8407d86af0e2d82a3a3cdaf539821ffc26a20d  statehead-d32-fp8-99m-model_005168.pt
a924edbe444cd3bc4da442daea85d264cec023ddcb24745bd13e667001fbcad2  statehead-d32-fp8-99m-core.csv
d649c345ee7fe64c4f00b9377f4bb467dcf3ea2f8c4053efad099304695dc42d  statehead-d32-fp8-99m-meta_005168.json
```

All eight optimizer shards were present and passed remote SHA-256
verification before deletion. They were not downloaded; their hashes remain
in `statehead-d32-fp8-99m.sha256`.

Other retained files:

- `calibration.json`
- `environment.txt`
- `nvidia-smi.csv`
- exact dataset and tokenizer logs/checksums
- production FP8/DDP gate JSON/log
- calibration metadata/log
- complete training and evaluation logs
- final CORE CSV and metadata
- run exit and completion markers

## Final local verification

```text
Focused StateHead suite: 53 passed, 36 skipped in 2.97s
Full suite excluding the known macOS test_memory_limit:
  96 passed, 50 skipped, 1 deselected in 5.33s
StateHead BF16/MPS five-step smoke:
  loss 5.924298 -> 5.923664; finite run and checkpoint save passed
Runner shell syntax and d32 dry run: passed
Manifest, exit, completion marker, metadata, CORE, and tracked hashes: passed
Tracked CORE/metadata byte comparison against the full bundle: passed
Ignored final-model SHA-256: passed
Model/training source unchanged from f68feae: passed
compileall and authored tracked diff check: passed
```

The first evidence validator used the padded raw CSV header as a literal
`Task` key and raised `KeyError: 'Task'`. It also lacked shell fail-fast, so
subsequent successful commands masked that failure in the shell status. The
validator was corrected to strip CSV keys/values and run under `set -e`; the
complete corrected check passed. This was a local QA-harness error, not a
model, training, artifact, or evaluation error.

## Cost and cleanup

The replacement instance invoice was `$41.385`, including `$41.214` GPU,
`$0.120` disk, and `$0.051` bandwidth. The failed initial instance was
`$2.628`. Total experiment invoicing was therefore **$44.013**, below the
approved `$70` ceiling.

After local checksum verification, instance `45647442` was destroyed and the
independent termination guard was cancelled. The final Vast audit showed zero
instances and zero volumes. No billable resource remains.

## Changed files

The run and its closure changed:

- `runs/statehead_d32_fp8_99m.sh`
- `dev/experiments/statehead-nanochat-d32-fp8-99m-v1.yaml`
- `dev/STATEHEAD_LOG.md`
- this `REPORT.md`
- `COMPLETED`
- `calibration.json`
- `dataset-all.log`
- `dataset-first-eight.log`
- `dataset-verification.txt`
- `environment.txt`
- `nvidia-smi.csv`
- `run-exit.txt`
- `statehead-d32-fp8-99m-core.csv`
- `statehead-d32-fp8-99m-eval.log`
- `statehead-d32-fp8-99m-meta_005168.json`
- `statehead-d32-fp8-99m-train-wall-seconds.txt`
- `statehead-d32-fp8-99m-train.log`
- `statehead-d32-fp8-99m.sha256`
- `statehead-d32-fp8-calibration-meta_000125.json`
- `statehead-d32-fp8-calibration-train.log`
- `statehead-d32-fp8-ddp-gate.json`
- `statehead-d32-fp8-ddp-gate.log`
- `tokenizer-train.log`
- `tokenizer.sha256`
- `uv-sync.log`

The final model exists only in the ignored `dev-ignore/` bundle. Pre-existing
untracked `.DS_Store`, `.agents/`, and
`STATEHEAD_NANOCHAT_CODEX_BRIEF.md` were not modified.

## Command ledger

Commands are grouped only where an identical read-only monitoring command was
repeated. Ephemeral SSH endpoints and the API key are omitted. No paid action
was taken outside the user's approved ceiling.

Repository and plan inspection:

```bash
git rev-parse HEAD
git status --short
sed -n '1,240p' /Users/haybales/.codex/skills/vastai/SKILL.md
sed -n '241,520p' /Users/haybales/.codex/skills/vastai/SKILL.md
wc -l STATEHEAD_NANOCHAT_CODEX_BRIEF.md README.md
sed -n '1,240p' STATEHEAD_NANOCHAT_CODEX_BRIEF.md
sed -n '241,480p' STATEHEAD_NANOCHAT_CODEX_BRIEF.md
sed -n '481,720p' STATEHEAD_NANOCHAT_CODEX_BRIEF.md
sed -n '721,960p' STATEHEAD_NANOCHAT_CODEX_BRIEF.md
sed -n '961,1147p' STATEHEAD_NANOCHAT_CODEX_BRIEF.md
sed -n '1,230p' README.md
sed -n '1,430p' runs/statehead_d32_fp8_99m.sh
sed -n '1,170p' dev/experiments/statehead-nanochat-d32-fp8-99m-v1.yaml
sed -n '200,230p' dev/LEADERBOARD.md
git diff --quiet f68feae68a6be965c8f76c40026cb134aba2c216 -- nanochat/gpt.py nanochat/statehead.py nanochat/statehead_cuda.py nanochat/csrc/statehead_cuda.cpp nanochat/csrc/statehead_cuda_kernel.cu nanochat/fp8.py nanochat/optim.py nanochat/dataloader.py nanochat/checkpoint_manager.py scripts/base_train.py scripts/base_eval.py
```

Vast lifecycle and safety guard:

```bash
vastai show instance 45647442
vastai start instance 45647442
sleep 14100
vastai destroy instance 45647442
vastai show instances --raw
vastai show volumes --raw
vastai show invoices
```

The `sleep` plus destroy command ran as a separate independent guard. Its
destroy action was cancelled after verified artifact retrieval because the
instance had already been manually destroyed.

Remote setup, validation, and launches, executed through SSH:

```bash
git fetch origin codex/statehead-nanochat
git checkout ede67fb538fe8e947c6b1525002d465bf39bb076
git diff --quiet f68feae68a6be965c8f76c40026cb134aba2c216 -- nanochat/gpt.py nanochat/statehead.py nanochat/statehead_cuda.py nanochat/csrc/statehead_cuda.cpp nanochat/csrc/statehead_cuda_kernel.cu nanochat/fp8.py nanochat/optim.py nanochat/dataloader.py nanochat/checkpoint_manager.py scripts/base_train.py scripts/base_eval.py
SETUP_ENV=1 PREPARE_DATA=1 NANOCHAT_BASE_DIR=/workspace/nanochat-statehead-d32 bash runs/statehead_d32_fp8_99m.sh
/usr/local/bin/uv pip install --python .venv/bin/python setuptools==83.0.0
SETUP_ENV=0 PREPARE_DATA=0 NANOCHAT_BASE_DIR=/workspace/nanochat-statehead-d32 bash runs/statehead_d32_fp8_99m.sh
git fetch origin codex/statehead-nanochat
git checkout ede67fb538fe8e947c6b1525002d465bf39bb076
SETUP_ENV=0 PREPARE_DATA=0 REUSE_CALIBRATION=1 NANOCHAT_BASE_DIR=/workspace/nanochat-statehead-d32 bash runs/statehead_d32_fp8_99m.sh
```

The runner expanded those commands into:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 --module dev.statehead_cuda_preflight --arch=statehead --device-batch-size=8 --steps=4 --warmup-steps=1 --layers=32 --model-width=2048 --heads=16 --sequence-length=2048 --scan-chunk-size=32 --scan-backend=cuda --fp8 --verify-gradients
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --arch=statehead --device-type=cuda --depth=32 --aspect-ratio=64 --head-dim=128 --max-seq-len=2048 --statehead-scan-backend=cuda --statehead-scan-chunk-size=32 --num-iterations=125 --device-batch-size=8 --total-batch-size=1048576 --seed=1337 --fp8 --fp8-recipe=tensorwise
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --arch=statehead --device-type=cuda --depth=32 --aspect-ratio=64 --head-dim=128 --max-seq-len=2048 --statehead-scan-backend=cuda --statehead-scan-chunk-size=32 --num-iterations=5168 --device-batch-size=8 --total-batch-size=1048576 --seed=1337 --fp8 --fp8-recipe=tensorwise
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-type=cuda --eval=core,bpb --model-tag=statehead-d32-fp8-99m --step=5168 --max-per-task=-1 --device-batch-size=8 --split-tokens=41943040
```

Progress was checked with repeated read-only SSH commands using `pgrep`,
`tail`, `nvidia-smi`, and result/checkpoint `find`/`stat` calls. Ten identical
10-minute waits (`sleep 600`) and shorter waits of 25, 30, 40, 45, 50, and
240 seconds were used between checks. No process was changed by a monitoring
command.

Runner fix and source control:

```bash
bash -n runs/statehead_d32_fp8_99m.sh
DRY_RUN=1 SETUP_ENV=0 PREPARE_DATA=0 NANOCHAT_BASE_DIR=/workspace/nanochat-statehead-d32 bash runs/statehead_d32_fp8_99m.sh
python -c 'import yaml; yaml.safe_load(open("dev/experiments/statehead-nanochat-d32-fp8-99m-v1.yaml"))'
git diff --check
git add runs/statehead_d32_fp8_99m.sh dev/experiments/statehead-nanochat-d32-fp8-99m-v1.yaml
git commit -m "Resume StateHead run after verified calibration"
git push origin codex/statehead-nanochat
```

Artifact retrieval, verification, and cleanup:

```bash
sha256sum -c /workspace/nanochat-statehead-d32/results/statehead-d32-fp8-99m.sha256
rsync -az --partial <VAST_SSH_ENDPOINT>:/workspace/nanochat-statehead-d32/results/ dev-ignore/statehead-vast-d32-fp8-99m-20260724/results/
rsync -a --exclude='statehead-d32-fp8-99m-model_005168.pt' dev-ignore/statehead-vast-d32-fp8-99m-20260724/results/ dev/results/statehead-vast-d32-fp8-99m-20260724/
shasum -a 256 dev-ignore/statehead-vast-d32-fp8-99m-20260724/results/statehead-d32-fp8-99m-model_005168.pt dev-ignore/statehead-vast-d32-fp8-99m-20260724/results/statehead-d32-fp8-99m-core.csv dev-ignore/statehead-vast-d32-fp8-99m-20260724/results/statehead-d32-fp8-99m-meta_005168.json
du -sh dev-ignore/statehead-vast-d32-fp8-99m-20260724/results
vastai destroy instance 45647442
vastai show instances --raw
vastai show volumes --raw
vastai show invoices
```

Final local sizing and verification commands are recorded in the final
repository commit alongside their results.

```bash
NANOCHAT_DTYPE=float32 .venv/bin/python -m pytest tests/test_statehead.py -q
NANOCHAT_DTYPE=float32 .venv/bin/python -m pytest -q -k 'not test_memory_limit'
bash -n runs/statehead_d32_fp8_99m.sh
DRY_RUN=1 SETUP_ENV=0 PREPARE_DATA=0 NANOCHAT_BASE_DIR=/workspace/nanochat-statehead-d32 bash runs/statehead_d32_fp8_99m.sh
.venv/bin/python -c '<validate YAML, exit, completion marker, metadata, CORE, and tracked hashes>'
cmp dev/results/statehead-vast-d32-fp8-99m-20260724/statehead-d32-fp8-99m-core.csv dev-ignore/statehead-vast-d32-fp8-99m-20260724/results/statehead-d32-fp8-99m-core.csv
cmp dev/results/statehead-vast-d32-fp8-99m-20260724/statehead-d32-fp8-99m-meta_005168.json dev-ignore/statehead-vast-d32-fp8-99m-20260724/results/statehead-d32-fp8-99m-meta_005168.json
shasum -a 256 dev-ignore/statehead-vast-d32-fp8-99m-20260724/results/statehead-d32-fp8-99m-model_005168.pt
git diff --quiet f68feae68a6be965c8f76c40026cb134aba2c216 -- nanochat/gpt.py nanochat/statehead.py nanochat/statehead_cuda.py nanochat/csrc/statehead_cuda.cpp nanochat/csrc/statehead_cuda_kernel.cu nanochat/fp8.py nanochat/optim.py nanochat/dataloader.py nanochat/checkpoint_manager.py scripts/base_train.py scripts/base_eval.py
.venv/bin/python -m compileall -q nanochat scripts tests dev/statehead_cuda_preflight.py
git diff --check -- dev/STATEHEAD_LOG.md dev/experiments/statehead-nanochat-d32-fp8-99m-v1.yaml runs/statehead_d32_fp8_99m.sh
NANOCHAT_BASE_DIR=/tmp/nanochat-statehead-smoke.W8Wrbv NANOCHAT_DTYPE=bfloat16 .venv/bin/python -m scripts.base_train --arch=statehead --seed=1337 --depth=1 --aspect-ratio=32 --head-dim=32 --max-seq-len=8 --device-batch-size=1 --total-batch-size=8 --num-iterations=5 --eval-every=-1 --eval-tokens=16 --core-metric-every=-1 --sample-every=-1 --save-every=-1 --device-type=mps --model-tag=statehead-d32-closeout-mps
NANOCHAT_DTYPE=float32 .venv/bin/python -c 'import torch; torch.set_default_device("meta"); from nanochat.statehead import StateHead, StateHeadConfig; m=StateHead(StateHeadConfig(n_layer=24,n_embd=1536,n_head=12,sequence_len=2048,vocab_size=32768,scan_chunk_size=32,scan_backend="cuda")); print(m.num_scaling_params()); print("flops_per_token", m.estimate_flops()); print("state_bytes_per_row_bf16", m.recurrent_state_bytes(dtype=torch.bfloat16))'
git add dev/STATEHEAD_LOG.md dev/experiments/statehead-nanochat-d32-fp8-99m-v1.yaml dev/results/statehead-vast-d32-fp8-99m-20260724
git commit -m "Record StateHead d32 full-run result"
git push origin codex/statehead-nanochat
```

## Unresolved questions

1. The parameter-matched d32 result missed the target by a large margin.
   Whether StateHead's higher throughput can compensate at d24 is still
   untested, but d24/width-1,536 has only `333,447,168` scaling parameters,
   45.69% of GPT d24's `729,810,624`. A d24 run is therefore a throughput and
   optimization-schedule experiment, not another parameter-matched test.
2. The d24 device batch and real-data step time are unknown. They require a
   separately approved production-shape gate and calibration; the d32 horizon
   must not be reused.
3. The final evaluation took only about 4.3 minutes on this host, much less
   than the conservative 99-minute reserve. This does not change the
   leaderboard's training-only time definition.
4. The frozen GPU environment currently relies on a remote-only
   `setuptools==83.0.0` repair before extension compilation. That dependency
   should be made reproducible before another paid run.
5. The full optimizer state is no longer recoverable after instance deletion.
   Its eight remote hashes are retained, and the final model needed for
   evaluation is local.

## Exact next recommended command

Do not launch another paid instance yet. First reproduce the requested d24
same-depth-and-width sizing locally:

```bash
NANOCHAT_DTYPE=float32 .venv/bin/python -c 'import torch; torch.set_default_device("meta"); from nanochat.statehead import StateHead, StateHeadConfig; m=StateHead(StateHeadConfig(n_layer=24,n_embd=1536,n_head=12,sequence_len=2048,vocab_size=32768,scan_chunk_size=32,scan_backend="cuda")); print(m.num_scaling_params()); print("flops_per_token", m.estimate_flops()); print("state_bytes_per_row_bf16", m.recurrent_state_bytes(dtype=torch.bfloat16))'
```

Observed output is 383,963,210 total parameters, 333,447,168 scaling
parameters, 2,001,014,928 estimated FLOPs/token, and 73,728 persistent BF16
state bytes per row. The next implementation task is a separately named d24
runner/manifest with a dry run and an approved capacity/calibration gate; it
must not overwrite this completed d32 result.
