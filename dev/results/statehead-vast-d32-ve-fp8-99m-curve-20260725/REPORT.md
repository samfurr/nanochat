# StateHead d32 value-embedding 99-minute checkpoint curve

## Outcome

The exact single-node 8x H100 SXM run completed, all ten checkpoints received
full CORE and BPB evaluation, all final optimizer shards were retained, and the
complete artifact bundle was verified locally before the instance was
destroyed.

The final 99.87-minute checkpoint reached:

- CORE: `0.142582`
- validation BPB: `0.803146`
- train BPB: `0.804883`
- tokens: `5,156,896,768`

This does **not** demonstrate GPT parity. The matched-wall-clock GPT d24
reference reached CORE `0.256525`.

## Fixed model and training recipe

- StateHead depth: 32
- width: 2,048
- recurrent heads: 16
- head dimension: 128
- sequence length: 2,048
- value embeddings: 16 banks, on GPT-rule odd zero-based layers
- total parameters: `1,879,379,034`
- scaling parameters: `738,200,576`
- value-embedding parameters: `1,073,741,824`
- scan: native CUDA, chunk size 32, FP32 recurrence
- projections: tensorwise FP8 where eligible, 65 converted linears
- activations and embeddings: BF16
- world size: 8
- device batch: 8
- global batch: 1,048,576 tokens
- measured training target: 5,940 seconds
- actual measured training: 5,992.328 seconds / 99.872 minutes
- full process wall time for the training command: 6,090 seconds

The model and training code remained pinned to
`0e8e426c33d46ef28a4da1362ec5c7a5a181621a`. The runner-only schedule fix was
applied at checkout `6d40506d875403e46b287ddf3ee069a1cd4928a9`.

## CUDA and learning gates

The one-H100 gate had already passed 9 focused CUDA numerical/gradient tests.
The full host then passed the exact production-shape eight-rank gate:

| Check | Result |
|---|---:|
| World size | 8 |
| FP8 eligible linears | 65 |
| Populated gradient finiteness | passed |
| Parameter checksum spread | 0.0 |
| Median synthetic throughput | 661,374.68 tokens/s |
| Peak allocated VRAM per rank | 30,685,987,328 bytes |
| Peak reserved VRAM per rank | 31,226,593,280 bytes |

The fresh 125-step real-data calibration also passed:

| Metric | Initial | Final |
|---|---:|---:|
| validation BPB | 3.160622 | 1.244822 |
| training loss | 10.400154 | 4.204415 |

Calibration measured 1.210427 seconds per step and fixed the full run at 4,918
steps before it began.

## Full checkpoint curve

| Measured minutes | Step | Tokens | Train BPB | Validation BPB | CORE |
|---:|---:|---:|---:|---:|---:|
| 10.082 | 507 | 531,628,032 | 1.001712 | 1.001484 | 0.059933 |
| 20.170 | 1,002 | 1,050,673,152 | 0.938517 | 0.937683 | 0.074475 |
| 30.274 | 1,498 | 1,570,766,848 | 0.912539 | 0.911416 | 0.088259 |
| 40.373 | 1,994 | 2,090,860,544 | 0.898480 | 0.897150 | 0.099069 |
| 50.448 | 2,489 | 2,609,905,664 | 0.879836 | 0.878349 | 0.107033 |
| 60.541 | 2,985 | 3,129,999,360 | 0.857238 | 0.855761 | 0.114039 |
| 70.632 | 3,481 | 3,650,093,056 | 0.838837 | 0.837278 | 0.116155 |
| 80.724 | 3,977 | 4,170,186,752 | 0.824207 | 0.822519 | 0.125528 |
| 90.798 | 4,472 | 4,689,231,872 | 0.812248 | 0.810504 | 0.135660 |
| 99.872 | 4,918 | 5,156,896,768 | 0.804883 | 0.803146 | 0.142582 |

Both metrics were still improving at the final checkpoint. From 90.80 to 99.87
minutes, CORE increased by `0.006922` and validation BPB decreased by
`0.007358`. The curve therefore does not support a claim that training had
fully flattened by 99 minutes.

## Comparison

Against the previous d32 StateHead without value embeddings:

- CORE improved from `0.137938` to `0.142582`;
- absolute CORE gain: `0.004644`;
- relative CORE gain: `3.37%`;
- validation BPB improved from `0.824316` to `0.803146`.

Against the matched-wall-clock GPT d24 reference:

- StateHead final CORE: `0.142582`;
- GPT reference CORE: `0.256525`;
- absolute gap: `-0.113943`;
- StateHead reached `55.58%` of the GPT CORE value.

The value embeddings help, and the late curve remains positive, but the
observed gain is far too small to claim capability parity. Raw parameter count
is also not matched: the StateHead model has 1.879B total parameters versus
1.384B for GPT, although the scaling-parameter counts are within about 1.15%.

## Execution incidents

Three orchestration issues did not alter the model recipe or scientific run:

1. The Vast image did not expose a global `python` shim. The first launcher
   exited before downloads, compilation, or training. Relaunching with the
   locked repository virtual environment first on `PATH` fixed it.
2. After successful calibration, the wrapper attempted to assign an
   environment value to the readonly shell constant
   `CHECKPOINT_INTERVAL_SECONDS`. The runner-only variable was renamed to
   `CALIBRATION_INTERVAL_SECONDS`, committed, and the completed gates were
   reused without repetition.
3. The original artifact manifest included `launcher.log`, which continued to
   grow after being hashed. That was the sole original checksum mismatch; the
   other 71 entries passed. A post-completion manifest was generated and all 76
   of its entries passed remotely and locally. The runner now excludes the
   still-growing launcher log from its pre-exit manifest.

## Artifacts and verification

Local directory:

`dev/results/statehead-vast-d32-ve-fp8-99m-curve-20260725`

Retained large artifacts:

- 10 model checkpoints: 52,358,796,220 bytes / 48.763 GiB
- 8 final optimizer shards: 7,789,030,056 bytes / 7.254 GiB
- full local directory: approximately 56 GiB

Important hashes:

```text
221e4f03618d72679d64c78b074f352d883f7d62757b83e221c224eb9923803c  post-completion.sha256
dd85cc2ae08ef852441c61383c71d8e1b2a7b41ed8c85dcc377f66ee6bc78292  statehead-d32-ve-fp8-99m-model_004918.pt
c7f1610a90dd358153875839a084e5bd13844935279ce9a8c5533b9e333eface  statehead-d32-ve-fp8-99m-curve.json
bb0fff8a4312b8da3f3e103061dc0a44b0b6f46496a1ec1db0b4ae44d8116704  statehead-d32-ve-fp8-99m-curve.csv
```

`local-post-completion-verification.txt` records 76 successful local checksum
checks.

## Infrastructure and cost

- Vast offer: `45759640`
- instance: `45763000`
- hardware: verified single-node 8x NVIDIA H100 80GB HBM3
- NVLink reported by the offer: 478.116 GB/s
- storage: 150 GB
- quoted all-in rate: `$28.3361111111/hour`
- approved ceiling: `$115.00`
- final invoice: `$98.27`
- post-cleanup instances: 0
- post-cleanup volumes: 0

The independent four-hour destruction guard was cancelled only after local
artifact verification and explicit instance destruction.

## Primary execution commands

Read-only offer and account gate:

```bash
vastai search offers 'gpu_name=H100_SXM num_gpus=8 rentable=true verified=true' -n --storage 150 -o dph --raw
vastai show instances --raw
vastai show volumes --raw
vastai show user --raw
```

Provisioning and hard guard:

```bash
vastai create instance 45759640 --image 'vastai/pytorch:@vastai-automatic-tag' --disk 150 --ssh --direct --cancel-unavail --label statehead-d32-ve-fp8-99m-curve
sleep 14400; vastai destroy instance 45763000 -y
```

Disposable-host environment:

```bash
uv sync --extra gpu --frozen
uv pip install setuptools==83.0.0
```

Full runner:

```bash
PATH=/workspace/nanochat/.venv/bin:/root/.local/bin:$PATH \
SETUP_ENV=0 PREPARE_DATA=1 DOWNLOAD_WORKERS=16 \
NANOCHAT_BASE_DIR=/workspace/nanochat-statehead-d32-ve WANDB_RUN=dummy \
bash runs/statehead_d32_ve_fp8_99m_curve.sh
```

Resume after the wrapper-only schedule fix:

```bash
PATH=/workspace/nanochat/.venv/bin:/root/.local/bin:$PATH \
SETUP_ENV=0 PREPARE_DATA=0 REUSE_CALIBRATION=1 \
NANOCHAT_BASE_DIR=/workspace/nanochat-statehead-d32-ve WANDB_RUN=dummy \
bash runs/statehead_d32_ve_fp8_99m_curve.sh
```

Final verification and cleanup:

```bash
shasum -a 256 -c post-completion.sha256
vastai destroy instance 45763000 -y
vastai show instances --raw
vastai show volumes --raw
```

Read-only monitoring, log-tail, disk, GPU-utilization, and transfer-progress
polls are preserved in the task transcript but omitted from this command
summary because they did not mutate the run.

## Unresolved questions

1. How much of the remaining GPT gap can be recovered by additional training
   from the final restartable checkpoint, given that both curves still improve?
2. Would a parameter-matched architecture change, especially a d24-like
   compute allocation, outperform simply extending this d32 schedule?
3. Are the value embeddings' modest quality gains worth their 1.074B raw
   parameters and checkpoint cost?

No continuation or architecture sweep was launched.
