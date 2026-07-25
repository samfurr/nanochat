# StateHead d32 value-embedding 99-minute checkpoint-curve plan

## Decision

Keep the completed StateHead d32 shape unchanged:

- depth 32
- width 2,048
- 16 recurrent heads
- head dimension 128
- sequence length 2,048
- CUDA v4 scan with chunk size 32
- tensorwise FP8 for the same 65 eligible large linears
- BF16 activations and FP32 recurrent accumulation

Add value embeddings using GPT's exact layer-selection rule. At even depth,
`has_ve(layer_idx, n_layer)` selects every odd zero-based layer, so d32 has
16 banks at layers 1, 3, ..., 31.

The GPT value path is not modified. The default StateHead configuration also
remains value-embedding-free; the experiment requires the explicit
`--statehead-value-embeddings` flag.

## Injection rule

GPT injects an input-gated token embedding into the attention value:

```text
v = c_v(x) + 3 * sigmoid(W_ve * x[:12]) * E[token]
```

StateHead has no attention value tensor. Its closest recurrent analogue is the
candidate after the bounded candidate projection and before the write gate:

```text
candidate = tanh(c) + 3 * sigmoid(W_ve * norm(x)[:12]) * E[token]
u = sigmoid(b) * candidate
state_t = sigmoid(a) * state_(t-1) + u
```

This preserves GPT's vocabulary-sized, layer-specific embeddings, 12-channel
per-head gate, gate range, initialization, BF16 embedding storage, and AdamW
hyperparameters. Whether this candidate injection recovers GPT-like capacity
is an experimental hypothesis, not an established equivalence.

The native CUDA path has a separate value-residual custom op. The original
no-value CUDA op is unchanged. In the value backward kernel, recurrence replay
uses the combined candidate, the raw candidate derivative uses
`combined_candidate - value_residual`, and the value-residual gradient is the
write-gated state gradient.

## Parameter accounting

At vocab size 32,768:

| Group | Parameters |
|---|---:|
| Existing d32 StateHead total | 805,634,138 |
| 16 value-embedding banks (`32768 x 2048`) | 1,073,741,824 |
| 16 tiny value gates (`16 x 12`) | 3,072 |
| New raw total | **1,879,379,034** |
| Scaling parameters (`transformer_matrices + lm_head`) | **738,200,576** |

For context, GPT d24 Run 6 has 729,810,624 scaling parameters and
1,384,122,122 raw parameters. This follow-up keeps the d32 recurrent model; it
is therefore scaling-parameter matched but deliberately exceeds GPT's raw
parameter count.

## Fixed run and checkpoint curve

The production runner is
`runs/statehead_d32_ve_fp8_99m_curve.sh`, pinned to model-code commit
`0e8e426c33d46ef28a4da1362ec5c7a5a181621a`.

It performs:

1. a four-step compiled FP8/DDP production-shape CUDA and finite-gradient gate;
2. a fresh 125-step dataset calibration/learning gate;
3. a fresh seed-1337 run calibrated to 5,940 measured trainer seconds;
4. model-only checkpoints calibrated to 10, 20, ..., 90 minutes;
5. a final checkpoint at 99 minutes with optimizer state;
6. full CORE (`max_per_task=-1`) and 40 Mi-token train/validation BPB at all
   ten checkpoints;
7. a JSON and CSV curve keyed by actual measured trainer time.

The checkpoints are fixed before the full run from calibration throughput.
They are not selected from validation or CORE results. Checkpoint serialization
time is outside nanochat's measured trainer clock. Intermediate optimizer state
is intentionally omitted; the final checkpoint is exactly resumable if the
curve is still improving.

## Required paid gate

Do not start the 99-minute run unless all of these pass on the intended single
8x H100 80GB HBM3 node:

- native extension compilation;
- CUDA value-residual forward numerical parity;
- CUDA gradients for gates, initial state, and value residual;
- compiled full-model CUDA loss and all-parameter gradient parity versus the
  PyTorch scan;
- four compiled FP8/DDP steps with finite loss and every populated gradient
  finite;
- cross-rank parameter checksum spread exactly zero;
- peak memory leaves safe headroom at device batch 8;
- dataset calibration train loss and validation BPB both decrease;
- at least 100 GiB free before calibration.

Local tests do not satisfy the CUDA gate. As of preparation, the reference,
checkpoint, optimizer, meta-device, eager/compiled PyTorch, and six-step host
MPS smoke tests pass. CUDA compilation/parity and FP8 stability remain
unverified until the paid gate runs.

## Time, storage, and spend planning

Measured evidence from the preceding d32 run:

- training command: about 100.3 wall minutes;
- one full CORE+BPP evaluation: about 4.3 minutes;
- final model checkpoint: about 2.9 GiB without value embeddings.

Planning estimates for this run:

- training: about 100 minutes;
- ten full evaluations: about 43 minutes;
- setup, calibration, ten approximately 5.2 GiB model writes, hashing, and
  artifact transfer: roughly 35-65 minutes;
- remote checkpoint/result storage: roughly 65-80 GiB;
- safe instance-to-destruction guard: four hours, including artifact retrieval.

These are planning estimates, not measurements of the new value-embedding
model. The exact Vast offer and four-hour ceiling require fresh user approval
before provisioning.

## Interpretation

The primary plot is CORE and validation BPB versus actual measured trainer
minutes. The decision after the run is:

- if both curves are still improving materially at 99 minutes, continue from
  the final optimizer checkpoint under a separately approved budget;
- if CORE is noisy but validation BPB continues to improve, use the full curve
  rather than a single checkpoint to judge continuation;
- if both flatten well before 99 minutes, more steps alone are not the next
  experiment;
- do not claim GPT parity unless the fixed full evaluations demonstrate it.
