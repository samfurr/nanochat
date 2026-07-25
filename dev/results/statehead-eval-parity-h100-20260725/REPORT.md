# StateHead d32 H100 evaluation parity audit

## Outcome

The production CORE batching path is row- and candidate-independent for the
tested trained checkpoint, but reduced-precision recurrence is numerically
sensitive at d32 depth.

FP32 establishes that the native CUDA scan is algorithmically consistent with
the PyTorch parallel scan and sequential oracle. BF16 does not preserve strict
full-sequence versus token-by-token or CUDA versus sequential parity at this
trained depth. FP8 evaluation is not acceptable for reporting benchmark
scores.

This is a token-level synthetic-candidate audit of the actual trained
checkpoint, not a re-evaluation of the full CORE benchmark. It therefore does
not claim that the reported CORE aggregate is invariant to evaluation dtype or
scan implementation.

## Provenance

- checkpoint step: `4918`
- checkpoint SHA-256:
  `dd85cc2ae08ef852441c61383c71d8e1b2a7b41ed8c85dcc377f66ee6bc78292`
- code commit: `133fb666b9a542aa25395c669233328a3972fa82`
- GPU: NVIDIA H100 80GB HBM3, SM90
- PyTorch: `2.9.1+cu128`
- CUDA runtime reported by PyTorch: `12.8`

The checkpoint is the final artifact from the 99-minute d32 value-embedding
run. Its configuration is 32 layers, width 2,048, 16 recurrent heads, 2,048
tokens, CUDA scan chunk size 32, and value embeddings enabled.

## Persistent test results

```text
tests/test_core_eval_statehead.py
4 passed in 3.36s

tests/test_statehead.py -k \
  "native_cuda_scan_compiles_fullgraph or \
   native_cuda_value_forward_and_gradients_match_reference or \
   compiled_native_cuda_full_model_loss_and_gradients_match_pytorch"
9 passed, 95 deselected in 7.12s
```

The selected CUDA tests cover full-graph compilation, native value-residual
forward and gradients in FP32/BF16, and compiled full-model loss/gradient
comparison with value embeddings both disabled and enabled.

## Trained-checkpoint BF16 results

### Evaluator invariants

- Batched padded candidate scores versus individually scored CUDA candidates:
  same choice, maximum average-loss difference `9.54e-7`.
- Candidate-order permutation: exact score equality and same choice.
- The evaluator feeds the complete prompt-plus-answer sequence and slices only
  the answer-token losses. Right padding occurs after the scored tokens, so it
  cannot causally change them.
- Every evaluator forward starts without an externally carried recurrent
  state. The order-permutation and batched-versus-individual checks confirm no
  row or candidate leakage in the tested path.

### Scan sensitivity

For 64 independently generated four-choice sets:

| Comparison | Choice flips | Mean absolute average-loss difference | Maximum |
|---|---:|---:|---:|
| CUDA production vs PyTorch parallel | 0 / 64 | 0.024943 | 0.154080 |
| CUDA production vs sequential | 1 / 64 | 0.025103 | 0.127329 |
| PyTorch parallel vs sequential | 1 / 64 | 0.022278 | 0.158552 |

On two 65-token rows, CUDA versus PyTorch parallel matched 126/130 token
argmaxes; CUDA versus sequential matched 125/130. Maximum logit differences
were `0.605055` and `0.557661`, respectively.

For one 16-token recurrent decode, a full CUDA sequence versus token-at-a-time
CUDA state carry matched 15/16 token argmaxes. The returned CUDA state was
BF16. A sequential scan with explicitly FP32 carried state matched full versus
token-at-a-time exactly: 16/16 argmaxes, zero measured logit and state
difference in this BF16-compute run.

### FP8 evaluation sensitivity

Converting eligible projection linears to the training FP8 recipe for
evaluation produced:

- 256 four-choice sets / 1,024 candidate scores
- 20 / 256 prediction flips (`7.8125%`) versus BF16
- mean absolute average-loss difference `0.162278`
- maximum absolute average-loss difference `1.364250`

FP8 evaluation must not be used for the reported CORE score.

## Trained-checkpoint FP32 results

FP32 removed the material scan disagreement:

- CUDA versus PyTorch parallel: 130/130 token argmaxes, maximum logit
  difference `2.93e-5`.
- CUDA versus sequential: 130/130 token argmaxes, maximum logit difference
  `2.86e-5`.
- 64 four-choice sets: zero choice flips for all three pairwise
  implementation comparisons.
- Largest average-loss difference across implementations: `9.54e-6`.
- Full CUDA sequence versus token-at-a-time CUDA: 16/16 token argmaxes,
  maximum logit difference `3.86e-5`.

This supports an FP32 roundoff explanation for the implementation differences,
not an algorithmic error in the native CUDA recurrence.

## Interpretation and next gate

The reported full-sequence BF16 CUDA CORE score is defined by that exact
evaluation recipe, and the evaluator's candidate batching/order behavior is
correct in this audit.
However, the score should not be described as numerically invariant to the
sequential recurrence. The trained d32 model has close enough candidate margins
for BF16 scan order to change some choices.

The repository brief requires scan accumulation in FP32 around reduced-
precision activations. The CUDA kernel does use FP32 internal recurrence and
FP32 chunk summaries, but activated gates, outputs, and returned state remain
in the surrounding dtype. The returned BF16 boundary state is sufficient to
break segmented/token-at-a-time parity.

Before using recurrent decoding as a correctness claim, preserve carried state
in FP32 across calls and rerun this gate. Before making a dtype-independent
CORE claim, run a bounded set of real CORE examples under BF16 CUDA, BF16
PyTorch parallel, and FP32 CUDA and compare per-option margins and choices.

## Evidence files

- `statehead-eval-parity-h100-bf16-final.json`
- `statehead-eval-parity-h100-bf16-final.log`
- `statehead-eval-parity-h100-fp32-final.json`
- `statehead-eval-parity-h100-fp32-final.log`
- `statehead-eval-core-tests-h100.log`
- `statehead-eval-cuda-tests-h100.log`
- `statehead-eval-parity-h100-provenance.txt`

## Infrastructure

- Vast instance: `45807305`
- quoted all-in rate: `$1.7444444444/hour`
- observed account-credit delta: `$1.16636`
- post-cleanup active instances: `0`
- post-cleanup volumes: `0`
