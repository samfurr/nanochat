# StateHead d32 value-embedding single-H100 CUDA gate

## Outcome

The bounded single-H100 gate passed and was cleaned up.

| Check | Result |
|---|---|
| Native extension compilation | passed |
| FP32/BF16 value-residual forward parity | passed |
| Gate, initial-state, and value-residual gradient parity | passed |
| Compiled full-model parity against PyTorch scan | passed, value embeddings off and on |
| Focused CUDA suite | 9 passed, 95 deselected |
| Production d32/2048 compiled FP8 execution | passed |
| Production-shape populated gradients | all finite |
| FP8 eligible linears | 65 |
| Parameter checksum spread | 0.0 |
| Peak allocated/reserved VRAM | 29,656,220,160 / 34,441,527,296 bytes |
| Final Vast invoice | $0.405, below the approved $1.00 ceiling |
| Post-cleanup resources | zero instances, zero volumes |

This is a CUDA correctness and single-rank FP8 feasibility result. It is not
the eight-rank DDP/FP8 gate, training-quality evidence, or GPT parity.

## Fixed implementation

- Checkout:
  `b5abb8aa841ec8525bcba9651697d6a27b32fe11`
- Model-code pin:
  `0e8e426c33d46ef28a4da1362ec5c7a5a181621a`
- PyTorch: `2.9.1+cu128`
- CUDA runtime: `12.8`
- GPU: NVIDIA H100 80GB HBM3
- StateHead: depth 32, width 2,048, 16 heads, sequence length 2,048
- Value embeddings: 16 GPT-rule banks
- Scan: native CUDA, chunk size 32
- Precision probe: BF16 activations, tensorwise FP8 eligible linears,
  FP32 recurrent accumulation

The production shape contained:

- total parameters: `1,879,379,034`
- scaling parameters: `738,200,576`
- value-embedding parameters: `1,073,741,824`
- estimated FLOPs/token: `4,429,793,424`

## Numerical and gradient parity

The focused test invocation exercised:

- value-residual CUDA forward and backward in FP32 and BF16;
- sequence lengths 33, 65, and 2,048;
- chunk sizes 16 and 32;
- gradients for raw gates, learned initial state, and candidate residual;
- compiled full-model logits, loss, and every parameter gradient against the
  PyTorch scan, both with value embeddings disabled and enabled.

Result:

```text
9 passed, 95 deselected in 138.32s
```

The first invocation failed before compilation because the locked environment
did not contain `setuptools`, which PyTorch's extension loader imports. The
known production-image repair, `setuptools==83.0.0`, was installed in the
disposable remote environment. The repository was not modified.

## Production-shape FP8 probe

The single-rank probe used device batch 1 for two compiled optimizer steps,
excluding the first compile step from the steady-state record:

| Step | Loss | Seconds | Tokens/s |
|---:|---:|---:|---:|
| 0 | 10.399179 | 132.388 | 15 |
| 1 | 3.131690 | 0.1502 | 13,634 |

All populated gradients were finite. Peak reserved VRAM was 34.44 GB, leaving
substantial room on the 80 GB device at batch 1. This does not establish that
device batch 8 fits or performs correctly under eight-rank optimizer sharding;
the full runner performs that gate before dataset calibration.

## Infrastructure and cost

- Vast offer: `41819343`
- Instance: `45761418`
- Observed all-in hourly rate: `$2.2888888889`
- Independent hard-destruction guard: 20 minutes
- GPU charge: `$0.400`
- Disk charge: `$0.004`
- Bandwidth charge: `$0.001`
- Total: `$0.405`

The evidence files were downloaded and hashed before destruction. Vast then
reported zero instances and zero volumes, and the independent guard was
cancelled.

## Evidence hashes

```text
d2dd348138e2d374a5cb310021d53c552f07d44b2b43e89608593f9e21c6ff44  statehead-d32-ve-fp8-single-h100.json
e5ed2dee1a7bc2a60c5fbc5e109f4ed754634f8d16bcc42dfd4b706f60371dd6  statehead-d32-ve-fp8-single-h100.log
cca7b6786365155044707abb81f96898e49c57c5c1a620c86031fce1644ee76d  cuda-value-parity.log
```

## Next gate

On an exact single-node 8x H100 SXM instance:

1. repeat the compiled production-shape probe with eight ranks and device
   batch 8;
2. require finite gradients, checksum spread zero, and safe peak VRAM;
3. run the fresh 125-step dataset calibration;
4. only then start the fixed 99-minute run and ten-checkpoint full evaluation.
