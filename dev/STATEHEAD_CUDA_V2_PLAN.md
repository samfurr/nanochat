# StateHead CUDA v2 optimization plan

Status: v2 hierarchical backward verified; optimization continues

This document is the durable plan for optimizing StateHead after the Phase 3
comparison. It complements `STATEHEAD_NANOCHAT_CODEX_BRIEF.md`; the checked-out
code and measured artifacts remain the source of truth.

## Current implementation checkpoint

Base commit: `9fc014568537bf8f59e5e44df086fb213ed4b66a`

Implemented locally:

- Replaced the single serial all-chunk backward kernel with reverse chunk
  summaries, a reverse chunk-boundary scan, and parallel per-chunk gradient
  replay.
- Mapped backward work as one CUDA block per `(batch, head, chunk)`, with
  feature dimension contiguous across the block.
- Kept the existing forward, gate-major layout, checkpoint representation,
  Python API, and Transformer baseline unchanged.
- Expanded CUDA forward/gradient tests across chunk sizes 1, 16, 32, and 64,
  exact and partial chunks, sequence length 2048, FP32, and BF16.
- Added a device-independent test of the reverse affine-summary algebra.

Local evidence:

- StateHead/reference suite excluding CUDA-only cases: 51 passed.
- Three-step MPS StateHead optimizer smoke: finite losses and gradients.
- Repository-wide suite: 96 passed, 50 CUDA skips, and one unrelated failure in
  the sandbox memory-limit enforcement test.

One-H100 verification at commit
`52dd72e184bf1459009e9d99a4819a2634f872aa` subsequently established:

- CUDA 12.8/SM90 compilation and load passed.
- FP32 suite: 89 passed.
- Focused BF16 native-CUDA/reverse-summary gate: 45 passed.
- At the d12/768/2048/batch-32 full-step shape, CUDA v2 chunk 32 reached
  824,653 tok/s versus 772,758 for chunk 64, 615,599 for compiled PyTorch
  StateHead, and 530,591 for GPT/FA3 on the same H100.
- CUDA v2 chunk 32 peak allocation was 13.15 GiB versus 36.42 GiB for compiled
  PyTorch StateHead.

These are bounded fixed-synthetic-batch verification results. Dataset learning
parity, FP8 stability, chunk 16, and Nsight profiling remain untested. The full
evidence and command transcript are in
`dev/results/statehead-cuda-v2-gate-20260723/REPORT.md`.

## Motivation

At d12, the completed eight-H100 runs showed that StateHead used substantially
fewer modeled FLOPs than GPT but delivered only a modest full-step throughput
advantage. FLOPs are therefore not a sufficient proxy for performance. The
optimization target is end-to-end training-step throughput while preserving
forward, gradient, learning, and training-stability behavior.

The first native CUDA implementation reduced peak memory substantially, but its
backward kernel is only storage-chunked. One thread owns one state element and
walks backward through every chunk serially, replaying each chunk before
calculating its gradients. With sequence length 2048, chunk size 64 bounds
shared-memory usage but leaves a 2048-step dependency chain in every thread.

## Non-negotiable constraints

- Keep the large gate and output projections in optimized GEMMs.
- Keep recurrent state transitions and chunk summaries in FP32.
- Do not change the Transformer baseline.
- Do not infer speed from FLOPs or isolated forward timings.
- Verify forward and gradient parity after every numerical or layout change.
- Require a short dataset learning-parity run before another full training run.
- Treat FP8 as a later GEMM optimization; do not lower recurrence precision.

## Target architecture

### Forward

1. Gate projection GEMM.
2. Parallel affine summary for every `(batch, head, chunk, dim)`.
3. Small forward scan over chunk summaries for every state element.
4. Parallel local replay/output for every chunk.
5. Output projection GEMM with residual epilogue where practical.

The initial v2 implementation keeps the existing three-pass forward. A later
persistent or decoupled-lookback kernel may combine steps 2-4 so that each gate
tile is loaded and activated once.

### Backward

Let `q_t = grad_y_t * o_t`. If `g_t` is the gradient carry arriving at state
`s_t` from later timesteps, the reverse recurrence is

```text
ds_t      = g_t + q_t
g_(t - 1) = a_t * ds_t
```

Each chunk is consequently an affine map from its carry-out to its carry-in:

```text
g_before = A_chunk * g_after + U_chunk
```

The replacement backward has three kernels:

1. `backward_chunk_summary`: process all chunks in parallel. Traverse only the
   local chunk in reverse and produce FP32 `A_chunk` and `U_chunk`.
2. `backward_chunk_boundary`: traverse the small number of chunk summaries in
   reverse per state element. Save the FP32 carry arriving after each chunk and
   produce the initial-state gradient.
3. `backward_chunk_grad`: process all chunks in parallel. Replay the local
   forward states from the saved FP32 chunk initial state, start from that
   chunk's saved reverse carry, traverse only the local chunk backward, and
   emit gate gradients.

This reduces the long per-thread backward dependency path from `T` steps to
`K` steps plus a small `T/K` boundary scan, while exposing
`batch * heads * chunks` blocks of useful work.

## Memory layout

### v2.0: isolate hierarchical backward

Retain existing layouts so the algorithmic change is independently measurable:

- gates: `[B, T, 4, H, Dh]`, input dtype
- output and `grad_output`: `[B, T, H, Dh]`, input dtype
- chunk summaries: `[B, n_chunks, H, Dh]`, FP32
- forward chunk initial states: `[B, n_chunks, H, Dh]`, FP32
- reverse chunk carries: `[B, n_chunks, H, Dh]`, FP32

Use `dim` as the contiguous lane. Initially retain 128-thread backward blocks
and sweep chunk sizes 16, 32, and 64. Chunk size 32 is the expected first H100
candidate because it halves the local dependency chain and shared-memory tile
relative to 64 while keeping workspace modest.

### v2.1: interleaved gates

After v2.0 parity and profiling, benchmark physical gate layout
`[B, T, H, Dh, 4]`. Interleave projection output rows so `(a, b, c, o)` for a
state element are adjacent. This preserves a `[BT, 4D]` projection GEMM while
allowing one vector gate load and one vector gradient store per lane.

This layout change requires:

- an explicit weight/bias permutation helper;
- checkpoint layout versioning or load-time conversion;
- reference-path and native-path parity tests;
- an isolated benchmark against the existing gate-major layout.

Do not silently reinterpret an existing checkpoint.

## Staged implementation

### Stage A: measurement baseline

- Capture Nsight Systems and/or PyTorch profiler traces for compiled PyTorch and
  native CUDA BF16 on the same H100.
- Record scan forward, scan backward, GEMMs, loss, optimizer, and communication.
- Preserve full-step tokens/second and peak-memory measurements.

### Stage B: hierarchical backward

- Implement the three backward kernels above with the existing gate layout.
- Add coverage for partial chunks and multiple chunk sizes.
- Verify FP32 and BF16 forward/gradient parity.
- Verify compiled full-model loss and parameter-gradient parity.
- Benchmark isolated scan backward and full optimizer steps.

### Stage C: launch and tile tuning

- Sweep chunk sizes 16, 32, and 64.
- Sweep block geometry only after parity passes.
- Measure occupancy, achieved bandwidth, register pressure, and shared memory.
- Select using median full-step throughput, not microkernel timing alone.

### Stage D: gate traffic and epilogues

- Benchmark interleaved gate layout and vectorized loads/stores.
- Fold gate bias into the projection GEMM epilogue.
- Fold output residual into the output projection epilogue where the compiler or
  GEMM library supports it.
- Confirm that layout conversion does not add a runtime transpose/materialization.

### Stage E: forward fusion

If profiling still shows forward gate rereads or activation recomputation as a
material bottleneck, prototype a persistent chunk kernel with decoupled
lookback:

- load and activate a gate tile once;
- compute its local affine summary;
- obtain the chunk prefix safely;
- replay from shared state and emit output.

Keep the three-pass forward as the correctness reference and fallback.

### Stage F: FP8 GEMMs

- Apply FP8 only to eligible gate/output GEMMs.
- Use native fused scaling/epilogue support rather than Python-side wrappers.
- Keep recurrence, summaries, and boundary state FP32.
- Repeat numerical, gradient, fixed-batch stability, learning-parity, and
  full-step throughput gates.

## Paths explicitly not selected

- Full-sequence state materialization in FP32: excessive HBM traffic and memory.
- Global associative-doubling scan: too many whole-sequence passes.
- Warp lanes assigned to time with feature-contiguous GEMM output: strided loads.
- Hand-written replacements for the large projection GEMMs.
- Reconstructing prior state with `(s_t - u_t) / a_t` as the primary training
  path: poorly conditioned when the retention gate is small.
- A complex decoupled-lookback forward before the simpler hierarchical backward
  has established its value.

## Verification gates

Each stage must record:

1. CUDA extension compilation.
2. FP32 forward parity, including partial chunks.
3. FP32 raw-gate and initial-state gradient parity.
4. BF16 forward and gradient parity with documented tolerances.
5. `torch.compile(fullgraph=True)` compatibility.
6. Compiled full-model loss and parameter-gradient parity.
7. Finite short training smoke test.
8. Dataset learning parity before a full-scale run.
9. Isolated scan forward/backward timings.
10. Median full training-step tokens/second and peak allocated/reserved memory.

Parity may only be claimed for gates that actually passed. Performance claims
must name the hardware, shape, precision, software revision, warmup, sample
count, and whether the number is scan-only or full-step.

## Immediate implementation slice

The first code change is Stage B only: replace the serial all-chunk backward
kernel with parallel chunk summaries, a reverse chunk-boundary scan, and
parallel local chunk-gradient kernels. Keep the forward, public Python API,
gate layout, checkpoint layout, and Transformer baseline unchanged.

## 2026-07-23 CUDA v4 execution update

The bounded one-H100 optimization pass completed the first useful part of
Stage D and refined Stage E:

- CUDA v3 caches sigmoid/tanh gate activations once instead of recalculating
  them in every recurrence kernel.
- CUDA v4 folds gate bias into that activation pass, eliminating the external
  compiled pointwise bias-add materialization.
- CUDA v4 reached 938,891 median full-step tok/s at
  d12/768/T2048/batch-32 BF16, 13.06% above same-host CUDA v2 chunk 32,
  54.40% above compiled PyTorch StateHead, and 80.78% above GPT/FA3.
- Peak allocation remained 13.15 GiB.
- FP32 and BF16 CUDA numerical/gradient suites, partial chunks, sequence 2048,
  fullgraph compilation, compiled full-model parity, and a final finite-gradient
  smoke passed.
- Dataset learning parity, FP8, and eight-GPU DDP remain open.

Two probes were measured and explicitly reverted. Removing flat indexing was
neutral at -0.02%. Fusing activation into forward summary generation was 1.80%
slower because it reduced activation parallelism and still needed the cache for
other passes.

Nsight Systems shows the standalone activation pass is now the largest
isolated opportunity at 5.675 ms/step. The next separately revertible probe is
a tensor-core gate projection with a custom mixed sigmoid/tanh output epilogue
that emits the existing interleaved gate layout. Keep v4 as the correctness
fallback and recurrence state FP32. If the full activation pass disappeared,
the theoretical step would be about 64.13 ms, or 1.97x the measured GPT
control; another roughly 1 ms would still be needed to exceed 2x GPT.

Do not retry the rejected forward-summary fusion unchanged. Do not advance
FP8 or a full training run until the projection/activation probe passes the
existing parity gates and a short dataset-backed learning-parity comparison.

Full report, profiles, tests, rejected probes, cost, commands, and checksums:

```text
dev/results/statehead-cuda-v4-optimization-20260723/REPORT.md
```

## CUDA v7 projection/activation pipeline probe

The next reversible probe keeps CUDA v4 available as `scan_backend=cuda` and
adds an explicit `scan_backend=cuda_projected`. It does not replace the large
gate GEMM with a handwritten recurrence kernel.

For the flattened gate projection
`[B*T,D] @ [D,4D] -> [B*T,4D]`, v7:

1. divides only the flattened token-row dimension into a small number of large
   tiles;
2. runs every tile through PyTorch's cuBLAS-backed BF16 GEMM on a projection
   stream;
3. runs the existing mixed sigmoid/tanh plus bias CUDA activation on a second
   stream;
4. double-buffers raw gate tiles so activation for tile `n` can overlap the
   tensor-core GEMM for tile `n+1`;
5. writes activated gates directly into the existing contiguous
   `[B,T,4,H,Dh]` layout;
6. invokes the verified CUDA v4 scan only after the activation stream has
   completed;
7. uses the existing scan backward to produce raw-logit gradients, then the
   same two projection-gradient GEMMs and bias reduction as the regular linear.

The initial production-shape tile size is 16,384 flattened rows, which yields
four tiles for batch 32 and sequence length 2,048. It is an explicit config and
preflight parameter so 8,192 and 32,768 can be measured without source edits.

Acceptance gates:

- CUDA 12.8 compilation and extension load;
- FP32 and BF16 projected full-model loss/parameter-gradient parity against
  the compiled PyTorch backend;
- partial chunks and sequence length 2,048;
- `torch.compile(fullgraph=True)` compatibility;
- finite-gradient short training;
- full-step throughput strictly above v4 on the same H100, with v4 rerun as
  the same-session control;
- no change to `scan_backend=cuda`, PyTorch StateHead, or GPT behavior.

Reject v7 if cuBLAS tiling loses more GEMM efficiency than the activation
overlap saves, if multistream execution prevents graph compilation/capture, or
if its memory/workspace overhead materially erodes v4's 13.15 GiB advantage.

### v7 measurement and v8 gate-parallel follow-up

On the same H100 and commit, the first short d12/768/T2048/batch-32 BF16
measurements were:

- CUDA v4 control: 948,838 median full-step tok/s;
- v7, four 16,384-row tiles: 934,513 tok/s, 1.51% below v4;
- v7, two 32,768-row tiles: 940,540 tok/s, 0.87% below v4.

The complete BF16 suite reached 90 passes with only the two known GPT/FA3 CPU
checkpoint failures; focused FP32 and BF16 projected/native CUDA parity each
passed 48 tests. Nsight confirms that v7 launches 72 activation tiles across
three measured optimizer steps, totaling 5.72 ms per step. Row tiling therefore
does not eliminate the activation work; it only tries to hide it, and the
smaller GEMMs plus stream/event overhead exceed the overlap benefit.

The next bounded v8 probe uses the gate-major projection layout already present
in the checkpoint. It splits the contiguous `[4D,D]` weight into four `[D,D]`
gate blocks, launches the four projections on independent cuBLAS streams, and
immediately applies each gate's sigmoid or tanh on that same stream while the
other projections remain in flight. Each activation writes directly to its
gate block in `[B,T,4,H,Dh]`; there is no runtime transpose or checkpoint
conversion. The CUDA v4 and row-tiled v7 backends remain separate controls.

Accept v8 only if it passes the same FP32/BF16 compiled full-model parity and
finite-gradient gates and beats the same-host v4 full-step control. If four
`[D,D]` GEMMs lose more tensor-core efficiency than concurrency recovers,
reject it and retain v4. A two-group `[2D,D]` split is the only justified
follow-up before moving to a CUTLASS/cuBLASLt custom mixed-activation epilogue.
