# StateHead research log

## 2026-07-22 — Phase 0 and Phase 1

### Checkout and environment

- Starting commit: `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`
- Pre-existing `git status --short`:

  ```text
  ?? .DS_Store
  ?? .agents/
  ?? STATEHEAD_NANOCHAT_CODEX_BRIEF.md
  ```

- The repository requests Python 3.10 in `.python-version`, but that pyenv version was not installed. The local environment was created with uv-managed CPython 3.12.8 and the CPU/MPS PyTorch 2.9.1 wheel.
- Host MPS check: `is_built=True`, `is_available=True`. The restricted sandbox reports MPS unavailable, so device validation was run with host permission.
- No CUDA device was used. No paid job was launched.

### Files audited before source edits

- `STATEHEAD_NANOCHAT_CODEX_BRIEF.md` (all 1,147 lines)
- `nanochat/gpt.py`
- `nanochat/dataloader.py`
- `nanochat/checkpoint_manager.py`
- `nanochat/engine.py`
- `nanochat/optim.py`
- `nanochat/dataset.py`
- `nanochat/tokenizer.py`
- `nanochat/common.py`
- `scripts/base_train.py`
- `scripts/base_eval.py`
- `scripts/tok_train.py`
- `runs/speedrun.sh`
- `runs/runcpu.sh`
- `dev/LEADERBOARD.md`
- all existing modules under `tests/`
- `pyproject.toml`

### Phase 0 baseline

Pre-edit tests:

```text
43 passed, 14 skipped, 1 failed in 3.42s
```

The failure was pre-existing and remains reproducible:

```text
tests/test_execution.py::test_memory_limit
Expected a 1 GiB allocation to be rejected by the 256 MiB child-process limit,
but it succeeded on this macOS host.
```

The disposable local smoke fixture lives at `/tmp/nanochat-statehead-smoke.W8Wrbv`. It contains two tiny synthetic parquet shards and a tiny tokenizer; it does not change the repository or the experiment dataset contract.

The first GPT smoke attempt used the not-yet-existing `--arch=gpt` argument and correctly failed argument parsing. A width-16 attempt then exposed the baseline's fixed 24-channel smear requirement. The smallest valid smoke was width 32:

```text
architecture: default GPT (no architecture flag)
device: CPU
precision: FP32
layers/width/heads: 1/32/1
sequence length: 8
steps/tokens: 5/40
parameters: 49,192
loss: 5.925983 -> 5.925332
result: finite, decreasing, checkpoint saved
```

### Phase 1 architecture decisions

- MLP-free block only: one RMS normalization, StateHead gate projection, recurrent scan, output projection, and residual update. No attention, MLP, RoPE, positional embeddings, value embeddings, CUDA, Triton, or FP8 were added.
- Gate matrix initialization uses nanochat's unscaled input-matrix uniform convention; output projection is zero initialized.
- Retention bias is `2.0`; write/candidate/output biases and learned initial state are zero initialized.
- The parallel reference uses functional associative doubling, 64-token chunks by default, identity padding, and FP32 recurrence accumulation for BF16/FP16 inputs.
- State is reset to the learned initial state on every ordinary model forward, so packed rows and gradient-accumulation microbatches do not carry recurrent state.
- Nanochat smear, residual lambdas, input reinjection, backout, logit softcap, and untied embeddings/head are preserved. Internal BOS does not hard-reset state.
- StateHead head count initially follows nanochat's `model_dim // head_dim` construction.
- Standard Muon is used for `gate.weight` and `out_proj.weight`. Gate bias and learned initial state use AdamW at the width-scaled unembedding learning rate. Exact optimizer coverage is asserted.
- Checkpoints write `model_type`; missing `model_type` remains legacy GPT. StateHead uses a separate default checkpoint tag, `statehead-d{depth}`.
- Fixed-size engine cache integration remains deferred. Phase 1 includes explicit-state prefill/decode correctness and a naive full-prefix generation fallback for evaluation samples.

### Phase 1 results

Focused correctness suite:

```text
39 passed in 0.99s
```

Coverage includes:

- sequential/parallel FP32 forward parity for batches 1 and 2 at lengths 1, 2, 3, 7, 63, 64, 65, 127, 128, 129, and 2,048;
- BF16 scan parity;
- scan input/initial-state gradient parity;
- gate weight, gate bias, output weight, initial state, and input-activation gradient parity;
- segmentation at 1, 63, 64, 65, and 128;
- batch independence and row reset;
- smear-aware full-prefill versus one-token recurrent decode;
- exact legacy GPT, tagged GPT, and StateHead checkpoint round trips;
- meta-device initialization;
- exact optimizer partition;
- tiny repeated-batch overfit.

Numerical tolerances are FP32 forward `rtol=2e-5, atol=2e-6`, FP32 scan gradient `rtol=2e-4, atol=2e-5`, full bank gradient `rtol=3e-4, atol=3e-5`, segmentation/decode `rtol=3e-5, atol=3e-6`, and BF16 forward `rtol=atol=2e-2`.

Tiny overfit result:

```text
40 AdamW steps on one repeated token batch
loss: 4.158207 -> 0.000409
ratio: 0.000098
all losses finite: true
```

MPS forward/backward:

```text
device: mps:0
precision: BF16 activations/embeddings, FP32 recurrence accumulation
loss: 4.157590
finite loss and all populated gradients: true
```

MPS training entry-point smoke:

```text
architecture: StateHead
layers/width/heads: 1/32/1
sequence length: 8
steps/tokens: 5/40
parameters: 29,884
  wte: 12,288
  lm_head: 12,288
  statehead_matrices: 5,120
  statehead_vectors: 160
  scalars: 28
analytical training FLOPs/token: 104,880
persistent BF16 state/row: 64 bytes
loss: 5.923017 -> 5.922367
result: finite, decreasing, compiled training and checkpoint save passed
checkpoint: /tmp/nanochat-statehead-smoke.W8Wrbv/base_checkpoints/phase1-statehead-mps
```

The saved StateHead checkpoint reloaded through `scripts.base_eval` on MPS and produced finite tiny-fixture BPB (`train=1.752195`, `val=1.752195`).

Post-edit default-GPT MPS smoke (no `--arch` flag):

```text
model config included n_kv_head and window_pattern, confirming GPT construction
parameters: 49,192
loss: 5.923016 -> 5.922373
result: finite, decreasing, checkpoint saved
```

Post-edit full tests:

```text
82 passed, 14 skipped, 1 failed in 3.41s
```

The only failure is the same pre-edit macOS memory-limit test. All other existing and new tests pass.

### Known limitations and unresolved questions

- Direct eager-versus-compiled numerical output/gradient comparison is Phase 2. Phase 1 only establishes that the compiled MPS training path runs with finite decreasing loss.
- `StateHeadCache` and the generic `Engine` integration are deferred; naive full-prefix sampling is available, and explicit-state prefill/decode parity is tested.
- CORE was not run or tuned.
- Multi-rank optimizer execution was not available locally. Before an 8-device run, confirm that AdamW sharding of a learned initial state larger than 1,024 elements is compatible with the optimizer's first-dimension divisibility requirement.
- The vector-parameter AdamW learning rate is a documented first-port choice, not a tuned result.
- Analytical recurrence FLOPs are an estimate and make no wall-clock performance claim.
- The baseline macOS memory-limit test needs a platform-specific follow-up outside StateHead Phase 0/1.

### Command ledger

Commands are listed in execution order. Inspection commands are included because the checkout, not the brief, was treated as authoritative.

```bash
pwd && git rev-parse HEAD && git status --short && rg --files -g 'AGENTS.md' -g 'STATEHEAD_NANOCHAT_CODEX_BRIEF.md' -g '!**/.git/**' && rg -n "nanochat|STATEHEAD_NANOCHAT|Phase 0|Phase 1" /Users/haybales/.codex/memories/MEMORY.md
wc -l -c STATEHEAD_NANOCHAT_CODEX_BRIEF.md && sed -n '1,260p' STATEHEAD_NANOCHAT_CODEX_BRIEF.md
sed -n '261,560p' STATEHEAD_NANOCHAT_CODEX_BRIEF.md
sed -n '561,860p' STATEHEAD_NANOCHAT_CODEX_BRIEF.md
sed -n '861,1147p' STATEHEAD_NANOCHAT_CODEX_BRIEF.md
python --version && uv --version && find . -maxdepth 2 -type f | sort && wc -l nanochat/gpt.py nanochat/dataloader.py nanochat/checkpoint_manager.py nanochat/engine.py nanochat/optim.py scripts/base_train.py scripts/base_eval.py runs/speedrun.sh dev/LEADERBOARD.md pyproject.toml && rg --files tests | sort
uv --version
find . -maxdepth 2 -type f | sort
wc -l nanochat/gpt.py nanochat/dataloader.py nanochat/checkpoint_manager.py nanochat/engine.py nanochat/optim.py scripts/base_train.py scripts/base_eval.py runs/speedrun.sh dev/LEADERBOARD.md pyproject.toml && rg --files tests | sort
sed -n '1,300p' nanochat/gpt.py && sed -n '301,620p' nanochat/gpt.py
sed -n '1,240p' nanochat/dataloader.py && sed -n '1,260p' nanochat/checkpoint_manager.py
sed -n '1,400p' nanochat/engine.py
sed -n '1,520p' nanochat/optim.py
sed -n '1,420p' nanochat/engine.py
sed -n '1,240p' nanochat/optim.py && sed -n '241,520p' nanochat/optim.py
sed -n '1,330p' scripts/base_train.py && sed -n '331,680p' scripts/base_train.py
sed -n '1,300p' scripts/base_eval.py
sed -n '1,140p' runs/speedrun.sh && sed -n '1,260p' dev/LEADERBOARD.md && sed -n '1,120p' pyproject.toml
wc -l tests/*.py && sed -n '1,260p' tests/test_engine.py && sed -n '1,260p' tests/test_optim.py && sed -n '1,260p' tests/test_attention_fallback.py
sed -n '241,340p' tests/test_engine.py && sed -n '261,430p' tests/test_attention_fallback.py
sed -n '1,180p' tests/test_execution.py && sed -n '1,180p' tests/test_tasks.py && sed -n '1,220p' tests/test_tokenizer.py
sed -n '1,20p' .python-version && ls -la .venv 2>/dev/null || true && pyenv versions
uv sync --python 3.12.8 --extra cpu --group dev
uv sync --python 3.12.8 --extra cpu --group dev
.venv/bin/python --version && git status --short
git diff --stat -- uv.lock && git diff -- uv.lock | sed -n '1,240p'
.venv/bin/python -m scripts.base_train --help
.venv/bin/python -m scripts.base_train --help
.venv/bin/python -m pytest -q
find /Users/haybales/.cache/nanochat -maxdepth 3 -type f | sort | sed -n '1,160p'
sed -n '1,260p' nanochat/dataset.py && sed -n '1,240p' nanochat/tokenizer.py
sed -n '1,220p' scripts/tok_train.py && sed -n '1,180p' runs/runcpu.sh && sed -n '1,180p' nanochat/common.py
mktemp -d /tmp/nanochat-statehead-smoke.XXXXXX
.venv/bin/python -c 'from pathlib import Path; import pyarrow as pa, pyarrow.parquet as pq, torch; from nanochat.tokenizer import RustBPETokenizer; root=Path("/tmp/nanochat-statehead-smoke.W8Wrbv"); data=root/"base_data_climbmix"; data.mkdir(); docs=["The quick brown fox jumps over the lazy dog. Nanochat local smoke training. "+str(i) for i in range(64)]; pq.write_table(pa.table({"text": docs}), data/"shard_00000.parquet", row_group_size=16); pq.write_table(pa.table({"text": docs[::-1]}), data/"shard_06542.parquet", row_group_size=16); tok=RustBPETokenizer.train_from_iterator(iter(docs*4), 512); tok_dir=root/"tokenizer"; tok.save(tok_dir); special=set(tok.encode_special(s) for s in tok.get_special_tokens()); token_bytes=torch.tensor([0 if i in special else len(tok.decode_single_token_bytes(i)) for i in range(tok.get_vocab_size())], dtype=torch.int32); torch.save(token_bytes, tok_dir/"token_bytes.pt"); print(root); print(tok.get_vocab_size())'
NANOCHAT_BASE_DIR=/tmp/nanochat-statehead-smoke.W8Wrbv NANOCHAT_DTYPE=float32 .venv/bin/python -m scripts.base_train --arch=gpt --depth=1 --aspect-ratio=16 --head-dim=16 --max-seq-len=8 --device-batch-size=1 --total-batch-size=8 --num-iterations=5 --eval-every=-1 --eval-tokens=16 --core-metric-every=-1 --sample-every=-1 --save-every=-1 --device-type=cpu --model-tag=phase0-gpt
NANOCHAT_BASE_DIR=/tmp/nanochat-statehead-smoke.W8Wrbv NANOCHAT_DTYPE=float32 .venv/bin/python -m scripts.base_train --depth=1 --aspect-ratio=16 --head-dim=16 --max-seq-len=8 --device-batch-size=1 --total-batch-size=8 --num-iterations=5 --eval-every=-1 --eval-tokens=16 --core-metric-every=-1 --sample-every=-1 --save-every=-1 --device-type=cpu --model-tag=phase0-gpt
NANOCHAT_BASE_DIR=/tmp/nanochat-statehead-smoke.W8Wrbv NANOCHAT_DTYPE=float32 .venv/bin/python -m scripts.base_train --depth=1 --aspect-ratio=32 --head-dim=32 --max-seq-len=8 --window-pattern=L --device-batch-size=1 --total-batch-size=8 --num-iterations=5 --eval-every=-1 --eval-tokens=16 --core-metric-every=-1 --sample-every=-1 --save-every=-1 --device-type=cpu --model-tag=phase0-gpt
NANOCHAT_DTYPE=float32 .venv/bin/python -c 'import torch; from nanochat.statehead import *; torch.manual_seed(0); a=torch.sigmoid(torch.randn(2,65,3,4)); u=torch.randn_like(a); o=torch.sigmoid(torch.randn_like(a)); s=torch.randn(2,3,4); ys,ss=statehead_scan_sequential(a,u,o,s); yp,sp=statehead_scan_parallel(a,u,o,s); print((ys-yp).abs().max().item(), (ss-sp).abs().max().item()); c=StateHeadConfig(sequence_len=8,vocab_size=64,n_layer=2,n_head=4,n_embd=32); m=StateHead(c); m.init_weights(); idx=torch.randint(0,64,(2,8)); loss=m(idx,idx); loss.backward(); print(loss.item(), m.num_scaling_params(), m.recurrent_state_bytes())'
NANOCHAT_DTYPE=float32 .venv/bin/python -m pytest tests/test_statehead.py -q
.venv/bin/python -c 'import torch; print(torch.__version__); print(torch.backends.mps.is_built(), torch.backends.mps.is_available())'
.venv/bin/python -c 'import torch; print(torch.__version__); print(torch.backends.mps.is_built(), torch.backends.mps.is_available())'
git diff --check && .venv/bin/python -m compileall -q nanochat scripts tests
git diff --stat && git status --short && git diff -- nanochat/gpt.py
NANOCHAT_DTYPE=float32 .venv/bin/python -m pytest -q
NANOCHAT_DTYPE=bfloat16 .venv/bin/python -c 'import torch; from nanochat.statehead import StateHead, StateHeadConfig; config=StateHeadConfig(sequence_len=8,vocab_size=64,n_layer=2,n_head=4,n_embd=32,scan_chunk_size=4); device=torch.device("mps");
with torch.device("meta"): model=StateHead(config)
model.to_empty(device=device); model.init_weights(); tokens=torch.randint(0,64,(2,8),device=device); targets=torch.roll(tokens,-1,1); loss=model(tokens,targets); loss.backward(); finite=bool(torch.isfinite(loss).item()) and all(bool(torch.isfinite(p.grad).all().item()) for p in model.parameters() if p.grad is not None); print(f"loss={loss.item():.6f} finite_loss_and_grads={finite} device={model.get_device()} dtype={model.transformer.wte.weight.dtype}")'
NANOCHAT_BASE_DIR=/tmp/nanochat-statehead-smoke.W8Wrbv NANOCHAT_DTYPE=bfloat16 .venv/bin/python -m scripts.base_train --arch=statehead --depth=1 --aspect-ratio=32 --head-dim=32 --max-seq-len=8 --device-batch-size=1 --total-batch-size=8 --num-iterations=5 --eval-every=-1 --eval-tokens=16 --core-metric-every=-1 --sample-every=-1 --save-every=-1 --device-type=mps --model-tag=phase1-statehead-mps
sed -n '1,120p' /tmp/nanochat-statehead-smoke.W8Wrbv/base_checkpoints/phase1-statehead-mps/meta_000005.json
NANOCHAT_BASE_DIR=/tmp/nanochat-statehead-smoke.W8Wrbv NANOCHAT_DTYPE=bfloat16 .venv/bin/python -m scripts.base_eval --eval=bpb --model-tag=phase1-statehead-mps --step=5 --device-batch-size=1 --split-tokens=16 --device-type=mps
NANOCHAT_BASE_DIR=/tmp/nanochat-statehead-smoke.W8Wrbv NANOCHAT_DTYPE=bfloat16 .venv/bin/python -m scripts.base_train --depth=1 --aspect-ratio=32 --head-dim=32 --max-seq-len=8 --window-pattern=L --device-batch-size=1 --total-batch-size=8 --num-iterations=5 --eval-every=-1 --eval-tokens=16 --core-metric-every=-1 --sample-every=-1 --save-every=-1 --device-type=mps --model-tag=phase1-default-gpt-mps
NANOCHAT_DTYPE=float32 .venv/bin/python -c 'import torch; from nanochat.statehead import StateHead, StateHeadConfig; torch.manual_seed(14); config=StateHeadConfig(sequence_len=8,vocab_size=64,n_layer=1,n_head=4,n_embd=32,scan_chunk_size=4); model=StateHead(config); model.init_weights(); inputs=torch.arange(8).repeat(4,1); targets=torch.roll(inputs,-1,1); optimizer=torch.optim.AdamW(model.parameters(),lr=0.03); losses=[];
for _ in range(40):
 optimizer.zero_grad(set_to_none=True); loss=model(inputs,targets); loss.backward(); optimizer.step(); losses.append(loss.item())
print(f"initial={losses[0]:.6f} final={losses[-1]:.6f} ratio={losses[-1]/losses[0]:.6f} finite={all(torch.isfinite(torch.tensor(losses)))}")'
git diff -- nanochat/checkpoint_manager.py scripts/base_train.py scripts/base_eval.py && sed -n '1,420p' nanochat/statehead.py && sed -n '1,420p' tests/test_statehead.py
```

Commands after this ledger was first written are appended below during final verification.

```bash
NANOCHAT_BASE_DIR=/tmp/nanochat-statehead-smoke.W8Wrbv .venv/bin/python -m scripts.base_train --help
NANOCHAT_DTYPE=float32 .venv/bin/python -m pytest tests/test_statehead.py -q
NANOCHAT_DTYPE=float32 .venv/bin/python -m pytest -q -k 'not test_memory_limit'
git diff --check && .venv/bin/python -m compileall -q nanochat scripts tests && git diff --quiet -- nanochat/gpt.py && git rev-parse HEAD && git status --short && git diff --numstat && wc -l nanochat/statehead.py tests/test_statehead.py dev/STATEHEAD_LOG.md
git diff --check && git status --short
```

Final focused result: `39 passed in 1.27s`. Final suite excluding the documented pre-existing platform failure: `82 passed, 14 skipped, 1 deselected in 3.68s`.

## 2026-07-22 — Phase 2 local behavior and compile checks

### Changes

- Added optional `--seed`; its default `-1` preserves the existing unseeded training behavior.
- MPS step timing now synchronizes before and after each measured training step.
- MPS peak allocated memory is sampled after synchronized steps and reported by the existing peak-memory line.
- Added portable full-model eager/`torch.compile` output, loss, and parameter-gradient comparison using the `aot_eager` backend.
- Added two-run, bitwise fixed-seed CPU loss-curve repeatability coverage.
- Added a direct test for StateHead's naive generation fallback.

### Correctness results

Focused suite after Phase 2 additions:

```text
42 passed in 2.15s
```

Full suite excluding the documented pre-existing macOS memory-limit failure:

```text
85 passed, 14 skipped, 1 deselected in 4.44s
```

Direct default-backend compile comparison on host MPS with BF16 activations and FP32 recurrence accumulation:

```text
max absolute logits difference:    0.00014495861
absolute loss difference:          0.00001335144
max absolute parameter-grad diff:  0.001953125
eager loss:                        4.15477228
compiled loss:                     4.15475893
```

The portable CPU compile test uses `rtol=1e-5, atol=1e-6` for logits/loss and `rtol=2e-4, atol=2e-5` for gradients. The direct MPS values above are recorded as BF16 evidence rather than forced through the tighter FP32 tolerances.

Existing Phase 1 tests continue to establish:

- full-row scan versus segmented and one-token recurrent processing;
- smear-aware prefill/decode equivalence;
- exact row-reset determinism;
- scan and full-bank forward/gradient parity.

### Fixed-seed MPS training repetitions

Both runs used:

```text
seed: 1337
architecture: StateHead
device/precision: MPS/BF16
layers/width/heads: 1/32/1
sequence length: 8
steps/tokens: 20/160
warmup steps: 5
warmdown ratio: 0.2
parameters: 29,884
synthetic local fixture: /tmp/nanochat-statehead-smoke.W8Wrbv
```

The logged debiased-smoothed loss curve was identical in both runs:

```text
5.924298, 5.923702, 5.922661, 5.921167, 5.919217,
5.916775, 5.913998, 5.910952, 5.907698, 5.904271,
5.900675, 5.896925, 5.893023, 5.888976, 5.884794,
5.880474, 5.876019, 5.871433, 5.866886, 5.862553
```

The final checkpoints contain 11 tensors and are bitwise identical (`max_abs=0.0`). All losses were finite.

Measured diagnostics after the first ten steps:

| Run | median tok/s | mean tok/s | peak MPS allocation |
|---|---:|---:|---:|
| seed1337-a | 3,496 | 3,535.8 | 0.31 MiB |
| seed1337-b | 3,239 | 3,315.3 | 0.31 MiB |

These are tiny-sequence, tiny-model local diagnostics. They are not evidence of speed relative to GPT or of large-model throughput.

Checkpoints:

- `/tmp/nanochat-statehead-smoke.W8Wrbv/base_checkpoints/phase2-statehead-mps-seed1337-a`
- `/tmp/nanochat-statehead-smoke.W8Wrbv/base_checkpoints/phase2-statehead-mps-seed1337-b`

### Phase 2 command ledger

```bash
git rev-parse HEAD && git status --short && rg -n "device-type|synchronize =|get_max_memory|seed|Peak memory|single training step|t1 =" scripts/base_train.py && tail -80 tests/test_statehead.py && tail -60 dev/STATEHEAD_LOG.md
NANOCHAT_DTYPE=float32 .venv/bin/python -m pytest tests/test_statehead.py -q
NANOCHAT_DTYPE=bfloat16 .venv/bin/python -c 'import copy, torch; from nanochat.statehead import StateHead, StateHeadConfig; torch.manual_seed(1337); device=torch.device("mps"); config=StateHeadConfig(sequence_len=8,vocab_size=64,n_layer=2,n_head=4,n_embd=32,scan_chunk_size=4); eager=StateHead(config); eager.init_weights(); eager.to(device); eager.smear_lambda.data.fill_(0.4); [block.state_bank.out_proj.weight.data.normal_(std=0.05) for block in eager.transformer.h]; compiled_base=copy.deepcopy(eager); compiled=torch.compile(compiled_base,dynamic=False); inputs=torch.randint(0,64,(2,8),device=device); targets=torch.roll(inputs,-1,1); eager_logits=eager(inputs); compiled_logits=compiled(inputs); output_diff=(compiled_logits-eager_logits).abs().max().item(); eager_loss=eager(inputs,targets); compiled_loss=compiled(inputs,targets); eager_loss.backward(); compiled_loss.backward(); grad_diffs=[];
for (name_a,param_a),(name_b,param_b) in zip(eager.named_parameters(),compiled_base.named_parameters()):
 assert name_a==name_b and param_a.grad is not None and param_b.grad is not None; grad_diffs.append((param_b.grad.float()-param_a.grad.float()).abs().max().item())
torch.mps.synchronize(); print(f"output_max_abs={output_diff:.8g} loss_abs={abs(compiled_loss.item()-eager_loss.item()):.8g} grad_max_abs={max(grad_diffs):.8g} eager_loss={eager_loss.item():.8f} compiled_loss={compiled_loss.item():.8f}")'
NANOCHAT_DTYPE=float32 .venv/bin/python -m pytest -q -k 'not test_memory_limit'
git diff --check && .venv/bin/python -m compileall -q nanochat scripts tests && git diff --quiet -- nanochat/gpt.py
NANOCHAT_BASE_DIR=/tmp/nanochat-statehead-smoke.W8Wrbv NANOCHAT_DTYPE=bfloat16 .venv/bin/python -m scripts.base_train --arch=statehead --seed=1337 --depth=1 --aspect-ratio=32 --head-dim=32 --max-seq-len=8 --device-batch-size=1 --total-batch-size=8 --num-iterations=20 --warmup-steps=5 --warmdown-ratio=0.2 --eval-every=-1 --eval-tokens=16 --core-metric-every=-1 --sample-every=-1 --save-every=-1 --device-type=mps --model-tag=phase2-statehead-mps-seed1337-a
NANOCHAT_BASE_DIR=/tmp/nanochat-statehead-smoke.W8Wrbv NANOCHAT_DTYPE=bfloat16 .venv/bin/python -m scripts.base_train --arch=statehead --seed=1337 --depth=1 --aspect-ratio=32 --head-dim=32 --max-seq-len=8 --device-batch-size=1 --total-batch-size=8 --num-iterations=20 --warmup-steps=5 --warmdown-ratio=0.2 --eval-every=-1 --eval-tokens=16 --core-metric-every=-1 --sample-every=-1 --save-every=-1 --device-type=mps --model-tag=phase2-statehead-mps-seed1337-b
.venv/bin/python -c 'import torch; a=torch.load("/tmp/nanochat-statehead-smoke.W8Wrbv/base_checkpoints/phase2-statehead-mps-seed1337-a/model_000020.pt",map_location="cpu"); b=torch.load("/tmp/nanochat-statehead-smoke.W8Wrbv/base_checkpoints/phase2-statehead-mps-seed1337-b/model_000020.pt",map_location="cpu"); assert a.keys()==b.keys(); diffs={key:(a[key].float()-b[key].float()).abs().max().item() for key in a}; print(f"tensor_count={len(diffs)} bitwise_equal={all(torch.equal(a[key],b[key]) for key in a)} max_abs={max(diffs.values())}")'
.venv/bin/python -c 'import statistics as s; a=[3415,3750,3497,3496,3175,3368,3408,3757,3956]; b=[3233,2897,3906,3228,3239,3332,3247,3239,3517]; print(f"run_a_postwarmup_median={s.median(a):.0f} mean={s.mean(a):.1f}"); print(f"run_b_postwarmup_median={s.median(b):.0f} mean={s.mean(b):.1f}")'
git diff --check && git status --short
```

The final command above is run after this log update.

## 2026-07-22 — Phase 3 RunPod preparation (no paid resources created)

### RunPod preflight state

- `runpodctl` authentication succeeds with version `2.7.2-309512b`, which is the latest upstream release at the time of this check.
- Account state before provisioning: zero current spend, no pods, and no network volumes. Two SSH public keys are already registered.
- Live inventory lists `NVIDIA H100 80GB HBM3` as available with low aggregate stock. Multiple secure-cloud data centers report H100 SXM inventory, but the list does not guarantee an eight-GPU allocation.
- The official RunPod H100 SXM page currently lists Secure Cloud at `$2.99/GPU-hour` and Community Cloud at `$2.69/GPU-hour`. The initial proposal uses Secure Cloud and a 30-minute hard termination guard: maximum GPU charge about `$1.50` for a one-H100 preflight, excluding negligible container-disk charges.
- No pod, volume, endpoint, or other paid resource was created during preparation.

### Controlled d12 contract

The checked-out d12 shapes were derived directly from the current model implementations at width 768, 6 heads, 128 channels/head, sequence length 2,048, and vocabulary 32,768:

| Model | Total params | Scaling params | Estimated train FLOPs/token |
|---|---:|---:|---:|
| GPT d12 | 286,261,730 | 110,100,912 | 759,696,048 |
| StateHead d12 | 85,767,218 | 60,555,264 | 363,414,672 |

The fixed controlled horizon is based on GPT d12's 12× scaling-parameter token rule and then floored to complete 524,288-token global batches:

```text
training steps: 2,520
global batch:   524,288 tokens
actual tokens: 1,321,205,760
seed:          1,337
precision:     BF16
FP8:           disabled
checkpoint:    predetermined final step
```

StateHead intentionally receives the same token horizon rather than its own parameter-derived horizon. This is the controlled-comparison choice required by the brief. The StateHead device batch remains unresolved until CUDA memory evidence exists; the global batch will remain fixed.

### Added preparation artifacts

The tested implementation and preflight code are pinned by local commit
`2944ed65dfb26809073e7b3446ff6255513c83d4` on branch
`codex/statehead-nanochat`. Nothing has been pushed remotely.

- `dev/statehead_cuda_preflight.py`: full-shape synthetic-token CUDA/DDP feasibility probe for compiled forward/backward and the real Muon/AdamW grouping. It records loss/gradient finiteness, step/compile timing, global throughput, max-rank peak VRAM, optimizer groups, and cross-rank parameter checksum spread. It is explicitly not a scientific learning result.
- `runs/statehead_cuda_preflight.sh`: module-mode launcher using BF16 and configurable world/device batch size.
- `dev/experiments/statehead-nanochat-cuda-preflight-v1.yaml`: exact bounded one-H100 preflight plan using the pinned `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404` image and a required 30-minute termination guard.
- `dev/experiments/statehead-nanochat-d12-controlled-v1.yaml`: controlled-run draft with explicit tokens, steps, data, evaluation policy, and the still-unresolved StateHead device batch.

The first direct script-path help check failed with `ModuleNotFoundError: No module named 'nanochat'`; the launcher now uses `python -m torch.distributed.run --module dev.statehead_cuda_preflight`, and module-mode help succeeds. The host has no system `torchrun`, so the launcher deliberately uses the current Python interpreter's distributed module.

### Local verification after preparation

```text
focused StateHead suite: 42 passed in 2.45s
full suite excluding known macOS memory-limit test: 85 passed, 14 skipped, 1 deselected in 4.88s
bash syntax, YAML parse, compileall, diff check: passed
nanochat/gpt.py diff: empty
CUDA execution: not run locally
```

### Phase 3 preparation command ledger

```bash
wc -l /Users/haybales/.agents/skills/runpod-usage/SKILL.md /Users/haybales/.agents/skills/runpodctl/SKILL.md && sed -n '1,240p' /Users/haybales/.agents/skills/runpod-usage/SKILL.md && sed -n '1,320p' /Users/haybales/.agents/skills/runpodctl/SKILL.md
for f in development-loop.md pod-workflows.md storage.md gpu-selection.md on-pod-setup.md getting-started.md; do wc -l "/Users/haybales/.agents/skills/runpod-usage/reference/$f"; done
sed -n '1,220p' /Users/haybales/.agents/skills/runpod-usage/reference/development-loop.md && sed -n '1,240p' /Users/haybales/.agents/skills/runpod-usage/reference/gpu-selection.md && sed -n '1,240p' /Users/haybales/.agents/skills/runpod-usage/reference/storage.md
sed -n '1,260p' /Users/haybales/.agents/skills/runpod-usage/reference/pod-workflows.md && sed -n '1,220p' /Users/haybales/.agents/skills/runpod-usage/reference/on-pod-setup.md && sed -n '1,220p' /Users/haybales/.agents/skills/runpod-usage/reference/getting-started.md
command -v runpodctl
runpodctl version
git rev-parse HEAD
git status --short
runpodctl user
runpodctl pod list --all
runpodctl network-volume list
runpodctl ssh list-keys
runpodctl gpu list --include-unavailable
runpodctl pod create --help
runpodctl datacenter list
rg -n "Phase 3|experiment manifest|Before any expensive run|8.?H100|fixed-token|num-iterations|checkpoint selection" STATEHEAD_NANOCHAT_CODEX_BRIEF.md dev scripts tests nanochat -g '*.md' -g '*.py' -g '*.sh' -g '*.yaml' -g '*.yml'
rg --files -g 'AGENTS.md' -g '*experiment*' -g '*.yaml' -g '*.yml' -g '*.sh' dev scripts . | sort
sed -n '680,770p' STATEHEAD_NANOCHAT_CODEX_BRIEF.md && sed -n '890,980p' STATEHEAD_NANOCHAT_CODEX_BRIEF.md && sed -n '1,180p' runs/speedrun.sh && sed -n '1,180p' runs/miniseries.sh && sed -n '1,220p' scripts/base_train.py
sed -n '220,560p' scripts/base_train.py
sed -n '560,760p' scripts/base_train.py && sed -n '1,260p' nanochat/statehead.py && sed -n '260,620p' nanochat/statehead.py
NANOCHAT_DTYPE=bfloat16 .venv/bin/python -c '<derive d12 GPT/StateHead parameter counts, FLOPs, state bytes, and fixed steps>'
sed -n '980,1060p' STATEHEAD_NANOCHAT_CODEX_BRIEF.md && sed -n '1,420p' dev/STATEHEAD_LOG.md
runpodctl template search pytorch --output json
sed -n '1,360p' nanochat/dataloader.py && sed -n '1,260p' nanochat/dataset.py && sed -n '1,220p' scripts/tok_train.py
git branch --show-current && git remote -v && git log -1 --oneline --decorate
runpodctl network-volume create --help && runpodctl template get --help
runpodctl template search torch291 --output json
sed -n '1,380p' scripts/base_eval.py && sed -n '1,300p' nanochat/checkpoint_manager.py && sed -n '1,220p' pyproject.toml && sed -n '1,240p' nanochat/common.py
sed -n '1,100p' nanochat/gpt.py && sed -n '1,100p' nanochat/optim.py && sed -n '1,120p' nanochat/flash_attention.py && .venv/bin/python -m pip --version
ls -l dev/statehead_cuda_preflight.py runs/statehead_cuda_preflight.sh dev/experiments/statehead-nanochat-cuda-preflight-v1.yaml dev/experiments/statehead-nanochat-d12-controlled-v1.yaml && sed -n '1,320p' dev/statehead_cuda_preflight.py && sed -n '1,180p' runs/statehead_cuda_preflight.sh && sed -n '1,220p' dev/experiments/statehead-nanochat-cuda-preflight-v1.yaml && sed -n '1,260p' dev/experiments/statehead-nanochat-d12-controlled-v1.yaml
chmod +x runs/statehead_cuda_preflight.sh && bash -n runs/statehead_cuda_preflight.sh
NANOCHAT_DTYPE=bfloat16 .venv/bin/python dev/statehead_cuda_preflight.py --help
.venv/bin/python -c '<parse both Phase 3 YAML manifests>'
git diff --check && .venv/bin/python -m compileall -q nanochat scripts tests dev/statehead_cuda_preflight.py
bash -n runs/statehead_cuda_preflight.sh && NANOCHAT_DTYPE=bfloat16 .venv/bin/python -m dev.statehead_cuda_preflight --help
torchrun --help | rg -n -- '--module|-m,'
.venv/bin/python -m torch.distributed.run --help | rg -n -- '--module|-m,'
NANOCHAT_DTYPE=float32 .venv/bin/python -m pytest tests/test_statehead.py -q
NANOCHAT_DTYPE=float32 .venv/bin/python -m pytest -q -k 'not test_memory_limit'
bash -n runs/statehead_cuda_preflight.sh && git diff --check && .venv/bin/python -m compileall -q nanochat scripts tests dev/statehead_cuda_preflight.py && git diff --quiet -- nanochat/gpt.py
rg -n "total = wte" nanochat/statehead.py && sed -n '210,245p' nanochat/statehead.py && git diff --stat && git diff --numstat && git branch --list 'codex/statehead-nanochat'
```

The first five RunPod API commands initially failed inside the network-restricted sandbox with DNS resolution errors and were immediately repeated with approved network access; the repeated commands succeeded.

## 2026-07-22 — bounded one-H100 CUDA preflight

The explicitly approved infrastructure-only preflight ran on one Secure Cloud
`NVIDIA H100 80GB HBM3` pod at `$2.99/hour`. The pod had a hard 30-minute
termination deadline and was manually deleted immediately after the artifacts were
retrieved. RunPod reported no remaining pods and `$0/hour` afterward. The observed
account-balance delta was approximately `$0.11`.

Environment:

```text
model code commit: 2944ed65dfb26809073e7b3446ff6255513c83d4
checked-out branch head: 39548ab5508b6937e87c48ecd6ab4f6e1a520b9c
image: runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404
GPU: NVIDIA H100 80GB HBM3, 81,559 MiB reported
driver: 570.195.03
Python: 3.12.3
PyTorch: 2.9.1+cu128
CUDA runtime: 12.8
precision: BF16 activations, FP32 scan accumulation
world/device batch/sequence: 1 / 1 / 2,048
compile: true
synthetic fixed token row: true
```

Result:

```text
process exit: 0
finite loss and gradients: yes
optimizer steps: 2
step 0, including compilation: 55.75203488022089 s, loss 10.398395538330078
step 1, steady shape: 0.018427319824695587 s, loss 5.794830322265625
steady global throughput: 111,139.33113894018 tokens/s
peak allocated VRAM: 2,310,125,568 bytes (2.151 GiB)
peak reserved VRAM: 3,634,364,416 bytes (3.385 GiB)
parameter checksum spread: 0.0 (trivial one-rank check)
```

The loss change is only a synthetic repeated-row optimizer sanity check. This run
does not test dataset learning, multi-rank optimizer communication, GPT parity,
validation BPB, CORE, or speedrun performance. No parity claim is made.

Artifacts:

- `dev/results/statehead-nanochat-cuda-preflight-v1.json`
- retrieved raw JSON SHA-256: `51e7f680b56e83341f81a02b9fc2a6489e6dd32effbb67c045b062c706155d0f`
- retrieved raw log SHA-256: `3fa716cd10d27d5fb4423889d9f8280b4286bb21399ea52b561c803f6391d151`

The raw log also contained a benign warning that NumPy was not installed in the
prebuilt container. The probe itself does not use NumPy.

### CUDA preflight command ledger

```bash
sed -n '1,240p' /Users/haybales/.agents/skills/runpod/SKILL.md && sed -n '1,220p' /Users/haybales/.agents/skills/runpod-usage/reference/pod-workflows.md
sed -n '1,320p' /Users/haybales/.agents/skills/runpodctl/SKILL.md && sed -n '1,280p' /Users/haybales/.agents/skills/runpod/golden-paths/06-dev-pod.md
date -u -v+30M '+%Y-%m-%dT%H:%M:%SZ'
runpodctl version
runpodctl user
runpodctl pod list --all
runpodctl pod create --help
runpodctl pod create --name statehead-cuda-preflight-v1 --image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 --gpu-id "NVIDIA H100 80GB HBM3" --gpu-count 1 --cloud-type SECURE --container-disk-in-gb 30 --ports "22/tcp" --ssh --terminate-after 2026-07-23T02:04:45Z
runpodctl pod get 219undjfa90mm8
runpodctl ssh info 219undjfa90mm8
ssh -i /Users/haybales/.runpod/ssh/runpodctl-ssh-key -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -p 10508 root@216.243.220.202 'hostname; nvidia-smi -L; python --version; python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"; df -h /workspace'
ssh -i /Users/haybales/.runpod/ssh/runpodctl-ssh-key -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 10508 root@216.243.220.202 'git clone --branch codex/statehead-nanochat --depth 1 https://github.com/samfurr/nanochat.git /workspace/nanochat'
ssh -i /Users/haybales/.runpod/ssh/runpodctl-ssh-key -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 10508 root@216.243.220.202 'cd /workspace/nanochat && git rev-parse HEAD && git status --short && bash -n runs/statehead_cuda_preflight.sh && python -c "import filelock, torch; print(filelock.__version__, torch.__version__)"'
ssh -i /Users/haybales/.runpod/ssh/runpodctl-ssh-key -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 10508 root@216.243.220.202 'cd /workspace/nanochat && NANOCHAT_REPO_COMMIT=2944ed65dfb26809073e7b3446ff6255513c83d4 NPROC_PER_NODE=1 DEVICE_BATCH_SIZE=1 PREFLIGHT_STEPS=2 RESULTS_DIR=/workspace/statehead-preflight-results bash runs/statehead_cuda_preflight.sh'
scp -i /Users/haybales/.runpod/ssh/runpodctl-ssh-key -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P 10508 root@216.243.220.202:/workspace/statehead-preflight-results/statehead-d12-b1-w1.json /private/tmp/statehead-d12-b1-w1.json
scp -i /Users/haybales/.runpod/ssh/runpodctl-ssh-key -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P 10508 root@216.243.220.202:/workspace/statehead-preflight-results/statehead-d12-b1-w1.log /private/tmp/statehead-d12-b1-w1.log
.venv/bin/python -c '<validate retrieved JSON fields and print key metrics>' && shasum -a 256 /private/tmp/statehead-d12-b1-w1.json /private/tmp/statehead-d12-b1-w1.log && wc -c /private/tmp/statehead-d12-b1-w1.json /private/tmp/statehead-d12-b1-w1.log
runpodctl pod delete 219undjfa90mm8
runpodctl pod list --all
runpodctl user
sed -n '1,260p' /private/tmp/statehead-d12-b1-w1.json && sed -n '1,260p' /private/tmp/statehead-d12-b1-w1.log
```

Post-retrieval local verification:

```text
result JSON byte-for-byte SHA-256 match: passed
result/manifest consistency assertions: passed
focused StateHead suite: 42 passed in 2.38s
full suite excluding known macOS memory-limit test: 85 passed, 14 skipped, 1 deselected in 4.90s
diff check, compileall, and unchanged nanochat/gpt.py assertion: passed
```

```bash
.venv/bin/python -c '<assert result JSON and both manifests are consistent>' && shasum -a 256 dev/results/statehead-nanochat-cuda-preflight-v1.json /private/tmp/statehead-d12-b1-w1.json
NANOCHAT_DTYPE=float32 .venv/bin/python -m pytest tests/test_statehead.py -q
NANOCHAT_DTYPE=float32 .venv/bin/python -m pytest -q -k 'not test_memory_limit'
git diff --check && .venv/bin/python -m compileall -q nanochat scripts tests dev/statehead_cuda_preflight.py && git diff --quiet -- nanochat/gpt.py && git status --short
```

## 2026-07-23 — Vast.ai eight-H100 StateHead preflight

The approved Vast.ai offer `40228016` was rented as instance `45629291` for a
bounded infrastructure/compiler/DDP check. The unverified Japan host supplied
the exact single-node shape: eight `NVIDIA H100 80GB HBM3` GPUs with all-to-all
`NV18` topology. The frozen checkout at
`df6e4161b2def519d7bf9061c15250f31b307abc` installed PyTorch
`2.9.1+cu128`.

The compiled BF16 PyTorch-reference StateHead d12 probe completed two
batch-32 optimizer steps on eight ranks. All checked parameter gradients were
finite, cross-rank checksum spread was `0.0`, the compile-inclusive first step
took `53.1073154178448` seconds, and the steady step took
`0.12498901505023241` seconds for `4,194,672.626144717` aggregate tokens per
second. This is an infrastructure result, not a learning-quality or parity
claim.

The optional host-local reference parity suite did not run. `pytest` was not
present in the production-only environment, and the destruction guard fired
while the locked development group was installing. The instance and its
remote result files were destroyed before retrieval. The captured JSON stdout
is preserved at
`dev/results/statehead-vast-8gpu-preflight-20260723/captured-result.json`; its
provenance and the complete command ledger are in the adjacent `REPORT.md`.

Vast.ai posted `$1.749`: `$1.734` GPU, `$0.004` disk, and `$0.011` download.
The observed credit delta was `$1.7503081587`. Cleanup verification returned
zero instances and zero volumes.

## 2026-07-22 — Phase 5 native CUDA candidate and winner-matching FP8 wiring

The user explicitly moved the work into the performance phase and asked to
match the GPT leaderboard run's quantization while taking implementation
inspiration from the fused MLX kernel in
`/Users/haybales/projects/statehead-speed`.

### Source-of-truth precision decision

The checked-out leaderboard winner and `runs/speedrun.sh` use `--fp8` with the
`tensorwise` recipe. This is mixed-precision training, not an 8-bit stored
model:

- master matrix weights and optimizer state remain FP32;
- ordinary CUDA compute/activations remain BF16;
- each eligible large `nn.Linear` dynamically quantizes one scale per tensor;
- forward inputs and weights use `float8_e4m3fn`;
- backward grad-output uses `float8_e5m2` while saved inputs/weights remain
  `float8_e4m3fn`;
- cuBLAS `torch._scaled_mm` performs the three Linear GEMMs;
- the current winner filter requires both dimensions divisible by 16 and the
  smaller dimension at least 128;
- evaluation temporarily restores the BF16 Linear path.

`STATEHEAD_NANOCHAT_CODEX_BRIEF.md` already specifies the same Phase 5
boundary: fuse gate activations plus forward/reverse scan, retain large
projections as high-performance GEMMs, use FP8 only around gate/output GEMMs,
and keep recurrence accumulation in FP32.

### Implementation

- Added a lazy-built PyTorch C++/CUDA extension. Its blocked algorithm mirrors
  the MLX implementation: FP32 affine chunk summaries, FP32 chunk-boundary
  scan, FP32 per-chunk output replay, and a reverse kernel that replays one
  chunk into shared memory before applying the analytical gate gradients.
- The native op consumes raw `[B,T,4,H,Dh]` gates and fuses sigmoid/tanh gate
  activation into forward and reverse recurrence kernels. The gate projection
  and output projections remain GEMMs.
- The GPU extra now declares and locks Ninja, which PyTorch's lazy C++/CUDA
  extension builder requires.
- The autograd boundary uses `torch.library.custom_op` with fake-tensor shape
  registrations so `torch.compile` can treat forward and backward as opaque
  native operations.
- `scan_backend=auto` selects native CUDA on CUDA and the existing functional
  PyTorch reference on CPU/MPS. Both `pytorch` and `cuda` are explicit CLI
  choices.
- StateHead may now use the same FP8 conversion recipe/filter as GPT. At a
  representative aligned test shape this converts the StateHead gate, block
  output projection, and LM output projection; the recurrent scan itself never
  receives FP8 tensors and continues to accumulate in FP32.
- The existing preflight runner now accepts explicit scan backend and FP8
  switches. Its defaults remain the historical PyTorch/BF16 behavior.

No Transformer module code changed.

### Local verification

```text
native CUDA extension actually compiled with nvcc: not available locally
host C++ binding syntax check: passed
custom-op fake CUDA shape/dtype dispatch: passed
custom-op registered autograd exercised with a CPU test double: passed
StateHead FP32 focused suite: 46 passed, 14 CUDA-only skipped
StateHead BF16 focused suite: 46 passed, 14 CUDA-only skipped
full suite excluding known macOS memory-limit test: 89 passed, 28 skipped, 1 deselected
CPU five-step compiled smoke loss: 5.925142 -> 5.915248
MPS/BF16 five-step compiled smoke loss: 5.924298 -> 5.914391
MPS/BF16 finite checkpoint save: passed
shell syntax, Python compileall, diff check: passed
unchanged nanochat/gpt.py assertion: passed
```

The first BF16 focused invocation exposed that the existing prefill-versus-
token-decode assertion still used FP32 tolerances even when global compute was
BF16. Its maximum state difference was `0.003173828125`. The assertion now uses
the same `2e-2` reduced-precision tolerance as the existing BF16 scan parity
test; both FP32 and BF16 suites pass.

### Claims still unresolved

The local machine cannot compile or execute CUDA. Therefore none of the
following is claimed yet:

- successful nvcc build against PyTorch 2.9.1/CUDA 12.8;
- native CUDA forward or gradient parity;
- native custom-op fullgraph compilation on H100;
- FP8 StateHead finite training or learning parity;
- native scan speedup over the PyTorch scan;
- end-to-end StateHead speed approaching the FlashAttention GPT baseline.

The committed preflight manifest is
`dev/experiments/statehead-nanochat-fused-cuda-preflight-v1.yaml`. On an already
provisioned H100 checkout, the next correctness command is:

```bash
NANOCHAT_DTYPE=float32 python -m pytest tests/test_statehead.py -q
```

Implementation commit:

```text
b952753243ea14ac391d617244f2fbd52ba0a487
```

The existing Phase 3 paired runner is pinned to that model-code commit but
explicitly selects `--statehead-scan-backend=pytorch`, preserving the approved
BF16 scientific comparison. The separate Phase 5 synthetic full-step gate uses
these commands after CUDA correctness passes:

```bash
NANOCHAT_REPO_COMMIT=b952753243ea14ac391d617244f2fbd52ba0a487 NPROC_PER_NODE=1 DEVICE_BATCH_SIZE=32 PREFLIGHT_STEPS=5 SCAN_BACKEND=pytorch FP8=0 PREFLIGHT_TAG=statehead-d12-pytorch-bf16 RESULTS_DIR=/workspace/statehead-fused-results bash runs/statehead_cuda_preflight.sh
NANOCHAT_REPO_COMMIT=b952753243ea14ac391d617244f2fbd52ba0a487 NPROC_PER_NODE=1 DEVICE_BATCH_SIZE=32 PREFLIGHT_STEPS=5 SCAN_BACKEND=cuda FP8=0 PREFLIGHT_TAG=statehead-d12-cuda-bf16 RESULTS_DIR=/workspace/statehead-fused-results bash runs/statehead_cuda_preflight.sh
NANOCHAT_REPO_COMMIT=b952753243ea14ac391d617244f2fbd52ba0a487 NPROC_PER_NODE=1 DEVICE_BATCH_SIZE=32 PREFLIGHT_STEPS=5 SCAN_BACKEND=cuda FP8=1 PREFLIGHT_TAG=statehead-d12-cuda-fp8 RESULTS_DIR=/workspace/statehead-fused-results bash runs/statehead_cuda_preflight.sh
```

The remaining same-day continuation entries are recorded newest-first: paid-run
capacity attempt, controlled paired-run preparation, eight-H100 DDP preflight,
then the earlier batch-32 capacity probe.

## 2026-07-22 — approved paired run blocked by exact capacity

The `$24.00` paired GPT/StateHead d12 run was explicitly approved. No training
pod could be allocated, so the dataset-backed run did not start and produced no
scientific result.

The pre-launch state was clean:

```text
checkout head: 1533bde849ae70fa6e1c89120cebb5f4f13189f4
launcher SHA-256: b3a0056b4d9ac0c4f534792f684a9a5cd6c2ebe5cf14dda77a2bf0ff675be540
starting balance: $206.882227635
starting spend rate: $0/hour
starting pods: 0
starting network volumes: 0
```

`CA-MTL-1` reported low H100 inventory but rejected network-volume creation
because that data center does not support network volumes; no resource was
created. A 100 GB volume was then created in `US-NE-1`, but RunPod reported no
remaining instance matching the exact Secure Cloud eight-H100 request. That
unused volume (`psiorukdgq`) was deleted immediately. One final exact-hardware
retry used `AP-JP-1`; its temporary volume (`nc29r10zf4`) was also deleted
immediately after the same capacity response.

No GPU pod was created in any data center. The final RunPod checks reported zero
pods, zero network volumes, `$0/hour`, and the unchanged balance
`$206.882227635`. No substitute GPU, cloud type, world size, storage design, or
training recipe was used. The approved run remains waiting for exact eight-H100
capacity; no parity or quality claim can be made.

Post-attempt local verification passed: both manifests agree on approved/capacity
wait status, the launcher hash remains unchanged, its dry run still succeeds,
the focused StateHead suite reported `42 passed in 2.30s`, and the full suite
excluding the known macOS memory-limit test reported `85 passed, 14 skipped,
1 deselected in 4.72s`. Compileall, diff checks, and the pinned model/training
source assertion also passed.

### Capacity-attempt command ledger

```bash
wc -l /Users/haybales/.agents/skills/runpod/SKILL.md && sed -n '1,520p' /Users/haybales/.agents/skills/runpod/SKILL.md
wc -l /Users/haybales/.agents/skills/runpodctl/SKILL.md && sed -n '1,520p' /Users/haybales/.agents/skills/runpodctl/SKILL.md
wc -l /Users/haybales/.agents/skills/runpod-usage/SKILL.md && sed -n '1,520p' /Users/haybales/.agents/skills/runpod-usage/SKILL.md
wc -l /Users/haybales/.agents/skills/runpod-usage/reference/development-loop.md && sed -n '1,520p' /Users/haybales/.agents/skills/runpod-usage/reference/development-loop.md
wc -l /Users/haybales/.agents/skills/runpod-usage/reference/pod-workflows.md && sed -n '1,520p' /Users/haybales/.agents/skills/runpod-usage/reference/pod-workflows.md
wc -l /Users/haybales/.agents/skills/runpod-usage/reference/storage.md && sed -n '1,520p' /Users/haybales/.agents/skills/runpod-usage/reference/storage.md
wc -l /Users/haybales/.agents/skills/runpod-usage/reference/on-pod-setup.md && sed -n '1,520p' /Users/haybales/.agents/skills/runpod-usage/reference/on-pod-setup.md
sed -n '1,140p' /Users/haybales/.agents/skills/runpodctl/SKILL.md && sed -n '141,300p' /Users/haybales/.agents/skills/runpodctl/SKILL.md
sed -n '1,120p' /Users/haybales/.agents/skills/runpod-usage/SKILL.md && sed -n '1,140p' /Users/haybales/.agents/skills/runpod-usage/reference/development-loop.md
sed -n '1,180p' /Users/haybales/.agents/skills/runpod-usage/reference/pod-workflows.md
sed -n '1,180p' /Users/haybales/.agents/skills/runpod-usage/reference/storage.md && sed -n '1,140p' /Users/haybales/.agents/skills/runpod-usage/reference/on-pod-setup.md
date -u -v+60M '+%Y-%m-%dT%H:%M:%SZ' && git rev-parse HEAD && git status --short && shasum -a 256 runs/statehead_d12_controlled.sh && sed -n '1,180p' dev/experiments/statehead-nanochat-d12-controlled-v1.yaml
runpodctl version && runpodctl network-volume create --help && runpodctl pod create --help
runpodctl user
runpodctl pod list --all
runpodctl network-volume list
runpodctl datacenter list | jq '[.[] | {id, h100: [.gpuAvailability[]? | select(.gpuId == "NVIDIA H100 80GB HBM3")] } | select((.h100 | length) > 0)]'
runpodctl network-volume create --name statehead-d12-controlled-v1 --size 100 --data-center-id CA-MTL-1
runpodctl network-volume create --name statehead-d12-controlled-v1 --size 100 --data-center-id US-NE-1
date -u -v+60M '+%Y-%m-%dT%H:%M:%SZ'
runpodctl pod create --name statehead-d12-paired-v1 --image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 --gpu-id "NVIDIA H100 80GB HBM3" --gpu-count 8 --cloud-type SECURE --data-center-ids US-NE-1 --network-volume-id psiorukdgq --volume-mount-path /workspace --container-disk-in-gb 30 --ports "22/tcp" --ssh --terminate-after 2026-07-23T03:19:45Z
runpodctl network-volume delete psiorukdgq
runpodctl network-volume list
runpodctl datacenter list | jq '[.[] | select(.id == "AP-JP-1" or .id == "EUR-IS-3" or .id == "US-NE-1") | {id, h100: [.gpuAvailability[]? | select(.gpuId == "NVIDIA H100 80GB HBM3")]}]'
runpodctl network-volume create --name statehead-d12-controlled-v1 --size 100 --data-center-id AP-JP-1
date -u -v+60M '+%Y-%m-%dT%H:%M:%SZ'
runpodctl pod create --name statehead-d12-paired-v1 --image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 --gpu-id "NVIDIA H100 80GB HBM3" --gpu-count 8 --cloud-type SECURE --data-center-ids AP-JP-1 --network-volume-id nc29r10zf4 --volume-mount-path /workspace --container-disk-in-gb 30 --ports "22/tcp" --ssh --terminate-after 2026-07-23T03:20:46Z
runpodctl network-volume delete nc29r10zf4
runpodctl pod list --all
runpodctl network-volume list
runpodctl user
.venv/bin/python -c '<assert approved capacity-wait manifests consistent>' && DRY_RUN=1 bash runs/statehead_d12_controlled.sh > /private/tmp/statehead-controlled-dry-run.txt
NANOCHAT_DTYPE=float32 .venv/bin/python -m pytest tests/test_statehead.py -q
NANOCHAT_DTYPE=float32 .venv/bin/python -m pytest -q -k 'not test_memory_limit'
git diff --check && .venv/bin/python -m compileall -q nanochat scripts tests dev/statehead_cuda_preflight.py && git diff --quiet -- nanochat/gpt.py && git diff --quiet 2944ed65dfb26809073e7b3446ff6255513c83d4 -- nanochat/gpt.py nanochat/statehead.py nanochat/optim.py nanochat/dataloader.py nanochat/checkpoint_manager.py scripts/base_train.py scripts/base_eval.py && git status --short && git diff --stat
```

## 2026-07-22 — controlled paired-run launcher preparation

No paid resource was launched in this step. The checked-out source of truth was
`795d07e1d7d62c62504a7970f8615ada0d427be2`; the pre-existing untracked
`.DS_Store`, `.agents/`, and `STATEHEAD_NANOCHAT_CODEX_BRIEF.md` remained
untouched.

`runs/statehead_d12_controlled.sh` now encodes the first dataset-backed paired
comparison. Its default order is GPT followed by StateHead on one eight-H100
node. Both use:

```text
model/training code commit: 2944ed65dfb26809073e7b3446ff6255513c83d4
seed: 1337
precision: BF16
FP8: false
depth/width/heads: 12 / 768 / 6
sequence length: 2,048
device batch: 32 rows per rank
world size: 8
global batch: 524,288 tokens
steps: 2,520
tokens per model: 1,321,205,760
train shards: 00000 through 00169
validation shard: 06542
tokenizer: 2B characters, vocabulary 32,768
validation BPB: every 250 steps over 41,943,040 tokens
final evaluation: full CORE plus train/validation BPB
checkpoint selection: predetermined final step 2,520
```

The launcher has a dry-run mode, rejects a world size other than eight, rejects
unknown or duplicate architecture selectors, verifies that all model/training
paths still match the pinned code commit, verifies the exact 171 dataset shards,
and refuses to overwrite an existing checkpoint directory. It retains raw train
and evaluation logs, final model/metadata, all optimizer hashes, per-task CORE
CSV, tokenizer hashes, environment details, and exit status. It does not create
RunPod infrastructure or silently enable FP8.

The RunPod workflow uses a temporary 100 GB network volume so data, checkpoints,
and logs survive pod termination until retrieval. The volume and pod will be
deleted after successful artifact retrieval, or the volume will be deleted
immediately if exact eight-H100 pod allocation fails. The target data center is
selected at launch because eight-H100 capacity is dynamic and currently sparse.

The live eight-H100 rate observed in the completed preflight was `$23.92/hour`.
RunPod's pricing page also listed Secure Cloud H100 SXM at `$2.99/GPU-hour` on
2026-07-23. Official network-volume pricing was `$0.07/GB/month` below 1 TB,
billed hourly, making one hour of 100 GB approximately `$0.0098`. The paired-run
manifest therefore proposes a 60-minute hard pod termination deadline,
`$23.92` maximum GPU charge, and `$24.00` total infrastructure ceiling. This
ceiling is awaiting explicit approval.

Preparation verification:

```text
bash syntax: passed
paired dry run: passed; both exact train/eval commands printed and no work executed
FP8 absence assertion: passed
world-size rejection guard: passed
duplicate-architecture rejection guard: passed
paired YAML consistency assertions: passed
launcher SHA-256: b3a0056b4d9ac0c4f534792f684a9a5cd6c2ebe5cf14dda77a2bf0ff675be540
focused StateHead suite: 42 passed in 2.39s
full suite excluding known macOS memory-limit test: 85 passed, 14 skipped, 1 deselected in 4.68s
base train/eval CLI flag checks: passed
compileall and diff check: passed
model/training paths match pinned commit: passed
nanochat/gpt.py working-tree diff: empty
RunPod resources after preparation: zero pods, zero network volumes, $0/hour
```

### Controlled paired-run preparation command ledger

```bash
wc -l /Users/haybales/.agents/skills/runpod/SKILL.md && sed -n '1,260p' /Users/haybales/.agents/skills/runpod/SKILL.md
wc -l /Users/haybales/.agents/skills/runpodctl/SKILL.md && sed -n '1,420p' /Users/haybales/.agents/skills/runpodctl/SKILL.md
wc -l /Users/haybales/.agents/skills/runpod-usage/SKILL.md && sed -n '1,360p' /Users/haybales/.agents/skills/runpod-usage/SKILL.md
wc -l /Users/haybales/.agents/skills/runpod-usage/reference/development-loop.md && sed -n '1,520p' /Users/haybales/.agents/skills/runpod-usage/reference/development-loop.md
wc -l /Users/haybales/.agents/skills/runpod-usage/reference/pod-workflows.md && sed -n '1,520p' /Users/haybales/.agents/skills/runpod-usage/reference/pod-workflows.md
wc -l /Users/haybales/.agents/skills/runpod-usage/reference/storage.md && sed -n '1,520p' /Users/haybales/.agents/skills/runpod-usage/reference/storage.md
wc -l /Users/haybales/.agents/skills/runpod-usage/reference/gpu-selection.md && sed -n '1,520p' /Users/haybales/.agents/skills/runpod-usage/reference/gpu-selection.md
wc -l /Users/haybales/.agents/skills/runpod-usage/reference/on-pod-setup.md && sed -n '1,520p' /Users/haybales/.agents/skills/runpod-usage/reference/on-pod-setup.md
git rev-parse HEAD && git status --short && git branch --show-current && git remote -v && rg --files -g 'AGENTS.md' -g '!**/.git/**'
sed -n '860,1060p' STATEHEAD_NANOCHAT_CODEX_BRIEF.md
sed -n '1,380p' scripts/base_train.py && sed -n '380,760p' scripts/base_train.py
sed -n '1,260p' runs/speedrun.sh && sed -n '1,300p' scripts/base_eval.py
sed -n '1,300p' nanochat/dataset.py && sed -n '1,260p' scripts/tok_train.py && sed -n '1,240p' nanochat/checkpoint_manager.py
sed -n '1,220p' dev/experiments/statehead-nanochat-d12-controlled-v1.yaml && sed -n '1,220p' pyproject.toml && rg -n 'NANOCHAT_BASE_DIR|tokenizer|climbmix|hf_hub_download|download' nanochat scripts runs -g '*.py' -g '*.sh'
nl -ba runs/speedrun.sh | sed -n '1,150p' && nl -ba runs/miniseries.sh | sed -n '1,180p' && nl -ba runs/scaling_laws.sh | sed -n '1,180p'
nl -ba scripts/base_train.py | sed -n '220,580p'
nl -ba scripts/base_eval.py | sed -n '1,280p' && nl -ba scripts/tok_train.py | sed -n '1,130p' && nl -ba nanochat/dataset.py | sed -n '1,190p'
nl -ba nanochat/dataloader.py | sed -n '1,260p' && nl -ba nanochat/common.py | sed -n '1,130p' && nl -ba nanochat/common.py | sed -n '240,390p'
git log --oneline --decorate -8 && git show --stat --oneline 2944ed65dfb26809073e7b3446ff6255513c83d4 && git diff 2944ed65dfb26809073e7b3446ff6255513c83d4..HEAD -- scripts/base_train.py nanochat/gpt.py nanochat/statehead.py nanochat/dataloader.py runs
nl -ba scripts/base_train.py | sed -n '288,560p'
nl -ba scripts/base_eval.py | sed -n '120,270p'
nl -ba nanochat/dataset.py | sed -n '1,185p' && nl -ba scripts/tok_train.py | sed -n '1,120p'
find dev -maxdepth 3 -type f -print | sort && find runs -maxdepth 1 -type f -print | sort
rg -n 'resume|dataloader|save_checkpoint|checkpoint_dir|train_loader|training_time|tokens per second|tok/sec|tokens_per' tests scripts dev -g '*.py' -g '*.sh' -g '*.md'
nl -ba scripts/base_train.py | sed -n '540,655p' && nl -ba scripts/base_eval.py | sed -n '205,260p' && nl -ba nanochat/dataset.py | sed -n '1,105p'
nl -ba nanochat/checkpoint_manager.py | sed -n '1,145p' && ls -la uv.lock && du -h uv.lock
chmod +x runs/statehead_d12_controlled.sh && bash -n runs/statehead_d12_controlled.sh
git status --short && git diff --stat && git diff --check && sed -n '1,320p' runs/statehead_d12_controlled.sh
sed -n '1,260p' dev/experiments/statehead-nanochat-d12-controlled-v1.yaml && sed -n '1,240p' dev/experiments/gpt-d12-controlled-v1.yaml
DRY_RUN=1 bash runs/statehead_d12_controlled.sh
DRY_RUN=1 bash runs/statehead_d12_controlled.sh > /private/tmp/statehead-controlled-dry-run.txt
if DRY_RUN=1 NPROC_PER_NODE=4 bash runs/statehead_d12_controlled.sh > /private/tmp/statehead-controlled-invalid-world.txt 2>&1; then exit 1; fi
if DRY_RUN=1 RUN_ARCHES=gpt,gpt bash runs/statehead_d12_controlled.sh > /private/tmp/statehead-controlled-invalid-arches.txt 2>&1; then exit 1; fi
.venv/bin/python -c '<assert paired manifests consistent>'
command -v shellcheck || true
runpodctl user
runpodctl pod list --all
runpodctl network-volume list
runpodctl datacenter list
runpodctl network-volume create --help && runpodctl pod create --help
NANOCHAT_DTYPE=float32 .venv/bin/python -m pytest tests/test_statehead.py -q
NANOCHAT_DTYPE=float32 .venv/bin/python -m pytest -q -k 'not test_memory_limit'
bash -n runs/statehead_d12_controlled.sh && NANOCHAT_DTYPE=float32 .venv/bin/python -m scripts.base_train --help > /private/tmp/statehead-base-train-help.txt && NANOCHAT_DTYPE=float32 .venv/bin/python -m scripts.base_eval --help > /private/tmp/statehead-base-eval-help.txt
git diff --check && .venv/bin/python -m compileall -q nanochat scripts tests dev/statehead_cuda_preflight.py && git diff --quiet -- nanochat/gpt.py && git diff --quiet 2944ed65dfb26809073e7b3446ff6255513c83d4 -- nanochat/gpt.py nanochat/statehead.py nanochat/optim.py nanochat/dataloader.py nanochat/checkpoint_manager.py scripts/base_train.py scripts/base_eval.py
bash -n runs/statehead_d12_controlled.sh && DRY_RUN=1 bash runs/statehead_d12_controlled.sh > /private/tmp/statehead-controlled-dry-run.txt
shasum -a 256 runs/statehead_d12_controlled.sh
rg -n '^## 2026-07-22 — controlled paired-run launcher preparation|^## 2026-07-22 — eight-H100|^## 2026-07-22 — one-H100 device-batch-32' dev/STATEHEAD_LOG.md && tail -n 30 dev/STATEHEAD_LOG.md
rg -n '^## 2026-07-22' dev/STATEHEAD_LOG.md
NANOCHAT_DTYPE=float32 .venv/bin/python -m pytest tests/test_statehead.py -q
NANOCHAT_DTYPE=float32 .venv/bin/python -m pytest -q -k 'not test_memory_limit'
bash -n runs/statehead_d12_controlled.sh && DRY_RUN=1 bash runs/statehead_d12_controlled.sh > /private/tmp/statehead-controlled-dry-run.txt && .venv/bin/python -c '<assert launcher hash and paired manifests consistent>'
git diff --check && .venv/bin/python -m compileall -q nanochat scripts tests dev/statehead_cuda_preflight.py && git diff --quiet -- nanochat/gpt.py && git diff --quiet 2944ed65dfb26809073e7b3446ff6255513c83d4 -- nanochat/gpt.py nanochat/statehead.py nanochat/optim.py nanochat/dataloader.py nanochat/checkpoint_manager.py scripts/base_train.py scripts/base_eval.py && git status --short && git diff --stat && git diff --summary
```

## 2026-07-22 — eight-H100 DDP/NCCL preflight

The explicitly approved distributed preflight ran on one Secure Cloud node with
eight `NVIDIA H100 80GB HBM3` GPUs at `$23.92/hour`. It had a hard 15-minute
termination deadline. Both synthetic optimizer steps completed on all eight
ranks; the result and raw log were retrieved and validated before the pod was
manually deleted. RunPod then reported no remaining pods and `$0/hour`. The
immediately observed account-balance delta was `$0.8828623333`, below the
approved `$5.98` ceiling; later billing adjustments may change that value.

Environment:

```text
model code commit: 2944ed65dfb26809073e7b3446ff6255513c83d4
checked-out branch head: ad9aab2f4f9bbbd100aad6e384243b4a16a085d2
image: runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404
RunPod pod: v1n834ijpcebmu (deleted)
cloud/location: Secure Cloud / CA
GPU: 8 × NVIDIA H100 80GB HBM3, 81,559 MiB reported per rank
driver: 580.126.09
PyTorch: 2.9.1+cu128
CUDA runtime: 12.8
precision: BF16 activations, FP32 scan accumulation
world/device batch/sequence: 8 / 32 / 2,048
global tokens per step: 524,288
compile: true
synthetic fixed token rows: true
```

Result:

```text
process exit: 0
all eight ranks completed: yes
finite loss and gradients on all ranks: yes
optimizer steps: 2
step 0, including compilation, max rank: 65.17646897956729 s, rank-0 loss 10.397625923156738
step 1, steady shape, max rank: 0.12632401660084724 s, rank-0 loss 9.237276077270508
steady global throughput: 4,150,343.0155852376 tokens/s
peak allocated VRAM, max rank: 43,207,041,024 bytes (40.240 GiB)
peak reserved VRAM, max rank: 47,355,789,312 bytes (44.104 GiB)
parameter checksum spread across ranks: 0.0
```

This establishes that the intended device/global batch, NCCL process group, and
distributed Muon/AdamW path can complete two compiled steps on the target
hardware. It does not test data loading, dataset learning, validation BPB, CORE,
GPT parity, end-to-end training stability, or sustained throughput. The loss
change is only a synthetic optimizer sanity check. No parity claim is made.

Artifacts:

- `dev/results/statehead-nanochat-cuda-8gpu-preflight-v1.json`
- retrieved raw JSON SHA-256: `43f6c4b9fb769125c77d85b6c3e3cf86434df0257f00f60975fd11c9ab153a00`
- retrieved raw log SHA-256: `fba0b084577932b9ff3fb220b7882fe38a678cf38cfa8f5680ded94910e6d74d`

The raw log contained benign warnings that NumPy was not installed in the
prebuilt container. The probe itself does not use NumPy.

### Eight-H100 preflight command ledger

```bash
sed -n '1,240p' /Users/haybales/.agents/skills/runpod/SKILL.md
sed -n '1,320p' /Users/haybales/.agents/skills/runpodctl/SKILL.md
date -u -v+15M '+%Y-%m-%dT%H:%M:%SZ'
git rev-parse HEAD && git status --short && sed -n '1,240p' dev/experiments/statehead-nanochat-cuda-8gpu-preflight-v1.yaml
runpodctl user
runpodctl pod list --all
runpodctl pod create --help
runpodctl pod create --name statehead-cuda-8gpu-preflight-v1 --image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 --gpu-id "NVIDIA H100 80GB HBM3" --gpu-count 8 --cloud-type SECURE --container-disk-in-gb 30 --ports "22/tcp" --ssh --terminate-after 2026-07-23T02:09:49Z
runpodctl pod get v1n834ijpcebmu
runpodctl ssh info v1n834ijpcebmu
ssh -i /Users/haybales/.runpod/ssh/runpodctl-ssh-key -p 22185 -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 root@69.30.85.162 'nvidia-smi -L && python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.device_count())"'
ssh -i /Users/haybales/.runpod/ssh/runpodctl-ssh-key -p 22185 -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@69.30.85.162 'git clone --branch codex/statehead-nanochat --depth 1 https://github.com/samfurr/nanochat.git /workspace/nanochat && cd /workspace/nanochat && git rev-parse HEAD && bash -n runs/statehead_cuda_preflight.sh'
ssh -i /Users/haybales/.runpod/ssh/runpodctl-ssh-key -p 22185 -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@69.30.85.162 'cd /workspace/nanochat && NANOCHAT_REPO_COMMIT=2944ed65dfb26809073e7b3446ff6255513c83d4 NPROC_PER_NODE=8 DEVICE_BATCH_SIZE=32 PREFLIGHT_STEPS=2 RESULTS_DIR=/workspace/statehead-8gpu-results bash runs/statehead_cuda_preflight.sh'
scp -i /Users/haybales/.runpod/ssh/runpodctl-ssh-key -P 22185 -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@69.30.85.162:/workspace/statehead-8gpu-results/statehead-d12-b32-w8.json /private/tmp/statehead-d12-b32-w8.json
scp -i /Users/haybales/.runpod/ssh/runpodctl-ssh-key -P 22185 -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@69.30.85.162:/workspace/statehead-8gpu-results/statehead-d12-b32-w8.log /private/tmp/statehead-d12-b32-w8.log
jq -e '.repo_commit == "2944ed65dfb26809073e7b3446ff6255513c83d4" and .world_size == 8 and .compiled == true and .config.device_batch_size == 32 and (.steps | length) == 2 and .parameter_checksum_spread == 0' /private/tmp/statehead-d12-b32-w8.json
shasum -a 256 /private/tmp/statehead-d12-b32-w8.json /private/tmp/statehead-d12-b32-w8.log
wc -c /private/tmp/statehead-d12-b32-w8.json /private/tmp/statehead-d12-b32-w8.log
runpodctl pod delete v1n834ijpcebmu
runpodctl pod list --all
runpodctl user
sed -n '1,240p' dev/experiments/statehead-nanochat-cuda-8gpu-preflight-v1.yaml
sed -n '1,280p' dev/experiments/statehead-nanochat-d12-controlled-v1.yaml
tail -n 260 dev/STATEHEAD_LOG.md
sed -n '1,260p' /private/tmp/statehead-d12-b32-w8.json
sed -n '1,260p' /private/tmp/statehead-d12-b32-w8.log
awk 'BEGIN { printf "cost_delta=%.10f\\npeak_allocated_gib=%.6f\\npeak_reserved_gib=%.6f\\n", 208.0753390127-207.1924766794, 43207041024/1073741824, 47355789312/1073741824 }'
sed -n '1,260p' dev/experiments/statehead-nanochat-cuda-batch32-preflight-v1.yaml && sed -n '1,220p' dev/experiments/statehead-nanochat-cuda-preflight-v1.yaml
```

The controlled dataset-backed manifest is now gated only on a separate paid-run
approval. It was not launched by this preflight.

Post-retrieval local verification:

```text
result JSON byte-for-byte SHA-256 match: passed
eight-GPU result and both dependent manifests: consistent
focused StateHead suite: 42 passed in 2.33s
full suite excluding known macOS memory-limit test: 85 passed, 14 skipped, 1 deselected in 4.77s
bash syntax, compileall, diff check, and unchanged nanochat/gpt.py assertion: passed
```

```bash
git status --short && git diff --stat && git diff --check
shasum -a 256 dev/results/statehead-nanochat-cuda-8gpu-preflight-v1.json /private/tmp/statehead-d12-b32-w8.json
.venv/bin/python -c '<assert artifact byte match and manifests consistent>'
NANOCHAT_DTYPE=float32 .venv/bin/python -m pytest tests/test_statehead.py -q
NANOCHAT_DTYPE=float32 .venv/bin/python -m pytest -q -k 'not test_memory_limit'
bash -n runs/statehead_cuda_preflight.sh && .venv/bin/python -m compileall -q nanochat scripts tests dev/statehead_cuda_preflight.py && git diff --check && git diff --quiet -- nanochat/gpt.py
git diff -- dev/STATEHEAD_LOG.md dev/experiments/statehead-nanochat-cuda-8gpu-preflight-v1.yaml dev/experiments/statehead-nanochat-d12-controlled-v1.yaml dev/results/statehead-nanochat-cuda-8gpu-preflight-v1.json
sed -n '675,735p' dev/STATEHEAD_LOG.md
tail -n 80 dev/STATEHEAD_LOG.md
rg -n '^## 2026-07-22 — eight-H100|^### Next gated probe|^## 2026-07-22 — one-H100 device-batch-32|^$' dev/STATEHEAD_LOG.md | tail -n 20
rg -n '^## 2026-07-22 — eight-H100|^### Next gated probe|^## 2026-07-22 — one-H100 device-batch-32' dev/STATEHEAD_LOG.md
git status --short
git add dev/STATEHEAD_LOG.md dev/experiments/statehead-nanochat-cuda-8gpu-preflight-v1.yaml dev/experiments/statehead-nanochat-d12-controlled-v1.yaml dev/results/statehead-nanochat-cuda-8gpu-preflight-v1.json
git diff --cached --check && git diff --cached --stat
git commit -m 'Record eight-H100 StateHead DDP preflight'
git push origin codex/statehead-nanochat
git rev-parse HEAD && git status --short
```

### Device-batch capacity gate (subsequently completed)

The controlled 8-GPU global batch is exactly `32 × 2,048 × 8 = 524,288`
tokens. The successful device-batch-1 probe did not establish that device batch
32 fit, so `dev/experiments/statehead-nanochat-cuda-batch32-preflight-v1.yaml`
defined the separate one-H100, two-step capacity probe documented below.

## 2026-07-22 — one-H100 device-batch-32 capacity probe

The explicitly approved capacity probe ran on one Secure Cloud
`NVIDIA H100 80GB HBM3` pod at `$2.99/hour`, with a 15-minute hard termination
deadline. Batch 32 completed directly, so the planned batch-16 OOM fallback was
not needed. The result and log were retrieved before the pod was manually
deleted. RunPod then reported no pods and `$0/hour`. The immediately observed
account-balance delta was approximately `$0.06`; later billing adjustments may
change that value slightly.

Environment:

```text
model code commit: 2944ed65dfb26809073e7b3446ff6255513c83d4
checked-out branch head: 55da9c146d7fe42c2bb1f1412b91e1cb0fb68142
image: runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404
GPU: NVIDIA H100 80GB HBM3, 81,559 MiB reported
driver: 580.126.09
Python: 3.12.3
PyTorch: 2.9.1+cu128
CUDA runtime: 12.8
precision: BF16 activations, FP32 scan accumulation
world/device batch/sequence: 1 / 32 / 2,048
tokens per step: 65,536
compile: true
synthetic fixed token rows: true
```

Result:

```text
process exit: 0
finite loss and gradients: yes
optimizer steps: 2
step 0, including compilation: 62.84214859455824 s, loss 10.397625923156738
step 1, steady shape: 0.12126892618834972 s, loss 7.433448791503906
steady global throughput: 540,418.7375932749 tokens/s
peak allocated VRAM: 43,420,313,600 bytes (40.438 GiB)
peak reserved VRAM: 47,355,789,312 bytes (44.103 GiB)
parameter checksum spread: 0.0 (trivial one-rank check)
batch-16 fallback: not run
```

This establishes only that the intended device batch fits on one H100. It does
not test NCCL, eight-rank optimizer sharding/synchronization, data loading,
validation BPB, CORE, GPT parity, or end-to-end training throughput. No parity
claim is made.

Artifacts:

- `dev/results/statehead-nanochat-cuda-batch32-preflight-v1.json`
- retrieved raw JSON SHA-256: `3581691c4be67c2fe7e5ff3dfdb36c98e975ac4f51c66b822d229ef5ad0d2554`
- retrieved raw log SHA-256: `86fe7b6f12f74ace7effd1143842160e7724f7ef9d753c1fee58824dbbda40ee`

### Batch-32 probe command ledger

```bash
sed -n '1,240p' /Users/haybales/.agents/skills/runpod/SKILL.md && sed -n '1,320p' /Users/haybales/.agents/skills/runpodctl/SKILL.md
date -u -v+15M '+%Y-%m-%dT%H:%M:%SZ'
git rev-parse HEAD && git status --short && sed -n '1,180p' dev/experiments/statehead-nanochat-cuda-batch32-preflight-v1.yaml
runpodctl user
runpodctl pod list --all
runpodctl pod create --help
runpodctl pod create --name statehead-cuda-b32-preflight-v1 --image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 --gpu-id "NVIDIA H100 80GB HBM3" --gpu-count 1 --cloud-type SECURE --container-disk-in-gb 30 --ports "22/tcp" --ssh --terminate-after 2026-07-23T02:00:36Z
runpodctl pod get 6gmczse2pceyqy
runpodctl ssh info 6gmczse2pceyqy
runpodctl ssh info 6gmczse2pceyqy
runpodctl pod get 6gmczse2pceyqy
ssh -i /Users/haybales/.runpod/ssh/runpodctl-ssh-key -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -p 12593 root@103.207.149.112 'hostname; nvidia-smi -L; python --version; python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"'
ssh -i /Users/haybales/.runpod/ssh/runpodctl-ssh-key -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 12593 root@103.207.149.112 'git clone --branch codex/statehead-nanochat --depth 1 https://github.com/samfurr/nanochat.git /workspace/nanochat'
ssh -i /Users/haybales/.runpod/ssh/runpodctl-ssh-key -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 12593 root@103.207.149.112 'cd /workspace/nanochat && git rev-parse HEAD && git status --short && bash -n runs/statehead_cuda_preflight.sh'
ssh -i /Users/haybales/.runpod/ssh/runpodctl-ssh-key -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 12593 root@103.207.149.112 'cd /workspace/nanochat && NANOCHAT_REPO_COMMIT=2944ed65dfb26809073e7b3446ff6255513c83d4 NPROC_PER_NODE=1 DEVICE_BATCH_SIZE=32 PREFLIGHT_STEPS=2 RESULTS_DIR=/workspace/statehead-batch32-results bash runs/statehead_cuda_preflight.sh'
scp -i /Users/haybales/.runpod/ssh/runpodctl-ssh-key -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P 12593 root@103.207.149.112:/workspace/statehead-batch32-results/statehead-d12-b32-w1.json /private/tmp/statehead-d12-b32-w1.json
scp -i /Users/haybales/.runpod/ssh/runpodctl-ssh-key -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P 12593 root@103.207.149.112:/workspace/statehead-batch32-results/statehead-d12-b32-w1.log /private/tmp/statehead-d12-b32-w1.log
.venv/bin/python -c '<validate retrieved batch-32 JSON fields and print key metrics>' && shasum -a 256 /private/tmp/statehead-d12-b32-w1.json /private/tmp/statehead-d12-b32-w1.log && wc -c /private/tmp/statehead-d12-b32-w1.json /private/tmp/statehead-d12-b32-w1.log
runpodctl pod delete 6gmczse2pceyqy
runpodctl pod list --all
runpodctl user
```

`dev/experiments/statehead-nanochat-cuda-8gpu-preflight-v1.yaml` was the
subsequent gated experiment. Its completed NCCL and distributed Muon/AdamW
result is documented above; no dataset-backed run was launched.

Post-retrieval local verification:

```text
result JSON byte-for-byte SHA-256 match: passed
batch-32 result and all three dependent manifests: consistent
focused StateHead suite: 42 passed in 2.31s
full suite excluding known macOS memory-limit test: 85 passed, 14 skipped, 1 deselected in 4.84s
diff check, compileall, and unchanged nanochat/gpt.py assertion: passed
```

```bash
.venv/bin/python -c '<assert batch-32 result and dependent manifests are consistent>' && shasum -a 256 dev/results/statehead-nanochat-cuda-batch32-preflight-v1.json /private/tmp/statehead-d12-b32-w1.json
NANOCHAT_DTYPE=float32 .venv/bin/python -m pytest tests/test_statehead.py -q
NANOCHAT_DTYPE=float32 .venv/bin/python -m pytest -q -k 'not test_memory_limit'
git diff --check && .venv/bin/python -m compileall -q nanochat scripts tests dev/statehead_cuda_preflight.py && git diff --quiet -- nanochat/gpt.py && git status --short
```

## 2026-07-23 — Vast eight-H100 controlled d12 pair

The approved Phase 3 dataset-backed comparison completed on Vast.ai instance
`45630532` using one node with eight NVIDIA H100 80GB HBM3 GPUs. StateHead ran
first, followed by a newly approved matched GPT control on the same node,
tokenizer, exact shard set, seed, data order, 2,520 steps, and
1,321,205,760-token budget. Both used BF16 and `torch.compile`; StateHead used
the explicit PyTorch parallel scan, while GPT retained its checked-in Flash
Attention 3 path. FP8 was disabled.

| Model | Params | Scaling params | Trainer time | Median tok/s | Peak VRAM | Held-out val BPB | CORE |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPT d12 | 286,261,730 | 110,100,912 | 324.950 s | 4,053,035 | 28,014.59 MiB | 0.845578 | 0.146506 |
| StateHead d12, same shape | 85,767,218 | 60,555,264 | 304.807 s | 4,321,656 | 41,432.27 MiB | 0.980587 | 0.067406 |

StateHead did **not** demonstrate learning parity. Its BPB was `0.135009`
higher and CORE `0.079100` lower. It was only 6.63% faster by median logged
step throughput despite an estimated training FLOP count 52.16% lower, and its
peak VRAM was 47.90% higher. StateHead beat GPT on three CORE tasks, tied one,
and lost eighteen. No NaN, Inf, compiler fallback, or runtime failure was
logged.

The full-run Vast invoice was `$10.261` for `0.6364842` billed hours,
including disk and bandwidth. The separate preflight invoice was `$1.749`.
Both are below the approved `$50` full-run ceiling. The instance was manually
destroyed after local artifact verification; the automatic three-hour guard
was then cancelled. The final account audit showed zero instances and zero
volumes.

Both final model/metadata/CORE triples matched their downloaded SHA-256 values.
All eight optimizer shards per architecture passed remote SHA-256 verification
before deletion, but the optimizer payloads were not downloaded. The complete
1.0 GiB local bundle, including the two final model checkpoints, is at:

```text
dev-ignore/statehead-vast-d12-controlled-20260723/controlled-results/
```

Tracked lightweight evidence and the full per-task/command report are at:

```text
dev/results/statehead-vast-d12-controlled-20260723/
```

The initial shallow remote clone did not contain the pinned model commit and
failed before setup, data preparation, or training. After `git fetch
--unshallow`, the pinned model-code diff was reverified as empty and the runner
was relaunched successfully.

This run answers the same-shape question only. A depth-12, width-1,152
StateHead has 117,374,976 scaling parameters, 6.61% above GPT's 110,100,912,
and is the first reasonable parameter-matched candidate. Do not spend multiple
same-shape seeds or advance to SFT from this result. First create a separately
named parameter-matched manifest/runner and gate its likely batch-16 shape on
one H100.

See
`dev/results/statehead-vast-d12-controlled-20260723/REPORT.md` for every
retrieved artifact, the complete per-task table, infrastructure provenance,
command ledger, unresolved questions, and the exact next local sizing command.

Final local verification:

```text
focused StateHead suite: 46 passed, 15 skipped in 2.18s
full suite excluding test_memory_limit: 89 passed, 29 skipped,
  1 deselected in 4.62s
StateHead BF16/MPS five-step smoke: loss 5.924298 -> 5.923664
default GPT BF16/MPS five-step smoke: loss 5.924309 -> 5.923669
runner syntax and paired dry run: passed
compileall and diff check: passed
model/training source unchanged from b952753: passed
manifests, exits, markers, shards, metadata, 5,040 finite losses,
  CORE aggregates, and both downloaded checkpoint hashes: passed
```

## 2026-07-23 — StateHead CUDA v2 hierarchical backward gate

Commit `52dd72e184bf1459009e9d99a4819a2634f872aa` replaced the native
CUDA backward's serial all-chunk traversal with parallel reverse chunk
summaries, a short reverse boundary scan, and parallel per-chunk gradient
replay. The existing forward, gate/checkpoint layout, Python API, and GPT
implementation remained unchanged.

One verified Vast H100 SXM compiled the extension with PyTorch 2.9.1+cu128 and
CUDA 12.8. The FP32 StateHead suite passed 89 tests, and the focused BF16
native-CUDA/reverse-summary gate passed 45. At the compiled
d12/768/2048/batch-32 fixed-synthetic-batch full-step shape:

| Mode | Median tok/s | Peak allocated |
|---|---:|---:|
| StateHead CUDA v2, chunk 32 | 824,653 | 13.15 GiB |
| StateHead CUDA v2, chunk 64 | 772,758 | 13.12 GiB |
| StateHead compiled PyTorch, chunk 32 | 615,599 | 36.42 GiB |
| GPT/FA3 | 530,591 | 27.40 GiB |

Thus chunk 32 was 6.72% faster than chunk 64, 33.96% faster than the
same-host compiled PyTorch StateHead control, and 55.42% faster than the
same-host GPT/FA3 control. These are bounded synthetic speed results, not
dataset learning parity. Chunk 16, Nsight profiling, FP8, and dataset-backed
learning parity remain open.

Vast instance `45636045` ran for approximately 23.17 minutes at an observed
$3.02/hour, for an estimated $1.17 before deletion. Deletion and zero remaining
instances were confirmed.

Full evidence, limitations, checksums, commands, and the exact next command:

```text
dev/results/statehead-cuda-v2-gate-20260723/REPORT.md
```
