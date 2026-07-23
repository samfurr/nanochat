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
