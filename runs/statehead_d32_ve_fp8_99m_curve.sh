#!/usr/bin/env bash

# StateHead-only d32 FP8 run with GPT-style alternating value embeddings.
# The measured trainer budget is 99 minutes. Model-only checkpoints are saved
# at calibrated 10-minute marks and every checkpoint receives full CORE + BPB
# evaluation after training. The final checkpoint also retains optimizer state.
# Run on one already-provisioned 8x H100 SXM node. This script does not create
# or destroy infrastructure.

set -euo pipefail

readonly MODEL_CODE_COMMIT="0e8e426c33d46ef28a4da1362ec5c7a5a181621a"
readonly WORLD_SIZE=8
readonly DEPTH=32
readonly MODEL_WIDTH=2048
readonly STATE_HEADS=16
readonly HEAD_DIM=128
readonly VALUE_EMBED_LAYERS=16
readonly DEVICE_BATCH_SIZE=8
readonly SEQUENCE_LENGTH=2048
readonly TOTAL_BATCH_TOKENS=1048576
readonly SCAN_CHUNK_SIZE=32
readonly TARGET_TRAINING_SECONDS=5940
readonly CHECKPOINT_INTERVAL_SECONDS=600
readonly GATE_STEPS=125
readonly GATE_EVAL_EVERY=125
readonly GATE_EVAL_TOKENS=4194304
readonly FULL_EVAL_EVERY=-1
readonly FULL_EVAL_TOKENS=41943040
readonly SEED=1337
readonly DATASET_TRAIN_SHARDS=170
readonly REQUIRED_FREE_GIB=100

DRY_RUN="${DRY_RUN:-0}"
SETUP_ENV="${SETUP_ENV:-1}"
PREPARE_DATA="${PREPARE_DATA:-1}"
REUSE_CALIBRATION="${REUSE_CALIBRATION:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-16}"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/workspace/nanochat-statehead-d32-ve}"
RESULTS_DIR="${RESULTS_DIR:-${NANOCHAT_BASE_DIR}/results}"
WANDB_RUN="${WANDB_RUN:-dummy}"

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

require_boolean() {
    local name="$1"
    local value="$2"
    [[ "$value" == "0" || "$value" == "1" ]] \
        || fail "$name must be 0 or 1 (got '$value')"
}

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

build_train_command() {
    local model_tag="$1"
    local num_iterations="$2"
    local eval_every="$3"
    local eval_tokens="$4"
    local save_steps="$5"
    TRAIN_COMMAND=(
        torchrun --standalone "--nproc_per_node=$WORLD_SIZE" -m scripts.base_train --
        --arch=statehead
        --device-type=cuda
        "--depth=$DEPTH"
        --aspect-ratio=64
        "--head-dim=$HEAD_DIM"
        "--max-seq-len=$SEQUENCE_LENGTH"
        --statehead-scan-backend=cuda
        "--statehead-scan-chunk-size=$SCAN_CHUNK_SIZE"
        --statehead-value-embeddings
        "--num-iterations=$num_iterations"
        --target-flops=-1
        --target-param-data-ratio=-1
        "--device-batch-size=$DEVICE_BATCH_SIZE"
        "--total-batch-size=$TOTAL_BATCH_TOKENS"
        --embedding-lr=0.3
        --unembedding-lr=0.008
        --weight-decay=0.28
        --matrix-lr=0.02
        --scalar-lr=0.5
        --warmup-steps=40
        --warmdown-ratio=0.65
        --final-lr-frac=0.05
        "--seed=$SEED"
        "--eval-every=$eval_every"
        "--eval-tokens=$eval_tokens"
        --core-metric-every=-1
        --sample-every=-1
        --save-every=-1
        "--save-steps=$save_steps"
        --save-intermediate-optimizer=0
        "--model-tag=$model_tag"
        --fp8
        --fp8-recipe=tensorwise
        "--run=$WANDB_RUN"
    )
}

require_boolean DRY_RUN "$DRY_RUN"
require_boolean SETUP_ENV "$SETUP_ENV"
require_boolean PREPARE_DATA "$PREPARE_DATA"
require_boolean REUSE_CALIBRATION "$REUSE_CALIBRATION"
[[ "$NPROC_PER_NODE" == "$WORLD_SIZE" ]] \
    || fail "NPROC_PER_NODE must be exactly $WORLD_SIZE"
[[ "$DOWNLOAD_WORKERS" =~ ^[1-9][0-9]*$ ]] \
    || fail "DOWNLOAD_WORKERS must be a positive integer"
[[ -f pyproject.toml && -d nanochat && -d scripts ]] \
    || fail "run from the nanochat repository root"

git cat-file -e "${MODEL_CODE_COMMIT}^{commit}"
model_paths=(
    nanochat/gpt.py
    nanochat/statehead.py
    nanochat/statehead_cuda.py
    nanochat/csrc/statehead_cuda.cpp
    nanochat/csrc/statehead_cuda_kernel.cu
    nanochat/fp8.py
    nanochat/optim.py
    nanochat/dataloader.py
    nanochat/checkpoint_manager.py
    scripts/base_train.py
    scripts/base_eval.py
    dev/statehead_cuda_preflight.py
)
git diff --quiet "$MODEL_CODE_COMMIT" -- "${model_paths[@]}" \
    || fail "model/training code differs from pinned commit $MODEL_CODE_COMMIT"

echo "StateHead d32 + 16 GPT-style value embeddings, FP8, 99-minute curve"
echo "  model code pin: $MODEL_CODE_COMMIT"
echo "  checkout head: $(git rev-parse HEAD)"
echo "  depth/width/heads/head_dim: $DEPTH/$MODEL_WIDTH/$STATE_HEADS/$HEAD_DIM"
echo "  value embedding layers: odd layers 1..31 ($VALUE_EMBED_LAYERS banks)"
echo "  world/device batch/sequence: $WORLD_SIZE/$DEVICE_BATCH_SIZE/$SEQUENCE_LENGTH"
echo "  global batch tokens: $TOTAL_BATCH_TOKENS"
echo "  precision: tensorwise FP8 projections, BF16 activations, FP32 recurrence"
echo "  CUDA scan chunk size: $SCAN_CHUNK_SIZE"
echo "  target measured training seconds: $TARGET_TRAINING_SECONDS"
echo "  checkpoint interval target: $CHECKPOINT_INTERVAL_SECONDS seconds"
echo "  base directory: $NANOCHAT_BASE_DIR"
echo "  results directory: $RESULTS_DIR"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "Dry run; no setup, download, CUDA work, training, or evaluation will execute."
    if [[ "$SETUP_ENV" == "1" ]]; then
        print_command uv sync --extra gpu --frozen
    fi
    if [[ "$PREPARE_DATA" == "1" ]]; then
        print_command python -m nanochat.dataset -n 8 -w "$DOWNLOAD_WORKERS"
        print_command python -m nanochat.dataset -n "$DATASET_TRAIN_SHARDS" -w "$DOWNLOAD_WORKERS"
        print_command python -m scripts.tok_train --max-chars=2000000000 --vocab-size=32768
    fi
    print_command env NANOCHAT_DTYPE=bfloat16 \
        python -m torch.distributed.run --standalone \
        "--nproc_per_node=$WORLD_SIZE" --module dev.statehead_cuda_preflight \
        --arch=statehead "--device-batch-size=$DEVICE_BATCH_SIZE" \
        --steps=4 --warmup-steps=1 "--layers=$DEPTH" \
        "--model-width=$MODEL_WIDTH" "--heads=$STATE_HEADS" \
        "--sequence-length=$SEQUENCE_LENGTH" \
        "--scan-chunk-size=$SCAN_CHUNK_SIZE" --scan-backend=cuda \
        --value-embeddings --fp8 --verify-gradients \
        --output="$RESULTS_DIR/statehead-d32-ve-fp8-ddp-gate.json"
    build_train_command statehead-d32-ve-fp8-calibration "$GATE_STEPS" \
        "$GATE_EVAL_EVERY" "$GATE_EVAL_TOKENS" ""
    echo "Dataset calibration/learning gate:"
    print_command env NANOCHAT_DTYPE=bfloat16 "${TRAIN_COMMAND[@]}"
    build_train_command statehead-d32-ve-fp8-99m CALIBRATED_STEPS \
        "$FULL_EVAL_EVERY" "$FULL_EVAL_TOKENS" CALIBRATED_CHECKPOINT_STEPS
    echo "Calibrated full run:"
    print_command env NANOCHAT_DTYPE=bfloat16 "${TRAIN_COMMAND[@]}"
    echo "Then run full CORE + 40 Mi-token train/val BPB at all 10 checkpoints."
    exit 0
fi

mkdir -p "$RESULTS_DIR"
DATASET_DOWNLOAD_PID=""
on_exit() {
    local status=$?
    trap - EXIT
    if [[ -n "$DATASET_DOWNLOAD_PID" ]] \
        && kill -0 "$DATASET_DOWNLOAD_PID" 2>/dev/null; then
        kill "$DATASET_DOWNLOAD_PID" 2>/dev/null || true
        wait "$DATASET_DOWNLOAD_PID" 2>/dev/null || true
    fi
    {
        echo "exit_status=$status"
        echo "finished_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    } > "$RESULTS_DIR/run-exit.txt"
    exit "$status"
}
trap on_exit EXIT

python -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 8; names = [torch.cuda.get_device_name(i) for i in range(8)]; assert all("H100" in name for name in names), names; print(torch.__version__, torch.version.cuda, names)'
nvidia-smi --query-gpu=name,memory.total,power.limit,driver_version \
    --format=csv | tee "$RESULTS_DIR/nvidia-smi.csv"

if [[ "$SETUP_ENV" == "1" ]]; then
    if ! command -v uv >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="${XDG_BIN_HOME:-${HOME}/.local/bin}:$PATH"
    fi
    uv sync --extra gpu --frozen 2>&1 | tee "$RESULTS_DIR/uv-sync.log"
fi
[[ -x .venv/bin/python ]] || fail ".venv is missing; run with SETUP_ENV=1"
source .venv/bin/activate

{
    echo "started_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "model_code_commit=$MODEL_CODE_COMMIT"
    echo "checkout_head=$(git rev-parse HEAD)"
    echo "checkout_status_begin"
    git status --short
    echo "checkout_status_end"
    echo "python=$(python --version 2>&1)"
    echo "torch=$(python -c 'import torch; print(torch.__version__)')"
    echo "cuda=$(python -c 'import torch; print(torch.version.cuda)')"
} | tee "$RESULTS_DIR/environment.txt"

if [[ "$PREPARE_DATA" == "1" ]]; then
    python -m nanochat.dataset -n 8 -w "$DOWNLOAD_WORKERS" \
        2>&1 | tee "$RESULTS_DIR/dataset-first-eight.log"
    python -m nanochat.dataset -n "$DATASET_TRAIN_SHARDS" \
        -w "$DOWNLOAD_WORKERS" > "$RESULTS_DIR/dataset-all.log" 2>&1 &
    DATASET_DOWNLOAD_PID=$!
    python -m scripts.tok_train --max-chars=2000000000 --vocab-size=32768 \
        2>&1 | tee "$RESULTS_DIR/tokenizer-train.log"
    wait "$DATASET_DOWNLOAD_PID"
    DATASET_DOWNLOAD_PID=""
fi

EXPECTED_TRAIN_SHARDS="$DATASET_TRAIN_SHARDS" python - <<'PY' \
    | tee "$RESULTS_DIR/dataset-verification.txt"
import os
from pathlib import Path
from nanochat.dataset import DATA_DIR

count = int(os.environ["EXPECTED_TRAIN_SHARDS"])
expected = {f"shard_{i:05d}.parquet" for i in range(count)}
expected.add("shard_06542.parquet")
actual = {path.name for path in Path(DATA_DIR).glob("*.parquet")}
assert expected == actual, (
    len(expected),
    len(actual),
    sorted(expected - actual)[:5],
    sorted(actual - expected)[:5],
)
print(f"verified {len(actual)} exact dataset shards")
PY
sha256sum "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" \
    "$NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt" \
    > "$RESULTS_DIR/tokenizer.sha256"

available_kib="$(df -Pk "$NANOCHAT_BASE_DIR" | awk 'NR == 2 {print $4}')"
required_kib="$((REQUIRED_FREE_GIB * 1024 * 1024))"
(( available_kib >= required_kib )) \
    || fail "need at least ${REQUIRED_FREE_GIB} GiB free, found $((available_kib / 1024 / 1024)) GiB"

GATE_TAG="statehead-d32-ve-fp8-calibration"
GATE_CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$GATE_TAG"
printf -v GATE_STEP_PADDED "%06d" "$GATE_STEPS"
GATE_META="$GATE_CHECKPOINT_DIR/meta_${GATE_STEP_PADDED}.json"
GATE_LOG="$RESULTS_DIR/${GATE_TAG}-train.log"
PREFLIGHT_JSON="$RESULTS_DIR/statehead-d32-ve-fp8-ddp-gate.json"

if [[ "$REUSE_CALIBRATION" == "0" ]]; then
    echo "Starting production-shape FP8/DDP finite-gradient gate."
    NANOCHAT_DTYPE=bfloat16 NANOCHAT_REPO_COMMIT="$(git rev-parse HEAD)" \
        python -m torch.distributed.run --standalone \
        "--nproc_per_node=$WORLD_SIZE" --module dev.statehead_cuda_preflight \
        --arch=statehead \
        "--device-batch-size=$DEVICE_BATCH_SIZE" \
        --steps=4 \
        --warmup-steps=1 \
        "--layers=$DEPTH" \
        "--model-width=$MODEL_WIDTH" \
        "--heads=$STATE_HEADS" \
        "--sequence-length=$SEQUENCE_LENGTH" \
        "--scan-chunk-size=$SCAN_CHUNK_SIZE" \
        --scan-backend=cuda \
        --value-embeddings \
        --fp8 \
        --verify-gradients \
        --output="$PREFLIGHT_JSON" \
        2>&1 | tee "$RESULTS_DIR/statehead-d32-ve-fp8-ddp-gate.log"

    [[ ! -e "$GATE_CHECKPOINT_DIR" ]] \
        || fail "gate checkpoint directory already exists: $GATE_CHECKPOINT_DIR"
    build_train_command "$GATE_TAG" "$GATE_STEPS" \
        "$GATE_EVAL_EVERY" "$GATE_EVAL_TOKENS" ""
    echo "Starting dataset calibration/learning gate."
    print_command env NANOCHAT_DTYPE=bfloat16 "${TRAIN_COMMAND[@]}"
    NANOCHAT_DTYPE=bfloat16 "${TRAIN_COMMAND[@]}" 2>&1 | tee "$GATE_LOG"
else
    echo "Reusing completed production-shape preflight and calibration evidence."
fi

[[ -f "$PREFLIGHT_JSON" ]] || fail "preflight result missing: $PREFLIGHT_JSON"
PREFLIGHT_JSON="$PREFLIGHT_JSON" python - <<'PY'
import json
import os

with open(os.environ["PREFLIGHT_JSON"], encoding="utf-8") as handle:
    result = json.load(handle)
config = result["config"]
counts = result["parameter_counts"]
assert result["architecture"] == "statehead"
assert result["world_size"] == 8
assert result["fp8"] is True
assert result["fp8_linear_count"] == 65
assert result["gradient_finiteness_checked"] is True
assert result["parameter_checksum_spread"] == 0.0
assert config["layers"] == 32
assert config["model_width"] == 2048
assert config["heads"] == 16
assert config["device_batch_size"] == 8
assert config["sequence_length"] == 2048
assert config["scan_chunk_size"] == 32
assert config["scan_backend"] == "cuda"
assert config["value_embeddings"] is True
assert counts["value_embeds"] == 1_073_741_824
assert counts["total"] == 1_879_379_034
print("verified exact d32 value-embedding FP8/DDP preflight evidence")
PY

[[ -f "$GATE_META" ]] || fail "gate metadata missing: $GATE_META"
[[ -f "$GATE_LOG" ]] || fail "gate log missing: $GATE_LOG"
cp "$GATE_META" "$RESULTS_DIR/${GATE_TAG}-meta_${GATE_STEP_PADDED}.json"

CALIBRATION_OUTPUT="$RESULTS_DIR/calibration.json"
SCHEDULE_OUTPUT="$RESULTS_DIR/checkpoint-schedule.json"
GATE_META="$GATE_META" \
GATE_LOG="$GATE_LOG" \
CALIBRATION_TARGET_SECONDS="$TARGET_TRAINING_SECONDS" \
CALIBRATION_INTERVAL_SECONDS="$CHECKPOINT_INTERVAL_SECONDS" \
CALIBRATION_OUTPUT="$CALIBRATION_OUTPUT" \
SCHEDULE_OUTPUT="$SCHEDULE_OUTPUT" python - <<'PY'
import json
import math
import os
import re

with open(os.environ["GATE_META"], encoding="utf-8") as handle:
    meta = json.load(handle)
with open(os.environ["GATE_LOG"], encoding="utf-8") as handle:
    log = handle.read()

validation = [float(value) for value in re.findall(r"Validation bpb: ([0-9.]+)", log)]
losses = [float(value) for value in re.findall(r"\| loss: ([0-9.]+) \|", log)]
if len(validation) < 2 or not all(math.isfinite(value) for value in validation):
    raise SystemExit(f"invalid validation curve: {validation}")
if validation[-1] >= validation[0]:
    raise SystemExit(f"validation BPB did not decrease: {validation}")
if len(losses) < 2 or not all(math.isfinite(value) for value in losses):
    raise SystemExit("training losses are missing or non-finite")
if losses[-1] >= losses[0]:
    raise SystemExit(
        f"training loss did not decrease: first={losses[0]} last={losses[-1]}"
    )

completed_step = int(meta["step"])
timed_steps = completed_step - 11
training_seconds = float(meta["loop_state"]["total_training_time"])
if timed_steps <= 0 or not math.isfinite(training_seconds) or training_seconds <= 0:
    raise SystemExit(
        f"invalid timing: completed_step={completed_step} "
        f"training_seconds={training_seconds}"
    )
seconds_per_step = training_seconds / timed_steps
target_seconds = int(os.environ["CALIBRATION_TARGET_SECONDS"])
interval_seconds = int(os.environ["CALIBRATION_INTERVAL_SECONDS"])
training_steps = round(target_seconds / seconds_per_step) + 11
if not 1000 <= training_steps <= 20000:
    raise SystemExit(
        f"calibrated steps outside safety range: {training_steps} "
        f"from {seconds_per_step:.6f} sec/step"
    )

checkpoint_targets = list(range(interval_seconds, target_seconds, interval_seconds))
checkpoints = [
    {
        "target_training_seconds": seconds,
        "step": round(seconds / seconds_per_step) + 11,
        "kind": "interval",
    }
    for seconds in checkpoint_targets
]
checkpoints.append({
    "target_training_seconds": target_seconds,
    "step": training_steps,
    "kind": "final",
})
steps = [item["step"] for item in checkpoints]
if len(checkpoints) != 10 or len(set(steps)) != len(steps) or steps != sorted(steps):
    raise SystemExit(f"invalid checkpoint schedule: {checkpoints}")

calibration = {
    "gate_step": completed_step,
    "timed_gate_steps": timed_steps,
    "gate_training_seconds": training_seconds,
    "seconds_per_step": seconds_per_step,
    "target_training_seconds": target_seconds,
    "calibrated_training_steps": training_steps,
    "initial_validation_bpb": validation[0],
    "final_validation_bpb": validation[-1],
    "initial_training_loss": losses[0],
    "final_training_loss": losses[-1],
}
schedule = {
    "seconds_per_step_from_calibration": seconds_per_step,
    "target_training_seconds": target_seconds,
    "checkpoint_interval_seconds": interval_seconds,
    "checkpoints": checkpoints,
}
with open(os.environ["CALIBRATION_OUTPUT"], "w", encoding="utf-8") as handle:
    json.dump(calibration, handle, indent=2)
    handle.write("\n")
with open(os.environ["SCHEDULE_OUTPUT"], "w", encoding="utf-8") as handle:
    json.dump(schedule, handle, indent=2)
    handle.write("\n")
print(json.dumps(calibration, indent=2))
print(json.dumps(schedule, indent=2))
PY

TRAINING_STEPS="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["calibrated_training_steps"])' "$CALIBRATION_OUTPUT")"
SAVE_STEPS="$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(",".join(str(x["step"]) for x in d["checkpoints"] if x["kind"] == "interval"))' "$SCHEDULE_OUTPUT")"
[[ "$TRAINING_STEPS" =~ ^[0-9]+$ ]] \
    || fail "invalid calibrated training steps: $TRAINING_STEPS"
[[ "$SAVE_STEPS" =~ ^[0-9]+(,[0-9]+){8}$ ]] \
    || fail "invalid calibrated checkpoint steps: $SAVE_STEPS"
printf -v FINAL_STEP_PADDED "%06d" "$TRAINING_STEPS"

FULL_TAG="statehead-d32-ve-fp8-99m"
FULL_CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$FULL_TAG"
[[ ! -e "$FULL_CHECKPOINT_DIR" ]] \
    || fail "full checkpoint directory already exists: $FULL_CHECKPOINT_DIR"
build_train_command "$FULL_TAG" "$TRAINING_STEPS" \
    "$FULL_EVAL_EVERY" "$FULL_EVAL_TOKENS" "$SAVE_STEPS"
echo "Starting calibrated full run for $TRAINING_STEPS steps."
echo "Model-only intermediate checkpoint steps: $SAVE_STEPS"
print_command env NANOCHAT_DTYPE=bfloat16 "${TRAIN_COMMAND[@]}"
FULL_START_SECONDS=$SECONDS
NANOCHAT_DTYPE=bfloat16 "${TRAIN_COMMAND[@]}" \
    2>&1 | tee "$RESULTS_DIR/${FULL_TAG}-train.log"
echo "$((SECONDS - FULL_START_SECONDS))" \
    > "$RESULTS_DIR/${FULL_TAG}-train-wall-seconds.txt"

python - "$FULL_CHECKPOINT_DIR" "$SCHEDULE_OUTPUT" <<'PY'
import json
import sys
from pathlib import Path

checkpoint_dir = Path(sys.argv[1])
schedule = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
expected = [int(item["step"]) for item in schedule["checkpoints"]]
models = sorted(int(path.stem.split("_")[1]) for path in checkpoint_dir.glob("model_*.pt"))
metadata = sorted(int(path.stem.split("_")[1]) for path in checkpoint_dir.glob("meta_*.json"))
assert models == expected, (models, expected)
assert metadata == expected, (metadata, expected)
print(f"verified {len(expected)} scheduled model/metadata checkpoints")
PY

echo "Evaluating all calibrated checkpoints with full CORE and BPB."
while read -r checkpoint_step; do
    printf -v checkpoint_padded "%06d" "$checkpoint_step"
    EVAL_COMMAND=(
        torchrun --standalone "--nproc_per_node=$WORLD_SIZE" -m scripts.base_eval --
        --device-type=cuda
        --eval=core,bpb
        "--model-tag=$FULL_TAG"
        "--step=$checkpoint_step"
        --max-per-task=-1
        "--device-batch-size=$DEVICE_BATCH_SIZE"
        "--split-tokens=$FULL_EVAL_TOKENS"
    )
    echo "Evaluating checkpoint step $checkpoint_step."
    print_command env NANOCHAT_DTYPE=bfloat16 "${EVAL_COMMAND[@]}"
    NANOCHAT_DTYPE=bfloat16 "${EVAL_COMMAND[@]}" \
        2>&1 | tee "$RESULTS_DIR/${FULL_TAG}-eval_${checkpoint_padded}.log"

    core_csv="$NANOCHAT_BASE_DIR/base_eval/base_model_${checkpoint_padded}.csv"
    model_file="$FULL_CHECKPOINT_DIR/model_${checkpoint_padded}.pt"
    meta_file="$FULL_CHECKPOINT_DIR/meta_${checkpoint_padded}.json"
    [[ -f "$core_csv" ]] || fail "CORE CSV missing: $core_csv"
    [[ -f "$model_file" ]] || fail "model missing: $model_file"
    [[ -f "$meta_file" ]] || fail "metadata missing: $meta_file"
    cp "$core_csv" "$RESULTS_DIR/${FULL_TAG}-core_${checkpoint_padded}.csv"
    cp "$meta_file" "$RESULTS_DIR/${FULL_TAG}-meta_${checkpoint_padded}.json"
    ln "$model_file" "$RESULTS_DIR/${FULL_TAG}-model_${checkpoint_padded}.pt"
done < <(
    python -c 'import json,sys; print(*[x["step"] for x in json.load(open(sys.argv[1]))["checkpoints"]], sep="\n")' \
        "$SCHEDULE_OUTPUT"
)

python dev/aggregate_statehead_checkpoint_curve.py \
    "--checkpoint-dir=$FULL_CHECKPOINT_DIR" \
    "--results-dir=$RESULTS_DIR" \
    "--tag=$FULL_TAG" \
    "--schedule=$SCHEDULE_OUTPUT" \
    "--output-json=$RESULTS_DIR/${FULL_TAG}-curve.json" \
    "--output-csv=$RESULTS_DIR/${FULL_TAG}-curve.csv" \
    | tee "$RESULTS_DIR/${FULL_TAG}-curve.log"

shopt -s nullglob
optimizer_shards=(
    "$FULL_CHECKPOINT_DIR"/optim_"${FINAL_STEP_PADDED}"_rank*.pt
)
all_optimizer_shards=("$FULL_CHECKPOINT_DIR"/optim_*_rank*.pt)
shopt -u nullglob
[[ "${#optimizer_shards[@]}" == "$WORLD_SIZE" ]] \
    || fail "expected $WORLD_SIZE final optimizer shards, found ${#optimizer_shards[@]}"
[[ "${#all_optimizer_shards[@]}" == "$WORLD_SIZE" ]] \
    || fail "intermediate optimizer shards were unexpectedly written"
for optimizer_shard in "${optimizer_shards[@]}"; do
    ln "$optimizer_shard" \
        "$RESULTS_DIR/${FULL_TAG}-$(basename "$optimizer_shard")"
done

(
    cd "$RESULTS_DIR"
    find . -maxdepth 1 -type f \
        ! -name "${FULL_TAG}.sha256" \
        ! -name "COMPLETED" \
        ! -name "run-exit.txt" \
        -print0 \
        | sort -z \
        | xargs -0 sha256sum \
        > "${FULL_TAG}.sha256"
)

touch "$RESULTS_DIR/COMPLETED"
find "$RESULTS_DIR" -maxdepth 1 -type f -print | sort
