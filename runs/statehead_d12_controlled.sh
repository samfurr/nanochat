#!/usr/bin/env bash

# Paired, dataset-backed d12 GPT/StateHead controlled comparison.
# This script runs on an already-provisioned 8x H100 pod. It does not create or
# delete RunPod infrastructure. Use DRY_RUN=1 to print the exact workload only.

set -euo pipefail

readonly MODEL_CODE_COMMIT="b952753243ea14ac391d617244f2fbd52ba0a487"
readonly WORLD_SIZE=8
readonly DEVICE_BATCH_SIZE=32
readonly SEQUENCE_LENGTH=2048
readonly TOTAL_BATCH_TOKENS=524288
readonly TRAINING_STEPS=2520
readonly EVAL_TOKENS=41943040
readonly SEED=1337
readonly DATASET_TRAIN_SHARDS=170

DRY_RUN="${DRY_RUN:-0}"
SETUP_ENV="${SETUP_ENV:-1}"
PREPARE_DATA="${PREPARE_DATA:-1}"
RUN_ARCHES="${RUN_ARCHES:-gpt,statehead}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-16}"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/workspace/nanochat-controlled}"
RESULTS_DIR="${RESULTS_DIR:-${NANOCHAT_BASE_DIR}/controlled-results}"
WANDB_RUN_PREFIX="${WANDB_RUN_PREFIX:-dummy}"

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

require_boolean() {
    local name="$1"
    local value="$2"
    [[ "$value" == "0" || "$value" == "1" ]] || fail "$name must be 0 or 1 (got '$value')"
}

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

require_boolean DRY_RUN "$DRY_RUN"
require_boolean SETUP_ENV "$SETUP_ENV"
require_boolean PREPARE_DATA "$PREPARE_DATA"
[[ "$NPROC_PER_NODE" == "$WORLD_SIZE" ]] || fail "NPROC_PER_NODE must be exactly $WORLD_SIZE"
[[ "$DOWNLOAD_WORKERS" =~ ^[1-9][0-9]*$ ]] || fail "DOWNLOAD_WORKERS must be a positive integer"
[[ -f pyproject.toml && -d nanochat && -d scripts ]] || fail "run from the nanochat repository root"

git cat-file -e "${MODEL_CODE_COMMIT}^{commit}"
model_paths=(
    nanochat/gpt.py
    nanochat/statehead.py
    nanochat/optim.py
    nanochat/dataloader.py
    nanochat/checkpoint_manager.py
    scripts/base_train.py
    scripts/base_eval.py
)
if ! git diff --quiet "$MODEL_CODE_COMMIT" -- "${model_paths[@]}"; then
    fail "model/training code differs from pinned commit $MODEL_CODE_COMMIT"
fi

IFS=',' read -r -a arches <<< "$RUN_ARCHES"
[[ "${#arches[@]}" -gt 0 ]] || fail "RUN_ARCHES must select gpt, statehead, or both"
seen_gpt=0
seen_statehead=0
for arch in "${arches[@]}"; do
    case "$arch" in
        gpt)
            [[ "$seen_gpt" == 0 ]] || fail "RUN_ARCHES contains duplicate gpt"
            seen_gpt=1
            ;;
        statehead)
            [[ "$seen_statehead" == 0 ]] || fail "RUN_ARCHES contains duplicate statehead"
            seen_statehead=1
            ;;
        *) fail "unsupported architecture '$arch'" ;;
    esac
done

echo "Controlled d12 comparison"
echo "  model code commit: $MODEL_CODE_COMMIT"
echo "  checkout head: $(git rev-parse HEAD)"
echo "  architectures: $RUN_ARCHES"
echo "  world/device batch/sequence: $WORLD_SIZE/$DEVICE_BATCH_SIZE/$SEQUENCE_LENGTH"
echo "  global tokens per step: $TOTAL_BATCH_TOKENS"
echo "  steps/total tokens: $TRAINING_STEPS/$((TRAINING_STEPS * TOTAL_BATCH_TOKENS))"
echo "  precision/FP8: bfloat16/false"
echo "  base directory: $NANOCHAT_BASE_DIR"
echo "  results directory: $RESULTS_DIR"

build_commands() {
    local arch="$1"
    local tag
    if [[ "$arch" == "gpt" ]]; then
        tag="gpt-d12-controlled-v1"
    else
        tag="statehead-nanochat-d12-controlled-v1"
    fi

    local wandb_run="$WANDB_RUN_PREFIX"
    if [[ "$WANDB_RUN_PREFIX" != "dummy" ]]; then
        wandb_run="${WANDB_RUN_PREFIX}-${tag}"
    fi

    TRAIN_COMMAND=(
        torchrun --standalone "--nproc_per_node=$WORLD_SIZE" -m scripts.base_train --
        "--arch=$arch"
        --device-type=cuda
        --depth=12
        --aspect-ratio=64
        --head-dim=128
        "--max-seq-len=$SEQUENCE_LENGTH"
        --window-pattern=SSSL
        "--num-iterations=$TRAINING_STEPS"
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
        --eval-every=250
        "--eval-tokens=$EVAL_TOKENS"
        --core-metric-every=-1
        --sample-every=-1
        --save-every=-1
        "--model-tag=$tag"
        "--run=$wandb_run"
    )
    if [[ "$arch" == "statehead" ]]; then
        # Keep the original Phase 3 scientific comparison on the explicit
        # PyTorch/BF16 reference path. Native CUDA/FP8 has its own Phase 5 gate.
        TRAIN_COMMAND+=(--statehead-scan-backend=pytorch)
    fi
    EVAL_COMMAND=(
        torchrun --standalone "--nproc_per_node=$WORLD_SIZE" -m scripts.base_eval --
        --device-type=cuda
        --eval=core,bpb
        "--model-tag=$tag"
        "--step=$TRAINING_STEPS"
        --max-per-task=-1
        "--device-batch-size=$DEVICE_BATCH_SIZE"
        "--split-tokens=$EVAL_TOKENS"
    )
    RUN_TAG="$tag"
}

if [[ "$DRY_RUN" == "1" ]]; then
    echo "Dry run; no environment setup, download, training, or evaluation will execute."
    if [[ "$SETUP_ENV" == "1" ]]; then
        print_command uv sync --extra gpu --frozen
    fi
    if [[ "$PREPARE_DATA" == "1" ]]; then
        print_command python -m nanochat.dataset -n 8 -w "$DOWNLOAD_WORKERS"
        print_command python -m nanochat.dataset -n "$DATASET_TRAIN_SHARDS" -w "$DOWNLOAD_WORKERS"
        print_command python -m scripts.tok_train --max-chars=2000000000 --vocab-size=32768
    fi
    for arch in "${arches[@]}"; do
        build_commands "$arch"
        echo "$RUN_TAG training:"
        print_command env NANOCHAT_DTYPE=bfloat16 "${TRAIN_COMMAND[@]}"
        echo "$RUN_TAG evaluation:"
        print_command env NANOCHAT_DTYPE=bfloat16 "${EVAL_COMMAND[@]}"
    done
    exit 0
fi

mkdir -p "$RESULTS_DIR"
DATASET_DOWNLOAD_PID=""
on_exit() {
    local status=$?
    trap - EXIT
    if [[ -n "$DATASET_DOWNLOAD_PID" ]] && kill -0 "$DATASET_DOWNLOAD_PID" 2>/dev/null; then
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
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee "$RESULTS_DIR/nvidia-smi.csv"

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
    echo "run_arches=$RUN_ARCHES"
} | tee "$RESULTS_DIR/environment.txt"

if [[ "$PREPARE_DATA" == "1" ]]; then
    python -m nanochat.dataset -n 8 -w "$DOWNLOAD_WORKERS" 2>&1 | tee "$RESULTS_DIR/dataset-first-eight.log"
    python -m nanochat.dataset -n "$DATASET_TRAIN_SHARDS" -w "$DOWNLOAD_WORKERS" > "$RESULTS_DIR/dataset-all.log" 2>&1 &
    DATASET_DOWNLOAD_PID=$!
    python -m scripts.tok_train --max-chars=2000000000 --vocab-size=32768 2>&1 | tee "$RESULTS_DIR/tokenizer-train.log"
    wait "$DATASET_DOWNLOAD_PID"
    DATASET_DOWNLOAD_PID=""
fi

EXPECTED_TRAIN_SHARDS="$DATASET_TRAIN_SHARDS" python -c 'import os; from pathlib import Path; from nanochat.dataset import DATA_DIR; count = int(os.environ["EXPECTED_TRAIN_SHARDS"]); expected = {f"shard_{i:05d}.parquet" for i in range(count)} | {"shard_06542.parquet"}; actual = {p.name for p in Path(DATA_DIR).glob("*.parquet")}; assert expected == actual, (len(expected), len(actual), sorted(expected - actual)[:5], sorted(actual - expected)[:5]); print(f"verified {len(actual)} exact dataset shards")' | tee "$RESULTS_DIR/dataset-verification.txt"
sha256sum "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" "$NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt" > "$RESULTS_DIR/tokenizer.sha256"

run_architecture() {
    local arch="$1"
    build_commands "$arch"
    local checkpoint_dir="$NANOCHAT_BASE_DIR/base_checkpoints/$RUN_TAG"
    [[ ! -e "$checkpoint_dir" ]] || fail "checkpoint directory already exists: $checkpoint_dir"

    echo "Starting $RUN_TAG at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    print_command env NANOCHAT_DTYPE=bfloat16 "${TRAIN_COMMAND[@]}"
    local start_seconds=$SECONDS
    NANOCHAT_DTYPE=bfloat16 "${TRAIN_COMMAND[@]}" 2>&1 | tee "$RESULTS_DIR/${RUN_TAG}-train.log"
    local train_seconds=$((SECONDS - start_seconds))
    echo "$train_seconds" > "$RESULTS_DIR/${RUN_TAG}-train-wall-seconds.txt"

    [[ -f "$checkpoint_dir/model_002520.pt" ]] || fail "$RUN_TAG final model checkpoint missing"
    [[ -f "$checkpoint_dir/meta_002520.json" ]] || fail "$RUN_TAG final metadata missing"

    echo "Evaluating $RUN_TAG at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    print_command env NANOCHAT_DTYPE=bfloat16 "${EVAL_COMMAND[@]}"
    NANOCHAT_DTYPE=bfloat16 "${EVAL_COMMAND[@]}" 2>&1 | tee "$RESULTS_DIR/${RUN_TAG}-eval.log"

    local core_csv="$NANOCHAT_BASE_DIR/base_eval/base_model_002520.csv"
    [[ -f "$core_csv" ]] || fail "$RUN_TAG CORE CSV missing"
    cp "$core_csv" "$RESULTS_DIR/${RUN_TAG}-core.csv"
    cp "$checkpoint_dir/meta_002520.json" "$RESULTS_DIR/${RUN_TAG}-meta_002520.json"
    cp "$checkpoint_dir/model_002520.pt" "$RESULTS_DIR/${RUN_TAG}-model_002520.pt"
    shopt -s nullglob
    local optimizer_shards=("$checkpoint_dir"/optim_002520_rank*.pt)
    shopt -u nullglob
    [[ "${#optimizer_shards[@]}" == "$WORLD_SIZE" ]] || fail "$RUN_TAG expected $WORLD_SIZE optimizer shards, found ${#optimizer_shards[@]}"
    sha256sum \
        "$RESULTS_DIR/${RUN_TAG}-core.csv" \
        "$RESULTS_DIR/${RUN_TAG}-meta_002520.json" \
        "$RESULTS_DIR/${RUN_TAG}-model_002520.pt" \
        "${optimizer_shards[@]}" \
        > "$RESULTS_DIR/${RUN_TAG}.sha256"
    echo "Completed $RUN_TAG at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
}

for arch in "${arches[@]}"; do
    run_architecture "$arch"
done

touch "$RESULTS_DIR/COMPLETED"
find "$RESULTS_DIR" -maxdepth 1 -type f -print | sort
