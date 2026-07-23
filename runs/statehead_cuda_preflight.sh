#!/usr/bin/env bash
set -euo pipefail

# Infrastructure-only probe. This does not download the training dataset and does
# not produce a learning-quality result.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_DTYPE=bfloat16

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-1}"
PREFLIGHT_STEPS="${PREFLIGHT_STEPS:-2}"
SCAN_BACKEND="${SCAN_BACKEND:-pytorch}"
FP8="${FP8:-0}"
RESULTS_DIR="${RESULTS_DIR:-/workspace/statehead-preflight-results}"
NANOCHAT_REPO_COMMIT="${NANOCHAT_REPO_COMMIT:-unknown}"
export NANOCHAT_REPO_COMMIT
PREFLIGHT_TAG="${PREFLIGHT_TAG:-statehead-d12-b${DEVICE_BATCH_SIZE}-w${NPROC_PER_NODE}}"

[[ "$SCAN_BACKEND" == "pytorch" || "$SCAN_BACKEND" == "cuda" ]] || {
  echo "SCAN_BACKEND must be pytorch or cuda" >&2
  exit 2
}
[[ "$FP8" == "0" || "$FP8" == "1" ]] || {
  echo "FP8 must be 0 or 1" >&2
  exit 2
}

mkdir -p "$RESULTS_DIR"

python -c 'import filelock, torch; print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()} devices={torch.cuda.device_count()}")'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

probe_args=(
  --device-batch-size="$DEVICE_BATCH_SIZE"
  --steps="$PREFLIGHT_STEPS"
  --scan-backend="$SCAN_BACKEND"
  --output="$RESULTS_DIR/${PREFLIGHT_TAG}.json"
)
if [[ "$FP8" == "1" ]]; then
  probe_args+=(--fp8)
fi

python -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" --module \
  dev.statehead_cuda_preflight "${probe_args[@]}" \
  2>&1 | tee "$RESULTS_DIR/${PREFLIGHT_TAG}.log"
