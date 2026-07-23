#!/usr/bin/env bash
set -euo pipefail

# Infrastructure-only probe. This does not download the training dataset and does
# not produce a learning-quality result.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_DTYPE=bfloat16

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-1}"
PREFLIGHT_STEPS="${PREFLIGHT_STEPS:-2}"
RESULTS_DIR="${RESULTS_DIR:-/workspace/statehead-preflight-results}"
NANOCHAT_REPO_COMMIT="${NANOCHAT_REPO_COMMIT:-unknown}"
export NANOCHAT_REPO_COMMIT

mkdir -p "$RESULTS_DIR"

python -c 'import filelock, torch; print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()} devices={torch.cuda.device_count()}")'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

python -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" --module \
  dev.statehead_cuda_preflight \
  --device-batch-size="$DEVICE_BATCH_SIZE" \
  --steps="$PREFLIGHT_STEPS" \
  --output="$RESULTS_DIR/statehead-d12-b${DEVICE_BATCH_SIZE}-w${NPROC_PER_NODE}.json" \
  2>&1 | tee "$RESULTS_DIR/statehead-d12-b${DEVICE_BATCH_SIZE}-w${NPROC_PER_NODE}.log"
