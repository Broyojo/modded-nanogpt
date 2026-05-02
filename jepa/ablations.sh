#!/usr/bin/env bash
# Run the EMA / NoPE / lag4 ablations sequentially on the 51M baseline.
# Each takes ~25 min on a single 4090; total ~75 min.
# Designed to be launched as a single background task after the 12L/768D
# capacity-scaling run finishes.
set -euo pipefail
cd "$(dirname "$0")/.."

LOG_DIR=/tmp
date

echo "=== ablation 1/3: EMA target encoder (config=ema) ==="
uv run python jepa/train_jepa.py --config=ema --name=fineweb-ema-baseline 2>&1 | tee "$LOG_DIR/fineweb_ema.log"
date

echo "=== ablation 2/3: NoPE (config=nope) ==="
uv run python jepa/train_jepa.py --config=nope --name=fineweb-nope-baseline 2>&1 | tee "$LOG_DIR/fineweb_nope.log"
date

echo "=== ablation 3/3: prediction lag k=4 (config=lag4) ==="
uv run python jepa/train_jepa.py --config=lag4 --name=fineweb-lag4-baseline 2>&1 | tee "$LOG_DIR/fineweb_lag4.log"
date

echo "=== all ablations finished ==="
