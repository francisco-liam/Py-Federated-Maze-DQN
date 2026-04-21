#!/usr/bin/env bash
# run_all.sh — run every experiment needed for the paper in order.
#
# Usage:
#   bash run_all.sh            # run everything
#   bash run_all.sh --skip-fl  # skip FL runs (use existing checkpoints)
#
# Logs for each run are saved to logs/<name>.log in addition to stdout.
# If any step fails the script stops immediately (set -e).

set -euo pipefail

PYTHON=".venv/bin/python"
LOG_DIR="logs"
mkdir -p "$LOG_DIR" checkpoints

SKIP_FL=false
for arg in "$@"; do
  [[ "$arg" == "--skip-fl" ]] && SKIP_FL=true
done

run_step() {
  local name="$1"
  shift
  echo ""
  echo "════════════════════════════════════════════════════════════════════"
  echo "  STEP: $name"
  echo "  CMD : $*"
  echo "  LOG : $LOG_DIR/$name.log"
  echo "════════════════════════════════════════════════════════════════════"
  "$@" 2>&1 | tee "$LOG_DIR/$name.log"
  echo "  ✓ $name complete"
}

# ── 1. Local-only baseline (lower bound) ────────────────────────────────────
# 3 independent agents, no sharing. ~1500 eps each.
run_step "local_only" $PYTHON train_local_only.py

# ── 2. Centralized DQN (upper bound) ────────────────────────────────────────
# 1 agent on all 3 seeds pooled. Matched compute budget to FL.
run_step "centralized" $PYTHON train_centralized.py

# ── 3. FedAvg — all layers (historical failure, for comparison table) ────────
if [[ "$SKIP_FL" == false ]]; then
  run_step "fedavg_all_layers" $PYTHON train_federated.py
fi

# ── 4. FedAvg — partial conv sharing, equal weight (our fix) ────────────────
if [[ "$SKIP_FL" == false ]]; then
  run_step "fedavg_partial_conv" $PYTHON train_federated.py --equal_weight
fi

# ── 5. FedProx — partial conv sharing (mu=0.1) ──────────────────────────────
if [[ "$SKIP_FL" == false ]]; then
  run_step "fedprox_partial_conv" $PYTHON train_federated.py --equal_weight --mu 0.1
fi

# ── 6. Transfer test — FL conv vs random init on held-out seed=7 ────────────
# Uses fl_fedavg_eqw_final.pt (step 4 output).
run_step "transfer_test_fedavg" $PYTHON transfer_test.py \
  --checkpoint checkpoints/fl_fedavg_eqw_final.pt \
  --finetune 200 --eval 50

# Also test FedProx checkpoint if it exists
if [[ -f "checkpoints/fl_fedprox_mu0.1_eqw_final.pt" ]]; then
  run_step "transfer_test_fedprox" $PYTHON transfer_test.py \
    --checkpoint checkpoints/fl_fedprox_mu0.1_eqw_final.pt \
    --finetune 200 --eval 50
fi

# ── Done ────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  ALL STEPS COMPLETE"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "Summary files:"
for f in checkpoints/summary_*.txt; do
  echo "  $f"
done
echo ""
echo "Transfer test logs:"
for f in logs/transfer_test*.log; do
  echo "  $f"
done
echo ""
echo "To compare results:"
echo "  cat checkpoints/summary_local_only.txt"
echo "  cat checkpoints/summary_centralized.txt"
echo "  cat checkpoints/summary_fedavg_eqw.txt"
echo "  cat checkpoints/summary_fedprox_mu0.1_eqw.txt"
echo "  cat logs/transfer_test_fedavg.log | tail -20"
