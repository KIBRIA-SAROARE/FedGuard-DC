#!/usr/bin/env bash
# Complete board experiment -> figures.  Usage (inside tmux):  ./run_experiment.sh [full|quick]
# 1) measurements (resumable)  2) live displays with snapshots (separate, after measurements)  3) paper figures
set -euo pipefail
cd "$(dirname "$0")"
PLAN="${1:-full}"
OUT=results_rt
mkdir -p "$OUT"
{
echo "=== $(date) plan=$PLAN ==="
python3 run_full_experiment.py --plan "$PLAN" --out "$OUT"

echo "=== live displays (real time, snapshots saved to $OUT/live) ==="
[ -n "${DISPLAY:-}" ] || export MPLBACKEND=Agg
python3 live_capture.py --out "$OUT" --fold S02_fault_100ms --test A1
python3 live_capture.py --out "$OUT" --fold S02_fault_100ms --test no_attack
python3 live_capture.py --out "$OUT" --fold S02_fault_100ms --test T3
python3 live_capture.py --out "$OUT" --fold FULL --test ctrl --scenario S07_atk_false_ups

echo "=== paper figures ==="
python3 paper_figures.py --root "$OUT"
echo "=== done $(date) ==="
} 2>&1 | tee -a "$OUT/run_experiment_console.log"
