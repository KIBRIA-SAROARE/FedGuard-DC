#!/usr/bin/env bash
# Full edge experiment. Usage: ./run_all.sh [--realtime]
set -euo pipefail
cd "$(dirname "$0")"
RES=results; mkdir -p "$RES"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1   # one core, as a site monitor would be provisioned
{
echo "== 0. idle baseline ==";                     python3 measure_idle.py --results $RES --seconds 30
echo "== 1. federated training (LOSO fold, S02 held out) + fusion =="; python3 train_edge.py --results $RES
echo "== 2. streaming DC1 monitor on held-out tests =="; python3 run_monitor.py --results $RES ${1:-}
echo "== 3. latency sweep ==";                      env -u OMP_NUM_THREADS -u OPENBLAS_NUM_THREADS python3 bench_latency.py --results $RES
echo "== 4. figures + summary ==";                  python3 make_figures.py --results $RES
} 2>&1 | tee "$RES/run_log.txt"
