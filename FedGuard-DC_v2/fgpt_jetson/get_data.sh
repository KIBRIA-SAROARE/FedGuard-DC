#!/usr/bin/env bash
# Sparse download of what the edge experiment needs: all 66 clean records (six-site federation + CSEC peers),
# the repository attacked files for one site in the test scenario, and attack_log.csv.
# Usage: ./get_data.sh [DC=1] [SCENARIO=S02_fault_100ms]
set -euo pipefail
DC="${1:-1}"; SC="${2:-S02_fault_100ms}"
REPO_URL="${REPO_URL:-https://github.com/KIBRIA-SAROARE/FedGuard-DC.git}"
cd "$(dirname "$0")"; mkdir -p data
[ -d data/repo/.git ] || git clone --filter=blob:none --no-checkout --depth 1 "$REPO_URL" data/repo
cd data/repo
git sparse-checkout init --no-cone
D=/FedGuard-DC_v2/dataset
git sparse-checkout set "$D/attack_log.csv" "$D/manifest.csv" "$D/clean/*.csv" "$D/attacked/${SC}_DC${DC}_A*.csv"
git checkout
cd ../..
ln -sfn repo/FedGuard-DC_v2/dataset data/dataset
echo "clean: $(ls data/dataset/clean | wc -l) (expect 66)   attacked: $(ls data/dataset/attacked | wc -l)"
du -shL data/dataset
