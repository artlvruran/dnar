#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
SWEEP_DIR="${SWEEP_DIR:-$REPO_ROOT/src/data/real/clean_full_enriched/sweep_7task_7agents}"
INDEX_CSV="${INDEX_CSV:-$SWEEP_DIR/sweep_index.csv}"
LOCAL_ROOT="${LOCAL_ROOT:-$REPO_ROOT/scripts/local/full_enriched_clean_sweep}"
ALGORITHMS="${ALGORITHMS:-real_milp}"
CASE_TIMEOUT_SEC="${CASE_TIMEOUT_SEC:-600}"

"$PYTHON_BIN" scripts/py_scripts/full_enriched_clean_dataset_sweep.py run \
  --sweep-dir "$SWEEP_DIR" \
  --index-csv "$INDEX_CSV" \
  --local-root "$LOCAL_ROOT" \
  --algorithms "$ALGORITHMS" \
  --case-timeout-sec "$CASE_TIMEOUT_SEC" \
  "$@"
