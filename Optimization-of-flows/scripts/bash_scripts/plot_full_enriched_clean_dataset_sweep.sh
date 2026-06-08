#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
LOCAL_ROOT="${LOCAL_ROOT:-$REPO_ROOT/scripts/local/full_enriched_clean_sweep}"
ALGORITHMS="${ALGORITHMS:-real_milp}"

"$PYTHON_BIN" scripts/py_scripts/full_enriched_clean_dataset_sweep.py plot \
  --local-root "$LOCAL_ROOT" \
  --algorithms "$ALGORITHMS" \
  "$@"
