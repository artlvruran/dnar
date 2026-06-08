from __future__ import annotations

# Backward-compatible facade. Main implementation lives in flowopt.pipelines.runs.
# Use getattr fallbacks so partial updates do not break imports in notebooks.
from .pipelines import runs as _runs

DATA_ROOT = _runs.DATA_ROOT
REAL_DATASET_PATH = _runs.REAL_DATASET_PATH
REAL_SIMPLE_DATASET_PATH = _runs.REAL_SIMPLE_DATASET_PATH
REAL_FULL_29K_DATASET_PATH = getattr(_runs, "REAL_FULL_29K_DATASET_PATH", REAL_DATASET_PATH)
REAL_CLEAN_FULL_ENRICHED_DATASET_PATH = getattr(
    _runs,
    "REAL_CLEAN_FULL_ENRICHED_DATASET_PATH",
    REAL_DATASET_PATH,
)
REAL_CLEAN_FULL_ENRICHED_SIMPLE_DATASET_PATH = getattr(
    _runs,
    "REAL_CLEAN_FULL_ENRICHED_SIMPLE_DATASET_PATH",
    REAL_SIMPLE_DATASET_PATH,
)
SYNTHETIC_DATASET_PATH = _runs.SYNTHETIC_DATASET_PATH
RunMetrics = _runs.RunMetrics

benchmark_synthetic = _runs.benchmark_synthetic
load_dataset = _runs.load_dataset
load_payload = _runs.load_payload

run_dummy = _runs.run_dummy
run_gap_vrp = _runs.run_gap_vrp
run_genetic = _runs.run_genetic
run_milp = _runs.run_milp

run_real_gap_vrp = getattr(_runs, "run_real_gap_vrp", run_gap_vrp)
run_real_gap_vrp_alns = getattr(_runs, "run_real_gap_vrp_alns", run_real_gap_vrp)
run_real_gap_vrp_saa = getattr(_runs, "run_real_gap_vrp_saa", run_real_gap_vrp)
run_real_milp = getattr(_runs, "run_real_milp", run_milp)
run_real_genetic = getattr(_runs, "run_real_genetic", run_genetic)
run_real_stochastic_grasp = getattr(_runs, "run_real_stochastic_grasp", run_real_genetic)
run_real_stochastic_rr = getattr(_runs, "run_real_stochastic_rr", run_real_genetic)

__all__ = [
    "DATA_ROOT",
    "REAL_DATASET_PATH",
    "REAL_SIMPLE_DATASET_PATH",
    "REAL_FULL_29K_DATASET_PATH",
    "REAL_CLEAN_FULL_ENRICHED_DATASET_PATH",
    "REAL_CLEAN_FULL_ENRICHED_SIMPLE_DATASET_PATH",
    "SYNTHETIC_DATASET_PATH",
    "RunMetrics",
    "benchmark_synthetic",
    "load_dataset",
    "load_payload",
    "run_dummy",
    "run_gap_vrp",
    "run_genetic",
    "run_milp",
    "run_real_gap_vrp",
    "run_real_gap_vrp_alns",
    "run_real_gap_vrp_saa",
    "run_real_milp",
    "run_real_genetic",
    "run_real_stochastic_grasp",
    "run_real_stochastic_rr",
]
