from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from flowopt.volume_core import (
    GreedyBatchConfig,
    VolumeDataset,
    VolumeGapVRPLikeSolver,
    VolumeGapVRPStochasticSolver,
    VolumeGeneticLikeSolver,
    VolumeGeneticStochasticSolver,
    VolumeMilpStochasticSolver,
    VolumeMilpLikeSolver,
    assert_dataset_input,
    assert_solution_output,
    save_solution_artifacts,
)


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "demo/data/object_volume_feasible_fullfleet/full_all_tasks_all_agents_stage4_volume_only"
DEFAULT_DATASET = (
    DATA_DIR / "dataset_real_spb_clean_full_all_tasks_all_agents_stage4_volume_only_greedy_full_subset_with_distances_v2_100.json"
)
if not DEFAULT_DATASET.exists():
    DEFAULT_DATASET = DATA_DIR / "dataset_real_spb_clean_full_all_tasks_all_agents_stage4_volume_only_greedy_full_subset_with_distances.json"
if not DEFAULT_DATASET.exists():
    DEFAULT_DATASET = DATA_DIR / "dataset_real_spb_clean_full_all_tasks_all_agents_stage4_volume_only_feasible_subset_with_distances.json"
OUT_ROOT = ROOT / "experiments/local/exp13_volume_core_3algo"


def main() -> None:
    p = argparse.ArgumentParser(description="EXP13: volume-core 3 algorithms benchmark")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--max-runtime-sec", type=float, default=180.0)
    p.add_argument("--top-k-agents", type=int, default=30)
    p.add_argument("--top-k-destinations", type=int, default=4)
    p.add_argument("--max-tasks-in-trip", type=int, default=500)
    p.add_argument("--log-every-sec", type=float, default=20.0)
    p.add_argument("--trip-log-every", type=int, default=400)
    p.add_argument("--override-speed-kmph", type=float, default=0.0)
    p.add_argument("--include-stochastic", action="store_true", help="Run stochastic variants too")
    p.add_argument("--stochastic-only", action="store_true", help="Run only stochastic variants")
    args = p.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUT_ROOT / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = GreedyBatchConfig(
        max_runtime_sec=float(args.max_runtime_sec),
        top_k_agents=int(args.top_k_agents),
        top_k_destinations=int(args.top_k_destinations),
        max_tasks_in_trip=int(args.max_tasks_in_trip),
        log_every_sec=float(args.log_every_sec),
        trip_log_every=int(args.trip_log_every),
        verbose=True,
    )

    base_solvers: list[tuple[str, object]] = [
        ("gap_vrp_like", VolumeGapVRPLikeSolver(cfg)),
        ("milp_like", VolumeMilpLikeSolver(cfg)),
        ("genetic_like", VolumeGeneticLikeSolver(cfg)),
    ]
    stoch_solvers: list[tuple[str, object]] = [
        ("gap_vrp_stoch", VolumeGapVRPStochasticSolver(cfg)),
        ("milp_stoch", VolumeMilpStochasticSolver(cfg)),
        ("genetic_stoch", VolumeGeneticStochasticSolver(cfg)),
    ]
    if args.stochastic_only:
        solvers = stoch_solvers
    elif args.include_stochastic:
        solvers = base_solvers + stoch_solvers
    else:
        solvers = base_solvers

    rows: list[dict] = []
    artifacts_index: dict[str, dict[str, str]] = {}

    for name, solver in solvers:
        print(f"\n=== {name.upper()} START ===", flush=True)
        dataset = VolumeDataset.from_json(args.dataset)
        assert_dataset_input(dataset)
        if args.override_speed_kmph and args.override_speed_kmph > 0:
            dataset.agents = tuple(
                a.__class__(**{**a.__dict__, "avg_speed_kmph": float(args.override_speed_kmph)})
                if (a.is_active and a.depot_node_id is not None)
                else a
                for a in dataset.agents
            )

        t0 = time.perf_counter()
        if hasattr(solver, "solve_checked"):
            solution = solver.solve_checked(dataset)
        else:
            solution = solver.solve(dataset)
            solution = assert_solution_output(solution, dataset_path=str(dataset.dataset_path))
        elapsed = time.perf_counter() - t0
        evaluation = dataset.evaluate(solution)

        algo_dir = run_dir / name
        algo_dir.mkdir(parents=True, exist_ok=True)
        paths = save_solution_artifacts(dataset=dataset, solution=solution, evaluation=evaluation, out_dir=algo_dir)
        artifacts_index[name] = paths

        row = evaluation.as_dict()
        row["runner_runtime_sec"] = round(elapsed, 3)
        rows.append(row)
        print(
            f"=== {name.upper()} DONE === assigned={row['assigned_tasks']}/{row['total_tasks']} "
            f"coverage={row['task_coverage_pct']}% feasible={row['feasible']} runtime={row['runtime_sec']}s",
            flush=True,
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["feasible", "assigned_tasks", "runtime_sec"], ascending=[False, False, True]).reset_index(drop=True)
    (run_dir / "summary.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "artifacts_index.json").write_text(json.dumps(artifacts_index, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "params.json").write_text(
        json.dumps(
            {
                "dataset": str(Path(args.dataset).resolve()),
                "max_runtime_sec": float(args.max_runtime_sec),
                "top_k_agents": int(args.top_k_agents),
                "top_k_destinations": int(args.top_k_destinations),
                "max_tasks_in_trip": int(args.max_tasks_in_trip),
                "log_every_sec": float(args.log_every_sec),
                "trip_log_every": int(args.trip_log_every),
                "override_speed_kmph": float(args.override_speed_kmph),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nRun dir: {run_dir}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
