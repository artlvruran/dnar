from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from flowopt.solvers import (
    solve_enriched_batched_greedy,
    solve_enriched_milp_ablation_adaptive_k,
    solve_enriched_milp_ablation_baseline,
    solve_enriched_milp_ablation_penalty_sweep,
    solve_enriched_milp_ablation_portfolio,
    solve_enriched_milp_ablation_zone_bundle,
    solve_enriched_milp_batch_cascaded,
    solve_enriched_milp_batch_portfolio,
    solve_enriched_milp_batch_then_milp,
    solve_enriched_milp_hybrid_seeded,
    solve_enriched_milp_lns_rounds,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = (
    ROOT
    / "demo/data/object_mass_feasible_fullfleet/container_full_split_all_agents/sweeps_task_agent_5pct/"
    / "dataset_real_spb_clean_full_split_by_containers_all_agents_with_distances_t001_micro200_a100.json"
)
OUT_DIR = ROOT / "experiments/local/exp8_batching_ablation"


@dataclass(frozen=True)
class SolverSpec:
    name: str
    fn: Callable[..., object]
    kwargs: dict[str, object]


def _build_specs() -> list[SolverSpec]:
    return [
        SolverSpec(
            name="greedy_ref",
            fn=solve_enriched_batched_greedy,
            kwargs={"top_k_agents": 30, "balance_penalty": 0.02, "random_seed": 42},
        ),
        SolverSpec(
            name="milp_zone_bundle",
            fn=solve_enriched_milp_ablation_zone_bundle,
            kwargs={
                "time_limit_sec_per_zone": 16,
                "max_pairs_per_bundle": 180,
                "bundle_fill_factor": 0.93,
                "bundle_max_tasks": 12,
                "unassigned_penalty": 1e5,
                "objective": "tasks",
            },
        ),
        SolverSpec(
            name="milp_hybrid_seeded",
            fn=solve_enriched_milp_hybrid_seeded,
            kwargs={"time_budget_sec": 90, "max_pairs_per_task": 220, "unassigned_penalty": 1e6},
        ),
        SolverSpec(
            name="milp_batch_then_milp",
            fn=solve_enriched_milp_batch_then_milp,
            kwargs={
                "time_budget_sec": 90,
                "bundle_max_tasks": 16,
                "bundle_fill_factor": 0.95,
                "max_pairs_per_bundle": 200,
                "max_pairs_per_task": 220,
                "unassigned_penalty": 1e6,
            },
        ),
        SolverSpec(
            name="milp_batch_cascaded",
            fn=solve_enriched_milp_batch_cascaded,
            kwargs={"time_budget_sec": 120, "unassigned_penalty": 1e6},
        ),
        SolverSpec(
            name="milp_batch_portfolio",
            fn=solve_enriched_milp_batch_portfolio,
            kwargs={"time_budget_sec": 120, "unassigned_penalty": 1e6},
        ),
        SolverSpec(
            name="milp_lns_rounds",
            fn=solve_enriched_milp_lns_rounds,
            kwargs={"time_budget_sec": 90, "max_pairs_per_task": 200, "max_rounds": 5, "unassigned_penalty": 1e6},
        ),
        SolverSpec(
            name="milp_portfolio",
            fn=solve_enriched_milp_ablation_portfolio,
            kwargs={
                "time_budget_sec": 90,
                "max_starts": 10,
                "per_start_time_limit_sec": 8,
                "max_pairs_per_task": 120,
                "jitter": 0.08,
                "unassigned_penalty": 1e6,
            },
        ),
        SolverSpec(
            name="milp_adaptive_k",
            fn=solve_enriched_milp_ablation_adaptive_k,
            kwargs={"time_budget_sec": 90, "pair_schedule": (60, 120, 180, 260), "unassigned_penalty": 1e6},
        ),
        SolverSpec(
            name="milp_penalty_sweep",
            fn=solve_enriched_milp_ablation_penalty_sweep,
            kwargs={
                "time_budget_sec": 90,
                "max_pairs_per_task": 180,
                "penalty_schedule": (1e5, 1e6, 1e7, 1e8),
            },
        ),
        SolverSpec(
            name="milp_baseline",
            fn=solve_enriched_milp_ablation_baseline,
            kwargs={"time_limit_sec": 90, "max_pairs_per_task": 180, "unassigned_penalty": 1e6},
        ),
    ]


def _result_row(name: str, result: object) -> dict[str, object]:
    d = result.as_dict()
    total = int((d.get("assigned_routes") or 0) + (d.get("unassigned_tasks") or 0))
    cov = (100.0 * (d.get("assigned_routes") or 0) / total) if total > 0 else 0.0
    return {
        "solver": name,
        "algorithm": d.get("algorithm"),
        "feasible": bool(d.get("feasible")),
        "all_checks_ok": bool(((d.get("details") or {}).get("checks") or {}).get("all_checks_ok", False)),
        "assigned_tasks": int(d.get("assigned_routes") or 0),
        "unassigned_tasks": int(d.get("unassigned_tasks") or 0),
        "coverage_pct": round(cov, 3),
        "active_agents": int(d.get("active_agents") or 0),
        "total_km": d.get("total_km"),
        "total_hours": d.get("total_hours"),
        "transport_work_ton_km": d.get("transport_work_ton_km"),
        "runtime_sec": float(d.get("runtime_sec") or 0.0),
        "solver_error": d.get("solver_error"),
    }


def _error_row(name: str, exc: Exception) -> dict[str, object]:
    return {
        "solver": name,
        "algorithm": name,
        "feasible": False,
        "all_checks_ok": False,
        "assigned_tasks": 0,
        "unassigned_tasks": None,
        "coverage_pct": 0.0,
        "active_agents": 0,
        "total_km": None,
        "total_hours": None,
        "transport_work_ton_km": None,
        "runtime_sec": 0.0,
        "solver_error": f"{type(exc).__name__}: {exc}",
    }


def _make_plots(df: pd.DataFrame, out_dir: Path) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    chart = df.sort_values(["coverage_pct", "runtime_sec"], ascending=[False, True]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(chart["solver"], chart["coverage_pct"])
    ax.set_ylabel("coverage %")
    ax.set_title("EXP8: coverage by solver")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(plots_dir / "coverage.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(chart["runtime_sec"], chart["coverage_pct"])
    for _, r in chart.iterrows():
        ax.annotate(str(r["solver"]), (float(r["runtime_sec"]), float(r["coverage_pct"])), fontsize=8)
    ax.set_xlabel("runtime sec")
    ax.set_ylabel("coverage %")
    ax.set_title("EXP8: runtime vs coverage")
    fig.tight_layout()
    fig.savefig(plots_dir / "runtime_vs_coverage.png", dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="EXP8: internal batching ablation for enriched solvers")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--only", type=str, default="", help="comma-separated solver names")
    args = p.parse_args()

    dataset = args.dataset.resolve()
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")

    specs = _build_specs()
    if args.only.strip():
        selected = {x.strip() for x in args.only.split(",") if x.strip()}
        specs = [s for s in specs if s.name in selected]

    rows: list[dict[str, object]] = []
    progress = tqdm(specs, desc="EXP8 solvers", unit="solver")
    for spec in progress:
        progress.set_postfix_str(spec.name)
        try:
            result = spec.fn(dataset_path=dataset, **spec.kwargs)
            rows.append(_result_row(spec.name, result))
        except Exception as exc:
            rows.append(_error_row(spec.name, exc))

    df = pd.DataFrame(rows)
    df = df.sort_values(["coverage_pct", "runtime_sec"], ascending=[False, True], na_position="last").reset_index(
        drop=True
    )

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = out_dir / f"exp8_batching_ablation_{stamp}.json"
    csv_path = out_dir / f"exp8_batching_ablation_{stamp}.csv"
    latest_path = out_dir / "exp8_batching_ablation_latest.csv"

    json_path.write_text(json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2), encoding="utf-8")
    df.to_csv(csv_path, index=False)
    df.to_csv(latest_path, index=False)
    _make_plots(df, out_dir)

    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {latest_path}")
    print(df[["solver", "feasible", "assigned_tasks", "unassigned_tasks", "coverage_pct", "runtime_sec"]])


if __name__ == "__main__":
    main()
