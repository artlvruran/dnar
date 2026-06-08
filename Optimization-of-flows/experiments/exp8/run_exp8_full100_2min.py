from __future__ import annotations

import json
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable

from flowopt.solvers import (
    solve_enriched_batched_greedy,
    solve_enriched_milp_ablation_zone_bundle,
    solve_enriched_milp_batch_cascaded,
    solve_enriched_milp_batch_then_milp,
)

ROOT = Path(__file__).resolve().parents[2]
DATASET = (
    ROOT
    / "demo/data/object_mass_feasible_fullfleet/container_full_split_all_agents/with_distances/"
    / "dataset_real_spb_clean_full_split_by_containers_all_agents_with_distances.json"
)
OUT_DIR = ROOT / "experiments/local/exp8_batching_ablation"
TIMEOUT_SEC = 120
HARD_TIMEOUT_SEC = 180


def _worker(
    fn: Callable[..., Any],
    kwargs: dict[str, Any],
    dataset_path: str,
    queue: Any,
) -> None:
    try:
        t0 = time.perf_counter()
        result = fn(dataset_path=dataset_path, **kwargs)
        d = result.as_dict()
        d["wall_runtime_sec"] = round(time.perf_counter() - t0, 3)
        queue.put(("ok", d))
    except Exception as exc:  # pragma: no cover
        queue.put(("err", f"{type(exc).__name__}: {exc}"))


def _run_one(name: str, fn: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    ctx = get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(fn, kwargs, str(DATASET), q))
    t0 = time.perf_counter()
    p.start()
    p.join(HARD_TIMEOUT_SEC)
    elapsed = round(time.perf_counter() - t0, 3)

    if p.is_alive():
        p.terminate()
        p.join(5)
        return {
            "solver": name,
            "status": "timeout",
            "elapsed_sec": elapsed,
            "feasible": False,
            "assigned_tasks": 0,
            "unassigned_tasks": None,
            "coverage_pct": None,
            "active_agents": 0,
            "runtime_sec": elapsed,
            "solver_error": f"Hard timeout > {HARD_TIMEOUT_SEC}s",
        }

    if q.empty():
        return {
            "solver": name,
            "status": "no_result",
            "elapsed_sec": elapsed,
            "feasible": False,
            "assigned_tasks": 0,
            "unassigned_tasks": None,
            "coverage_pct": None,
            "active_agents": 0,
            "runtime_sec": elapsed,
            "solver_error": "No result returned",
        }

    status, payload = q.get()
    if status == "err":
        return {
            "solver": name,
            "status": "error",
            "elapsed_sec": elapsed,
            "feasible": False,
            "assigned_tasks": 0,
            "unassigned_tasks": None,
            "coverage_pct": None,
            "active_agents": 0,
            "runtime_sec": elapsed,
            "solver_error": payload,
        }

    assigned = int(payload.get("assigned_routes") or 0)
    unassigned = payload.get("unassigned_tasks")
    total = assigned + (int(unassigned) if unassigned is not None else 0)
    cov = (100.0 * assigned / total) if total > 0 else None
    details = payload.get("details") or {}
    cfg_details = details.get("config") or {}
    milp_cfg = details.get("milp_config") or {}
    return {
        "solver": name,
        "status": "ok",
        "elapsed_sec": elapsed,
        "feasible": bool(payload.get("feasible")),
        "all_checks_ok": bool(((payload.get("details") or {}).get("checks") or {}).get("all_checks_ok", False)),
        "assigned_tasks": assigned,
        "unassigned_tasks": unassigned,
        "coverage_pct": round(cov, 3) if cov is not None else None,
        "active_agents": int(payload.get("active_agents") or 0),
        "total_km": payload.get("total_km"),
        "total_hours": payload.get("total_hours"),
        "transport_work_ton_km": payload.get("transport_work_ton_km"),
        "runtime_sec": payload.get("runtime_sec"),
        "wall_runtime_sec": payload.get("wall_runtime_sec"),
        "soft_cutoff_hit": bool(
            details.get("global_cutoff_hit")
            or cfg_details.get("global_cutoff_hit")
            or milp_cfg.get("candidate_cutoff_hit")
        ),
        "solver_error": payload.get("solver_error"),
    }


def main() -> None:
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    solvers: list[tuple[str, Callable[..., Any], dict[str, Any]]] = [
        (
            "greedy_ref",
            solve_enriched_batched_greedy,
            {"top_k_agents": 30, "balance_penalty": 0.02, "random_seed": 42, "max_runtime_sec": float(TIMEOUT_SEC)},
        ),
        (
            "milp_zone_bundle",
            solve_enriched_milp_ablation_zone_bundle,
            {
                "time_limit_sec_per_zone": 14,
                "max_runtime_sec": float(TIMEOUT_SEC),
                "max_pairs_per_bundle": 160,
                "bundle_fill_factor": 0.93,
                "bundle_max_tasks": 12,
                "unassigned_penalty": 1e5,
                "objective": "tasks",
            },
        ),
        (
            "milp_batch_then_milp",
            solve_enriched_milp_batch_then_milp,
            {
                "time_budget_sec": float(TIMEOUT_SEC),
                "bundle_max_tasks": 16,
                "bundle_fill_factor": 0.95,
                "max_pairs_per_bundle": 180,
                "max_pairs_per_task": 200,
                "unassigned_penalty": 1e6,
            },
        ),
        (
            "milp_batch_cascaded",
            solve_enriched_milp_batch_cascaded,
            {
                "time_budget_sec": float(TIMEOUT_SEC),
                "unassigned_penalty": 1e6,
            },
        ),
    ]

    rows: list[dict[str, Any]] = []
    for name, fn, kwargs in solvers:
        print(f"RUN {name} ...")
        row = _run_one(name, fn, kwargs)
        rows.append(row)
        print(
            f"  -> {row['status']} assigned={row.get('assigned_tasks')} "
            f"unassigned={row.get('unassigned_tasks')} cov={row.get('coverage_pct')} elapsed={row.get('elapsed_sec')}s"
        )

    rows.sort(key=lambda r: ((r.get("coverage_pct") is None), -(r.get("coverage_pct") or -1), r.get("elapsed_sec", 1e9)))
    out_json = OUT_DIR / "exp8_full100_2min_results.json"
    out_csv = OUT_DIR / "exp8_full100_2min_results.csv"
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    header = list(rows[0].keys()) if rows else []
    with out_csv.open("w", encoding="utf-8") as fh:
        fh.write(",".join(header) + "\n")
        for row in rows:
            vals = []
            for k in header:
                v = row.get(k)
                if isinstance(v, str):
                    vals.append('"' + v.replace('"', '""') + '"')
                else:
                    vals.append("" if v is None else str(v))
            fh.write(",".join(vals) + "\n")

    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
