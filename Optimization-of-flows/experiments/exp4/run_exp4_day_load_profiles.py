from __future__ import annotations

import argparse
from datetime import datetime
import json
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any, Callable

import pandas as pd

from flowopt.pipeline import (
    run_real_gap_vrp,
    run_real_genetic,
    run_real_milp,
    run_real_stochastic_grasp,
    run_real_stochastic_rr,
)


def _log(t0: float, msg: str) -> None:
    dt = time.perf_counter() - t0
    print(f"[+{dt:7.1f}s] {msg}", flush=True)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    try:
        import numpy as np  # type: ignore

        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass
    return str(obj)


def _load_index(day_load_root: Path) -> dict[str, Any]:
    idx = day_load_root / "index_day_load_profiles.json"
    if not idx.exists():
        return {}
    try:
        return json.loads(idx.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _discover_datasets(day_load_root: Path) -> list[dict[str, Any]]:
    index = _load_index(day_load_root)
    target_by_tag: dict[str, float] = {}
    for row in index.get("profiles", []):
        tag = str(row.get("profile_tag", "")).strip()
        if tag:
            target_by_tag[tag] = float(row.get("target_utilization", 0.0))

    datasets: list[dict[str, Any]] = []
    for p in sorted(day_load_root.glob("u*/dataset_*.json")):
        tag = p.parent.name
        payload = json.loads(p.read_text(encoding="utf-8"))
        datasets.append(
            {
                "tag": tag,
                "dataset_path": p.resolve(),
                "tasks_total": len(payload.get("tasks", [])),
                "agents_total": len(payload.get("agents", [])),
                "target_utilization": target_by_tag.get(tag),
            }
        )
    return datasets


def _worker(func_name: str, kwargs: dict[str, Any], queue: mp.Queue) -> None:
    fn_map: dict[str, Callable[..., Any]] = {
        "real_gap_vrp": run_real_gap_vrp,
        "real_milp": run_real_milp,
        "real_genetic": run_real_genetic,
        "real_stochastic_grasp": run_real_stochastic_grasp,
        "real_stochastic_rr": run_real_stochastic_rr,
    }
    try:
        from flowopt import core

        fn = fn_map[func_name]
        params = dict(kwargs)
        capacity_scale = float(params.pop("__capacity_scale", 1.0))
        base_km = core.MAX_DAILY_KM_BY_TYPE.copy()
        base_h = core.MAX_SHIFT_HOURS_BY_TYPE.copy()
        if capacity_scale != 1.0:
            core.MAX_DAILY_KM_BY_TYPE.update({k: v * capacity_scale for k, v in base_km.items()})
            core.MAX_SHIFT_HOURS_BY_TYPE.update({k: v * capacity_scale for k, v in base_h.items()})

        try:
            metrics = fn(**params)
        finally:
            core.MAX_DAILY_KM_BY_TYPE.clear()
            core.MAX_DAILY_KM_BY_TYPE.update(base_km)
            core.MAX_SHIFT_HOURS_BY_TYPE.clear()
            core.MAX_SHIFT_HOURS_BY_TYPE.update(base_h)

        data = metrics.as_dict()
        data.pop("agent_solution_rows", None)
        data["capacity_scale"] = capacity_scale
        queue.put({"ok": True, "data": data})
    except Exception as exc:
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _timeout_row(
    *,
    dataset_tag: str,
    dataset_path: str,
    tasks_total: int,
    agents_total: int,
    target_utilization: float | None,
    algorithm: str,
    elapsed: float,
    timeout_sec: int,
) -> dict[str, Any]:
    return {
        "dataset_tag": dataset_tag,
        "dataset_path": dataset_path,
        "tasks_total": tasks_total,
        "agents_total": agents_total,
        "target_utilization": target_utilization,
        "algorithm": algorithm,
        "feasible": False,
        "all_checks_ok": False,
        "assigned_routes": 0,
        "unassigned_tasks": tasks_total,
        "active_agents": 0,
        "transport_work_ton_km": None,
        "total_km": None,
        "deadhead_km": None,
        "deadhead_share_pct": None,
        "total_hours": None,
        "runtime_sec": round(elapsed, 3),
        "solver_error": f"TimeoutError: exceeded timeout {timeout_sec}s",
    }


def _run_with_timeout(
    *,
    t0: float,
    dataset_row: dict[str, Any],
    algorithm: str,
    func_name: str,
    kwargs: dict[str, Any],
    timeout_sec: int,
) -> dict[str, Any]:
    queue: mp.Queue = mp.Queue()
    p = mp.Process(target=_worker, args=(func_name, kwargs, queue), daemon=True)

    _log(t0, f"{dataset_row['tag']} | {algorithm}: START")
    w0 = time.perf_counter()
    p.start()
    p.join(timeout=timeout_sec)

    elapsed = time.perf_counter() - w0
    if p.is_alive():
        p.terminate()
        p.join(timeout=5)
        _log(t0, f"{dataset_row['tag']} | {algorithm}: TIMEOUT after {elapsed:.1f}s")
        return _timeout_row(
            dataset_tag=dataset_row["tag"],
            dataset_path=str(dataset_row["dataset_path"]),
            tasks_total=int(dataset_row["tasks_total"]),
            agents_total=int(dataset_row["agents_total"]),
            target_utilization=dataset_row.get("target_utilization"),
            algorithm=algorithm,
            elapsed=elapsed,
            timeout_sec=timeout_sec,
        )

    result: dict[str, Any] | None = None
    if not queue.empty():
        result = queue.get()

    if not result or not result.get("ok", False):
        err = "Worker failed without error payload"
        if result and "error" in result:
            err = str(result["error"])
        _log(t0, f"{dataset_row['tag']} | {algorithm}: ERROR after {elapsed:.1f}s: {err}")
        row = _timeout_row(
            dataset_tag=dataset_row["tag"],
            dataset_path=str(dataset_row["dataset_path"]),
            tasks_total=int(dataset_row["tasks_total"]),
            agents_total=int(dataset_row["agents_total"]),
            target_utilization=dataset_row.get("target_utilization"),
            algorithm=algorithm,
            elapsed=elapsed,
            timeout_sec=timeout_sec,
        )
        row["solver_error"] = err
        return row

    row = dict(result["data"])
    row["dataset_tag"] = dataset_row["tag"]
    row["dataset_path"] = str(dataset_row["dataset_path"])
    row["tasks_total"] = int(dataset_row["tasks_total"])
    row["agents_total"] = int(dataset_row["agents_total"])
    row["target_utilization"] = dataset_row.get("target_utilization")
    row["runtime_sec"] = round(elapsed, 3)
    if row.get("unassigned_tasks") is not None and row["tasks_total"]:
        row["assignment_coverage_pct"] = round(
            100.0 * (row["tasks_total"] - row["unassigned_tasks"]) / row["tasks_total"],
            3,
        )
    else:
        row["assignment_coverage_pct"] = None
    _log(t0, f"{dataset_row['tag']} | {algorithm}: DONE in {elapsed:.1f}s")
    return row


def _build_method_specs(args: argparse.Namespace) -> list[tuple[str, str, dict[str, Any]]]:
    return [
        (
            "real_gap_vrp",
            "real_gap_vrp",
            {
                "step1_method": "greedy",
                "gap_iter": args.gap_iter,
                "use_repair": True,
                "show_progress": False,
                "verbose": False,
            },
        ),
        (
            "real_milp",
            "real_milp",
            {
                "time_limit_sec": args.milp_time_limit_sec,
                "unassigned_penalty": 1e5,
                "show_progress": False,
            },
        ),
        (
            "real_genetic",
            "real_genetic",
            {
                "population_size": args.genetic_population,
                "generations": args.genetic_generations,
                "generation_scale": args.genetic_generation_scale,
                "elite_size": args.genetic_elite,
                "seed": args.seed,
                "show_progress": False,
            },
        ),
        (
            "real_stochastic_grasp",
            "real_stochastic_grasp",
            {
                "time_budget_sec": args.stochastic_budget_sec,
                "max_starts": 3,
                "candidate_k": args.stochastic_candidate_k,
                "rcl_size": args.stochastic_rcl_size,
                "seed": args.seed,
                "show_progress": False,
            },
        ),
        (
            "real_stochastic_rr",
            "real_stochastic_rr",
            {
                "time_budget_sec": args.stochastic_budget_sec,
                "candidate_k": args.stochastic_candidate_k,
                "rcl_size": args.stochastic_rcl_size,
                "batch_size": args.stochastic_batch_size,
                "seed": args.seed,
                "show_progress": False,
            },
        ),
    ]


def _make_markdown_report(df: pd.DataFrame, out_path: Path) -> None:
    cols = [
        "dataset_tag",
        "target_utilization",
        "algorithm",
        "feasible",
        "all_checks_ok",
        "assigned_routes",
        "unassigned_tasks",
        "assignment_coverage_pct",
        "active_agents",
        "runtime_sec",
        "solver_error",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None

    lines: list[str] = []
    lines.append("# EXP4 Report: Day Load Profiles\n")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")

    for tag in sorted(df["dataset_tag"].dropna().unique().tolist()):
        lines.append(f"## Dataset `{tag}`\n")
        part = df[df["dataset_tag"] == tag][cols].copy()
        lines.append(part.to_markdown(index=False))
        lines.append("")

    lines.append("## Overall summary\n")
    piv = (
        df.groupby("algorithm", dropna=False)
        .agg(
            runs=("algorithm", "count"),
            feasible_runs=("feasible", lambda s: int(pd.Series(s).fillna(False).sum())),
            avg_runtime_sec=("runtime_sec", "mean"),
            avg_coverage_pct=("assignment_coverage_pct", "mean"),
        )
        .reset_index()
    )
    lines.append(piv.to_markdown(index=False))

    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EXP4: benchmark methods on day load profiles (u50/u90)")
    parser.add_argument(
        "--day-load-root",
        type=Path,
        default=Path("src/data/real/day_load_profiles"),
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        default="real_gap_vrp,real_milp,real_genetic,real_stochastic_grasp,real_stochastic_rr",
        help="Comma-separated subset of methods",
    )
    parser.add_argument("--timeout-sec", type=int, default=600, help="Hard timeout per method")
    parser.add_argument(
        "--capacity-scale",
        type=float,
        default=1.0,
        help="Scale for daily km/h limits (1.0 means original one-day limits)",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--gap-iter", type=int, default=20)
    parser.add_argument("--milp-time-limit-sec", type=int, default=300)

    parser.add_argument("--genetic-population", type=int, default=30)
    parser.add_argument("--genetic-generations", type=int, default=60)
    parser.add_argument("--genetic-generation-scale", type=float, default=0.15)
    parser.add_argument("--genetic-elite", type=int, default=4)

    parser.add_argument("--stochastic-budget-sec", type=int, default=300)
    parser.add_argument("--stochastic-candidate-k", type=int, default=20)
    parser.add_argument("--stochastic-rcl-size", type=int, default=5)
    parser.add_argument("--stochastic-batch-size", type=int, default=1000)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    datasets = _discover_datasets(args.day_load_root)
    if not datasets:
        raise SystemExit(f"No datasets found in: {args.day_load_root}")

    _log(t0, f"datasets discovered: {[d['tag'] for d in datasets]}")

    all_specs = _build_method_specs(args)
    selected = {x.strip() for x in args.algorithms.split(",") if x.strip()}
    method_specs = [spec for spec in all_specs if spec[0] in selected]
    if not method_specs:
        raise SystemExit("No selected algorithms matched available methods")

    _log(t0, f"methods selected: {[m[0] for m in method_specs]}")

    rows: list[dict[str, Any]] = []
    for ds in datasets:
        _log(
            t0,
            f"dataset {ds['tag']}: tasks={ds['tasks_total']}, agents={ds['agents_total']}, target_util={ds.get('target_utilization')}",
        )
        for algorithm, func_name, kwargs_base in method_specs:
            kwargs = dict(kwargs_base)
            kwargs["dataset_path"] = ds["dataset_path"]
            kwargs["__capacity_scale"] = float(args.capacity_scale)

            row = _run_with_timeout(
                t0=t0,
                dataset_row=ds,
                algorithm=algorithm,
                func_name=func_name,
                kwargs=kwargs,
                timeout_sec=max(10, int(args.timeout_sec)),
            )
            rows.append(row)

    df = pd.DataFrame(rows)

    primary_cols = [
        "dataset_tag",
        "target_utilization",
        "algorithm",
        "tasks_total",
        "agents_total",
        "feasible",
        "all_checks_ok",
        "assigned_routes",
        "unassigned_tasks",
        "assignment_coverage_pct",
        "active_agents",
        "transport_work_ton_km",
        "total_km",
        "deadhead_km",
        "deadhead_share_pct",
        "total_hours",
        "runtime_sec",
        "capacity_scale",
        "solver_error",
    ]
    for col in primary_cols:
        if col not in df.columns:
            df[col] = None
    df = df[primary_cols + [c for c in df.columns if c not in primary_cols]]

    print("\n=== EXP4 results ===")
    print(df[primary_cols].to_string(index=False))

    out_dir = Path("experiments/exp4/local")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_json = out_dir / f"exp4_day_load_profiles_{stamp}.json"
    out_csv = out_dir / f"exp4_day_load_profiles_{stamp}.csv"
    out_md = out_dir / f"exp4_day_load_profiles_{stamp}.md"

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "day_load_root": str(args.day_load_root.resolve()),
        "datasets": [
            {
                **d,
                "dataset_path": str(d.get("dataset_path")),
            }
            for d in datasets
        ],
        "methods": [m[0] for m in method_specs],
        "timeout_sec": int(args.timeout_sec),
        "capacity_scale": float(args.capacity_scale),
        "params": {
            "gap_iter": args.gap_iter,
            "milp_time_limit_sec": args.milp_time_limit_sec,
            "genetic_population": args.genetic_population,
            "genetic_generations": args.genetic_generations,
            "genetic_generation_scale": args.genetic_generation_scale,
            "genetic_elite": args.genetic_elite,
            "stochastic_budget_sec": args.stochastic_budget_sec,
            "stochastic_candidate_k": args.stochastic_candidate_k,
            "stochastic_rcl_size": args.stochastic_rcl_size,
            "stochastic_batch_size": args.stochastic_batch_size,
            "seed": args.seed,
        },
        "results": rows,
    }

    out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    df.to_csv(out_csv, index=False)
    _make_markdown_report(df, out_md)

    _log(t0, f"saved JSON: {out_json.resolve()}")
    _log(t0, f"saved CSV : {out_csv.resolve()}")
    _log(t0, f"saved MD  : {out_md.resolve()}")


if __name__ == "__main__":
    main()
