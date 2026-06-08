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
    REAL_FULL_29K_DATASET_PATH,
    run_real_gap_vrp,
    run_real_genetic,
    run_real_milp,
    run_real_stochastic_grasp,
    run_real_stochastic_rr,
)


def _progress_factory(global_t0: float) -> Callable[[str], None]:
    def _log(message: str) -> None:
        dt = time.perf_counter() - global_t0
        print(f"[+{dt:7.1f}s] {message}", flush=True)

    return _log


def _worker(func_name: str, kwargs: dict[str, Any], queue: mp.Queue) -> None:
    fn_map: dict[str, Callable[..., Any]] = {
        "real_gap_vrp": run_real_gap_vrp,
        "real_milp": run_real_milp,
        "real_genetic": run_real_genetic,
        "real_stochastic_grasp": run_real_stochastic_grasp,
        "real_stochastic_rr": run_real_stochastic_rr,
    }
    try:
        fn = fn_map[func_name]
        metrics = fn(**kwargs)
        data = metrics.as_dict()
        data.pop("agent_solution_rows", None)
        queue.put({"ok": True, "data": data})
    except Exception as exc:
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _timeout_row(*, algorithm: str, dataset_path: str, elapsed: float, timeout_sec: int) -> dict[str, Any]:
    return {
        "algorithm": algorithm,
        "dataset_path": dataset_path,
        "feasible": False,
        "all_checks_ok": False,
        "assigned_routes": 0,
        "unassigned_tasks": None,
        "active_agents": 0,
        "transport_work_ton_km": None,
        "total_km": None,
        "deadhead_km": None,
        "deadhead_share_pct": None,
        "total_hours": None,
        "runtime_sec": round(elapsed, 3),
        "solver_error": f"TimeoutError: exceeded timeout {timeout_sec}s",
    }


def run_with_timeout(
    *,
    algorithm: str,
    func_name: str,
    kwargs: dict[str, Any],
    timeout_sec: int,
    progress_log: Callable[[str], None],
) -> dict[str, Any]:
    queue: mp.Queue = mp.Queue()
    process = mp.Process(target=_worker, args=(func_name, kwargs, queue), daemon=True)

    t0 = time.perf_counter()
    progress_log(f"{algorithm}: START")
    process.start()
    process.join(timeout=timeout_sec)

    elapsed = time.perf_counter() - t0
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        progress_log(f"{algorithm}: TIMEOUT after {elapsed:.1f}s")
        return _timeout_row(
            algorithm=algorithm,
            dataset_path=str(kwargs.get("dataset_path", "")),
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
        progress_log(f"{algorithm}: ERROR after {elapsed:.1f}s: {err}")
        row = _timeout_row(
            algorithm=algorithm,
            dataset_path=str(kwargs.get("dataset_path", "")),
            elapsed=elapsed,
            timeout_sec=timeout_sec,
        )
        row["solver_error"] = err
        return row

    row = dict(result["data"])
    row["runtime_sec"] = round(elapsed, 3)
    progress_log(f"{algorithm}: DONE in {elapsed:.1f}s")
    return row


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp1: fast stochastic experiments on real datasets")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=REAL_FULL_29K_DATASET_PATH,
        help="Path to dataset json",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        default="real_stochastic_grasp,real_stochastic_rr",
        help=(
            "Comma-separated list. Supported: "
            "real_stochastic_grasp,real_stochastic_rr,real_gap_vrp,real_milp,real_genetic"
        ),
    )
    parser.add_argument("--timeout-sec", type=int, default=190, help="Hard timeout per algorithm")
    parser.add_argument("--budget-sec", type=int, default=180, help="Internal budget for stochastic solvers")
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rcl-size", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional head() view for compact output")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    t0 = time.perf_counter()
    log = _progress_factory(t0)

    dataset_path = args.dataset_path.resolve()
    selected = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    log(f"dataset: {dataset_path}")
    log(f"selected: {selected}")

    run_specs: dict[str, tuple[str, dict[str, Any]]] = {
        "real_stochastic_grasp": (
            "real_stochastic_grasp",
            {
                "dataset_path": dataset_path,
                "time_budget_sec": args.budget_sec,
                "max_starts": 2,
                "candidate_k": args.candidate_k,
                "rcl_size": args.rcl_size,
                "seed": args.seed,
                "show_progress": True,
            },
        ),
        "real_stochastic_rr": (
            "real_stochastic_rr",
            {
                "dataset_path": dataset_path,
                "time_budget_sec": args.budget_sec,
                "candidate_k": args.candidate_k,
                "rcl_size": args.rcl_size,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "show_progress": True,
            },
        ),
        "real_gap_vrp": (
            "real_gap_vrp",
            {
                "dataset_path": dataset_path,
                "step1_method": "greedy",
                "gap_iter": 5,
                "use_repair": False,
                "show_progress": True,
                "verbose": False,
            },
        ),
        "real_milp": (
            "real_milp",
            {
                "dataset_path": dataset_path,
                "time_limit_sec": min(60, args.budget_sec),
                "unassigned_penalty": 1e5,
                "show_progress": True,
            },
        ),
        "real_genetic": (
            "real_genetic",
            {
                "dataset_path": dataset_path,
                "population_size": 20,
                "generations": 20,
                "generation_scale": 0.25,
                "elite_size": 2,
                "seed": args.seed,
                "show_progress": True,
            },
        ),
    }

    rows: list[dict[str, Any]] = []
    for algo in selected:
        if algo not in run_specs:
            log(f"skip unknown algorithm: {algo}")
            continue
        func_name, kwargs = run_specs[algo]
        row = run_with_timeout(
            algorithm=algo,
            func_name=func_name,
            kwargs=kwargs,
            timeout_sec=max(10, int(args.timeout_sec)),
            progress_log=log,
        )
        rows.append(row)

    if not rows:
        raise SystemExit("No valid algorithms selected")

    frame = pd.DataFrame(rows)
    primary_cols = [
        "algorithm",
        "feasible",
        "all_checks_ok",
        "assigned_routes",
        "unassigned_tasks",
        "active_agents",
        "transport_work_ton_km",
        "total_km",
        "deadhead_km",
        "deadhead_share_pct",
        "total_hours",
        "runtime_sec",
        "solver_error",
    ]
    for col in primary_cols:
        if col not in frame.columns:
            frame[col] = None

    frame = frame[primary_cols + [c for c in frame.columns if c not in primary_cols]]
    print()
    if args.max_rows is not None and args.max_rows > 0:
        print(frame.head(args.max_rows).to_string(index=False))
    else:
        print(frame.to_string(index=False))

    out_dir = Path("experiments/exp1/local")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"exp1_{stamp}.json"
    out_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": str(dataset_path),
        "algorithms": selected,
        "timeout_sec": int(args.timeout_sec),
        "budget_sec": int(args.budget_sec),
        "results": rows,
    }
    out_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
