from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import time
from typing import Any, Callable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

TMP_ROOT = REPO_ROOT / ".test_tmp" / "sweep_runtime"
os.environ.setdefault("MPLCONFIGDIR", str((TMP_ROOT / "mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((TMP_ROOT / "xdg-cache").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).joinpath("fontconfig").mkdir(parents=True, exist_ok=True)

from flowopt.pipeline_runtime import execute_solver

DEFAULT_SWEEP_DIR = REPO_ROOT / "src" / "data" / "real" / "clean_full_enriched" / "sweep_7task_7agents"
DEFAULT_INDEX_CSV = DEFAULT_SWEEP_DIR / "sweep_index.csv"
DEFAULT_LOCAL_ROOT = REPO_ROOT / "scripts" / "local" / "full_enriched_clean_sweep"


@dataclass(frozen=True)
class SweepCase:
    task_percent: int
    agent_percent: int
    dataset_file: str
    tasks: int
    agents: int

    @property
    def key(self) -> str:
        return f"{self.dataset_file}|t{self.task_percent:03d}|a{self.agent_percent:03d}"


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _log(t0: float, message: str) -> None:
    dt = time.perf_counter() - t0
    print(f"[+{dt:7.1f}s] {message}", flush=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _read_index(index_csv: Path) -> list[SweepCase]:
    df = pd.read_csv(index_csv)
    expected = {"task_percent", "agent_percent", "dataset_file", "tasks", "agents"}
    missing = sorted(expected - set(df.columns))
    if missing:
        raise ValueError(f"Index file is missing columns: {missing}")

    cases: list[SweepCase] = []
    for row in df.to_dict(orient="records"):
        cases.append(
            SweepCase(
                task_percent=int(row["task_percent"]),
                agent_percent=int(row["agent_percent"]),
                dataset_file=str(row["dataset_file"]),
                tasks=int(row["tasks"]),
                agents=int(row["agents"]),
            )
        )
    return cases


def _parse_percent_list(raw: str | None) -> set[int] | None:
    if raw is None or not raw.strip():
        return None
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 1 or value > 100:
            raise ValueError(f"Percent value out of range [1..100]: {value}")
        out.add(value)
    return out or None


def _filter_cases(
    cases: list[SweepCase],
    *,
    task_percents: set[int] | None,
    agent_percents: set[int] | None,
) -> list[SweepCase]:
    out: list[SweepCase] = []
    for case in cases:
        if task_percents is not None and case.task_percent not in task_percents:
            continue
        if agent_percents is not None and case.agent_percent not in agent_percents:
            continue
        out.append(case)
    out.sort(key=lambda c: (c.task_percent, c.agent_percent))
    return out


def _load_state_rows(state_path: Path) -> list[dict[str, Any]]:
    if not state_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in state_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
        except Exception:
            continue
    return rows


def _state_map(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        dataset_key = str(row.get("dataset_key", ""))
        algorithm = str(row.get("algorithm", ""))
        if not dataset_key or not algorithm:
            continue
        by_key[(dataset_key, algorithm)] = row
    return by_key


def _append_state_row(state_path: Path, row: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def _build_summary_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    if "unassigned_tasks" in df.columns and "tasks_total" in df.columns:
        coverage = []
        for u, total in zip(df["unassigned_tasks"], df["tasks_total"]):
            if pd.isna(u) or pd.isna(total) or float(total) <= 0:
                coverage.append(None)
            else:
                coverage.append(round(100.0 * (float(total) - float(u)) / float(total), 3))
        df["task_coverage_pct"] = coverage

    if "active_agents" in df.columns and "agents_total" in df.columns:
        active_share = []
        for a, total in zip(df["active_agents"], df["agents_total"]):
            if pd.isna(a) or pd.isna(total) or float(total) <= 0:
                active_share.append(None)
            else:
                active_share.append(round(100.0 * float(a) / float(total), 3))
        df["active_agents_share_pct"] = active_share

    ordered_cols = [
        "run_id",
        "finished_at",
        "algorithm",
        "task_percent",
        "agent_percent",
        "tasks_total",
        "agents_total",
        "status",
        "feasible",
        "all_checks_ok",
        "assigned_routes",
        "unassigned_tasks",
        "task_coverage_pct",
        "active_agents",
        "active_agents_share_pct",
        "transport_work_ton_km",
        "total_km",
        "deadhead_share_pct",
        "total_hours",
        "runtime_sec",
        "solver_error",
        "dataset_path",
    ]
    for col in ordered_cols:
        if col not in df.columns:
            df[col] = None
    return df[ordered_cols]


def _emit_queue_progress(msg: str, q: mp.Queue) -> None:
    q.put({"type": "progress", "message": msg})


def _worker_run(
    *,
    dataset_path: str,
    algorithm: str,
    solver_kwargs: dict[str, Any],
    show_solver_progress: bool,
    q: mp.Queue,
) -> None:
    try:
        execution = execute_solver(
            algorithm=algorithm,
            dataset_path=dataset_path,
            solver_kwargs=solver_kwargs,
            show_progress=show_solver_progress,
            verbose=False,
            progress_hook=(lambda message: _emit_queue_progress(message, q)),
        )
        q.put({"type": "result", "ok": True, "data": execution.metrics.as_dict()})
    except Exception as exc:  # pragma: no cover - subprocess guard
        q.put({"type": "result", "ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _run_case_with_timeout(
    *,
    t0: float,
    case: SweepCase,
    dataset_path: Path,
    algorithm: str,
    solver_kwargs: dict[str, Any],
    timeout_sec: int,
    show_solver_progress: bool,
    show_live_solver_logs: bool,
) -> dict[str, Any]:
    q: mp.Queue = mp.Queue()
    p = mp.Process(
        target=_worker_run,
        kwargs={
            "dataset_path": str(dataset_path),
            "algorithm": algorithm,
            "solver_kwargs": solver_kwargs,
            "show_solver_progress": show_solver_progress,
            "q": q,
        },
        daemon=True,
    )

    p.start()
    started = time.perf_counter()
    result_payload: dict[str, Any] | None = None

    while True:
        if result_payload is not None:
            break
        elapsed = time.perf_counter() - started
        if elapsed > timeout_sec:
            p.terminate()
            p.join(timeout=5)
            return {
                "status": "timeout",
                "solver_error": f"TimeoutError: exceeded {timeout_sec}s",
                "runtime_sec": round(elapsed, 3),
                "algorithm": algorithm,
                "feasible": False,
                "all_checks_ok": False,
                "assigned_routes": 0,
                "unassigned_tasks": case.tasks,
                "active_agents": 0,
                "transport_work_ton_km": None,
                "total_km": None,
                "deadhead_km": None,
                "deadhead_share_pct": None,
                "total_hours": None,
            }

        try:
            item = q.get(timeout=1.0)
        except queue.Empty:
            if not p.is_alive():
                # process finished, drain queue below
                pass
            continue

        item_type = item.get("type")
        if item_type == "progress":
            if show_live_solver_logs:
                _log(t0, f"{case.dataset_file} | {algorithm}: {item.get('message')}")
            continue

        if item_type == "result":
            result_payload = item
            break

    p.join(timeout=5)

    # Drain any buffered progress/result messages.
    while True:
        try:
            item = q.get_nowait()
        except queue.Empty:
            break
        if item.get("type") == "progress":
            if show_live_solver_logs:
                _log(t0, f"{case.dataset_file} | {algorithm}: {item.get('message')}")
            continue
        if item.get("type") == "result" and result_payload is None:
            result_payload = item

    elapsed = time.perf_counter() - started
    if not result_payload:
        return {
            "status": "error",
            "solver_error": "Worker returned no payload",
            "runtime_sec": round(elapsed, 3),
            "algorithm": algorithm,
            "feasible": False,
            "all_checks_ok": False,
            "assigned_routes": 0,
            "unassigned_tasks": case.tasks,
            "active_agents": 0,
            "transport_work_ton_km": None,
            "total_km": None,
            "deadhead_km": None,
            "deadhead_share_pct": None,
            "total_hours": None,
        }

    if not result_payload.get("ok", False):
        return {
            "status": "error",
            "solver_error": str(result_payload.get("error", "Worker error")),
            "runtime_sec": round(elapsed, 3),
            "algorithm": algorithm,
            "feasible": False,
            "all_checks_ok": False,
            "assigned_routes": 0,
            "unassigned_tasks": case.tasks,
            "active_agents": 0,
            "transport_work_ton_km": None,
            "total_km": None,
            "deadhead_km": None,
            "deadhead_share_pct": None,
            "total_hours": None,
        }

    data = dict(result_payload.get("data") or {})
    data["runtime_sec"] = round(elapsed, 3)
    data["status"] = "ok"
    return data


def _solver_kwargs_for_algorithm(args: argparse.Namespace, algorithm: str) -> dict[str, Any]:
    if algorithm == "real_milp":
        return {
            "time_limit_sec": int(args.milp_time_limit_sec),
            "unassigned_penalty": float(args.milp_unassigned_penalty),
        }
    if algorithm == "real_gap_vrp":
        return {
            "step1_method": str(args.gap_step1_method),
            "gap_iter": int(args.gap_iter),
            "use_repair": True,
        }
    if algorithm == "real_genetic":
        return {
            "population_size": int(args.genetic_population),
            "generations": int(args.genetic_generations),
            "generation_scale": float(args.genetic_generation_scale),
            "min_generations": int(args.genetic_min_generations),
            "elite_size": int(args.genetic_elite_size),
            "seed": int(args.seed),
            "max_runtime_sec": float(args.genetic_time_limit_sec),
        }
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def _save_materialized_outputs(local_root: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    state_dir = local_root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    df = _build_summary_frame(rows)
    csv_path = state_dir / "results_latest.csv"
    json_path = state_dir / "results_latest.json"

    if df.empty:
        csv_path.write_text("", encoding="utf-8")
        json_path.write_text("[]", encoding="utf-8")
        return csv_path, json_path

    df.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return csv_path, json_path


def _save_run_snapshot(local_root: Path, run_id: str, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    out_dir = local_root / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _build_summary_frame(rows)

    csv_path = out_dir / f"sweep_run_{run_id}.csv"
    json_path = out_dir / f"sweep_run_{run_id}.json"
    md_path = out_dir / f"sweep_run_{run_id}.md"

    df.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    lines: list[str] = []
    lines.append(f"# Sweep run `{run_id}`")
    lines.append("")
    lines.append(df.to_markdown(index=False))
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, json_path


def _draw_heatmap(
    *,
    df: pd.DataFrame,
    algorithm: str,
    metric: str,
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    part = df[df["algorithm"] == algorithm].copy()
    if part.empty:
        return

    pivot = part.pivot_table(
        index="task_percent",
        columns="agent_percent",
        values=metric,
        aggfunc="mean",
    ).sort_index().sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(10, 8))
    arr = pivot.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(arr)
    img = ax.imshow(masked, aspect="auto", cmap="viridis")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(x) for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(x) for x in pivot.index])
    ax.set_xlabel("Agent percent")
    ax.set_ylabel("Task percent")
    ax.set_title(title)

    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            v = arr[r, c]
            if pd.isna(v):
                txt = "-"
            else:
                txt = f"{v:.1f}"
            ax.text(c, r, txt, ha="center", va="center", color="white", fontsize=8)

    fig.colorbar(img, ax=ax, shrink=0.85)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _draw_3d_surface(
    *,
    df: pd.DataFrame,
    algorithm: str,
    metric: str,
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    part = df[df["algorithm"] == algorithm].copy()
    if part.empty:
        return

    pivot = part.pivot_table(
        index="task_percent",
        columns="agent_percent",
        values=metric,
        aggfunc="mean",
    ).sort_index().sort_index(axis=1)

    z = pivot.to_numpy(dtype=float)
    x_vals = pivot.columns.to_numpy(dtype=float)
    y_vals = pivot.index.to_numpy(dtype=float)
    xx, yy = np.meshgrid(x_vals, y_vals)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xx, yy, np.nan_to_num(z, nan=0.0), cmap="viridis", linewidth=0.0, antialiased=True)
    ax.set_xlabel("Agent percent")
    ax.set_ylabel("Task percent")
    ax.set_zlabel(metric)
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _draw_full_agents_lines(
    *,
    df: pd.DataFrame,
    algorithm: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    part = df[(df["algorithm"] == algorithm) & (df["agent_percent"] == 100)].copy()
    if part.empty:
        return

    part = part.sort_values("task_percent")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(part["task_percent"], part["task_coverage_pct"], marker="o")
    axes[0].set_title("Coverage vs Task% (agents=100%)")
    axes[0].set_xlabel("Task percent")
    axes[0].set_ylabel("Coverage %")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(part["task_percent"], part["active_agents_share_pct"], marker="o", color="#e67e22")
    axes[1].set_title("Active agents share vs Task%")
    axes[1].set_xlabel("Task percent")
    axes[1].set_ylabel("Active agents %")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(part["task_percent"], part["runtime_sec"], marker="o", color="#16a085")
    axes[2].set_title("Runtime vs Task%")
    axes[2].set_xlabel("Task percent")
    axes[2].set_ylabel("Runtime sec")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"{algorithm}: full-agent profile", y=1.03)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_sweep(args: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    mp.set_start_method("spawn", force=True)

    sweep_dir = Path(args.sweep_dir).resolve()
    index_csv = Path(args.index_csv).resolve()
    local_root = Path(args.local_root).resolve()
    state_path = local_root / "state" / "results.jsonl"

    if not sweep_dir.exists():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_dir}")
    if not index_csv.exists():
        raise FileNotFoundError(f"Index CSV not found: {index_csv}")

    _log(t0, f"sweep_dir={sweep_dir}")
    _log(t0, f"index_csv={index_csv}")
    _log(t0, f"local_root={local_root}")

    cases = _read_index(index_csv)
    cases = _filter_cases(
        cases,
        task_percents=_parse_percent_list(args.task_percents),
        agent_percents=_parse_percent_list(args.agent_percents),
    )
    if not cases:
        raise RuntimeError("No sweep cases matched the selected filters")

    prev_rows = _load_state_rows(state_path)
    prev_map = _state_map(prev_rows)

    algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    run_id = _ts()

    total_jobs = len(cases) * len(algorithms)
    _log(t0, f"run_id={run_id}, cases={len(cases)}, algorithms={algorithms}, total_jobs={total_jobs}")

    try:
        from tqdm.auto import tqdm

        progress = tqdm(total=total_jobs, desc="sweep", unit="job")
    except Exception:
        progress = None

    new_rows: list[dict[str, Any]] = []
    for case in cases:
        dataset_path = sweep_dir / case.dataset_file
        if not dataset_path.exists():
            _log(t0, f"dataset missing, skip: {dataset_path}")
            if progress is not None:
                progress.update(len(algorithms))
            continue

        for algorithm in algorithms:
            key = (case.key, algorithm)
            if (not args.force) and key in prev_map:
                if progress is not None:
                    progress.update(1)
                continue

            kwargs = _solver_kwargs_for_algorithm(args, algorithm)
            _log(
                t0,
                f"start {case.dataset_file} | {algorithm} | "
                f"tasks={case.tasks}, agents={case.agents}",
            )
            row = _run_case_with_timeout(
                t0=t0,
                case=case,
                dataset_path=dataset_path,
                algorithm=algorithm,
                solver_kwargs=kwargs,
                timeout_sec=int(args.case_timeout_sec),
                show_solver_progress=bool(args.show_solver_progress),
                show_live_solver_logs=bool(args.show_live_solver_logs),
            )
            row.update(
                {
                    "run_id": run_id,
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "dataset_key": case.key,
                    "dataset_file": case.dataset_file,
                    "dataset_path": str(dataset_path),
                    "task_percent": case.task_percent,
                    "agent_percent": case.agent_percent,
                    "tasks_total": case.tasks,
                    "agents_total": case.agents,
                    "solver_kwargs": kwargs,
                }
            )
            _append_state_row(state_path, row)
            new_rows.append(row)

            # refresh materialized views after each completed job for crash-safe resume
            all_rows = _load_state_rows(state_path)
            _save_materialized_outputs(local_root, all_rows)

            if progress is not None:
                progress.update(1)
                progress.set_postfix(
                    {
                        "algo": algorithm,
                        "t%": case.task_percent,
                        "a%": case.agent_percent,
                        "status": row.get("status", "ok"),
                    }
                )

    if progress is not None:
        progress.close()

    all_rows = _load_state_rows(state_path)
    latest_csv, latest_json = _save_materialized_outputs(local_root, all_rows)
    run_csv, run_json = _save_run_snapshot(local_root, run_id, new_rows)

    _log(t0, f"run snapshot csv: {run_csv}")
    _log(t0, f"run snapshot json: {run_json}")
    _log(t0, f"latest csv      : {latest_csv}")
    _log(t0, f"latest json     : {latest_json}")
    _log(t0, f"new jobs saved  : {len(new_rows)}")

    return 0


def plot_sweep(args: argparse.Namespace) -> int:
    local_root = Path(args.local_root).resolve()
    state_path = local_root / "state" / "results.jsonl"
    rows = _load_state_rows(state_path)
    if not rows:
        print(f"No state rows found: {state_path}")
        return 0

    df = _build_summary_frame(rows)
    if df.empty:
        print("No usable rows in state file")
        return 0

    if args.run_id:
        df = df[df["run_id"] == args.run_id].copy()

    algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    if algorithms:
        df = df[df["algorithm"].isin(algorithms)].copy()

    plots_dir = local_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for algorithm in sorted(df["algorithm"].dropna().unique().tolist()):
        _draw_heatmap(
            df=df,
            algorithm=algorithm,
            metric="task_coverage_pct",
            out_path=plots_dir / f"heatmap_{algorithm}_coverage.png",
            title=f"{algorithm}: task coverage (%)",
        )
        _draw_heatmap(
            df=df,
            algorithm=algorithm,
            metric="active_agents_share_pct",
            out_path=plots_dir / f"heatmap_{algorithm}_active_agents_share.png",
            title=f"{algorithm}: active agents share (%)",
        )
        _draw_heatmap(
            df=df,
            algorithm=algorithm,
            metric="runtime_sec",
            out_path=plots_dir / f"heatmap_{algorithm}_runtime_sec.png",
            title=f"{algorithm}: runtime (sec)",
        )
        _draw_3d_surface(
            df=df,
            algorithm=algorithm,
            metric="task_coverage_pct",
            out_path=plots_dir / f"surface_{algorithm}_coverage.png",
            title=f"{algorithm}: coverage surface",
        )
        _draw_full_agents_lines(
            df=df,
            algorithm=algorithm,
            out_path=plots_dir / f"full_agents_{algorithm}.png",
        )

    summary_csv = plots_dir / "plot_source_summary.csv"
    df.to_csv(summary_csv, index=False)
    print(f"Saved plots to: {plots_dir}")
    print(f"Saved source summary: {summary_csv}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep runner for clean_full_enriched 7x7 datasets with resume and plotting"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run scheduling sweep")
    run.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    run.add_argument("--index-csv", type=Path, default=DEFAULT_INDEX_CSV)
    run.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    run.add_argument("--algorithms", type=str, default="real_milp")
    run.add_argument("--task-percents", type=str, default=None, help="Comma-separated filter, e.g. 5,35,100")
    run.add_argument("--agent-percents", type=str, default=None, help="Comma-separated filter, e.g. 50,100")
    run.add_argument("--force", action="store_true", help="Re-run even if state already has this dataset+algorithm")

    run.add_argument("--case-timeout-sec", type=int, default=600)
    run.add_argument("--show-solver-progress", action="store_true")
    run.add_argument("--show-live-solver-logs", action="store_true")

    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--gap-step1-method", type=str, default="dataset", choices=["dataset", "greedy", "lp"])
    run.add_argument("--gap-iter", type=int, default=20)
    run.add_argument("--milp-time-limit-sec", type=int, default=60)
    run.add_argument("--milp-unassigned-penalty", type=float, default=1e5)
    run.add_argument("--genetic-population", type=int, default=20)
    run.add_argument("--genetic-generations", type=int, default=20)
    run.add_argument("--genetic-generation-scale", type=float, default=1.0)
    run.add_argument("--genetic-min-generations", type=int, default=3)
    run.add_argument("--genetic-elite-size", type=int, default=3)
    run.add_argument("--genetic-time-limit-sec", type=float, default=20.0)

    plot = sub.add_parser("plot", help="Build heatmaps/3D/line plots from saved state")
    plot.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    plot.add_argument("--algorithms", type=str, default="real_milp")
    plot.add_argument("--run-id", type=str, default=None)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "run":
        return run_sweep(args)
    if args.cmd == "plot":
        return plot_sweep(args)

    raise RuntimeError(f"Unsupported command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
