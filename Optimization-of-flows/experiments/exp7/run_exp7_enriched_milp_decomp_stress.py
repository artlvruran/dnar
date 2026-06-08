#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class Exp7Config:
    sweeps_dir: Path
    timeout_sec: int
    objective: str
    time_limit_sec_per_zone: int
    max_pairs_per_bundle: int
    bundle_mass_quantile: float
    bundle_vol_quantile: float
    bundle_fill_factor: float
    bundle_max_tasks: int
    out_dir: Path
    pattern: str
    task_pcts: tuple[int, ...] | None
    max_cases: int | None


def _parse_args() -> Exp7Config:
    p = argparse.ArgumentParser(description="EXP7: decomposed MILP stress test on enriched sweep datasets.")
    p.add_argument(
        "--sweeps-dir",
        type=Path,
        default=Path("demo/data/object_mass_feasible_fullfleet/container_full_split_all_agents/sweeps_task_agent_5pct"),
    )
    p.add_argument("--timeout-sec", type=int, default=300)
    p.add_argument("--objective", choices=["tasks", "volume"], default="tasks")
    p.add_argument("--time-limit-sec-per-zone", type=int, default=25)
    p.add_argument("--max-pairs-per-bundle", type=int, default=90)
    p.add_argument("--bundle-mass-quantile", type=float, default=0.35)
    p.add_argument("--bundle-vol-quantile", type=float, default=0.35)
    p.add_argument("--bundle-fill-factor", type=float, default=0.90)
    p.add_argument("--bundle-max-tasks", type=int, default=8)
    p.add_argument("--out-dir", type=Path, default=Path("experiments/exp7/local"))
    p.add_argument(
        "--pattern",
        type=str,
        default="dataset_real_spb_clean_full_split_by_containers_all_agents_with_distances_t*_a100.json",
    )
    p.add_argument(
        "--task-pcts",
        type=str,
        default="",
        help="Comma-separated task percentages filter, e.g. '1,6,11,16,21'",
    )
    p.add_argument("--max-cases", type=int, default=0, help="Limit number of cases after filtering (0 = all)")
    a = p.parse_args()
    task_pcts: tuple[int, ...] | None = None
    if a.task_pcts.strip():
        vals = [int(x.strip()) for x in a.task_pcts.split(",") if x.strip()]
        task_pcts = tuple(sorted(set(vals)))
    return Exp7Config(
        sweeps_dir=a.sweeps_dir,
        timeout_sec=a.timeout_sec,
        objective=a.objective,
        time_limit_sec_per_zone=a.time_limit_sec_per_zone,
        max_pairs_per_bundle=a.max_pairs_per_bundle,
        bundle_mass_quantile=a.bundle_mass_quantile,
        bundle_vol_quantile=a.bundle_vol_quantile,
        bundle_fill_factor=a.bundle_fill_factor,
        bundle_max_tasks=a.bundle_max_tasks,
        out_dir=a.out_dir,
        pattern=a.pattern,
        task_pcts=task_pcts,
        max_cases=(int(a.max_cases) if int(a.max_cases) > 0 else None),
    )


def _task_pct(dataset_path: Path) -> int:
    return int(dataset_path.stem.split("_t")[-1].split("_a")[0])


def _run_single(dataset_path: Path, cfg: Exp7Config) -> dict[str, object]:
    child_code = r"""
import json
from pathlib import Path
from flowopt.solvers.enriched import solve_enriched_milp_decomposed

p = Path(__import__("sys").argv[1])
objective = __import__("sys").argv[2]
time_limit_sec_per_zone = int(__import__("sys").argv[3])
max_pairs_per_bundle = int(__import__("sys").argv[4])
bundle_mass_quantile = float(__import__("sys").argv[5])
bundle_vol_quantile = float(__import__("sys").argv[6])
bundle_fill_factor = float(__import__("sys").argv[7])
bundle_max_tasks = int(__import__("sys").argv[8])

res = solve_enriched_milp_decomposed(
    dataset_path=p,
    objective=objective,
    time_limit_sec_per_zone=time_limit_sec_per_zone,
    max_pairs_per_bundle=max_pairs_per_bundle,
    bundle_mass_quantile=bundle_mass_quantile,
    bundle_vol_quantile=bundle_vol_quantile,
    bundle_fill_factor=bundle_fill_factor,
    bundle_max_tasks=bundle_max_tasks,
    verbose=False,
)
d = res.as_dict()
checks = (d.get("details") or {}).get("checks") or {}
print(json.dumps({
    "algorithm": d.get("algorithm"),
    "feasible": d.get("feasible"),
    "all_checks_ok": checks.get("all_checks_ok"),
    "assigned_tasks": d.get("assigned_routes"),
    "assigned_trips": d.get("assigned_trips"),
    "unassigned_tasks": d.get("unassigned_tasks"),
    "active_agents": d.get("active_agents"),
    "total_km": d.get("total_km"),
    "transport_work_ton_km": d.get("transport_work_ton_km"),
    "runtime_sec": d.get("runtime_sec"),
    "volume_coverage_pct": (d.get("details") or {}).get("volume_coverage_pct"),
}, ensure_ascii=False))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    try:
        cp = subprocess.run(
            [
                sys.executable,
                "-c",
                child_code,
                str(dataset_path),
                cfg.objective,
                str(cfg.time_limit_sec_per_zone),
                str(cfg.max_pairs_per_bundle),
                str(cfg.bundle_mass_quantile),
                str(cfg.bundle_vol_quantile),
                str(cfg.bundle_fill_factor),
                str(cfg.bundle_max_tasks),
            ],
            text=True,
            capture_output=True,
            timeout=cfg.timeout_sec,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "feasible": False,
            "all_checks_ok": False,
            "assigned_tasks": None,
            "assigned_trips": None,
            "unassigned_tasks": None,
            "active_agents": None,
            "total_km": None,
            "transport_work_ton_km": None,
            "runtime_sec": cfg.timeout_sec,
            "volume_coverage_pct": None,
            "solver_error": f"Timeout>{cfg.timeout_sec}s",
        }

    if cp.returncode != 0:
        tail = (cp.stderr or cp.stdout or "").strip().splitlines()
        msg = tail[-1] if tail else f"returncode={cp.returncode}"
        return {
            "status": "error",
            "feasible": False,
            "all_checks_ok": False,
            "assigned_tasks": None,
            "assigned_trips": None,
            "unassigned_tasks": None,
            "active_agents": None,
            "total_km": None,
            "transport_work_ton_km": None,
            "runtime_sec": None,
            "volume_coverage_pct": None,
            "solver_error": msg,
        }

    lines = (cp.stdout or "").strip().splitlines()
    if not lines:
        return {
            "status": "error",
            "feasible": False,
            "all_checks_ok": False,
            "assigned_tasks": None,
            "assigned_trips": None,
            "unassigned_tasks": None,
            "active_agents": None,
            "total_km": None,
            "transport_work_ton_km": None,
            "runtime_sec": None,
            "volume_coverage_pct": None,
            "solver_error": "No JSON output from child process",
        }

    try:
        ans = json.loads(lines[-1])
    except Exception as e:  # noqa: BLE001
        return {
            "status": "error",
            "feasible": False,
            "all_checks_ok": False,
            "assigned_tasks": None,
            "assigned_trips": None,
            "unassigned_tasks": None,
            "active_agents": None,
            "total_km": None,
            "transport_work_ton_km": None,
            "runtime_sec": None,
            "volume_coverage_pct": None,
            "solver_error": f"JSON parse error: {e}",
        }

    return {
        "status": "ok",
        "feasible": bool(ans.get("feasible")),
        "all_checks_ok": bool(ans.get("all_checks_ok")),
        "assigned_tasks": ans.get("assigned_tasks"),
        "assigned_trips": ans.get("assigned_trips"),
        "unassigned_tasks": ans.get("unassigned_tasks"),
        "active_agents": ans.get("active_agents"),
        "total_km": ans.get("total_km"),
        "transport_work_ton_km": ans.get("transport_work_ton_km"),
        "runtime_sec": ans.get("runtime_sec"),
        "volume_coverage_pct": ans.get("volume_coverage_pct"),
        "solver_error": None,
        "algorithm": ans.get("algorithm"),
    }


def _save_plots(df: pd.DataFrame, out_dir: Path) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    x = df["task_pct"].astype(float)
    total_tasks = df["assigned_tasks"].fillna(0) + df["unassigned_tasks"].fillna(0)
    coverage = (100.0 * df["assigned_tasks"] / total_tasks).fillna(0.0)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(x, coverage, marker="o", color="#1f77b4", label="Task coverage %")
    ax1.set_xlabel("Task percent in sweep dataset")
    ax1.set_ylabel("Task coverage %", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_ylim(0, 105)
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, df["runtime_sec"], marker="s", color="#d62728", label="Runtime sec")
    ax2.set_ylabel("Runtime (sec)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    plt.title("EXP7: MILP-decomposed stress (coverage vs runtime)")
    fig.tight_layout()
    fig.savefig(plots_dir / "coverage_runtime.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, df["assigned_tasks"], marker="o", label="assigned_tasks")
    ax.plot(x, df["unassigned_tasks"], marker="o", label="unassigned_tasks")
    ax.set_xlabel("Task percent")
    ax.set_ylabel("Tasks count")
    ax.set_title("EXP7: Assigned vs unassigned tasks")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "assigned_unassigned.png", dpi=160)
    plt.close(fig)

    df.assign(task_coverage_pct=coverage).to_csv(plots_dir / "summary_with_coverage.csv", index=False)


def main() -> None:
    cfg = _parse_args()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(cfg.sweeps_dir.glob(cfg.pattern), key=_task_pct)
    if cfg.task_pcts is not None:
        allowed = set(cfg.task_pcts)
        files = [f for f in files if _task_pct(f) in allowed]
    if cfg.max_cases is not None:
        files = files[: cfg.max_cases]
    if not files:
        raise FileNotFoundError(f"No datasets by pattern {cfg.pattern} in {cfg.sweeps_dir}")

    rows: list[dict[str, object]] = []
    for i, dataset_file in enumerate(files, start=1):
        tp = _task_pct(dataset_file)
        print(f"[{i}/{len(files)}] t={tp:03d}% -> {dataset_file.name}", flush=True)
        row = _run_single(dataset_file, cfg)
        row["task_pct"] = tp
        row["dataset_file"] = dataset_file.name
        rows.append(row)
        print(
            f"    status={row.get('status')} feasible={row.get('feasible')} "
            f"assigned={row.get('assigned_tasks')} unassigned={row.get('unassigned_tasks')} "
            f"runtime={row.get('runtime_sec')} volume_cov={row.get('volume_coverage_pct')}",
            flush=True,
        )

    df = pd.DataFrame(rows).sort_values("task_pct").reset_index(drop=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_out = cfg.out_dir / f"exp7_enriched_milp_decomp_stress_{stamp}.json"
    csv_out = cfg.out_dir / f"exp7_enriched_milp_decomp_stress_{stamp}.csv"
    latest_out = cfg.out_dir / "exp7_enriched_milp_decomp_stress_latest.csv"

    payload = {
        "created_at": stamp,
        "timeout_sec": cfg.timeout_sec,
        "solver": "enriched_milp_decomp_v1",
        "config": {
            "objective": cfg.objective,
            "time_limit_sec_per_zone": cfg.time_limit_sec_per_zone,
            "max_pairs_per_bundle": cfg.max_pairs_per_bundle,
            "bundle_mass_quantile": cfg.bundle_mass_quantile,
            "bundle_vol_quantile": cfg.bundle_vol_quantile,
            "bundle_fill_factor": cfg.bundle_fill_factor,
            "bundle_max_tasks": cfg.bundle_max_tasks,
            "pattern": cfg.pattern,
        },
        "rows": rows,
    }
    json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    df.to_csv(csv_out, index=False)
    df.to_csv(latest_out, index=False)
    _save_plots(df, cfg.out_dir)

    print("\nSaved:")
    print(json_out)
    print(csv_out)
    print(latest_out)
    print("\nSummary:")
    cols = ["task_pct", "status", "feasible", "assigned_tasks", "unassigned_tasks", "runtime_sec", "volume_coverage_pct"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
