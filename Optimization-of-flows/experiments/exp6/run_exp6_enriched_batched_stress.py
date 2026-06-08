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
class Exp6Config:
    sweeps_dir: Path
    timeout_sec: int
    top_k_agents: int
    balance_penalty: float
    random_seed: int
    out_dir: Path
    pattern: str


def _parse_args() -> Exp6Config:
    p = argparse.ArgumentParser(description="EXP6: stress-test enriched batched greedy solver on task sweep.")
    p.add_argument(
        "--sweeps-dir",
        type=Path,
        default=Path("demo/data/object_mass_feasible_fullfleet/container_full_split_all_agents/sweeps_task_agent_5pct"),
    )
    p.add_argument("--timeout-sec", type=int, default=180)
    p.add_argument("--top-k-agents", type=int, default=20)
    p.add_argument("--balance-penalty", type=float, default=0.03)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=Path("experiments/exp6/local"))
    p.add_argument(
        "--pattern",
        type=str,
        default="dataset_real_spb_clean_full_split_by_containers_all_agents_with_distances_t*_a100.json",
    )
    a = p.parse_args()
    return Exp6Config(
        sweeps_dir=a.sweeps_dir,
        timeout_sec=a.timeout_sec,
        top_k_agents=a.top_k_agents,
        balance_penalty=a.balance_penalty,
        random_seed=a.random_seed,
        out_dir=a.out_dir,
        pattern=a.pattern,
    )


def _task_pct(dataset_path: Path) -> int:
    return int(dataset_path.stem.split("_t")[-1].split("_a")[0])


def _run_single(dataset_path: Path, cfg: Exp6Config) -> dict[str, object]:
    child_code = r"""
import json
from pathlib import Path
from flowopt.solvers.enriched import solve_enriched_batched_greedy

p = Path(__import__("sys").argv[1])
seed = int(__import__("sys").argv[2])
top_k = int(__import__("sys").argv[3])
balance = float(__import__("sys").argv[4])
res = solve_enriched_batched_greedy(
    dataset_path=p,
    random_seed=seed,
    top_k_agents=top_k,
    balance_penalty=balance,
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
                str(cfg.random_seed),
                str(cfg.top_k_agents),
                str(cfg.balance_penalty),
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
    ax1.plot(x, coverage, marker="o", color="#13795b", label="Coverage %")
    ax1.set_xlabel("Task percent in sweep dataset")
    ax1.set_ylabel("Coverage %", color="#13795b")
    ax1.tick_params(axis="y", labelcolor="#13795b")
    ax1.set_ylim(0, 105)
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, df["runtime_sec"], marker="s", color="#c0392b", label="Runtime sec")
    ax2.set_ylabel("Runtime (sec)", color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")

    plt.title("EXP6: Enriched batched greedy stress (coverage vs runtime)")
    fig.tight_layout()
    fig.savefig(plots_dir / "coverage_runtime.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, df["assigned_tasks"], marker="o", label="assigned_tasks")
    ax.plot(x, df["unassigned_tasks"], marker="o", label="unassigned_tasks")
    ax.set_xlabel("Task percent")
    ax.set_ylabel("Tasks count")
    ax.set_title("EXP6: Assigned vs unassigned tasks")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "assigned_unassigned.png", dpi=160)
    plt.close(fig)

    out_cols = [
        "task_pct",
        "status",
        "feasible",
        "all_checks_ok",
        "assigned_tasks",
        "unassigned_tasks",
        "runtime_sec",
        "total_km",
        "transport_work_ton_km",
    ]
    df.assign(coverage_pct=coverage).to_csv(plots_dir / "summary_with_coverage.csv", index=False, columns=out_cols + ["coverage_pct"])


def main() -> None:
    cfg = _parse_args()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(cfg.sweeps_dir.glob(cfg.pattern), key=_task_pct)
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
            f"runtime={row.get('runtime_sec')}",
            flush=True,
        )

    df = pd.DataFrame(rows).sort_values("task_pct").reset_index(drop=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_out = cfg.out_dir / f"exp6_enriched_batched_stress_{stamp}.json"
    csv_out = cfg.out_dir / f"exp6_enriched_batched_stress_{stamp}.csv"
    latest_out = cfg.out_dir / "exp6_enriched_batched_stress_latest.csv"
    payload = {
        "created_at": stamp,
        "timeout_sec": cfg.timeout_sec,
        "solver": "enriched_batched_greedy_v1",
        "legacy_alias": "solve_enriched_gap_vrp",
        "config": {
            "top_k_agents": cfg.top_k_agents,
            "balance_penalty": cfg.balance_penalty,
            "random_seed": cfg.random_seed,
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
    cols = ["task_pct", "status", "feasible", "assigned_tasks", "unassigned_tasks", "runtime_sec"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()

