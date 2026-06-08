from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from collections import defaultdict

from flowopt.volume_core import (
    GreedyBatchConfig,
    GreedyBatchVolumeSolver,
    VolumeDataset,
    save_solution_artifacts,
)


ROOT = Path(__file__).resolve().parents[2]
BASE_DATASET = (
    ROOT
    / "demo/data/object_volume_feasible_fullfleet/full_all_tasks_all_agents_stage4_volume_only"
    / "dataset_real_spb_clean_full_all_tasks_all_agents_stage4_volume_only.json"
)
COVERED_ONLY_DATASET = (
    ROOT
    / "demo/data/object_volume_feasible_fullfleet/full_all_tasks_all_agents_stage4_volume_only"
    / "dataset_real_spb_clean_full_all_tasks_all_agents_stage4_volume_only_covered_only.json"
)
DEFAULT_DATASET = COVERED_ONLY_DATASET if COVERED_ONLY_DATASET.exists() else BASE_DATASET
OUT_ROOT = ROOT / "experiments/local/exp13_volume_core_batch"


def _select_tasks_stratified(dataset: VolumeDataset, limit: int) -> None:
    groups: dict[int | None, list] = defaultdict(list)
    for t in dataset.tasks:
        groups[t.source_zone_num].append(t)
    for g in groups.values():
        g.sort(key=lambda x: x.volume_raw_m3, reverse=True)
    keys = list(groups.keys())
    picked = []
    idx = {k: 0 for k in keys}
    while len(picked) < limit:
        progress = False
        for k in keys:
            i = idx[k]
            if i < len(groups[k]):
                picked.append(groups[k][i])
                idx[k] = i + 1
                progress = True
                if len(picked) >= limit:
                    break
        if not progress:
            break
    dataset.tasks = picked


def _diagnose_unassigned(dataset: VolumeDataset, solution, eval_result) -> dict[str, int]:
    task_by_id = {t.task_id: t for t in dataset.tasks}
    agent_by_id = {a.agent_id: a for a in dataset.agents if a.is_active and a.depot_node_id is not None}

    used_km = {aid: 0.0 for aid in agent_by_id}
    used_h = {aid: 0.0 for aid in agent_by_id}
    for tr in solution.trips:
        used_km[tr.agent_id] = used_km.get(tr.agent_id, 0.0) + tr.total_km
        used_h[tr.agent_id] = used_h.get(tr.agent_id, 0.0) + tr.total_hours

    object_used = eval_result.object_volume_used_m3
    object_cap = eval_result.object_volume_capacity_m3

    diag = {
        "no_compatible_agent_static": 0,
        "object_capacity_reached": 0,
        "no_agent_with_remaining_time_or_km": 0,
        "no_agent_with_remaining_volume_capacity": 0,
        "other": 0,
    }

    for tid in solution.unassigned_task_ids:
        t = task_by_id.get(tid)
        if t is None:
            diag["other"] += 1
            continue
        comp = [a for a in agent_by_id.values() if dataset.agent_can_take_task(t, a)]
        if not comp:
            diag["no_compatible_agent_static"] += 1
            continue

        cap = float(object_cap.get(t.destination_node_id, 0.0) or 0.0)
        used = float(object_used.get(t.destination_node_id, 0.0) or 0.0)
        if cap > 0 and used >= cap - 1e-9:
            diag["object_capacity_reached"] += 1
            continue

        time_km_ok = []
        vol_ok = []
        for a in comp:
            rem_km = a.max_daily_km - used_km.get(a.agent_id, 0.0)
            rem_h = a.max_hours - used_h.get(a.agent_id, 0.0)
            if rem_km <= 1e-9 or rem_h <= 1e-9:
                continue
            time_km_ok.append(a)
            eff = dataset.effective_task_volume(t, a)
            if a.max_raw_volume_m3 <= 0 or eff <= a.max_raw_volume_m3 + 1e-9:
                vol_ok.append(a)
        if not time_km_ok:
            diag["no_agent_with_remaining_time_or_km"] += 1
            continue
        if not vol_ok:
            diag["no_agent_with_remaining_volume_capacity"] += 1
            continue
        diag["other"] += 1
    return diag


def main() -> None:
    p = argparse.ArgumentParser(description="EXP13: volume-core greedy batch solver")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--max-runtime-sec", type=float, default=120.0)
    p.add_argument("--top-k-agents", type=int, default=30)
    p.add_argument("--top-k-destinations", type=int, default=3)
    p.add_argument("--max-tasks-in-trip", type=int, default=32)
    p.add_argument("--log-every-sec", type=float, default=5.0)
    p.add_argument("--trip-log-every", type=int, default=100)
    p.add_argument("--override-speed-kmph", type=float, default=0.0)
    p.add_argument("--task-limit", type=int, default=0)
    p.add_argument("--task-sampling", type=str, default="head", choices=["head", "stratified"])
    args = p.parse_args()

    dataset = VolumeDataset.from_json(args.dataset)
    if args.override_speed_kmph and args.override_speed_kmph > 0:
        dataset.agents = tuple(
            a.__class__(**{**a.__dict__, "avg_speed_kmph": float(args.override_speed_kmph)})
            if (a.is_active and a.depot_node_id is not None)
            else a
            for a in dataset.agents
        )
    if args.task_limit and args.task_limit > 0:
        if args.task_sampling == "stratified":
            _select_tasks_stratified(dataset, int(args.task_limit))
        else:
            dataset.tasks = dataset.tasks[: int(args.task_limit)]

    solver = GreedyBatchVolumeSolver(
        GreedyBatchConfig(
            max_runtime_sec=float(args.max_runtime_sec),
            top_k_agents=int(args.top_k_agents),
            top_k_destinations=int(args.top_k_destinations),
            max_tasks_in_trip=int(args.max_tasks_in_trip),
            log_every_sec=float(args.log_every_sec),
            trip_log_every=int(args.trip_log_every),
        )
    )

    solution = solver.solve(dataset)
    evaluation = dataset.evaluate(solution)
    unassigned_diag = _diagnose_unassigned(dataset, solution, evaluation)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / f"run_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = save_solution_artifacts(dataset=dataset, solution=solution, evaluation=evaluation, out_dir=out_dir)

    summary_dict = evaluation.as_dict()
    summary_dict["unassigned_diagnostics"] = unassigned_diag
    summary_df = pd.DataFrame([summary_dict])
    summary_csv = out_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    latest_dir = OUT_ROOT / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    (latest_dir / "summary.json").write_text(json.dumps(summary_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_df.to_csv(latest_dir / "summary.csv", index=False)

    idx = {
        "run_dir": str(out_dir),
        "dataset": str(Path(args.dataset).resolve()),
        "params": {
            "max_runtime_sec": float(args.max_runtime_sec),
            "top_k_agents": int(args.top_k_agents),
            "top_k_destinations": int(args.top_k_destinations),
            "max_tasks_in_trip": int(args.max_tasks_in_trip),
            "log_every_sec": float(args.log_every_sec),
            "trip_log_every": int(args.trip_log_every),
            "override_speed_kmph": float(args.override_speed_kmph),
            "task_limit": int(args.task_limit),
        },
        "summary": summary_dict,
        "artifacts": {**paths, "summary_csv": str(summary_csv)},
    }
    (out_dir / "index.json").write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")
    (latest_dir / "index.json").write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Run dir: {out_dir}")
    print(summary_df.to_string(index=False))
    print("Artifacts:")
    for k, v in paths.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
