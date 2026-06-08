from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from flowopt.backend.io import load_payload
from flowopt.dataset import CONTAINER_TO_VEHICLE_TYPES
from flowopt.solvers.enriched.common import SERVICE_HOURS_BY_CONTAINER
from flowopt.solvers.enriched.problem import build_enriched_problem


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_JSON = (
    ROOT
    / "demo/data/object_mass_feasible_fullfleet/container_full_split_all_agents"
    / "dataset_real_spb_clean_full_split_by_containers_all_agents.json"
)
DEFAULT_DATASET_GZ = DEFAULT_DATASET_JSON.with_suffix(".json.gz")
OUT_DIR = ROOT / "experiments/local/exp9_potential_capacity"


@dataclass(frozen=True)
class TaskCandidate:
    task_id: str
    destination_node_id: str
    mass_tons: float
    volume_raw_m3: float
    loaded_km: float
    min_trip_hours: float


def ensure_dataset_json(path: Path) -> Path:
    if path.exists():
        return path
    gz_path = path.with_suffix(".json.gz")
    if not gz_path.exists():
        raise FileNotFoundError(f"Dataset not found: {path} and {gz_path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(gz_path, "rt", encoding="utf-8") as src, path.open("w", encoding="utf-8") as dst:
        dst.write(src.read())
    return path


def _agent_matches_task(agent: Any, task: Any) -> bool:
    if not agent.is_available:
        return False
    if task.source_zone_num is not None and agent.zone_num is not None and task.source_zone_num != agent.zone_num:
        return False
    if task.container_type not in CONTAINER_TO_VEHICLE_TYPES.get(task.container_type, set()) and False:
        return False
    if agent.vehicle_type not in CONTAINER_TO_VEHICLE_TYPES.get(task.container_type, set()):
        return False
    if task.compatible_vehicle_types and agent.vehicle_type not in task.compatible_vehicle_types:
        return False
    if agent.cap_container_types and task.container_type not in agent.cap_container_types:
        return False
    if task.source_center and not agent.is_compact:
        return False
    if task.mass_tons > agent.capacity_tons + 1e-9:
        return False
    if agent.max_raw_volume_m3 > 0 and task.volume_raw_m3 > agent.max_raw_volume_m3 + 1e-9:
        return False
    return True


def build_candidates(dataset_path: Path) -> tuple[list[TaskCandidate], dict[str, Any]]:
    problem = build_enriched_problem(dataset_path)
    payload = load_payload(dataset_path)
    raw_task_by_id = {str(t.get("task_id")): t for t in payload.get("tasks", [])}

    active_agents = [a for a in problem.agents if a.is_available]
    agents_zone_none = [a for a in active_agents if a.zone_num is None]
    agents_by_zone: dict[int, list[Any]] = {}
    for a in active_agents:
        if a.zone_num is None:
            continue
        agents_by_zone.setdefault(int(a.zone_num), []).append(a)

    candidates: list[TaskCandidate] = []
    infeasible_no_agent = 0
    for task in tqdm(problem.tasks, desc="exp9:candidate-scan", unit="task"):
        pool = list(agents_zone_none)
        if task.source_zone_num is not None:
            pool.extend(agents_by_zone.get(int(task.source_zone_num), []))
        else:
            pool = active_agents

        best_speed = 0.0
        feasible = False
        for agent in pool:
            if not _agent_matches_task(agent, task):
                continue
            feasible = True
            if agent.avg_speed_kmph > best_speed:
                best_speed = float(agent.avg_speed_kmph)
        if not feasible:
            infeasible_no_agent += 1
            continue

        raw = raw_task_by_id.get(task.task_id, {})
        loaded_km = float(raw.get("nearest_object_distance_km", 0.0) or 0.0)
        if loaded_km <= 0:
            loaded_km = 1e-6
        service_h = float(SERVICE_HOURS_BY_CONTAINER.get(task.container_type, 0.25))
        speed = max(best_speed, 1e-6)
        min_trip_hours = loaded_km / speed + service_h
        candidates.append(
            TaskCandidate(
                task_id=task.task_id,
                destination_node_id=task.destination_node_id,
                mass_tons=float(task.mass_tons),
                volume_raw_m3=float(task.volume_raw_m3),
                loaded_km=float(loaded_km),
                min_trip_hours=float(min_trip_hours),
            )
        )

    total_hours_budget = float(sum(a.max_shift_hours for a in active_agents))
    total_km_budget = float(sum(a.max_daily_km for a in active_agents))
    total_mass_capacity_per_trip = float(sum(a.capacity_tons for a in active_agents))
    total_vol_capacity_per_trip = float(sum(max(0.0, a.max_raw_volume_m3) for a in active_agents))

    total_task_mass = float(sum(t.mass_tons for t in problem.tasks))
    total_task_vol = float(sum(t.volume_raw_m3 for t in problem.tasks))
    obj_mass_caps = {k: float(v) for k, v in problem.object_day_capacity_tons.items()}
    obj_vol_caps = {k: float(v) for k, v in problem.object_day_capacity_volume_m3.items()}

    meta = {
        "dataset_path": str(dataset_path),
        "total_tasks": int(len(problem.tasks)),
        "active_agents": int(len(active_agents)),
        "infeasible_no_agent_tasks": int(infeasible_no_agent),
        "feasible_candidate_tasks": int(len(candidates)),
        "total_task_mass_tons": round(total_task_mass, 3),
        "total_task_volume_raw_m3": round(total_task_vol, 3),
        "total_object_day_mass_cap_tons": round(sum(obj_mass_caps.values()), 3),
        "total_object_day_volume_cap_m3": round(sum(obj_vol_caps.values()), 3),
        "total_fleet_hours_budget": round(total_hours_budget, 3),
        "total_fleet_km_budget": round(total_km_budget, 3),
        "total_fleet_mass_capacity_per_trip_tons": round(total_mass_capacity_per_trip, 3),
        "total_fleet_raw_volume_capacity_per_trip_m3": round(total_vol_capacity_per_trip, 3),
        "object_mass_caps": obj_mass_caps,
        "object_vol_caps": obj_vol_caps,
        "hours_budget": total_hours_budget,
        "km_budget": total_km_budget,
    }
    return candidates, meta


def run_object_only(candidates: list[TaskCandidate], meta: dict[str, Any]) -> dict[str, Any]:
    by_obj: dict[str, list[TaskCandidate]] = {}
    for t in candidates:
        by_obj.setdefault(t.destination_node_id, []).append(t)

    selected = 0
    mass = 0.0
    vol = 0.0
    for oid, arr in by_obj.items():
        m_cap = float(meta["object_mass_caps"].get(oid, 0.0))
        v_cap = float(meta["object_vol_caps"].get(oid, 0.0))
        if m_cap <= 0 and v_cap <= 0:
            continue
        arr = sorted(
            arr,
            key=lambda x: (x.mass_tons / max(m_cap, 1e-9)) + (x.volume_raw_m3 / max(v_cap, 1e-9)),
        )
        m_used = 0.0
        v_used = 0.0
        for t in arr:
            if m_cap > 0 and (m_used + t.mass_tons) > m_cap + 1e-9:
                continue
            if v_cap > 0 and (v_used + t.volume_raw_m3) > v_cap + 1e-9:
                continue
            selected += 1
            mass += t.mass_tons
            vol += t.volume_raw_m3
            m_used += t.mass_tons
            v_used += t.volume_raw_m3

    total_tasks = int(meta["total_tasks"])
    return {
        "scenario": "object_only",
        "selected_tasks": int(selected),
        "coverage_pct": round(100.0 * selected / max(total_tasks, 1), 3),
        "selected_mass_tons": round(mass, 3),
        "selected_volume_m3": round(vol, 3),
    }


def run_aggregate_budget(
    candidates: list[TaskCandidate],
    meta: dict[str, Any],
    *,
    trip_km_factor: float,
) -> dict[str, Any]:
    rem_h = float(meta["hours_budget"])
    rem_km = float(meta["km_budget"])
    rem_m = {k: float(v) for k, v in meta["object_mass_caps"].items()}
    rem_v = {k: float(v) for k, v in meta["object_vol_caps"].items()}

    arr = sorted(
        candidates,
        key=lambda t: (t.min_trip_hours / max(meta["hours_budget"], 1e-9))
        + ((t.loaded_km * trip_km_factor) / max(meta["km_budget"], 1e-9)),
    )

    selected = 0
    mass = 0.0
    vol = 0.0
    used_h = 0.0
    used_km = 0.0
    for t in arr:
        oid = t.destination_node_id
        need_h = float(t.min_trip_hours)
        need_km = float(t.loaded_km) * float(trip_km_factor)
        if need_h > rem_h + 1e-9:
            continue
        if need_km > rem_km + 1e-9:
            continue
        if rem_m.get(oid, 0.0) < t.mass_tons - 1e-9:
            continue
        if rem_v.get(oid, 0.0) < t.volume_raw_m3 - 1e-9:
            continue
        rem_h -= need_h
        rem_km -= need_km
        rem_m[oid] = rem_m.get(oid, 0.0) - t.mass_tons
        rem_v[oid] = rem_v.get(oid, 0.0) - t.volume_raw_m3
        selected += 1
        mass += t.mass_tons
        vol += t.volume_raw_m3
        used_h += need_h
        used_km += need_km

    total_tasks = int(meta["total_tasks"])
    return {
        "scenario": f"aggregate_budget_km_factor_{trip_km_factor:.2f}",
        "selected_tasks": int(selected),
        "coverage_pct": round(100.0 * selected / max(total_tasks, 1), 3),
        "selected_mass_tons": round(mass, 3),
        "selected_volume_m3": round(vol, 3),
        "used_hours": round(used_h, 3),
        "used_km": round(used_km, 3),
        "hours_utilization_pct": round(100.0 * used_h / max(float(meta["hours_budget"]), 1e-9), 3),
        "km_utilization_pct": round(100.0 * used_km / max(float(meta["km_budget"]), 1e-9), 3),
    }


def plot_scenarios(df: pd.DataFrame, out_dir: Path) -> None:
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    chart = df.copy()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(chart["scenario"], chart["coverage_pct"])
    ax.set_ylabel("Coverage, % of tasks")
    ax.set_title("EXP9: Potential task coverage under different bounds")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(plots / "coverage_scenarios.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(chart["scenario"], chart["selected_mass_tons"], label="selected mass (t)")
    ax2 = ax.twinx()
    ax2.plot(chart["scenario"], chart["selected_tasks"], color="black", marker="o", label="selected tasks")
    ax.set_ylabel("Mass, tons")
    ax2.set_ylabel("Tasks")
    ax.set_title("EXP9: Selected mass and tasks by scenario")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(plots / "mass_tasks_scenarios.png", dpi=170)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="EXP9: potential capacity report for enriched full dataset")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_JSON)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--km-factors", type=str, default="1.0,2.5")
    args = p.parse_args()

    ds = ensure_dataset_json(args.dataset.resolve())
    candidates, meta = build_candidates(ds)

    scenarios: list[dict[str, Any]] = []
    scenarios.append(run_object_only(candidates, meta))
    for raw in [x.strip() for x in args.km_factors.split(",") if x.strip()]:
        scenarios.append(run_aggregate_budget(candidates, meta, trip_km_factor=float(raw)))

    df = pd.DataFrame(scenarios).sort_values("coverage_pct", ascending=False).reset_index(drop=True)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta_path = out_dir / f"exp9_potential_meta_{stamp}.json"
    csv_path = out_dir / f"exp9_potential_scenarios_{stamp}.csv"
    json_path = out_dir / f"exp9_potential_scenarios_{stamp}.json"
    latest_csv = out_dir / "exp9_potential_scenarios_latest.csv"
    latest_meta = out_dir / "exp9_potential_meta_latest.json"

    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    df.to_csv(csv_path, index=False)
    df.to_csv(latest_csv, index=False)
    json_path.write_text(json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2), encoding="utf-8")
    plot_scenarios(df, out_dir)

    print(f"Saved: {meta_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
