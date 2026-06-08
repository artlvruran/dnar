from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from flowopt import core
from flowopt.backend.io import load_dataset
from flowopt.solvers.enriched.common import SERVICE_HOURS_BY_CONTAINER
from flowopt.solvers.enriched.distance_oracle import DistanceOracleWithFallback, PrecomputedDistanceOracle
from flowopt.solvers.enriched.problem import build_enriched_problem
from flowopt.solvers.enriched.runner_v2 import solve_enriched_milp_ablation_zone_bundle


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = (
    ROOT
    / "demo/data/object_mass_feasible_fullfleet/container_full_split_all_agents/sweeps_task_agent_5pct"
    / "dataset_real_spb_clean_full_split_by_containers_all_agents_with_distances_t031_a100.json"
)
OUT_DIR = ROOT / "experiments/local/exp10_bundle_route_cost_compare"


@dataclass(frozen=True)
class RouteRecalc:
    route_id: str
    old_total_km: float
    new_total_km: float
    old_loaded_km: float
    new_loaded_km: float
    old_hours: float
    new_hours: float
    task_count: int
    unique_sources: int
    destination: str


def _build_oracle(dataset_path: Path) -> DistanceOracleWithFallback:
    dataset, payload = load_dataset(dataset_path)
    nx_graph = core.build_nx_graph(dataset)
    precomputed = PrecomputedDistanceOracle.from_dataset_payload(dataset_path=dataset_path, payload=payload)
    return DistanceOracleWithFallback(nx_graph=nx_graph, precomputed=precomputed)


def _nn_path_length(oracle: DistanceOracleWithFallback, depot: str, sources: list[str], dest: str) -> tuple[float, float]:
    # Greedy nearest-neighbor over pickup sources:
    # depot -> sources... -> destination -> depot
    if not sources:
        d1 = oracle.dist(depot, dest)
        d2 = oracle.dist(dest, depot)
        total = d1 + d2 if d1 < float("inf") and d2 < float("inf") else float("inf")
        return total, 0.0

    remaining = sources[:]
    cur = depot
    total = 0.0
    pickup_seq: list[str] = []
    while remaining:
        best_i = -1
        best_d = float("inf")
        for i, s in enumerate(remaining):
            d = oracle.dist(cur, s)
            if d < best_d:
                best_d = d
                best_i = i
        if best_i < 0 or best_d == float("inf"):
            return float("inf"), float("inf")
        nxt = remaining.pop(best_i)
        total += best_d
        cur = nxt
        pickup_seq.append(nxt)

    to_dest = oracle.dist(cur, dest)
    to_depot = oracle.dist(dest, depot)
    if to_dest == float("inf") or to_depot == float("inf"):
        return float("inf"), float("inf")
    total += to_dest + to_depot
    loaded_km = to_dest  # proxy: loaded on leg from last pickup to dump point
    return total, loaded_km


def main() -> None:
    p = argparse.ArgumentParser(description="EXP10: compare bundle cost model vs route-aware pickup sequencing")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--time-limit-sec-per-zone", type=int, default=20)
    p.add_argument("--max-pairs-per-bundle", type=int, default=160)
    p.add_argument("--bundle-max-tasks", type=int, default=12)
    args = p.parse_args()

    dataset_path = args.dataset.resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    result = solve_enriched_milp_ablation_zone_bundle(
        dataset_path=dataset_path,
        time_limit_sec_per_zone=args.time_limit_sec_per_zone,
        max_pairs_per_bundle=args.max_pairs_per_bundle,
        bundle_fill_factor=0.93,
        bundle_max_tasks=args.bundle_max_tasks,
        unassigned_penalty=1e5,
        objective="tasks",
    )
    d = result.as_dict()

    problem = build_enriched_problem(dataset_path)
    task_by_id = {t.task_id: t for t in problem.tasks}
    agent_by_id = {a.agent_id: a for a in problem.agents}
    oracle = _build_oracle(dataset_path)

    rows: list[RouteRecalc] = []
    tr_old = 0.0
    tr_new = 0.0
    total_old = 0.0
    total_new = 0.0
    hours_old = 0.0
    hours_new = 0.0

    for route in result.routes:
        tasks = [task_by_id[tid] for tid in route.task_ids if tid in task_by_id]
        if not tasks:
            continue
        agent = agent_by_id.get(route.agent_id)
        if agent is None or agent.depot_node_id is None:
            continue
        depot = str(agent.depot_node_id)
        dest = str(tasks[0].destination_node_id)
        sources = [str(t.source_node_id) for t in tasks]
        new_total, new_loaded = _nn_path_length(oracle, depot, sources, dest)

        service_h = sum(float(SERVICE_HOURS_BY_CONTAINER.get(t.container_type, 0.25)) for t in tasks)
        new_hours = (new_total / max(agent.avg_speed_kmph, 1e-6)) + service_h if new_total < float("inf") else float("inf")

        old_total = float(route.total_distance_km)
        old_loaded = float(route.loaded_distance_km)
        old_hours = float(route.total_hours)

        rows.append(
            RouteRecalc(
                route_id=route.route_id,
                old_total_km=old_total,
                new_total_km=float(new_total),
                old_loaded_km=old_loaded,
                new_loaded_km=float(new_loaded),
                old_hours=old_hours,
                new_hours=float(new_hours),
                task_count=len(tasks),
                unique_sources=len(set(sources)),
                destination=dest,
            )
        )

        mass_sum = sum(float(t.mass_tons) for t in tasks)
        tr_old += mass_sum * old_loaded
        tr_new += mass_sum * float(new_loaded)
        total_old += old_total
        total_new += float(new_total)
        hours_old += old_hours
        hours_new += float(new_hours)

    df = pd.DataFrame([r.__dict__ for r in rows])
    if not df.empty:
        df["total_km_delta"] = df["new_total_km"] - df["old_total_km"]
        df["total_km_delta_pct"] = 100.0 * df["total_km_delta"] / df["old_total_km"].clip(lower=1e-9)
        df["hours_delta"] = df["new_hours"] - df["old_hours"]

    summary = {
        "dataset_path": str(dataset_path),
        "solver_algorithm": d.get("algorithm"),
        "assigned_tasks": int(d.get("assigned_routes") or 0),
        "unassigned_tasks": int(d.get("unassigned_tasks") or 0),
        "routes_count": int(len(result.routes)),
        "old_total_km": round(total_old, 3),
        "new_total_km_route_aware": round(total_new, 3),
        "total_km_delta": round(total_new - total_old, 3),
        "total_km_delta_pct": round(100.0 * (total_new - total_old) / max(total_old, 1e-9), 3),
        "old_total_hours": round(hours_old, 3),
        "new_total_hours_route_aware": round(hours_new, 3),
        "old_transport_work_ton_km": round(tr_old, 3),
        "new_transport_work_ton_km_route_aware": round(tr_new, 3),
        "transport_work_delta": round(tr_new - tr_old, 3),
        "transport_work_delta_pct": round(100.0 * (tr_new - tr_old) / max(tr_old, 1e-9), 3),
        "multi_source_routes": int((df["unique_sources"] > 1).sum()) if not df.empty else 0,
        "multi_task_routes": int((df["task_count"] > 1).sum()) if not df.empty else 0,
    }

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    route_csv = out_dir / f"exp10_route_compare_{stamp}.csv"
    summary_json = out_dir / f"exp10_summary_{stamp}.json"
    latest_csv = out_dir / "exp10_route_compare_latest.csv"
    latest_json = out_dir / "exp10_summary_latest.json"

    if not df.empty:
        df.sort_values(["total_km_delta", "task_count"], ascending=[False, False]).to_csv(route_csv, index=False)
        df.sort_values(["total_km_delta", "task_count"], ascending=[False, False]).to_csv(latest_csv, index=False)
    else:
        pd.DataFrame().to_csv(route_csv, index=False)
        pd.DataFrame().to_csv(latest_csv, index=False)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved:", route_csv)
    print("Saved:", summary_json)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
