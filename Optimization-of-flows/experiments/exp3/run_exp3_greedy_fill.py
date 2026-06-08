from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from flowopt import core
from flowopt.backend.io import load_dataset
from flowopt.dataset import CONTAINER_TO_VEHICLE_TYPES, RoutingDataset


@dataclass
class AgentRun:
    agent_id: str
    vehicle_type: str
    depot_node: str | None
    tasks_count: int
    trips_count: int
    mass_tons: float
    pickup_km: float
    unload_km: float
    return_km: float
    total_km: float
    total_hours: float
    km_limit: float
    hour_limit: float
    km_ok: bool
    hours_ok: bool
    assigned_task_ids: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "vehicle_type": self.vehicle_type,
            "depot_node": self.depot_node,
            "tasks_count": self.tasks_count,
            "trips_count": self.trips_count,
            "mass_tons": round(self.mass_tons, 3),
            "pickup_km": round(self.pickup_km, 3),
            "unload_km": round(self.unload_km, 3),
            "return_km": round(self.return_km, 3),
            "total_km": round(self.total_km, 3),
            "total_hours": round(self.total_hours, 3),
            "km_limit": round(self.km_limit, 3),
            "hour_limit": round(self.hour_limit, 3),
            "km_ok": self.km_ok,
            "hours_ok": self.hours_ok,
            "assigned_task_ids": self.assigned_task_ids,
        }


def _log(enabled: bool, t0: float, msg: str) -> None:
    if not enabled:
        return
    dt = time.perf_counter() - t0
    print(f"[+{dt:7.1f}s] {msg}", flush=True)


def _sp_dist(
    graph,
    cache: dict[tuple[str, str], tuple[list[str], float] | None],
    src: str,
    dst: str,
) -> float:
    res = core.shortest_path_cached(graph, cache, src, dst)
    if res is None:
        return float("inf")
    return float(res[1])


def _pick_nearest_candidate(
    *,
    current_node: str,
    current_xy: tuple[float, float],
    candidate_indices: np.ndarray,
    source_nodes: list[str],
    source_xy: np.ndarray,
    graph,
    cache: dict[tuple[str, str], tuple[list[str], float] | None],
    shortlist_k: int,
) -> tuple[int | None, float]:
    if candidate_indices.size == 0:
        return None, float("inf")

    cx, cy = current_xy
    dx = source_xy[candidate_indices, 0] - cx
    dy = source_xy[candidate_indices, 1] - cy
    e2 = dx * dx + dy * dy

    k = min(max(1, int(shortlist_k)), candidate_indices.size)
    if k < candidate_indices.size:
        local = np.argpartition(e2, k - 1)[:k]
        cands = candidate_indices[local]
    else:
        cands = candidate_indices

    best_idx: int | None = None
    best_dist = float("inf")
    for tidx in cands.tolist():
        d = _sp_dist(graph, cache, current_node, source_nodes[tidx])
        if d < best_dist:
            best_dist = d
            best_idx = tidx

    return best_idx, best_dist


def run_greedy_fill(
    *,
    dataset: RoutingDataset,
    payload: dict[str, Any],
    max_agents: int | None,
    max_trips_per_agent: int,
    max_pickups_per_trip: int,
    shortlist_k: int,
    show_progress: bool,
) -> dict[str, Any]:
    t0 = time.perf_counter()

    tasks = list(dataset.tasks)
    n_tasks = len(tasks)
    graph = core.build_nx_graph(dataset)
    cache: dict[tuple[str, str], tuple[list[str], float] | None] = {}

    states = core.initialize_agent_states(dataset, payload)
    agent_ids = list(states.keys())
    if max_agents is not None:
        agent_ids = agent_ids[: max(0, int(max_agents))]

    _log(show_progress, t0, f"dataset tasks={n_tasks}, agents={len(agent_ids)}")

    node_xy = {nid: (node.x, node.y) for nid, node in dataset.graph.nodes.items()}

    source_nodes = [t.source_node_id for t in tasks]
    dest_nodes = [t.destination_node_id for t in tasks]
    dest_arr = np.array(dest_nodes, dtype=object)
    source_xy = np.array([node_xy[s] for s in source_nodes], dtype=float)

    task_ids = [t.task_id for t in tasks]
    task_mass = np.array([t.mass_tons for t in tasks], dtype=float)
    task_service = np.array([core.SERVICE_HOURS_BY_CONTAINER[t.container_type] for t in tasks], dtype=float)
    task_container = np.array([t.container_type for t in tasks], dtype=object)
    task_center = np.array([dataset.graph.nodes[t.source_node_id].center for t in tasks], dtype=bool)

    # Unassigned task mask (True -> still available)
    free = np.ones(n_tasks, dtype=bool)

    # Container compatibility masks per vehicle type.
    vehicle_container_masks: dict[str, np.ndarray] = {}
    for aid in agent_ids:
        vt = states[aid].vehicle_type
        if vt in vehicle_container_masks:
            continue
        allowed = {c for c, vset in CONTAINER_TO_VEHICLE_TYPES.items() if vt in vset}
        vehicle_container_masks[vt] = np.isin(task_container, np.array(sorted(allowed), dtype=object))

    # Agent order: larger capacity first.
    agent_ids_sorted = sorted(
        agent_ids,
        key=lambda aid: states[aid].capacity_tons,
        reverse=True,
    )

    task_to_agent: dict[str, str] = {}
    task_to_trip: dict[str, int] = {}
    per_agent_runs: list[AgentRun] = []

    report_every = max(1, len(agent_ids_sorted) // 10)

    for a_pos, aid in enumerate(agent_ids_sorted, start=1):
        st = states[aid]
        depot = st.depot_node
        if depot is None:
            continue

        speed = core.AVG_SPEED_KMPH_BY_TYPE[st.vehicle_type]
        km_limit = core.MAX_DAILY_KM_BY_TYPE[st.vehicle_type]
        h_limit = core.MAX_SHIFT_HOURS_BY_TYPE[st.vehicle_type]

        compat_mask_base = vehicle_container_masks[st.vehicle_type].copy()
        compat_mask_base &= (task_mass <= st.capacity_tons + 1e-9)
        if not st.is_compact:
            compat_mask_base &= ~task_center

        current_node = depot
        current_xy = node_xy[depot]

        day_pickup_km = 0.0
        day_unload_km = 0.0
        day_return_km = 0.0
        day_service_h = 0.0
        day_km_no_return = 0.0
        trip_counter = 0
        assigned_idxs: list[int] = []

        while trip_counter < max_trips_per_agent:
            cap_left = st.capacity_tons
            trip_dest: str | None = None
            trip_last_node = current_node
            trip_last_xy = current_xy
            trip_pickup_km = 0.0
            trip_service_h = 0.0
            trip_task_idxs: list[int] = []

            for _ in range(max_pickups_per_trip):
                cand = free & compat_mask_base & (task_mass <= cap_left + 1e-9)
                if trip_dest is not None:
                    cand &= (dest_arr == trip_dest)
                cand_idx = np.flatnonzero(cand)
                if cand_idx.size == 0:
                    break

                pick_idx, leg_to_src = _pick_nearest_candidate(
                    current_node=trip_last_node,
                    current_xy=trip_last_xy,
                    candidate_indices=cand_idx,
                    source_nodes=source_nodes,
                    source_xy=source_xy,
                    graph=graph,
                    cache=cache,
                    shortlist_k=shortlist_k,
                )
                if pick_idx is None or not np.isfinite(leg_to_src):
                    break

                this_dest = dest_nodes[pick_idx] if trip_dest is None else trip_dest
                leg_to_unload = _sp_dist(graph, cache, source_nodes[pick_idx], this_dest)
                leg_back = _sp_dist(graph, cache, this_dest, depot)
                if not np.isfinite(leg_to_unload) or not np.isfinite(leg_back):
                    break

                projected_km = (
                    day_km_no_return
                    + trip_pickup_km
                    + leg_to_src
                    + leg_to_unload
                    + leg_back
                )
                projected_service = day_service_h + trip_service_h + float(task_service[pick_idx])
                projected_hours = projected_service + projected_km / speed

                if projected_km > km_limit + 1e-9 or projected_hours > h_limit + 1e-9:
                    break

                # accept task
                trip_dest = this_dest
                trip_task_idxs.append(int(pick_idx))
                free[pick_idx] = False

                trip_pickup_km += float(leg_to_src)
                trip_service_h += float(task_service[pick_idx])
                cap_left -= float(task_mass[pick_idx])

                trip_last_node = source_nodes[pick_idx]
                trip_last_xy = node_xy[trip_last_node]

            if not trip_task_idxs:
                break

            assert trip_dest is not None
            leg_unload = _sp_dist(graph, cache, trip_last_node, trip_dest)
            if not np.isfinite(leg_unload):
                # rollback defensive
                for tidx in trip_task_idxs:
                    free[tidx] = True
                break

            trip_counter += 1
            day_pickup_km += trip_pickup_km
            day_unload_km += float(leg_unload)
            day_service_h += trip_service_h
            day_km_no_return += trip_pickup_km + float(leg_unload)

            current_node = trip_dest
            current_xy = node_xy[current_node]

            for tidx in trip_task_idxs:
                task_to_agent[task_ids[tidx]] = aid
                task_to_trip[task_ids[tidx]] = trip_counter
            assigned_idxs.extend(trip_task_idxs)

        if current_node != depot:
            ret = _sp_dist(graph, cache, current_node, depot)
            if np.isfinite(ret):
                day_return_km += float(ret)
            else:
                day_return_km += 1e9

        total_km = day_pickup_km + day_unload_km + day_return_km
        total_hours = day_service_h + total_km / speed

        ar = AgentRun(
            agent_id=aid,
            vehicle_type=st.vehicle_type,
            depot_node=depot,
            tasks_count=len(assigned_idxs),
            trips_count=trip_counter,
            mass_tons=float(np.sum(task_mass[assigned_idxs])) if assigned_idxs else 0.0,
            pickup_km=day_pickup_km,
            unload_km=day_unload_km,
            return_km=day_return_km,
            total_km=total_km,
            total_hours=total_hours,
            km_limit=km_limit,
            hour_limit=h_limit,
            km_ok=bool(total_km <= km_limit + 1e-9),
            hours_ok=bool(total_hours <= h_limit + 1e-9),
            assigned_task_ids=[task_ids[i] for i in assigned_idxs],
        )
        per_agent_runs.append(ar)

        if a_pos % report_every == 0 or a_pos == len(agent_ids_sorted):
            assigned_cnt = int(np.sum(~free))
            _log(
                show_progress,
                t0,
                f"agents {a_pos}/{len(agent_ids_sorted)} processed, assigned={assigned_cnt}/{n_tasks}",
            )

    assigned_count = int(np.sum(~free))
    unassigned_indices = np.flatnonzero(free)
    unassigned_ids = [task_ids[i] for i in unassigned_indices.tolist()]

    active_agents = [r for r in per_agent_runs if r.tasks_count > 0]
    limit_violations = [r for r in active_agents if (not r.km_ok or not r.hours_ok)]

    total_km = float(sum(r.total_km for r in active_agents))
    total_hours = float(sum(r.total_hours for r in active_agents))
    total_pickup_km = float(sum(r.pickup_km for r in active_agents))
    total_unload_km = float(sum(r.unload_km for r in active_agents))
    total_return_km = float(sum(r.return_km for r in active_agents))
    total_mass = float(sum(r.mass_tons for r in active_agents))

    # Approximate transport-work by service legs only.
    task_by_id = {t.task_id: t for t in tasks}
    transport_work = 0.0
    for tid in task_to_agent:
        task = task_by_id[tid]
        d = _sp_dist(graph, cache, task.source_node_id, task.destination_node_id)
        if np.isfinite(d):
            transport_work += task.mass_tons * d

    feasible = (len(unassigned_ids) == 0) and (len(limit_violations) == 0)

    summary = {
        "algorithm": "exp3_greedy_fill",
        "dataset_tasks": n_tasks,
        "agents_used": len(agent_ids_sorted),
        "active_agents": len(active_agents),
        "assigned_tasks": assigned_count,
        "unassigned_tasks": len(unassigned_ids),
        "assignment_coverage_pct": round(100.0 * assigned_count / n_tasks, 3) if n_tasks else 0.0,
        "feasible": feasible,
        "limit_violating_agents": len(limit_violations),
        "transport_work_ton_km_approx": round(transport_work, 3),
        "total_mass_tons_assigned": round(total_mass, 3),
        "total_km": round(total_km, 3),
        "total_hours": round(total_hours, 3),
        "pickup_km": round(total_pickup_km, 3),
        "unload_km": round(total_unload_km, 3),
        "return_km": round(total_return_km, 3),
        "runtime_sec": round(time.perf_counter() - t0, 3),
        "params": {
            "max_agents": max_agents,
            "max_trips_per_agent": max_trips_per_agent,
            "max_pickups_per_trip": max_pickups_per_trip,
            "shortlist_k": shortlist_k,
        },
    }

    return {
        "summary": summary,
        "agent_rows": [r.as_dict() for r in per_agent_runs],
        "unassigned_task_ids": unassigned_ids,
        "task_assignment": task_to_agent,
        "task_trip": task_to_trip,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp3: greedy fill by nearest MNO tasks")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("src/data/real/full_29k/dataset_real_spb_full_29k.json"),
    )
    parser.add_argument("--max-agents", type=int, default=None)
    parser.add_argument("--max-trips-per-agent", type=int, default=8)
    parser.add_argument("--max-pickups-per-trip", type=int, default=16)
    parser.add_argument("--shortlist-k", type=int, default=40)
    parser.add_argument("--show-progress", action="store_true")
    args = parser.parse_args()

    dataset, payload = load_dataset(args.dataset_path)
    result = run_greedy_fill(
        dataset=dataset,
        payload=payload,
        max_agents=args.max_agents,
        max_trips_per_agent=max(1, int(args.max_trips_per_agent)),
        max_pickups_per_trip=max(1, int(args.max_pickups_per_trip)),
        shortlist_k=max(1, int(args.shortlist_k)),
        show_progress=args.show_progress,
    )

    summary = result["summary"]
    print("\n=== EXP3 summary ===")
    print(pd.DataFrame([summary]).to_string(index=False))

    df_agents = pd.DataFrame(result["agent_rows"])
    if not df_agents.empty:
        df_active = df_agents[df_agents["tasks_count"] > 0].copy()
        if not df_active.empty:
            cols = [
                "agent_id",
                "vehicle_type",
                "tasks_count",
                "trips_count",
                "mass_tons",
                "pickup_km",
                "unload_km",
                "return_km",
                "total_km",
                "total_hours",
                "km_ok",
                "hours_ok",
            ]
            print("\n=== EXP3 active agents (top 30 by tasks_count) ===")
            print(
                df_active.sort_values(["tasks_count", "total_km"], ascending=[False, True])[cols]
                .head(30)
                .to_string(index=False)
            )

    out_dir = Path("experiments/exp3/local")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_json = out_dir / f"exp3_greedy_fill_{stamp}.json"
    out_summary_csv = out_dir / f"exp3_summary_{stamp}.csv"
    out_agents_csv = out_dir / f"exp3_agents_{stamp}.csv"

    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(out_summary_csv, index=False)
    pd.DataFrame(result["agent_rows"]).to_csv(out_agents_csv, index=False)

    print(f"\nSaved JSON: {out_json.resolve()}")
    print(f"Saved CSV : {out_summary_csv.resolve()}")
    print(f"Saved CSV : {out_agents_csv.resolve()}")


if __name__ == "__main__":
    main()
