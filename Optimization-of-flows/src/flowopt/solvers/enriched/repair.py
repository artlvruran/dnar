from __future__ import annotations

from dataclasses import dataclass

from .common import SERVICE_HOURS_BY_CONTAINER, build_batched_route, pair_cost
from .distance_oracle import DistanceOracleWithFallback
from .problem import EnrichedProblem, task_agent_compatible
from .types import AssignmentRoute


@dataclass(frozen=True)
class RepairStats:
    repaired_tasks: int
    created_routes: int


def greedy_repair_unassigned(
    *,
    problem: EnrichedProblem,
    routes: list[AssignmentRoute],
    unassigned_task_ids: list[str],
    oracle: DistanceOracleWithFallback,
) -> tuple[list[AssignmentRoute], list[str], RepairStats]:
    if not unassigned_task_ids:
        return routes, [], RepairStats(repaired_tasks=0, created_routes=0)

    task_by_id = {t.task_id: t for t in problem.tasks}
    agent_by_id = {a.agent_id: a for a in problem.agents}
    open_tasks = [task_by_id[tid] for tid in unassigned_task_ids if tid in task_by_id]
    if not open_tasks:
        return routes, sorted(set(unassigned_task_ids)), RepairStats(repaired_tasks=0, created_routes=0)

    used_km = {a.agent_id: 0.0 for a in problem.agents}
    used_h = {a.agent_id: 0.0 for a in problem.agents}
    object_mass_used: dict[str, float] = {}
    object_vol_used: dict[str, float] = {}
    for r in routes:
        used_km[r.agent_id] = used_km.get(r.agent_id, 0.0) + float(r.total_distance_km)
        used_h[r.agent_id] = used_h.get(r.agent_id, 0.0) + float(r.total_hours)
        for tid in r.task_ids:
            t = task_by_id.get(tid)
            if t is None:
                continue
            object_mass_used[t.destination_node_id] = object_mass_used.get(t.destination_node_id, 0.0) + float(t.mass_tons)
            object_vol_used[t.destination_node_id] = object_vol_used.get(t.destination_node_id, 0.0) + float(t.volume_raw_m3)

    batches: dict[tuple[str, str, str], list] = {}
    for t in open_tasks:
        key = (t.source_node_id, t.destination_node_id, t.container_type)
        batches.setdefault(key, []).append(t)
    for group in batches.values():
        group.sort(key=lambda x: (x.mass_tons, x.volume_raw_m3), reverse=True)

    routes_out = list(routes)
    unresolved: list[str] = []
    repaired = 0
    route_counter = len(routes_out) + 1

    for key in sorted(batches.keys(), key=lambda k: len(batches[k]), reverse=True):
        group = batches[key]
        while group:
            rep = group[0]
            best: tuple[str, list, float, float, float] | None = None
            for agent in problem.agents:
                if not task_agent_compatible(rep, agent):
                    continue
                pc = pair_cost(rep, agent, oracle)
                if pc is None:
                    continue
                rem_km = agent.max_daily_km - used_km.get(agent.agent_id, 0.0)
                rem_h = agent.max_shift_hours - used_h.get(agent.agent_id, 0.0)
                if pc.total_km > rem_km + 1e-9:
                    continue
                service_h = SERVICE_HOURS_BY_CONTAINER.get(rep.container_type, 0.25)
                travel_h = pc.total_km / max(agent.avg_speed_kmph, 1e-6)
                max_by_h = int((rem_h - travel_h + 1e-9) // max(service_h, 1e-6))
                if max_by_h <= 0:
                    continue
                cur_mass = 0.0
                cur_vol = 0.0
                pick = []
                cap_m = max(agent.capacity_tons, 0.0)
                cap_v = agent.max_raw_volume_m3 if agent.max_raw_volume_m3 > 0 else float("inf")
                cap_obj_m = float(problem.object_day_capacity_tons.get(rep.destination_node_id, 0.0) or 0.0)
                cap_obj_v = float(problem.object_day_capacity_volume_m3.get(rep.destination_node_id, 0.0) or 0.0)
                obj_m_used = object_mass_used.get(rep.destination_node_id, 0.0)
                obj_v_used = object_vol_used.get(rep.destination_node_id, 0.0)

                for t in group:
                    if not task_agent_compatible(t, agent):
                        continue
                    if len(pick) >= max_by_h:
                        break
                    nm = cur_mass + float(t.mass_tons)
                    nv = cur_vol + float(t.volume_raw_m3)
                    if nm > cap_m + 1e-9 or nv > cap_v + 1e-9:
                        continue
                    if cap_obj_m > 0 and (obj_m_used + nm) > cap_obj_m + 1e-9:
                        continue
                    if cap_obj_v > 0 and (obj_v_used + nv) > cap_obj_v + 1e-9:
                        continue
                    pick.append(t)
                    cur_mass = nm
                    cur_vol = nv
                if not pick:
                    continue
                trip_h = travel_h + service_h * len(pick)
                if trip_h > rem_h + 1e-9:
                    continue
                score = pc.total_km / max(1, len(pick))
                cand = (agent.agent_id, pick, pc.total_km, pc.loaded_km, trip_h)
                if best is None or len(pick) > len(best[1]) or (len(pick) == len(best[1]) and score < (best[2] / max(1, len(best[1])))):
                    best = cand

            if best is None:
                unresolved.extend(t.task_id for t in group)
                break

            aid, picked, trip_km, loaded_km, trip_h = best
            agent = agent_by_id[aid]
            routes_out.append(
                build_batched_route(
                    route_id=f"EREPAIR_ROUTE_{route_counter:06d}",
                    agent=agent,
                    tasks=list(picked),
                    loaded_distance_km=trip_km if loaded_km <= 0 else loaded_km,
                    total_distance_km=trip_km,
                    total_hours=trip_h,
                )
            )
            route_counter += 1
            used_km[aid] = used_km.get(aid, 0.0) + float(trip_km)
            used_h[aid] = used_h.get(aid, 0.0) + float(trip_h)
            for t in picked:
                object_mass_used[t.destination_node_id] = object_mass_used.get(t.destination_node_id, 0.0) + float(t.mass_tons)
                object_vol_used[t.destination_node_id] = object_vol_used.get(t.destination_node_id, 0.0) + float(t.volume_raw_m3)
            picked_ids = {t.task_id for t in picked}
            repaired += len(picked_ids)
            group = [t for t in group if t.task_id not in picked_ids]

    unresolved_ids = sorted(set(unresolved))
    return routes_out, unresolved_ids, RepairStats(repaired_tasks=repaired, created_routes=max(0, len(routes_out) - len(routes)))

