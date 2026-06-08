from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .distance_oracle import DistanceOracleWithFallback
from .problem import EnrichedProblem, task_agent_compatible
from .types import AssignmentRoute


@dataclass(frozen=True)
class EnrichedConstraintReport:
    checks: dict[str, Any]
    normalized_unassigned: list[str]
    transport_work_ton_km: float


def evaluate_constraints(
    *,
    problem: EnrichedProblem,
    routes: list[AssignmentRoute],
    unassigned_task_ids: list[str],
    oracle: DistanceOracleWithFallback,
) -> EnrichedConstraintReport:
    task_by_id = {t.task_id: t for t in problem.tasks}
    agent_by_id = {a.agent_id: a for a in problem.agents}
    all_task_ids = set(task_by_id)

    unknown_task_refs = 0
    unknown_agent_refs = 0
    incompatible_assignments = 0
    unreachable_assignments = 0
    duplicate_tasks = 0
    per_agent_km: dict[str, float] = {aid: 0.0 for aid in agent_by_id}
    per_agent_h: dict[str, float] = {aid: 0.0 for aid in agent_by_id}
    object_mass: dict[str, float] = {}
    object_vol: dict[str, float] = {}

    seen: set[str] = set()
    for route in routes:
        agent = agent_by_id.get(route.agent_id)
        if agent is None:
            unknown_agent_refs += 1
            continue
        per_agent_km[agent.agent_id] = per_agent_km.get(agent.agent_id, 0.0) + float(route.total_distance_km)
        per_agent_h[agent.agent_id] = per_agent_h.get(agent.agent_id, 0.0) + float(route.total_hours)

        for tid in route.task_ids:
            task = task_by_id.get(tid)
            if task is None:
                unknown_task_refs += 1
                continue
            if tid in seen:
                duplicate_tasks += 1
                continue
            seen.add(tid)

            if not task_agent_compatible(task, agent):
                incompatible_assignments += 1

            if agent.depot_node_id is None:
                unreachable_assignments += 1
            else:
                d1 = oracle.dist(agent.depot_node_id, task.source_node_id)
                d2 = oracle.dist(task.source_node_id, task.destination_node_id)
                d3 = oracle.dist(task.destination_node_id, agent.depot_node_id)
                if not (d1 < float("inf") and d2 < float("inf") and d3 < float("inf")):
                    unreachable_assignments += 1

            object_mass[task.destination_node_id] = object_mass.get(task.destination_node_id, 0.0) + float(task.mass_tons)
            object_vol[task.destination_node_id] = object_vol.get(task.destination_node_id, 0.0) + float(task.volume_raw_m3)

    missing_tasks = sorted(all_task_ids - seen)
    unknown_unassigned = sorted(set(unassigned_task_ids) - all_task_ids)
    normalized_unassigned = sorted(set(missing_tasks) | set(unassigned_task_ids)) if unknown_unassigned else sorted(
        set(missing_tasks) | set(unassigned_task_ids)
    )

    overflow_km_agents = 0
    overflow_hours_agents = 0
    unavailable_used_agents = 0
    for agent in problem.agents:
        km = per_agent_km.get(agent.agent_id, 0.0)
        h = per_agent_h.get(agent.agent_id, 0.0)
        if km > agent.max_daily_km + 1e-9:
            overflow_km_agents += 1
        if h > agent.max_shift_hours + 1e-9:
            overflow_hours_agents += 1
        if (km > 0 or h > 0) and not agent.is_available:
            unavailable_used_agents += 1

    object_mass_violations = 0
    object_vol_violations = 0
    for oid, used in object_mass.items():
        cap = float(problem.object_day_capacity_tons.get(oid, 0.0) or 0.0)
        if cap > 0 and used > cap + 1e-9:
            object_mass_violations += 1
    for oid, used in object_vol.items():
        cap = float(problem.object_day_capacity_volume_m3.get(oid, 0.0) or 0.0)
        if cap > 0 and used > cap + 1e-9:
            object_vol_violations += 1

    total_tr = 0.0
    for route in routes:
        for tid in route.task_ids:
            task = task_by_id.get(tid)
            if task is None:
                continue
            total_tr += float(task.mass_tons) * float(route.loaded_distance_km)

    all_assigned_once = (
        duplicate_tasks == 0
        and unknown_task_refs == 0
        and len(normalized_unassigned) == 0
        and len(seen) == len(all_task_ids)
    )
    daily_limits_ok = (overflow_km_agents == 0 and overflow_hours_agents == 0)
    object_limits_ok = (object_mass_violations == 0 and object_vol_violations == 0)
    compatibility_ok = (incompatible_assignments == 0 and unavailable_used_agents == 0)
    reachability_ok = (unreachable_assignments == 0 and unknown_agent_refs == 0)

    checks = {
        "hard_constraints_ok": bool(
            all_assigned_once and daily_limits_ok and object_limits_ok and compatibility_ok and reachability_ok
        ),
        "daily_limits_ok": bool(daily_limits_ok),
        "object_limits_ok": bool(object_limits_ok),
        "reachability_ok": bool(reachability_ok),
        "compatibility_ok": bool(compatibility_ok),
        "all_tasks_assigned_once": bool(all_assigned_once),
        "all_tasks_assigned": len(normalized_unassigned) == 0,
        "mno_coverage_ok": len(normalized_unassigned) == 0,
        "all_checks_ok": bool(
            all_assigned_once and daily_limits_ok and object_limits_ok and compatibility_ok and reachability_ok
        ),
        "task_space_match": True,
        "metric_task_space": "dataset_reference",
        "overflow_km_agents": int(overflow_km_agents),
        "overflow_hours_agents": int(overflow_hours_agents),
        "object_mass_violations": int(object_mass_violations),
        "object_volume_violations": int(object_vol_violations),
        "incompatible_assignments": int(incompatible_assignments),
        "unavailable_used_agents": int(unavailable_used_agents),
        "unreachable_assignments": int(unreachable_assignments),
        "duplicate_tasks": int(duplicate_tasks),
        "unknown_task_refs": int(unknown_task_refs),
        "unknown_agent_refs": int(unknown_agent_refs),
        "unknown_unassigned_refs": int(len(unknown_unassigned)),
    }

    return EnrichedConstraintReport(
        checks=checks,
        normalized_unassigned=normalized_unassigned,
        transport_work_ton_km=float(total_tr),
    )

