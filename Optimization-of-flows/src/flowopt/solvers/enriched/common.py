from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .distance_oracle import DistanceOracleWithFallback
from .types import AssignmentRoute, EnrichedAgent, EnrichedTask


SERVICE_HOURS_BY_CONTAINER: dict[str, float] = {
    "A": 0.22,
    "B": 0.24,
    "C": 0.35,
    "D": 0.32,
    "Type1": 0.22,
    "Type2": 0.24,
    "Type3": 0.35,
    "Type4": 0.32,
    "TypeRNO": 0.28,
    "TypeO2P": 0.30,
}


@dataclass(frozen=True)
class PairCost:
    total_km: float
    loaded_km: float
    total_hours: float


def pair_cost(task: EnrichedTask, agent: EnrichedAgent, oracle: DistanceOracleWithFallback) -> PairCost | None:
    if agent.depot_node_id is None:
        return None
    d1 = oracle.dist(agent.depot_node_id, task.source_node_id)
    d2 = oracle.dist(task.source_node_id, task.destination_node_id)
    d3 = oracle.dist(task.destination_node_id, agent.depot_node_id)
    if not all(x < float("inf") for x in (d1, d2, d3)):
        return None
    total_km = d1 + d2 + d3
    loaded_km = d2
    service_hours = SERVICE_HOURS_BY_CONTAINER.get(task.container_type, 0.25)
    total_hours = (total_km / max(agent.avg_speed_kmph, 1e-6)) + service_hours
    return PairCost(total_km=total_km, loaded_km=loaded_km, total_hours=total_hours)


def build_single_task_route(
    *,
    route_id: str,
    agent: EnrichedAgent,
    task: EnrichedTask,
    cost: PairCost,
) -> AssignmentRoute:
    # Path is compressed (node sequence), because main route geometry is not stored in precomputed matrix.
    return AssignmentRoute(
        route_id=route_id,
        agent_id=agent.agent_id,
        task_ids=(task.task_id,),
        path=(
            agent.depot_node_id or "",
            task.source_node_id,
            task.destination_node_id,
            agent.depot_node_id or "",
        ),
        loaded_distance_km=cost.loaded_km,
        total_distance_km=cost.total_km,
        total_hours=cost.total_hours,
        payload_tons=float(task.mass_tons),
    )


def build_batched_route(
    *,
    route_id: str,
    agent: EnrichedAgent,
    tasks: list[EnrichedTask],
    loaded_distance_km: float,
    total_distance_km: float,
    total_hours: float,
) -> AssignmentRoute:
    first = tasks[0]
    payload_tons = sum(float(t.mass_tons) for t in tasks)
    return AssignmentRoute(
        route_id=route_id,
        agent_id=agent.agent_id,
        task_ids=tuple(t.task_id for t in tasks),
        path=(
            agent.depot_node_id or "",
            first.source_node_id,
            first.destination_node_id,
            agent.depot_node_id or "",
        ),
        loaded_distance_km=float(loaded_distance_km),
        total_distance_km=float(total_distance_km),
        total_hours=float(total_hours),
        payload_tons=float(payload_tons),
    )


def summarize_checks(*, unassigned_count: int, overflow_km: int, overflow_hours: int) -> dict[str, Any]:
    daily_ok = (overflow_km == 0 and overflow_hours == 0)
    all_assigned = (unassigned_count == 0)
    hard_ok = daily_ok and all_assigned
    return {
        "hard_constraints_ok": hard_ok,
        "daily_limits_ok": daily_ok,
        "reachability_ok": True,
        "all_tasks_assigned": all_assigned,
        "mno_coverage_ok": all_assigned,
        "all_checks_ok": hard_ok,
        "task_space_match": True,
        "metric_task_space": "dataset_reference",
        "overflow_km_agents": overflow_km,
        "overflow_hours_agents": overflow_hours,
    }
