from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EnrichedAgent:
    agent_id: str
    vehicle_type: str
    capacity_tons: float
    max_raw_volume_m3: float
    is_compact: bool
    is_available: bool
    depot_node_id: str | None
    cap_container_types: frozenset[str]
    max_daily_km: float
    max_shift_hours: float
    avg_speed_kmph: float
    zone_num: int | None


@dataclass(frozen=True)
class EnrichedTask:
    task_id: str
    source_node_id: str
    destination_node_id: str
    container_type: str
    mass_tons: float
    volume_raw_m3: float
    compatible_vehicle_types: frozenset[str]
    source_center: bool
    source_zone_num: int | None


@dataclass
class AssignmentRoute:
    route_id: str
    agent_id: str
    task_ids: tuple[str, ...]
    path: tuple[str, ...]
    loaded_distance_km: float
    total_distance_km: float
    total_hours: float
    payload_tons: float


@dataclass
class AgentUsage:
    agent_id: str
    tasks: list[str] = field(default_factory=list)
    total_km: float = 0.0
    total_hours: float = 0.0
    loaded_km: float = 0.0


@dataclass
class EnrichedSolveResult:
    algorithm: str
    feasible: bool
    routes: list[AssignmentRoute]
    unassigned_task_ids: list[str]
    agent_usage: dict[str, AgentUsage]
    runtime_sec: float
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        active_agents = sum(1 for usage in self.agent_usage.values() if usage.tasks)
        loaded_km = sum(route.loaded_distance_km for route in self.routes)
        total_km = sum(route.total_distance_km for route in self.routes)
        total_hours = sum(route.total_hours for route in self.routes)
        assigned_tasks = sum(len(route.task_ids) for route in self.routes)
        trip_count = len(self.routes)
        transport_work = self.details.get("transport_work_ton_km")
        if transport_work is None:
            transport_work = sum(float(route.payload_tons) * float(route.loaded_distance_km) for route in self.routes)

        out = {
            "algorithm": self.algorithm,
            "feasible": self.feasible,
            "assigned_routes": assigned_tasks,
            "assigned_trips": trip_count,
            "unassigned_tasks": len(self.unassigned_task_ids),
            "active_agents": active_agents,
            "transport_work_ton_km": round(float(transport_work), 3),
            "total_km": round(float(total_km), 3),
            "deadhead_km": round(float(total_km - loaded_km), 3),
            "deadhead_share_pct": round(float(100.0 * (total_km - loaded_km) / total_km), 3) if total_km > 0 else None,
            "total_hours": round(float(total_hours), 3),
            "runtime_sec": round(float(self.runtime_sec), 3),
            "solver_error": self.details.get("solver_error"),
            "details": self.details,
        }
        return out
