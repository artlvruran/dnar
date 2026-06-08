from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VolumeTask:
    task_id: str
    source_node_id: str
    destination_node_id: str
    source_zone_num: int | None
    volume_raw_m3: float
    is_compactable: bool
    requires_compact_d: bool
    required_container_types: tuple[str, ...]
    container_type: str


@dataclass(frozen=True)
class VolumeAgent:
    agent_id: str
    depot_node_id: str | None
    zone_num: int | None
    is_active: bool
    cap_a: bool
    cap_b: bool
    cap_c: bool
    cap_d: bool
    compaction_coeff: float
    max_raw_volume_m3: float
    max_hours: float
    max_daily_km: float
    avg_speed_kmph: float


@dataclass(frozen=True)
class GraphNodeLite:
    node_id: str
    kind: str
    x: float
    y: float
    object_day_capacity_volume_m3: float = 0.0


@dataclass(frozen=True)
class TaskPickupLog:
    task_id: str
    source_node_id: str
    effective_volume_m3: float
    carried_distance_to_object_km: float


@dataclass(frozen=True)
class TripPlan:
    trip_id: str
    agent_id: str
    depot_node_id: str
    destination_object_id: str
    ordered_task_ids: tuple[str, ...]
    visit_nodes: tuple[str, ...]  # depot, s1..sn, object, depot
    leg_distances_km: tuple[float, ...]
    total_km: float
    total_hours: float
    payload_effective_volume_m3: float
    task_pickups: tuple[TaskPickupLog, ...]
    path_nodes_full: tuple[str, ...] = ()


@dataclass(frozen=True)
class AssignmentSolution:
    algorithm: str
    dataset_path: str
    trips: tuple[TripPlan, ...]
    unassigned_task_ids: tuple[str, ...]
    runtime_sec: float
    solver_logs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConstraintReport:
    all_checks_ok: bool
    feasible: bool
    checks: dict[str, Any]
    bottlenecks: dict[str, Any]
    summary: dict[str, Any]


@dataclass(frozen=True)
class EvaluationResult:
    algorithm: str
    dataset_path: str
    total_tasks: int
    assigned_tasks: int
    unassigned_tasks: int
    task_coverage_pct: float
    active_agents: int
    trips_count: int
    total_km: float
    total_hours: float
    transport_work_volume_m3_km: float
    object_volume_used_m3: dict[str, float] = field(default_factory=dict)
    object_volume_capacity_m3: dict[str, float] = field(default_factory=dict)
    constraints: ConstraintReport | None = None
    runtime_sec: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "dataset_path": self.dataset_path,
            "total_tasks": self.total_tasks,
            "assigned_tasks": self.assigned_tasks,
            "unassigned_tasks": self.unassigned_tasks,
            "task_coverage_pct": self.task_coverage_pct,
            "active_agents": self.active_agents,
            "trips_count": self.trips_count,
            "total_km": self.total_km,
            "total_hours": self.total_hours,
            "transport_work_volume_m3_km": self.transport_work_volume_m3_km,
            "runtime_sec": self.runtime_sec,
            "all_checks_ok": bool(self.constraints.all_checks_ok) if self.constraints else False,
            "feasible": bool(self.constraints.feasible) if self.constraints else False,
            "checks": self.constraints.checks if self.constraints else {},
            "bottlenecks": self.constraints.bottlenecks if self.constraints else {},
            "object_volume_used_m3": self.object_volume_used_m3,
            "object_volume_capacity_m3": self.object_volume_capacity_m3,
        }
