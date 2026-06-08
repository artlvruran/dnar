from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .. import core
from .. import genetic_solver_components as ga


@dataclass(frozen=True)
class ConstraintBundle:
    dataset_profile: str
    max_daily_km_by_type: dict[str, float]
    max_shift_hours_by_type: dict[str, float]
    avg_speed_kmph_by_type: dict[str, float]
    service_hours_by_container: dict[str, float]
    uses_edge_vehicle_type_limits: bool
    uses_agent_payload_limits: bool
    uses_agent_volume_limits: bool
    uses_object_mass_limits: bool
    uses_object_volume_limits: bool
    uses_task_vehicle_compatibility: bool

    def apply(self) -> None:
        core.MAX_DAILY_KM_BY_TYPE.update(self.max_daily_km_by_type)
        core.MAX_SHIFT_HOURS_BY_TYPE.update(self.max_shift_hours_by_type)
        core.AVG_SPEED_KMPH_BY_TYPE.update(self.avg_speed_kmph_by_type)
        core.SERVICE_HOURS_BY_CONTAINER.update(self.service_hours_by_container)

        ga.MAX_DAILY_KM_BY_TYPE.update(self.max_daily_km_by_type)
        ga.MAX_SHIFT_HOURS_BY_TYPE.update(self.max_shift_hours_by_type)
        ga.AVG_SPEED_KMPH_BY_TYPE.update(self.avg_speed_kmph_by_type)
        ga.SERVICE_HOURS_BY_CONTAINER.update(self.service_hours_by_container)

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_profile": self.dataset_profile,
            "max_daily_km_by_type": dict(self.max_daily_km_by_type),
            "max_shift_hours_by_type": dict(self.max_shift_hours_by_type),
            "avg_speed_kmph_by_type": dict(self.avg_speed_kmph_by_type),
            "service_hours_by_container": dict(self.service_hours_by_container),
            "uses_edge_vehicle_type_limits": self.uses_edge_vehicle_type_limits,
            "uses_agent_payload_limits": self.uses_agent_payload_limits,
            "uses_agent_volume_limits": self.uses_agent_volume_limits,
            "uses_object_mass_limits": self.uses_object_mass_limits,
            "uses_object_volume_limits": self.uses_object_volume_limits,
            "uses_task_vehicle_compatibility": self.uses_task_vehicle_compatibility,
        }


def build_constraint_bundle(
    *,
    payload: dict[str, Any],
    dataset_profile: str,
) -> ConstraintBundle:
    meta = payload.get("metadata") or {}
    vehicle_profiles = meta.get("vehicle_profiles") or {}

    max_daily_km_by_type = dict(core.MAX_DAILY_KM_BY_TYPE)
    max_shift_hours_by_type = dict(core.MAX_SHIFT_HOURS_BY_TYPE)
    avg_speed_kmph_by_type = dict(core.AVG_SPEED_KMPH_BY_TYPE)
    service_hours_by_container = dict(core.SERVICE_HOURS_BY_CONTAINER)

    for vt, profile in vehicle_profiles.items():
        max_daily_km_by_type[str(vt)] = float(profile.get("max_daily_km", max_daily_km_by_type.get(vt, 130.0)))
        max_shift_hours_by_type[str(vt)] = float(
            profile.get("max_shift_hours", max_shift_hours_by_type.get(vt, 10.0))
        )
        avg_speed_kmph_by_type[str(vt)] = float(profile.get("avg_speed_kmph", avg_speed_kmph_by_type.get(vt, 24.0)))

    for container_type, service_hours in (meta.get("service_hours_by_container") or {}).items():
        service_hours_by_container[str(container_type)] = float(service_hours)

    edges = (payload.get("graph") or {}).get("edges") or []
    agents = payload.get("agents") or []
    tasks = payload.get("tasks") or []
    nodes = (payload.get("graph") or {}).get("nodes") or []

    uses_edge_vehicle_type_limits = any(bool(edge.get("allowed_vehicle_types")) for edge in edges)
    uses_agent_payload_limits = any(float(agent.get("capacity_tons", 0.0) or 0.0) > 0 for agent in agents)
    uses_agent_volume_limits = any(
        (float(agent.get("body_volume_m3", 0.0) or 0.0) > 0)
        or (float(agent.get("max_raw_volume_m3", 0.0) or 0.0) > 0)
        for agent in agents
    ) and any(float(task.get("volume_raw_m3", 0.0) or 0.0) > 0 for task in tasks)

    uses_object_mass_limits = any(float(node.get("object_day_capacity_tons", 0.0) or 0.0) > 0 for node in nodes)
    uses_object_volume_limits = any(
        float(node.get("object_day_capacity_volume_m3", 0.0) or 0.0) > 0 for node in nodes
    )
    uses_task_vehicle_compatibility = any(bool(task.get("compatible_vehicle_types")) for task in tasks)

    return ConstraintBundle(
        dataset_profile=dataset_profile,
        max_daily_km_by_type=max_daily_km_by_type,
        max_shift_hours_by_type=max_shift_hours_by_type,
        avg_speed_kmph_by_type=avg_speed_kmph_by_type,
        service_hours_by_container=service_hours_by_container,
        uses_edge_vehicle_type_limits=uses_edge_vehicle_type_limits,
        uses_agent_payload_limits=uses_agent_payload_limits,
        uses_agent_volume_limits=uses_agent_volume_limits,
        uses_object_mass_limits=uses_object_mass_limits,
        uses_object_volume_limits=uses_object_volume_limits,
        uses_task_vehicle_compatibility=uses_task_vehicle_compatibility,
    )
