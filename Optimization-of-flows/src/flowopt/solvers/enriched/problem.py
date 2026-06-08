from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ... import core
from ...backend.io import load_dataset
from ...dataset import CONTAINER_TO_VEHICLE_TYPES
from .types import EnrichedAgent, EnrichedTask


@dataclass(frozen=True)
class EnrichedProblem:
    dataset_path: Path
    payload: dict[str, Any]
    agents: list[EnrichedAgent]
    tasks: list[EnrichedTask]
    task_mass_by_id: dict[str, float]
    object_day_capacity_tons: dict[str, float]
    object_day_capacity_volume_m3: dict[str, float]


def _agent_container_caps(agent_raw: dict[str, Any]) -> frozenset[str]:
    explicit = agent_raw.get("cap_container_types")
    if explicit:
        return frozenset(str(x) for x in explicit)
    caps: list[str] = []
    for c in ("A", "B", "C", "D"):
        if bool(agent_raw.get(f"cap_container_{c}", False)):
            caps.append(c)
    return frozenset(caps)


def _max_raw_volume(agent_raw: dict[str, Any]) -> float:
    raw = float(agent_raw.get("max_raw_volume_m3", 0.0) or 0.0)
    if raw > 0:
        return raw
    body = float(agent_raw.get("body_volume_m3", 0.0) or 0.0)
    comp = float(agent_raw.get("compaction_coeff", 1.0) or 1.0)
    if body > 0 and comp > 0:
        return body * comp
    return 0.0


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _agent_is_available(raw: dict[str, Any]) -> bool:
    direct = raw.get("is_active_work_1st_shoulder")
    if direct is not None:
        return bool(direct)
    status = str(raw.get("status", "")).strip().lower()
    shoulder = str(raw.get("shoulder", "")).strip().lower()
    if status and shoulder:
        return ("в работе" in status) and shoulder.startswith("1")
    return True


def build_enriched_problem(dataset_path: Path | str) -> EnrichedProblem:
    dataset_path = Path(dataset_path)
    dataset, payload = load_dataset(dataset_path)

    profiles = (payload.get("metadata") or {}).get("vehicle_profiles") or {}
    agent_depots = (payload.get("metadata") or {}).get("agent_depots") or {}
    raw_agents_by_id = {
        str(a.get("agent_id")): a
        for a in payload.get("agents", [])
        if a.get("agent_id") is not None
    }
    raw_tasks_by_id = {
        str(t.get("task_id")): t
        for t in payload.get("tasks", [])
        if t.get("task_id") is not None
    }
    node_caps_mass: dict[str, float] = {}
    node_caps_vol: dict[str, float] = {}
    for node in (payload.get("graph") or {}).get("nodes", []) or []:
        if not str(node.get("kind", "")).startswith("object"):
            continue
        nid = str(node.get("node_id"))
        node_caps_mass[nid] = float(node.get("object_day_capacity_tons", 0.0) or 0.0)
        node_caps_vol[nid] = float(node.get("object_day_capacity_volume_m3", 0.0) or 0.0)

    agents: list[EnrichedAgent] = []
    for aid, agent in dataset.fleet.agents.items():
        p = profiles.get(agent.vehicle_type, {})
        max_daily_km = float(p.get("max_daily_km", core.MAX_DAILY_KM_BY_TYPE.get(agent.vehicle_type, 130.0)))
        max_shift_h = float(p.get("max_shift_hours", core.MAX_SHIFT_HOURS_BY_TYPE.get(agent.vehicle_type, 10.0)))
        speed = float(p.get("avg_speed_kmph", core.AVG_SPEED_KMPH_BY_TYPE.get(agent.vehicle_type, 24.0)))

        raw = raw_agents_by_id.get(aid)
        if raw is None:
            raw = {
                "cap_container_types": list(agent.cap_container_types),
                "max_raw_volume_m3": agent.raw_volume_limit_m3,
                "is_compact": agent.is_compact,
                "zone_num": None,
                "depot_node_id": None,
            }
        depot_from_meta = agent_depots.get(aid)
        depot_from_raw = raw.get("depot_node_id")
        depot_node_id = depot_from_meta if depot_from_meta is not None else depot_from_raw
        is_compact = raw.get("is_compact")
        if is_compact is None:
            # For enriched datasets compact requirement is often encoded via D-capability only.
            is_compact = bool(raw.get("cap_container_D", False))

        agents.append(
            EnrichedAgent(
                agent_id=aid,
                vehicle_type=agent.vehicle_type,
                capacity_tons=float(agent.capacity_tons),
                max_raw_volume_m3=float(_max_raw_volume(raw)),
                is_compact=bool(is_compact),
                is_available=_agent_is_available(raw),
                depot_node_id=str(depot_node_id) if depot_node_id is not None else None,
                cap_container_types=_agent_container_caps(raw),
                max_daily_km=max_daily_km,
                max_shift_hours=max_shift_h,
                avg_speed_kmph=max(speed, 1e-6),
                zone_num=_safe_int(raw.get("zone_num")),
            )
        )

    tasks: list[EnrichedTask] = []
    for task in dataset.tasks:
        raw_task = raw_tasks_by_id.get(task.task_id, {})
        source_center = bool(task.source_center or raw_task.get("requires_compact_d", False))
        source_zone = raw_task.get("source_zone_num")
        comp_types = task.compatible_vehicle_types or tuple(raw_task.get("compatible_vehicle_types") or ())
        tasks.append(
            EnrichedTask(
                task_id=task.task_id,
                source_node_id=task.source_node_id,
                destination_node_id=task.destination_node_id,
                container_type=task.container_type,
                mass_tons=float(task.mass_tons),
                volume_raw_m3=float(task.volume_raw_m3),
                compatible_vehicle_types=frozenset(str(x) for x in comp_types),
                source_center=source_center,
                source_zone_num=_safe_int(source_zone),
            )
        )

    return EnrichedProblem(
        dataset_path=dataset_path,
        payload=payload,
        agents=agents,
        tasks=tasks,
        task_mass_by_id={t.task_id: float(t.mass_tons) for t in tasks},
        object_day_capacity_tons=node_caps_mass,
        object_day_capacity_volume_m3=node_caps_vol,
    )


def task_agent_compatible(task: EnrichedTask, agent: EnrichedAgent) -> bool:
    if not agent.is_available:
        return False
    if task.source_zone_num is not None and agent.zone_num is not None and task.source_zone_num != agent.zone_num:
        return False
    allowed_types = CONTAINER_TO_VEHICLE_TYPES.get(task.container_type, set())
    if agent.vehicle_type not in allowed_types:
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
