from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NormalizedPayload:
    payload: dict[str, Any]
    dataset_profile: str
    notes: tuple[str, ...]


def _coalesce(raw: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        if key in raw and raw[key] is not None:
            return raw[key]
    return default


def detect_dataset_profile(payload: dict[str, Any]) -> str:
    meta = payload.get("metadata") or {}
    profile = str(meta.get("profile", "")).strip().lower()
    if profile:
        if "clean_full_enriched" in profile:
            return "clean_full_enriched"
        if "fast_uniform" in profile or "u50" in profile:
            return "u50_fast_uniform"
        return profile

    if payload.get("objects") or payload.get("depots"):
        return "clean_full_enriched"
    if (meta.get("day_load_profile") or meta.get("fast_profile")):
        return "u50_fast_uniform"
    return "generic_real"


class BaseDatasetAdapter(ABC):
    profile_name: str = "generic_real"

    @abstractmethod
    def normalize(self, payload: dict[str, Any]) -> NormalizedPayload:
        raise NotImplementedError


class GenericRealDatasetAdapter(BaseDatasetAdapter):
    profile_name = "generic_real"

    def normalize(self, payload: dict[str, Any]) -> NormalizedPayload:
        return _normalize_payload_common(payload, dataset_profile=self.profile_name)


class FastU50DatasetAdapter(BaseDatasetAdapter):
    profile_name = "u50_fast_uniform"

    def normalize(self, payload: dict[str, Any]) -> NormalizedPayload:
        return _normalize_payload_common(payload, dataset_profile=self.profile_name)


class CleanFullEnrichedDatasetAdapter(BaseDatasetAdapter):
    profile_name = "clean_full_enriched"

    def normalize(self, payload: dict[str, Any]) -> NormalizedPayload:
        return _normalize_payload_common(payload, dataset_profile=self.profile_name)


def _normalize_nodes(nodes_raw: list[dict[str, Any]], notes: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for node in nodes_raw:
        norm = dict(node)
        node_id = _coalesce(node, ("node_id", "id", "node"))
        kind = _coalesce(node, ("kind", "node_kind", "type"), "road")
        if node_id is None:
            notes.append("node without id skipped")
            continue
        norm["node_id"] = str(node_id)
        norm["kind"] = str(kind)
        norm["x"] = float(_coalesce(node, ("x", "lon", "lng"), 0.0))
        norm["y"] = float(_coalesce(node, ("y", "lat"), 0.0))
        norm.setdefault("center", False)
        norm.setdefault("container_types", [])
        norm.setdefault("daily_mass_tons", 0.0)
        norm.setdefault("object_day_capacity_tons", 0.0)
        norm.setdefault("object_year_capacity_tons", 0.0)
        norm.setdefault("object_day_capacity_volume_m3", 0.0)
        out.append(norm)
    return out


def _normalize_edges(edges_raw: list[dict[str, Any]], notes: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for edge in edges_raw:
        norm = dict(edge)
        source_id = _coalesce(edge, ("source_id", "from", "source"))
        target_id = _coalesce(edge, ("target_id", "to", "target"))
        if source_id is None or target_id is None:
            notes.append("edge without endpoints skipped")
            continue
        norm["source_id"] = str(source_id)
        norm["target_id"] = str(target_id)
        norm["distance_km"] = float(_coalesce(edge, ("distance_km", "distance", "length_km"), 0.0))
        allowed = _coalesce(edge, ("allowed_vehicle_types", "vehicle_types"), None)
        if not allowed:
            # Keep permissive fallback to avoid loader failure on legacy exports.
            allowed = [
                "VT_A",
                "VT_AB",
                "VT_ABD",
                "VT_AD",
                "VT_C",
                "VT_CD",
                "Type1",
                "Type2",
                "Type1-2",
                "Type3",
                "Type4",
                "TypeRNO",
                "TypeO2P",
            ]
        norm["allowed_vehicle_types"] = list(allowed)
        out.append(norm)
    return out


def _normalize_agents(agents_raw: list[dict[str, Any]], notes: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for agent in agents_raw:
        norm = dict(agent)
        agent_id = _coalesce(agent, ("agent_id", "vehicle_id", "id"))
        vehicle_type = _coalesce(agent, ("vehicle_type", "type"), "VT_A")
        capacity_tons = _coalesce(agent, ("capacity_tons",), None)
        if capacity_tons is None:
            capacity_kg = _coalesce(agent, ("capacity_kg",), 0.0)
            capacity_tons = float(capacity_kg) / 1000.0
            notes.append("agent capacity_kg converted to capacity_tons")

        if agent_id is None:
            notes.append("agent without id skipped")
            continue

        norm["agent_id"] = str(agent_id)
        norm["vehicle_type"] = str(vehicle_type)
        norm["capacity_tons"] = float(capacity_tons)
        norm["is_compact"] = bool(_coalesce(agent, ("is_compact", "compact"), False))
        norm.setdefault("body_volume_m3", 0.0)
        norm.setdefault("compaction_coeff", 1.0)
        norm.setdefault("max_raw_volume_m3", 0.0)
        out.append(norm)
    return out


def _normalize_tasks(tasks_raw: list[dict[str, Any]], notes: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for task in tasks_raw:
        norm = dict(task)
        task_id = _coalesce(task, ("task_id", "id", "job_id"))
        source_node_id = _coalesce(task, ("source_node_id", "source_id", "from_node_id", "mno_node_id"))
        destination_node_id = _coalesce(
            task,
            ("destination_node_id", "target_node_id", "to_node_id", "object_node_id"),
        )
        mass_tons = _coalesce(task, ("mass_tons",), None)
        if mass_tons is None:
            mass_kg = _coalesce(task, ("mass_kg", "mass"), 0.0)
            mass_tons = float(mass_kg) / 1000.0
            notes.append("task mass_kg converted to mass_tons")

        if task_id is None or source_node_id is None or destination_node_id is None:
            notes.append("task without required ids skipped")
            continue

        norm["task_id"] = str(task_id)
        norm["source_node_id"] = str(source_node_id)
        norm["destination_node_id"] = str(destination_node_id)
        norm["container_type"] = str(_coalesce(task, ("container_type", "fraction", "type"), "A"))
        norm["mass_tons"] = float(mass_tons)
        norm["periodicity"] = str(_coalesce(task, ("periodicity", "service_periodicity"), "P1"))

        norm.setdefault("mass_kg", float(norm["mass_tons"]) * 1000.0)
        norm.setdefault("volume_raw_m3", 0.0)
        norm.setdefault("density_kg_m3_assumed", 0.0)
        norm.setdefault("volume_body_m3_by_vehicle_type", {})
        norm.setdefault("compatible_vehicle_types", [])
        norm.setdefault("source_center", False)
        norm.setdefault("source_container_types", [])
        norm.setdefault("source_node_daily_mass_tons", 0.0)
        norm.setdefault("destination_object_alias", None)
        norm.setdefault("destination_object_day_capacity_tons", 0.0)
        norm.setdefault("destination_object_day_capacity_volume_m3", 0.0)
        norm.setdefault("periodicity_days", 1)
        norm.setdefault("periodicity_weekly_frequency", 1.0)
        out.append(norm)
    return out


def _normalize_payload_common(
    payload: dict[str, Any],
    *,
    dataset_profile: str,
) -> NormalizedPayload:
    notes: list[str] = []
    out = dict(payload)

    meta = dict(payload.get("metadata") or {})
    out["metadata"] = meta

    graph_raw = payload.get("graph") or {}
    nodes_raw = list(_coalesce(graph_raw, ("nodes", "road_nodes"), []))
    edges_raw = list(_coalesce(graph_raw, ("edges", "road_edges"), []))
    out["graph"] = {
        "nodes": _normalize_nodes(nodes_raw, notes),
        "edges": _normalize_edges(edges_raw, notes),
    }

    out["agents"] = _normalize_agents(list(_coalesce(payload, ("agents", "fleet", "vehicles"), [])), notes)
    out["tasks"] = _normalize_tasks(list(_coalesce(payload, ("tasks", "jobs", "orders"), [])), notes)

    routes_raw = _coalesce(payload, ("routes", "solution_routes"), [])
    out["routes"] = list(routes_raw) if routes_raw is not None else []
    out.setdefault("derived", dict(payload.get("derived") or {}))

    # Keep extra top-level keys (depots/objects) untouched for downstream tools.
    for extra in ("objects", "depots"):
        if extra in payload:
            out[extra] = payload[extra]

    return NormalizedPayload(payload=out, dataset_profile=dataset_profile, notes=tuple(notes))


def adapter_for_payload(payload: dict[str, Any]) -> BaseDatasetAdapter:
    profile = detect_dataset_profile(payload)
    if profile == "clean_full_enriched":
        return CleanFullEnrichedDatasetAdapter()
    if profile == "u50_fast_uniform":
        return FastU50DatasetAdapter()
    return GenericRealDatasetAdapter()


def normalize_payload(payload: dict[str, Any]) -> NormalizedPayload:
    adapter = adapter_for_payload(payload)
    return adapter.normalize(payload)
