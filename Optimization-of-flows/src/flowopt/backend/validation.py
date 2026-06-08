from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import core
from ..dataset import CONTAINER_TO_VEHICLE_TYPES, RoutingDataset
from .io import load_dataset


@dataclass
class InputDatasetReport:
    dataset_path: str
    basic_checks_ok: bool
    reachability_ok: bool
    compatibility_ok: bool
    all_checks_ok: bool
    counts: dict[str, int]
    issues: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "basic_checks_ok": self.basic_checks_ok,
            "reachability_ok": self.reachability_ok,
            "compatibility_ok": self.compatibility_ok,
            "all_checks_ok": self.all_checks_ok,
            "counts": self.counts,
            "issues": self.issues,
        }


def _task_compatible_agents_count(dataset: RoutingDataset, task) -> int:
    source = dataset.graph.nodes.get(task.source_node_id)
    if source is None:
        return 0
    allowed_types = CONTAINER_TO_VEHICLE_TYPES.get(task.container_type, set())

    count = 0
    for agent in dataset.fleet.agents.values():
        if agent.vehicle_type not in allowed_types:
            continue
        if task.compatible_vehicle_types and agent.vehicle_type not in task.compatible_vehicle_types:
            continue
        if not agent.supports_container(task.container_type):
            continue
        if agent.capacity_tons + 1e-9 < task.mass_tons:
            continue
        if source.center and not agent.is_compact:
            continue
        if agent.raw_volume_limit_m3 > 0 and task.volume_raw_m3 > agent.raw_volume_limit_m3 + 1e-9:
            continue
        body_need = task.body_volume_for_vehicle(agent.vehicle_type)
        if body_need <= 0 and task.volume_raw_m3 > 0 and agent.compaction_coeff > 0:
            body_need = task.volume_raw_m3 / agent.compaction_coeff
        if agent.body_volume_m3 > 0 and body_need > agent.body_volume_m3 + 1e-9:
            continue
        count += 1
    return count


def summarize_input_dataset(
    dataset: RoutingDataset,
    payload: dict[str, Any],
    *,
    dataset_path: Path | str | None = None,
) -> InputDatasetReport:
    basic_errors: list[str] = []
    schema_errors: list[str] = []
    uncovered_tasks: list[str] = []

    try:
        dataset.graph.validate()
    except Exception as exc:  # pragma: no cover - diagnostics path
        basic_errors.append(f"graph_validate: {exc}")

    try:
        dataset.fleet.validate()
    except Exception as exc:  # pragma: no cover - diagnostics path
        basic_errors.append(f"fleet_validate: {exc}")

    for task in dataset.tasks:
        if task.container_type not in CONTAINER_TO_VEHICLE_TYPES:
            schema_errors.append(f"unknown_container_type:{task.task_id}:{task.container_type}")
            continue

        if task.source_node_id not in dataset.graph.nodes:
            schema_errors.append(f"missing_source_node:{task.task_id}:{task.source_node_id}")
            continue
        if task.destination_node_id not in dataset.graph.nodes:
            schema_errors.append(f"missing_destination_node:{task.task_id}:{task.destination_node_id}")
            continue

        src = dataset.graph.nodes[task.source_node_id]
        if src.kind == "mno" and src.container_types and task.container_type not in src.container_types:
            schema_errors.append(
                f"task_container_mismatch:{task.task_id}:{task.container_type}:source={src.node_id}"
            )

        if _task_compatible_agents_count(dataset, task) == 0:
            uncovered_tasks.append(task.task_id)

    graph = core.build_nx_graph(dataset)
    cache: dict[tuple[str, str], tuple[list[str], float] | None] = {}
    reachability = core.check_reachability(graph, dataset, cache)

    reachability_ok = (
        len(reachability.get("unreachable_tasks", [])) == 0
        and len(reachability.get("unreachable_special_nodes", [])) == 0
    )
    compatibility_ok = len(uncovered_tasks) == 0
    basic_checks_ok = len(basic_errors) == 0 and len(schema_errors) == 0
    all_checks_ok = basic_checks_ok and reachability_ok and compatibility_ok

    counts = {
        "nodes": len(dataset.graph.nodes),
        "edges": len(dataset.graph.edges),
        "agents": len(dataset.fleet.agents),
        "tasks": len(dataset.tasks),
        "routes": len(dataset.routes),
    }

    issues = {
        "basic_errors": basic_errors,
        "schema_errors": schema_errors,
        "uncovered_tasks": uncovered_tasks,
        "reachability": reachability,
        "metadata_keys": sorted((payload.get("metadata") or {}).keys()),
    }

    path_text = str(dataset_path) if dataset_path is not None else "<in-memory>"
    return InputDatasetReport(
        dataset_path=path_text,
        basic_checks_ok=basic_checks_ok,
        reachability_ok=reachability_ok,
        compatibility_ok=compatibility_ok,
        all_checks_ok=all_checks_ok,
        counts=counts,
        issues=issues,
    )


def summarize_dataset_path(dataset_path: Path | str) -> InputDatasetReport:
    dataset, payload = load_dataset(dataset_path)
    return summarize_input_dataset(dataset, payload, dataset_path=dataset_path)
