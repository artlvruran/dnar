from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import core
from ..dataset import CONTAINER_TO_VEHICLE_TYPES, dataset_from_dict
from .io import load_payload, save_payload
from .validation import summarize_input_dataset


SRC_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = SRC_ROOT / "data"

DEFAULT_BASE_DATASET_PATH = (
    DATA_ROOT / "real" / "clean_full_enriched" / "dataset_real_spb_clean_full_enriched.json"
)
DEFAULT_OUT_DIR = DATA_ROOT / "real" / "clean_full_enriched" / "simple"
DEFAULT_OUT_DATASET_PATH = DEFAULT_OUT_DIR / "dataset_real_spb_clean_full_enriched_simple.json"
DEFAULT_OUT_SUMMARY_PATH = DEFAULT_OUT_DIR / "summary_real_spb_clean_full_enriched_simple.json"


@dataclass
class CleanEnrichedSimpleBuildConfig:
    max_tasks: int = 16
    max_agents: int = 80
    selected_depot_id: str | None = None
    max_task_pool: int = 500


def _choose_depot(payload: dict[str, Any], preferred: str | None = None) -> str:
    depot_nodes = [n["node_id"] for n in payload["graph"]["nodes"] if n.get("kind") == "depot"]
    if preferred is not None:
        if preferred not in depot_nodes:
            raise ValueError(f"Requested depot {preferred} is not present in graph.")
        return preferred

    agent_depots: dict[str, str] = payload.get("metadata", {}).get("agent_depots", {})
    counts = Counter(agent_depots.values())
    if counts:
        return counts.most_common(1)[0][0]
    if not depot_nodes:
        raise ValueError("Dataset has no depot nodes.")
    return sorted(depot_nodes)[0]


def _task_compatible_with_agent(task: dict[str, Any], agent: dict[str, Any], source_node: dict[str, Any]) -> bool:
    container = str(task["container_type"])
    allowed_types = CONTAINER_TO_VEHICLE_TYPES.get(container, set())
    if agent["vehicle_type"] not in allowed_types:
        return False
    if task.get("compatible_vehicle_types") and agent["vehicle_type"] not in task["compatible_vehicle_types"]:
        return False
    cap_flag = f"cap_container_{container}"
    if cap_flag in agent and not bool(agent.get(cap_flag, False)):
        return False
    if float(agent.get("capacity_tons", 0.0)) + 1e-9 < float(task.get("mass_tons", 0.0)):
        return False
    if bool(source_node.get("center", False)) and not bool(agent.get("is_compact", False)):
        return False

    raw_vol = float(task.get("volume_raw_m3", 0.0) or 0.0)
    body_vol = float(agent.get("body_volume_m3", 0.0) or 0.0)
    comp = float(agent.get("compaction_coeff", 1.0) or 1.0)
    raw_limit = float(agent.get("max_raw_volume_m3", 0.0) or 0.0)
    if raw_limit <= 0 and body_vol > 0 and comp > 0:
        raw_limit = body_vol * comp
    if raw_vol > 0 and raw_limit > 0 and raw_vol > raw_limit + 1e-9:
        return False

    body_by_type = (task.get("volume_body_m3_by_vehicle_type") or {}).get(agent["vehicle_type"])
    if body_by_type is not None and body_vol > 0 and float(body_by_type) > body_vol + 1e-9:
        return False
    return True


def build_clean_enriched_simple_dataset(
    *,
    base_dataset_path: Path | str = DEFAULT_BASE_DATASET_PATH,
    out_dataset_path: Path | str = DEFAULT_OUT_DATASET_PATH,
    out_summary_path: Path | str | None = DEFAULT_OUT_SUMMARY_PATH,
    config: CleanEnrichedSimpleBuildConfig | None = None,
) -> dict[str, Any]:
    cfg = config or CleanEnrichedSimpleBuildConfig()
    payload = load_payload(base_dataset_path)
    dataset = dataset_from_dict(payload)

    selected_depot = _choose_depot(payload, cfg.selected_depot_id)
    node_by_id = {node["node_id"]: node for node in payload["graph"]["nodes"]}
    depot_node = node_by_id[selected_depot]

    agent_depots = payload.get("metadata", {}).get("agent_depots", {})
    depot_agents = [
        agent for agent in payload["agents"]
        if agent_depots.get(agent["agent_id"]) == selected_depot
    ]
    if not depot_agents:
        raise ValueError(f"No agents attached to depot {selected_depot}.")
    depot_agents.sort(key=lambda row: row["agent_id"])
    selected_agents = depot_agents[: max(1, int(cfg.max_agents))]
    selected_agent_ids = {agent["agent_id"] for agent in selected_agents}

    # Fast preselection by compatibility + local proximity in XY.
    task_candidates: list[tuple[float, dict[str, Any]]] = []
    dx0, dy0 = float(depot_node["x"]), float(depot_node["y"])
    for task in payload["tasks"]:
        source = node_by_id.get(task["source_node_id"])
        if source is None:
            continue
        if not any(_task_compatible_with_agent(task, agent, source) for agent in selected_agents):
            continue
        dx = float(source["x"]) - dx0
        dy = float(source["y"]) - dy0
        xy_dist2 = dx * dx + dy * dy
        task_candidates.append((xy_dist2, task))
    if not task_candidates:
        raise ValueError("No compatible tasks for selected depot/agents.")
    task_candidates.sort(key=lambda item: (item[0], item[1]["task_id"]))
    pooled = [task for _, task in task_candidates[: max(1, int(cfg.max_task_pool))]]
    selected_tasks = pooled[: max(1, int(cfg.max_tasks))]
    selected_source_ids = {task["source_node_id"] for task in selected_tasks}
    selected_destination_ids = {task["destination_node_id"] for task in selected_tasks}

    graph = core.build_nx_graph(dataset)
    cache: dict[tuple[str, str], tuple[list[str], float] | None] = {}
    keep_nodes: set[str] = {selected_depot}
    for task in selected_tasks:
        src = task["source_node_id"]
        dst = task["destination_node_id"]
        keep_nodes.add(src)
        keep_nodes.add(dst)
        for a, b in ((selected_depot, src), (src, dst), (dst, selected_depot)):
            path = core.shortest_path_cached(graph, cache, a, b)
            if path is not None:
                keep_nodes.update(path[0])

    nodes = []
    for raw_node in payload["graph"]["nodes"]:
        node_id = raw_node["node_id"]
        if node_id not in keep_nodes:
            continue
        node = deepcopy(raw_node)
        if node.get("kind") == "mno" and node_id not in selected_source_ids:
            node["kind"] = "road"
            node["daily_mass_tons"] = 0.0
            node["container_types"] = []
            node["center"] = False
        if str(node.get("kind", "")).startswith("object") and node_id not in selected_destination_ids:
            node["kind"] = "road"
            node["object_day_capacity_tons"] = 0.0
            node["object_year_capacity_tons"] = 0.0
            node["object_day_capacity_volume_m3"] = 0.0
        if node.get("kind") == "depot" and node_id != selected_depot:
            node["kind"] = "road"
        nodes.append(node)
    keep_node_ids = {node["node_id"] for node in nodes}
    edges = [
        deepcopy(edge)
        for edge in payload["graph"]["edges"]
        if edge["source_id"] in keep_node_ids and edge["target_id"] in keep_node_ids
    ]

    metadata = deepcopy(payload.get("metadata", {}))
    metadata["name"] = f"{metadata.get('name', 'real_spb')}_clean_enriched_simple"
    metadata["profile"] = "clean_enriched_simple"
    metadata["profile_description"] = "Reduced one-depot subset for clean_full_enriched debug runs"
    metadata["selected_depot_node_id"] = selected_depot
    metadata["source_dataset"] = str(Path(base_dataset_path))
    metadata["agent_depots"] = {agent_id: selected_depot for agent_id in selected_agent_ids}
    metadata["depot_node_ids"] = [selected_depot]
    metadata["counts"] = {
        "nodes": len(nodes),
        "road_edges": len(edges),
        "agents": len(selected_agents),
        "tasks": len(selected_tasks),
        "routes": 0,
    }
    metadata["clean_enriched_simple"] = {
        "max_tasks": cfg.max_tasks,
        "max_agents": cfg.max_agents,
        "max_task_pool": cfg.max_task_pool,
    }

    simple_payload = {
        "metadata": metadata,
        "graph": {"nodes": nodes, "edges": edges},
        "agents": selected_agents,
        "tasks": selected_tasks,
        "routes": [],
        "derived": {},
    }

    out_dataset_path = save_payload(simple_payload, out_dataset_path)
    simple_dataset = dataset_from_dict(simple_payload)
    report = summarize_input_dataset(simple_dataset, simple_payload, dataset_path=out_dataset_path)

    summary = {
        "dataset_path": str(out_dataset_path),
        "selected_depot": selected_depot,
        "selected_agents": len(selected_agents),
        "selected_tasks": len(selected_tasks),
        "selected_nodes": len(nodes),
        "selected_edges": len(edges),
        "report": report.as_dict(),
    }
    if out_summary_path is not None:
        save_payload(summary, out_summary_path)
    return summary
