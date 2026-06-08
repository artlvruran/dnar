from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import core
from ..dataset import CONTAINER_TO_VEHICLE_TYPES, RoutingDataset, dataset_from_dict
from .io import load_payload, save_payload
from .validation import summarize_input_dataset


SRC_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = SRC_ROOT / "data"
DEFAULT_BASE_DATASET_PATH = DATA_ROOT / "real" / "dataset_real_spb_ready.json"
DEFAULT_OUT_DATASET_PATH = DATA_ROOT / "real" / "real_simple" / "dataset_real_spb_simple.json"
DEFAULT_OUT_SUMMARY_PATH = DATA_ROOT / "real" / "real_simple" / "summary_real_spb_simple.json"


@dataclass
class RealSimpleBuildConfig:
    max_tasks: int = 32
    max_agents: int = 64
    seed: int = 42
    selected_depot_id: str | None = None


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
    allowed_types = CONTAINER_TO_VEHICLE_TYPES.get(task["container_type"], set())
    if agent["vehicle_type"] not in allowed_types:
        return False
    if task.get("compatible_vehicle_types") and agent["vehicle_type"] not in task["compatible_vehicle_types"]:
        return False
    cap_flag = f"cap_container_{task['container_type']}"
    if cap_flag in agent and not bool(agent.get(cap_flag, False)):
        return False
    if float(agent["capacity_tons"]) + 1e-9 < float(task["mass_tons"]):
        return False
    if bool(source_node.get("center", False)) and not bool(agent.get("is_compact", False)):
        return False
    task_raw_volume = float(task.get("volume_raw_m3", 0.0) or 0.0)
    max_raw_volume = float(agent.get("max_raw_volume_m3", 0.0) or 0.0)
    if max_raw_volume <= 0:
        body_volume = float(agent.get("body_volume_m3", 0.0) or 0.0)
        compaction = float(agent.get("compaction_coeff", 1.0) or 1.0)
        if body_volume > 0 and compaction > 0:
            max_raw_volume = body_volume * compaction
    if task_raw_volume > 0 and max_raw_volume > 0 and task_raw_volume > max_raw_volume + 1e-9:
        return False
    body_by_vt = (task.get("volume_body_m3_by_vehicle_type") or {}).get(agent["vehicle_type"])
    if body_by_vt is not None and float(agent.get("body_volume_m3", 0.0) or 0.0) > 0:
        if float(body_by_vt) > float(agent["body_volume_m3"]) + 1e-9:
            return False
    return True


def _select_tasks(
    *,
    payload: dict[str, Any],
    dataset: RoutingDataset,
    selected_depot: str,
    depot_agents: list[dict[str, Any]],
    max_tasks: int,
) -> list[dict[str, Any]]:
    graph = core.build_nx_graph(dataset)
    cache: dict[tuple[str, str], tuple[list[str], float] | None] = {}

    task_obj_by_id = {task.task_id: task for task in dataset.tasks}
    source_node_by_id = {node["node_id"]: node for node in payload["graph"]["nodes"]}

    scored: list[tuple[float, str]] = []
    for raw_task in payload["tasks"]:
        task = task_obj_by_id.get(raw_task["task_id"])
        if task is None:
            continue

        source_node = source_node_by_id.get(raw_task["source_node_id"])
        if source_node is None:
            continue

        if not any(_task_compatible_with_agent(raw_task, agent, source_node) for agent in depot_agents):
            continue

        legs = [
            core.shortest_path_cached(graph, cache, selected_depot, task.source_node_id),
            core.shortest_path_cached(graph, cache, task.source_node_id, task.destination_node_id),
            core.shortest_path_cached(graph, cache, task.destination_node_id, selected_depot),
        ]
        if any(leg is None for leg in legs):
            continue

        score = float(sum(leg[1] for leg in legs if leg is not None))
        scored.append((score, raw_task["task_id"]))

    if not scored:
        raise ValueError("No candidate tasks reachable from selected depot.")

    scored.sort(key=lambda item: (item[0], item[1]))
    selected_ids = {task_id for _, task_id in scored[:max_tasks]}

    rank = {task_id: i for i, (_, task_id) in enumerate(scored)}
    selected_tasks = [task for task in payload["tasks"] if task["task_id"] in selected_ids]
    selected_tasks.sort(key=lambda task: rank.get(task["task_id"], 10**9))
    return selected_tasks


def _select_agents(
    *,
    selected_tasks: list[dict[str, Any]],
    source_node_by_id: dict[str, dict[str, Any]],
    depot_agents: list[dict[str, Any]],
    max_agents: int,
) -> list[dict[str, Any]]:
    agents_sorted = sorted(depot_agents, key=lambda a: a["agent_id"])
    selected_ids: set[str] = set()

    for task in selected_tasks:
        source = source_node_by_id[task["source_node_id"]]
        already_covered = False
        for agent in agents_sorted:
            if agent["agent_id"] in selected_ids and _task_compatible_with_agent(task, agent, source):
                already_covered = True
                break
        if already_covered:
            continue

        chosen = None
        for agent in agents_sorted:
            if _task_compatible_with_agent(task, agent, source):
                chosen = agent
                break
        if chosen is None:
            raise ValueError(f"Task {task['task_id']} has no compatible agent in selected depot.")
        selected_ids.add(chosen["agent_id"])

    target_agents = max(max_agents, len(selected_ids))
    for agent in agents_sorted:
        if len(selected_ids) >= target_agents:
            break
        selected_ids.add(agent["agent_id"])

    selected_agents = [agent for agent in agents_sorted if agent["agent_id"] in selected_ids]
    return selected_agents


def _collect_subgraph_nodes(
    *,
    dataset: RoutingDataset,
    selected_depot: str,
    selected_tasks: list[dict[str, Any]],
) -> set[str]:
    graph = core.build_nx_graph(dataset)
    cache: dict[tuple[str, str], tuple[list[str], float] | None] = {}

    keep_nodes: set[str] = {selected_depot}
    for task in selected_tasks:
        src = task["source_node_id"]
        dst = task["destination_node_id"]
        keep_nodes.add(src)
        keep_nodes.add(dst)

        legs = [
            core.shortest_path_cached(graph, cache, selected_depot, src),
            core.shortest_path_cached(graph, cache, src, dst),
            core.shortest_path_cached(graph, cache, dst, selected_depot),
        ]
        for leg in legs:
            if leg is None:
                continue
            keep_nodes.update(leg[0])

    return keep_nodes


def _count_node_kinds(nodes: list[dict[str, Any]]) -> dict[str, int]:
    c = Counter(node.get("kind", "road") for node in nodes)
    return {
        "road_nodes": int(c.get("road", 0)),
        "mno": int(c.get("mno", 0)),
        "object1": int(c.get("object1", 0)),
        "object2": int(c.get("object2", 0)),
        "depots": int(c.get("depot", 0)),
    }


def build_real_simple_dataset(
    *,
    base_dataset_path: Path | str = DEFAULT_BASE_DATASET_PATH,
    out_dataset_path: Path | str = DEFAULT_OUT_DATASET_PATH,
    out_summary_path: Path | str | None = DEFAULT_OUT_SUMMARY_PATH,
    config: RealSimpleBuildConfig | None = None,
) -> dict[str, Any]:
    cfg = config or RealSimpleBuildConfig()

    base_payload = load_payload(base_dataset_path)
    base_dataset = dataset_from_dict(base_payload)
    base_report = summarize_input_dataset(base_dataset, base_payload, dataset_path=base_dataset_path)
    if not base_report.all_checks_ok:
        raise ValueError(
            "Base dataset failed input checks; see base_report in returned summary for details."
        )

    selected_depot = _choose_depot(base_payload, cfg.selected_depot_id)

    source_node_by_id = {node["node_id"]: node for node in base_payload["graph"]["nodes"]}
    agent_depots = base_payload.get("metadata", {}).get("agent_depots", {})
    depot_agents = [
        agent for agent in base_payload["agents"]
        if agent_depots.get(agent["agent_id"]) == selected_depot
    ]
    if not depot_agents:
        raise ValueError(f"No agents attached to depot {selected_depot}.")

    selected_tasks = _select_tasks(
        payload=base_payload,
        dataset=base_dataset,
        selected_depot=selected_depot,
        depot_agents=depot_agents,
        max_tasks=cfg.max_tasks,
    )
    selected_agents = _select_agents(
        selected_tasks=selected_tasks,
        source_node_by_id=source_node_by_id,
        depot_agents=depot_agents,
        max_agents=cfg.max_agents,
    )
    selected_agent_ids = {agent["agent_id"] for agent in selected_agents}

    keep_nodes = _collect_subgraph_nodes(
        dataset=base_dataset,
        selected_depot=selected_depot,
        selected_tasks=selected_tasks,
    )

    nodes: list[dict[str, Any]] = []
    for raw_node in base_payload["graph"]["nodes"]:
        node_id = raw_node["node_id"]
        if node_id not in keep_nodes:
            continue
        node = deepcopy(raw_node)
        if node.get("kind") == "depot" and node_id != selected_depot:
            node["kind"] = "road"
        nodes.append(node)

    keep_node_ids = {node["node_id"] for node in nodes}
    edges = [
        deepcopy(edge)
        for edge in base_payload["graph"]["edges"]
        if edge["source_id"] in keep_node_ids and edge["target_id"] in keep_node_ids
    ]

    metadata = deepcopy(base_payload.get("metadata", {}))
    metadata["name"] = f"{metadata.get('name', 'real_spb')}_real_simple"
    metadata["profile"] = "real_simple"
    metadata["profile_description"] = "Reduced one-depot subset for fast smoke runs"
    metadata["selected_depot_node_id"] = selected_depot
    metadata["source_dataset"] = str(Path(base_dataset_path))
    metadata["real_simple"] = {
        "max_tasks": cfg.max_tasks,
        "max_agents": cfg.max_agents,
        "seed": cfg.seed,
    }
    metadata["agent_depots"] = {agent_id: selected_depot for agent_id in selected_agent_ids}
    metadata["depot_node_ids"] = [selected_depot]

    node_kind_counts = _count_node_kinds(nodes)
    metadata["counts"] = {
        **node_kind_counts,
        "road_edges": len(edges),
        "agents": len(selected_agents),
        "tasks": len(selected_tasks),
        "routes": 0,
    }

    simple_payload = {
        "metadata": metadata,
        "graph": {
            "nodes": nodes,
            "edges": edges,
        },
        "agents": selected_agents,
        "tasks": selected_tasks,
        "routes": [],
        "derived": {},
    }

    out_dataset_path = save_payload(simple_payload, out_dataset_path)

    simple_dataset = dataset_from_dict(simple_payload)
    simple_report = summarize_input_dataset(simple_dataset, simple_payload, dataset_path=out_dataset_path)

    summary = {
        "base_report": base_report.as_dict(),
        "simple_report": simple_report.as_dict(),
        "selected_depot": selected_depot,
        "selected_agents": len(selected_agents),
        "selected_tasks": len(selected_tasks),
        "selected_nodes": len(nodes),
        "selected_edges": len(edges),
        "dataset_path": str(out_dataset_path),
    }

    if out_summary_path is not None:
        save_payload(summary, out_summary_path)

    return summary
