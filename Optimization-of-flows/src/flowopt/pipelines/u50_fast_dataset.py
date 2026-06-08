from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_graph(edges: list[dict[str, Any]], node_ids: set[str]) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(node_ids)
    g.add_weighted_edges_from(
        (e["source_id"], e["target_id"], float(e.get("distance_km", 1.0) or 1.0))
        for e in edges
        if e["source_id"] in node_ids and e["target_id"] in node_ids
    )
    return g


def _stratified_task_sample(
    task_ids: list[str],
    task_xy: dict[str, tuple[float, float]],
    target_count: int,
    grid_size: int,
    rng: random.Random,
) -> list[str]:
    if target_count >= len(task_ids):
        return list(task_ids)

    xs = [task_xy[tid][0] for tid in task_ids]
    ys = [task_xy[tid][1] for tid in task_ids]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)

    buckets: dict[tuple[int, int], list[str]] = defaultdict(list)
    for tid in task_ids:
        x, y = task_xy[tid]
        cx = min(grid_size - 1, max(0, int((x - min_x) / span_x * grid_size)))
        cy = min(grid_size - 1, max(0, int((y - min_y) / span_y * grid_size)))
        buckets[(cx, cy)].append(tid)

    keys = list(buckets.keys())
    for key in keys:
        rng.shuffle(buckets[key])

    picked: list[str] = []
    while len(picked) < target_count:
        progressed = False
        rng.shuffle(keys)
        for key in keys:
            if not buckets[key]:
                continue
            picked.append(buckets[key].pop())
            progressed = True
            if len(picked) >= target_count:
                break
        if not progressed:
            break

    return picked


def _select_agents_in_depot(
    depot_agents: list[dict[str, Any]],
    target_count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if target_count >= len(depot_agents):
        return list(depot_agents)

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for agent in depot_agents:
        by_type[agent.get("vehicle_type", "unknown")].append(agent)

    for vt in by_type:
        by_type[vt].sort(key=lambda a: float(a.get("capacity_tons", 0.0) or 0.0), reverse=True)

    total = len(depot_agents)
    quotas: dict[str, int] = {}
    taken = 0
    for vt, items in by_type.items():
        q = int(math.floor(target_count * len(items) / total))
        if q > 0:
            quotas[vt] = min(q, len(items))
            taken += quotas[vt]
        else:
            quotas[vt] = 0

    remaining = target_count - taken
    type_order = sorted(by_type.keys(), key=lambda vt: len(by_type[vt]), reverse=True)
    idx = 0
    while remaining > 0 and type_order:
        vt = type_order[idx % len(type_order)]
        if quotas[vt] < len(by_type[vt]):
            quotas[vt] += 1
            remaining -= 1
        idx += 1
        if idx > 100000:
            break

    selected: list[dict[str, Any]] = []
    for vt, q in quotas.items():
        selected.extend(by_type[vt][:q])

    if len(selected) < target_count:
        selected_ids = {a["agent_id"] for a in selected}
        rest = [a for a in depot_agents if a["agent_id"] not in selected_ids]
        rest.sort(key=lambda a: float(a.get("capacity_tons", 0.0) or 0.0), reverse=True)
        selected.extend(rest[: target_count - len(selected)])

    rng.shuffle(selected)
    return selected[:target_count]


def build_fast_u50_dataset(
    *,
    base_dataset_path: Path | str,
    out_dataset_path: Path | str,
    downscale_factor: float = 10.0,
    seed: int = 42,
    grid_size: int = 6,
    safety_utilization: float = 0.85,
    min_tasks_per_depot: int = 1,
    min_agents_per_depot: int = 1,
) -> dict[str, Any]:
    base_path = Path(base_dataset_path)
    out_path = Path(out_dataset_path)
    if downscale_factor <= 0:
        raise ValueError("downscale_factor must be > 0")

    data = _load_json(base_path)
    meta = dict(data.get("metadata", {}))

    nodes = [dict(n) for n in data["graph"]["nodes"]]
    edges = [dict(e) for e in data["graph"]["edges"]]
    agents = [dict(a) for a in data.get("agents", [])]
    tasks = [dict(t) for t in data.get("tasks", [])]

    rng = random.Random(seed)
    node_by_id = {n["node_id"]: n for n in nodes}
    depot_ids = list(meta.get("depot_node_ids", []))
    if not depot_ids:
        raise RuntimeError("metadata.depot_node_ids is empty")

    # Build graph for distance estimation.
    graph = _build_graph(edges, set(node_by_id.keys()))
    rev = graph.reverse(copy=False)

    dep_to_source_dist: dict[str, dict[str, float]] = {}
    for dep in depot_ids:
        dep_to_source_dist[dep] = nx.single_source_dijkstra_path_length(rev, dep, weight="weight")

    def nearest_depot_for_source(source_id: str) -> str:
        best_dep, best = None, float("inf")
        for dep in depot_ids:
            d = dep_to_source_dist[dep].get(source_id)
            if d is not None and d < best:
                best = d
                best_dep = dep
        if best_dep is not None:
            return best_dep
        sx, sy = node_by_id[source_id]["x"], node_by_id[source_id]["y"]
        return min(
            depot_ids,
            key=lambda dep: (node_by_id[dep]["x"] - sx) ** 2 + (node_by_id[dep]["y"] - sy) ** 2,
        )

    # Group tasks by nearest depot.
    tasks_by_depot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    task_xy: dict[str, tuple[float, float]] = {}
    for task in tasks:
        source_id = task["source_node_id"]
        dep = nearest_depot_for_source(source_id)
        tasks_by_depot[dep].append(task)
        sn = node_by_id[source_id]
        task_xy[task["task_id"]] = (float(sn["x"]), float(sn["y"]))

    # Uniform reduction inside each depot zone + spatial stratified pick.
    selected_tasks: list[dict[str, Any]] = []
    for dep in depot_ids:
        depot_tasks = tasks_by_depot.get(dep, [])
        if not depot_tasks:
            continue
        target = int(math.ceil(len(depot_tasks) / downscale_factor))
        target = max(min_tasks_per_depot, target)
        target = min(target, len(depot_tasks))

        task_ids = [t["task_id"] for t in depot_tasks]
        picked_ids = set(_stratified_task_sample(task_ids, task_xy, target, grid_size, rng))
        selected_tasks.extend([t for t in depot_tasks if t["task_id"] in picked_ids])

    selected_tasks.sort(key=lambda t: t["task_id"])

    # Recompute nearest depot for selected tasks.
    sel_task_dep: dict[str, str] = {}
    for t in selected_tasks:
        sel_task_dep[t["task_id"]] = nearest_depot_for_source(t["source_node_id"])

    # Conservative required agents by depot from cycle-time budget.
    profiles = meta.get("vehicle_profiles", {})
    service_hours = meta.get("service_hours_by_container", {})

    # Depot-specific average speed/shift from available agents in that depot.
    agent_depots = meta.get("agent_depots", {})
    agents_by_depot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for agent in agents:
        dep = agent_depots.get(agent["agent_id"])
        if dep is None:
            continue
        agents_by_depot[dep].append(agent)

    depot_speed: dict[str, float] = {}
    depot_shift: dict[str, float] = {}
    for dep in depot_ids:
        dep_agents = agents_by_depot.get(dep, [])
        if not dep_agents:
            depot_speed[dep] = 24.0
            depot_shift[dep] = 10.0
            continue
        speeds = []
        shifts = []
        for a in dep_agents:
            vt = a.get("vehicle_type")
            p = profiles.get(vt, {})
            speeds.append(float(p.get("avg_speed_kmph", 24.0)))
            shifts.append(float(p.get("max_shift_hours", 10.0)))
        depot_speed[dep] = sum(speeds) / len(speeds)
        depot_shift[dep] = sum(shifts) / len(shifts)

    selected_objects = sorted({t["destination_node_id"] for t in selected_tasks})
    all_to_obj_dist: dict[str, dict[str, float]] = {}
    obj_to_all_dist: dict[str, dict[str, float]] = {}
    for obj in selected_objects:
        all_to_obj_dist[obj] = nx.single_source_dijkstra_path_length(rev, obj, weight="weight")
        obj_to_all_dist[obj] = nx.single_source_dijkstra_path_length(graph, obj, weight="weight")

    dep_to_all_dist: dict[str, dict[str, float]] = {}
    for dep in depot_ids:
        dep_to_all_dist[dep] = nx.single_source_dijkstra_path_length(graph, dep, weight="weight")

    required_agents: dict[str, int] = {dep: 0 for dep in depot_ids}
    required_hours: dict[str, float] = {dep: 0.0 for dep in depot_ids}

    for task in selected_tasks:
        tid = task["task_id"]
        dep = sel_task_dep[tid]
        src = task["source_node_id"]
        obj = task["destination_node_id"]

        d1 = dep_to_all_dist.get(dep, {}).get(src)
        d2 = all_to_obj_dist.get(obj, {}).get(src)
        d3 = obj_to_all_dist.get(obj, {}).get(dep)
        if d1 is None or d2 is None or d3 is None:
            # Skip from hours model if path estimate is unavailable.
            continue
        speed = max(depot_speed.get(dep, 24.0), 1e-6)
        service_h = float(service_hours.get(task.get("container_type", "A"), 0.22))
        cycle_h = (d1 + d2 + d3) / speed + service_h
        required_hours[dep] += cycle_h

    for dep in depot_ids:
        sel_count = sum(1 for t in selected_tasks if sel_task_dep[t["task_id"]] == dep)
        if sel_count == 0:
            required_agents[dep] = 0
            continue
        shift_h = max(depot_shift.get(dep, 10.0), 1e-6)
        cap_h = shift_h * max(safety_utilization, 1e-3)
        req = int(math.ceil(required_hours[dep] / cap_h)) if cap_h > 0 else sel_count
        req = max(min_agents_per_depot, req)
        required_agents[dep] = req

    # Select agents: uniform downscale per depot, then bump to required.
    selected_agents: list[dict[str, Any]] = []
    selected_agent_ids: set[str] = set()
    selected_agents_by_depot: dict[str, int] = {}

    for dep in depot_ids:
        dep_agents = agents_by_depot.get(dep, [])
        if not dep_agents:
            selected_agents_by_depot[dep] = 0
            continue

        target = int(math.ceil(len(dep_agents) / downscale_factor))
        target = max(min_agents_per_depot, target)
        target = max(target, required_agents.get(dep, 0))
        target = min(target, len(dep_agents))

        chosen = _select_agents_in_depot(dep_agents, target, rng)
        selected_agents.extend(chosen)
        selected_agents_by_depot[dep] = len(chosen)
        for a in chosen:
            selected_agent_ids.add(a["agent_id"])

    # Rebuild nodes: preserve topology, demote unused mno to road, align mass to selected tasks.
    mass_by_source: dict[str, float] = defaultdict(float)
    ctype_by_source: dict[str, set[str]] = defaultdict(set)
    for t in selected_tasks:
        src = t["source_node_id"]
        mass_by_source[src] += float(t.get("mass_tons", 0.0) or 0.0)
        ctype_by_source[src].add(str(t.get("container_type", "A")))

    selected_sources = set(mass_by_source.keys())
    demoted = 0
    for n in nodes:
        if n.get("kind") == "mno":
            if n["node_id"] in selected_sources:
                n["daily_mass_tons"] = round(float(mass_by_source[n["node_id"]]), 6)
                n["container_types"] = sorted(ctype_by_source[n["node_id"]])
            else:
                n["kind"] = "road"
                n["daily_mass_tons"] = 0.0
                n["container_types"] = []
                n["center"] = False
                demoted += 1

    # Filter agent depots mapping.
    new_agent_depots = {
        aid: dep
        for aid, dep in agent_depots.items()
        if aid in selected_agent_ids
    }

    # Update metadata counts.
    kind_counts: dict[str, int] = defaultdict(int)
    node_kind = {}
    for n in nodes:
        kind = n.get("kind", "unknown")
        kind_counts[kind] += 1
        node_kind[n["node_id"]] = kind
    road_edges = sum(
        1
        for e in edges
        if node_kind.get(e["source_id"]) == "road" and node_kind.get(e["target_id"]) == "road"
    )

    new_meta = dict(meta)
    new_meta["name"] = f"real_spb_u50_fast_f{downscale_factor:g}"
    new_meta["agent_depots"] = new_agent_depots
    new_meta.setdefault("fast_profile", {})
    new_meta["fast_profile"] = {
        "source_dataset": str(base_path),
        "downscale_factor": downscale_factor,
        "seed": seed,
        "grid_size": grid_size,
        "safety_utilization": safety_utilization,
        "selected_tasks": len(selected_tasks),
        "selected_agents": len(selected_agents),
        "demoted_mno_to_road": demoted,
        "required_agents_by_depot": required_agents,
        "selected_agents_by_depot": selected_agents_by_depot,
        "selected_tasks_by_depot": {
            dep: sum(1 for t in selected_tasks if sel_task_dep[t["task_id"]] == dep)
            for dep in depot_ids
        },
        "required_hours_by_depot": {dep: round(required_hours[dep], 6) for dep in depot_ids},
    }
    new_meta["counts"] = {
        "road_nodes": int(kind_counts.get("road", 0)),
        "road_edges": int(road_edges),
        "mno": int(kind_counts.get("mno", 0)),
        "object1": int(kind_counts.get("object1", 0)),
        "depots": int(kind_counts.get("depot", 0)),
        "agents": int(len(selected_agents)),
        "tasks": int(len(selected_tasks)),
        "routes": 0,
    }

    out_dataset = {
        "metadata": new_meta,
        "graph": {"nodes": nodes, "edges": edges},
        "agents": selected_agents,
        "tasks": selected_tasks,
        "routes": [],
        "derived": data.get("derived", {}),
    }

    _save_json(out_path, out_dataset)

    summary = {
        "source_dataset": str(base_path),
        "output_dataset": str(out_path),
        "downscale_factor": downscale_factor,
        "seed": seed,
        "before": {
            "agents": len(agents),
            "tasks": len(tasks),
            "mno_nodes": sum(1 for n in data["graph"]["nodes"] if n.get("kind") == "mno"),
        },
        "after": {
            "agents": len(selected_agents),
            "tasks": len(selected_tasks),
            "mno_nodes": sum(1 for n in nodes if n.get("kind") == "mno"),
            "task_sources": len(selected_sources),
        },
        "required_agents_by_depot": required_agents,
        "selected_agents_by_depot": selected_agents_by_depot,
        "selected_tasks_by_depot": new_meta["fast_profile"]["selected_tasks_by_depot"],
        "required_hours_by_depot": new_meta["fast_profile"]["required_hours_by_depot"],
    }

    summary_path = out_path.with_name(out_path.stem.replace("dataset", "summary") + ".json")
    _save_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fast balanced U50 dataset")
    parser.add_argument("--base-dataset", required=True)
    parser.add_argument("--out-dataset", required=True)
    parser.add_argument("--factor", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument("--safety-utilization", type=float, default=0.85)
    args = parser.parse_args()

    summary = build_fast_u50_dataset(
        base_dataset_path=Path(args.base_dataset),
        out_dataset_path=Path(args.out_dataset),
        downscale_factor=args.factor,
        seed=args.seed,
        grid_size=args.grid_size,
        safety_utilization=args.safety_utilization,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
