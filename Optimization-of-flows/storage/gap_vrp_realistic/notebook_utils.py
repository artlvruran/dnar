from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import math
import random
import sys
from typing import Any

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import plotly.graph_objects as go


def fraction_tag(agent_fraction: float) -> str:
    if not (0.0 < agent_fraction <= 1.0):
        raise ValueError("agent_fraction must be in (0, 1].")
    text = f"{agent_fraction:.4f}".rstrip("0").rstrip(".")
    return f"af_{text.replace('.', 'p')}"


def _sanitize_release_metadata(payload: dict[str, Any]) -> None:
    metadata = payload.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        return

    # Remove raw-source references (absolute paths, original XLSX names, cache links).
    drop_top = {
        "source_xlsx",
        "source_excel",
        "source_file",
        "source_path",
        "input_xlsx",
        "input_file",
        "input_path",
        "artifacts",
    }
    for key in drop_top:
        metadata.pop(key, None)

    preprocess = metadata.get("preprocess")
    if isinstance(preprocess, dict):
        drop_preprocess = {
            "source_xlsx",
            "source_excel",
            "source_file",
            "source_path",
            "input_xlsx",
            "input_file",
            "input_path",
            "graph_cache",
            "graphml_path",
            "osm_cache_path",
        }
        for key in drop_preprocess:
            preprocess.pop(key, None)

    metadata["is_anonymized"] = True
    metadata["source_description"] = "anonymized_real_export_spb_march_2026"


def _build_nx_graph(payload: dict[str, Any]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for node in payload["graph"]["nodes"]:
        graph.add_node(node["node_id"], x=float(node["x"]), y=float(node["y"]), kind=node.get("kind", "road"))
    for edge in payload["graph"]["edges"]:
        u = edge["source_id"]
        v = edge["target_id"]
        d = float(edge["distance_km"])
        if graph.has_edge(u, v):
            if d < graph[u][v]["distance_km"]:
                graph[u][v]["distance_km"] = d
        else:
            graph.add_edge(u, v, distance_km=d)
    return graph


def _shortest_path_distance_cached(
    graph: nx.DiGraph,
    cache: dict[tuple[str, str], float | None],
    source: str,
    target: str,
) -> float | None:
    key = (source, target)
    if key in cache:
        return cache[key]
    if source == target:
        cache[key] = 0.0
        return 0.0
    try:
        d = nx.shortest_path_length(graph, source=source, target=target, weight="distance_km")
    except nx.NetworkXNoPath:
        cache[key] = None
        return None
    cache[key] = float(d)
    return cache[key]


def _build_solver_ready_derived(payload: dict[str, Any]) -> dict[str, Any]:
    tasks = payload.get("tasks", [])
    agents = payload.get("agents", [])
    nodes = payload.get("graph", {}).get("nodes", [])
    metadata = payload.get("metadata", {})

    node_by_id = {n["node_id"]: n for n in nodes}
    object_nodes = [n for n in nodes if str(n.get("kind", "")).startswith("object")]
    object_ids = [n["node_id"] for n in object_nodes]

    task_ids = [t["task_id"] for t in tasks]
    agent_ids = [a["agent_id"] for a in agents]

    task_index = {tid: i for i, tid in enumerate(task_ids)}
    agent_index = {aid: i for i, aid in enumerate(agent_ids)}
    object_index = {oid: i for i, oid in enumerate(object_ids)}

    container_to_vehicle = metadata.get("container_to_vehicle_types", {})
    vehicle_profiles = metadata.get("vehicle_profiles", {})
    agent_depots = metadata.get("agent_depots", {})

    graph = _build_nx_graph(payload)
    dist_cache: dict[tuple[str, str], float | None] = {}

    task_agent_dense: list[list[int]] = [[0] * len(agent_ids) for _ in range(len(task_ids))]
    task_agent_sparse: dict[str, list[str]] = {}

    for ti, task in enumerate(tasks):
        source_node = node_by_id.get(task["source_node_id"], {})
        allowed_vt = set(container_to_vehicle.get(task["container_type"], []))
        comp_ids: list[str] = []
        for ai, agent in enumerate(agents):
            ok = True
            if allowed_vt and agent["vehicle_type"] not in allowed_vt:
                ok = False
            if float(agent["capacity_tons"]) + 1e-9 < float(task["mass_tons"]):
                ok = False
            if bool(source_node.get("center", False)) and not bool(agent.get("is_compact", False)):
                ok = False
            if ok:
                task_agent_dense[ti][ai] = 1
                comp_ids.append(agent["agent_id"])
        task_agent_sparse[task["task_id"]] = comp_ids

    task_object_assigned_dense: list[list[int]] = [[0] * len(object_ids) for _ in range(len(task_ids))]
    task_object_reachable_dense: list[list[int]] = [[0] * len(object_ids) for _ in range(len(task_ids))]
    task_object_distance_dense: list[list[float | None]] = [[None] * len(object_ids) for _ in range(len(task_ids))]

    task_object_assigned_sparse: dict[str, list[str]] = {}
    task_object_reachable_sparse: dict[str, list[str]] = {}

    for ti, task in enumerate(tasks):
        src = task["source_node_id"]
        assigned = task["destination_node_id"]
        assigned_list: list[str] = []
        reachable_list: list[str] = []

        for oi, obj_id in enumerate(object_ids):
            if obj_id == assigned:
                task_object_assigned_dense[ti][oi] = 1
                assigned_list.append(obj_id)

            d = _shortest_path_distance_cached(graph, dist_cache, src, obj_id)
            if d is not None:
                task_object_reachable_dense[ti][oi] = 1
                task_object_distance_dense[ti][oi] = round(float(d), 4)
                reachable_list.append(obj_id)

        task_object_assigned_sparse[task["task_id"]] = assigned_list
        task_object_reachable_sparse[task["task_id"]] = reachable_list

    agent_limits: list[dict[str, Any]] = []
    for agent in agents:
        vt = str(agent["vehicle_type"])
        profile = vehicle_profiles.get(vt, {})
        agent_limits.append(
            {
                "agent_id": agent["agent_id"],
                "vehicle_type": vt,
                "capacity_tons": float(agent["capacity_tons"]),
                "is_compact": bool(agent.get("is_compact", False)),
                "home_depot_node_id": agent_depots.get(agent["agent_id"]),
                "max_daily_km": float(profile.get("max_daily_km", 0.0)),
                "max_shift_hours": float(profile.get("max_shift_hours", 0.0)),
                "avg_speed_kmph": float(profile.get("avg_speed_kmph", 0.0)),
            }
        )

    object_limits: list[dict[str, Any]] = []
    for node in object_nodes:
        object_limits.append(
            {
                "object_node_id": node["node_id"],
                "day_capacity_tons": float(node.get("object_day_capacity_tons", 0.0)),
                "year_capacity_tons": float(node.get("object_year_capacity_tons", 0.0)),
            }
        )

    task_requirements: list[dict[str, Any]] = []
    for task in tasks:
        task_requirements.append(
            {
                "task_id": task["task_id"],
                "source_node_id": task["source_node_id"],
                "assigned_destination_node_id": task["destination_node_id"],
                "container_type": task["container_type"],
                "mass_tons": float(task["mass_tons"]),
            }
        )

    mathematical_model = {
        "decision_variables": {
            "x_ta": "x_{t,a} in {0,1}: task t assigned to agent a",
            "y_to": "y_{t,o} in {0,1}: task t assigned to object o",
        },
        "constraints": [
            "(C1) sum_a x_{t,a} = 1 for each task t",
            "(C2) x_{t,a} <= M_task_agent[t,a]",
            "(C3) sum_o y_{t,o} = 1 for each task t",
            "(C4) y_{t,o} <= M_task_object_assigned[t,o] (strict current setup)",
            "(C5) sum_t mass_t * y_{t,o} <= day_capacity_o for each object o",
            "(C6) for each agent a: total_km_a <= max_daily_km_a, total_hours_a <= max_shift_hours_a",
            "(C7) for each agent-task pair: mass_t <= capacity_a if x_{t,a}=1",
        ],
        "notes": [
            "C6 requires routing sequence/time model; this dataset provides limits and compatibility matrices.",
            "For alternative object assignment experiments use M_task_object_reachable instead of assigned-only matrix.",
        ],
    }

    return {
        "solver_ready": {
            "schema_version": "1.0",
            "sets": {
                "tasks": task_ids,
                "agents": agent_ids,
                "objects": object_ids,
            },
            "indices": {
                "task_index": task_index,
                "agent_index": agent_index,
                "object_index": object_index,
            },
            "limits": {
                "agent_limits": agent_limits,
                "object_limits": object_limits,
                "task_requirements": task_requirements,
                "service_hours_by_container": metadata.get("service_hours_by_container", {}),
            },
            "compatibility": {
                "task_agent_dense": task_agent_dense,
                "task_agent_sparse": task_agent_sparse,
                "task_object_assigned_dense": task_object_assigned_dense,
                "task_object_assigned_sparse": task_object_assigned_sparse,
                "task_object_reachable_dense": task_object_reachable_dense,
                "task_object_reachable_sparse": task_object_reachable_sparse,
            },
            "costs": {
                "task_object_distance_km_dense": task_object_distance_dense,
                "task_assigned_distance_km": {
                    t["task_id"]: (
                        task_object_distance_dense[task_index[t["task_id"]]][object_index[t["destination_node_id"]]]
                        if t["destination_node_id"] in object_index
                        else None
                    )
                    for t in tasks
                },
            },
            "model": mathematical_model,
        }
    }


def prepare_dataset_variant(
    *,
    base_dataset_path: Path,
    out_dataset_path: Path,
    agent_fraction: float,
    seed: int,
) -> dict[str, Any]:
    payload = json.loads(base_dataset_path.read_text(encoding="utf-8"))
    _sanitize_release_metadata(payload)
    if not (0.0 < agent_fraction <= 1.0):
        raise ValueError("agent_fraction must be in (0, 1].")

    agents = payload.get("agents", [])
    if not agents:
        raise ValueError("Dataset has no agents.")

    if agent_fraction < 1.0:
        keep_n = max(1, int(round(len(agents) * agent_fraction)))
        rng = random.Random(seed)
        keep_idx = sorted(rng.sample(range(len(agents)), keep_n))
        agents_new = [agents[i] for i in keep_idx]
    else:
        agents_new = list(agents)

    keep_agent_ids = {a["agent_id"] for a in agents_new}
    payload["agents"] = agents_new

    metadata = payload.setdefault("metadata", {})
    old_depots = metadata.get("agent_depots", {})
    new_depots = {aid: dep for aid, dep in old_depots.items() if aid in keep_agent_ids}
    metadata["agent_depots"] = new_depots
    metadata["depot_node_ids"] = sorted(set(new_depots.values()))
    metadata["agent_fraction"] = float(agent_fraction)
    metadata["fraction_tag"] = fraction_tag(agent_fraction)

    counts = metadata.setdefault("counts", {})
    counts["agents"] = len(agents_new)
    counts["depots"] = len(set(new_depots.values()))

    # Build explicit constraints/matrices layer.
    derived = payload.setdefault("derived", {})
    derived.update(_build_solver_ready_derived(payload))

    out_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    out_dataset_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _ensure_notebook_sys_path(notebook_dir: Path) -> None:
    if str(notebook_dir) not in sys.path:
        sys.path.insert(0, str(notebook_dir))


def run_solver(
    *,
    notebook_dir: Path,
    dataset_path: Path,
    output_root: Path,
    days: int,
    seed: int,
    mass_noise: float,
    render_first_day: bool,
    render_utilization: bool,
) -> pd.DataFrame:
    _ensure_notebook_sys_path(notebook_dir)
    import simple_solver_min as sm

    runner = sm.MultiDayRealRunner(dataset_path, output_root)
    results = runner.run(
        days=days,
        seed=seed,
        mass_noise=mass_noise,
        render_first_day=render_first_day,
        render_utilization=render_utilization,
    )

    rows = [
        {
            "day": r.day_index,
            "status": r.solver_status,
            "assigned_routes": r.assigned_routes,
            "unassigned_tasks": r.unassigned_tasks,
            "active_agents": r.active_agents,
            "transport_work_ton_km": r.transport_work_ton_km,
            "all_checks_ok": r.all_checks_ok,
            "output_dir": str(r.output_dir),
        }
        for r in results
    ]

    output_root.mkdir(parents=True, exist_ok=True)
    run_results_path = output_root / "latest_run_results.json"
    run_results_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return pd.DataFrame(rows)


def load_results_df(output_root: Path) -> pd.DataFrame:
    run_results_path = output_root / "latest_run_results.json"
    if not run_results_path.exists():
        raise FileNotFoundError(f"Run summary file not found: {run_results_path}")
    rows = json.loads(run_results_path.read_text(encoding="utf-8"))
    return pd.DataFrame(rows)


def plot_daily_metrics(results_df: pd.DataFrame) -> None:
    x = results_df["day"].tolist()
    tr = [v if pd.notna(v) else 0.0 for v in results_df["transport_work_ton_km"].tolist()]
    unassigned = results_df["unassigned_tasks"].tolist()

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(x, tr, marker="o", color="#1f77b4", label="TR, ton*km")
    ax1.set_xlabel("Day")
    ax1.set_ylabel("Transport work (ton*km)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(x, unassigned, marker="s", color="#d62728", label="Unassigned tasks")
    ax2.set_ylabel("Unassigned tasks", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    ax1.set_title("Real-data pack daily run summary")
    plt.tight_layout()
    plt.show()


def show_day_classic_solution(
    *,
    output_root: Path,
    day_index: int = 1,
    title_suffix: str = "",
) -> dict[str, Any]:
    day_dir = output_root / f"day_{day_index:03d}"
    report_path = day_dir / "feasibility_report.json"
    map_path = day_dir / "solution_map.png"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing report: {report_path}")
    if not map_path.exists():
        raise FileNotFoundError(f"Missing map: {map_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    img = mpimg.imread(map_path)
    plt.figure(figsize=(14, 10))
    plt.imshow(img)
    plt.axis("off")
    suffix = f" | {title_suffix}" if title_suffix else ""
    plt.title(f"Day {day_index} solution map (classic){suffix}")
    plt.show()

    task_cov = report.get("task_source_coverage", {})
    mno_cov = report.get("mno_scope_coverage", {})
    total_tasks = int(task_cov.get("total_tasks", 0))
    covered_tasks = int(task_cov.get("covered_tasks", 0))
    total_sources = int(task_cov.get("total_unique_task_sources", 0))
    covered_sources = int(task_cov.get("covered_unique_task_sources", 0))
    eligible_mno = int(mno_cov.get("eligible_mno_count", 0))
    covered_mno = int(mno_cov.get("covered_eligible_mno_count", 0))

    metrics_df = pd.DataFrame(
        [
            {
                "solver_status": report.get("solver_status"),
                "all_checks_ok": bool(report.get("constraints_check_summary", {}).get("all_checks_ok", False)),
                "assigned_routes": int(report.get("assigned_routes", 0)),
                "assigned_tasks": int(report.get("assigned_tasks", 0)),
                "unassigned_tasks": len(report.get("unassigned_tasks", [])),
                "active_agents": int(report.get("active_agents", 0)),
                "transport_work_ton_km": report.get("transport_work_ton_km"),
            }
        ]
    )

    coverage_df = pd.DataFrame(
        [
            {
                "scope": "tasks",
                "covered": covered_tasks,
                "total": total_tasks,
                "coverage_pct": round(100.0 * covered_tasks / total_tasks, 2) if total_tasks > 0 else None,
            },
            {
                "scope": "task_sources",
                "covered": covered_sources,
                "total": total_sources,
                "coverage_pct": round(100.0 * covered_sources / total_sources, 2) if total_sources > 0 else None,
            },
            {
                "scope": "eligible_mno",
                "covered": covered_mno,
                "total": eligible_mno,
                "coverage_pct": round(100.0 * covered_mno / eligible_mno, 2) if eligible_mno > 0 else None,
            },
        ]
    )

    return {
        "day_dir": day_dir,
        "report": report,
        "metrics_df": metrics_df,
        "coverage_df": coverage_df,
        "map_path": map_path,
    }


def _build_graph(dataset: dict[str, Any]) -> tuple[nx.DiGraph, dict[str, tuple[float, float, str]]]:
    node_xy: dict[str, tuple[float, float, str]] = {}
    for node in dataset["graph"]["nodes"]:
        node_xy[node["node_id"]] = (float(node["x"]), float(node["y"]), node.get("kind", "road"))

    graph = nx.DiGraph()
    for node_id, (x, y, kind) in node_xy.items():
        graph.add_node(node_id, x=x, y=y, kind=kind)
    for edge in dataset["graph"]["edges"]:
        u = edge["source_id"]
        v = edge["target_id"]
        w = float(edge["distance_km"])
        if graph.has_edge(u, v):
            if w < graph[u][v]["distance_km"]:
                graph[u][v]["distance_km"] = w
        else:
            graph.add_edge(u, v, distance_km=w)
    return graph, node_xy


def _route_order_key(route: dict[str, Any]) -> int:
    rid = str(route.get("route_id", ""))
    try:
        return int(rid.split("_")[-1])
    except Exception:
        return 10**9


def _resolve_selected_agents(
    routes_by_agent: dict[str, list[dict[str, Any]]],
    selected_agent_indices: list[int] | None,
    selected_agent_ids: list[str] | None,
) -> tuple[list[str], pd.DataFrame]:
    active_agents = sorted(routes_by_agent.keys())
    rows = [{"index": i + 1, "agent_id": aid} for i, aid in enumerate(active_agents)]
    index_df = pd.DataFrame(rows)

    if selected_agent_ids:
        selected = [aid for aid in selected_agent_ids if aid in routes_by_agent]
    elif selected_agent_indices:
        selected = []
        for idx in selected_agent_indices:
            if 1 <= int(idx) <= len(active_agents):
                selected.append(active_agents[int(idx) - 1])
    else:
        selected = active_agents

    return selected, index_df


def _build_segments(
    dataset: dict[str, Any],
    solution_plan: dict[str, Any],
    selected_agents: list[str],
) -> tuple[list[dict[str, Any]], dict[str, tuple[float, float, str]], nx.DiGraph]:
    graph, node_xy = _build_graph(dataset)
    routes_by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for route in solution_plan["routes"]:
        routes_by_agent[route["agent_id"]].append(route)
    for aid in routes_by_agent:
        routes_by_agent[aid].sort(key=_route_order_key)

    agent_depots = dataset.get("metadata", {}).get("agent_depots", {})

    path_cache: dict[tuple[str, str], tuple[list[str], float] | None] = {}

    def shortest_path_cached(source: str, target: str) -> tuple[list[str], float] | None:
        key = (source, target)
        if key in path_cache:
            return path_cache[key]
        if source == target:
            path_cache[key] = ([source], 0.0)
            return path_cache[key]
        try:
            p = nx.shortest_path(graph, source=source, target=target, weight="distance_km")
        except nx.NetworkXNoPath:
            path_cache[key] = None
            return None
        dist = 0.0
        for i in range(len(p) - 1):
            dist += float(graph[p[i]][p[i + 1]]["distance_km"])
        path_cache[key] = (p, dist)
        return path_cache[key]

    segments: list[dict[str, Any]] = []
    for agent_id in selected_agents:
        agent_routes = routes_by_agent.get(agent_id, [])
        if not agent_routes:
            continue
        depot = agent_depots.get(agent_id)
        if depot not in node_xy:
            continue
        current = depot

        for idx, route in enumerate(agent_routes):
            service_path = [nid for nid in route.get("path", []) if nid in node_xy]
            if len(service_path) < 2:
                continue

            service_start = service_path[0]
            service_end = service_path[-1]
            route_id = route.get("route_id", "")
            task_id = route.get("task_id", "")
            mass = float(route.get("mass_tons", 0.0))

            if current != service_start:
                dead = shortest_path_cached(current, service_start)
                if dead is not None and len(dead[0]) >= 2:
                    segments.append(
                        {
                            "agent_id": agent_id,
                            "segment_type": "depot_to_source" if idx == 0 else "between_tasks",
                            "route_id": route_id,
                            "task_id": task_id,
                            "from_node": current,
                            "to_node": service_start,
                            "load_tons": 0.0,
                            "distance_km": float(dead[1]),
                            "path": dead[0],
                        }
                    )

            service_dist = 0.0
            for i in range(len(service_path) - 1):
                u = service_path[i]
                v = service_path[i + 1]
                if graph.has_edge(u, v):
                    service_dist += float(graph[u][v]["distance_km"])

            segments.append(
                {
                    "agent_id": agent_id,
                    "segment_type": "loaded_to_object",
                    "route_id": route_id,
                    "task_id": task_id,
                    "from_node": service_start,
                    "to_node": service_end,
                    "load_tons": mass,
                    "distance_km": service_dist,
                    "path": service_path,
                }
            )
            current = service_end

        if current != depot:
            back = shortest_path_cached(current, depot)
            if back is not None and len(back[0]) >= 2:
                segments.append(
                    {
                        "agent_id": agent_id,
                        "segment_type": "return_to_depot",
                        "route_id": "",
                        "task_id": "",
                        "from_node": current,
                        "to_node": depot,
                        "load_tons": 0.0,
                        "distance_km": float(back[1]),
                        "path": back[0],
                    }
                )

    return segments, node_xy, graph


def render_selected_agents_cycles_static(
    *,
    dataset_path: Path,
    output_root: Path,
    day_index: int = 1,
    selected_agent_indices: list[int] | None = None,
    selected_agent_ids: list[str] | None = None,
    max_edges_to_draw: int = 12000,
) -> dict[str, Any]:
    day_dir = output_root / f"day_{day_index:03d}"
    solution_plan = json.loads((day_dir / "solution_plan.json").read_text(encoding="utf-8"))
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    routes_by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for route in solution_plan["routes"]:
        routes_by_agent[route["agent_id"]].append(route)

    selected_agents, active_agents_df = _resolve_selected_agents(
        routes_by_agent=routes_by_agent,
        selected_agent_indices=selected_agent_indices,
        selected_agent_ids=selected_agent_ids,
    )

    segments, node_xy, graph = _build_segments(dataset, solution_plan, selected_agents)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_title(
        f"Day {day_index} selected agent cycles | agents={len(selected_agents)}"
    )

    # base road graph
    edges = list(dataset["graph"]["edges"])
    step = max(1, len(edges) // max_edges_to_draw)
    for i, edge in enumerate(edges):
        if i % step != 0:
            continue
        s = node_xy[edge["source_id"]]
        t = node_xy[edge["target_id"]]
        ax.plot([s[0], t[0]], [s[1], t[1]], color="#d4d8dc", linewidth=0.35, alpha=0.5, zorder=1)

    # draw segments in old style
    cmap = plt.get_cmap("tab20")
    agent_color = {aid: cmap(i % 20) for i, aid in enumerate(selected_agents)}

    def annotate_load(xs: list[float], ys: list[float], text: str, color: Any, zorder: int) -> None:
        if len(xs) < 2:
            return
        mid = len(xs) // 2
        i0 = max(0, mid - 1)
        i1 = min(len(xs) - 1, mid + 1)
        dx = xs[i1] - xs[i0]
        dy = ys[i1] - ys[i0]
        norm = math.hypot(dx, dy)
        nx0, ny0 = (0.0, 0.0) if norm == 0 else (-dy / norm, dx / norm)
        px = xs[mid] + nx0 * 0.0009
        py = ys[mid] + ny0 * 0.0009
        ax.text(
            px,
            py,
            text,
            fontsize=6.5,
            color=color,
            zorder=zorder,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
        )

    for seg in segments:
        aid = seg["agent_id"]
        color = agent_color[aid]
        xs = [node_xy[nid][0] for nid in seg["path"]]
        ys = [node_xy[nid][1] for nid in seg["path"]]

        if seg["segment_type"] == "loaded_to_object":
            ax.plot(xs, ys, color=color, linewidth=2.2, alpha=0.92, zorder=4)
            annotate_load(xs, ys, f"{seg['load_tons']:.1f}t", color, zorder=7)
        else:
            ax.plot(xs, ys, color=color, linewidth=1.35, alpha=0.5, zorder=2)
            annotate_load(xs, ys, "0t", "#555", zorder=6)

    nodes = dataset["graph"]["nodes"]
    mno = [n for n in nodes if n["kind"] == "mno"]
    obj = [n for n in nodes if n["kind"] == "object1"]
    depots = [n for n in nodes if n["kind"] == "depot"]

    if mno:
        ax.scatter([n["x"] for n in mno], [n["y"] for n in mno], c="#2b7bba", s=16, alpha=0.65, label="MNO", zorder=3)
    if obj:
        ax.scatter([n["x"] for n in obj], [n["y"] for n in obj], c="#2ca02c", s=90, marker="^", edgecolors="#111", linewidths=0.5, label="Object1", zorder=5)
    if depots:
        ax.scatter([n["x"] for n in depots], [n["y"] for n in depots], c="#111", s=80, marker="X", label="Depot", zorder=6)

    ax.grid(alpha=0.12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.show()

    segments_df = pd.DataFrame(
        [
            {
                "agent_id": seg["agent_id"],
                "segment_type": seg["segment_type"],
                "route_id": seg["route_id"],
                "task_id": seg["task_id"],
                "from_node": seg["from_node"],
                "to_node": seg["to_node"],
                "current_load_tons": round(seg["load_tons"], 3),
                "distance_km": round(seg["distance_km"], 3),
                "path_nodes": len(seg["path"]),
            }
            for seg in segments
        ]
    )

    return {
        "segments_df": segments_df,
        "active_agents_df": active_agents_df,
        "selected_agent_ids": selected_agents,
        "day_dir": day_dir,
    }


def render_day_agent_cycles(
    *,
    dataset_path: Path,
    output_root: Path,
    day_index: int = 1,
    selected_agent_indices: list[int] | None = None,
    selected_agent_ids: list[str] | None = None,
    open_browser: bool = True,
) -> dict[str, Any]:
    day_dir = output_root / f"day_{day_index:03d}"
    report = json.loads((day_dir / "feasibility_report.json").read_text(encoding="utf-8"))
    solution_plan = json.loads((day_dir / "solution_plan.json").read_text(encoding="utf-8"))
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    routes_by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for route in solution_plan["routes"]:
        routes_by_agent[route["agent_id"]].append(route)

    selected_agents, active_agents_df = _resolve_selected_agents(
        routes_by_agent=routes_by_agent,
        selected_agent_indices=selected_agent_indices,
        selected_agent_ids=selected_agent_ids,
    )

    segments, node_xy, _ = _build_segments(dataset, solution_plan, selected_agents)

    points_by_kind = {"mno": {"x": [], "y": []}, "object1": {"x": [], "y": []}, "depot": {"x": [], "y": []}}
    for _, (x, y, kind) in node_xy.items():
        if kind in points_by_kind:
            points_by_kind[kind]["x"].append(x)
            points_by_kind[kind]["y"].append(y)

    segment_style = {
        "depot_to_source": {"name": "Depot -> first source (0t)", "color": "#94a3b8", "width": 1.8},
        "between_tasks": {"name": "Between tasks (0t)", "color": "#64748b", "width": 1.5},
        "loaded_to_object": {"name": "Loaded -> object", "color": "#2563eb", "width": 2.6},
        "return_to_depot": {"name": "Return to depot (0t)", "color": "#111827", "width": 2.0},
    }

    fig = go.Figure()
    for stype, cfg in segment_style.items():
        line_x: list[float | None] = []
        line_y: list[float | None] = []
        hover_x: list[float] = []
        hover_y: list[float] = []
        hover_text: list[str] = []

        for seg in segments:
            if seg["segment_type"] != stype:
                continue
            path = seg["path"]
            if len(path) < 2:
                continue

            for nid in path:
                x, y, _ = node_xy[nid]
                line_x.append(x)
                line_y.append(y)
            line_x.append(None)
            line_y.append(None)

            mid = path[len(path) // 2]
            mx, my, _ = node_xy[mid]
            hover_x.append(mx)
            hover_y.append(my)
            hover_text.append(
                "<br>".join(
                    [
                        f"agent_id={seg['agent_id']}",
                        f"segment={seg['segment_type']}",
                        f"route_id={seg['route_id']}",
                        f"task_id={seg['task_id']}",
                        f"from={seg['from_node']} -> to={seg['to_node']}",
                        f"current_load_tons={seg['load_tons']:.3f}",
                        f"distance_km={seg['distance_km']:.3f}",
                    ]
                )
            )

        if line_x:
            fig.add_trace(
                go.Scattergl(
                    x=line_x,
                    y=line_y,
                    mode="lines",
                    line=dict(color=cfg["color"], width=cfg["width"]),
                    name=cfg["name"],
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scattergl(
                    x=hover_x,
                    y=hover_y,
                    mode="markers",
                    marker=dict(size=8, color=cfg["color"], opacity=0.06),
                    text=hover_text,
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=False,
                )
            )

    point_styles = {
        "mno": {"name": "MNO points", "color": "#16a34a", "size": 6},
        "object1": {"name": "Objects", "color": "#7c3aed", "size": 10},
        "depot": {"name": "Depots", "color": "#dc2626", "size": 11},
    }
    for kind, cfg in point_styles.items():
        pts = points_by_kind[kind]
        if pts["x"]:
            fig.add_trace(
                go.Scattergl(
                    x=pts["x"],
                    y=pts["y"],
                    mode="markers",
                    name=cfg["name"],
                    marker=dict(size=cfg["size"], color=cfg["color"], opacity=0.9),
                    hovertemplate=f"kind={kind}<extra></extra>",
                )
            )

    fig.update_layout(
        title=(
            f"Day {day_index} full agent trajectories (depot -> tasks -> depot) | "
            f"agents={len(selected_agents)}, segments={len(segments)}"
        ),
        width=1280,
        height=920,
        template="plotly_white",
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.0),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1)

    html_path = day_dir / f"day_{day_index:03d}_interactive_map.html"
    fig.write_html(html_path, include_plotlyjs="cdn", auto_open=open_browser)
    if open_browser:
        fig.show(renderer="browser")

    segments_df = pd.DataFrame(
        [
            {
                "agent_id": seg["agent_id"],
                "segment_type": seg["segment_type"],
                "route_id": seg["route_id"],
                "task_id": seg["task_id"],
                "from_node": seg["from_node"],
                "to_node": seg["to_node"],
                "current_load_tons": round(seg["load_tons"], 3),
                "distance_km": round(seg["distance_km"], 3),
                "path_nodes": len(seg["path"]),
            }
            for seg in segments
        ]
    )

    return {
        "report": report,
        "segments_df": segments_df,
        "html_path": html_path,
        "day_dir": day_dir,
        "active_agents_df": active_agents_df,
        "selected_agent_ids": selected_agents,
    }
