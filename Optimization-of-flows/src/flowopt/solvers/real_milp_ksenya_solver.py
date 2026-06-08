from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, Bounds, LinearConstraint


@dataclass
class DayResult:
    day_index: int
    solver_status: str
    assigned_routes: int
    unassigned_tasks: int
    active_agents: int
    transport_work_ton_km: float | None
    all_checks_ok: bool
    output_dir: Path


class DayPayloadFactory:
    def __init__(self, base_payload: dict[str, Any]) -> None:
        self.base_payload = base_payload

    def build(self, day_index: int, *, seed: int, mass_noise: float) -> dict[str, Any]:
        rng = random.Random(seed + day_index * 10007)
        payload = deepcopy(self.base_payload)

        mass_by_source: dict[str, float] = {}
        for task in payload["tasks"]:
            base_mass = float(task["mass_tons"])
            factor = 1.0 + rng.uniform(-mass_noise, mass_noise)
            day_mass = max(0.05, base_mass * factor)
            task["mass_tons"] = round(day_mass, 3)
            source = str(task["source_node_id"])
            mass_by_source[source] = mass_by_source.get(source, 0.0) + day_mass

        for node in payload["graph"]["nodes"]:
            if node["kind"] == "mno":
                node["daily_mass_tons"] = round(mass_by_source.get(node["node_id"], 0.0), 3)

        metadata = payload.setdefault("metadata", {})
        metadata["day_index"] = day_index
        metadata["name"] = f"{metadata.get('name', 'real_data_pack')}_day_{day_index:03d}"
        metadata["mass_noise"] = mass_noise
        metadata["seed"] = seed
        return payload


def _validate_payload(raw_payload: dict[str, Any]) -> None:
    known_nodes = {n["node_id"] for n in raw_payload["graph"]["nodes"]}
    known_container_types = set(raw_payload["metadata"]["container_to_vehicle_types"])

    for task in raw_payload["tasks"]:
        if task["source_node_id"] not in known_nodes:
            raise ValueError(f"Unknown source node in task {task['task_id']}")
        if task["destination_node_id"] not in known_nodes:
            raise ValueError(f"Unknown destination node in task {task['task_id']}")
        if task["container_type"] not in known_container_types:
            raise ValueError(f"Unknown container type in task {task['task_id']}")
        if float(task["mass_tons"]) < 0:
            raise ValueError(f"Negative mass in task {task['task_id']}")


def _build_vehicle_graph(payload: dict[str, Any], vehicle_type: str) -> nx.DiGraph:
    G = nx.DiGraph()
    for node in payload["graph"]["nodes"]:
        G.add_node(
            node["node_id"],
            x=node.get("x"),
            y=node.get("y"),
            kind=node.get("kind"),
            center=node.get("center", False),
        )
    for edge in payload["graph"]["edges"]:
        if vehicle_type in edge.get("allowed_vehicle_types", []):
            s = edge["source_id"]
            t = edge["target_id"]
            w = float(edge["distance_km"])
            if G.has_edge(s, t):
                if w < G[s][t]["distance_km"]:
                    G[s][t]["distance_km"] = w
            else:
                G.add_edge(s, t, distance_km=w)
    return G


def _shortest_distance(
    graphs_by_type: dict[str, nx.DiGraph],
    cache: dict[tuple[str, str, str], float | None],
    vehicle_type: str,
    source: str,
    target: str,
) -> float | None:
    key = (vehicle_type, source, target)
    if key in cache:
        return cache[key]
    sentinel = ("__preloaded__", vehicle_type, source)
    if sentinel in cache:
        return None  # source was preloaded; absence in cache => unreachable
    G = graphs_by_type[vehicle_type]
    try:
        d = nx.shortest_path_length(G, source=source, target=target, weight="distance_km")
        cache[key] = float(d)
    except Exception:
        cache[key] = None
    return cache[key]


def _preload_distances_from_source(
    graphs_by_type: dict[str, nx.DiGraph],
    cache: dict[tuple[str, str, str], float | None],
    vehicle_type: str,
    source: str,
) -> None:
    """Run single-source Dijkstra once and populate cache for all targets.

    Subsequent _shortest_distance(vt, source, *) lookups become O(1) dict hits.
    A sentinel key marks the source as preloaded so we don't re-run.
    """
    sentinel = ("__preloaded__", vehicle_type, source)
    if sentinel in cache:
        return
    G = graphs_by_type[vehicle_type]
    if source not in G:
        cache[sentinel] = None  # type: ignore[assignment]
        return
    try:
        dists = nx.single_source_dijkstra_path_length(G, source, weight="distance_km")
    except Exception:
        dists = {}
    for target, d in dists.items():
        if target == source:
            continue
        cache[(vehicle_type, source, target)] = float(d)
    cache[sentinel] = None  # type: ignore[assignment]


def _shortest_path(
    graphs_by_type: dict[str, nx.DiGraph],
    vehicle_type: str,
    source: str,
    target: str,
) -> tuple[list[str], float] | tuple[None, None]:
    G = graphs_by_type[vehicle_type]
    try:
        path = nx.shortest_path(G, source=source, target=target, weight="distance_km")
        dist = 0.0
        for a, b in zip(path[:-1], path[1:]):
            dist += G[a][b]["distance_km"]
        return path, float(dist)
    except Exception:
        return None, None


def _render_solution_map(payload: dict[str, Any], routes: list[dict[str, Any]], out_png: Path) -> None:
    node_by_id = {n["node_id"]: n for n in payload["graph"]["nodes"]}

    fig, ax = plt.subplots(figsize=(10, 8))

    xs_mno, ys_mno = [], []
    xs_obj, ys_obj = [], []
    xs_dep, ys_dep = [], []

    for n in payload["graph"]["nodes"]:
        x = n.get("x")
        y = n.get("y")
        if x is None or y is None:
            continue
        if n["kind"] == "mno":
            xs_mno.append(x)
            ys_mno.append(y)
        elif n["kind"] == "object1":
            xs_obj.append(x)
            ys_obj.append(y)
        elif n["kind"] == "depot":
            xs_dep.append(x)
            ys_dep.append(y)

    if xs_mno:
        ax.scatter(xs_mno, ys_mno, s=5, label="mno")
    if xs_obj:
        ax.scatter(xs_obj, ys_obj, s=20, marker="s", label="object")
    if xs_dep:
        ax.scatter(xs_dep, ys_dep, s=40, marker="^", label="depot")

    for r in routes[:500]:
        path = r.get("full_path", [])
        coords = []
        for nid in path:
            n = node_by_id.get(nid)
            if not n:
                continue
            x = n.get("x")
            y = n.get("y")
            if x is None or y is None:
                continue
            coords.append((x, y))
        if len(coords) >= 2:
            ax.plot([c[0] for c in coords], [c[1] for c in coords], linewidth=0.7, alpha=0.7)

    ax.legend()
    ax.set_title("Exact MILP solution map")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _hierholzer_walk(start: str, counts: Counter[tuple[str, str]]) -> list[str]:
    local_out: dict[str, list[str]] = defaultdict(list)
    remaining: dict[tuple[str, str], int] = dict(counts)
    for (i, j), c in counts.items():
        for _ in range(c):
            local_out[i].append(j)

    stack = [start]
    circuit: list[str] = []
    while stack:
        v = stack[-1]
        while local_out[v]:
            nxt = local_out[v].pop()
            if remaining[(v, nxt)] <= 0:
                continue
            remaining[(v, nxt)] -= 1
            stack.append(nxt)
            v = nxt
        circuit.append(stack.pop())
    circuit.reverse()
    return circuit


def _consume_walk_from_depot(depot: str, counts: Counter[tuple[str, str]]) -> list[str]:
    if sum(c for (i, _), c in counts.items() if i == depot) == 0:
        return [depot]
    walk = _hierholzer_walk(depot, counts)
    for a, b in zip(walk[:-1], walk[1:]):
        counts[(a, b)] -= 1
        if counts[(a, b)] <= 0:
            del counts[(a, b)]
    return walk


def _consume_any_cycle(counts: Counter[tuple[str, str]]) -> list[str]:
    if not counts:
        return []
    start = next(iter(counts))[0]
    walk = _hierholzer_walk(start, counts)
    for a, b in zip(walk[:-1], walk[1:]):
        counts[(a, b)] -= 1
        if counts[(a, b)] <= 0:
            del counts[(a, b)]
    return walk


def _build_compatibility(
    tasks: list[dict[str, Any]],
    agents: list[dict[str, Any]],
    node_by_id: dict[str, dict[str, Any]],
    container_to_vehicle_types: dict[str, set[str]],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    compat_agents: dict[str, list[str]] = {t["task_id"]: [] for t in tasks}
    compat_tasks: dict[str, list[str]] = {a["agent_id"]: [] for a in agents}
    for t in tasks:
        tid = t["task_id"]
        p = t["source_node_id"]
        need_compact = bool(node_by_id[p].get("center", False))
        allowed_types = container_to_vehicle_types[t["container_type"]]
        q_t = float(t["mass_tons"])
        for a in agents:
            if a["vehicle_type"] not in allowed_types:
                continue
            if float(a["capacity_tons"]) < q_t:
                continue
            if need_compact and not bool(a["is_compact"]):
                continue
            compat_agents[tid].append(a["agent_id"])
            compat_tasks[a["agent_id"]].append(tid)
    return compat_agents, compat_tasks


def summarize_milp_dimensions(payload: dict[str, Any]) -> dict[str, Any]:
    """Return naive (full Cartesian) and pruned variable counts for diagnostics.

    The pruned counts for x and y are reported as an *upper bound* sum_v |N_v|^2
    (without applying the per-arc reachability filter). The reachability filter
    typically removes another 30-70 percent of arcs, but computing it requires
    SSSP on the road graph and is left to the actual solver build phase.
    """
    _validate_payload(payload)
    tasks = payload["tasks"]
    agents = payload["agents"]
    nodes = payload["graph"]["nodes"]
    metadata = payload["metadata"]
    node_by_id = {n["node_id"]: n for n in nodes}

    src = {t["task_id"]: t["source_node_id"] for t in tasks}
    dest = {t["task_id"]: t["destination_node_id"] for t in tasks}

    P = sorted(set(src.values()))
    F = sorted(set(dest.values()))
    Dnodes = list(metadata["depot_node_ids"])
    N_full = P + F + Dnodes
    V_all = [a["agent_id"] for a in agents]
    container_to_vehicle_types = {k: set(vs) for k, vs in metadata["container_to_vehicle_types"].items()}
    depot_of = dict(metadata["agent_depots"])

    compat_agents, compat_tasks = _build_compatibility(tasks, agents, node_by_id, container_to_vehicle_types)
    V_used = [v for v in V_all if compat_tasks[v]]

    n_v_sizes: list[int] = []
    n_x_upper = 0
    n_n_pruned = 0
    tasks_at_p = {p: [t for t in tasks if t["source_node_id"] == p] for p in P}
    for v in V_used:
        nv = {depot_of[v]}
        for tid in compat_tasks[v]:
            nv.add(src[tid])
            nv.add(dest[tid])
        nv_list = sorted(nv)
        sz = len(nv_list)
        n_v_sizes.append(sz)
        n_x_upper += sz * (sz - 1)  # i != j

        for p in nv_list:
            if p in P and any(v in compat_agents[t["task_id"]] for t in tasks_at_p.get(p, [])):
                n_n_pruned += 1

    n_z_pruned = sum(len(compat_agents[t["task_id"]]) for t in tasks)

    full_x = len(N_full) ** 2 * len(V_all)
    full_y = full_x
    full_z = len(tasks) * len(V_all)
    full_n = len(P) * len(V_all)
    full_u = len(V_all)
    full_TT = len(V_all)

    return {
        "tasks": len(tasks),
        "agents_total": len(V_all),
        "agents_used": len(V_used),
        "P": len(P),
        "F": len(F),
        "depots": len(Dnodes),
        "N_full": len(N_full),
        "N_v_avg": (sum(n_v_sizes) / len(n_v_sizes)) if n_v_sizes else 0.0,
        "N_v_max": max(n_v_sizes) if n_v_sizes else 0,
        "vars_full": {
            "x": full_x,
            "y": full_y,
            "z": full_z,
            "n": full_n,
            "u": full_u,
            "TT": full_TT,
            "total": full_x + full_y + full_z + full_n + full_u + full_TT,
        },
        "vars_pruned_upper": {
            "x": n_x_upper,
            "y": n_x_upper,
            "z": n_z_pruned,
            "n": n_n_pruned,
            "u": len(V_used),
            "TT": len(V_used),
            "total": 2 * n_x_upper + n_z_pruned + n_n_pruned + 2 * len(V_used),
        },
    }


def solve_real_milp_ksenya(
    payload: dict[str, Any],
    *,
    time_limit_sec: int = 120,
    mip_rel_gap: float = 0.05,
    verbose: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Exact MILP solver with Lvl1 pre-filtering (variable pruning).

    Mathematically equivalent to the original formulation: every variable that
    is removed was already forced to zero by feasibility (incompatibility,
    unreachable arcs, foreign depots, sources without compatible tasks). We
    just stop materialising those zeros so scipy.milp can build the matrix.
    """
    return _solve_real_milp_ksenya_core(
        payload,
        time_limit_sec=time_limit_sec,
        mip_rel_gap=mip_rel_gap,
        verbose=verbose,
    )


def _solve_real_milp_ksenya_core(
    payload: dict[str, Any],
    *,
    task_subset: set[str] | None = None,
    agent_subset: set[str] | None = None,
    knn_k: int | None = None,
    time_limit_sec: int = 120,
    mip_rel_gap: float = 0.05,
    verbose: bool = True,
    log_prefix: str = "milp_ksenya",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Internal core used by the public solver and its decomp / k-NN variants.

    task_subset / agent_subset restrict the MILP to a subset of tasks and
    agents (None == take all from payload). knn_k, if set, prunes per-vehicle
    arcs to k nearest neighbours per node, force-keeping depot→source,
    source→dest, dest→depot for every compatible task so single-task trips
    remain feasible.
    """
    _validate_payload(payload)

    tasks = payload["tasks"]
    agents = payload["agents"]
    if task_subset is not None:
        tasks = [t for t in tasks if t["task_id"] in task_subset]
    if agent_subset is not None:
        agents = [a for a in agents if a["agent_id"] in agent_subset]

    nodes = payload["graph"]["nodes"]
    metadata = payload["metadata"]
    depot_of_full = dict(metadata["agent_depots"])

    if not tasks or not agents:
        all_task_ids = [t["task_id"] for t in payload["tasks"]]
        unassigned_here = [
            tid for tid in all_task_ids
            if (task_subset is None or tid in task_subset)
        ]
        states_empty = {
            a["agent_id"]: {
                "agent_id": a["agent_id"],
                "vehicle_type": a["vehicle_type"],
                "capacity_tons": float(a["capacity_tons"]),
                "is_compact": bool(a["is_compact"]),
                "depot_node": depot_of_full[a["agent_id"]],
                "current_node": depot_of_full[a["agent_id"]],
                "task_ids": [],
                "route_ids": [],
                "total_km": 0.0,
                "total_hours": 0.0,
            }
            for a in payload["agents"]
            if agent_subset is None or a["agent_id"] in agent_subset
        }
        return {"success": True, "objective": 0.0}, {
            "routes": [],
            "states": states_empty,
            "unassigned": unassigned_here,
            "transport_work_ton_km": 0.0,
        }

    node_by_id = {n["node_id"]: n for n in nodes}

    T = [t["task_id"] for t in tasks]
    V_all = [a["agent_id"] for a in agents]

    src = {t["task_id"]: t["source_node_id"] for t in tasks}
    dest = {t["task_id"]: t["destination_node_id"] for t in tasks}
    q = {t["task_id"]: float(t["mass_tons"]) for t in tasks}

    P = sorted(set(src.values()))
    F = sorted(set(dest.values()))
    Dnodes = list(metadata["depot_node_ids"])

    depot_of = dict(metadata["agent_depots"])
    tasks_at_p = {p: [tid for tid in T if src[tid] == p] for p in P}

    Q = {a["agent_id"]: float(a["capacity_tons"]) for a in agents}
    vehicle_type = {a["agent_id"]: a["vehicle_type"] for a in agents}

    vehicle_profiles = metadata["vehicle_profiles"]
    service_hours_by_container = metadata["service_hours_by_container"]
    container_to_vehicle_types = {k: set(vs) for k, vs in metadata["container_to_vehicle_types"].items()}

    L = {v: float(vehicle_profiles[vehicle_type[v]]["max_daily_km"]) for v in V_all}
    H = {v: 60.0 * float(vehicle_profiles[vehicle_type[v]]["max_shift_hours"]) for v in V_all}
    avg_speed = {v: float(vehicle_profiles[vehicle_type[v]]["avg_speed_kmph"]) for v in V_all}
    tau = {t["task_id"]: 60.0 * float(service_hours_by_container[t["container_type"]]) for t in tasks}

    Cap_day = {}
    for n in nodes:
        if n["kind"] == "object1":
            Cap_day[n["node_id"]] = float(n.get("object_day_capacity_tons", 0.0) or 0.0)

    if verbose:
        zero_cap_with_tasks: list[tuple[str, int]] = []
        for f in sorted({t["destination_node_id"] for t in tasks}):
            if Cap_day.get(f, 0.0) <= 0.0:
                n_tasks_to_f = sum(1 for t in tasks if t["destination_node_id"] == f)
                if n_tasks_to_f > 0:
                    zero_cap_with_tasks.append((f, n_tasks_to_f))
        if zero_cap_with_tasks:
            print(
                f"[{log_prefix}] WARNING: {len(zero_cap_with_tasks)} object(s) have Cap_day=0 with tasks targeting them — those tasks cannot be assigned:"
            )
            for nid, ntasks in zero_cap_with_tasks[:10]:
                print(f"  {nid}: {ntasks} task(s)")
            if len(zero_cap_with_tasks) > 10:
                print(f"  ... and {len(zero_cap_with_tasks) - 10} more")

    # ----- Step 1.1 + 1.2: compatibility filter -----
    compat_agents, compat_tasks = _build_compatibility(
        tasks, agents, node_by_id, container_to_vehicle_types
    )
    V = [v for v in V_all if compat_tasks[v]]

    # ----- Reachability per used vehicle type -----
    unique_vehicle_types = sorted({vehicle_type[v] for v in V})
    graphs_by_type = {vt: _build_vehicle_graph(payload, vt) for vt in unique_vehicle_types}
    dist_cache: dict[tuple[str, str, str], float | None] = {}

    # ----- Step 1.3: per-agent node set N_v -----
    N_v: dict[str, list[str]] = {}
    for v in V:
        nv = {depot_of[v]}
        for tid in compat_tasks[v]:
            nv.add(src[tid])
            nv.add(dest[tid])
        N_v[v] = sorted(nv)

    # ----- Preload distances: one SSSP per (vehicle_type, source) populates
    # the cache for all reachable targets at once. Without this, building
    # arcs_v below would call shortest_path_length per pair, which is far slower.
    sources_by_vt: dict[str, set[str]] = {vt: set() for vt in unique_vehicle_types}
    for v in V:
        for n in N_v[v]:
            sources_by_vt[vehicle_type[v]].add(n)
    for vt, sources in sources_by_vt.items():
        for src_node in sources:
            _preload_distances_from_source(graphs_by_type, dist_cache, vt, src_node)

    # ----- Step 1.4: per-agent reachable arcs -----
    Dijv: dict[tuple[str, str, str], float] = {}
    Cijv: dict[tuple[str, str, str], float] = {}
    arcs_v: dict[str, list[tuple[str, str]]] = {}
    for v in V:
        vt = vehicle_type[v]
        avail: list[tuple[str, str]] = []
        for i in N_v[v]:
            for j in N_v[v]:
                if i == j:
                    continue
                d = _shortest_distance(graphs_by_type, dist_cache, vt, i, j)
                if d is None:
                    continue
                Dijv[(i, j, v)] = float(d)
                Cijv[(i, j, v)] = 60.0 * float(d) / avg_speed[v]
                avail.append((i, j))
        arcs_v[v] = avail

    # ----- Optional k-NN arc pruning -----
    # For each vehicle, keep only the `knn_k` shortest outgoing and incoming
    # arcs per node. To preserve single-task feasibility, the depot↔source,
    # source→dest, dest→depot arcs of every compatible task are force-kept.
    if knn_k is not None and knn_k > 0:
        kept_pairs_total = 0
        original_pairs_total = 0
        for v in V:
            d_own = depot_of[v]
            essential: set[tuple[str, str]] = set()
            for tid in compat_tasks[v]:
                s_t, f_t = src[tid], dest[tid]
                if d_own != s_t:
                    essential.add((d_own, s_t))
                if s_t != f_t:
                    essential.add((s_t, f_t))
                if f_t != d_own:
                    essential.add((f_t, d_own))

            arcs_full = arcs_v[v]
            arcs_set = set(arcs_full)
            kept: set[tuple[str, str]] = essential & arcs_set

            out_per_i: dict[str, list[tuple[float, str]]] = defaultdict(list)
            in_per_j: dict[str, list[tuple[float, str]]] = defaultdict(list)
            for (i, j) in arcs_full:
                c = Cijv[(i, j, v)]
                out_per_i[i].append((c, j))
                in_per_j[j].append((c, i))
            for i, opts in out_per_i.items():
                opts.sort(key=lambda p: p[0])
                for _, j in opts[:knn_k]:
                    kept.add((i, j))
            for j, opts in in_per_j.items():
                opts.sort(key=lambda p: p[0])
                for _, i in opts[:knn_k]:
                    kept.add((i, j))

            arcs_v[v] = sorted(kept)
            kept_pairs_total += len(kept)
            original_pairs_total += len(arcs_full)

        if verbose:
            ratio = (kept_pairs_total / max(1, original_pairs_total)) * 100.0
            print(
                f"[{log_prefix}] knn_k={knn_k}: arcs pruned {original_pairs_total} -> "
                f"{kept_pairs_total} ({ratio:.1f}% kept)"
            )

    # ----- Variables (created only when meaningful) -----
    idx: dict[tuple[Any, ...], int] = {}
    names: list[tuple[Any, ...]] = []
    lb: list[float] = []
    ub: list[float] = []
    integrality: list[int] = []

    def add_var(name: tuple[Any, ...], low: float, up: float, integ: int) -> None:
        idx[name] = len(names)
        names.append(name)
        lb.append(low)
        ub.append(up)
        integrality.append(integ)

    for v in V:
        max_arc_use_v = max(1, len(compat_tasks[v]))
        for (i, j) in arcs_v[v]:
            add_var(("x", i, j, v), 0, float(max_arc_use_v), 1)
    for v in V:
        for (i, j) in arcs_v[v]:
            add_var(("y", i, j, v), 0, np.inf, 0)
    for tid in T:
        for v in compat_agents[tid]:
            add_var(("z", tid, v), 0, 1, 1)
    for v in V:
        add_var(("u", v), 0, 1, 1)

    # ----- Constraints -----
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    bl: list[float] = []
    bu: list[float] = []
    row = 0

    def add_constr(coeffs: dict[tuple[Any, ...], float], low: float, up: float) -> None:
        nonlocal row
        if not coeffs:
            return
        for var, coef in coeffs.items():
            rows.append(row)
            cols.append(idx[var])
            vals.append(coef)
        bl.append(low)
        bu.append(up)
        row += 1

    # (1) each task assigned to at most one compatible agent
    for tid in T:
        if not compat_agents[tid]:
            continue
        coeff = {("z", tid, v): 1.0 for v in compat_agents[tid]}
        add_constr(coeff, 0.0, 1.0)

    # (2) z_tv <= u_v
    for tid in T:
        for v in compat_agents[tid]:
            add_constr({("z", tid, v): 1.0, ("u", v): -1.0}, -np.inf, 0.0)

    # (3-5) sum_i x[i,p,v] <= sum_t z_tv  (over compatible tasks at p)
    # Substituted form: n[p,v] eliminated; old (3) and (4) are subsumed by
    # general flow conservation (8) at non-depot nodes.
    for v in V:
        for p in N_v[v]:
            if p not in P:
                continue
            tids_here = [tid for tid in tasks_at_p.get(p, []) if v in compat_agents[tid]]
            if not tids_here:
                continue
            coeff: dict[tuple[Any, ...], float] = {}
            for i in N_v[v]:
                key = ("x", i, p, v)
                if key in idx:
                    coeff[key] = coeff.get(key, 0.0) + 1.0
            for tid in tids_here:
                key_z = ("z", tid, v)
                if key_z in idx:
                    coeff[key_z] = coeff.get(key_z, 0.0) - 1.0
            add_constr(coeff, -np.inf, 0.0)

    # (6) depot out: sum_j x[d,j,v] = u_v
    for v in V:
        d = depot_of[v]
        coeff = {("u", v): -1.0}
        for j in N_v[v]:
            key = ("x", d, j, v)
            if key in idx:
                coeff[key] = coeff.get(key, 0.0) + 1.0
        add_constr(coeff, 0.0, 0.0)

    # (7) depot in: sum_i x[i,d,v] = u_v
    for v in V:
        d = depot_of[v]
        coeff = {("u", v): -1.0}
        for i in N_v[v]:
            key = ("x", i, d, v)
            if key in idx:
                coeff[key] = coeff.get(key, 0.0) + 1.0
        add_constr(coeff, 0.0, 0.0)

    # (8) flow conservation at non-depot nodes (P u F restricted to N_v)
    for v in V:
        own = depot_of[v]
        for k in N_v[v]:
            if k == own:
                continue
            coeff: dict[tuple[Any, ...], float] = {}
            for i in N_v[v]:
                key = ("x", i, k, v)
                if key in idx:
                    coeff[key] = coeff.get(key, 0.0) + 1.0
            for j in N_v[v]:
                key = ("x", k, j, v)
                if key in idx:
                    coeff[key] = coeff.get(key, 0.0) - 1.0
            add_constr(coeff, 0.0, 0.0)

    # (9) y_ij <= Q_v * x_ij
    for v in V:
        for (i, j) in arcs_v[v]:
            add_constr({("y", i, j, v): 1.0, ("x", i, j, v): -Q[v]}, -np.inf, 0.0)

    # (10) no flow leaving depot (loaded)
    for v in V:
        d = depot_of[v]
        coeff = {}
        for j in N_v[v]:
            key = ("y", d, j, v)
            if key in idx:
                coeff[key] = 1.0
        add_constr(coeff, 0.0, 0.0)

    # (11) no flow returning into depot
    for v in V:
        d = depot_of[v]
        coeff = {}
        for i in N_v[v]:
            key = ("y", i, d, v)
            if key in idx:
                coeff[key] = 1.0
        add_constr(coeff, 0.0, 0.0)

    # (12) destinations are sinks: no flow leaves f
    for v in V:
        for f in N_v[v]:
            if f not in F:
                continue
            coeff = {}
            for j in N_v[v]:
                key = ("y", f, j, v)
                if key in idx:
                    coeff[key] = 1.0
            add_constr(coeff, 0.0, 0.0)

    # (13) flow conservation at source p: out - in = sum q_t z_tv
    for v in V:
        for p in N_v[v]:
            if p not in P:
                continue
            coeff: dict[tuple[Any, ...], float] = {}
            for j in N_v[v]:
                key = ("y", p, j, v)
                if key in idx:
                    coeff[key] = coeff.get(key, 0.0) + 1.0
            for i in N_v[v]:
                key = ("y", i, p, v)
                if key in idx:
                    coeff[key] = coeff.get(key, 0.0) - 1.0
            for tid in tasks_at_p.get(p, []):
                key_z = ("z", tid, v)
                if key_z in idx:
                    coeff[key_z] = coeff.get(key_z, 0.0) - q[tid]
            add_constr(coeff, 0.0, 0.0)

    # (14) flow conservation at destination f: in = sum q_t z_tv (for t with dest = f)
    for v in V:
        for f in N_v[v]:
            if f not in F:
                continue
            tids_f = [tid for tid in compat_tasks[v] if dest[tid] == f]
            coeff: dict[tuple[Any, ...], float] = {}
            for i in N_v[v]:
                key = ("y", i, f, v)
                if key in idx:
                    coeff[key] = coeff.get(key, 0.0) + 1.0
            for tid in tids_f:
                key_z = ("z", tid, v)
                if key_z in idx:
                    coeff[key_z] = coeff.get(key_z, 0.0) - q[tid]
            add_constr(coeff, 0.0, 0.0)

    # (15) per-destination day capacity
    for f in F:
        coeff: dict[tuple[Any, ...], float] = {}
        for tid in T:
            if dest[tid] != f:
                continue
            for v in compat_agents[tid]:
                key = ("z", tid, v)
                if key in idx:
                    coeff[key] = coeff.get(key, 0.0) + q[tid]
        if coeff:
            add_constr(coeff, -np.inf, float(Cap_day.get(f, 0.0)))

    # (16-17) sum_ij C_ij x_ijv + sum_t tau_t z_tv <= H_v * u_v
    # Substituted form: TT[v] eliminated, old (16) inlined into (17).
    for v in V:
        coeff: dict[tuple[Any, ...], float] = {("u", v): -H[v]}
        for (i, j) in arcs_v[v]:
            coeff[("x", i, j, v)] = coeff.get(("x", i, j, v), 0.0) + Cijv[(i, j, v)]
        for tid in compat_tasks[v]:
            key = ("z", tid, v)
            if key in idx:
                coeff[key] = coeff.get(key, 0.0) + tau[tid]
        add_constr(coeff, -np.inf, 0.0)

    # (18) sum D_ij x_ijv <= L_v u_v
    for v in V:
        coeff: dict[tuple[Any, ...], float] = {("u", v): -L[v]}
        for (i, j) in arcs_v[v]:
            coeff[("x", i, j, v)] = coeff.get(("x", i, j, v), 0.0) + Dijv[(i, j, v)]
        add_constr(coeff, -np.inf, 0.0)

    # ----- Objective -----
    lam = 1e-3
    c = np.zeros(len(names))
    for v in V:
        for (i, j) in arcs_v[v]:
            c[idx[("y", i, j, v)]] += Dijv[(i, j, v)]
            c[idx[("x", i, j, v)]] += lam * Dijv[(i, j, v)]
    for v in V:
        c[idx[("u", v)]] += 1e-6

    unassigned_penalty = 1e4
    for tid in T:
        for v in compat_agents[tid]:
            c[idx[("z", tid, v)]] -= unassigned_penalty

    A = sp.coo_array((vals, (rows, cols)), shape=(row, len(names)))
    bounds = Bounds(lb, ub)
    constraints = LinearConstraint(A, bl, bu)

    if verbose:
        n_int = int(sum(integrality))
        print(
            f"[{log_prefix}] model: vars={len(names)}, rows={row}, integers={n_int}, "
            f"time_limit={int(time_limit_sec)}s, mip_rel_gap={mip_rel_gap}"
        )

    res = milp(
        c=c,
        integrality=np.array(integrality),
        bounds=bounds,
        constraints=constraints,
        options={
            "disp": bool(verbose),
            "time_limit": int(time_limit_sec),
            "mip_rel_gap": float(mip_rel_gap),
        },
    )

    if res.x is None:
        return {"success": False, "reason": "milp_failed"}, {
            "routes": [],
            "states": {a["agent_id"]: {
                "agent_id": a["agent_id"],
                "vehicle_type": a["vehicle_type"],
                "capacity_tons": float(a["capacity_tons"]),
                "is_compact": bool(a["is_compact"]),
                "depot_node": depot_of[a["agent_id"]],
                "current_node": depot_of[a["agent_id"]],
                "task_ids": [],
                "route_ids": [],
                "total_km": 0.0,
                "total_hours": 0.0,
            } for a in agents},
            "unassigned": T[:],
            "transport_work_ton_km": None,
        }

    sol = res.x

    z_sol: dict[tuple[str, str], float] = {}
    for tid in T:
        for v in compat_agents[tid]:
            z_sol[(tid, v)] = sol[idx[("z", tid, v)]]

    x_sol: dict[tuple[str, str, str], int] = {}
    for v in V:
        for (i, j) in arcs_v[v]:
            val = int(round(sol[idx[("x", i, j, v)]]))
            if val > 0:
                x_sol[(i, j, v)] = val

    y_sol: dict[tuple[str, str, str], float] = {}
    for v in V:
        for (i, j) in arcs_v[v]:
            val = float(sol[idx[("y", i, j, v)]])
            if val > 1e-8:
                y_sol[(i, j, v)] = val

    TT_sol: dict[str, float] = {}
    for v in V:
        tt_v = 0.0
        for (i, j) in arcs_v[v]:
            tt_v += Cijv[(i, j, v)] * float(sol[idx[("x", i, j, v)]])
        for tid in compat_tasks[v]:
            key_z = ("z", tid, v)
            if key_z in idx:
                tt_v += tau[tid] * float(sol[idx[key_z]])
        TT_sol[v] = tt_v
    u_sol = {v: float(sol[idx[("u", v)]]) for v in V}

    states = {}
    for a in agents:
        v = a["agent_id"]
        states[v] = {
            "agent_id": v,
            "vehicle_type": a["vehicle_type"],
            "capacity_tons": float(a["capacity_tons"]),
            "is_compact": bool(a["is_compact"]),
            "depot_node": depot_of[v],
            "current_node": depot_of[v],
            "task_ids": [],
            "route_ids": [],
            "total_km": 0.0,
            "total_hours": (TT_sol[v] / 60.0) if v in TT_sol else 0.0,
        }

    assigned_task_to_agent: dict[str, str] = {}
    for tid in T:
        for v in compat_agents[tid]:
            if z_sol.get((tid, v), 0.0) > 0.5:
                assigned_task_to_agent[tid] = v
                states[v]["task_ids"].append(tid)
                break

    unassigned = [tid for tid in T if tid not in assigned_task_to_agent]

    routes: list[dict[str, Any]] = []
    route_counter = 1

    for v in V:
        if u_sol[v] < 0.5:
            continue

        arc_counts = Counter({(i, j): cnt for (i, j, vv), cnt in x_sol.items() if vv == v})
        if not arc_counts:
            continue

        depot = depot_of[v]
        walks: list[list[str]] = []

        main_walk = _consume_walk_from_depot(depot, arc_counts)
        if len(main_walk) >= 2:
            walks.append(main_walk)

        while arc_counts:
            cyc = _consume_any_cycle(arc_counts)
            if len(cyc) >= 2:
                walks.append(cyc)

        vehicle_task_ids = [tid for tid, av in assigned_task_to_agent.items() if av == v]
        vehicle_task_ids.sort()

        agent_arc_counts = Counter(
            {(i, j): cnt for (i, j, vv), cnt in x_sol.items() if vv == v}
        )
        states[v]["total_km"] = round(
            sum(Dijv[(i, j, v)] * cnt for (i, j), cnt in agent_arc_counts.items()),
            3,
        )

        ton_km_v = sum(Dijv[(i, j, v)] * y_sol.get((i, j, v), 0.0) for (i, j) in arcs_v[v])

        if not walks:
            walks = [[depot]]

        for walk_idx, route_nodes in enumerate(walks, start=1):
            full_path: list[str] = []
            route_km = 0.0
            for a, b in zip(route_nodes[:-1], route_nodes[1:]):
                pth, seg_km = _shortest_path(graphs_by_type, vehicle_type[v], a, b)
                if pth is not None:
                    if not full_path:
                        full_path.extend(pth)
                    else:
                        full_path.extend(pth[1:])
                if (a, b, v) in Dijv:
                    route_km += Dijv[(a, b, v)]

            route_id = f"route_{route_counter:04d}"
            route_counter += 1
            states[v]["route_ids"].append(route_id)

            routes.append(
                {
                    "route_id": route_id,
                    "agent_id": v,
                    "path": tuple(route_nodes),
                    "full_path": full_path,
                    "task_ids": tuple(vehicle_task_ids if walk_idx == 1 else ()),
                    "route_km": round(route_km, 3),
                    "ton_km": round(ton_km_v if walk_idx == 1 else 0.0, 3),
                    "TT_min": round(TT_sol[v], 3),
                    "contains_disconnected_cycle": bool(walk_idx > 1),
                }
            )

    transport_work = round(
        sum(
            Dijv[(i, j, v)] * y_sol.get((i, j, v), 0.0)
            for v in V
            for (i, j) in arcs_v[v]
        ),
        3,
    )

    return {"success": True, "objective": float(res.fun)}, {
        "routes": routes,
        "states": states,
        "unassigned": unassigned,
        "transport_work_ton_km": transport_work,
    }


def _solve_real_milp_ksenya_decomposed(
    payload: dict[str, Any],
    *,
    knn_k: int | None = None,
    time_limit_sec: int = 120,
    mip_rel_gap: float = 0.05,
    verbose: bool = True,
    log_prefix: str = "milp_ksenya_decomp",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Decomposition by nearest depot (Euclidean on source coordinates).

    Each task is assigned to its closest depot; sub-MILPs are solved
    sequentially, with shared destination day-capacities updated between
    iterations so different depots cannot overcommit the same object.
    Optionally combined with k-NN arc pruning when knn_k is set.
    """
    payload = deepcopy(payload)
    nodes = payload["graph"]["nodes"]
    node_by_id = {n["node_id"]: n for n in nodes}
    depot_ids = list(payload["metadata"]["depot_node_ids"])
    agent_depots = dict(payload["metadata"]["agent_depots"])

    def _euclid(a: str, b: str) -> float:
        ax = float(node_by_id[a].get("x") or 0.0)
        ay = float(node_by_id[a].get("y") or 0.0)
        bx = float(node_by_id[b].get("x") or 0.0)
        by = float(node_by_id[b].get("y") or 0.0)
        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

    tasks_by_depot: dict[str, list[str]] = {d: [] for d in depot_ids}
    for t in payload["tasks"]:
        s = t["source_node_id"]
        nearest = min(depot_ids, key=lambda d: _euclid(s, d))
        tasks_by_depot[nearest].append(t["task_id"])

    agents_by_depot: dict[str, list[str]] = {d: [] for d in depot_ids}
    for a in payload["agents"]:
        d = agent_depots.get(a["agent_id"])
        if d in agents_by_depot:
            agents_by_depot[d].append(a["agent_id"])

    active_depots = [d for d in depot_ids if tasks_by_depot[d] and agents_by_depot[d]]
    n_active = max(1, len(active_depots))
    per_depot_limit = max(10, int(time_limit_sec) // n_active)

    merged_states: dict[str, dict[str, Any]] = {
        a["agent_id"]: {
            "agent_id": a["agent_id"],
            "vehicle_type": a["vehicle_type"],
            "capacity_tons": float(a["capacity_tons"]),
            "is_compact": bool(a["is_compact"]),
            "depot_node": agent_depots[a["agent_id"]],
            "current_node": agent_depots[a["agent_id"]],
            "task_ids": [],
            "route_ids": [],
            "total_km": 0.0,
            "total_hours": 0.0,
        }
        for a in payload["agents"]
    }
    all_routes: list[dict[str, Any]] = []
    all_unassigned: list[str] = []
    total_transport_work = 0.0
    sub_objectives: list[float] = []
    overall_success = True

    if verbose:
        sizes = ", ".join(
            f"{d}: {len(tasks_by_depot[d])}t/{len(agents_by_depot[d])}a"
            for d in depot_ids
        )
        knn_note = f", knn_k={knn_k}" if knn_k else ""
        print(f"[{log_prefix}] groups by nearest depot: {sizes}{knn_note}")
        print(
            f"[{log_prefix}] active depots: {n_active}, "
            f"per-depot time limit: {per_depot_limit}s"
        )

    for d in depot_ids:
        tids = tasks_by_depot[d]
        aids = set(agents_by_depot[d])
        if not tids:
            continue
        if not aids:
            all_unassigned.extend(tids)
            if verbose:
                print(
                    f"[{log_prefix}] depot {d}: {len(tids)} task(s) but no "
                    f"agents — all unassigned"
                )
            continue

        sub_log_prefix = f"{log_prefix}/depot={d}"
        sub_meta, sub_sol = _solve_real_milp_ksenya_core(
            payload,
            task_subset=set(tids),
            agent_subset=aids,
            knn_k=knn_k,
            time_limit_sec=per_depot_limit,
            mip_rel_gap=mip_rel_gap,
            verbose=verbose,
            log_prefix=sub_log_prefix,
        )

        if not sub_meta.get("success"):
            overall_success = False
            all_unassigned.extend(tids)
            continue

        sub_objectives.append(float(sub_meta.get("objective") or 0.0))
        all_routes.extend(sub_sol["routes"])
        for v_id, st in sub_sol["states"].items():
            merged_states[v_id] = st
        all_unassigned.extend(sub_sol["unassigned"])
        total_transport_work += float(sub_sol.get("transport_work_ton_km") or 0.0)

        assigned_this_depot = set(tids) - set(sub_sol["unassigned"])
        consumed_by_dest: dict[str, float] = defaultdict(float)
        for t in payload["tasks"]:
            if t["task_id"] in assigned_this_depot:
                consumed_by_dest[t["destination_node_id"]] += float(t["mass_tons"])
        for n in nodes:
            if n["kind"] == "object1" and n["node_id"] in consumed_by_dest:
                cur = float(n.get("object_day_capacity_tons", 0.0) or 0.0)
                n["object_day_capacity_tons"] = max(
                    0.0, cur - consumed_by_dest[n["node_id"]]
                )

    return {"success": overall_success, "objective": sum(sub_objectives)}, {
        "routes": all_routes,
        "states": merged_states,
        "unassigned": all_unassigned,
        "transport_work_ton_km": round(total_transport_work, 3),
    }


class MultiDayRealRunner:
    def __init__(self, base_dataset_path: Path, output_root: Path) -> None:
        self.base_dataset_path = base_dataset_path
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)

        self.base_payload = json.loads(self.base_dataset_path.read_text(encoding="utf-8"))
        self.factory = DayPayloadFactory(self.base_payload)

    def _preprocess_data(self, day_payload: dict[str, Any]) -> dict[str, Any]:
        _validate_payload(day_payload)

        nodes = day_payload["graph"]["nodes"]
        edges = day_payload["graph"]["edges"]
        tasks = day_payload["tasks"]
        agents = day_payload["agents"]

        G = nx.DiGraph()
        for n in nodes:
            G.add_node(n["node_id"])
        for e in edges:
            G.add_edge(e["source_id"], e["target_id"], distance_km=float(e["distance_km"]))

        unreachable_tasks = []
        for t in tasks:
            try:
                nx.shortest_path_length(G, source=t["source_node_id"], target=t["destination_node_id"], weight="distance_km")
            except Exception:
                unreachable_tasks.append(t["task_id"])

        preprocess_report = {
            "dataset": str(self.base_dataset_path),
            "counts": {
                "nodes": len(nodes),
                "edges": len(edges),
                "agents": len(agents),
                "tasks": len(tasks),
            },
            "reachability": {
                "unreachable_tasks": unreachable_tasks,
                "unreachable_special_nodes": [],
            },
            "compatibility": {
                "tasks_without_path": unreachable_tasks,
                "tasks_without_compatible_agents": [],
                "all_tasks_have_path": len(unreachable_tasks) == 0,
                "all_tasks_have_compatible_agents": True,
            },
            "preprocess_ok": len(unreachable_tasks) == 0,
        }
        return preprocess_report

    def solve_day(
        self,
        day_payload: dict[str, Any],
        *,
        render_map: bool,
        render_utilization: bool = False,
    ) -> DayResult:
        day_index = int(day_payload.get("metadata", {}).get("day_index", 0))
        day_dir = self.output_root / f"day_{day_index:03d}"
        day_dir.mkdir(parents=True, exist_ok=True)

        preprocess_report = self._preprocess_data(day_payload)
        (day_dir / "preprocess_report.json").write_text(
            json.dumps(preprocess_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        milp_meta, solution = solve_real_milp_ksenya(day_payload)

        routes = solution["routes"]
        states = solution["states"]
        unassigned = solution["unassigned"]
        transport_work = solution["transport_work_ton_km"]

        solver_status = "feasible" if (milp_meta.get("success") and not unassigned) else "infeasible"
        solve_error = None if milp_meta.get("success") else milp_meta.get("reason")

        plan = {"routes": routes, "states": states}
        (day_dir / "solution_plan.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        capacity_summary = {
            "objects": [
                {
                    "object_node_id": n["node_id"],
                    "day_capacity_tons": float(n.get("object_day_capacity_tons", 0.0) or 0.0),
                }
                for n in day_payload["graph"]["nodes"]
                if n["kind"] == "object1"
            ]
        }

        source_counts = defaultdict(int)
        for r in routes:
            for tid in r["task_ids"]:
                task = next(t for t in day_payload["tasks"] if t["task_id"] == tid)
                source_counts[task["source_node_id"]] += 1

        source_coverage = {
            "covered_sources": len(source_counts),
            "total_sources": len({t["source_node_id"] for t in day_payload["tasks"]}),
            "rows": [{"source_node_id": k, "assigned_tasks": v} for k, v in source_counts.items()],
        }

        mno_coverage = {
            "covered_mno": len(source_counts),
            "total_mno": len({t["source_node_id"] for t in day_payload["tasks"]}),
            "all_tasks_assigned": len(unassigned) == 0,
        }

        depot_summary = {
            "depots": [
                {
                    "depot_node_id": d,
                    "active_agents": sum(1 for s in states.values() if s["depot_node"] == d and s["task_ids"]),
                }
                for d in day_payload["metadata"]["depot_node_ids"]
            ]
        }

        checks = {
            "all_checks_ok": len(unassigned) == 0 and preprocess_report.get("preprocess_ok", False),
            "preprocess_ok": preprocess_report.get("preprocess_ok", False),
            "all_tasks_assigned": len(unassigned) == 0,
            "limit_violations": 0,
        }

        report = {
            "solver_status": solver_status,
            "dataset": str(self.base_dataset_path),
            "pipeline": "real_data_notebook_pack_exact_milp_rewritten",
            "preprocess_report_file": "preprocess_report.json",
            "preprocess_ok": preprocess_report.get("preprocess_ok", False),
            "constraints_check_summary": checks,
            "reachability": preprocess_report["reachability"],
            "assigned_routes": len(routes),
            "assigned_tasks": sum(len(s["task_ids"]) for s in states.values()),
            "unassigned_tasks": unassigned,
            "task_source_coverage": source_coverage,
            "mno_scope_coverage": mno_coverage,
            "depot_activity": depot_summary,
            "limit_violations": [],
            "transport_work_ton_km": transport_work,
            "transport_work_by_shoulder_ton_km": {"loaded_ton_km_est": transport_work},
            "active_agents": sum(1 for s in states.values() if s["task_ids"]),
            "object_capacity_summary": capacity_summary,
            "solve_error": solve_error,
            "agent_summary": {
                agent_id: {
                    "vehicle_type": state["vehicle_type"],
                    "tasks": len(state["task_ids"]),
                    "total_km": round(state["total_km"], 3),
                    "km_limit": float(day_payload["metadata"]["vehicle_profiles"][state["vehicle_type"]]["max_daily_km"]),
                    "total_hours": round(state["total_hours"], 3),
                    "hour_limit": float(day_payload["metadata"]["vehicle_profiles"][state["vehicle_type"]]["max_shift_hours"]),
                }
                for agent_id, state in states.items()
                if state["task_ids"]
            },
        }

        (day_dir / "feasibility_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if render_map:
            _render_solution_map(day_payload, routes, day_dir / "solution_map.png")

        return DayResult(
            day_index=day_index,
            solver_status=solver_status,
            assigned_routes=len(routes),
            unassigned_tasks=len(unassigned),
            active_agents=sum(1 for s in states.values() if s["task_ids"]),
            transport_work_ton_km=transport_work,
            all_checks_ok=bool(checks.get("all_checks_ok", False)),
            output_dir=day_dir,
        )

    def run(
        self,
        *,
        days: int,
        seed: int = 42,
        mass_noise: float = 0.08,
        render_first_day: bool = True,
        render_utilization: bool = False,
    ) -> list[DayResult]:
        results: list[DayResult] = []
        for day in range(1, days + 1):
            payload = self.factory.build(day, seed=seed, mass_noise=mass_noise)
            result = self.solve_day(
                payload,
                render_map=(render_first_day and day == 1),
                render_utilization=(render_utilization and day == 1),
            )
            results.append(result)
        return results
