
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import time
from typing import Any, Callable

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import milp, Bounds, LinearConstraint

from .core import (
    AgentState,
    MAX_DAILY_KM_BY_TYPE,
    MAX_SHIFT_HOURS_BY_TYPE,
    AVG_SPEED_KMPH_BY_TYPE,
    SERVICE_HOURS_BY_CONTAINER,
    build_nx_graph,
    check_reachability,
    build_solution_dataset,
    validate_daily_limits,
    constraints_check_summary,
    mno_scope_coverage_summary,
    build_solution_plan,
    render_solution_map,
    render_utilization,
)
from .dataset import Route, dataset_from_dict, CONTAINER_TO_VEHICLE_TYPES


def _task_service_hours(task) -> float:
    return SERVICE_HOURS_BY_CONTAINER.get(task.container_type, 0.25)


def _vehicle_graph_for_type(dataset, vehicle_type: str) -> nx.DiGraph:
    G = nx.DiGraph()
    for node_id, node in dataset.graph.nodes.items():
        G.add_node(node_id, x=node.x, y=node.y, kind=node.kind)
    for edge in dataset.graph.edges:
        if vehicle_type in edge.allowed_vehicle_types:
            if G.has_edge(edge.source_id, edge.target_id):
                curr = G[edge.source_id][edge.target_id]["distance_km"]
                if edge.distance_km < curr:
                    G[edge.source_id][edge.target_id]["distance_km"] = edge.distance_km
            else:
                G.add_edge(edge.source_id, edge.target_id, distance_km=edge.distance_km)
    return G


def _shortest_path_for_type(dataset, vehicle_type: str, source: str, target: str):
    G = _vehicle_graph_for_type(dataset, vehicle_type)
    path = nx.shortest_path(G, source=source, target=target, weight="distance_km")
    dist = 0.0
    for a, b in zip(path[:-1], path[1:]):
        dist += G[a][b]["distance_km"]
    return path, dist


def solve_sandbox_milp(
    payload: dict[str, Any],
    avg_speed_kmph: float = 20.0,
    service_mno_min: float = 10.0,
    service_dump_min: float = 20.0,
    shift_minutes: float = 480.0,
    show_progress: bool = False,
    progress_hook: Callable[[str], None] | None = None,
):
    def _emit(message: str) -> None:
        if progress_hook is not None:
            progress_hook(message)
        elif show_progress:
            print(f"[milp] {message}", flush=True)

    t0 = time.perf_counter()
    data = payload

    nodes = data["graph"]["nodes"]
    edges = data["graph"]["edges"]
    agents = data["agents"]
    tasks = data["tasks"]

    I = [t["source_node_id"] for t in tasks]
    F = sorted(set(t["destination_node_id"] for t in tasks))
    Dnodes = data["metadata"]["depot_node_ids"]
    N = I + F + Dnodes
    V = [a["agent_id"] for a in agents]
    _emit(f"prepare problem: tasks={len(I)}, dumps={len(F)}, agents={len(V)}")

    q = {t["source_node_id"]: float(t["mass_tons"]) for t in tasks}
    dest = {t["source_node_id"]: t["destination_node_id"] for t in tasks}
    Q = {a["agent_id"]: float(a["capacity_tons"]) for a in agents}
    depot_of = data["metadata"]["agent_depots"]
    task_of_source = {t["source_node_id"]: t["task_id"] for t in tasks}

    UG = nx.Graph()
    for n in nodes:
        UG.add_node(n["node_id"])
    for e in edges:
        s, t, w = e["source_id"], e["target_id"], float(e["distance_km"])
        if UG.has_edge(s, t):
            if w < UG[s][t]["distance_km"]:
                UG[s][t]["distance_km"] = w
        else:
            UG.add_edge(s, t, distance_km=w)

    t_dist = time.perf_counter()
    lengths_all = {}
    src_log_every = max(1, len(N) // 10) if N else 1
    for idx_src, src in enumerate(N):
        lengths_all[src] = nx.single_source_dijkstra_path_length(UG, src, weight="distance_km")
        if (idx_src + 1) % src_log_every == 0 or idx_src + 1 == len(N):
            _emit(f"all-pairs shortest paths: {idx_src + 1}/{len(N)} sources")
    _emit(f"distance matrix built in {time.perf_counter() - t_dist:.1f}s")
    Dij = {(i, j): (0.0 if i == j else float(lengths_all[i][j])) for i in N for j in N}
    Cij = {(i, j): Dij[(i, j)] / avg_speed_kmph * 60.0 for i in N for j in N}

    S = {k: (service_mno_min if k in I else service_dump_min) for k in I + F}
    T_L = float(shift_minutes)
    L = {v: avg_speed_kmph * (T_L / 60.0) for v in V}

    idx, names, lb, ub, integrality = {}, [], [], [], []

    def add_var(name, low, up, integ):
        idx[name] = len(names)
        names.append(name)
        lb.append(low)
        ub.append(up)
        integrality.append(integ)

    for v in V:
        for i in N:
            for j in N:
                add_var(("x", i, j, v), 0, 1, 1)
    for v in V:
        for i in N:
            for j in N:
                add_var(("y", i, j, v), 0, np.inf, 0)
    for v in V:
        add_var(("u", v), 0, 1, 1)
    for v in V:
        for f in F:
            add_var(("s", f, v), 0, 1, 1)
    for v in V:
        add_var(("TT", v), 0, np.inf, 0)
    _emit(f"variables created: {len(names)}")

    rows, cols, vals, bl, bu = [], [], [], [], []
    row = 0

    def add_constr(coeffs, low, up):
        nonlocal row
        for var, coef in coeffs.items():
            rows.append(row)
            cols.append(idx[var])
            vals.append(coef)
        bl.append(low)
        bu.append(up)
        row += 1

    for i in I:
        coeff = {}
        for v in V:
            for j in N:
                coeff[("x", i, j, v)] = coeff.get(("x", i, j, v), 0.0) + 1.0
        add_constr(coeff, 1.0, 1.0)

    for v in V:
        d = depot_of[v]
        coeff = {("u", v): -1.0}
        for j in N:
            coeff[("x", d, j, v)] = coeff.get(("x", d, j, v), 0.0) + 1.0
        add_constr(coeff, 0.0, 0.0)

    for v in V:
        d = depot_of[v]
        coeff = {("u", v): -1.0}
        for i in N:
            coeff[("x", i, d, v)] = coeff.get(("x", i, d, v), 0.0) + 1.0
        add_constr(coeff, 0.0, 0.0)

    for v in V:
        d = depot_of[v]
        for k in N:
            if k == d:
                continue
            coeff = {}
            for i in N:
                coeff[("x", i, k, v)] = coeff.get(("x", i, k, v), 0.0) + 1.0
            for j in N:
                coeff[("x", k, j, v)] = coeff.get(("x", k, j, v), 0.0) - 1.0
            add_constr(coeff, 0.0, 0.0)

    for v in V:
        for i in N:
            add_constr({("x", i, i, v): 1.0}, 0.0, 0.0)

    for v in V:
        coeff = {("u", v): -1.0}
        for f in F:
            coeff[("s", f, v)] = 1.0
        add_constr(coeff, 0.0, 0.0)

    for v in V:
        for f in F:
            coeff = {("s", f, v): -1.0}
            for i in N:
                coeff[("x", i, f, v)] = coeff.get(("x", i, f, v), 0.0) + 1.0
            add_constr(coeff, 0.0, 0.0)

    for i in I:
        df = dest[i]
        for v in V:
            coeff = {("s", df, v): -1.0}
            for j in N:
                coeff[("x", i, j, v)] = coeff.get(("x", i, j, v), 0.0) + 1.0
            add_constr(coeff, -np.inf, 0.0)

    for v in V:
        d = depot_of[v]
        for f in F:
            add_constr({("x", f, d, v): 1.0, ("s", f, v): -1.0}, 0.0, 0.0)
            coeff = {}
            for j in N:
                if j != d:
                    coeff[("x", f, j, v)] = 1.0
            add_constr(coeff, 0.0, 0.0)

    for v in V:
        Qv = Q[v]
        for i in N:
            for j in N:
                add_constr({("y", i, j, v): 1.0, ("x", i, j, v): -Qv}, -np.inf, 0.0)

    for v in V:
        d = depot_of[v]
        coeff = {}
        for j in N:
            coeff[("y", d, j, v)] = 1.0
        add_constr(coeff, 0.0, 0.0)

    for v in V:
        for i in I:
            coeff = {}
            for j in N:
                coeff[("y", i, j, v)] = coeff.get(("y", i, j, v), 0.0) + 1.0
                coeff[("y", j, i, v)] = coeff.get(("y", j, i, v), 0.0) - 1.0
                coeff[("x", i, j, v)] = coeff.get(("x", i, j, v), 0.0) - q[i]
            add_constr(coeff, 0.0, 0.0)

    for v in V:
        own = depot_of[v]
        for d in Dnodes:
            if d == own:
                continue
            coeff = {}
            for i in N:
                coeff[("x", i, d, v)] = 1.0
            add_constr(coeff, 0.0, 0.0)

    for v in V:
        own = depot_of[v]
        for d in Dnodes:
            if d == own:
                continue
            coeff = {}
            for j in N:
                coeff[("x", d, j, v)] = 1.0
            add_constr(coeff, 0.0, 0.0)

    for v in V:
        for f in F:
            coeff = {}
            for j in N:
                coeff[("y", f, j, v)] = 1.0
            add_constr(coeff, 0.0, 0.0)

    for v in V:
        coeff = {("TT", v): 1.0}
        for i in N:
            for j in N:
                coeff[("x", i, j, v)] = coeff.get(("x", i, j, v), 0.0) - Cij[(i, j)]
        for k in I + F:
            for i in N:
                coeff[("x", i, k, v)] = coeff.get(("x", i, k, v), 0.0) - S[k]
        add_constr(coeff, 0.0, 0.0)
        add_constr({("TT", v): 1.0, ("u", v): -T_L}, -np.inf, 0.0)

    for v in V:
        d = depot_of[v]
        for f in F:
            add_constr({("x", d, f, v): 1.0}, 0.0, 0.0)

    for v in V:
        coeff = {("u", v): -L[v]}
        for i in N:
            for j in N:
                coeff[("x", i, j, v)] = coeff.get(("x", i, j, v), 0.0) + Dij[(i, j)]
        add_constr(coeff, -np.inf, 0.0)
    _emit(f"constraints created: {row}")

    c = np.zeros(len(names))
    for v in V:
        for i in N:
            for j in N:
                c[idx[("y", i, j, v)]] = Dij[(i, j)]

    A = sp.coo_array((vals, (rows, cols)), shape=(row, len(names)))
    bounds = Bounds(lb, ub)
    constraints = LinearConstraint(A, bl, bu)

    _emit("solve MILP (time_limit=60s)")
    t_solve = time.perf_counter()
    res = milp(
        c=c,
        integrality=np.array(integrality),
        bounds=bounds,
        constraints=constraints,
        options={"disp": False, "time_limit": 60},
    )
    _emit(
        f"MILP solved in {time.perf_counter() - t_solve:.1f}s "
        f"(status={getattr(res, 'status', None)})"
    )

    if res is None or getattr(res, "x", None) is None:
        status = getattr(res, "status", None)
        message = getattr(res, "message", "MILP solver returned no solution vector")
        raise RuntimeError(f"MILP failed (status={status}): {message}")

    sol = res.x
    xsol, ysol, u_sol, TT_sol = {}, {}, {}, {}
    for v in V:
        u_sol[v] = sol[idx[("u", v)]]
        TT_sol[v] = sol[idx[("TT", v)]]
        for i in N:
            for j in N:
                xv = sol[idx[("x", i, j, v)]]
                yv = sol[idx[("y", i, j, v)]]
                if xv > 0.5:
                    xsol[(i, j, v)] = xv
                if yv > 1e-8:
                    ysol[(i, j, v)] = yv

    out_arcs = defaultdict(list)
    for (i, j, v), _ in xsol.items():
        out_arcs[(v, i)].append(j)

    def route_for(v):
        d = depot_of[v]
        route = [d]
        cur = d
        seen = set()
        while True:
            nxts = out_arcs.get((v, cur), [])
            if not nxts:
                break
            nxt = nxts[0]
            route.append(nxt)
            if nxt == d:
                break
            if (cur, nxt) in seen:
                route.append("LOOP")
                break
            seen.add((cur, nxt))
            cur = nxt
        return route

    used = [v for v in V if u_sol[v] > 0.5]
    rows_out = []
    for v in used:
        route = route_for(v)
        served = [n for n in route if n in I]
        dump_nodes = [n for n in route if n in F]
        route_km = sum(Dij[(a, b)] for a, b in zip(route, route[1:]))
        tonkm = sum(Dij[(i, j)] * ysol.get((i, j, v), 0.0) for i in N for j in N)
        rows_out.append({
            "agent_id": v,
            "depot": depot_of[v],
            "route_nodes": route,
            "served_task_ids": [task_of_source[s] for s in served],
            "dump": dump_nodes[0] if dump_nodes else None,
            "load_tons": sum(q[i] for i in served),
            "route_km": route_km,
            "ton_km": tonkm,
            "TT_min": TT_sol[v],
        })
    _emit(f"decode done in {time.perf_counter() - t0:.1f}s, used_agents={len(used)}")

    return res, pd.DataFrame(rows_out)


def assign_tasks_greedy(
    dataset,
    graph,
    payload,
    cache,
    *,
    show_progress: bool = False,
    progress_hook: Callable[[str], None] | None = None,
):
    res, routes_df = solve_sandbox_milp(
        payload,
        show_progress=show_progress,
        progress_hook=progress_hook,
    )

    source_to_task = {t.source_node_id: t for t in dataset.tasks}

    states = {}
    for agent_id, agent in dataset.fleet.agents.items():
        agent_depots = payload["metadata"]["agent_depots"]

        states[agent_id] = AgentState(
            agent_id=agent_id,
            vehicle_type=agent.vehicle_type,
            capacity_tons=agent.capacity_tons,
            is_compact=agent.is_compact,
            depot_node=agent_depots[agent_id],
            current_node=agent_depots[agent_id],
        )

    routes = []
    assigned = set()
    route_counter = 1

    for rec in routes_df.to_dict("records"):
        preferred_agent_id = rec["agent_id"]

        for source_node in rec["route_nodes"]:
            if source_node not in source_to_task:
                continue

            task = source_to_task[source_node]
            task_id = task.task_id
            if task_id in assigned:
                continue

            source_graph_node = dataset.graph.nodes[task.source_node_id]

            candidate_agent_ids = [preferred_agent_id] + [
                aid for aid in states.keys() if aid != preferred_agent_id
            ]

            chosen_agent_id = None
            chosen_agent = None

            for aid in candidate_agent_ids:
                agent = dataset.fleet.agents[aid]

                if source_graph_node.center and not agent.is_compact:
                    continue

                try:
                    path = nx.shortest_path(
                        graph,
                        source=task.source_node_id,
                        target=task.destination_node_id,
                        weight="distance_km",
                    )
                except Exception:
                    continue

                chosen_agent_id = aid
                chosen_agent = agent
                break

            if chosen_agent_id is None:
                continue

            state = states[chosen_agent_id]

            route_id = f"route_{route_counter:04d}"
            routes.append(
                Route(
                    route_id=route_id,
                    agent_id=chosen_agent_id,
                    path=tuple(path),
                    task_ids=(task_id,),
                )
            )
            state.task_ids.append(task_id)
            state.route_ids.append(route_id)
            route_counter += 1
            assigned.add(task_id)

    unassigned = [t.task_id for t in dataset.tasks if t.task_id not in assigned]
    if progress_hook is not None:
        progress_hook(
            f"assignment built: routes={len(routes)}, assigned={len(assigned)}, unassigned={len(unassigned)}"
        )
    elif show_progress:
        print(
            f"[milp] assignment built: routes={len(routes)}, assigned={len(assigned)}, "
            f"unassigned={len(unassigned)}",
            flush=True,
        )
    return routes, states, unassigned
