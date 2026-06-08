from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

import networkx as nx
import numpy as np

from .. import core
from ..dataset import CONTAINER_TO_VEHICLE_TYPES, Route, RoutingDataset, Task
from ..gap_vrp_solver import SolverResult


@dataclass
class _Prepared:
    tasks: list[Task]
    task_ids: list[str]
    task_sources: list[str]
    task_destinations: list[str]
    task_masses: np.ndarray
    task_is_center: np.ndarray
    task_service_km: np.ndarray
    task_service_hours: np.ndarray
    task_containers: np.ndarray
    task_paths: list[tuple[str, ...] | None]
    candidates: list[np.ndarray]
    candidate_base_km: list[np.ndarray]
    candidate_base_h: list[np.ndarray]
    agent_ids: list[str]
    agent_depots: list[str | None]
    agent_km_limit: np.ndarray
    agent_h_limit: np.ndarray
    agent_speed: np.ndarray
    agent_capacity: np.ndarray
    agent_is_compact: np.ndarray
    agent_raw_volume_limit: np.ndarray
    agent_body_volume: np.ndarray
    agent_compaction: np.ndarray
    agent_vehicle_types: list[str]
    depot_x: np.ndarray
    depot_y: np.ndarray
    task_source_x: np.ndarray
    task_source_y: np.ndarray
    task_dest_x: np.ndarray
    task_dest_y: np.ndarray
    container_masks: dict[str, np.ndarray]


def _emit(
    message: str,
    *,
    show_progress: bool,
    progress_hook: Callable[[str], None] | None,
    prefix: str,
) -> None:
    if progress_hook is not None:
        progress_hook(f"[{prefix}] {message}")
    elif show_progress:
        print(f"[{prefix}] {message}", flush=True)


def _pair_service_lookup(
    *,
    tasks: list[Task],
    graph: nx.DiGraph,
    show_progress: bool,
    progress_hook: Callable[[str], None] | None,
    prefix: str,
) -> dict[tuple[str, str], tuple[float, tuple[str, ...]]]:
    pairs_by_dest: dict[str, set[str]] = {}
    for task in tasks:
        pairs_by_dest.setdefault(task.destination_node_id, set()).add(task.source_node_id)

    rev = graph.reverse(copy=False)
    lookup: dict[tuple[str, str], tuple[float, tuple[str, ...]]] = {}

    destinations = sorted(pairs_by_dest)
    for idx, dst in enumerate(destinations, start=1):
        t0 = time.perf_counter()
        dist_map, path_map = nx.single_source_dijkstra(rev, dst, weight="distance_km")
        added = 0
        for src in pairs_by_dest[dst]:
            if src not in dist_map or src not in path_map:
                continue
            rev_path = path_map[src]
            lookup[(src, dst)] = (float(dist_map[src]), tuple(reversed(rev_path)))
            added += 1
        _emit(
            f"service matrix {idx}/{len(destinations)}: dest={dst}, pairs={added}, sec={time.perf_counter() - t0:.2f}",
            show_progress=show_progress,
            progress_hook=progress_hook,
            prefix=prefix,
        )

    return lookup


def _prepare(
    *,
    dataset: RoutingDataset,
    payload: dict[str, Any],
    graph: nx.DiGraph,
    candidate_k: int,
    road_factor: float,
    trip_share: float,
    show_progress: bool,
    progress_hook: Callable[[str], None] | None,
    prefix: str,
) -> _Prepared:
    tasks = list(dataset.tasks)
    states = core.initialize_agent_states(dataset, payload)
    agent_ids = list(states.keys())

    node_xy = {node_id: (node.x, node.y) for node_id, node in dataset.graph.nodes.items()}

    agent_depots = [states[aid].depot_node for aid in agent_ids]
    agent_speed = np.array([core.AVG_SPEED_KMPH_BY_TYPE[states[aid].vehicle_type] for aid in agent_ids], dtype=float)
    agent_km_limit = np.array([core.MAX_DAILY_KM_BY_TYPE[states[aid].vehicle_type] for aid in agent_ids], dtype=float)
    agent_h_limit = np.array([core.MAX_SHIFT_HOURS_BY_TYPE[states[aid].vehicle_type] for aid in agent_ids], dtype=float)
    agent_capacity = np.array([states[aid].capacity_tons for aid in agent_ids], dtype=float)
    agent_is_compact = np.array([states[aid].is_compact for aid in agent_ids], dtype=bool)
    agent_raw_volume_limit = np.array([states[aid].max_raw_volume_m3 for aid in agent_ids], dtype=float)
    agent_body_volume = np.array([states[aid].body_volume_m3 for aid in agent_ids], dtype=float)
    agent_compaction = np.array([states[aid].compaction_coeff for aid in agent_ids], dtype=float)
    agent_vehicle_types = [states[aid].vehicle_type for aid in agent_ids]

    depot_xy = np.array(
        [node_xy.get(depot, (0.0, 0.0)) if depot is not None else (0.0, 0.0) for depot in agent_depots],
        dtype=float,
    )
    depot_x = depot_xy[:, 0]
    depot_y = depot_xy[:, 1]

    container_masks: dict[str, np.ndarray] = {}
    for container, vehicle_types in CONTAINER_TO_VEHICLE_TYPES.items():
        container_masks[container] = np.array(
            [
                (
                    states[aid].vehicle_type in vehicle_types
                    and (
                        not states[aid].cap_container_types
                        or container in states[aid].cap_container_types
                    )
                )
                for aid in agent_ids
            ],
            dtype=bool,
        )

    _emit(
        f"precompute service paths: tasks={len(tasks)}, agents={len(agent_ids)}",
        show_progress=show_progress,
        progress_hook=progress_hook,
        prefix=prefix,
    )
    service_lookup = _pair_service_lookup(
        tasks=tasks,
        graph=graph,
        show_progress=show_progress,
        progress_hook=progress_hook,
        prefix=prefix,
    )

    n_tasks = len(tasks)
    task_ids = [task.task_id for task in tasks]
    task_sources = [task.source_node_id for task in tasks]
    task_destinations = [task.destination_node_id for task in tasks]
    task_masses = np.array([task.mass_tons for task in tasks], dtype=float)
    task_is_center = np.array([dataset.graph.nodes[task.source_node_id].center for task in tasks], dtype=bool)
    task_service_hours = np.array([core.SERVICE_HOURS_BY_CONTAINER[task.container_type] for task in tasks], dtype=float)
    task_containers = np.array([task.container_type for task in tasks], dtype=object)
    task_service_km = np.full(n_tasks, np.inf, dtype=float)
    task_paths: list[tuple[str, ...] | None] = [None] * n_tasks

    candidates: list[np.ndarray] = [np.empty(0, dtype=np.int32) for _ in range(n_tasks)]
    candidate_base_km: list[np.ndarray] = [np.empty(0, dtype=float) for _ in range(n_tasks)]
    candidate_base_h: list[np.ndarray] = [np.empty(0, dtype=float) for _ in range(n_tasks)]

    report_every = max(1, n_tasks // 10)
    for tidx, task in enumerate(tasks):
        pair = service_lookup.get((task.source_node_id, task.destination_node_id))
        if pair is None:
            continue
        service_km, path = pair
        task_service_km[tidx] = service_km
        task_paths[tidx] = path

        src_x, src_y = node_xy[task.source_node_id]
        dst_x, dst_y = node_xy[task.destination_node_id]

        mask = container_masks.get(task.container_type, np.zeros(len(agent_ids), dtype=bool)).copy()
        if task.compatible_vehicle_types:
            allowed_task_types = set(task.compatible_vehicle_types)
            mask &= np.array([vt in allowed_task_types for vt in agent_vehicle_types], dtype=bool)
        mask &= agent_capacity + 1e-9 >= task.mass_tons
        if dataset.graph.nodes[task.source_node_id].center:
            mask &= agent_is_compact
        if task.volume_raw_m3 > 0:
            mask &= (agent_raw_volume_limit <= 0) | (agent_raw_volume_limit + 1e-9 >= task.volume_raw_m3)

        cands = np.flatnonzero(mask)
        if cands.size == 0:
            continue

        body_req = np.array(
            [task.body_volume_for_vehicle(agent_vehicle_types[i]) for i in cands],
            dtype=float,
        )
        if task.volume_raw_m3 > 0:
            fallback = np.maximum(agent_compaction[cands], 1e-9)
            body_req = np.where(body_req > 0, body_req, task.volume_raw_m3 / fallback)
        body_ok = (agent_body_volume[cands] <= 0) | (body_req <= agent_body_volume[cands] + 1e-9)
        if not body_ok.any():
            continue
        cands = cands[body_ok]
        body_req = body_req[body_ok]
        if cands.size == 0:
            continue

        depot_to_src = np.hypot(depot_x[cands] - src_x, depot_y[cands] - src_y)
        dst_to_depot = np.hypot(dst_x - depot_x[cands], dst_y - depot_y[cands])
        base_km = service_km + trip_share * road_factor * (depot_to_src + dst_to_depot)
        if cands.size > candidate_k:
            take = np.argpartition(base_km, candidate_k - 1)[:candidate_k]
            cands = cands[take]
            base_km = base_km[take]
            body_req = body_req[take]

        base_h = base_km / agent_speed[cands] + task_service_hours[tidx]

        candidates[tidx] = cands.astype(np.int32, copy=False)
        candidate_base_km[tidx] = base_km.astype(float, copy=False)
        candidate_base_h[tidx] = base_h.astype(float, copy=False)

        if (tidx + 1) % report_every == 0 or tidx + 1 == n_tasks:
            _emit(
                f"prepare compatibility: {tidx + 1}/{n_tasks}",
                show_progress=show_progress,
                progress_hook=progress_hook,
                prefix=prefix,
            )

    return _Prepared(
        tasks=tasks,
        task_ids=task_ids,
        task_sources=task_sources,
        task_destinations=task_destinations,
        task_masses=task_masses,
        task_is_center=task_is_center,
        task_service_km=task_service_km,
        task_service_hours=task_service_hours,
        task_containers=task_containers,
        task_paths=task_paths,
        candidates=candidates,
        candidate_base_km=candidate_base_km,
        candidate_base_h=candidate_base_h,
        agent_ids=agent_ids,
        agent_depots=agent_depots,
        agent_km_limit=agent_km_limit,
        agent_h_limit=agent_h_limit,
        agent_speed=agent_speed,
        agent_capacity=agent_capacity,
        agent_is_compact=agent_is_compact,
        agent_raw_volume_limit=agent_raw_volume_limit,
        agent_body_volume=agent_body_volume,
        agent_compaction=agent_compaction,
        agent_vehicle_types=agent_vehicle_types,
        depot_x=depot_x,
        depot_y=depot_y,
        task_source_x=np.array([node_xy[s][0] for s in task_sources], dtype=float),
        task_source_y=np.array([node_xy[s][1] for s in task_sources], dtype=float),
        task_dest_x=np.array([node_xy[d][0] for d in task_destinations], dtype=float),
        task_dest_y=np.array([node_xy[d][1] for d in task_destinations], dtype=float),
        container_masks=container_masks,
    )


def _priority_order(prep: _Prepared, rng: np.random.Generator, noise: float) -> list[int]:
    idxs = list(range(len(prep.tasks)))
    idxs.sort(
        key=lambda i: (
            -int(prep.task_is_center[i]),
            -prep.task_masses[i],
            rng.random() * max(noise, 1e-12),
        )
    )
    return idxs


def _constructive_assign(
    *,
    prep: _Prepared,
    order: list[int],
    rng: np.random.Generator,
    rcl_size: int,
    score_noise: float,
    load_penalty: float,
    overload_slack: float,
    deadline: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_agents = len(prep.agent_ids)
    n_tasks = len(prep.tasks)

    owner = np.full(n_tasks, -1, dtype=np.int32)
    used_km = np.zeros(n_agents, dtype=float)
    used_h = np.zeros(n_agents, dtype=float)
    used_task_km = np.zeros(n_agents, dtype=float)
    used_service_h = np.zeros(n_agents, dtype=float)

    for pos, tidx in enumerate(order):
        if pos % 256 == 0 and time.perf_counter() > deadline:
            break

        cands = prep.candidates[tidx]
        if cands.size == 0 or not np.isfinite(prep.task_service_km[tidx]):
            continue
        inc_km = prep.candidate_base_km[tidx]
        inc_h = prep.candidate_base_h[tidx]

        km_cap = prep.agent_km_limit[cands] * overload_slack
        h_cap = prep.agent_h_limit[cands] * overload_slack
        feasible = (used_km[cands] + inc_km <= km_cap) & (used_h[cands] + inc_h <= h_cap)
        if not feasible.any():
            continue

        cands_f = cands[feasible]
        inc_km_f = inc_km[feasible]
        inc_h_f = inc_h[feasible]

        load = used_km[cands_f] / np.maximum(prep.agent_km_limit[cands_f], 1e-6)
        score = inc_km_f + load_penalty * load
        if score_noise > 0:
            score = score * (1.0 + score_noise * rng.random(score.size))

        rcl = min(max(1, rcl_size), score.size)
        if rcl == score.size:
            pick_local = int(rng.integers(0, score.size))
        else:
            top = np.argpartition(score, rcl - 1)[:rcl]
            pick_local = int(top[int(rng.integers(0, top.size))])

        aidx = int(cands_f[pick_local])
        owner[tidx] = aidx
        used_km[aidx] += float(inc_km_f[pick_local])
        used_h[aidx] += float(inc_h_f[pick_local])
        used_task_km[aidx] += float(prep.task_service_km[tidx])
        used_service_h[aidx] += float(prep.task_service_hours[tidx])

    return owner, used_km, used_h, used_task_km, used_service_h


def _objective(owner: np.ndarray, used_km: np.ndarray) -> float:
    unassigned = int(np.sum(owner < 0))
    return float(unassigned) * 1_000_000.0 + float(np.sum(used_km))


def _repair_unassigned(
    *,
    prep: _Prepared,
    owner: np.ndarray,
    used_km: np.ndarray,
    used_h: np.ndarray,
    used_task_km: np.ndarray,
    used_service_h: np.ndarray,
    road_factor: float,
    load_penalty: float,
    overload_slack: float,
    repair_trip_share: float,
    deadline: float,
    max_passes: int,
    show_progress: bool,
    progress_hook: Callable[[str], None] | None,
    prefix: str,
) -> None:
    n_tasks = len(prep.tasks)
    if n_tasks == 0:
        return

    for pidx in range(1, max_passes + 1):
        if time.perf_counter() >= deadline:
            break
        unassigned = np.flatnonzero(owner < 0)
        if unassigned.size == 0:
            break

        # Harder tasks first: center + heavier mass.
        order = sorted(
            unassigned.tolist(),
            key=lambda i: (-int(prep.task_is_center[i]), -float(prep.task_masses[i])),
        )
        inserted = 0

        for tidx in order:
            if time.perf_counter() >= deadline:
                break
            if owner[tidx] >= 0:
                continue
            service_km = float(prep.task_service_km[tidx])
            if not np.isfinite(service_km):
                continue

            container = str(prep.task_containers[tidx])
            mask = prep.container_masks.get(container)
            if mask is None:
                continue
            feasible_agents = mask.copy()
            feasible_agents &= prep.agent_capacity + 1e-9 >= prep.task_masses[tidx]
            if prep.task_is_center[tidx]:
                feasible_agents &= prep.agent_is_compact
            if prep.tasks[tidx].compatible_vehicle_types:
                allowed = set(prep.tasks[tidx].compatible_vehicle_types)
                feasible_agents &= np.array([vt in allowed for vt in prep.agent_vehicle_types], dtype=bool)
            if prep.tasks[tidx].volume_raw_m3 > 0:
                feasible_agents &= (
                    (prep.agent_raw_volume_limit <= 0)
                    | (prep.agent_raw_volume_limit + 1e-9 >= prep.tasks[tidx].volume_raw_m3)
                )

            cands = np.flatnonzero(feasible_agents)
            if cands.size == 0:
                continue

            body_req = np.array(
                [prep.tasks[tidx].body_volume_for_vehicle(prep.agent_vehicle_types[i]) for i in cands],
                dtype=float,
            )
            if prep.tasks[tidx].volume_raw_m3 > 0:
                fallback = np.maximum(prep.agent_compaction[cands], 1e-9)
                body_req = np.where(body_req > 0, body_req, prep.tasks[tidx].volume_raw_m3 / fallback)
            body_ok = (prep.agent_body_volume[cands] <= 0) | (body_req <= prep.agent_body_volume[cands] + 1e-9)
            if not body_ok.any():
                continue
            cands = cands[body_ok]
            if cands.size == 0:
                continue

            depot_to_src = np.hypot(
                prep.depot_x[cands] - prep.task_source_x[tidx],
                prep.depot_y[cands] - prep.task_source_y[tidx],
            )
            dst_to_depot = np.hypot(
                prep.task_dest_x[tidx] - prep.depot_x[cands],
                prep.task_dest_y[tidx] - prep.depot_y[cands],
            )
            inc_km = service_km + repair_trip_share * road_factor * (depot_to_src + dst_to_depot)
            inc_h = inc_km / np.maximum(prep.agent_speed[cands], 1e-6) + prep.task_service_hours[tidx]

            km_cap = prep.agent_km_limit[cands] * overload_slack
            h_cap = prep.agent_h_limit[cands] * overload_slack
            feasible = (used_km[cands] + inc_km <= km_cap) & (used_h[cands] + inc_h <= h_cap)
            if not feasible.any():
                continue

            cands_f = cands[feasible]
            inc_km_f = inc_km[feasible]
            inc_h_f = inc_h[feasible]

            load = used_km[cands_f] / np.maximum(prep.agent_km_limit[cands_f], 1e-6)
            score = inc_km_f + load_penalty * load
            best_local = int(np.argmin(score))
            aidx = int(cands_f[best_local])

            owner[tidx] = aidx
            used_km[aidx] += float(inc_km_f[best_local])
            used_h[aidx] += float(inc_h_f[best_local])
            used_task_km[aidx] += service_km
            used_service_h[aidx] += float(prep.task_service_hours[tidx])
            inserted += 1

        _emit(
            f"repair pass {pidx}/{max_passes}: inserted={inserted}, remaining={int(np.sum(owner < 0))}",
            show_progress=show_progress,
            progress_hook=progress_hook,
            prefix=prefix,
        )
        if inserted == 0:
            break


def _build_result(
    *,
    prep: _Prepared,
    dataset: RoutingDataset,
    payload: dict[str, Any],
    owner: np.ndarray,
    used_km: np.ndarray,
    used_h: np.ndarray,
    used_task_km: np.ndarray,
    used_service_h: np.ndarray,
    method_label: str,
    route_prefix: str,
    validate_solution_dataset: bool = False,
) -> SolverResult:
    states = core.initialize_agent_states(dataset, payload)

    routes: list[Route] = []
    route_counter = 1

    unassigned_task_ids: list[str] = []
    tasks_by_agent: dict[int, list[int]] = {aidx: [] for aidx in range(len(prep.agent_ids))}

    for tidx, aidx in enumerate(owner.tolist()):
        if aidx < 0:
            unassigned_task_ids.append(prep.task_ids[tidx])
            continue
        tasks_by_agent[aidx].append(tidx)

    for aidx, task_indices in tasks_by_agent.items():
        if not task_indices:
            continue
        agent_id = prep.agent_ids[aidx]
        state = states[agent_id]
        state.task_ids = [prep.task_ids[tidx] for tidx in task_indices]
        state.task_km = round(float(used_task_km[aidx]), 6)
        state.deadhead_km = round(max(0.0, float(used_km[aidx] - used_task_km[aidx])), 6)
        state.service_hours = round(float(used_service_h[aidx]), 6)
        state.drive_hours = round(max(0.0, float(used_h[aidx] - used_service_h[aidx])), 6)
        state.route_ids = []

        for tidx in task_indices:
            task = prep.tasks[tidx]
            path = prep.task_paths[tidx]
            if path is None:
                unassigned_task_ids.append(task.task_id)
                continue
            route_id = f"{route_prefix}_{route_counter:06d}"
            route_counter += 1
            state.route_ids.append(route_id)
            routes.append(
                Route(
                    route_id=route_id,
                    agent_id=agent_id,
                    path=path,
                    task_ids=(task.task_id,),
                )
            )

    limit_violations = core.validate_daily_limits(states)

    assigned_ids = {task_id for route in routes for task_id in route.task_ids}
    for task_id in prep.task_ids:
        if task_id not in assigned_ids and task_id not in unassigned_task_ids:
            unassigned_task_ids.append(task_id)
    unassigned_task_ids = sorted(set(unassigned_task_ids))

    feasible = len(unassigned_task_ids) == 0 and len(limit_violations) == 0
    if feasible and validate_solution_dataset:
        try:
            _ = core.build_solution_dataset(dataset, routes)
        except Exception:
            feasible = False

    return SolverResult(
        method_label=method_label,
        routes=routes,
        states=states,
        unassigned=unassigned_task_ids,
        transport_work_ton_km=None,
        feasible=feasible,
        limit_violations=limit_violations,
        n_assigned=len(routes),
        n_unassigned=len(unassigned_task_ids),
        active_agents=sum(1 for state in states.values() if state.task_ids),
    )


def solve_real_stochastic_grasp(
    *,
    dataset: RoutingDataset,
    payload: dict[str, Any],
    graph: nx.DiGraph,
    cache: dict[tuple[str, str], tuple[list[str], float] | None],
    time_budget_sec: int = 180,
    max_starts: int = 3,
    candidate_k: int = 20,
    rcl_size: int = 5,
    load_penalty: float = 8.0,
    road_factor: float = 1.25,
    trip_share: float = 0.35,
    overload_slack: float = 1.02,
    seed: int | None = 42,
    show_progress: bool = False,
    progress_hook: Callable[[str], None] | None = None,
) -> SolverResult:
    del cache

    prefix = "real_stochastic_grasp"
    start = time.perf_counter()
    deadline = start + max(1, int(time_budget_sec))
    rng = np.random.default_rng(seed)

    _emit(
        f"start: tasks={len(dataset.tasks)}, agents={len(dataset.fleet.agents)}, budget={time_budget_sec}s",
        show_progress=show_progress,
        progress_hook=progress_hook,
        prefix=prefix,
    )

    prep = _prepare(
        dataset=dataset,
        payload=payload,
        graph=graph,
        candidate_k=max(4, int(candidate_k)),
        road_factor=road_factor,
        trip_share=trip_share,
        show_progress=show_progress,
        progress_hook=progress_hook,
        prefix=prefix,
    )

    best_owner: np.ndarray | None = None
    best_used_km: np.ndarray | None = None
    best_used_h: np.ndarray | None = None
    best_used_task_km: np.ndarray | None = None
    best_used_service_h: np.ndarray | None = None
    best_obj = float("inf")

    starts_done = 0
    effective_starts = max(1, int(max_starts))
    if len(prep.tasks) > 10_000:
        effective_starts = min(effective_starts, 2)

    for sidx in range(effective_starts):
        if time.perf_counter() >= deadline:
            break
        order_noise = 1.0 + 0.5 * sidx
        score_noise = 0.20 + 0.10 * sidx
        order = _priority_order(prep, rng, noise=order_noise)

        owner, used_km, used_h, used_task_km, used_service_h = _constructive_assign(
            prep=prep,
            order=order,
            rng=rng,
            rcl_size=max(1, int(rcl_size)),
            score_noise=score_noise,
            load_penalty=load_penalty,
            overload_slack=max(1.0, float(overload_slack)),
            deadline=deadline,
        )

        obj = _objective(owner, used_km)
        unassigned = int(np.sum(owner < 0))
        _emit(
            f"start {sidx + 1}/{effective_starts}: unassigned={unassigned}, km={used_km.sum():.1f}, obj={obj:.1f}",
            show_progress=show_progress,
            progress_hook=progress_hook,
            prefix=prefix,
        )

        starts_done += 1
        if obj < best_obj:
            best_obj = obj
            best_owner = owner.copy()
            best_used_km = used_km.copy()
            best_used_h = used_h.copy()
            best_used_task_km = used_task_km.copy()
            best_used_service_h = used_service_h.copy()

    if best_owner is None:
        n_tasks = len(prep.tasks)
        best_owner = np.full(n_tasks, -1, dtype=np.int32)
        best_used_km = np.zeros(len(prep.agent_ids), dtype=float)
        best_used_h = np.zeros(len(prep.agent_ids), dtype=float)
        best_used_task_km = np.zeros(len(prep.agent_ids), dtype=float)
        best_used_service_h = np.zeros(len(prep.agent_ids), dtype=float)

    _emit(
        f"done starts={starts_done}, elapsed={time.perf_counter() - start:.1f}s",
        show_progress=show_progress,
        progress_hook=progress_hook,
        prefix=prefix,
    )

    if len(prep.tasks) <= 8_000 and time.perf_counter() < deadline:
        _repair_unassigned(
            prep=prep,
            owner=best_owner,
            used_km=best_used_km,
            used_h=best_used_h,
            used_task_km=best_used_task_km,
            used_service_h=best_used_service_h,
            road_factor=road_factor,
            load_penalty=load_penalty,
            overload_slack=max(1.0, float(overload_slack)),
            repair_trip_share=min(0.05, max(0.0, float(trip_share))),
            deadline=deadline,
            max_passes=3,
            show_progress=show_progress,
            progress_hook=progress_hook,
            prefix=prefix,
        )

    return _build_result(
        prep=prep,
        dataset=dataset,
        payload=payload,
        owner=best_owner,
        used_km=best_used_km,
        used_h=best_used_h,
        used_task_km=best_used_task_km,
        used_service_h=best_used_service_h,
        method_label="Real Stochastic GRASP",
        route_prefix="RSGRASP_ROUTE",
        validate_solution_dataset=False,
    )


def solve_real_stochastic_rr(
    *,
    dataset: RoutingDataset,
    payload: dict[str, Any],
    graph: nx.DiGraph,
    cache: dict[tuple[str, str], tuple[list[str], float] | None],
    time_budget_sec: int = 180,
    candidate_k: int = 20,
    rcl_size: int = 5,
    load_penalty: float = 8.0,
    road_factor: float = 1.25,
    trip_share: float = 0.35,
    overload_slack: float = 1.02,
    batch_size: int = 800,
    seed: int | None = 42,
    show_progress: bool = False,
    progress_hook: Callable[[str], None] | None = None,
) -> SolverResult:
    del cache

    prefix = "real_stochastic_rr"
    t0 = time.perf_counter()
    deadline = t0 + max(1, int(time_budget_sec))
    rng = np.random.default_rng(seed)

    _emit(
        f"start: tasks={len(dataset.tasks)}, agents={len(dataset.fleet.agents)}, budget={time_budget_sec}s",
        show_progress=show_progress,
        progress_hook=progress_hook,
        prefix=prefix,
    )

    prep = _prepare(
        dataset=dataset,
        payload=payload,
        graph=graph,
        candidate_k=max(4, int(candidate_k)),
        road_factor=road_factor,
        trip_share=trip_share,
        show_progress=show_progress,
        progress_hook=progress_hook,
        prefix=prefix,
    )

    order = _priority_order(prep, rng, noise=1.0)
    owner, used_km, used_h, used_task_km, used_service_h = _constructive_assign(
        prep=prep,
        order=order,
        rng=rng,
        rcl_size=max(1, int(rcl_size)),
        score_noise=0.25,
        load_penalty=load_penalty,
        overload_slack=max(1.0, float(overload_slack)),
        deadline=deadline,
    )

    best_owner = owner.copy()
    best_used_km = used_km.copy()
    best_used_h = used_h.copy()
    best_used_task_km = used_task_km.copy()
    best_used_service_h = used_service_h.copy()
    best_obj = _objective(owner, used_km)
    _emit(
        f"initial: unassigned={int(np.sum(owner < 0))}, km={used_km.sum():.1f}",
        show_progress=show_progress,
        progress_hook=progress_hook,
        prefix=prefix,
    )

    iter_idx = 0
    last_report = t0
    while time.perf_counter() < deadline:
        iter_idx += 1
        assigned_idx = np.flatnonzero(owner >= 0)
        if assigned_idx.size == 0:
            break

        cur_batch = min(max(50, int(batch_size)), int(assigned_idx.size))
        batch = rng.choice(assigned_idx, size=cur_batch, replace=False)

        for tidx in batch.tolist():
            old_aidx = int(owner[tidx])
            if old_aidx < 0:
                continue
            match = np.where(prep.candidates[tidx] == old_aidx)[0]
            if match.size == 0:
                owner[tidx] = -1
                continue
            old_pos = int(match[0])
            owner[tidx] = -1
            used_km[old_aidx] -= float(prep.candidate_base_km[tidx][old_pos])
            used_h[old_aidx] -= float(prep.candidate_base_h[tidx][old_pos])
            used_task_km[old_aidx] -= float(prep.task_service_km[tidx])
            used_service_h[old_aidx] -= float(prep.task_service_hours[tidx])

        rng.shuffle(batch)
        for tidx in batch.tolist():
            cands = prep.candidates[tidx]
            if cands.size == 0:
                continue
            inc_km = prep.candidate_base_km[tidx]
            inc_h = prep.candidate_base_h[tidx]

            km_cap = prep.agent_km_limit[cands] * max(1.0, float(overload_slack))
            h_cap = prep.agent_h_limit[cands] * max(1.0, float(overload_slack))
            feasible = (used_km[cands] + inc_km <= km_cap) & (used_h[cands] + inc_h <= h_cap)
            if not feasible.any():
                continue

            cands_f = cands[feasible]
            inc_km_f = inc_km[feasible]
            inc_h_f = inc_h[feasible]

            load = used_km[cands_f] / np.maximum(prep.agent_km_limit[cands_f], 1e-6)
            score = inc_km_f + load_penalty * load
            score = score * (1.0 + 0.15 * rng.random(score.size))
            rcl = min(max(1, int(rcl_size)), score.size)
            if rcl == score.size:
                pick_local = int(rng.integers(0, score.size))
            else:
                top = np.argpartition(score, rcl - 1)[:rcl]
                pick_local = int(top[int(rng.integers(0, top.size))])

            aidx = int(cands_f[pick_local])
            owner[tidx] = aidx
            used_km[aidx] += float(inc_km_f[pick_local])
            used_h[aidx] += float(inc_h_f[pick_local])
            used_task_km[aidx] += float(prep.task_service_km[tidx])
            used_service_h[aidx] += float(prep.task_service_hours[tidx])

        obj = _objective(owner, used_km)
        if obj < best_obj:
            best_obj = obj
            best_owner = owner.copy()
            best_used_km = used_km.copy()
            best_used_h = used_h.copy()
            best_used_task_km = used_task_km.copy()
            best_used_service_h = used_service_h.copy()

        now = time.perf_counter()
        if now - last_report >= 2.0:
            _emit(
                f"iter {iter_idx}: unassigned={int(np.sum(owner < 0))}, best_unassigned={int(np.sum(best_owner < 0))}",
                show_progress=show_progress,
                progress_hook=progress_hook,
                prefix=prefix,
            )
            last_report = now

    _emit(
        f"done: iters={iter_idx}, elapsed={time.perf_counter() - t0:.1f}s",
        show_progress=show_progress,
        progress_hook=progress_hook,
        prefix=prefix,
    )

    if len(prep.tasks) <= 8_000 and time.perf_counter() < deadline:
        _repair_unassigned(
            prep=prep,
            owner=best_owner,
            used_km=best_used_km,
            used_h=best_used_h,
            used_task_km=best_used_task_km,
            used_service_h=best_used_service_h,
            road_factor=road_factor,
            load_penalty=load_penalty,
            overload_slack=max(1.0, float(overload_slack)),
            repair_trip_share=min(0.05, max(0.0, float(trip_share))),
            deadline=deadline,
            max_passes=2,
            show_progress=show_progress,
            progress_hook=progress_hook,
            prefix=prefix,
        )

    return _build_result(
        prep=prep,
        dataset=dataset,
        payload=payload,
        owner=best_owner,
        used_km=best_used_km,
        used_h=best_used_h,
        used_task_km=best_used_task_km,
        used_service_h=best_used_service_h,
        method_label="Real Stochastic RR",
        route_prefix="RSRR_ROUTE",
        validate_solution_dataset=False,
    )
