from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import time
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import numpy as np

from ... import core
from ...backend.io import load_dataset
from ...dataset import Route, RoutingDataset, Task
from ..real_gap_vrp_solver import solve_real_gap_vrp as _solve_real_gap_vrp
from ..real_milp_solver import solve_real_milp as _solve_real_milp
from .common import SERVICE_HOURS_BY_CONTAINER, summarize_checks
from .distance_oracle import DistanceOracleWithFallback, PrecomputedDistanceOracle
from .problem import EnrichedProblem, build_enriched_problem
from .types import AgentUsage, AssignmentRoute, EnrichedSolveResult


@dataclass(frozen=True)
class EnrichedLegacyGapConfig:
    step1_method: str = "dataset"
    gap_iter: int = 40
    use_repair: bool = True
    show_progress: bool = False
    verbose: bool = False
    progress_hook: Callable[[str], None] | None = None


@dataclass(frozen=True)
class EnrichedLegacyMILPConfig:
    time_limit_sec: int = 60
    unassigned_penalty: float = 1e5
    show_progress: bool = False
    progress_hook: Callable[[str], None] | None = None


def _safe_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _zones_from_payload(payload: dict[str, Any]) -> tuple[dict[str, int | None], dict[str, int | None]]:
    task_zone: dict[str, int | None] = {}
    for raw in payload.get("tasks", []):
        tid = raw.get("task_id")
        if tid is None:
            continue
        task_zone[str(tid)] = _safe_int(raw.get("source_zone_num"))
    agent_zone: dict[str, int | None] = {}
    for raw in payload.get("agents", []):
        aid = raw.get("agent_id")
        if aid is None:
            continue
        agent_zone[str(aid)] = _safe_int(raw.get("zone_num"))
    return task_zone, agent_zone


def _ensure_full_agent_depots(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    meta = dict(out.get("metadata") or {})
    depots = dict(meta.get("agent_depots") or {})
    for a in out.get("agents", []):
        aid = a.get("agent_id")
        depot = a.get("depot_node_id")
        if aid is None:
            continue
        if str(aid) not in depots and depot is not None:
            depots[str(aid)] = str(depot)
    meta["agent_depots"] = depots
    out["metadata"] = meta
    return out


@contextmanager
def _patched_core(
    *,
    task_zone: dict[str, int | None],
    agent_zone: dict[str, int | None],
    precomputed: PrecomputedDistanceOracle | None,
):
    orig_compat = core.is_task_compatible_with_agent_state
    orig_sp = core.shortest_path_cached

    def compat_with_zone(*, dataset: RoutingDataset, state: core.AgentState, task: Task) -> bool:
        if not orig_compat(dataset=dataset, state=state, task=task):
            return False
        tz = task_zone.get(task.task_id)
        az = agent_zone.get(state.agent_id)
        if tz is not None and az is not None and tz != az:
            return False
        return True

    def sp_with_precomputed(
        graph: nx.DiGraph,
        cache: dict[tuple[str, str], tuple[list[str], float] | None],
        source: str,
        target: str,
    ) -> tuple[list[str], float] | None:
        out = orig_sp(graph, cache, source, target)
        if out is None:
            return None
        if precomputed is None:
            return out
        d = precomputed.dist(source, target)
        if not np.isfinite(d):
            return out
        path, _old = out
        cache[(str(source), str(target))] = (path, float(d))
        return cache[(str(source), str(target))]

    core.is_task_compatible_with_agent_state = compat_with_zone
    core.shortest_path_cached = sp_with_precomputed
    try:
        yield
    finally:
        core.is_task_compatible_with_agent_state = orig_compat
        core.shortest_path_cached = orig_sp


def _distance_from_path(route: Route, edge_dist: dict[tuple[str, str], float], oracle: DistanceOracleWithFallback) -> float:
    total = 0.0
    path = list(route.path)
    for i in range(len(path) - 1):
        u = str(path[i])
        v = str(path[i + 1])
        d = edge_dist.get((u, v))
        if d is None:
            d = oracle.dist(u, v)
        total += float(d)
    return total


def _convert_legacy_result(
    *,
    result: Any,
    problem: EnrichedProblem,
    dataset: RoutingDataset,
    oracle: DistanceOracleWithFallback,
    algorithm_name: str,
    extra_details: dict[str, Any] | None = None,
) -> EnrichedSolveResult:
    task_by_id = {t.task_id: t for t in dataset.tasks}
    agent_by_id = {a.agent_id: a for a in problem.agents}
    edge_dist = {(e.source_id, e.target_id): float(e.distance_km) for e in dataset.graph.edges}
    routes: list[AssignmentRoute] = []
    usage = {aid: AgentUsage(agent_id=aid) for aid in agent_by_id}
    unassigned = list(result.unassigned)

    for i, route in enumerate(result.routes, start=1):
        task_ids = [tid for tid in route.task_ids if tid in task_by_id]
        if not task_ids:
            continue
        agent = agent_by_id.get(route.agent_id)
        if agent is None:
            continue
        payload_tons = float(sum(task_by_id[tid].mass_tons for tid in task_ids))
        loaded_km = 0.0
        service_hours = 0.0
        for tid in task_ids:
            task = task_by_id[tid]
            loaded_km += float(oracle.dist(task.source_node_id, task.destination_node_id))
            service_hours += float(SERVICE_HOURS_BY_CONTAINER.get(task.container_type, 0.25))
        total_km = _distance_from_path(route, edge_dist, oracle)
        total_hours = (total_km / max(agent.avg_speed_kmph, 1e-6)) + service_hours

        ar = AssignmentRoute(
            route_id=f"{algorithm_name.upper()}_ROUTE_{i:07d}",
            agent_id=route.agent_id,
            task_ids=tuple(task_ids),
            path=tuple(str(x) for x in route.path),
            loaded_distance_km=loaded_km,
            total_distance_km=total_km,
            total_hours=total_hours,
            payload_tons=payload_tons,
        )
        routes.append(ar)

        u = usage[route.agent_id]
        u.tasks.extend(task_ids)
        u.total_km += total_km
        u.total_hours += total_hours
        u.loaded_km += loaded_km

    overflow_km = 0
    overflow_h = 0
    for aid, u in usage.items():
        a = agent_by_id.get(aid)
        if a is None:
            continue
        if u.total_km > a.max_daily_km + 1e-9:
            overflow_km += 1
        if u.total_hours > a.max_shift_hours + 1e-9:
            overflow_h += 1
    checks = summarize_checks(
        unassigned_count=len(unassigned),
        overflow_km=overflow_km,
        overflow_hours=overflow_h,
    )
    details: dict[str, Any] = {
        "checks": checks,
        "legacy_method_label": getattr(result, "method_label", None),
        "legacy_transport_work_ton_km": getattr(result, "transport_work_ton_km", None),
        "task_mass_by_id": problem.task_mass_by_id,
    }
    if extra_details:
        details.update(extra_details)

    return EnrichedSolveResult(
        algorithm=algorithm_name,
        feasible=bool(checks["all_checks_ok"]),
        routes=routes,
        unassigned_task_ids=sorted(set(unassigned)),
        agent_usage=usage,
        runtime_sec=float(extra_details.get("runtime_sec", 0.0) if extra_details else 0.0),
        details=details,
    )


def solve_enriched_legacy_gap_vrp(
    *,
    dataset_path: Path | str,
    config: EnrichedLegacyGapConfig | None = None,
) -> EnrichedSolveResult:
    cfg = config or EnrichedLegacyGapConfig()
    problem = build_enriched_problem(dataset_path)
    raw_dataset, payload = load_dataset(dataset_path)
    payload = _ensure_full_agent_depots(payload)
    graph = core.build_nx_graph(raw_dataset)
    cache: dict[tuple[str, str], tuple[list[str], float] | None] = {}
    precomputed = PrecomputedDistanceOracle.from_dataset_payload(dataset_path=problem.dataset_path, payload=problem.payload)
    oracle = DistanceOracleWithFallback(nx_graph=graph, precomputed=precomputed)
    task_zone, agent_zone = _zones_from_payload(payload)

    t0 = time.perf_counter()
    with _patched_core(task_zone=task_zone, agent_zone=agent_zone, precomputed=precomputed):
        legacy = _solve_real_gap_vrp(
            dataset=raw_dataset,
            payload=payload,
            graph=graph,
            cache=cache,
            step1_method=cfg.step1_method,
            gap_iter=cfg.gap_iter,
            use_repair=cfg.use_repair,
            show_progress=cfg.show_progress,
            verbose=cfg.verbose,
            progress_hook=cfg.progress_hook,
        )
    elapsed = time.perf_counter() - t0
    return _convert_legacy_result(
        result=legacy,
        problem=problem,
        dataset=raw_dataset,
        oracle=oracle,
        algorithm_name="enriched_legacy_gap_vrp_v1",
        extra_details={"runtime_sec": elapsed, "legacy_mode": "patched_core_zone+precomputed", "step1_method": cfg.step1_method},
    )


def solve_enriched_legacy_milp(
    *,
    dataset_path: Path | str,
    config: EnrichedLegacyMILPConfig | None = None,
) -> EnrichedSolveResult:
    cfg = config or EnrichedLegacyMILPConfig()
    problem = build_enriched_problem(dataset_path)
    raw_dataset, payload = load_dataset(dataset_path)
    payload = _ensure_full_agent_depots(payload)
    graph = core.build_nx_graph(raw_dataset)
    cache: dict[tuple[str, str], tuple[list[str], float] | None] = {}
    precomputed = PrecomputedDistanceOracle.from_dataset_payload(dataset_path=problem.dataset_path, payload=problem.payload)
    oracle = DistanceOracleWithFallback(nx_graph=graph, precomputed=precomputed)
    task_zone, agent_zone = _zones_from_payload(payload)

    t0 = time.perf_counter()
    with _patched_core(task_zone=task_zone, agent_zone=agent_zone, precomputed=precomputed):
        legacy = _solve_real_milp(
            dataset=raw_dataset,
            payload=payload,
            graph=graph,
            cache=cache,
            time_limit_sec=cfg.time_limit_sec,
            unassigned_penalty=cfg.unassigned_penalty,
            show_progress=cfg.show_progress,
            progress_hook=cfg.progress_hook,
        )
    elapsed = time.perf_counter() - t0
    return _convert_legacy_result(
        result=legacy,
        problem=problem,
        dataset=raw_dataset,
        oracle=oracle,
        algorithm_name="enriched_legacy_milp_v1",
        extra_details={
            "runtime_sec": elapsed,
            "legacy_mode": "patched_core_zone+precomputed",
            "time_limit_sec": cfg.time_limit_sec,
            "unassigned_penalty": cfg.unassigned_penalty,
        },
    )
