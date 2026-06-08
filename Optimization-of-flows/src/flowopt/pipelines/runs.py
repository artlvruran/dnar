from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
import time
from types import SimpleNamespace
from typing import Any, Callable

import pandas as pd

from .. import core
from .. import genetic_solver_components as ga
from .. import milp_solver as milp
from ..backend.io import load_dataset, load_payload
from ..dataset import RoutingDataset
from ..solvers import dummy as dummy_solver
from ..solvers import gap_vrp as gap
from ..solvers import gap_vrp_alns_solver as real_gap_alns_solver
from ..solvers import gap_vrp_saa_solver as real_gap_saa_solver
from ..solvers import real_gap_vrp_solver as real_gap_solver
from ..solvers import real_genetic_solver as real_genetic_solver
from ..solvers import real_milp_solver as real_milp_solver
from ..solvers import real_stochastic_grasp_solver as real_stochastic_grasp_solver
from ..solvers import real_stochastic_rr_solver as real_stochastic_rr_solver


SRC_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = SRC_ROOT / "data"
SYNTHETIC_DATASET_PATH = DATA_ROOT / "synthetic" / "dataset_sandbox_type2.json"
REAL_DATASET_PATH = DATA_ROOT / "real" / "dataset_real_spb_ready.json"
REAL_SIMPLE_DATASET_PATH = DATA_ROOT / "real" / "real_simple" / "dataset_real_spb_simple.json"
REAL_FULL_29K_DATASET_PATH = DATA_ROOT / "real" / "full_29k" / "dataset_real_spb_full_29k.json"
REAL_CLEAN_FULL_ENRICHED_DATASET_PATH = (
    DATA_ROOT / "real" / "clean_full_enriched" / "dataset_real_spb_clean_full_enriched.json"
)
REAL_CLEAN_FULL_ENRICHED_SIMPLE_DATASET_PATH = (
    DATA_ROOT / "real" / "clean_full_enriched" / "simple" / "dataset_real_spb_clean_full_enriched_simple.json"
)


def _emit_progress(
    *,
    show_progress: bool,
    progress_hook: Callable[[str], None] | None,
    prefix: str,
    message: str,
) -> None:
    if progress_hook is not None:
        progress_hook(f"[{prefix}] {message}")
    elif show_progress:
        print(f"[{prefix}] {message}", flush=True)


def _prefixed_progress_hook(
    *,
    show_progress: bool,
    progress_hook: Callable[[str], None] | None,
    prefix: str,
) -> Callable[[str], None]:
    return lambda message: _emit_progress(
        show_progress=show_progress,
        progress_hook=progress_hook,
        prefix=prefix,
        message=message,
    )


@dataclass
class RunMetrics:
    algorithm: str
    dataset_path: str
    feasible: bool
    all_checks_ok: bool
    assigned_routes: int
    unassigned_tasks: int
    active_agents: int
    transport_work_ton_km: float | None
    total_km: float | None
    deadhead_km: float | None
    deadhead_share_pct: float | None
    total_hours: float | None
    runtime_sec: float
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "dataset_path": self.dataset_path,
            "feasible": self.feasible,
            "all_checks_ok": self.all_checks_ok,
            "assigned_routes": self.assigned_routes,
            "unassigned_tasks": self.unassigned_tasks,
            "active_agents": self.active_agents,
            "transport_work_ton_km": self.transport_work_ton_km,
            "total_km": self.total_km,
            "deadhead_km": self.deadhead_km,
            "deadhead_share_pct": self.deadhead_share_pct,
            "total_hours": self.total_hours,
            "runtime_sec": round(self.runtime_sec, 3),
            **self.details,
        }


def _aggregate_state_metrics(states: dict[str, core.AgentState]) -> dict[str, float | None]:
    active_states = [state for state in states.values() if state.task_ids]
    if not active_states:
        return {
            "total_km": None,
            "deadhead_km": None,
            "deadhead_share_pct": None,
            "total_hours": None,
        }
    total_km = sum(state.total_km for state in active_states)
    deadhead_km = sum(state.deadhead_km for state in active_states)
    total_hours = sum(state.total_hours for state in active_states)
    deadhead_share = (100.0 * deadhead_km / total_km) if total_km > 0 else None
    return {
        "total_km": round(total_km, 3),
        "deadhead_km": round(deadhead_km, 3),
        "deadhead_share_pct": round(deadhead_share, 3) if deadhead_share is not None else None,
        "total_hours": round(total_hours, 3),
    }


def _agent_solution_rows(
    *,
    dataset: RoutingDataset,
    routes,
    states: dict[str, core.AgentState],
) -> list[dict[str, Any]]:
    task_by_id = {task.task_id: task for task in dataset.tasks}
    route_count_by_agent: dict[str, int] = defaultdict(int)
    for route in routes:
        route_count_by_agent[route.agent_id] += 1

    rows: list[dict[str, Any]] = []
    for agent_id, state in states.items():
        if not state.task_ids:
            continue
        task_ids = list(state.task_ids)
        total_mass = sum(task_by_id[tid].mass_tons for tid in task_ids if tid in task_by_id)
        total_km = state.total_km
        deadhead_share = (100.0 * state.deadhead_km / total_km) if total_km > 0 else None
        km_limit = core.MAX_DAILY_KM_BY_TYPE.get(state.vehicle_type)
        hours_limit = core.MAX_SHIFT_HOURS_BY_TYPE.get(state.vehicle_type)
        rows.append(
            {
                "agent_id": agent_id,
                "vehicle_type": state.vehicle_type,
                "is_compact": state.is_compact,
                "capacity_tons": round(state.capacity_tons, 3),
                "depot_node": state.depot_node,
                "tasks_count": len(task_ids),
                "routes_count": route_count_by_agent.get(agent_id, 0),
                "total_mass_tons": round(total_mass, 3),
                "total_km": round(total_km, 3),
                "task_km": round(state.task_km, 3),
                "deadhead_km": round(state.deadhead_km, 3),
                "deadhead_share_pct": round(deadhead_share, 3) if deadhead_share is not None else None,
                "total_hours": round(state.total_hours, 3),
                "drive_hours": round(state.drive_hours, 3),
                "service_hours": round(state.service_hours, 3),
                "true_transport_work_ton_km": round(state.true_transport_work_ton_km, 3),
                "peak_load_tons": round(state.peak_load_tons, 3),
                "movement_legs_count": len(state.movement_legs),
                "km_limit": km_limit,
                "hours_limit": hours_limit,
                "task_ids": task_ids,
                "route_ids": list(state.route_ids),
                "stop_sequence": [
                    {
                        "node_id": event.node_id,
                        "action": event.action,
                        "task_ids": list(event.task_ids),
                        "load_after_tons": event.load_after_tons,
                    }
                    for event in state.stop_events
                ],
                "movement_legs": [
                    {
                        "from_node_id": leg.from_node_id,
                        "to_node_id": leg.to_node_id,
                        "distance_km": round(leg.distance_km, 3),
                        "loaded_mass_tons": leg.loaded_mass_tons,
                        "carrying_task_ids": list(leg.carrying_task_ids),
                    }
                    for leg in state.movement_legs
                ],
            }
        )

    rows.sort(key=lambda row: (-row["tasks_count"], row["total_km"], row["agent_id"]))
    return rows


def _rebuild_state_plans_for_routes(
    *,
    dataset: RoutingDataset,
    states: dict[str, core.AgentState],
    graph,
    cache,
    routes,
) -> None:
    task_by_id = {task.task_id: task for task in dataset.tasks}
    tasks_by_agent: dict[str, list[Any]] = {}
    for route in routes:
        agent_tasks = tasks_by_agent.setdefault(route.agent_id, [])
        for task_id in route.task_ids:
            if task_id in task_by_id:
                agent_tasks.append(task_by_id[task_id])

    for agent_id, tasks in tasks_by_agent.items():
        state = states.get(agent_id)
        if state is None:
            continue
        plan = core.evaluate_agent_task_set(state, tasks, graph, cache)
        if plan is None:
            continue
        core.apply_plan_to_state(state, plan)


def _core_checks(
    dataset: RoutingDataset,
    graph,
    cache,
    routes,
    limit_violations: list[dict[str, Any]],
    unassigned: list[str],
) -> tuple[bool, dict[str, Any]]:
    reachability = core.check_reachability(graph, dataset, cache)
    route_task_ids = {task_id for route in routes for task_id in route.task_ids}
    base_task_ids = {task.task_id for task in dataset.tasks}
    same_task_space = route_task_ids.issubset(base_task_ids)

    solved_dataset_valid = False
    if not unassigned and same_task_space:
        try:
            _ = core.build_solution_dataset(dataset, routes)
            solved_dataset_valid = True
        except Exception:
            solved_dataset_valid = False

    if same_task_space:
        mno_coverage = core.mno_scope_coverage_summary(dataset, routes)
    else:
        mno_coverage = {
            "all_eligible_mno_covered": len(unassigned) == 0,
            "eligible_mno_count": None,
            "covered_eligible_mno_count": None,
        }

    checks = core.constraints_check_summary(
        solved_dataset_valid=solved_dataset_valid,
        limit_violations=limit_violations,
        reachability=reachability,
        unassigned=unassigned,
        mno_coverage=mno_coverage,
    )
    checks["metric_task_space"] = "dataset_reference" if same_task_space else "solver_generated_mismatch"
    checks["task_space_match"] = bool(same_task_space)
    return bool(checks.get("all_checks_ok", False)), checks


def _metrics_from_solver_result(
    *,
    algorithm: str,
    dataset_path: Path | str,
    dataset: RoutingDataset,
    graph,
    cache,
    result,
    elapsed: float,
    details: dict[str, Any] | None = None,
) -> RunMetrics:
    central_metrics = core.evaluate_solution_metrics(
        dataset=dataset,
        routes=result.routes,
        unassigned=result.unassigned,
    )
    task_space_match = bool(central_metrics.get("task_space_match", False))
    if task_space_match:
        _rebuild_state_plans_for_routes(
            dataset=dataset,
            states=result.states,
            graph=graph,
            cache=cache,
            routes=result.routes,
        )

    all_checks_ok, checks = _core_checks(
        dataset=dataset,
        graph=graph,
        cache=cache,
        routes=result.routes,
        limit_violations=result.limit_violations,
        unassigned=result.unassigned,
    )
    state_metrics = _aggregate_state_metrics(result.states)
    merged_details = {
        "checks": checks,
        **central_metrics,
        "agent_solution_rows": _agent_solution_rows(
            dataset=dataset,
            routes=result.routes,
            states=result.states,
        ),
    }
    if details:
        merged_details.update(details)
    normalized_feasible = bool(result.feasible and all_checks_ok)

    return RunMetrics(
        algorithm=algorithm,
        dataset_path=str(Path(dataset_path)),
        feasible=normalized_feasible,
        all_checks_ok=all_checks_ok,
        assigned_routes=result.n_assigned,
        unassigned_tasks=result.n_unassigned,
        active_agents=result.active_agents,
        transport_work_ton_km=central_metrics.get("TR"),
        total_km=state_metrics["total_km"],
        deadhead_km=state_metrics["deadhead_km"],
        deadhead_share_pct=state_metrics["deadhead_share_pct"],
        total_hours=state_metrics["total_hours"],
        runtime_sec=elapsed,
        details=merged_details,
    )


def _error_metrics(
    *,
    algorithm: str,
    dataset_path: Path | str,
    dataset: RoutingDataset,
    graph,
    cache,
    elapsed: float,
    solver_error: str,
    details: dict[str, Any] | None = None,
) -> RunMetrics:
    reachability = core.check_reachability(graph, dataset, cache)
    merged_details = {
        "solver_error": solver_error,
        "checks": {
            "hard_constraints_ok": False,
            "daily_limits_ok": False,
            "reachability_ok": len(reachability.get("unreachable_tasks", [])) == 0,
            "all_tasks_assigned": False,
            "mno_coverage_ok": False,
            "all_checks_ok": False,
        },
    }
    if details:
        merged_details.update(details)
    return RunMetrics(
        algorithm=algorithm,
        dataset_path=str(Path(dataset_path)),
        feasible=False,
        all_checks_ok=False,
        assigned_routes=0,
        unassigned_tasks=len(dataset.tasks),
        active_agents=0,
        transport_work_ton_km=None,
        total_km=None,
        deadhead_km=None,
        deadhead_share_pct=None,
        total_hours=None,
        runtime_sec=elapsed,
        details=merged_details,
    )


def run_gap_vrp(
    *,
    dataset_path: Path | str = REAL_DATASET_PATH,
    step1_method: str = "lp",
    gap_iter: int = 120,
    use_repair: bool = True,
    max_agents: int | None = None,
    show_progress: bool = False,
    verbose: bool = False,
    progress_hook: Callable[[str], None] | None = None,
) -> RunMetrics:
    emit = _prefixed_progress_hook(
        show_progress=show_progress,
        progress_hook=progress_hook,
        prefix="gap_vrp",
    )
    emit(f"load dataset: {dataset_path}")
    t_load = time.perf_counter()
    dataset, payload = load_dataset(dataset_path)
    emit(
        f"dataset loaded in {time.perf_counter() - t_load:.1f}s "
        f"(nodes={len(dataset.graph.nodes)}, edges={len(dataset.graph.edges)}, "
        f"agents={len(dataset.fleet.agents)}, tasks={len(dataset.tasks)})"
    )

    if max_agents is not None:
        emit(f"select agents subset: max_agents={max_agents}")
        dataset, payload = gap.select_agents_subset(dataset, payload, max_agents=max_agents)
        emit(f"active agents after subset: {len(dataset.fleet.agents)}")

    t_graph = time.perf_counter()
    graph = core.build_nx_graph(dataset)
    emit(f"graph built in {time.perf_counter() - t_graph:.1f}s")
    cache: dict[tuple[str, str], tuple[list[str], float] | None] = {}

    t0 = time.perf_counter()
    emit("solver start")
    result = gap.run_gap_vrp(
        dataset=dataset,
        payload=payload,
        graph=graph,
        cache=cache,
        step1_method=step1_method,
        gap_iter=gap_iter,
        use_repair=use_repair,
        verbose=verbose,
        show_progress=show_progress,
        progress_hook=emit,
    )
    elapsed = time.perf_counter() - t0
    emit(f"solver finished in {elapsed:.1f}s")
    return _metrics_from_solver_result(
        algorithm="gap_vrp",
        dataset_path=dataset_path,
        dataset=dataset,
        graph=graph,
        cache=cache,
        result=result,
        elapsed=elapsed,
        details={
            "step1_method": step1_method,
            "gap_iter": gap_iter,
            "used_agents": len(dataset.fleet.agents),
        },
    )


def run_dummy(
    *,
    dataset_path: Path | str = REAL_SIMPLE_DATASET_PATH,
) -> RunMetrics:
    dataset, payload = load_dataset(dataset_path)
    graph = core.build_nx_graph(dataset)
    cache: dict[tuple[str, str], tuple[list[str], float] | None] = {}

    t0 = time.perf_counter()
    result = dummy_solver.run_dummy_solver(
        dataset=dataset,
        payload=payload,
        graph=graph,
        cache=cache,
    )
    elapsed = time.perf_counter() - t0
    return _metrics_from_solver_result(
        algorithm="dummy",
        dataset_path=dataset_path,
        dataset=dataset,
        graph=graph,
        cache=cache,
        result=result,
        elapsed=elapsed,
    )


def run_real_gap_vrp(
    *,
    dataset_path: Path | str = REAL_SIMPLE_DATASET_PATH,
    step1_method: str = "lp",
    gap_iter: int = 120,
    use_repair: bool = True,
    max_agents: int | None = None,
    show_progress: bool = False,
    verbose: bool = False,
    progress_hook: Callable[[str], None] | None = None,
) -> RunMetrics:
    emit = _prefixed_progress_hook(
        show_progress=show_progress,
        progress_hook=progress_hook,
        prefix="real_gap_vrp",
    )
    emit(f"load dataset: {dataset_path}")
    dataset, payload = load_dataset(dataset_path)
    emit(
        f"dataset loaded (nodes={len(dataset.graph.nodes)}, edges={len(dataset.graph.edges)}, "
        f"agents={len(dataset.fleet.agents)}, tasks={len(dataset.tasks)})"
    )
    if max_agents is not None:
        emit(f"select agents subset: max_agents={max_agents}")
        dataset, payload = gap.select_agents_subset(dataset, payload, max_agents=max_agents)
        emit(f"active agents after subset: {len(dataset.fleet.agents)}")
    emit("build nx graph")
    graph = core.build_nx_graph(dataset)
    cache: dict[tuple[str, str], tuple[list[str], float] | None] = {}

    t0 = time.perf_counter()
    try:
        emit("solver start")
        result = real_gap_solver.solve_real_gap_vrp(
            dataset=dataset,
            payload=payload,
            graph=graph,
            cache=cache,
            step1_method=step1_method,
            gap_iter=gap_iter,
            use_repair=use_repair,
            show_progress=show_progress,
            verbose=verbose,
            progress_hook=emit,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        emit(f"solver error after {elapsed:.1f}s: {type(exc).__name__}")
        return _error_metrics(
            algorithm="real_gap_vrp",
            dataset_path=dataset_path,
            dataset=dataset,
            graph=graph,
            cache=cache,
            elapsed=elapsed,
            solver_error=f"{type(exc).__name__}: {exc}",
            details={
                "step1_method": step1_method,
                "gap_iter": gap_iter,
                "used_agents": len(dataset.fleet.agents),
            },
        )
    elapsed = time.perf_counter() - t0
    emit(f"solver done in {elapsed:.1f}s")
    return _metrics_from_solver_result(
        algorithm="real_gap_vrp",
        dataset_path=dataset_path,
        dataset=dataset,
        graph=graph,
        cache=cache,
        result=result,
        elapsed=elapsed,
        details={
            "step1_method": step1_method,
            "gap_iter": gap_iter,
            "used_agents": len(dataset.fleet.agents),
        },
    )


def run_real_gap_vrp_alns(
    *,
    dataset_path: Path | str = REAL_SIMPLE_DATASET_PATH,
    step1_method: str = "lp",
    gap_iter: int = 120,
    use_repair: bool = True,
    max_agents: int | None = None,
    alns_iterations: int = 150,
    alns_removal_q: int = 30,
    alns_seed: int = 42,
    alns_log_every: int = 10,
    start_temperature: float = 1000.0,
    end_temperature: float = 1.0,
    temperature_step: float = 0.999,
    show_progress: bool = False,
    verbose: bool = False,
    progress_hook: Callable[[str], None] | None = None,
) -> RunMetrics:
    emit = _prefixed_progress_hook(
        show_progress=show_progress,
        progress_hook=progress_hook,
        prefix="real_gap_vrp_alns",
    )
    emit(f"load dataset: {dataset_path}")
    dataset, payload = load_dataset(dataset_path)
    emit(
        f"dataset loaded (nodes={len(dataset.graph.nodes)}, edges={len(dataset.graph.edges)}, "
        f"agents={len(dataset.fleet.agents)}, tasks={len(dataset.tasks)})"
    )
    if max_agents is not None:
        emit(f"select agents subset: max_agents={max_agents}")
        dataset, payload = gap.select_agents_subset(dataset, payload, max_agents=max_agents)
        emit(f"active agents after subset: {len(dataset.fleet.agents)}")
    emit("build nx graph")
    graph = core.build_nx_graph(dataset)
    cache: dict[tuple[str, str], tuple[list[str], float] | None] = {}

    t0 = time.perf_counter()
    try:
        emit("solver start")
        alns_solution = real_gap_alns_solver.solve_real_gap_vrp_alns(
            dataset=dataset,
            payload=payload,
            graph=graph,
            cache=cache,
            step1_method=step1_method,
            gap_iter=gap_iter,
            use_repair=use_repair,
            alns_iterations=alns_iterations,
            alns_removal_q=alns_removal_q,
            alns_seed=alns_seed,
            alns_log_every=alns_log_every,
            start_temperature=start_temperature,
            end_temperature=end_temperature,
            temperature_step=temperature_step,
            show_progress=show_progress,
            verbose=verbose,
            progress_hook=emit,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        emit(f"solver error after {elapsed:.1f}s: {type(exc).__name__}")
        return _error_metrics(
            algorithm="real_gap_vrp_alns",
            dataset_path=dataset_path,
            dataset=dataset,
            graph=graph,
            cache=cache,
            elapsed=elapsed,
            solver_error=f"{type(exc).__name__}: {exc}",
            details={
                "step1_method": step1_method,
                "gap_iter": gap_iter,
                "used_agents": len(dataset.fleet.agents),
                "alns_iterations": alns_iterations,
                "alns_removal_q": alns_removal_q,
                "alns_seed": alns_seed,
                "alns_log_every": alns_log_every,
            },
        )
    elapsed = time.perf_counter() - t0
    emit(f"solver done in {elapsed:.1f}s")
    return _metrics_from_solver_result(
        algorithm="real_gap_vrp_alns",
        dataset_path=dataset_path,
        dataset=dataset,
        graph=graph,
        cache=cache,
        result=alns_solution.result,
        elapsed=elapsed,
        details={
            "step1_method": step1_method,
            "gap_iter": gap_iter,
            "used_agents": len(dataset.fleet.agents),
            "alns_iterations": alns_iterations,
            "alns_removal_q": alns_removal_q,
            "alns_seed": alns_seed,
            "alns_log_every": alns_log_every,
            "alns_base_objective": round(alns_solution.base_objective, 3),
            "alns_best_objective": round(alns_solution.best_objective, 3),
            "alns_gain": round(alns_solution.base_objective - alns_solution.best_objective, 3),
        },
    )


def run_real_gap_vrp_saa(
    *,
    dataset_path: Path | str = REAL_SIMPLE_DATASET_PATH,
    step1_method: str = "lp",
    gap_iter: int = 120,
    use_repair: bool = True,
    max_agents: int | None = None,
    saa_sample_size: int = 20,
    saa_iterations: int = 100,
    saa_route_max_size: int = 15,
    saa_seed: int = 42,
    saa_log_every: int = 10,
    show_progress: bool = False,
    verbose: bool = False,
    progress_hook: Callable[[str], None] | None = None,
) -> RunMetrics:
    emit = _prefixed_progress_hook(
        show_progress=show_progress,
        progress_hook=progress_hook,
        prefix="real_gap_vrp_saa",
    )
    emit(f"load dataset: {dataset_path}")
    dataset, payload = load_dataset(dataset_path)
    emit(
        f"dataset loaded (nodes={len(dataset.graph.nodes)}, edges={len(dataset.graph.edges)}, "
        f"agents={len(dataset.fleet.agents)}, tasks={len(dataset.tasks)})"
    )
    if max_agents is not None:
        emit(f"select agents subset: max_agents={max_agents}")
        dataset, payload = gap.select_agents_subset(dataset, payload, max_agents=max_agents)
        emit(f"active agents after subset: {len(dataset.fleet.agents)}")
    emit("build nx graph")
    graph = core.build_nx_graph(dataset)
    cache: dict[tuple[str, str], tuple[list[str], float] | None] = {}

    t0 = time.perf_counter()
    try:
        emit("solver start")
        saa_solution = real_gap_saa_solver.solve_real_gap_vrp_saa(
            dataset=dataset,
            payload=payload,
            graph=graph,
            cache=cache,
            step1_method=step1_method,
            gap_iter=gap_iter,
            use_repair=use_repair,
            saa_sample_size=saa_sample_size,
            saa_iterations=saa_iterations,
            saa_route_max_size=saa_route_max_size,
            saa_seed=saa_seed,
            saa_log_every=saa_log_every,
            show_progress=show_progress,
            verbose=verbose,
            progress_hook=emit,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        emit(f"solver error after {elapsed:.1f}s: {type(exc).__name__}")
        return _error_metrics(
            algorithm="real_gap_vrp_saa",
            dataset_path=dataset_path,
            dataset=dataset,
            graph=graph,
            cache=cache,
            elapsed=elapsed,
            solver_error=f"{type(exc).__name__}: {exc}",
            details={
                "step1_method": step1_method,
                "gap_iter": gap_iter,
                "used_agents": len(dataset.fleet.agents),
                "saa_sample_size": saa_sample_size,
                "saa_iterations": saa_iterations,
                "saa_route_max_size": saa_route_max_size,
                "saa_seed": saa_seed,
                "saa_log_every": saa_log_every,
            },
        )
    elapsed = time.perf_counter() - t0
    emit(f"solver done in {elapsed:.1f}s")
    return _metrics_from_solver_result(
        algorithm="real_gap_vrp_saa",
        dataset_path=dataset_path,
        dataset=dataset,
        graph=graph,
        cache=cache,
        result=saa_solution.result,
        elapsed=elapsed,
        details={
            "step1_method": step1_method,
            "gap_iter": gap_iter,
            "used_agents": len(dataset.fleet.agents),
            "saa_sample_size": saa_sample_size,
            "saa_iterations": saa_iterations,
            "saa_route_max_size": saa_route_max_size,
            "saa_seed": saa_seed,
            "saa_log_every": saa_log_every,
            "saa_route_pool_size": saa_solution.route_pool_size,
            "saa_assembled_route_count": saa_solution.assembled_route_count,
        },
    )


def run_real_milp(
    *,
    dataset_path: Path | str = REAL_SIMPLE_DATASET_PATH,
    time_limit_sec: int = 60,
    unassigned_penalty: float = 1e5,
    show_progress: bool = False,
    progress_hook: Callable[[str], None] | None = None,
) -> RunMetrics:
    emit = _prefixed_progress_hook(
        show_progress=show_progress,
        progress_hook=progress_hook,
        prefix="real_milp",
    )
    emit(f"load dataset: {dataset_path}")
    dataset, payload = load_dataset(dataset_path)
    emit(
        f"dataset loaded (nodes={len(dataset.graph.nodes)}, edges={len(dataset.graph.edges)}, "
        f"agents={len(dataset.fleet.agents)}, tasks={len(dataset.tasks)})"
    )
    emit("build nx graph")
    graph = core.build_nx_graph(dataset)
    cache: dict[tuple[str, str], tuple[list[str], float] | None] = {}

    t0 = time.perf_counter()
    try:
        emit("solver start")
        result = real_milp_solver.solve_real_milp(
            dataset=dataset,
            payload=payload,
            graph=graph,
            cache=cache,
            time_limit_sec=time_limit_sec,
            unassigned_penalty=unassigned_penalty,
            show_progress=show_progress,
            progress_hook=emit,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        emit(f"solver error after {elapsed:.1f}s: {type(exc).__name__}")
        return _error_metrics(
            algorithm="real_milp",
            dataset_path=dataset_path,
            dataset=dataset,
            graph=graph,
            cache=cache,
            elapsed=elapsed,
            solver_error=f"{type(exc).__name__}: {exc}",
            details={
                "time_limit_sec": time_limit_sec,
                "unassigned_penalty": unassigned_penalty,
            },
        )
    elapsed = time.perf_counter() - t0
    emit(f"solver done in {elapsed:.1f}s")
    return _metrics_from_solver_result(
        algorithm="real_milp",
        dataset_path=dataset_path,
        dataset=dataset,
        graph=graph,
        cache=cache,
        result=result,
        elapsed=elapsed,
        details={
            "time_limit_sec": time_limit_sec,
            "unassigned_penalty": unassigned_penalty,
        },
    )


def run_real_genetic(
    *,
    dataset_path: Path | str = REAL_SIMPLE_DATASET_PATH,
    population_size: int = 60,
    generations: int = 120,
    elite_size: int = 4,
    seed: int | None = 42,
    generation_scale: float = 0.1,
    min_generations: int = 3,
    max_runtime_sec: float | None = None,
    show_progress: bool = False,
    progress_hook: Callable[[str], None] | None = None,
) -> RunMetrics:
    emit = _prefixed_progress_hook(
        show_progress=show_progress,
        progress_hook=progress_hook,
        prefix="real_genetic",
    )
    emit(f"load dataset: {dataset_path}")
    dataset, payload = load_dataset(dataset_path)
    emit(
        f"dataset loaded (nodes={len(dataset.graph.nodes)}, edges={len(dataset.graph.edges)}, "
        f"agents={len(dataset.fleet.agents)}, tasks={len(dataset.tasks)})"
    )
    effective_generations = generations
    if generation_scale > 0:
        effective_generations = max(min_generations, int(round(generations * generation_scale)))
    if effective_generations != generations:
        emit(
            f"generation downscale applied: requested={generations}, "
            f"effective={effective_generations}, scale={generation_scale}"
        )
    emit("build nx graph")
    graph = core.build_nx_graph(dataset)
    cache: dict[tuple[str, str], tuple[list[str], float] | None] = {}

    t0 = time.perf_counter()
    try:
        emit("solver start")
        result = real_genetic_solver.solve_real_genetic(
            dataset=dataset,
            payload=payload,
            graph=graph,
            cache=cache,
            population_size=population_size,
            generations=effective_generations,
            elite_size=elite_size,
            seed=seed,
            max_runtime_sec=max_runtime_sec,
            show_progress=show_progress,
            progress_hook=emit,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        emit(f"solver error after {elapsed:.1f}s: {type(exc).__name__}")
        return _error_metrics(
            algorithm="real_genetic",
            dataset_path=dataset_path,
            dataset=dataset,
            graph=graph,
            cache=cache,
            elapsed=elapsed,
            solver_error=f"{type(exc).__name__}: {exc}",
            details={
                "population_size": population_size,
                "generations": effective_generations,
                "generations_requested": generations,
                "generation_scale": generation_scale,
                "elite_size": elite_size,
                "max_runtime_sec": max_runtime_sec,
            },
        )
    elapsed = time.perf_counter() - t0
    emit(f"solver done in {elapsed:.1f}s")
    return _metrics_from_solver_result(
        algorithm="real_genetic",
        dataset_path=dataset_path,
        dataset=dataset,
        graph=graph,
        cache=cache,
        result=result,
        elapsed=elapsed,
        details={
            "population_size": population_size,
            "generations": effective_generations,
            "generations_requested": generations,
            "generation_scale": generation_scale,
            "elite_size": elite_size,
            "max_runtime_sec": max_runtime_sec,
        },
    )


def run_real_stochastic_grasp(
    *,
    dataset_path: Path | str = REAL_FULL_29K_DATASET_PATH,
    time_budget_sec: int = 180,
    max_starts: int = 2,
    candidate_k: int = 20,
    rcl_size: int = 5,
    load_penalty: float = 8.0,
    road_factor: float = 1.25,
    trip_share: float = 0.35,
    overload_slack: float = 1.02,
    seed: int | None = 42,
    show_progress: bool = False,
    progress_hook: Callable[[str], None] | None = None,
) -> RunMetrics:
    emit = _prefixed_progress_hook(
        show_progress=show_progress,
        progress_hook=progress_hook,
        prefix="real_stochastic_grasp",
    )
    emit(f"load dataset: {dataset_path}")
    dataset, payload = load_dataset(dataset_path)
    emit(
        f"dataset loaded (nodes={len(dataset.graph.nodes)}, edges={len(dataset.graph.edges)}, "
        f"agents={len(dataset.fleet.agents)}, tasks={len(dataset.tasks)})"
    )
    emit("build nx graph")
    graph = core.build_nx_graph(dataset)
    cache: dict[tuple[str, str], tuple[list[str], float] | None] = {}

    t0 = time.perf_counter()
    try:
        emit("solver start")
        result = real_stochastic_grasp_solver.solve_real_stochastic_grasp(
            dataset=dataset,
            payload=payload,
            graph=graph,
            cache=cache,
            time_budget_sec=time_budget_sec,
            max_starts=max_starts,
            candidate_k=candidate_k,
            rcl_size=rcl_size,
            load_penalty=load_penalty,
            road_factor=road_factor,
            trip_share=trip_share,
            overload_slack=overload_slack,
            seed=seed,
            show_progress=show_progress,
            progress_hook=emit,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        emit(f"solver error after {elapsed:.1f}s: {type(exc).__name__}")
        return _error_metrics(
            algorithm="real_stochastic_grasp",
            dataset_path=dataset_path,
            dataset=dataset,
            graph=graph,
            cache=cache,
            elapsed=elapsed,
            solver_error=f"{type(exc).__name__}: {exc}",
            details={
                "time_budget_sec": time_budget_sec,
                "max_starts": max_starts,
                "candidate_k": candidate_k,
                "rcl_size": rcl_size,
            },
        )
    elapsed = time.perf_counter() - t0
    emit(f"solver done in {elapsed:.1f}s")
    return _metrics_from_solver_result(
        algorithm="real_stochastic_grasp",
        dataset_path=dataset_path,
        dataset=dataset,
        graph=graph,
        cache=cache,
        result=result,
        elapsed=elapsed,
        details={
            "time_budget_sec": time_budget_sec,
            "max_starts": max_starts,
            "candidate_k": candidate_k,
            "rcl_size": rcl_size,
            "load_penalty": load_penalty,
            "road_factor": road_factor,
            "trip_share": trip_share,
            "overload_slack": overload_slack,
            "seed": seed,
        },
    )


def run_real_stochastic_rr(
    *,
    dataset_path: Path | str = REAL_FULL_29K_DATASET_PATH,
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
) -> RunMetrics:
    emit = _prefixed_progress_hook(
        show_progress=show_progress,
        progress_hook=progress_hook,
        prefix="real_stochastic_rr",
    )
    emit(f"load dataset: {dataset_path}")
    dataset, payload = load_dataset(dataset_path)
    emit(
        f"dataset loaded (nodes={len(dataset.graph.nodes)}, edges={len(dataset.graph.edges)}, "
        f"agents={len(dataset.fleet.agents)}, tasks={len(dataset.tasks)})"
    )
    emit("build nx graph")
    graph = core.build_nx_graph(dataset)
    cache: dict[tuple[str, str], tuple[list[str], float] | None] = {}

    t0 = time.perf_counter()
    try:
        emit("solver start")
        result = real_stochastic_rr_solver.solve_real_stochastic_rr(
            dataset=dataset,
            payload=payload,
            graph=graph,
            cache=cache,
            time_budget_sec=time_budget_sec,
            candidate_k=candidate_k,
            rcl_size=rcl_size,
            load_penalty=load_penalty,
            road_factor=road_factor,
            trip_share=trip_share,
            overload_slack=overload_slack,
            batch_size=batch_size,
            seed=seed,
            show_progress=show_progress,
            progress_hook=emit,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        emit(f"solver error after {elapsed:.1f}s: {type(exc).__name__}")
        return _error_metrics(
            algorithm="real_stochastic_rr",
            dataset_path=dataset_path,
            dataset=dataset,
            graph=graph,
            cache=cache,
            elapsed=elapsed,
            solver_error=f"{type(exc).__name__}: {exc}",
            details={
                "time_budget_sec": time_budget_sec,
                "candidate_k": candidate_k,
                "rcl_size": rcl_size,
                "batch_size": batch_size,
            },
        )
    elapsed = time.perf_counter() - t0
    emit(f"solver done in {elapsed:.1f}s")
    return _metrics_from_solver_result(
        algorithm="real_stochastic_rr",
        dataset_path=dataset_path,
        dataset=dataset,
        graph=graph,
        cache=cache,
        result=result,
        elapsed=elapsed,
        details={
            "time_budget_sec": time_budget_sec,
            "candidate_k": candidate_k,
            "rcl_size": rcl_size,
            "batch_size": batch_size,
            "load_penalty": load_penalty,
            "road_factor": road_factor,
            "trip_share": trip_share,
            "overload_slack": overload_slack,
            "seed": seed,
        },
    )


def run_milp(
    *,
    dataset_path: Path | str = SYNTHETIC_DATASET_PATH,
    show_progress: bool = False,
    progress_hook: Callable[[str], None] | None = None,
) -> RunMetrics:
    emit = _prefixed_progress_hook(
        show_progress=show_progress,
        progress_hook=progress_hook,
        prefix="milp",
    )
    emit(f"load dataset: {dataset_path}")
    dataset, payload = load_dataset(dataset_path)
    emit(
        f"dataset loaded (nodes={len(dataset.graph.nodes)}, edges={len(dataset.graph.edges)}, "
        f"agents={len(dataset.fleet.agents)}, tasks={len(dataset.tasks)})"
    )
    graph = core.build_nx_graph(dataset)
    cache: dict[tuple[str, str], tuple[list[str], float] | None] = {}
    reachability = core.check_reachability(graph, dataset, cache)

    t0 = time.perf_counter()
    try:
        emit("solver start")
        routes, states, unassigned = milp.assign_tasks_greedy(
            dataset,
            graph,
            payload,
            cache,
            show_progress=show_progress,
            progress_hook=emit,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        emit(f"solver error after {elapsed:.1f}s: {type(exc).__name__}")
        return RunMetrics(
            algorithm="milp",
            dataset_path=str(Path(dataset_path)),
            feasible=False,
            all_checks_ok=False,
            assigned_routes=0,
            unassigned_tasks=len(dataset.tasks),
            active_agents=0,
            transport_work_ton_km=None,
            total_km=None,
            deadhead_km=None,
            deadhead_share_pct=None,
            total_hours=None,
            runtime_sec=elapsed,
            details={
                "solver_error": f"{type(exc).__name__}: {exc}",
                "checks": {
                    "hard_constraints_ok": False,
                    "daily_limits_ok": False,
                    "reachability_ok": len(reachability.get("unreachable_tasks", [])) == 0,
                    "all_tasks_assigned": False,
                    "mno_coverage_ok": False,
                    "all_checks_ok": False,
                },
            },
        )
    elapsed = time.perf_counter() - t0
    emit(f"solver done in {elapsed:.1f}s")
    limit_violations = core.validate_daily_limits(states)
    result = SimpleNamespace(
        routes=routes,
        states=states,
        unassigned=unassigned,
        feasible=(len(unassigned) == 0 and len(limit_violations) == 0),
        limit_violations=limit_violations,
        n_assigned=len(routes),
        n_unassigned=len(unassigned),
        active_agents=sum(1 for state in states.values() if state.task_ids),
    )
    return _metrics_from_solver_result(
        algorithm="milp",
        dataset_path=dataset_path,
        dataset=dataset,
        graph=graph,
        cache=cache,
        result=result,
        elapsed=elapsed,
        details={
            "reachability_summary": reachability,
        },
    )


def run_genetic(
    *,
    dataset_path: Path | str = SYNTHETIC_DATASET_PATH,
    population_size: int = 60,
    generations: int = 120,
    elite_size: int = 4,
    seed: int | None = 42,
    show_progress: bool = False,
    progress_hook: Callable[[str], None] | None = None,
) -> RunMetrics:
    emit = _prefixed_progress_hook(
        show_progress=show_progress,
        progress_hook=progress_hook,
        prefix="genetic",
    )
    emit(f"load dataset: {dataset_path}")
    dataset, payload = load_dataset(dataset_path)
    emit(
        f"dataset loaded (nodes={len(dataset.graph.nodes)}, edges={len(dataset.graph.edges)}, "
        f"agents={len(dataset.fleet.agents)}, tasks={len(dataset.tasks)})"
    )
    graph = ga.build_nx_graph(dataset)
    cache: dict[tuple[str, str], tuple[list[str], float] | None] = {}
    reachability = ga.check_reachability(graph, dataset, cache)

    t0 = time.perf_counter()
    try:
        emit("solver start")
        routes, states, unassigned = ga.assign_tasks_genetic(
            dataset,
            graph,
            payload,
            cache,
            population_size=population_size,
            generations=generations,
            elite_size=elite_size,
            seed=seed,
            show_progress=show_progress,
            progress_hook=emit,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        emit(f"solver error after {elapsed:.1f}s: {type(exc).__name__}")
        return RunMetrics(
            algorithm="genetic",
            dataset_path=str(Path(dataset_path)),
            feasible=False,
            all_checks_ok=False,
            assigned_routes=0,
            unassigned_tasks=len(dataset.tasks),
            active_agents=0,
            transport_work_ton_km=None,
            total_km=None,
            deadhead_km=None,
            deadhead_share_pct=None,
            total_hours=None,
            runtime_sec=elapsed,
            details={
                "population_size": population_size,
                "generations": generations,
                "elite_size": elite_size,
                "solver_error": f"{type(exc).__name__}: {exc}",
                "checks": {
                    "hard_constraints_ok": False,
                    "daily_limits_ok": False,
                    "reachability_ok": len(reachability.get("unreachable_tasks", [])) == 0,
                    "all_tasks_assigned": False,
                    "mno_coverage_ok": False,
                    "all_checks_ok": False,
                },
            },
        )
    elapsed = time.perf_counter() - t0
    emit(f"solver done in {elapsed:.1f}s")
    limit_violations = ga.validate_daily_limits(states)
    result = SimpleNamespace(
        routes=routes,
        states=states,
        unassigned=unassigned,
        feasible=(len(unassigned) == 0 and len(limit_violations) == 0),
        limit_violations=limit_violations,
        n_assigned=len(routes),
        n_unassigned=len(unassigned),
        active_agents=sum(1 for state in states.values() if state.task_ids),
    )
    return _metrics_from_solver_result(
        algorithm="genetic",
        dataset_path=dataset_path,
        dataset=dataset,
        graph=graph,
        cache=cache,
        result=result,
        elapsed=elapsed,
        details={
            "population_size": population_size,
            "generations": generations,
            "elite_size": elite_size,
            "reachability_summary": reachability,
        },
    )


def benchmark_synthetic(
    *,
    dataset_path: Path | str = SYNTHETIC_DATASET_PATH,
    gap_step1_method: str = "dataset",
    gap_iter: int = 120,
    ga_population_size: int = 60,
    ga_generations: int = 120,
    ga_elite_size: int = 4,
    ga_seed: int | None = 42,
    show_progress: bool = False,
    progress_hook: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    emit = _prefixed_progress_hook(
        show_progress=show_progress,
        progress_hook=progress_hook,
        prefix="benchmark_synthetic",
    )
    emit("run gap_vrp")
    gap_metrics = run_gap_vrp(
        dataset_path=dataset_path,
        step1_method=gap_step1_method,
        gap_iter=gap_iter,
        use_repair=True,
        verbose=False,
        show_progress=show_progress,
        progress_hook=progress_hook,
    )
    emit("run milp")
    milp_metrics = run_milp(
        dataset_path=dataset_path,
        show_progress=show_progress,
        progress_hook=progress_hook,
    )
    emit("run genetic")
    ga_metrics = run_genetic(
        dataset_path=dataset_path,
        population_size=ga_population_size,
        generations=ga_generations,
        elite_size=ga_elite_size,
        seed=ga_seed,
        show_progress=show_progress,
        progress_hook=progress_hook,
    )
    results = [gap_metrics, milp_metrics, ga_metrics]
    emit("all algorithms finished")

    table = pd.DataFrame([result.as_dict() for result in results])
    table = table.sort_values(
        by=["all_checks_ok", "transport_work_ton_km", "total_km", "runtime_sec"],
        ascending=[False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    return table
