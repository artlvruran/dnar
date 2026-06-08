from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable

from .. import core
from ..dataset import RoutingDataset, dataset_from_dict
from ..pipelines import runs as run_impl
from ..solvers import gap_vrp_alns_solver
from ..solvers import gap_vrp_saa_solver
from ..solvers import real_gap_vrp_solver
from ..solvers import real_genetic_solver
from ..solvers import real_milp_ksenya_solver
from ..solvers import real_milp_ksenya_decomp_solver
from ..solvers import real_milp_ksenya_knn_solver
from ..solvers import real_milp_ksenya_decomp_knn_solver
from ..solvers import real_milp_solver
from .constraints import ConstraintBundle, build_constraint_bundle
from .dataset_adapters import normalize_payload


@dataclass(frozen=True)
class ProblemContext:
    dataset_path: Path
    dataset: RoutingDataset
    payload: dict[str, Any]
    dataset_profile: str
    normalization_notes: tuple[str, ...]
    constraints: ConstraintBundle


@dataclass
class SolverExecution:
    metrics: run_impl.RunMetrics
    solver_result: Any | None
    problem: ProblemContext
    graph: Any
    cache: dict[tuple[str, str], tuple[list[str], float] | None]


def _prefixed_emitter(
    *,
    show_progress: bool,
    progress_hook: Callable[[str], None] | None,
    prefix: str,
) -> Callable[[str], None]:
    return lambda message: run_impl._emit_progress(  # type: ignore[attr-defined]
        show_progress=show_progress,
        progress_hook=progress_hook,
        prefix=prefix,
        message=message,
    )


def prepare_problem_context(dataset_path: Path | str) -> ProblemContext:
    dataset_path = Path(dataset_path)
    payload_raw = run_impl.load_payload(dataset_path)
    normalized = normalize_payload(payload_raw)

    dataset = dataset_from_dict(normalized.payload)
    dataset.graph.validate()
    dataset.fleet.validate()

    constraints = build_constraint_bundle(
        payload=normalized.payload,
        dataset_profile=normalized.dataset_profile,
    )
    constraints.apply()

    return ProblemContext(
        dataset_path=dataset_path,
        dataset=dataset,
        payload=normalized.payload,
        dataset_profile=normalized.dataset_profile,
        normalization_notes=normalized.notes,
        constraints=constraints,
    )


def execute_solver(
    *,
    algorithm: str,
    dataset_path: Path | str,
    solver_kwargs: dict[str, Any] | None = None,
    show_progress: bool = False,
    verbose: bool = False,
    progress_hook: Callable[[str], None] | None = None,
) -> SolverExecution:
    solver_kwargs = dict(solver_kwargs or {})
    problem = prepare_problem_context(dataset_path)

    emit = _prefixed_emitter(
        show_progress=show_progress,
        progress_hook=progress_hook,
        prefix=algorithm,
    )

    emit(f"dataset profile: {problem.dataset_profile}")
    if problem.normalization_notes:
        emit(f"normalization notes: {len(problem.normalization_notes)}")
    emit(
        "constraints: "
        f"edge_vt={problem.constraints.uses_edge_vehicle_type_limits}, "
        f"agent_volume={problem.constraints.uses_agent_volume_limits}, "
        f"object_caps_m={problem.constraints.uses_object_mass_limits}, "
        f"object_caps_v={problem.constraints.uses_object_volume_limits}"
    )

    emit("build nx graph")
    graph = core.build_nx_graph(problem.dataset)
    cache: dict[tuple[str, str], tuple[list[str], float] | None] = {}

    details_common = {
        "dataset_profile": problem.dataset_profile,
        "normalization_notes": list(problem.normalization_notes),
        "constraint_bundle": problem.constraints.as_dict(),
    }

    t0 = time.perf_counter()
    solver_result = None
    solver_details: dict[str, Any] = {}

    try:
        emit("solver start")
        if algorithm == "real_gap_vrp":
            step1_method = str(solver_kwargs.get("step1_method", "dataset"))
            gap_iter = int(solver_kwargs.get("gap_iter", 20))
            use_repair = bool(solver_kwargs.get("use_repair", True))
            max_agents = solver_kwargs.get("max_agents")

            dataset = problem.dataset
            payload = problem.payload
            if max_agents is not None:
                emit(f"select agents subset: max_agents={max_agents}")
                dataset, payload = run_impl.gap.select_agents_subset(
                    dataset,
                    payload,
                    max_agents=max_agents,
                )
                problem = ProblemContext(
                    dataset_path=problem.dataset_path,
                    dataset=dataset,
                    payload=payload,
                    dataset_profile=problem.dataset_profile,
                    normalization_notes=problem.normalization_notes,
                    constraints=problem.constraints,
                )
                graph = core.build_nx_graph(problem.dataset)
                cache = {}

            solver_result = real_gap_vrp_solver.solve_real_gap_vrp(
                dataset=problem.dataset,
                payload=problem.payload,
                graph=graph,
                cache=cache,
                step1_method=step1_method,
                gap_iter=gap_iter,
                use_repair=use_repair,
                show_progress=show_progress,
                verbose=verbose,
                progress_hook=emit,
            )
            solver_details = {
                "step1_method": step1_method,
                "gap_iter": gap_iter,
                "used_agents": len(problem.dataset.fleet.agents),
            }

        elif algorithm == "real_gap_vrp_alns":
            step1_method = str(solver_kwargs.get("step1_method", "dataset"))
            gap_iter = int(solver_kwargs.get("gap_iter", 20))
            use_repair = bool(solver_kwargs.get("use_repair", True))
            alns_iterations = int(solver_kwargs.get("alns_iterations", 20))
            alns_removal_q = int(solver_kwargs.get("alns_removal_q", 5))
            alns_seed = int(solver_kwargs.get("alns_seed", 42))
            alns_log_every = int(solver_kwargs.get("alns_log_every", 10))
            start_temperature = float(solver_kwargs.get("start_temperature", 80.0))
            end_temperature = float(solver_kwargs.get("end_temperature", 1.0))
            temperature_step = float(solver_kwargs.get("temperature_step", 0.98))

            bundle = gap_vrp_alns_solver.solve_real_gap_vrp_alns(
                dataset=problem.dataset,
                payload=problem.payload,
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
            solver_result = bundle.result
            solver_details = {
                "step1_method": step1_method,
                "gap_iter": gap_iter,
                "used_agents": len(problem.dataset.fleet.agents),
                "alns_iterations": alns_iterations,
                "alns_removal_q": alns_removal_q,
                "alns_seed": alns_seed,
                "alns_log_every": alns_log_every,
                "alns_base_objective": round(bundle.base_objective, 3),
                "alns_best_objective": round(bundle.best_objective, 3),
                "alns_gain": round(bundle.base_objective - bundle.best_objective, 3),
            }

        elif algorithm == "real_gap_vrp_saa":
            step1_method = str(solver_kwargs.get("step1_method", "dataset"))
            gap_iter = int(solver_kwargs.get("gap_iter", 20))
            use_repair = bool(solver_kwargs.get("use_repair", True))
            saa_sample_size = int(solver_kwargs.get("saa_sample_size", 12))
            saa_iterations = int(solver_kwargs.get("saa_iterations", 60))
            saa_route_max_size = int(solver_kwargs.get("saa_route_max_size", 8))
            saa_seed = int(solver_kwargs.get("saa_seed", 42))
            saa_log_every = int(solver_kwargs.get("saa_log_every", 10))

            bundle = gap_vrp_saa_solver.solve_real_gap_vrp_saa(
                dataset=problem.dataset,
                payload=problem.payload,
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
            solver_result = bundle.result
            solver_details = {
                "step1_method": step1_method,
                "gap_iter": gap_iter,
                "used_agents": len(problem.dataset.fleet.agents),
                "saa_sample_size": saa_sample_size,
                "saa_iterations": saa_iterations,
                "saa_route_max_size": saa_route_max_size,
                "saa_seed": saa_seed,
                "saa_log_every": saa_log_every,
                "saa_route_pool_size": bundle.route_pool_size,
                "saa_assembled_route_count": bundle.assembled_route_count,
            }

        elif algorithm == "real_milp":
            time_limit_sec = int(solver_kwargs.get("time_limit_sec", 60))
            unassigned_penalty = float(solver_kwargs.get("unassigned_penalty", 1e5))
            solver_result = real_milp_solver.solve_real_milp(
                dataset=problem.dataset,
                payload=problem.payload,
                graph=graph,
                cache=cache,
                time_limit_sec=time_limit_sec,
                unassigned_penalty=unassigned_penalty,
                show_progress=show_progress,
                progress_hook=emit,
            )
            solver_details = {
                "time_limit_sec": time_limit_sec,
                "unassigned_penalty": unassigned_penalty,
            }

        elif algorithm == "real_genetic":
            population_size = int(solver_kwargs.get("population_size", 20))
            generations_requested = int(solver_kwargs.get("generations", 20))
            generation_scale = float(solver_kwargs.get("generation_scale", 1.0))
            min_generations = int(solver_kwargs.get("min_generations", 3))
            elite_size = int(solver_kwargs.get("elite_size", 3))
            seed = solver_kwargs.get("seed", 42)
            max_runtime_sec = solver_kwargs.get("max_runtime_sec")
            if max_runtime_sec is not None:
                max_runtime_sec = float(max_runtime_sec)

            effective_generations = generations_requested
            if generation_scale > 0:
                effective_generations = max(min_generations, int(round(generations_requested * generation_scale)))
            if effective_generations != generations_requested:
                emit(
                    f"generation downscale applied: requested={generations_requested}, "
                    f"effective={effective_generations}, scale={generation_scale}"
                )

            solver_result = real_genetic_solver.solve_real_genetic(
                dataset=problem.dataset,
                payload=problem.payload,
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
            solver_details = {
                "population_size": population_size,
                "generations": effective_generations,
                "generations_requested": generations_requested,
                "generation_scale": generation_scale,
                "elite_size": elite_size,
                "max_runtime_sec": max_runtime_sec,
            }

        elif algorithm in (
            "milp_ksenya",
            "milp_ksenya_decomp",
            "milp_ksenya_knn",
            "milp_ksenya_decomp_knn",
        ):
            if algorithm == "milp_ksenya":
                milp_meta, solution = real_milp_ksenya_solver.solve_real_milp_ksenya(
                    problem.payload
                )
            elif algorithm == "milp_ksenya_decomp":
                milp_meta, solution = real_milp_ksenya_decomp_solver.solve_real_milp_ksenya_decomp(
                    problem.payload,
                    knn_k=solver_kwargs.get("knn_k"),
                    time_limit_sec=int(solver_kwargs.get("time_limit_sec", 120)),
                    mip_rel_gap=float(solver_kwargs.get("mip_rel_gap", 0.05)),
                    verbose=bool(solver_kwargs.get("verbose", verbose)),
                )
            elif algorithm == "milp_ksenya_knn":
                milp_meta, solution = real_milp_ksenya_knn_solver.solve_real_milp_ksenya_knn(
                    problem.payload,
                    knn_k=int(solver_kwargs.get("knn_k", 10)),
                    time_limit_sec=int(solver_kwargs.get("time_limit_sec", 120)),
                    mip_rel_gap=float(solver_kwargs.get("mip_rel_gap", 0.05)),
                    verbose=bool(solver_kwargs.get("verbose", verbose)),
                )
            else:
                milp_meta, solution = real_milp_ksenya_decomp_knn_solver.solve_real_milp_ksenya_decomp_knn(
                    problem.payload,
                    knn_k=int(solver_kwargs.get("knn_k", 5)),
                    time_limit_sec=int(solver_kwargs.get("time_limit_sec", 120)),
                    mip_rel_gap=float(solver_kwargs.get("mip_rel_gap", 0.05)),
                    verbose=bool(solver_kwargs.get("verbose", verbose)),
                )
            _elapsed = time.perf_counter() - t0
            emit(f"solver done in {_elapsed:.1f}s")

            _routes = solution["routes"]
            _unassigned = solution["unassigned"]
            _states = solution["states"]
            _total_km = sum(r["route_km"] for r in _routes)
            _total_hours = sum(
                r["TT_min"] for r in _routes if not r.get("contains_disconnected_cycle")
            ) / 60.0

            from ..pipelines.runs import RunMetrics as _RunMetrics
            _metrics = _RunMetrics(
                algorithm=algorithm,
                dataset_path=str(problem.dataset_path),
                feasible=bool(milp_meta.get("success", False)),
                all_checks_ok=bool(milp_meta.get("success", False)) and len(_unassigned) == 0,
                assigned_routes=len(_routes),
                unassigned_tasks=len(_unassigned),
                active_agents=sum(1 for s in _states.values() if s["task_ids"]),
                transport_work_ton_km=solution.get("transport_work_ton_km"),
                total_km=round(_total_km, 3),
                deadhead_km=None,
                deadhead_share_pct=None,
                total_hours=round(_total_hours, 3),
                runtime_sec=_elapsed,
                details={"solver_error": milp_meta.get("reason")} if not milp_meta.get("success") else {},
            )
            return SolverExecution(
                metrics=_metrics,
                solver_result=None,
                problem=problem,
                graph=graph,
                cache=cache,
            )

        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        emit(f"solver error after {elapsed:.1f}s: {type(exc).__name__}")
        metrics = run_impl._error_metrics(  # type: ignore[attr-defined]
            algorithm=algorithm,
            dataset_path=problem.dataset_path,
            dataset=problem.dataset,
            graph=graph,
            cache=cache,
            elapsed=elapsed,
            solver_error=f"{type(exc).__name__}: {exc}",
            details={**details_common, **solver_details},
        )
        return SolverExecution(
            metrics=metrics,
            solver_result=None,
            problem=problem,
            graph=graph,
            cache=cache,
        )

    elapsed = time.perf_counter() - t0
    emit(f"solver done in {elapsed:.1f}s")
    metrics = run_impl._metrics_from_solver_result(  # type: ignore[attr-defined]
        algorithm=algorithm,
        dataset_path=problem.dataset_path,
        dataset=problem.dataset,
        graph=graph,
        cache=cache,
        result=solver_result,
        elapsed=elapsed,
        details={**details_common, **solver_details},
    )
    return SolverExecution(
        metrics=metrics,
        solver_result=solver_result,
        problem=problem,
        graph=graph,
        cache=cache,
    )
