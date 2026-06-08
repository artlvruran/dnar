from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import time
from typing import Any, Callable

import pandas as pd

from .. import core
from .. import gap_vrp_solver as gap_core
from .. import genetic_solver_components as ga
from ..dataset import RoutingDataset
from ..pipeline_runtime import execute_solver
from ..pipelines.u50_fast_dataset import build_fast_u50_dataset


@dataclass(frozen=True)
class _HiddenConfig:
    gap_step1_method: str = "dataset"
    gap_iter: int = 20
    max_agents: int | None = None
    use_repair: bool = True

    alns_iterations: int = 20
    alns_removal_q: int = 5
    alns_seed: int = 42
    alns_start_temp: float = 80.0
    alns_end_temp: float = 1.0
    alns_step: float = 0.98

    milp_time_limit_sec: int = 90
    milp_unassigned_penalty: float = 1e7

    ga_population_size: int = 20
    ga_generations: int = 20
    ga_elite_size: int = 3
    ga_seed: int = 42
    ga_generation_scale: float = 1.0
    ga_min_generations: int = 10
    ga_max_runtime_sec: float | None = None

    saa_sample_size: int = 12
    saa_iterations: int = 60
    saa_route_max_size: int = 8
    saa_seed: int = 42
    saa_log_every: int = 10


DEFAULT_HIDDEN_CONFIG = _HiddenConfig()


@dataclass(frozen=True)
class DemoContext:
    repo_root: Path
    dataset_path: Path
    summary_path: Path
    output_dir: Path
    factor_tag: str
    summary: dict[str, Any]
    counts: dict[str, int]
    hidden_config: _HiddenConfig


def prepare_context_from_dataset_path(
    *,
    dataset_path: Path | str,
    output_subdir: str = "custom_dataset",
    summary_path: Path | str | None = None,
    repo_root: Path | None = None,
    hidden_config: _HiddenConfig = DEFAULT_HIDDEN_CONFIG,
) -> DemoContext:
    root = _find_repo_root((repo_root or Path.cwd()).resolve())
    dataset_path = Path(dataset_path).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")

    inferred_summary = summary_path
    if inferred_summary is None:
        candidate = dataset_path.with_name(dataset_path.name.replace("dataset_", "summary_"))
        if candidate.exists():
            inferred_summary = candidate
    resolved_summary = Path(inferred_summary).resolve() if inferred_summary is not None else dataset_path
    summary: dict[str, Any] = {}
    if resolved_summary.exists() and resolved_summary != dataset_path:
        try:
            summary = json.loads(resolved_summary.read_text(encoding="utf-8"))
        except Exception:
            summary = {}

    factor_tag = dataset_path.stem
    output_dir = root / "demo" / "local" / output_subdir / factor_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = _patch_limits_from_vehicle_profiles(dataset_path)
    return DemoContext(
        repo_root=root,
        dataset_path=dataset_path,
        summary_path=resolved_summary,
        output_dir=output_dir,
        factor_tag=factor_tag,
        summary=summary,
        counts=counts,
        hidden_config=hidden_config,
    )


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").exists() and (candidate / "notebooks").exists():
            return candidate
    raise RuntimeError("Repo root not found")


def _factor_tag(downscale_factor: float) -> str:
    if float(downscale_factor).is_integer():
        return f"f{int(downscale_factor)}"
    return f"f{str(downscale_factor).replace('.', '_')}"


def _resolve_u50_base_dataset_path(root: Path) -> Path:
    candidates = [
        root
        / "src"
        / "data"
        / "real"
        / "day_load_profiles"
        / "u50"
        / "simplified_a_only"
        / "dataset_real_spb_day_u50_A_only_simplified.json",
        root
        / "src"
        / "data"
        / "real"
        / "day_load_profiles"
        / "u50"
        / "fast_uniform"
        / "simplified_a_only"
        / "dataset_real_spb_day_u50_A_only_simplified.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    checked = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"U50 base dataset not found. Checked:\n{checked}")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return str(value)


def _patch_limits_from_vehicle_profiles(dataset_path: Path) -> dict[str, int]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    meta = payload.get("metadata", {})
    profiles = meta.get("vehicle_profiles", {})

    for vt, profile in profiles.items():
        km = float(profile.get("max_daily_km", core.MAX_DAILY_KM_BY_TYPE.get(vt, 130.0)))
        hh = float(profile.get("max_shift_hours", core.MAX_SHIFT_HOURS_BY_TYPE.get(vt, 10.0)))
        sp = float(profile.get("avg_speed_kmph", core.AVG_SPEED_KMPH_BY_TYPE.get(vt, 24.0)))
        core.MAX_DAILY_KM_BY_TYPE[vt] = km
        core.MAX_SHIFT_HOURS_BY_TYPE[vt] = hh
        core.AVG_SPEED_KMPH_BY_TYPE[vt] = sp
        ga.MAX_DAILY_KM_BY_TYPE[vt] = km
        ga.MAX_SHIFT_HOURS_BY_TYPE[vt] = hh
        ga.AVG_SPEED_KMPH_BY_TYPE[vt] = sp

    return {
        "nodes": len(payload.get("graph", {}).get("nodes", [])),
        "edges": len(payload.get("graph", {}).get("edges", [])),
        "agents": len(payload.get("agents", [])),
        "tasks": len(payload.get("tasks", [])),
        "objects": len({t.get("destination_node_id") for t in payload.get("tasks", [])}),
        "sources": len({t.get("source_node_id") for t in payload.get("tasks", [])}),
    }


def prepare_u50_fast_uniform_context(
    *,
    downscale_factor: float = 10.0,
    seed: int = 42,
    force_rebuild: bool = False,
    grid_size: int = 6,
    safety_utilization: float = 0.85,
    output_subdir: str = "u50_fast_uniform",
    repo_root: Path | None = None,
    hidden_config: _HiddenConfig = DEFAULT_HIDDEN_CONFIG,
) -> DemoContext:
    root = _find_repo_root((repo_root or Path.cwd()).resolve())
    base_dataset_path = _resolve_u50_base_dataset_path(root)
    fast_root = root / "src" / "data" / "real" / "day_load_profiles" / "u50" / "fast_uniform"
    fast_root.mkdir(parents=True, exist_ok=True)

    factor_tag = _factor_tag(downscale_factor)
    dataset_path = fast_root / f"dataset_real_spb_day_u50_fast_{factor_tag}.json"
    summary_path = fast_root / f"summary_real_spb_day_u50_fast_{factor_tag}.json"
    output_dir = root / "demo" / "local" / output_subdir / factor_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    if force_rebuild or not dataset_path.exists() or not summary_path.exists():
        summary = build_fast_u50_dataset(
            base_dataset_path=base_dataset_path,
            out_dataset_path=dataset_path,
            downscale_factor=float(downscale_factor),
            seed=seed,
            grid_size=grid_size,
            safety_utilization=safety_utilization,
        )
    else:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    counts = _patch_limits_from_vehicle_profiles(dataset_path)
    return DemoContext(
        repo_root=root,
        dataset_path=dataset_path,
        summary_path=summary_path,
        output_dir=output_dir,
        factor_tag=factor_tag,
        summary=summary,
        counts=counts,
        hidden_config=hidden_config,
    )


def _make_progress_logger(enabled: bool) -> Callable[[str], None] | None:
    if not enabled:
        return None
    t0 = time.perf_counter()

    def _log(message: str) -> None:
        dt = time.perf_counter() - t0
        print(f"[+{dt:7.1f}s] {message}", flush=True)

    return _log


def run_u50_fast_uniform_benchmark(
    context: DemoContext,
    *,
    show_algo_progress: bool = True,
    show_solver_details: bool = False,
    require_full_assignment: bool = False,
) -> tuple[pd.DataFrame, Path]:
    cfg = context.hidden_config
    progress_log = _make_progress_logger(show_algo_progress)
    dataset_path = context.dataset_path

    def _run_algo(label: str, algorithm: str, params: dict[str, Any]):
        if progress_log is not None:
            progress_log(f"{label}: START")
        t0 = time.perf_counter()
        execution = execute_solver(
            algorithm=algorithm,
            dataset_path=dataset_path,
            solver_kwargs=params,
            show_progress=show_solver_details,
            verbose=False,
            progress_hook=progress_log,
        )
        if progress_log is not None:
            progress_log(f"{label}: DONE in {time.perf_counter() - t0:.1f}s")
        return execution.metrics

    results = [
        _run_algo(
            "GAP-VRP",
            "real_gap_vrp",
            {
                "step1_method": cfg.gap_step1_method,
                "gap_iter": cfg.gap_iter,
                "max_agents": cfg.max_agents,
                "use_repair": cfg.use_repair,
            },
        ),
        _run_algo(
            "GAP-VRP-ALNS",
            "real_gap_vrp_alns",
            {
                "step1_method": cfg.gap_step1_method,
                "gap_iter": cfg.gap_iter,
                "max_agents": cfg.max_agents,
                "use_repair": cfg.use_repair,
                "alns_iterations": cfg.alns_iterations,
                "alns_removal_q": cfg.alns_removal_q,
                "alns_seed": cfg.alns_seed,
                "start_temperature": cfg.alns_start_temp,
                "end_temperature": cfg.alns_end_temp,
                "temperature_step": cfg.alns_step,
            },
        ),
        _run_algo(
            "REAL-MILP",
            "real_milp",
            {
                "time_limit_sec": cfg.milp_time_limit_sec,
                "unassigned_penalty": cfg.milp_unassigned_penalty,
            },
        ),
        _run_algo(
            "REAL-GENETIC",
            "real_genetic",
            {
                "population_size": cfg.ga_population_size,
                "generations": cfg.ga_generations,
                "elite_size": cfg.ga_elite_size,
                "seed": cfg.ga_seed,
                "generation_scale": cfg.ga_generation_scale,
                "min_generations": cfg.ga_min_generations,
                "max_runtime_sec": cfg.ga_max_runtime_sec,
            },
        ),
        _run_algo(
            "GAP-VRP-SAA",
            "real_gap_vrp_saa",
            {
                "step1_method": cfg.gap_step1_method,
                "gap_iter": cfg.gap_iter,
                "max_agents": cfg.max_agents,
                "use_repair": cfg.use_repair,
                "saa_sample_size": cfg.saa_sample_size,
                "saa_iterations": cfg.saa_iterations,
                "saa_route_max_size": cfg.saa_route_max_size,
                "saa_seed": cfg.saa_seed,
                "saa_log_every": cfg.saa_log_every,
            },
        ),
    ]

    benchmark_df = pd.DataFrame([r.as_dict() for r in results])
    benchmark_df = benchmark_df.sort_values(
        by=["all_checks_ok", "transport_work_ton_km", "total_km", "runtime_sec"],
        ascending=[False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)

    if require_full_assignment and (benchmark_df["unassigned_tasks"].fillna(10**9) > 0).any():
        raise AssertionError("Some algorithms still have unassigned tasks")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = context.output_dir / f"benchmark_u50_fast_alns_{context.factor_tag}_{ts}.json"
    records = benchmark_df.where(pd.notna(benchmark_df), None).to_dict(orient="records")
    artifact = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "notebook": "demo/real_u50_fast_uniform_3algo_benchmark_alns.ipynb",
        "dataset_path": str(dataset_path),
        "config": {
            "downscale_factor": context.factor_tag,
            "hidden_config": cfg.__dict__,
        },
        "results": records,
    }
    out_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return benchmark_df, out_path


def benchmark_main_table(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    main_cols = [
        "algorithm",
        "feasible",
        "all_checks_ok",
        "assigned_routes",
        "unassigned_tasks",
        "active_agents",
        "transport_work_ton_km",
        "total_km",
        "deadhead_km",
        "deadhead_share_pct",
        "total_hours",
        "runtime_sec",
        "alns_gain",
        "solver_error",
    ]
    return benchmark_df[[c for c in main_cols if c in benchmark_df.columns]]


def benchmark_detail_table(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    detail_cols = [
        "algorithm",
        "checks",
        "solver_error",
        "step1_method",
        "gap_iter",
        "used_agents",
        "time_limit_sec",
        "unassigned_penalty",
        "population_size",
        "generations",
        "generations_requested",
        "generation_scale",
        "elite_size",
        "alns_iterations",
        "alns_removal_q",
        "alns_seed",
        "alns_base_objective",
        "alns_best_objective",
        "alns_gain",
        "saa_sample_size",
        "saa_iterations",
        "saa_route_max_size",
        "saa_seed",
        "saa_log_every",
        "saa_route_pool_size",
        "saa_assembled_route_count",
    ]
    return benchmark_df[[c for c in detail_cols if c in benchmark_df.columns]]


def _dataset_for_render(
    *,
    dataset_obj: RoutingDataset,
    solver_result: Any,
    graph_obj,
    cache_obj: dict[tuple[str, str], tuple[list[str], float] | None],
    step1_method: str,
    show_solver_progress: bool,
    algorithm: str,
) -> RoutingDataset:
    route_task_ids = {tid for route in solver_result.routes for tid in route.task_ids}
    base_task_ids = {t.task_id for t in dataset_obj.tasks}
    gap_like_algos = {"real_gap_vrp", "real_gap_vrp_alns", "real_gap_vrp_saa"}

    if route_task_ids.issubset(base_task_ids):
        return dataset_obj

    if algorithm not in gap_like_algos:
        missing = sorted(route_task_ids - base_task_ids)[:5]
        raise RuntimeError(f"Route task ids missing in dataset tasks: sample={missing}")

    if step1_method == "lp":
        generated_tasks = gap_core.generate_tasks_lp(
            dataset_obj,
            graph_obj,
            cache_obj,
            show_progress=show_solver_progress,
        )
    else:
        generated_tasks = gap_core.generate_tasks_greedy(
            dataset_obj,
            graph_obj,
            cache_obj,
            show_progress=show_solver_progress,
        )

    generated_task_ids = {t.task_id for t in generated_tasks}
    if not route_task_ids.issubset(generated_task_ids):
        missing = sorted(route_task_ids - generated_task_ids)[:5]
        raise RuntimeError(f"Route task ids missing in generated step1 tasks: sample={missing}")

    return RoutingDataset(
        graph=dataset_obj.graph,
        fleet=dataset_obj.fleet,
        tasks=generated_tasks,
        routes=dataset_obj.routes,
        metadata=dataset_obj.metadata,
    )


def solve_for_visualization(
    context: DemoContext,
    *,
    algorithm: str = "real_genetic",
    show_solver_progress: bool = False,
) -> tuple[Any, RoutingDataset, Any, dict[tuple[str, str], tuple[list[str], float] | None]]:
    cfg = context.hidden_config
    params: dict[str, Any]
    if algorithm == "real_gap_vrp":
        params = {
            "step1_method": cfg.gap_step1_method,
            "gap_iter": cfg.gap_iter,
            "use_repair": cfg.use_repair,
            "max_agents": cfg.max_agents,
        }
    elif algorithm == "real_gap_vrp_alns":
        params = {
            "step1_method": cfg.gap_step1_method,
            "gap_iter": cfg.gap_iter,
            "use_repair": cfg.use_repair,
            "max_agents": cfg.max_agents,
            "alns_iterations": cfg.alns_iterations,
            "alns_removal_q": cfg.alns_removal_q,
            "alns_seed": cfg.alns_seed,
            "start_temperature": cfg.alns_start_temp,
            "end_temperature": cfg.alns_end_temp,
            "temperature_step": cfg.alns_step,
        }
    elif algorithm == "real_gap_vrp_saa":
        params = {
            "step1_method": cfg.gap_step1_method,
            "gap_iter": cfg.gap_iter,
            "use_repair": cfg.use_repair,
            "max_agents": cfg.max_agents,
            "saa_sample_size": cfg.saa_sample_size,
            "saa_iterations": cfg.saa_iterations,
            "saa_route_max_size": cfg.saa_route_max_size,
            "saa_seed": cfg.saa_seed,
            "saa_log_every": cfg.saa_log_every,
        }
    elif algorithm == "real_milp":
        params = {
            "time_limit_sec": cfg.milp_time_limit_sec,
            "unassigned_penalty": cfg.milp_unassigned_penalty,
        }
    elif algorithm == "real_genetic":
        params = {
            "population_size": cfg.ga_population_size,
            "generations": cfg.ga_generations,
            "elite_size": cfg.ga_elite_size,
            "seed": cfg.ga_seed,
            "generation_scale": cfg.ga_generation_scale,
            "min_generations": cfg.ga_min_generations,
            "max_runtime_sec": cfg.ga_max_runtime_sec,
        }
    else:
        raise ValueError(f"Unknown algorithm={algorithm}")

    execution = execute_solver(
        algorithm=algorithm,
        dataset_path=context.dataset_path,
        solver_kwargs=params,
        show_progress=show_solver_progress,
        verbose=False,
    )
    if execution.solver_result is None:
        raise RuntimeError(
            f"{algorithm} failed: {execution.metrics.as_dict().get('solver_error', 'unknown error')}"
        )
    solver_result = execution.solver_result
    dataset_obj = execution.problem.dataset
    graph_obj = execution.graph
    cache_obj = execution.cache

    dataset_render = _dataset_for_render(
        dataset_obj=dataset_obj,
        solver_result=solver_result,
        graph_obj=graph_obj,
        cache_obj=cache_obj,
        step1_method=cfg.gap_step1_method,
        show_solver_progress=show_solver_progress,
        algorithm=algorithm,
    )
    return solver_result, dataset_render, graph_obj, cache_obj


def render_solution_map_for_algorithm(
    context: DemoContext,
    *,
    algorithm: str = "real_genetic",
    show_solver_progress: bool = False,
) -> dict[str, Any]:
    solver_result, dataset_render, graph_obj, cache_obj = solve_for_visualization(
        context,
        algorithm=algorithm,
        show_solver_progress=show_solver_progress,
    )

    if not solver_result.routes:
        raise RuntimeError("No routes in solver result, schedule map cannot be rendered")

    map_path = context.output_dir / f"schedule_map_{algorithm}_{context.factor_tag}.png"
    plan_path = context.output_dir / f"schedule_plan_{algorithm}_{context.factor_tag}.json"
    core.render_solution_map(
        dataset_render,
        graph_obj,
        cache_obj,
        solver_result.routes,
        solver_result.states,
        map_path,
    )

    plan_payload = core.build_solution_plan(
        solver_result.routes,
        dataset_render,
        solver_result.states,
    )
    plan_path.write_text(json.dumps(plan_payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return {
        "algorithm": algorithm,
        "assigned_routes": len(solver_result.routes),
        "unassigned": len(solver_result.unassigned),
        "active_agents": sum(1 for s in solver_result.states.values() if s.task_ids),
        "feasible": bool(solver_result.feasible),
        "map_path": map_path,
        "plan_path": plan_path,
    }


def render_per_agent_maps_for_algorithm(
    context: DemoContext,
    *,
    algorithm: str = "real_genetic",
    show_solver_progress: bool = False,
    max_agents: int | None = None,
) -> dict[str, Any]:
    solver_result, dataset_render, graph_obj, cache_obj = solve_for_visualization(
        context,
        algorithm=algorithm,
        show_solver_progress=show_solver_progress,
    )
    active_agent_ids = sorted([agent_id for agent_id, state in solver_result.states.items() if state.task_ids])
    if max_agents is not None and max_agents > 0:
        active_agent_ids = active_agent_ids[:max_agents]

    per_agent_dir = context.output_dir / f"per_agent_maps_{algorithm}_{context.factor_tag}"
    per_agent_dir.mkdir(parents=True, exist_ok=True)

    saved: list[dict[str, Any]] = []
    for agent_id in active_agent_ids:
        state = solver_result.states[agent_id]
        agent_routes = [route for route in solver_result.routes if route.agent_id == agent_id]
        if not agent_routes:
            continue

        map_path = per_agent_dir / f"{agent_id}.png"
        plan_path = per_agent_dir / f"{agent_id}.json"
        core.render_solution_map(
            dataset_render,
            graph_obj,
            cache_obj,
            agent_routes,
            {agent_id: state},
            map_path,
        )
        plan_payload = core.build_solution_plan(agent_routes, dataset_render, {agent_id: state})
        plan_path.write_text(
            json.dumps(plan_payload, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        saved.append(
            {
                "agent_id": agent_id,
                "routes": len(agent_routes),
                "map_path": map_path,
                "plan_path": plan_path,
            }
        )

    return {
        "algorithm": algorithm,
        "output_dir": per_agent_dir,
        "saved_agents": len(saved),
        "active_agents_total": sum(1 for state in solver_result.states.values() if state.task_ids),
        "items": saved,
    }
