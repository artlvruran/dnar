from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ... import core
from ...backend.io import load_dataset
from .distance_oracle import DistanceOracleWithFallback, PrecomputedDistanceOracle
from .evaluator import finalize_enriched_result
from .gap_vrp_solver_v2 import EnrichedGapVRPConfig, EnrichedGapVRPSolver
from .milp_solver_v2 import EnrichedMILPConfig, EnrichedMILPSolver
from .milp_decomp_solver_v1 import EnrichedMILPDecompConfig, EnrichedMILPDecompSolver
from .legacy_pipeline_bridge_v1 import (
    EnrichedLegacyGapConfig,
    EnrichedLegacyMILPConfig,
    solve_enriched_legacy_gap_vrp as _solve_enriched_legacy_gap_vrp,
    solve_enriched_legacy_milp as _solve_enriched_legacy_milp,
)
from .problem import EnrichedProblem, build_enriched_problem
from .milp_stochastic_solver_v1 import EnrichedMILPStochasticConfig, EnrichedMILPStochasticSolver
from .repair import greedy_repair_unassigned
from .stochastic_solver_v2 import (
    EnrichedStochasticConfig,
    EnrichedStochasticGRASPSolver,
    EnrichedStochasticRRSolver,
)
from .milp_ablation_solvers_v1 import (
    AblationBatchCascadedConfig,
    AblationBatchPortfolioConfig,
    AblationBatchThenMilpConfig,
    AblationAdaptiveHardnessConfig,
    AblationAdaptiveKConfig,
    AblationBaselineConfig,
    AblationCriticalFirstConfig,
    AblationHybridSeededConfig,
    AblationLNSConfig,
    AblationLexicographic2StageConfig,
    AblationPenaltySweepConfig,
    AblationPortfolioConfig,
    AblationTimeWindowedConfig,
    AblationZoneBundleConfig,
    AblationZoneQuotaConfig,
    EnrichedMILPAblationAdaptiveHardnessSolver,
    EnrichedMILPAblationAdaptiveKSolver,
    EnrichedMILPAblationBatchCascadedSolver,
    EnrichedMILPAblationBatchPortfolioSolver,
    EnrichedMILPAblationBatchThenMilpSolver,
    EnrichedMILPAblationBaselineSolver,
    EnrichedMILPAblationCriticalFirstSolver,
    EnrichedMILPAblationHybridSeededSolver,
    EnrichedMILPAblationLNSSolver,
    EnrichedMILPAblationLexicographic2StageSolver,
    EnrichedMILPAblationPenaltySweepSolver,
    EnrichedMILPAblationPortfolioSolver,
    EnrichedMILPAblationTimeWindowedSolver,
    EnrichedMILPAblationZoneBundleSolver,
    EnrichedMILPAblationZoneQuotaSolver,
)
from .types import EnrichedSolveResult


def _build_oracle(problem: EnrichedProblem) -> DistanceOracleWithFallback:
    dataset, _payload = load_dataset(problem.dataset_path)
    nx_graph = core.build_nx_graph(dataset)
    precomputed = PrecomputedDistanceOracle.from_dataset_payload(
        dataset_path=Path(problem.dataset_path),
        payload=problem.payload,
    )
    return DistanceOracleWithFallback(nx_graph=nx_graph, precomputed=precomputed)


def _coverage(result: EnrichedSolveResult, total_tasks: int) -> int:
    return max(0, int(total_tasks) - len(result.unassigned_task_ids))


def _maybe_fallback_to_greedy(
    *,
    problem: EnrichedProblem,
    oracle: DistanceOracleWithFallback,
    result: EnrichedSolveResult,
    random_seed: int = 42,
) -> EnrichedSolveResult:
    if result.feasible:
        return result
    greedy = EnrichedGapVRPSolver(
        EnrichedGapVRPConfig(
            random_seed=random_seed,
            top_k_agents=30,
            balance_penalty=0.02,
        )
    ).solve(problem=problem, oracle=oracle)
    greedy = finalize_enriched_result(problem=problem, result=greedy, oracle=oracle)
    if _coverage(greedy, len(problem.tasks)) > _coverage(result, len(problem.tasks)):
        result.details["fallback"] = {
            "used": True,
            "fallback_solver": "enriched_batched_greedy_v1",
            "reason": "milp_family_partial_coverage",
        }
        greedy.details["fallback_from"] = result.algorithm
        greedy.algorithm = f"{result.algorithm}_fallback_greedy"
        return greedy
    return result


def solve_enriched_milp(
    *,
    dataset_path: Path | str,
    time_limit_sec: int = 60,
    max_runtime_sec: float | None = None,
    unassigned_penalty: float = 1e6,
    max_pairs_per_task: int = 80,
    use_repair: bool = False,
    fallback_to_greedy: bool = False,
) -> EnrichedSolveResult:
    problem = build_enriched_problem(dataset_path)
    oracle = _build_oracle(problem)
    solver = EnrichedMILPSolver(
        EnrichedMILPConfig(
            time_limit_sec=time_limit_sec,
            max_runtime_sec=max_runtime_sec,
            unassigned_penalty=unassigned_penalty,
            max_pairs_per_task=max_pairs_per_task,
        )
    )
    result = solver.solve(problem=problem, oracle=oracle)
    if use_repair and result.unassigned_task_ids:
        routes, unresolved, stats = greedy_repair_unassigned(
            problem=problem,
            routes=result.routes,
            unassigned_task_ids=result.unassigned_task_ids,
            oracle=oracle,
        )
        result.routes = routes
        result.unassigned_task_ids = unresolved
        result.details["repair"] = {
            "enabled": True,
            "repaired_tasks": stats.repaired_tasks,
            "created_routes": stats.created_routes,
        }
    result = finalize_enriched_result(problem=problem, result=result, oracle=oracle)
    if fallback_to_greedy:
        result = _maybe_fallback_to_greedy(problem=problem, oracle=oracle, result=result, random_seed=42)
    return result


def solve_enriched_milp_decomposed(
    *,
    dataset_path: Path | str,
    objective: str = "tasks",
    unassigned_penalty: float = 1e5,
    time_limit_sec_per_zone: int = 45,
    max_runtime_sec: float | None = None,
    max_pairs_per_bundle: int = 120,
    bundle_mass_quantile: float = 0.35,
    bundle_vol_quantile: float = 0.35,
    bundle_fill_factor: float = 0.90,
    bundle_max_tasks: int = 8,
    verbose: bool = False,
    use_repair: bool = False,
    fallback_to_greedy: bool = False,
) -> EnrichedSolveResult:
    problem = build_enriched_problem(dataset_path)
    oracle = _build_oracle(problem)
    solver = EnrichedMILPDecompSolver(
        EnrichedMILPDecompConfig(
            objective=objective,
            unassigned_penalty=unassigned_penalty,
            time_limit_sec_per_zone=time_limit_sec_per_zone,
            max_runtime_sec=max_runtime_sec,
            max_pairs_per_bundle=max_pairs_per_bundle,
            bundle_mass_quantile=bundle_mass_quantile,
            bundle_vol_quantile=bundle_vol_quantile,
            bundle_fill_factor=bundle_fill_factor,
            bundle_max_tasks=bundle_max_tasks,
            verbose=verbose,
        )
    )
    result = solver.solve(problem=problem, oracle=oracle)
    if use_repair and result.unassigned_task_ids:
        routes, unresolved, stats = greedy_repair_unassigned(
            problem=problem,
            routes=result.routes,
            unassigned_task_ids=result.unassigned_task_ids,
            oracle=oracle,
        )
        result.routes = routes
        result.unassigned_task_ids = unresolved
        result.details["repair"] = {
            "enabled": True,
            "repaired_tasks": stats.repaired_tasks,
            "created_routes": stats.created_routes,
        }
    result = finalize_enriched_result(problem=problem, result=result, oracle=oracle)
    if fallback_to_greedy:
        result = _maybe_fallback_to_greedy(problem=problem, oracle=oracle, result=result, random_seed=43)
    return result


def solve_enriched_milp_stochastic(
    *,
    dataset_path: Path | str,
    time_budget_sec: float = 20.0,
    max_starts: int = 8,
    per_start_time_limit_sec: int = 8,
    max_pairs_per_task: int = 40,
    unassigned_penalty: float = 1e6,
    jitter: float = 0.05,
    seed: int = 42,
    use_repair: bool = False,
    fallback_to_greedy: bool = False,
) -> EnrichedSolveResult:
    problem = build_enriched_problem(dataset_path)
    oracle = _build_oracle(problem)
    solver = EnrichedMILPStochasticSolver(
        EnrichedMILPStochasticConfig(
            time_budget_sec=time_budget_sec,
            max_starts=max_starts,
            per_start_time_limit_sec=per_start_time_limit_sec,
            max_pairs_per_task=max_pairs_per_task,
            unassigned_penalty=unassigned_penalty,
            jitter=jitter,
            seed=seed,
        )
    )
    result = solver.solve(problem=problem, oracle=oracle)
    if use_repair and result.unassigned_task_ids:
        routes, unresolved, stats = greedy_repair_unassigned(
            problem=problem,
            routes=result.routes,
            unassigned_task_ids=result.unassigned_task_ids,
            oracle=oracle,
        )
        result.routes = routes
        result.unassigned_task_ids = unresolved
        result.details["repair"] = {
            "enabled": True,
            "repaired_tasks": stats.repaired_tasks,
            "created_routes": stats.created_routes,
        }
    result = finalize_enriched_result(problem=problem, result=result, oracle=oracle)
    if fallback_to_greedy:
        result = _maybe_fallback_to_greedy(problem=problem, oracle=oracle, result=result, random_seed=seed)
    return result


def solve_enriched_batched_greedy(
    *,
    dataset_path: Path | str,
    random_seed: int = 42,
    top_k_agents: int = 20,
    balance_penalty: float = 0.05,
    max_runtime_sec: float | None = None,
) -> EnrichedSolveResult:
    problem = build_enriched_problem(dataset_path)
    oracle = _build_oracle(problem)
    solver = EnrichedGapVRPSolver(
        EnrichedGapVRPConfig(
            random_seed=random_seed,
            top_k_agents=top_k_agents,
            balance_penalty=balance_penalty,
            max_runtime_sec=max_runtime_sec,
        )
    )
    result = solver.solve(problem=problem, oracle=oracle)
    return finalize_enriched_result(problem=problem, result=result, oracle=oracle)


def solve_enriched_gap_vrp(
    *,
    dataset_path: Path | str,
    random_seed: int = 42,
    top_k_agents: int = 20,
    balance_penalty: float = 0.05,
    max_runtime_sec: float | None = None,
) -> EnrichedSolveResult:
    # Backward-compatible alias:
    # original name suggested GAP+VRP decomposition, but current enriched solver
    # is a batched greedy assignment with route-level trip estimation.
    return solve_enriched_batched_greedy(
        dataset_path=dataset_path,
        random_seed=random_seed,
        top_k_agents=top_k_agents,
        balance_penalty=balance_penalty,
        max_runtime_sec=max_runtime_sec,
    )


def solve_enriched_stochastic_rr(
    *,
    dataset_path: Path | str,
    time_budget_sec: float = 20.0,
    max_starts: int = 12,
    seed: int = 42,
) -> EnrichedSolveResult:
    problem = build_enriched_problem(dataset_path)
    oracle = _build_oracle(problem)
    solver = EnrichedStochasticRRSolver(
        EnrichedStochasticConfig(
            time_budget_sec=time_budget_sec,
            max_starts=max_starts,
            seed=seed,
        )
    )
    result = solver.solve(problem=problem, oracle=oracle)
    return finalize_enriched_result(problem=problem, result=result, oracle=oracle)


def solve_enriched_stochastic_grasp(
    *,
    dataset_path: Path | str,
    time_budget_sec: float = 20.0,
    max_starts: int = 12,
    seed: int = 42,
) -> EnrichedSolveResult:
    problem = build_enriched_problem(dataset_path)
    oracle = _build_oracle(problem)
    solver = EnrichedStochasticGRASPSolver(
        EnrichedStochasticConfig(
            time_budget_sec=time_budget_sec,
            max_starts=max_starts,
            seed=seed,
        )
    )
    result = solver.solve(problem=problem, oracle=oracle)
    return finalize_enriched_result(problem=problem, result=result, oracle=oracle)


def solve_enriched_milp_ablation_baseline(
    *,
    dataset_path: Path | str,
    time_limit_sec: int = 30,
    max_pairs_per_task: int = 80,
    unassigned_penalty: float = 1e6,
) -> EnrichedSolveResult:
    problem = build_enriched_problem(dataset_path)
    oracle = _build_oracle(problem)
    solver = EnrichedMILPAblationBaselineSolver(
        AblationBaselineConfig(
            time_limit_sec=time_limit_sec,
            max_pairs_per_task=max_pairs_per_task,
            unassigned_penalty=unassigned_penalty,
        )
    )
    result = solver.solve(problem=problem, oracle=oracle)
    return finalize_enriched_result(problem=problem, result=result, oracle=oracle)


def solve_enriched_milp_ablation_adaptive_k(
    *,
    dataset_path: Path | str,
    time_budget_sec: float = 90.0,
    pair_schedule: tuple[int, ...] = (40, 80, 120, 200),
    unassigned_penalty: float = 1e6,
) -> EnrichedSolveResult:
    problem = build_enriched_problem(dataset_path)
    oracle = _build_oracle(problem)
    solver = EnrichedMILPAblationAdaptiveKSolver(
        AblationAdaptiveKConfig(
            time_budget_sec=time_budget_sec,
            pair_schedule=pair_schedule,
            unassigned_penalty=unassigned_penalty,
        )
    )
    result = solver.solve(problem=problem, oracle=oracle)
    return finalize_enriched_result(problem=problem, result=result, oracle=oracle)


def solve_enriched_milp_ablation_penalty_sweep(
    *,
    dataset_path: Path | str,
    time_budget_sec: float = 90.0,
    max_pairs_per_task: int = 160,
    penalty_schedule: tuple[float, ...] = (1e5, 1e6, 1e7, 1e8),
) -> EnrichedSolveResult:
    problem = build_enriched_problem(dataset_path)
    oracle = _build_oracle(problem)
    solver = EnrichedMILPAblationPenaltySweepSolver(
        AblationPenaltySweepConfig(
            time_budget_sec=time_budget_sec,
            max_pairs_per_task=max_pairs_per_task,
            penalty_schedule=penalty_schedule,
        )
    )
    result = solver.solve(problem=problem, oracle=oracle)
    return finalize_enriched_result(problem=problem, result=result, oracle=oracle)


def solve_enriched_milp_ablation_zone_bundle(
    *,
    dataset_path: Path | str,
    time_limit_sec_per_zone: int = 20,
    max_runtime_sec: float | None = None,
    max_pairs_per_bundle: int = 120,
    bundle_fill_factor: float = 0.9,
    bundle_max_tasks: int = 8,
    unassigned_penalty: float = 1e5,
    objective: str = "tasks",
) -> EnrichedSolveResult:
    problem = build_enriched_problem(dataset_path)
    oracle = _build_oracle(problem)
    solver = EnrichedMILPAblationZoneBundleSolver(
        AblationZoneBundleConfig(
            time_limit_sec_per_zone=time_limit_sec_per_zone,
            max_runtime_sec=max_runtime_sec,
            max_pairs_per_bundle=max_pairs_per_bundle,
            bundle_fill_factor=bundle_fill_factor,
            bundle_max_tasks=bundle_max_tasks,
            unassigned_penalty=unassigned_penalty,
            objective=objective,
        )
    )
    result = solver.solve(problem=problem, oracle=oracle)
    return finalize_enriched_result(problem=problem, result=result, oracle=oracle)


def solve_enriched_milp_ablation_portfolio(
    *,
    dataset_path: Path | str,
    time_budget_sec: float = 60.0,
    max_starts: int = 12,
    per_start_time_limit_sec: int = 8,
    max_pairs_per_task: int = 80,
    jitter: float = 0.08,
    seed: int = 42,
    unassigned_penalty: float = 1e6,
) -> EnrichedSolveResult:
    problem = build_enriched_problem(dataset_path)
    oracle = _build_oracle(problem)
    solver = EnrichedMILPAblationPortfolioSolver(
        AblationPortfolioConfig(
            time_budget_sec=time_budget_sec,
            max_starts=max_starts,
            per_start_time_limit_sec=per_start_time_limit_sec,
            max_pairs_per_task=max_pairs_per_task,
            jitter=jitter,
            seed=seed,
            unassigned_penalty=unassigned_penalty,
        )
    )
    result = solver.solve(problem=problem, oracle=oracle)
    return finalize_enriched_result(problem=problem, result=result, oracle=oracle)


def solve_enriched_milp_lexicographic_2stage(
    *,
    dataset_path: Path | str,
    time_budget_sec: float = 120.0,
    max_pairs_per_task: int = 180,
    unassigned_penalty: float = 1e6,
) -> EnrichedSolveResult:
    problem = build_enriched_problem(dataset_path)
    oracle = _build_oracle(problem)
    solver = EnrichedMILPAblationLexicographic2StageSolver(
        AblationLexicographic2StageConfig(
            time_budget_sec=time_budget_sec,
            max_pairs_per_task=max_pairs_per_task,
            unassigned_penalty=unassigned_penalty,
        )
    )
    result = solver.solve(problem=problem, oracle=oracle)
    return finalize_enriched_result(problem=problem, result=result, oracle=oracle)


def solve_enriched_milp_critical_first(
    *,
    dataset_path: Path | str,
    time_limit_sec: int = 90,
    max_pairs_per_task: int = 200,
    unassigned_penalty: float = 1e6,
) -> EnrichedSolveResult:
    problem = build_enriched_problem(dataset_path)
    oracle = _build_oracle(problem)
    solver = EnrichedMILPAblationCriticalFirstSolver(
        AblationCriticalFirstConfig(
            time_limit_sec=time_limit_sec,
            max_pairs_per_task=max_pairs_per_task,
            unassigned_penalty=unassigned_penalty,
        )
    )
    result = solver.solve(problem=problem, oracle=oracle)
    return finalize_enriched_result(problem=problem, result=result, oracle=oracle)


def solve_enriched_milp_zone_quota(
    *,
    dataset_path: Path | str,
    time_limit_sec: int = 90,
    max_pairs_per_task: int = 180,
    unassigned_penalty: float = 1e6,
    zone_min_coverage_ratio: float = 0.92,
) -> EnrichedSolveResult:
    problem = build_enriched_problem(dataset_path)
    oracle = _build_oracle(problem)
    solver = EnrichedMILPAblationZoneQuotaSolver(
        AblationZoneQuotaConfig(
            time_limit_sec=time_limit_sec,
            max_pairs_per_task=max_pairs_per_task,
            unassigned_penalty=unassigned_penalty,
            zone_min_coverage_ratio=zone_min_coverage_ratio,
        )
    )
    result = solver.solve(problem=problem, oracle=oracle)
    return finalize_enriched_result(problem=problem, result=result, oracle=oracle)


def solve_enriched_milp_adaptive_hardness(
    *,
    dataset_path: Path | str,
    time_limit_sec: int = 90,
    max_pairs_per_task: int = 200,
    unassigned_penalty: float = 1e6,
) -> EnrichedSolveResult:
    problem = build_enriched_problem(dataset_path)
    oracle = _build_oracle(problem)
    solver = EnrichedMILPAblationAdaptiveHardnessSolver(
        AblationAdaptiveHardnessConfig(
            time_limit_sec=time_limit_sec,
            max_pairs_per_task=max_pairs_per_task,
            unassigned_penalty=unassigned_penalty,
        )
    )
    result = solver.solve(problem=problem, oracle=oracle)
    return finalize_enriched_result(problem=problem, result=result, oracle=oracle)


def solve_enriched_milp_time_windowed(
    *,
    dataset_path: Path | str,
    time_budget_sec: float = 120.0,
    max_pairs_per_task: int = 180,
    unassigned_penalty: float = 1e6,
) -> EnrichedSolveResult:
    problem = build_enriched_problem(dataset_path)
    oracle = _build_oracle(problem)
    solver = EnrichedMILPAblationTimeWindowedSolver(
        AblationTimeWindowedConfig(
            time_budget_sec=time_budget_sec,
            max_pairs_per_task=max_pairs_per_task,
            unassigned_penalty=unassigned_penalty,
        )
    )
    result = solver.solve(problem=problem, oracle=oracle)
    return finalize_enriched_result(problem=problem, result=result, oracle=oracle)


def solve_enriched_milp_lns_rounds(
    *,
    dataset_path: Path | str,
    time_budget_sec: float = 120.0,
    max_pairs_per_task: int = 180,
    unassigned_penalty: float = 1e6,
    max_rounds: int = 6,
) -> EnrichedSolveResult:
    problem = build_enriched_problem(dataset_path)
    oracle = _build_oracle(problem)
    solver = EnrichedMILPAblationLNSSolver(
        AblationLNSConfig(
            time_budget_sec=time_budget_sec,
            max_pairs_per_task=max_pairs_per_task,
            unassigned_penalty=unassigned_penalty,
            max_rounds=max_rounds,
        )
    )
    result = solver.solve(problem=problem, oracle=oracle)
    return finalize_enriched_result(problem=problem, result=result, oracle=oracle)


def solve_enriched_milp_hybrid_seeded(
    *,
    dataset_path: Path | str,
    time_budget_sec: float = 120.0,
    max_pairs_per_task: int = 200,
    unassigned_penalty: float = 1e6,
) -> EnrichedSolveResult:
    problem = build_enriched_problem(dataset_path)
    oracle = _build_oracle(problem)
    solver = EnrichedMILPAblationHybridSeededSolver(
        AblationHybridSeededConfig(
            time_budget_sec=time_budget_sec,
            max_pairs_per_task=max_pairs_per_task,
            unassigned_penalty=unassigned_penalty,
        )
    )
    result = solver.solve(problem=problem, oracle=oracle)
    return finalize_enriched_result(problem=problem, result=result, oracle=oracle)


def solve_enriched_milp_batch_then_milp(
    *,
    dataset_path: Path | str,
    time_budget_sec: float = 90.0,
    bundle_max_tasks: int = 16,
    bundle_fill_factor: float = 0.95,
    max_pairs_per_bundle: int = 180,
    max_pairs_per_task: int = 220,
    unassigned_penalty: float = 1e6,
) -> EnrichedSolveResult:
    problem = build_enriched_problem(dataset_path)
    oracle = _build_oracle(problem)
    solver = EnrichedMILPAblationBatchThenMilpSolver(
        AblationBatchThenMilpConfig(
            time_budget_sec=time_budget_sec,
            bundle_max_tasks=bundle_max_tasks,
            bundle_fill_factor=bundle_fill_factor,
            max_pairs_per_bundle=max_pairs_per_bundle,
            max_pairs_per_task=max_pairs_per_task,
            unassigned_penalty=unassigned_penalty,
        )
    )
    result = solver.solve(problem=problem, oracle=oracle)
    return finalize_enriched_result(problem=problem, result=result, oracle=oracle)


def solve_enriched_milp_batch_cascaded(
    *,
    dataset_path: Path | str,
    time_budget_sec: float = 120.0,
    unassigned_penalty: float = 1e6,
) -> EnrichedSolveResult:
    problem = build_enriched_problem(dataset_path)
    oracle = _build_oracle(problem)
    solver = EnrichedMILPAblationBatchCascadedSolver(
        AblationBatchCascadedConfig(
            time_budget_sec=time_budget_sec,
            unassigned_penalty=unassigned_penalty,
        )
    )
    result = solver.solve(problem=problem, oracle=oracle)
    return finalize_enriched_result(problem=problem, result=result, oracle=oracle)


def solve_enriched_milp_batch_portfolio(
    *,
    dataset_path: Path | str,
    time_budget_sec: float = 120.0,
    unassigned_penalty: float = 1e6,
) -> EnrichedSolveResult:
    problem = build_enriched_problem(dataset_path)
    oracle = _build_oracle(problem)
    solver = EnrichedMILPAblationBatchPortfolioSolver(
        AblationBatchPortfolioConfig(
            time_budget_sec=time_budget_sec,
            unassigned_penalty=unassigned_penalty,
        )
    )
    result = solver.solve(problem=problem, oracle=oracle)
    return finalize_enriched_result(problem=problem, result=result, oracle=oracle)


def solve_enriched_legacy_gap_vrp(
    *,
    dataset_path: Path | str,
    step1_method: str = "dataset",
    gap_iter: int = 40,
    use_repair: bool = True,
    show_progress: bool = False,
    verbose: bool = False,
    progress_hook: Callable[[str], None] | None = None,
) -> EnrichedSolveResult:
    problem = build_enriched_problem(dataset_path)
    oracle = _build_oracle(problem)
    result = _solve_enriched_legacy_gap_vrp(
        dataset_path=dataset_path,
        config=EnrichedLegacyGapConfig(
            step1_method=step1_method,
            gap_iter=gap_iter,
            use_repair=use_repair,
            show_progress=show_progress,
            verbose=verbose,
            progress_hook=progress_hook,
        ),
    )
    return finalize_enriched_result(problem=problem, result=result, oracle=oracle)


def solve_enriched_legacy_milp(
    *,
    dataset_path: Path | str,
    time_limit_sec: int = 60,
    unassigned_penalty: float = 1e5,
    show_progress: bool = False,
    progress_hook: Callable[[str], None] | None = None,
) -> EnrichedSolveResult:
    problem = build_enriched_problem(dataset_path)
    oracle = _build_oracle(problem)
    result = _solve_enriched_legacy_milp(
        dataset_path=dataset_path,
        config=EnrichedLegacyMILPConfig(
            time_limit_sec=time_limit_sec,
            unassigned_penalty=unassigned_penalty,
            show_progress=show_progress,
            progress_hook=progress_hook,
        ),
    )
    return finalize_enriched_result(problem=problem, result=result, oracle=oracle)


def benchmark_enriched_algorithms(
    *,
    dataset_path: Path | str,
    milp_kwargs: dict[str, Any] | None = None,
    gap_kwargs: dict[str, Any] | None = None,
    rr_kwargs: dict[str, Any] | None = None,
    grasp_kwargs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    milp_res = solve_enriched_milp(dataset_path=dataset_path, **(milp_kwargs or {}))
    results.append(milp_res.as_dict())

    gap_res = solve_enriched_batched_greedy(dataset_path=dataset_path, **(gap_kwargs or {}))
    results.append(gap_res.as_dict())

    rr_res = solve_enriched_stochastic_rr(dataset_path=dataset_path, **(rr_kwargs or {}))
    results.append(rr_res.as_dict())

    grasp_res = solve_enriched_stochastic_grasp(dataset_path=dataset_path, **(grasp_kwargs or {}))
    results.append(grasp_res.as_dict())

    return results
