from __future__ import annotations

from dataclasses import dataclass
import time

from .distance_oracle import DistanceOracleWithFallback
from .milp_decomp_solver_v1 import EnrichedMILPDecompConfig, EnrichedMILPDecompSolver
from .milp_solver_v2 import EnrichedMILPConfig, EnrichedMILPSolver
from .milp_stochastic_solver_v1 import EnrichedMILPStochasticConfig, EnrichedMILPStochasticSolver
from .problem import EnrichedProblem
from .types import AgentUsage, EnrichedAgent, EnrichedSolveResult


def _coverage(result: EnrichedSolveResult, total_tasks: int) -> int:
    return max(0, int(total_tasks) - len(result.unassigned_task_ids))


def _pick_better(
    *,
    best: EnrichedSolveResult | None,
    cur: EnrichedSolveResult,
    total_tasks: int,
) -> EnrichedSolveResult:
    if best is None:
        return cur
    b_cov = _coverage(best, total_tasks)
    c_cov = _coverage(cur, total_tasks)
    if c_cov > b_cov:
        return cur
    if c_cov < b_cov:
        return best
    b_d = best.as_dict()
    c_d = cur.as_dict()
    b_km = float(b_d.get("total_km") or 1e18)
    c_km = float(c_d.get("total_km") or 1e18)
    if c_km < b_km:
        return cur
    return best


def _merge_results(
    *,
    problem: EnrichedProblem,
    first: EnrichedSolveResult,
    second: EnrichedSolveResult,
    algorithm: str,
    runtime_sec: float,
    details: dict[str, object] | None = None,
) -> EnrichedSolveResult:
    usage: dict[str, AgentUsage] = {}
    for agent in problem.agents:
        a0 = first.agent_usage.get(agent.agent_id, AgentUsage(agent_id=agent.agent_id))
        a1 = second.agent_usage.get(agent.agent_id, AgentUsage(agent_id=agent.agent_id))
        usage[agent.agent_id] = AgentUsage(
            agent_id=agent.agent_id,
            tasks=list(a0.tasks) + list(a1.tasks),
            total_km=float(a0.total_km) + float(a1.total_km),
            total_hours=float(a0.total_hours) + float(a1.total_hours),
            loaded_km=float(a0.loaded_km) + float(a1.loaded_km),
        )
    merged = EnrichedSolveResult(
        algorithm=algorithm,
        feasible=False,
        routes=list(first.routes) + list(second.routes),
        unassigned_task_ids=sorted(set(second.unassigned_task_ids)),
        agent_usage=usage,
        runtime_sec=runtime_sec,
        details=dict(first.details),
    )
    merged.details.update(second.details)
    if details:
        merged.details.update(details)
    return merged


def _build_residual_problem(problem: EnrichedProblem, result: EnrichedSolveResult) -> EnrichedProblem:
    assigned: set[str] = set()
    used_mass: dict[str, float] = {}
    used_vol: dict[str, float] = {}
    for route in result.routes:
        for tid in route.task_ids:
            assigned.add(tid)
    task_by_id = {t.task_id: t for t in problem.tasks}
    for tid in assigned:
        task = task_by_id.get(tid)
        if task is None:
            continue
        did = task.destination_node_id
        used_mass[did] = used_mass.get(did, 0.0) + float(task.mass_tons)
        used_vol[did] = used_vol.get(did, 0.0) + float(task.volume_raw_m3)

    residual_tasks = [t for t in problem.tasks if t.task_id not in assigned]

    residual_agents: list[EnrichedAgent] = []
    for agent in problem.agents:
        u = result.agent_usage.get(agent.agent_id, AgentUsage(agent_id=agent.agent_id))
        rem_km = max(0.0, float(agent.max_daily_km) - float(u.total_km))
        rem_h = max(0.0, float(agent.max_shift_hours) - float(u.total_hours))
        if rem_km <= 1e-9 or rem_h <= 1e-9:
            continue
        residual_agents.append(
            EnrichedAgent(
                agent_id=agent.agent_id,
                vehicle_type=agent.vehicle_type,
                capacity_tons=float(agent.capacity_tons),
                max_raw_volume_m3=float(agent.max_raw_volume_m3),
                is_compact=bool(agent.is_compact),
                is_available=bool(agent.is_available),
                depot_node_id=agent.depot_node_id,
                cap_container_types=agent.cap_container_types,
                max_daily_km=rem_km,
                max_shift_hours=rem_h,
                avg_speed_kmph=float(agent.avg_speed_kmph),
                zone_num=agent.zone_num,
            )
        )

    residual_obj_mass: dict[str, float] = {}
    residual_obj_vol: dict[str, float] = {}
    for oid, cap in problem.object_day_capacity_tons.items():
        residual_obj_mass[oid] = max(0.0, float(cap) - used_mass.get(oid, 0.0))
    for oid, cap in problem.object_day_capacity_volume_m3.items():
        residual_obj_vol[oid] = max(0.0, float(cap) - used_vol.get(oid, 0.0))

    return EnrichedProblem(
        dataset_path=problem.dataset_path,
        payload=problem.payload,
        agents=residual_agents,
        tasks=residual_tasks,
        task_mass_by_id={t.task_id: float(t.mass_tons) for t in residual_tasks},
        object_day_capacity_tons=residual_obj_mass,
        object_day_capacity_volume_m3=residual_obj_vol,
    )


@dataclass(frozen=True)
class AblationBaselineConfig:
    time_limit_sec: int = 30
    max_pairs_per_task: int = 80
    unassigned_penalty: float = 1e6


class EnrichedMILPAblationBaselineSolver:
    """Strict baseline MILP without fallback/repair."""

    def __init__(self, config: AblationBaselineConfig | None = None) -> None:
        self.config = config or AblationBaselineConfig()

    def solve(self, *, problem: EnrichedProblem, oracle: DistanceOracleWithFallback) -> EnrichedSolveResult:
        solver = EnrichedMILPSolver(
            EnrichedMILPConfig(
                time_limit_sec=self.config.time_limit_sec,
                max_pairs_per_task=self.config.max_pairs_per_task,
                unassigned_penalty=self.config.unassigned_penalty,
            )
        )
        result = solver.solve(problem=problem, oracle=oracle)
        result.algorithm = "idea_milp_baseline_strict_v1"
        return result


@dataclass(frozen=True)
class AblationAdaptiveKConfig:
    time_budget_sec: float = 90.0
    pair_schedule: tuple[int, ...] = (40, 80, 120, 200)
    unassigned_penalty: float = 1e6


class EnrichedMILPAblationAdaptiveKSolver:
    """Run MILP with progressively wider candidate pools and keep best coverage."""

    def __init__(self, config: AblationAdaptiveKConfig | None = None) -> None:
        self.config = config or AblationAdaptiveKConfig()

    def solve(self, *, problem: EnrichedProblem, oracle: DistanceOracleWithFallback) -> EnrichedSolveResult:
        t0 = time.perf_counter()
        best: EnrichedSolveResult | None = None
        total_tasks = len(problem.tasks)
        schedule = self.config.pair_schedule or (80,)
        for i, k in enumerate(schedule):
            elapsed = time.perf_counter() - t0
            remain = self.config.time_budget_sec - elapsed
            if remain <= 1.0:
                break
            stages_left = max(1, len(schedule) - i)
            tl = max(5, int(remain / stages_left))
            solver = EnrichedMILPSolver(
                EnrichedMILPConfig(
                    time_limit_sec=tl,
                    max_pairs_per_task=int(k),
                    unassigned_penalty=self.config.unassigned_penalty,
                )
            )
            cur = solver.solve(problem=problem, oracle=oracle)
            cur.details["adaptive_stage"] = {
                "k": int(k),
                "time_limit_sec": int(tl),
            }
            best = _pick_better(best=best, cur=cur, total_tasks=total_tasks)
            if best.feasible:
                break
        if best is None:
            best = EnrichedMILPSolver().solve(problem=problem, oracle=oracle)
        best.algorithm = "idea_milp_adaptive_k_v1"
        best.runtime_sec = time.perf_counter() - t0
        return best


@dataclass(frozen=True)
class AblationPenaltySweepConfig:
    time_budget_sec: float = 90.0
    max_pairs_per_task: int = 160
    penalty_schedule: tuple[float, ...] = (1e5, 1e6, 1e7, 1e8)


class EnrichedMILPAblationPenaltySweepSolver:
    """Sweep unassigned penalties and keep best coverage."""

    def __init__(self, config: AblationPenaltySweepConfig | None = None) -> None:
        self.config = config or AblationPenaltySweepConfig()

    def solve(self, *, problem: EnrichedProblem, oracle: DistanceOracleWithFallback) -> EnrichedSolveResult:
        t0 = time.perf_counter()
        best: EnrichedSolveResult | None = None
        total_tasks = len(problem.tasks)
        schedule = self.config.penalty_schedule or (1e6,)
        for i, penalty in enumerate(schedule):
            elapsed = time.perf_counter() - t0
            remain = self.config.time_budget_sec - elapsed
            if remain <= 1.0:
                break
            stages_left = max(1, len(schedule) - i)
            tl = max(5, int(remain / stages_left))
            solver = EnrichedMILPSolver(
                EnrichedMILPConfig(
                    time_limit_sec=tl,
                    max_pairs_per_task=self.config.max_pairs_per_task,
                    unassigned_penalty=float(penalty),
                )
            )
            cur = solver.solve(problem=problem, oracle=oracle)
            cur.details["penalty_stage"] = {
                "unassigned_penalty": float(penalty),
                "time_limit_sec": int(tl),
            }
            best = _pick_better(best=best, cur=cur, total_tasks=total_tasks)
            if best.feasible:
                break
        if best is None:
            best = EnrichedMILPSolver().solve(problem=problem, oracle=oracle)
        best.algorithm = "idea_milp_penalty_sweep_v1"
        best.runtime_sec = time.perf_counter() - t0
        return best


@dataclass(frozen=True)
class AblationZoneBundleConfig:
    time_limit_sec_per_zone: int = 20
    max_runtime_sec: float | None = None
    max_pairs_per_bundle: int = 120
    bundle_fill_factor: float = 0.9
    bundle_max_tasks: int = 8
    unassigned_penalty: float = 1e5
    objective: str = "tasks"


class EnrichedMILPAblationZoneBundleSolver:
    """Zone decomposition + bundle MILP (multi-task trip approximation)."""

    def __init__(self, config: AblationZoneBundleConfig | None = None) -> None:
        self.config = config or AblationZoneBundleConfig()

    def solve(self, *, problem: EnrichedProblem, oracle: DistanceOracleWithFallback) -> EnrichedSolveResult:
        solver = EnrichedMILPDecompSolver(
            EnrichedMILPDecompConfig(
                objective=self.config.objective,
                unassigned_penalty=self.config.unassigned_penalty,
                time_limit_sec_per_zone=self.config.time_limit_sec_per_zone,
                max_runtime_sec=self.config.max_runtime_sec,
                max_pairs_per_bundle=self.config.max_pairs_per_bundle,
                bundle_fill_factor=self.config.bundle_fill_factor,
                bundle_max_tasks=self.config.bundle_max_tasks,
                verbose=False,
            )
        )
        result = solver.solve(problem=problem, oracle=oracle)
        result.algorithm = "idea_milp_zone_bundle_v1"
        return result


@dataclass(frozen=True)
class AblationPortfolioConfig:
    time_budget_sec: float = 60.0
    max_starts: int = 12
    per_start_time_limit_sec: int = 8
    max_pairs_per_task: int = 80
    jitter: float = 0.08
    seed: int = 42
    unassigned_penalty: float = 1e6


class EnrichedMILPAblationPortfolioSolver:
    """Stochastic MILP portfolio (multi-start with random candidate tie-breaking)."""

    def __init__(self, config: AblationPortfolioConfig | None = None) -> None:
        self.config = config or AblationPortfolioConfig()

    def solve(self, *, problem: EnrichedProblem, oracle: DistanceOracleWithFallback) -> EnrichedSolveResult:
        solver = EnrichedMILPStochasticSolver(
            EnrichedMILPStochasticConfig(
                time_budget_sec=self.config.time_budget_sec,
                max_starts=self.config.max_starts,
                per_start_time_limit_sec=self.config.per_start_time_limit_sec,
                max_pairs_per_task=self.config.max_pairs_per_task,
                jitter=self.config.jitter,
                seed=self.config.seed,
                unassigned_penalty=self.config.unassigned_penalty,
            )
        )
        result = solver.solve(problem=problem, oracle=oracle)
        result.algorithm = "idea_milp_portfolio_v1"
        return result


@dataclass(frozen=True)
class AblationLexicographic2StageConfig:
    time_budget_sec: float = 120.0
    max_pairs_per_task: int = 180
    unassigned_penalty: float = 1e6


class EnrichedMILPAblationLexicographic2StageSolver:
    """2-stage MILP: (1) min unassigned, (2) min km with fixed best coverage."""

    def __init__(self, config: AblationLexicographic2StageConfig | None = None) -> None:
        self.config = config or AblationLexicographic2StageConfig()

    def solve(self, *, problem: EnrichedProblem, oracle: DistanceOracleWithFallback) -> EnrichedSolveResult:
        t0 = time.perf_counter()
        stage1_t = max(10, int(self.config.time_budget_sec * 0.55))
        stage2_t = max(5, int(self.config.time_budget_sec - stage1_t))

        s1 = EnrichedMILPSolver(
            EnrichedMILPConfig(
                time_limit_sec=stage1_t,
                max_pairs_per_task=self.config.max_pairs_per_task,
                unassigned_penalty=self.config.unassigned_penalty,
                x_cost_weight=0.0,
                y_cost_weight=1.0,
                prioritize_hard_tasks=True,
                dynamic_candidate_caps=True,
            )
        ).solve(problem=problem, oracle=oracle)
        best_unassigned = len(s1.unassigned_task_ids)

        s2 = EnrichedMILPSolver(
            EnrichedMILPConfig(
                time_limit_sec=stage2_t,
                max_pairs_per_task=self.config.max_pairs_per_task,
                unassigned_penalty=self.config.unassigned_penalty,
                x_cost_weight=1.0,
                y_cost_weight=1e-6,
                max_unassigned_tasks=best_unassigned,
                dynamic_candidate_caps=True,
            )
        ).solve(problem=problem, oracle=oracle)

        best = _pick_better(best=s1, cur=s2, total_tasks=len(problem.tasks))
        best.algorithm = "idea_milp_lexicographic_2stage_v1"
        best.runtime_sec = time.perf_counter() - t0
        best.details["lexicographic"] = {
            "stage1_time_sec": stage1_t,
            "stage2_time_sec": stage2_t,
            "stage1_unassigned": best_unassigned,
        }
        return best


@dataclass(frozen=True)
class AblationCriticalFirstConfig:
    time_limit_sec: int = 90
    max_pairs_per_task: int = 200
    unassigned_penalty: float = 1e6


class EnrichedMILPAblationCriticalFirstSolver:
    """Hard-task-focused MILP via candidate-aware unassigned penalties."""

    def __init__(self, config: AblationCriticalFirstConfig | None = None) -> None:
        self.config = config or AblationCriticalFirstConfig()

    def solve(self, *, problem: EnrichedProblem, oracle: DistanceOracleWithFallback) -> EnrichedSolveResult:
        result = EnrichedMILPSolver(
            EnrichedMILPConfig(
                time_limit_sec=self.config.time_limit_sec,
                max_pairs_per_task=self.config.max_pairs_per_task,
                unassigned_penalty=self.config.unassigned_penalty,
                prioritize_hard_tasks=True,
                dynamic_candidate_caps=True,
            )
        ).solve(problem=problem, oracle=oracle)
        result.algorithm = "idea_milp_critical_first_v1"
        return result


@dataclass(frozen=True)
class AblationZoneQuotaConfig:
    time_limit_sec: int = 90
    max_pairs_per_task: int = 180
    unassigned_penalty: float = 1e6
    zone_min_coverage_ratio: float = 0.92


class EnrichedMILPAblationZoneQuotaSolver:
    """MILP with per-zone minimal coverage constraints."""

    def __init__(self, config: AblationZoneQuotaConfig | None = None) -> None:
        self.config = config or AblationZoneQuotaConfig()

    def solve(self, *, problem: EnrichedProblem, oracle: DistanceOracleWithFallback) -> EnrichedSolveResult:
        result = EnrichedMILPSolver(
            EnrichedMILPConfig(
                time_limit_sec=self.config.time_limit_sec,
                max_pairs_per_task=self.config.max_pairs_per_task,
                unassigned_penalty=self.config.unassigned_penalty,
                zone_min_coverage_ratio=self.config.zone_min_coverage_ratio,
                dynamic_candidate_caps=True,
            )
        ).solve(problem=problem, oracle=oracle)
        result.algorithm = "idea_milp_zone_quota_v1"
        return result


@dataclass(frozen=True)
class AblationAdaptiveHardnessConfig:
    time_limit_sec: int = 90
    max_pairs_per_task: int = 200
    unassigned_penalty: float = 1e6


class EnrichedMILPAblationAdaptiveHardnessSolver:
    """Dynamic candidate caps by task hardness."""

    def __init__(self, config: AblationAdaptiveHardnessConfig | None = None) -> None:
        self.config = config or AblationAdaptiveHardnessConfig()

    def solve(self, *, problem: EnrichedProblem, oracle: DistanceOracleWithFallback) -> EnrichedSolveResult:
        result = EnrichedMILPSolver(
            EnrichedMILPConfig(
                time_limit_sec=self.config.time_limit_sec,
                max_pairs_per_task=self.config.max_pairs_per_task,
                unassigned_penalty=self.config.unassigned_penalty,
                dynamic_candidate_caps=True,
                dynamic_min_k=60,
                dynamic_max_k=280,
                dynamic_hard_threshold=35,
            )
        ).solve(problem=problem, oracle=oracle)
        result.algorithm = "idea_milp_adaptive_hardness_v1"
        return result


@dataclass(frozen=True)
class AblationTimeWindowedConfig:
    time_budget_sec: float = 120.0
    max_pairs_per_task: int = 180
    unassigned_penalty: float = 1e6


class EnrichedMILPAblationTimeWindowedSolver:
    """Anytime schedule with changing objective weights."""

    def __init__(self, config: AblationTimeWindowedConfig | None = None) -> None:
        self.config = config or AblationTimeWindowedConfig()

    def solve(self, *, problem: EnrichedProblem, oracle: DistanceOracleWithFallback) -> EnrichedSolveResult:
        t0 = time.perf_counter()
        phases = [
            (0.35, 0.0, 1.0),
            (0.35, 0.2, 1.0),
            (0.30, 1.0, 0.2),
        ]
        best: EnrichedSolveResult | None = None
        for frac, xw, yw in phases:
            tl = max(5, int(self.config.time_budget_sec * frac))
            cur = EnrichedMILPSolver(
                EnrichedMILPConfig(
                    time_limit_sec=tl,
                    max_pairs_per_task=self.config.max_pairs_per_task,
                    unassigned_penalty=self.config.unassigned_penalty,
                    x_cost_weight=float(xw),
                    y_cost_weight=float(yw),
                    dynamic_candidate_caps=True,
                    prioritize_hard_tasks=True,
                )
            ).solve(problem=problem, oracle=oracle)
            best = _pick_better(best=best, cur=cur, total_tasks=len(problem.tasks))
            if best.feasible:
                break
        if best is None:
            best = EnrichedMILPSolver().solve(problem=problem, oracle=oracle)
        best.algorithm = "idea_milp_time_windowed_v1"
        best.runtime_sec = time.perf_counter() - t0
        return best


@dataclass(frozen=True)
class AblationLNSConfig:
    time_budget_sec: float = 120.0
    max_pairs_per_task: int = 180
    unassigned_penalty: float = 1e6
    max_rounds: int = 6


class EnrichedMILPAblationLNSSolver:
    """MILP rounds with tightening max-unassigned bound."""

    def __init__(self, config: AblationLNSConfig | None = None) -> None:
        self.config = config or AblationLNSConfig()

    def solve(self, *, problem: EnrichedProblem, oracle: DistanceOracleWithFallback) -> EnrichedSolveResult:
        t0 = time.perf_counter()
        best: EnrichedSolveResult | None = None
        target: int | None = None
        for i in range(self.config.max_rounds):
            elapsed = time.perf_counter() - t0
            remain = self.config.time_budget_sec - elapsed
            if remain <= 2.0:
                break
            tl = max(5, int(remain / max(1, self.config.max_rounds - i)))
            cur = EnrichedMILPSolver(
                EnrichedMILPConfig(
                    time_limit_sec=tl,
                    max_pairs_per_task=self.config.max_pairs_per_task,
                    unassigned_penalty=self.config.unassigned_penalty,
                    max_unassigned_tasks=target,
                    random_jitter=0.03 * (i + 1),
                    random_seed=42 + i * 13,
                    dynamic_candidate_caps=True,
                )
            ).solve(problem=problem, oracle=oracle)
            best = _pick_better(best=best, cur=cur, total_tasks=len(problem.tasks))
            cur_un = len(cur.unassigned_task_ids)
            if target is None:
                target = cur_un
            else:
                target = max(0, min(target, cur_un) - 1)
            if best.feasible:
                break
        if best is None:
            best = EnrichedMILPSolver().solve(problem=problem, oracle=oracle)
        best.algorithm = "idea_milp_lns_rounds_v1"
        best.runtime_sec = time.perf_counter() - t0
        return best


@dataclass(frozen=True)
class AblationHybridSeededConfig:
    time_budget_sec: float = 120.0
    max_pairs_per_task: int = 200
    unassigned_penalty: float = 1e6


class EnrichedMILPAblationHybridSeededSolver:
    """Seed from zone-bundle coverage, then tighten MILP by max-unassigned bound."""

    def __init__(self, config: AblationHybridSeededConfig | None = None) -> None:
        self.config = config or AblationHybridSeededConfig()

    def solve(self, *, problem: EnrichedProblem, oracle: DistanceOracleWithFallback) -> EnrichedSolveResult:
        t0 = time.perf_counter()
        seed_t = max(10, int(self.config.time_budget_sec * 0.35))
        main_t = max(10, int(self.config.time_budget_sec - seed_t))
        seed = EnrichedMILPDecompSolver(
            EnrichedMILPDecompConfig(
                objective="tasks",
                time_limit_sec_per_zone=max(4, int(seed_t / 4)),
                max_runtime_sec=float(seed_t),
                max_pairs_per_bundle=120,
                bundle_fill_factor=0.9,
                bundle_max_tasks=8,
                unassigned_penalty=1e5,
                verbose=False,
            )
        ).solve(problem=problem, oracle=oracle)
        bound = len(seed.unassigned_task_ids)

        main = EnrichedMILPSolver(
            EnrichedMILPConfig(
                time_limit_sec=main_t,
                max_runtime_sec=float(main_t),
                max_pairs_per_task=self.config.max_pairs_per_task,
                unassigned_penalty=self.config.unassigned_penalty,
                max_unassigned_tasks=bound,
                dynamic_candidate_caps=True,
                prioritize_hard_tasks=True,
            )
        ).solve(problem=problem, oracle=oracle)
        best = _pick_better(best=seed, cur=main, total_tasks=len(problem.tasks))
        best.algorithm = "idea_milp_hybrid_seeded_v1"
        best.runtime_sec = time.perf_counter() - t0
        best.details["hybrid_seed"] = {"seed_unassigned": bound}
        return best


@dataclass(frozen=True)
class AblationBatchThenMilpConfig:
    time_budget_sec: float = 90.0
    bundle_max_tasks: int = 16
    bundle_fill_factor: float = 0.95
    max_pairs_per_bundle: int = 180
    max_pairs_per_task: int = 220
    unassigned_penalty: float = 1e6


class EnrichedMILPAblationBatchThenMilpSolver:
    """Batch first to save km/h, then MILP only on residual tasks."""

    def __init__(self, config: AblationBatchThenMilpConfig | None = None) -> None:
        self.config = config or AblationBatchThenMilpConfig()

    def solve(self, *, problem: EnrichedProblem, oracle: DistanceOracleWithFallback) -> EnrichedSolveResult:
        t0 = time.perf_counter()
        seed_t = max(8, int(self.config.time_budget_sec * 0.55))
        main_t = max(8, int(self.config.time_budget_sec - seed_t))

        seed = EnrichedMILPDecompSolver(
            EnrichedMILPDecompConfig(
                objective="tasks",
                unassigned_penalty=1e5,
                time_limit_sec_per_zone=max(4, int(seed_t / 4)),
                max_runtime_sec=float(seed_t),
                max_pairs_per_bundle=self.config.max_pairs_per_bundle,
                bundle_fill_factor=self.config.bundle_fill_factor,
                bundle_max_tasks=self.config.bundle_max_tasks,
                verbose=False,
            )
        ).solve(problem=problem, oracle=oracle)

        if len(seed.unassigned_task_ids) == 0:
            seed.algorithm = "idea_batch_then_milp_v1"
            seed.runtime_sec = time.perf_counter() - t0
            return seed

        residual = _build_residual_problem(problem, seed)
        if not residual.tasks or not residual.agents:
            seed.algorithm = "idea_batch_then_milp_v1"
            seed.runtime_sec = time.perf_counter() - t0
            return seed

        tail = EnrichedMILPSolver(
            EnrichedMILPConfig(
                time_limit_sec=main_t,
                max_runtime_sec=float(main_t),
                max_pairs_per_task=self.config.max_pairs_per_task,
                unassigned_penalty=self.config.unassigned_penalty,
                dynamic_candidate_caps=True,
                prioritize_hard_tasks=True,
                random_jitter=0.02,
            )
        ).solve(problem=residual, oracle=oracle)
        merged = _merge_results(
            problem=problem,
            first=seed,
            second=tail,
            algorithm="idea_batch_then_milp_v1",
            runtime_sec=time.perf_counter() - t0,
            details={
                "batch_then_milp": {
                    "seed_unassigned": len(seed.unassigned_task_ids),
                    "tail_unassigned": len(tail.unassigned_task_ids),
                    "seed_time_sec": seed_t,
                    "tail_time_sec": main_t,
                }
            },
        )
        return merged


@dataclass(frozen=True)
class AblationBatchCascadedConfig:
    time_budget_sec: float = 120.0
    unassigned_penalty: float = 1e6


class EnrichedMILPAblationBatchCascadedSolver:
    """Two batching passes (coarse->fine) and MILP tail."""

    def __init__(self, config: AblationBatchCascadedConfig | None = None) -> None:
        self.config = config or AblationBatchCascadedConfig()

    def solve(self, *, problem: EnrichedProblem, oracle: DistanceOracleWithFallback) -> EnrichedSolveResult:
        t0 = time.perf_counter()
        b1_t = max(8, int(self.config.time_budget_sec * 0.35))
        b2_t = max(8, int(self.config.time_budget_sec * 0.35))
        tail_t = max(8, int(self.config.time_budget_sec - b1_t - b2_t))

        coarse = EnrichedMILPDecompSolver(
            EnrichedMILPDecompConfig(
                objective="tasks",
                unassigned_penalty=1e5,
                time_limit_sec_per_zone=max(4, int(b1_t / 4)),
                max_runtime_sec=float(b1_t),
                max_pairs_per_bundle=200,
                bundle_fill_factor=0.98,
                bundle_max_tasks=24,
                verbose=False,
            )
        ).solve(problem=problem, oracle=oracle)
        p1 = _build_residual_problem(problem, coarse)

        if p1.tasks and p1.agents:
            fine = EnrichedMILPDecompSolver(
                EnrichedMILPDecompConfig(
                    objective="tasks",
                    unassigned_penalty=1e5,
                    time_limit_sec_per_zone=max(4, int(b2_t / 4)),
                    max_runtime_sec=float(b2_t),
                    max_pairs_per_bundle=220,
                    bundle_fill_factor=0.92,
                    bundle_max_tasks=10,
                    verbose=False,
                )
            ).solve(problem=p1, oracle=oracle)
            merged12 = _merge_results(
                problem=problem,
                first=coarse,
                second=fine,
                algorithm="idea_batch_cascaded_v1",
                runtime_sec=0.0,
            )
        else:
            fine = EnrichedSolveResult(
                algorithm="empty",
                feasible=True,
                routes=[],
                unassigned_task_ids=[],
                agent_usage={a.agent_id: AgentUsage(agent_id=a.agent_id) for a in p1.agents},
                runtime_sec=0.0,
                details={},
            )
            merged12 = coarse

        p2 = _build_residual_problem(problem, merged12)
        if p2.tasks and p2.agents:
            tail = EnrichedMILPSolver(
                EnrichedMILPConfig(
                    time_limit_sec=tail_t,
                    max_runtime_sec=float(tail_t),
                    max_pairs_per_task=240,
                    unassigned_penalty=self.config.unassigned_penalty,
                    dynamic_candidate_caps=True,
                    prioritize_hard_tasks=True,
                    random_jitter=0.03,
                )
            ).solve(problem=p2, oracle=oracle)
            final = _merge_results(
                problem=problem,
                first=merged12,
                second=tail,
                algorithm="idea_batch_cascaded_v1",
                runtime_sec=time.perf_counter() - t0,
                details={
                    "batch_cascaded": {
                        "coarse_unassigned": len(coarse.unassigned_task_ids),
                        "fine_unassigned": len(fine.unassigned_task_ids),
                        "tail_unassigned": len(tail.unassigned_task_ids),
                        "coarse_time_sec": b1_t,
                        "fine_time_sec": b2_t,
                        "tail_time_sec": tail_t,
                    }
                },
            )
            return final

        merged12.algorithm = "idea_batch_cascaded_v1"
        merged12.runtime_sec = time.perf_counter() - t0
        return merged12


@dataclass(frozen=True)
class AblationBatchPortfolioConfig:
    time_budget_sec: float = 120.0
    unassigned_penalty: float = 1e6


class EnrichedMILPAblationBatchPortfolioSolver:
    """Portfolio over multiple batching shapes, best by coverage."""

    def __init__(self, config: AblationBatchPortfolioConfig | None = None) -> None:
        self.config = config or AblationBatchPortfolioConfig()

    def solve(self, *, problem: EnrichedProblem, oracle: DistanceOracleWithFallback) -> EnrichedSolveResult:
        t0 = time.perf_counter()
        total = len(problem.tasks)
        best: EnrichedSolveResult | None = None
        shapes = [
            (8, 0.88, 140),
            (12, 0.92, 180),
            (16, 0.95, 220),
        ]
        per = max(20, int(self.config.time_budget_sec / max(1, len(shapes))))
        for max_tasks, fill, pairs in shapes:
            cur = EnrichedMILPAblationBatchThenMilpSolver(
                AblationBatchThenMilpConfig(
                    time_budget_sec=float(per),
                    bundle_max_tasks=int(max_tasks),
                    bundle_fill_factor=float(fill),
                    max_pairs_per_bundle=int(pairs),
                    max_pairs_per_task=220,
                    unassigned_penalty=self.config.unassigned_penalty,
                )
            ).solve(problem=problem, oracle=oracle)
            best = _pick_better(best=best, cur=cur, total_tasks=total)
            if best is not None and len(best.unassigned_task_ids) == 0:
                break
        if best is None:
            best = EnrichedMILPSolver().solve(problem=problem, oracle=oracle)
        best.algorithm = "idea_batch_portfolio_v1"
        best.runtime_sec = time.perf_counter() - t0
        return best
