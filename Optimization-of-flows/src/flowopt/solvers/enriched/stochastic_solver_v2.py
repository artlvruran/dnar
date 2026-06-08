from __future__ import annotations

from dataclasses import dataclass
import time

from .distance_oracle import DistanceOracleWithFallback
from .gap_vrp_solver_v2 import EnrichedGapVRPConfig, EnrichedGapVRPSolver
from .problem import EnrichedProblem
from .types import EnrichedSolveResult


@dataclass(frozen=True)
class EnrichedStochasticConfig:
    time_budget_sec: float = 20.0
    max_starts: int = 12
    top_k_min: int = 2
    top_k_max: int = 30
    seed: int = 42


class EnrichedStochasticRRSolver:
    """Random-restart stochastic solver over GAP-v2 constructive policy."""

    def __init__(self, config: EnrichedStochasticConfig | None = None) -> None:
        self.config = config or EnrichedStochasticConfig()

    def solve(self, *, problem: EnrichedProblem, oracle: DistanceOracleWithFallback) -> EnrichedSolveResult:
        t0 = time.perf_counter()
        best: EnrichedSolveResult | None = None
        starts = 0
        while starts < self.config.max_starts and (time.perf_counter() - t0) < self.config.time_budget_sec:
            k = self.config.top_k_min + (starts % max(1, self.config.top_k_max - self.config.top_k_min + 1))
            solver = EnrichedGapVRPSolver(
                EnrichedGapVRPConfig(
                    random_seed=self.config.seed + starts * 97,
                    top_k_agents=k,
                    balance_penalty=0.03,
                )
            )
            result = solver.solve(problem=problem, oracle=oracle)
            starts += 1
            if best is None:
                best = result
                continue
            # prefer better coverage, then lower total km
            b = best.as_dict()
            r = result.as_dict()
            b_cov = (len(problem.tasks) - int(b["unassigned_tasks"]))
            r_cov = (len(problem.tasks) - int(r["unassigned_tasks"]))
            b_km = float(b.get("total_km") or 1e18)
            r_km = float(r.get("total_km") or 1e18)
            if (r_cov > b_cov) or (r_cov == b_cov and r_km < b_km):
                best = result

        if best is None:
            best = EnrichedGapVRPSolver().solve(problem=problem, oracle=oracle)
        best.algorithm = "enriched_stochastic_rr_v2"
        best.runtime_sec = time.perf_counter() - t0
        best.details["stochastic_starts"] = starts
        best.details["time_budget_sec"] = self.config.time_budget_sec
        return best


class EnrichedStochasticGRASPSolver:
    """GRASP-like variant: stronger randomness via wider top-k window."""

    def __init__(self, config: EnrichedStochasticConfig | None = None) -> None:
        self.config = config or EnrichedStochasticConfig()

    def solve(self, *, problem: EnrichedProblem, oracle: DistanceOracleWithFallback) -> EnrichedSolveResult:
        t0 = time.perf_counter()
        best: EnrichedSolveResult | None = None
        starts = 0
        while starts < self.config.max_starts and (time.perf_counter() - t0) < self.config.time_budget_sec:
            width = self.config.top_k_min + 2 * starts
            k = min(self.config.top_k_max, max(self.config.top_k_min, width))
            solver = EnrichedGapVRPSolver(
                EnrichedGapVRPConfig(
                    random_seed=self.config.seed + starts * 131,
                    top_k_agents=k,
                    balance_penalty=0.01,
                )
            )
            result = solver.solve(problem=problem, oracle=oracle)
            starts += 1
            if best is None:
                best = result
                continue
            b = best.as_dict()
            r = result.as_dict()
            b_cov = (len(problem.tasks) - int(b["unassigned_tasks"]))
            r_cov = (len(problem.tasks) - int(r["unassigned_tasks"]))
            b_tw = float(b.get("transport_work_ton_km") or 1e18)
            r_tw = float(r.get("transport_work_ton_km") or 1e18)
            if (r_cov > b_cov) or (r_cov == b_cov and r_tw < b_tw):
                best = result

        if best is None:
            best = EnrichedGapVRPSolver().solve(problem=problem, oracle=oracle)
        best.algorithm = "enriched_stochastic_grasp_v2"
        best.runtime_sec = time.perf_counter() - t0
        best.details["stochastic_starts"] = starts
        best.details["time_budget_sec"] = self.config.time_budget_sec
        return best
