from __future__ import annotations

from dataclasses import dataclass
import time

from .distance_oracle import DistanceOracleWithFallback
from .milp_solver_v2 import EnrichedMILPConfig, EnrichedMILPSolver
from .problem import EnrichedProblem
from .types import EnrichedSolveResult


@dataclass(frozen=True)
class EnrichedMILPStochasticConfig:
    time_budget_sec: float = 20.0
    max_starts: int = 8
    per_start_time_limit_sec: int = 8
    max_pairs_per_task: int = 40
    unassigned_penalty: float = 1e6
    jitter: float = 0.05
    seed: int = 42


class EnrichedMILPStochasticSolver:
    def __init__(self, config: EnrichedMILPStochasticConfig | None = None) -> None:
        self.config = config or EnrichedMILPStochasticConfig()

    def solve(self, *, problem: EnrichedProblem, oracle: DistanceOracleWithFallback) -> EnrichedSolveResult:
        t0 = time.perf_counter()
        best: EnrichedSolveResult | None = None
        starts = 0

        while starts < self.config.max_starts and (time.perf_counter() - t0) < self.config.time_budget_sec:
            solver = EnrichedMILPSolver(
                EnrichedMILPConfig(
                    time_limit_sec=self.config.per_start_time_limit_sec,
                    unassigned_penalty=self.config.unassigned_penalty,
                    max_pairs_per_task=self.config.max_pairs_per_task,
                    random_jitter=self.config.jitter,
                    random_seed=self.config.seed + starts * 1009,
                )
            )
            result = solver.solve(problem=problem, oracle=oracle)
            starts += 1

            if best is None:
                best = result
                continue

            b = best.as_dict()
            r = result.as_dict()
            b_cov = len(problem.tasks) - int(b.get("unassigned_tasks", len(problem.tasks)))
            r_cov = len(problem.tasks) - int(r.get("unassigned_tasks", len(problem.tasks)))
            b_km = float(b.get("total_km") or 1e18)
            r_km = float(r.get("total_km") or 1e18)
            if (r_cov > b_cov) or (r_cov == b_cov and r_km < b_km):
                best = result
            if result.feasible:
                best = result
                break

        if best is None:
            best = EnrichedMILPSolver().solve(problem=problem, oracle=oracle)

        best.algorithm = "enriched_milp_stochastic_v1"
        best.runtime_sec = time.perf_counter() - t0
        best.details["stochastic_starts"] = starts
        best.details["time_budget_sec"] = self.config.time_budget_sec
        return best

