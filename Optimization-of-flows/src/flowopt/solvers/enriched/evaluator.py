from __future__ import annotations

from .constraints import evaluate_constraints
from .distance_oracle import DistanceOracleWithFallback
from .problem import EnrichedProblem
from .types import EnrichedSolveResult


def finalize_enriched_result(
    *,
    problem: EnrichedProblem,
    result: EnrichedSolveResult,
    oracle: DistanceOracleWithFallback,
) -> EnrichedSolveResult:
    report = evaluate_constraints(
        problem=problem,
        routes=result.routes,
        unassigned_task_ids=result.unassigned_task_ids,
        oracle=oracle,
    )
    result.unassigned_task_ids = report.normalized_unassigned
    result.feasible = bool(report.checks.get("all_checks_ok", False))
    result.details["checks"] = report.checks
    result.details["transport_work_ton_km"] = round(float(report.transport_work_ton_km), 6)
    return result

