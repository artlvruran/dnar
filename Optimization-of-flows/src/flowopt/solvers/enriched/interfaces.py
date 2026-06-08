from __future__ import annotations

from typing import Protocol

from .distance_oracle import DistanceOracleWithFallback
from .problem import EnrichedProblem
from .types import EnrichedSolveResult


class EnrichedSolver(Protocol):
    def solve(self, *, problem: EnrichedProblem, oracle: DistanceOracleWithFallback) -> EnrichedSolveResult:
        ...

