from __future__ import annotations

from typing import Any, Callable

import networkx as nx

from ..dataset import RoutingDataset
from ..gap_vrp_solver import SolverResult
from . import gap_vrp as gap_solver


def solve_real_gap_vrp(
    *,
    dataset: RoutingDataset,
    payload: dict[str, Any],
    graph: nx.DiGraph,
    cache: dict[tuple[str, str], tuple[list[str], float] | None],
    step1_method: str = "lp",
    gap_iter: int = 120,
    use_repair: bool = True,
    show_progress: bool = False,
    verbose: bool = False,
    progress_hook: Callable[[str], None] | None = None,
) -> SolverResult:
    return gap_solver.run_gap_vrp(
        dataset=dataset,
        payload=payload,
        graph=graph,
        cache=cache,
        step1_method=step1_method,
        gap_iter=gap_iter,
        use_repair=use_repair,
        show_progress=show_progress,
        verbose=verbose,
        progress_hook=progress_hook,
    )
