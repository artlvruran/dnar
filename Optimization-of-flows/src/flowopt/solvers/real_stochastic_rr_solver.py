from __future__ import annotations

from typing import Callable

import networkx as nx

from ..dataset import RoutingDataset
from ..gap_vrp_solver import SolverResult
from .real_stochastic_solver import solve_real_stochastic_rr as _solve


def solve_real_stochastic_rr(
    *,
    dataset: RoutingDataset,
    payload: dict[str, Any],
    graph: nx.DiGraph,
    cache: dict[tuple[str, str], tuple[list[str], float] | None],
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
) -> SolverResult:
    return _solve(
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
        progress_hook=progress_hook,
    )
