from __future__ import annotations

import time
from typing import Any, Callable

import networkx as nx

from .. import core
from .. import genetic_solver_components as ga
from ..dataset import RoutingDataset
from ..gap_vrp_solver import SolverResult


def solve_real_genetic(
    *,
    dataset: RoutingDataset,
    payload: dict[str, Any],
    graph: nx.DiGraph,
    cache: dict[tuple[str, str], tuple[list[str], float] | None],
    population_size: int = 60,
    generations: int = 120,
    elite_size: int = 4,
    seed: int | None = 42,
    max_runtime_sec: float | None = None,
    show_progress: bool = False,
    progress_hook: Callable[[str], None] | None = None,
) -> SolverResult:
    def _emit(message: str) -> None:
        if progress_hook is not None:
            progress_hook(message)
        elif show_progress:
            print(f"[real_genetic] {message}", flush=True)

    _emit(
        f"start: tasks={len(dataset.tasks)}, agents={len(dataset.fleet.agents)}, "
        f"population={population_size}, generations={generations}, "
        f"max_runtime_sec={max_runtime_sec}"
    )
    t0 = time.perf_counter()
    routes, states, unassigned_ids = ga.assign_tasks_genetic(
        dataset,
        graph,
        payload,
        cache,
        population_size=population_size,
        generations=generations,
        elite_size=elite_size,
        seed=seed,
        max_runtime_sec=max_runtime_sec,
        show_progress=show_progress,
        progress_hook=progress_hook,
    )
    _emit(
        f"assignment done in {time.perf_counter() - t0:.1f}s: "
        f"routes={len(routes)}, unassigned={len(unassigned_ids)}"
    )

    limit_violations = core.validate_daily_limits(states)
    feasible = (len(unassigned_ids) == 0 and len(limit_violations) == 0)
    _emit(f"finalize: violations={len(limit_violations)}, feasible={feasible}")

    return SolverResult(
        method_label="Real Genetic",
        routes=routes,
        states=states,
        unassigned=unassigned_ids,
        transport_work_ton_km=None,
        feasible=feasible,
        limit_violations=limit_violations,
        n_assigned=len(routes),
        n_unassigned=len(unassigned_ids),
        active_agents=sum(1 for s in states.values() if s.task_ids),
    )
