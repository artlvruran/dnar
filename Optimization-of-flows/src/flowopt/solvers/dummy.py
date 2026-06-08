from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx

from .. import core
from ..dataset import Route, RoutingDataset


@dataclass
class DummySolverResult:
    method_label: str
    routes: list[Route]
    states: dict[str, core.AgentState]
    unassigned: list[str]
    transport_work_ton_km: float | None
    feasible: bool
    limit_violations: list[dict[str, Any]]
    n_assigned: int
    n_unassigned: int
    active_agents: int


def run_dummy_solver(
    *,
    dataset: RoutingDataset,
    payload: dict[str, Any],
    graph: nx.DiGraph,
    cache: dict[tuple[str, str], tuple[list[str], float] | None],
) -> DummySolverResult:
    routes, states, unassigned_ids = core.assign_tasks_greedy(dataset, graph, payload, cache)
    limit_violations = core.validate_daily_limits(states)
    feasible = (len(unassigned_ids) == 0 and len(limit_violations) == 0)

    return DummySolverResult(
        method_label="dummy_greedy",
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
