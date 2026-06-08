from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable

import networkx as nx
import numpy as np

from .. import core
from .. import gap_vrp_solver as gap_solver
from ..dataset import RoutingDataset, Task
from ..gap_vrp_solver import SolverResult


@dataclass
class GapVRPSAASolution:
    result: SolverResult
    route_pool_size: int
    assembled_route_count: int


class SAASubSamplingVRP:
    def __init__(
        self,
        *,
        tasks: list[Task],
        coords: np.ndarray,
        sample_size: int = 20,
        iterations: int = 100,
        route_max_size: int = 15,
        log_every: int = 10,
        progress_hook: Callable[[str], None] | None = None,
    ) -> None:
        self.tasks = tasks
        self.coords = coords
        self.sample_size = max(1, min(sample_size, len(tasks)))
        self.iterations = max(1, iterations)
        self.route_max_size = max(1, route_max_size)
        self.log_every = max(1, log_every)
        self.progress_hook = progress_hook
        self.n_total = len(coords)
        diffs = coords[:, None, :] - coords[None, :, :]
        self.dist_matrix = np.sqrt((diffs ** 2).sum(axis=2))
        self.route_pool: list[list[int]] = []

    def solve_local_deterministic_vrp(self, indices: list[int]) -> list[list[int]]:
        unvisited = list(indices)
        local_routes: list[list[int]] = []
        while unvisited:
            route: list[int] = []
            curr = 0
            for _ in range(min(self.route_max_size, len(unvisited))):
                next_node = min(unvisited, key=lambda x: self.dist_matrix[curr][x])
                route.append(next_node)
                unvisited.remove(next_node)
                curr = next_node
            local_routes.append(route)
        return local_routes

    def run_saa(self) -> None:
        all_indices = list(range(1, self.n_total))
        if self.progress_hook is not None:
            self.progress_hook(
                f"SAA start: total_nodes={self.n_total}, sample_size={self.sample_size}, iterations={self.iterations}"
            )
        for iteration in range(1, self.iterations + 1):
            sample_indices = random.sample(all_indices, self.sample_size)
            local_routes = self.solve_local_deterministic_vrp(sample_indices)
            self.route_pool.extend(local_routes)
            if self.progress_hook is not None and (
                iteration % self.log_every == 0 or iteration == self.iterations
            ):
                self.progress_hook(
                    f"SAA iter {iteration}/{self.iterations}: route_pool={len(self.route_pool)}"
                )

    def assemble_final_solution(self) -> list[list[int]]:
        uncovered = set(range(1, self.n_total))
        final_solution: list[list[int]] = []

        def route_efficiency(route: list[int]) -> float:
            if not route:
                return float("inf")
            dist = self.dist_matrix[0, route[0]]
            for idx in range(len(route) - 1):
                dist += self.dist_matrix[route[idx], route[idx + 1]]
            dist += self.dist_matrix[route[-1], 0]
            return dist / len(route)

        sorted_pool = sorted(self.route_pool, key=route_efficiency)
        for route in sorted_pool:
            if route and all(node in uncovered for node in route):
                final_solution.append(route)
                for node in route:
                    uncovered.remove(node)
            if not uncovered:
                break

        for node in sorted(uncovered):
            final_solution.append([node])
        return final_solution


def _build_task_coords(dataset: RoutingDataset, tasks: list[Task]) -> np.ndarray:
    coords = [(0.0, 0.0)]
    for task in tasks:
        node = dataset.graph.nodes[task.source_node_id]
        coords.append((float(node.x), float(node.y)))
    return np.asarray(coords, dtype=float)


def _best_agent_for_route(
    *,
    route_task_ids: list[str],
    task_by_id: dict[str, Task],
    available_agents: set[str],
    states: dict[str, core.AgentState],
    dataset: RoutingDataset,
    graph: nx.DiGraph,
    cache: dict[tuple[str, str], tuple[list[str], float] | None],
) -> tuple[str | None, core.AgentDayPlan | None]:
    best_agent_id = None
    best_plan = None
    best_score = None
    route_tasks = [task_by_id[task_id] for task_id in route_task_ids]
    for agent_id in available_agents:
        state = states[agent_id]
        if any(agent_id not in [s.agent_id for s in core.compatible_agents(dataset, states, task)] for task in route_tasks):
            continue
        plan = core.evaluate_agent_task_set(state, route_tasks, graph, cache)
        if plan is None or not plan.feasible:
            continue
        score = (plan.total_km, plan.total_hours, agent_id)
        if best_score is None or score < best_score:
            best_score = score
            best_agent_id = agent_id
            best_plan = plan
    return best_agent_id, best_plan


def _assign_routes_to_agents(
    *,
    dataset: RoutingDataset,
    tasks: list[Task],
    assembled_routes: list[list[int]],
    graph: nx.DiGraph,
    cache: dict[tuple[str, str], tuple[list[str], float] | None],
    route_prefix: str,
) -> SolverResult:
    task_by_id = {task.task_id: task for task in tasks}
    task_index = {idx + 1: task.task_id for idx, task in enumerate(tasks)}
    states = core.initialize_agent_states(dataset, {"metadata": dataset.metadata})
    available_agents = set(states.keys())
    routes: list[gap_solver.Route] = []
    unassigned: list[str] = []
    route_counter = 1

    def add_plan(agent_id: str, plan: core.AgentDayPlan) -> None:
        nonlocal route_counter
        agent_state = states[agent_id]
        core.apply_plan_to_state(agent_state, plan)
        for task in plan.ordered_tasks:
            route_id = f"{route_prefix}_{route_counter:04d}"
            route_counter += 1
            agent_state.route_ids.append(route_id)
            routes.append(
                gap_solver.Route(
                    route_id=route_id,
                    agent_id=agent_id,
                    path=plan.service_paths[task.task_id],
                    task_ids=(task.task_id,),
                )
            )

    sorted_routes = sorted(assembled_routes, key=lambda route: (-len(route), route))
    for route in sorted_routes:
        route_task_ids = [task_index[node_idx] for node_idx in route]
        agent_id, plan = _best_agent_for_route(
            route_task_ids=route_task_ids,
            task_by_id=task_by_id,
            available_agents=available_agents,
            states=states,
            dataset=dataset,
            graph=graph,
            cache=cache,
        )
        if agent_id is not None and plan is not None:
            add_plan(agent_id, plan)
            available_agents.remove(agent_id)
            continue

        for task_id in route_task_ids:
            single_agent_id, single_plan = _best_agent_for_route(
                route_task_ids=[task_id],
                task_by_id=task_by_id,
                available_agents=available_agents,
                states=states,
                dataset=dataset,
                graph=graph,
                cache=cache,
            )
            if single_agent_id is not None and single_plan is not None:
                add_plan(single_agent_id, single_plan)
                available_agents.remove(single_agent_id)
            else:
                unassigned.append(task_id)

    limit_violations = core.validate_daily_limits(states)
    feasible = (len(unassigned) == 0 and len(limit_violations) == 0)

    return SolverResult(
        method_label="GAP-VRP + SAA",
        routes=routes,
        states=states,
        unassigned=sorted(set(unassigned)),
        transport_work_ton_km=None,
        feasible=feasible,
        limit_violations=limit_violations,
        n_assigned=len(routes),
        n_unassigned=len(set(unassigned)),
        active_agents=sum(1 for state in states.values() if state.task_ids),
    )


def solve_real_gap_vrp_saa(
    *,
    dataset: RoutingDataset,
    payload: dict[str, Any],
    graph: nx.DiGraph,
    cache: dict[tuple[str, str], tuple[list[str], float] | None],
    step1_method: str = "lp",
    gap_iter: int = 120,
    use_repair: bool = True,
    saa_sample_size: int = 20,
    saa_iterations: int = 100,
    saa_route_max_size: int = 15,
    saa_seed: int = 42,
    saa_log_every: int = 10,
    show_progress: bool = False,
    verbose: bool = False,
    progress_hook: Callable[[str], None] | None = None,
) -> GapVRPSAASolution:
    del payload, gap_iter, use_repair, verbose

    def _emit(message: str) -> None:
        if progress_hook is not None:
            progress_hook(f"[GAP-VRP-SAA] {message}")
        elif show_progress:
            print(f"[GAP-VRP-SAA] {message}", flush=True)

    _emit("step1 start")
    if step1_method == "dataset":
        tasks = list(dataset.tasks)
    elif step1_method == "lp":
        tasks = gap_solver.generate_tasks_lp(
            dataset,
            graph,
            cache,
            show_progress=show_progress,
            progress_hook=progress_hook,
        )
    else:
        tasks = gap_solver.generate_tasks_greedy(
            dataset,
            graph,
            cache,
            show_progress=show_progress,
            progress_hook=progress_hook,
        )
    _emit(f"step1 done: tasks={len(tasks)}")

    random.seed(saa_seed)
    np.random.seed(saa_seed)
    coords = _build_task_coords(dataset, tasks)
    sample_size = min(max(1, saa_sample_size), len(tasks))
    saa = SAASubSamplingVRP(
        tasks=tasks,
        coords=coords,
        sample_size=sample_size,
        iterations=saa_iterations,
        route_max_size=saa_route_max_size,
        log_every=saa_log_every,
        progress_hook=_emit,
    )
    saa.run_saa()
    assembled_routes = saa.assemble_final_solution()
    _emit(
        f"SAA assemble done: route_pool={len(saa.route_pool)}, assembled_routes={len(assembled_routes)}"
    )

    result = _assign_routes_to_agents(
        dataset=dataset,
        tasks=tasks,
        assembled_routes=assembled_routes,
        graph=graph,
        cache=cache,
        route_prefix="SAA_ROUTE",
    )
    return GapVRPSAASolution(
        result=result,
        route_pool_size=len(saa.route_pool),
        assembled_route_count=len(assembled_routes),
    )
