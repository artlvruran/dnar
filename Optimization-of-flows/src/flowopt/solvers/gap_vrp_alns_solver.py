from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Callable

import networkx as nx
import numpy as np
try:
    from alns import ALNS, State
    from alns.accept import SimulatedAnnealing
    from alns.select import RouletteWheel
    from alns.stop import MaxIterations
except ModuleNotFoundError:  # pragma: no cover - depends on optional extra
    ALNS = None
    State = object
    SimulatedAnnealing = None
    RouletteWheel = None
    MaxIterations = None

from .. import core
from .. import gap_vrp_solver as gap_solver
from ..dataset import RoutingDataset, Task
from ..gap_vrp_solver import SolverResult


@dataclass
class GapVRPALNSSolution:
    result: SolverResult
    base_objective: float
    best_objective: float


def _solver_result_objective(result: SolverResult) -> float:
    active_states = [state for state in result.states.values() if state.task_ids]
    total_km = sum(state.total_km for state in active_states)
    return float(total_km + len(result.unassigned) * 10000.0)


class GapVRPALNSData:
    def __init__(
        self,
        *,
        dataset: RoutingDataset,
        tasks: list[Task],
        states: dict[str, core.AgentState],
        graph: nx.DiGraph,
        cache: dict[tuple[str, str], tuple[list[str], float] | None],
        q: int = 30,
        alns_iterations: int = 150,
        log_every: int = 10,
        progress_hook: Callable[[str], None] | None = None,
    ) -> None:
        self.dataset = dataset
        self.tasks = tasks
        self.task_by_id = {task.task_id: task for task in tasks}
        self.task_ids = [task.task_id for task in tasks]
        self.task_index = {task_id: idx + 1 for idx, task_id in enumerate(self.task_ids)}
        self.index_to_task_id = {idx: task_id for task_id, idx in self.task_index.items()}
        self.states = states
        self.graph = graph
        self.cache = cache
        self.removal_q = q
        self.alns_iterations = alns_iterations
        self.log_every = max(1, log_every)
        self.progress_hook = progress_hook
        self.iteration_counter = 0
        self.agent_ids = list(states.keys())
        self.compatible_agents = {
            task.task_id: [state.agent_id for state in core.compatible_agents(dataset, states, task)]
            for task in tasks
        }
        self.candidate_list = self._build_candidate_list()

    def _task_route_nodes(self, task_id: str) -> tuple[str, str]:
        task = self.task_by_id[task_id]
        return task.source_node_id, task.destination_node_id

    def _distance_between_tasks(self, left_task_id: str, right_task_id: str) -> float:
        _, left_dst = self._task_route_nodes(left_task_id)
        right_src, _ = self._task_route_nodes(right_task_id)
        shortest = core.shortest_path_cached(self.graph, self.cache, left_dst, right_src)
        return shortest[1] if shortest is not None else float("inf")

    def _build_candidate_list(self) -> dict[int, list[int]]:
        candidate_list: dict[int, list[int]] = {}
        for task_id in self.task_ids:
            scored: list[tuple[float, float, int, str]] = []
            task = self.task_by_id[task_id]
            for other_task_id in self.task_ids:
                if other_task_id == task_id:
                    continue
                other_task = self.task_by_id[other_task_id]
                service_gap = self._distance_between_tasks(task_id, other_task_id)
                scored.append(
                    (
                        service_gap,
                        abs(task.mass_tons - other_task.mass_tons),
                        0 if task.container_type == other_task.container_type else 1,
                        other_task_id,
                    )
                )
            scored.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
            candidate_list[self.task_index[task_id]] = [
                self.task_index[other_task_id]
                for _, _, _, other_task_id in scored[: min(50, len(scored))]
            ]
        return candidate_list

    def evaluate_routes(self, routes: list[list[int]], unassigned: list[int]) -> float:
        total_km = 0.0
        total_unassigned = len(unassigned)
        for route_idx, route in enumerate(routes):
            agent_id = self.agent_ids[route_idx]
            state = self.states[agent_id]
            task_ids = [self.index_to_task_id[node_idx] for node_idx in route]
            tasks = [self.task_by_id[task_id] for task_id in task_ids]
            plan = core.evaluate_agent_task_set(state, tasks, self.graph, self.cache)
            if plan is None or not plan.feasible:
                return float("inf")
            total_km += plan.total_km
        return total_km + total_unassigned * 10000


class GapVRPState(State):
    def __init__(self, routes: list[list[int]], unassigned: list[int], data: GapVRPALNSData) -> None:
        self.routes = routes
        self.unassigned = unassigned
        self.data = data

    def objective(self) -> float:
        return self.data.evaluate_routes(self.routes, self.unassigned)


def random_removal(state: GapVRPState, rnd: np.random.RandomState, q: int | None = None) -> GapVRPState:
    new_state = copy.deepcopy(state)
    q = q or new_state.data.removal_q
    all_assigned = [node for route in new_state.routes for node in route]
    if not all_assigned:
        return new_state
    to_remove = rnd.choice(all_assigned, size=min(q, len(all_assigned)), replace=False)
    removed = set(int(node) for node in np.asarray(to_remove).tolist())
    for route_idx in range(len(new_state.routes)):
        new_state.routes[route_idx] = [node for node in new_state.routes[route_idx] if node not in removed]
    new_state.unassigned.extend(sorted(removed))
    return new_state


def string_removal(state: GapVRPState, rnd: np.random.RandomState, q: int | None = None) -> GapVRPState:
    new_state = copy.deepcopy(state)
    q = q or new_state.data.removal_q
    non_empty_routes = [idx for idx, route in enumerate(new_state.routes) if route]
    if not non_empty_routes:
        return new_state
    route_idx = int(rnd.choice(non_empty_routes))
    route = new_state.routes[route_idx]
    length = min(q, len(route))
    start = int(rnd.randint(0, len(route) - length + 1))
    removed = route[start : start + length]
    new_state.routes[route_idx] = route[:start] + route[start + length :]
    new_state.unassigned.extend(removed)
    return new_state


def optimized_shaw_removal(state: GapVRPState, rnd: np.random.RandomState, q: int | None = None) -> GapVRPState:
    new_state = copy.deepcopy(state)
    q = q or new_state.data.removal_q
    all_assigned = [node for route in new_state.routes for node in route]
    if not all_assigned:
        return new_state
    seed_node = int(rnd.choice(all_assigned))
    candidates = state.data.candidate_list.get(seed_node, [])
    to_remove = [seed_node]
    assigned_set = set(all_assigned)
    for neighbor in candidates:
        if len(to_remove) >= q:
            break
        if neighbor in assigned_set and neighbor not in to_remove:
            to_remove.append(neighbor)
    removed = set(to_remove)
    for route_idx in range(len(new_state.routes)):
        new_state.routes[route_idx] = [node for node in new_state.routes[route_idx] if node not in removed]
    new_state.unassigned.extend(to_remove)
    return new_state


def regret_2_repair(state: GapVRPState, rnd: np.random.RandomState) -> GapVRPState:
    state.data.iteration_counter += 1
    while state.unassigned:
        best_regret: tuple[float, float, str] | None = None
        best_node: int | None = None
        best_position: tuple[int, int] | None = None
        for node_idx in list(state.unassigned):
            insertion_costs: list[tuple[float, int, int]] = []
            task_id = state.data.index_to_task_id[node_idx]
            for route_idx, route in enumerate(state.routes):
                agent_id = state.data.agent_ids[route_idx]
                if agent_id not in state.data.compatible_agents[task_id]:
                    continue
                base_score = state.data.evaluate_routes(
                    [r if idx != route_idx else list(route) for idx, r in enumerate(state.routes)],
                    state.unassigned,
                )
                for pos in range(len(route) + 1):
                    candidate_routes = [list(r) for r in state.routes]
                    candidate_routes[route_idx].insert(pos, node_idx)
                    candidate_score = state.data.evaluate_routes(candidate_routes, state.unassigned)
                    if not np.isfinite(candidate_score):
                        continue
                    insertion_costs.append((candidate_score - base_score, route_idx, pos))
            insertion_costs.sort(key=lambda item: item[0])
            if not insertion_costs:
                continue
            regret = (
                insertion_costs[1][0] - insertion_costs[0][0]
                if len(insertion_costs) >= 2
                else insertion_costs[0][0]
            )
            candidate = (-regret, insertion_costs[0][0], task_id)
            if best_regret is None or candidate < best_regret:
                best_regret = candidate
                best_node = node_idx
                best_position = (insertion_costs[0][1], insertion_costs[0][2])
        if best_node is None or best_position is None:
            break
        route_idx, pos = best_position
        state.routes[route_idx].insert(pos, best_node)
        state.unassigned.remove(best_node)

    if (
        state.data.progress_hook is not None
        and state.data.log_every > 0
        and state.data.iteration_counter % state.data.log_every == 0
    ):
        candidate_objective = state.objective()
        state.data.progress_hook(
            f"ALNS iter {state.data.iteration_counter}/{state.data.alns_iterations}: "
            f"candidate objective={candidate_objective:.3f}, unassigned={len(state.unassigned)}"
        )
    return state


def _build_routes_from_state(
    state: GapVRPState,
) -> tuple[list[gap_solver.Route], dict[str, core.AgentState], list[str]]:
    data = state.data
    routes: list[gap_solver.Route] = []
    states = core.initialize_agent_states(data.dataset, {"metadata": data.dataset.metadata})
    route_counter = 1
    unassigned = [data.index_to_task_id[node_idx] for node_idx in state.unassigned]
    for route_idx, route in enumerate(state.routes):
        if not route:
            continue
        agent_id = data.agent_ids[route_idx]
        agent_state = states[agent_id]
        task_ids = [data.index_to_task_id[node_idx] for node_idx in route]
        tasks = [data.task_by_id[task_id] for task_id in task_ids]
        plan = core.evaluate_agent_task_set(agent_state, tasks, data.graph, data.cache)
        if plan is None or not plan.feasible:
            unassigned.extend(task_ids)
            continue
        core.apply_plan_to_state(agent_state, plan)
        for task in plan.ordered_tasks:
            route_id = f"ALNS_ROUTE_{route_counter:04d}"
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
    return routes, states, sorted(set(unassigned))


def _build_solver_result(
    *,
    dataset: RoutingDataset,
    alns_state: GapVRPState,
) -> SolverResult:
    del dataset
    routes, states, unassigned = _build_routes_from_state(alns_state)
    limit_violations = core.validate_daily_limits(states)
    feasible = (len(unassigned) == 0 and len(limit_violations) == 0)
    return SolverResult(
        method_label="GAP-Lagrangean + VRP + ALNS",
        routes=routes,
        states=states,
        unassigned=unassigned,
        transport_work_ton_km=None,
        feasible=feasible,
        limit_violations=limit_violations,
        n_assigned=len(routes),
        n_unassigned=len(unassigned),
        active_agents=sum(1 for state in states.values() if state.task_ids),
    )


def solve_real_gap_vrp_alns(
    *,
    dataset: RoutingDataset,
    payload: dict[str, Any],
    graph: nx.DiGraph,
    cache: dict[tuple[str, str], tuple[list[str], float] | None],
    step1_method: str = "lp",
    gap_iter: int = 120,
    use_repair: bool = True,
    alns_iterations: int = 150,
    alns_removal_q: int = 30,
    alns_seed: int = 42,
    alns_log_every: int = 10,
    start_temperature: float = 1000.0,
    end_temperature: float = 1.0,
    temperature_step: float = 0.999,
    show_progress: bool = False,
    verbose: bool = False,
    progress_hook: Callable[[str], None] | None = None,
) -> GapVRPALNSSolution:
    def _emit(message: str) -> None:
        if progress_hook is not None:
            progress_hook(f"[GAP-VRP-ALNS] {message}")
        elif show_progress:
            print(f"[GAP-VRP-ALNS] {message}", flush=True)

    del verbose

    if ALNS is None or SimulatedAnnealing is None or RouletteWheel is None or MaxIterations is None:
        _emit("optional dependency 'alns' not found, fallback to base GAP-VRP")
        base_result = gap_solver.run_gap_vrp(
            dataset=dataset,
            payload=payload,
            graph=graph,
            cache=cache,
            step1_method=step1_method,
            gap_iter=gap_iter,
            use_repair=use_repair,
            verbose=False,
            show_progress=show_progress,
            progress_hook=progress_hook,
        )
        base_objective = _solver_result_objective(base_result)
        return GapVRPALNSSolution(
            result=base_result,
            base_objective=base_objective,
            best_objective=base_objective,
        )

    _emit("base solver start")
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

    states = core.initialize_agent_states(dataset, payload)
    allocation = gap_solver.solve_gap_lagrangean(
        tasks,
        list(states.values()),
        dataset,
        graph,
        cache,
        max_iter=gap_iter,
        show_progress=show_progress,
        progress_hook=progress_hook,
    )
    if use_repair:
        allocation = gap_solver.iterative_repair(allocation, states, dataset, graph, cache)

    task_index = {task.task_id: idx + 1 for idx, task in enumerate(tasks)}
    initial_routes: list[list[int]] = []
    assigned_task_ids: set[str] = set()
    for agent_id in states:
        agent_tasks = allocation.get(agent_id, [])
        plan = core.evaluate_agent_task_set(states[agent_id], agent_tasks, graph, cache)
        if plan is None:
            initial_routes.append([])
            continue
        route_nodes: list[int] = []
        for task in plan.ordered_tasks:
            node_idx = task_index[task.task_id]
            route_nodes.append(node_idx)
            assigned_task_ids.add(task.task_id)
        initial_routes.append(route_nodes)
    initial_unassigned = [task_index[task.task_id] for task in tasks if task.task_id not in assigned_task_ids]

    alns_data = GapVRPALNSData(
        dataset=dataset,
        tasks=tasks,
        states=states,
        graph=graph,
        cache=cache,
        q=alns_removal_q,
        alns_iterations=alns_iterations,
        log_every=alns_log_every,
        progress_hook=_emit,
    )
    initial_state = GapVRPState(initial_routes, initial_unassigned, alns_data)
    base_objective = initial_state.objective()
    _emit(f"base solution objective={base_objective:.3f}")

    rnd_state = np.random.RandomState(alns_seed)
    random.seed(alns_seed)
    alns = ALNS(rnd_state)
    alns.add_destroy_operator(random_removal)
    alns.add_destroy_operator(string_removal)
    alns.add_destroy_operator(optimized_shaw_removal)
    alns.add_repair_operator(regret_2_repair)

    select = RouletteWheel(scores=[50, 15, 5, 1], decay=0.8, num_destroy=3, num_repair=1)
    accept = SimulatedAnnealing(
        start_temperature=start_temperature,
        end_temperature=end_temperature,
        step=temperature_step,
    )
    stop = MaxIterations(alns_iterations)

    _emit(f"ALNS start: iterations={alns_iterations}, removal_q={alns_removal_q}, seed={alns_seed}")
    result = alns.iterate(initial_state, select, accept, stop)
    best_state = result.best_state
    best_objective = best_state.objective()
    _emit(f"ALNS done: best objective={best_objective:.3f}")

    return GapVRPALNSSolution(
        result=_build_solver_result(dataset=dataset, alns_state=best_state),
        base_objective=base_objective,
        best_objective=best_objective,
    )
