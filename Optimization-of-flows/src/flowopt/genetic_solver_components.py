from __future__ import annotations

from collections import defaultdict
import argparse
from dataclasses import dataclass, field
from pathlib import Path
import json
import math
import time
from typing import Any, Callable

import matplotlib.pyplot as plt
import networkx as nx

import random #для генерации популяций хромосом и их мутаций

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "data"

from . import core as core_solver
from .dataset import CONTAINER_TO_VEHICLE_TYPES, Route, RoutingDataset, Task, dataset_from_dict


DEFAULT_DATASET_PATH = DATA_DIR / "synthetic" / "dataset_sandbox_type2.json"

# Daily limits used by this simple solver to prove one-day feasibility.
MAX_DAILY_KM_BY_TYPE: dict[str, float] = {
    "VT_A": 130.0,
    "VT_AB": 140.0,
    "VT_ABD": 120.0,
    "VT_AD": 120.0,
    "VT_C": 140.0,
    "VT_CD": 120.0,
    "Type1": 130.0,
    "Type2": 140.0,
    "Type1-2": 95.0,
    "Type3": 140.0,
    "Type4": 120.0,
    "TypeRNO": 110.0,
    "TypeO2P": 200.0,
}

MAX_SHIFT_HOURS_BY_TYPE: dict[str, float] = {
    "VT_A": 10.0,
    "VT_AB": 10.0,
    "VT_ABD": 10.0,
    "VT_AD": 10.0,
    "VT_C": 10.0,
    "VT_CD": 10.0,
    "Type1": 10.0,
    "Type2": 10.0,
    "Type1-2": 9.0,
    "Type3": 10.0,
    "Type4": 10.0,
    "TypeRNO": 9.5,
    "TypeO2P": 11.0,
}

AVG_SPEED_KMPH_BY_TYPE: dict[str, float] = {
    "VT_A": 24.0,
    "VT_AB": 24.0,
    "VT_ABD": 22.0,
    "VT_AD": 22.0,
    "VT_C": 22.0,
    "VT_CD": 21.0,
    "Type1": 24.0,
    "Type2": 24.0,
    "Type1-2": 22.0,
    "Type3": 22.0,
    "Type4": 21.0,
    "TypeRNO": 23.0,
    "TypeO2P": 28.0,
}

SERVICE_HOURS_BY_CONTAINER: dict[str, float] = {
    "A": 0.22,
    "B": 0.24,
    "C": 0.35,
    "D": 0.32,
    "Type1": 0.22,
    "Type2": 0.24,
    "Type3": 0.35,
    "Type4": 0.32,
    "TypeRNO": 0.28,
    "TypeO2P": 0.30,
}


@dataclass
class AgentState:
    agent_id: str
    vehicle_type: str
    capacity_tons: float
    is_compact: bool
    depot_node: str | None
    current_node: str | None
    body_volume_m3: float = 0.0
    compaction_coeff: float = 1.0
    max_raw_volume_m3: float = 0.0
    cap_container_types: tuple[str, ...] = ()
    task_ids: list[str] = field(default_factory=list)
    route_ids: list[str] = field(default_factory=list)
    deadhead_km: float = 0.0
    task_km: float = 0.0
    service_hours: float = 0.0
    drive_hours: float = 0.0
    true_transport_work_ton_km: float = 0.0
    peak_load_tons: float = 0.0
    movement_legs: tuple[Any, ...] = ()
    stop_events: tuple[Any, ...] = ()

    @property
    def total_km(self) -> float:
        return self.deadhead_km + self.task_km

    @property
    def total_hours(self) -> float:
        return self.service_hours + self.drive_hours


@dataclass
class AgentDayPlan:
    ordered_tasks: list[Task]
    service_paths: dict[str, tuple[str, ...]]
    deadhead_km: float
    task_km: float
    drive_hours: float
    service_hours: float
    total_km: float
    total_hours: float
    feasible: bool
    true_transport_work_ton_km: float = 0.0
    peak_load_tons: float = 0.0
    movement_legs: tuple[Any, ...] = ()
    stop_events: tuple[Any, ...] = ()


def load_dataset(path: Path) -> tuple[RoutingDataset, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    dataset = dataset_from_dict(payload)
    dataset.validate()
    return dataset, payload


def build_nx_graph(dataset: RoutingDataset) -> nx.DiGraph:
    graph = nx.DiGraph()
    for node_id, node in dataset.graph.nodes.items():
        graph.add_node(node_id, x=node.x, y=node.y, kind=node.kind)
    for edge in dataset.graph.edges:
        if graph.has_edge(edge.source_id, edge.target_id):
            curr = graph[edge.source_id][edge.target_id]["distance_km"]
            if edge.distance_km < curr:
                graph[edge.source_id][edge.target_id]["distance_km"] = edge.distance_km
        else:
            graph.add_edge(edge.source_id, edge.target_id, distance_km=edge.distance_km)
    return graph


def path_distance(graph: nx.DiGraph, path: list[str]) -> float:
    return sum(graph[path[i]][path[i + 1]]["distance_km"] for i in range(len(path) - 1))


def shortest_path_cached(
    graph: nx.DiGraph,
    cache: dict[tuple[str, str], tuple[list[str], float] | None],
    source: str,
    target: str,
) -> tuple[list[str], float] | None:
    key = (source, target)
    if key in cache:
        return cache[key]
    if source == target:
        cache[key] = ([source], 0.0)
        return cache[key]
    try:
        path = nx.shortest_path(graph, source=source, target=target, weight="distance_km")
    except nx.NetworkXNoPath:
        cache[key] = None
        return None
    dist = path_distance(graph, path)
    cache[key] = (path, dist)
    return cache[key]


def check_reachability(
    graph: nx.DiGraph,
    dataset: RoutingDataset,
    cache: dict[tuple[str, str], tuple[list[str], float] | None],
) -> dict[str, Any]:
    weak_components = list(nx.weakly_connected_components(graph))
    largest_weak = max((len(c) for c in weak_components), default=0)
    strong_components = list(nx.strongly_connected_components(graph))
    largest_strong = max((len(c) for c in strong_components), default=0)

    unreachable_tasks: list[str] = []
    for task in dataset.tasks:
        if shortest_path_cached(graph, cache, task.source_node_id, task.destination_node_id) is None:
            unreachable_tasks.append(task.task_id)

    special_nodes = [
        node.node_id
        for node in dataset.graph.nodes.values()
        if node.kind in {"mno", "object1", "object2", "depot"}
    ]
    unreachable_special_nodes: list[str] = []
    object1_ids = [n.node_id for n in dataset.graph.nodes.values() if n.kind == "object1"]
    object2_ids = [n.node_id for n in dataset.graph.nodes.values() if n.kind == "object2"]

    for node_id in special_nodes:
        node_kind = dataset.graph.nodes[node_id].kind
        if node_kind == "mno":
            reachable = any(
                shortest_path_cached(graph, cache, node_id, target) is not None for target in object1_ids
            )
        elif node_kind == "object1":
            # Sandbox datasets may intentionally disable second shoulder.
            if not object2_ids:
                reachable = True
            else:
                reachable = any(
                    shortest_path_cached(graph, cache, node_id, target) is not None for target in object2_ids
                )
        else:
            reachable = True
        if not reachable:
            unreachable_special_nodes.append(node_id)

    return {
        "weakly_connected_components": len(weak_components),
        "largest_weak_component_size": largest_weak,
        "strongly_connected_components": len(strong_components),
        "largest_strong_component_size": largest_strong,
        "unreachable_tasks": unreachable_tasks,
        "unreachable_special_nodes": unreachable_special_nodes,
    }


def initialize_agent_states(dataset: RoutingDataset, payload: dict[str, Any]) -> dict[str, AgentState]:
    depots = payload.get("metadata", {}).get("agent_depots", {})
    depot_nodes = [n.node_id for n in dataset.graph.nodes.values() if n.kind == "depot"]
    fallback_depot = depot_nodes[0] if depot_nodes else None
    states: dict[str, AgentState] = {}
    for agent in dataset.fleet.agents.values():
        depot_node = depots.get(agent.agent_id, fallback_depot)
        states[agent.agent_id] = AgentState(
            agent_id=agent.agent_id,
            vehicle_type=agent.vehicle_type,
            capacity_tons=agent.capacity_tons,
            is_compact=agent.is_compact,
            depot_node=depot_node,
            current_node=depot_node,
            body_volume_m3=agent.body_volume_m3,
            compaction_coeff=agent.compaction_coeff,
            max_raw_volume_m3=agent.raw_volume_limit_m3,
            cap_container_types=agent.cap_container_types,
        )
    return states


def _task_body_volume_for_vehicle(task: Task, vehicle_type: str, compaction_coeff: float) -> float:
    body_volume = task.body_volume_for_vehicle(vehicle_type)
    if body_volume > 0:
        return body_volume
    if task.volume_raw_m3 > 0 and compaction_coeff > 0:
        return task.volume_raw_m3 / compaction_coeff
    return 0.0


def compatible_agents(dataset: RoutingDataset, states: dict[str, AgentState], task) -> list[AgentState]:
    source_node = dataset.graph.nodes[task.source_node_id]
    allowed_types = CONTAINER_TO_VEHICLE_TYPES.get(task.container_type, set())
    agents = []
    for state in states.values():
        if state.vehicle_type not in allowed_types:
            continue
        if task.compatible_vehicle_types and state.vehicle_type not in task.compatible_vehicle_types:
            continue
        if state.cap_container_types and task.container_type not in state.cap_container_types:
            continue
        if state.capacity_tons + 1e-9 < task.mass_tons:
            continue
        if source_node.center and not state.is_compact:
            continue
        if state.max_raw_volume_m3 > 0 and task.volume_raw_m3 > state.max_raw_volume_m3 + 1e-9:
            continue
        if state.body_volume_m3 > 0:
            body_need = _task_body_volume_for_vehicle(task, state.vehicle_type, state.compaction_coeff)
            if body_need > state.body_volume_m3 + 1e-9:
                continue
        agents.append(state)
    return agents


def order_agent_tasks_by_nearest_source(
    state: AgentState,
    tasks: list[Task],
    graph: nx.DiGraph,
    cache: dict[tuple[str, str], tuple[list[str], float] | None],
) -> tuple[list[Task], dict[str, tuple[str, ...]]] | None:
    if not tasks:
        return [], {}
    if state.depot_node is None:
        return None

    remaining = {task.task_id: task for task in tasks}
    ordered: list[Task] = []
    service_paths: dict[str, tuple[str, ...]] = {}
    current = state.depot_node

    while remaining:
        best: tuple[float, float, str, tuple[str, ...]] | None = None
        for task in remaining.values():
            reposition = shortest_path_cached(graph, cache, current, task.source_node_id)
            if reposition is None:
                continue
            service = shortest_path_cached(graph, cache, task.source_node_id, task.destination_node_id)
            if service is None:
                continue
            _, reposition_km = reposition
            service_nodes, service_km = service
            # First leg is closest next source; tie-break by service segment length.
            score = (reposition_km, service_km, task.task_id, tuple(service_nodes))
            if best is None or score < best:
                best = score

        if best is None:
            return None

        _, _, next_task_id, service_nodes = best
        next_task = remaining.pop(next_task_id)
        ordered.append(next_task)
        service_paths[next_task_id] = service_nodes
        current = next_task.destination_node_id

    return ordered, service_paths


def evaluate_agent_task_set(
    state: AgentState,
    tasks: list[Task],
    graph: nx.DiGraph,
    cache: dict[tuple[str, str], tuple[list[str], float] | None],
) -> AgentDayPlan | None:
    # Keep core and GA profile tables aligned so feasibility checks are identical.
    core_solver.MAX_DAILY_KM_BY_TYPE.update(MAX_DAILY_KM_BY_TYPE)
    core_solver.MAX_SHIFT_HOURS_BY_TYPE.update(MAX_SHIFT_HOURS_BY_TYPE)
    core_solver.AVG_SPEED_KMPH_BY_TYPE.update(AVG_SPEED_KMPH_BY_TYPE)
    core_solver.SERVICE_HOURS_BY_CONTAINER.update(SERVICE_HOURS_BY_CONTAINER)

    core_plan = core_solver.evaluate_agent_task_set(state, tasks, graph, cache)
    if core_plan is None:
        return None

    return AgentDayPlan(
        ordered_tasks=core_plan.ordered_tasks,
        service_paths=core_plan.service_paths,
        deadhead_km=core_plan.deadhead_km,
        task_km=core_plan.task_km,
        drive_hours=core_plan.drive_hours,
        service_hours=core_plan.service_hours,
        total_km=core_plan.total_km,
        total_hours=core_plan.total_hours,
        feasible=core_plan.feasible,
        true_transport_work_ton_km=core_plan.true_transport_work_ton_km,
        peak_load_tons=core_plan.peak_load_tons,
        movement_legs=core_plan.movement_legs,
        stop_events=core_plan.stop_events,
    )

def assign_tasks_greedy(
    dataset: RoutingDataset,
    graph: nx.DiGraph,
    payload: dict[str, Any],
    cache: dict[tuple[str, str], tuple[list[str], float] | None],
) -> tuple[list[Route], dict[str, AgentState], list[str]]:
    states = initialize_agent_states(dataset, payload)
    first_shoulder = [t for t in dataset.tasks if t.container_type != "TypeO2P"]
    second_shoulder = [t for t in dataset.tasks if t.container_type == "TypeO2P"]
    first_shoulder.sort(
        key=lambda t: (dataset.graph.nodes[t.source_node_id].center, t.mass_tons),
        reverse=True,
    )
    second_shoulder.sort(key=lambda t: t.mass_tons, reverse=True)
    ordered_tasks = first_shoulder + second_shoulder

    allocated: dict[str, list[Task]] = {agent_id: [] for agent_id in states}
    unassigned: list[str] = []

    # Phase 1: allocate points/tasks to agents.
    for task in ordered_tasks:
        if shortest_path_cached(graph, cache, task.source_node_id, task.destination_node_id) is None:
            unassigned.append(task.task_id)
            continue

        candidates = compatible_agents(dataset, states, task)
        if not candidates:
            unassigned.append(task.task_id)
            continue
        has_non_compact_candidate = any(not candidate.is_compact for candidate in candidates)

        best_choice: tuple[float, str, AgentDayPlan] | None = None
        for candidate in candidates:
            trial_tasks = allocated[candidate.agent_id] + [task]
            trial_plan = evaluate_agent_task_set(candidate, trial_tasks, graph, cache)
            if trial_plan is None or not trial_plan.feasible:
                continue

            compact_penalty = 0.0
            if has_non_compact_candidate and candidate.is_compact and not dataset.graph.nodes[task.source_node_id].center:
                # Preserve compact vehicles for center-constrained tasks.
                compact_penalty = 1000.0

            # Mild balancing term avoids collapsing all points onto one agent.
            balance_penalty = 0.45 * len(trial_tasks)
            score = trial_plan.total_km + compact_penalty + balance_penalty
            if best_choice is None or score < best_choice[0]:
                best_choice = (score, candidate.agent_id, trial_plan)

        if best_choice is None:
            unassigned.append(task.task_id)
            continue

        _, agent_id, _ = best_choice
        allocated[agent_id].append(task)

    # Phase 2: for each agent build ordered cycle (depot -> tasks -> depot).
    routes: list[Route] = []
    route_counter = 1

    for agent_id, state in states.items():
        tasks_for_agent = allocated[agent_id]
        if not tasks_for_agent:
            continue

        final_plan = evaluate_agent_task_set(state, tasks_for_agent, graph, cache)
        if final_plan is None or not final_plan.feasible:
            # Defensive fallback: keep feasibility guarantees explicit.
            for task in tasks_for_agent:
                unassigned.append(task.task_id)
            allocated[agent_id] = []
            continue

        state.task_ids = [task.task_id for task in final_plan.ordered_tasks]
        state.deadhead_km = final_plan.deadhead_km
        state.task_km = final_plan.task_km
        state.drive_hours = final_plan.drive_hours
        state.service_hours = final_plan.service_hours
        state.true_transport_work_ton_km = final_plan.true_transport_work_ton_km
        state.peak_load_tons = final_plan.peak_load_tons
        state.movement_legs = final_plan.movement_legs
        state.stop_events = final_plan.stop_events
        state.current_node = state.depot_node
        state.route_ids = []

        for task in final_plan.ordered_tasks:
            route_id = f"SOL_ROUTE_{route_counter:04d}"
            route_counter += 1
            state.route_ids.append(route_id)
            routes.append(
                Route(
                    route_id=route_id,
                    agent_id=agent_id,
                    path=final_plan.service_paths[task.task_id],
                    task_ids=(task.task_id,),
                )
            )

    return routes, states, unassigned

def _plan_transport_work(plan: AgentDayPlan, tasks: list[Task], graph: nx.DiGraph) -> float: #    Считает transport work (тонно-км) для набора задач одного агента. Учитываем только service-участки source -> destination, как и в dataset.transport_work().

    task_by_id = {task.task_id: task for task in tasks}
    total_tr = 0.0

    for task_id, service_path in plan.service_paths.items():
        task = task_by_id[task_id]
        service_dist = path_distance(graph, list(service_path))
        total_tr += task.mass_tons * service_dist

    return total_tr

def initialize_population(dataset, population_size):
    tasks = [task.task_id for task in dataset.tasks]

    population = []
    for _ in range(population_size):
        chromosome = tasks[:]
        random.shuffle(chromosome)
        population.append(chromosome)

    return population

def decode_chromosome(chromosome, dataset, graph, cache, states):
    """
    Хромосома -> распределение задач по агентам.
    Задачи берём в порядке из хромосомы и назначаем лучшему совместимому агенту.
    """
    task_map = {task.task_id: task for task in dataset.tasks}
    allocation = {agent_id: [] for agent_id in states}
    unassigned = []

    for task_id in chromosome:
        task = task_map[task_id]

        best_agent = None
        best_score = float("inf")

        candidates = compatible_agents(dataset, states, task)  # исправление: учитываем тип, capacity, compact
        if not candidates:
            unassigned.append(task)
            continue

        has_non_compact_candidate = any(not state.is_compact for state in candidates)  # улучшение: как в greedy

        for state in candidates:
            agent_id = state.agent_id
            trial_tasks = allocation[agent_id] + [task]

            plan = evaluate_agent_task_set(
                state,
                trial_tasks,
                graph,
                cache,
            )

            if plan is None or not plan.feasible:
                continue

            # Основная стоимость = тонно-км, а не просто км  # исправление: теперь fitness ближе к целевой функции
            trial_tr = _plan_transport_work(plan, trial_tasks, graph)

            # Небольшой штраф за пустые перегонки  # улучшение: помогает не накапливать лишний deadhead
            deadhead_penalty = 0.02 * plan.deadhead_km

            # Сохраняем компактные машины для действительно узких/центральных точек  # улучшение: перенесено из greedy-логики
            compact_penalty = 0.0
            source_node = dataset.graph.nodes[task.source_node_id]
            if has_non_compact_candidate and state.is_compact and not source_node.center:
                compact_penalty = 1000.0

            # Небольшой балансирующий штраф  # улучшение: уменьшает "схлопывание" всех задач на одну машину
            balance_penalty = 0.3 * len(trial_tasks)

            score = trial_tr + deadhead_penalty + compact_penalty + balance_penalty

            if score < best_score:
                best_score = score
                best_agent = agent_id

        if best_agent is None:
            unassigned.append(task)
        else:
            allocation[best_agent].append(task)

    return allocation, unassigned


def evaluate_solution(chromosome, dataset, graph, cache):
    """
    Fitness хромосомы: чем меньше, тем лучше.
    Основная цель — минимизировать тонно-км.
    """
    payload_stub = {"metadata": dataset.metadata}  # исправление: initialize_agent_states ждёт payload с metadata
    states = initialize_agent_states(dataset, payload_stub)

    allocation, unassigned = decode_chromosome(
        chromosome,
        dataset,
        graph,
        cache,
        states,
    )

    total_cost = 0.0

    for agent_id, tasks in allocation.items():
        if not tasks:
            continue

        state = states[agent_id]
        plan = evaluate_agent_task_set(
            state,
            tasks,
            graph,
            cache,
        )

        if plan is None or not plan.feasible:
            return float("inf")

        total_cost += _plan_transport_work(plan, tasks, graph)  # исправление: считаем именно transport work
        total_cost += 0.02 * plan.deadhead_km  # улучшение: слабый штраф за пустой пробег

    # Большой штраф за необслуженные задачи  # оставляем как жёсткое наказание infeasible-решениям
    if unassigned:
        total_cost += 1e6 * len(unassigned)
    total_cost += random.uniform(0, 0.001)

    return total_cost
    
    

def selection(population, fitnesses, k=3):
    selected = random.sample(list(zip(population, fitnesses)), k)
    selected.sort(key=lambda x: x[1])  # минимизация
    return selected[0][0][:]
    
    

def crossover(parent1, parent2):
    size = len(parent1)

    # выбираем сегмент
    a, b = sorted(random.sample(range(size), 2))

    child = [None] * size

    # копируем кусок из parent1
    child[a:b] = parent1[a:b]

    # заполняем из parent2
    p2_idx = 0
    for i in range(size):
        if child[i] is None:
            while parent2[p2_idx] in child:
                p2_idx += 1
            child[i] = parent2[p2_idx]

    return child


def mutate(chromosome, p=0.3):
    chromosome = chromosome[:]  # копия

    if random.random() < p:
        # swap mutation
        i, j = random.sample(range(len(chromosome)), 2)
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]

    if random.random() < p:
        # inversion mutation (очень важная!)
        i, j = sorted(random.sample(range(len(chromosome)), 2))
        chromosome[i:j] = reversed(chromosome[i:j])

    return chromosome
    


def genetic_solver(
    dataset,
    graph,
    cache,
    *,
    population_size: int = 80,
    generations: int = 300,
    elite_size: int = 4,
    seed: int | None = 42,
    max_runtime_sec: float | None = None,
    show_progress: bool = False,
    progress_hook: Callable[[str], None] | None = None,
):
    """
    Основной цикл генетического алгоритма.
    """
    if seed is not None:
        random.seed(seed)

    def _emit(message: str) -> None:
        if progress_hook is not None:
            progress_hook(message)
        elif show_progress:
            print(f"[genetic] {message}", flush=True)

    population = initialize_population(dataset, population_size)
    _emit(f"population initialized: {population_size} chromosomes")
    time_start = time.perf_counter()

    best_solution = None
    best_cost = float("inf")
    executed_generations = 0

    gen_iter = range(generations)
    if show_progress:
        try:
            from tqdm.auto import tqdm

            gen_iter = tqdm(gen_iter, total=generations, desc="[GA] generations", leave=False)
        except Exception:
            pass

    log_every = max(1, generations // 10) if generations else 1
    deadline = (time_start + max_runtime_sec) if (max_runtime_sec is not None and max_runtime_sec > 0) else None
    for generation in gen_iter:
        if deadline is not None and time.perf_counter() >= deadline:
            elapsed = time.perf_counter() - time_start
            _emit(
                f"time limit reached at generation {generation}/{generations}, "
                f"elapsed={elapsed:.1f}s"
            )
            break

        timed_out_mid_generation = False
        fitnesses: list[float] = []
        evaluated_population: list[list[str]] = []
        for chromosome in population:
            if deadline is not None and time.perf_counter() >= deadline:
                timed_out_mid_generation = True
                break
            fitnesses.append(evaluate_solution(chromosome, dataset, graph, cache))
            evaluated_population.append(chromosome)

        if not fitnesses:
            break
        executed_generations = generation + 1

        # Обновляем глобально лучшее решение
        for chromosome, fit in zip(evaluated_population, fitnesses):
            if fit < best_cost:
                best_cost = fit
                best_solution = chromosome[:]

        if timed_out_mid_generation:
            elapsed = time.perf_counter() - time_start
            _emit(
                f"time limit reached during generation {generation + 1}/{generations}, "
                f"evaluated={len(evaluated_population)}/{len(population)}, elapsed={elapsed:.1f}s"
            )
            break

        # Elitism: переносим несколько лучших особей без изменений  # главное исправление против "потери прогресса"
        elite_indices = sorted(range(len(evaluated_population)), key=lambda i: fitnesses[i])[:elite_size]
        new_population = [evaluated_population[i][:] for i in elite_indices]

        # Остальную популяцию заполняем потомками
        while len(new_population) < population_size:
            p1 = selection(evaluated_population, fitnesses)
            p2 = selection(evaluated_population, fitnesses)

            child = crossover(p1, p2)
            child = mutate(child)

            new_population.append(child)

        population = new_population

        if show_progress and hasattr(gen_iter, "set_postfix"):
            gen_iter.set_postfix(best=f"{best_cost:.3f}")
        if (generation + 1) % log_every == 0 or (generation + 1) == generations:
            _emit(f"generation {generation + 1}/{generations}, best_cost={best_cost:.3f}")

        # Можно раскомментировать для отладки:
        # if generation % 20 == 0:
        #     print(f"[GA] generation={generation}, best_cost={best_cost:.3f}")

    if best_solution is None:
        # Defensive fallback: keep solver output deterministic and non-empty.
        best_solution = population[0][:]
    _emit(
        f"genetic finished: executed_generations={executed_generations}/{generations}, "
        f"best_cost={best_cost:.3f}, elapsed={time.perf_counter() - time_start:.1f}s"
    )
    return best_solution
    
    
def assign_tasks_genetic(
    dataset,
    graph,
    payload,
    cache,
    *,
    population_size: int = 80,
    generations: int = 300,
    elite_size: int = 4,
    seed: int | None = 42,
    max_runtime_sec: float | None = None,
    show_progress: bool = False,
    progress_hook: Callable[[str], None] | None = None,
):
    """
    Генетический solver с тем же интерфейсом, что и greedy:
    возвращает routes, states, unassigned.
    """
    states = initialize_agent_states(dataset, payload)
    def _emit(message: str) -> None:
        if progress_hook is not None:
            progress_hook(message)
        elif show_progress:
            print(f"[genetic] {message}", flush=True)

    _emit(
        f"start assignment: tasks={len(dataset.tasks)}, agents={len(dataset.fleet.agents)}, "
        f"population={population_size}, generations={generations}"
    )
    best_chromosome = genetic_solver(
        dataset,
        graph,
        cache,
        population_size=population_size,
        generations=generations,
        elite_size=elite_size,
        seed=seed,
        max_runtime_sec=max_runtime_sec,
        show_progress=show_progress,
        progress_hook=progress_hook,
    )

    allocation, unassigned = decode_chromosome(
        best_chromosome,
        dataset,
        graph,
        cache,
        states,
    )
    _emit(
        f"chromosome decoded: assigned_tasks={len(dataset.tasks) - len(unassigned)}, "
        f"unassigned={len(unassigned)}"
    )

    routes = []
    route_counter = 1  # улучшение: стартуем с 1, как в greedy, для единообразия

    for agent_id, tasks in allocation.items():
        if not tasks:
            continue

        state = states[agent_id]
        plan = evaluate_agent_task_set(
            state,
            tasks,
            graph,
            cache,
        )

        if plan is None or not plan.feasible:
            # Защитный fallback  # улучшение: если декодирование дало странный результат, явно уводим задачи в unassigned
            for task in tasks:
                if task.task_id not in unassigned:
                    unassigned.append(task.task_id)
            continue

        # Обновляем state, чтобы build_solution_plan и render_utilization работали корректно  # исправление: раньше это не заполнялось
        state.task_ids = [task.task_id for task in plan.ordered_tasks]
        state.deadhead_km = plan.deadhead_km
        state.task_km = plan.task_km
        state.drive_hours = plan.drive_hours
        state.service_hours = plan.service_hours
        state.true_transport_work_ton_km = plan.true_transport_work_ton_km
        state.peak_load_tons = plan.peak_load_tons
        state.movement_legs = plan.movement_legs
        state.stop_events = plan.stop_events
        state.current_node = state.depot_node
        state.route_ids = []

        for task in plan.ordered_tasks:
            route_id = f"SOL_ROUTE_{route_counter:04d}"
            route_counter += 1

            state.route_ids.append(route_id)

            routes.append(
                Route(
                    route_id=route_id,
                    agent_id=agent_id,
                    path=tuple(plan.service_paths[task.task_id]),
                    task_ids=(task.task_id,),
                )
            )

    _emit(
        f"routes built: routes={len(routes)}, active_agents={sum(1 for s in states.values() if s.task_ids)}"
    )
    return routes, states, unassigned


def build_solution_dataset(base_dataset: RoutingDataset, routes: list[Route]) -> RoutingDataset:
    solved = RoutingDataset(
        graph=base_dataset.graph,
        fleet=base_dataset.fleet,
        tasks=base_dataset.tasks,
        routes=routes,
        metadata={
            **base_dataset.metadata,
            "solver": "simple_allocate_then_cycle_solver",
            "daily_limits": {
                "max_daily_km_by_type": MAX_DAILY_KM_BY_TYPE,
                "max_shift_hours_by_type": MAX_SHIFT_HOURS_BY_TYPE,
                "avg_speed_kmph_by_type": AVG_SPEED_KMPH_BY_TYPE,
                "service_hours_by_container": SERVICE_HOURS_BY_CONTAINER,
            },
        },
    )
    solved.validate()
    return solved


def validate_daily_limits(states: dict[str, AgentState]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for state in states.values():
        if not state.task_ids:
            continue
        km_limit = MAX_DAILY_KM_BY_TYPE[state.vehicle_type]
        h_limit = MAX_SHIFT_HOURS_BY_TYPE[state.vehicle_type]
        if state.total_km > km_limit + 1e-9:
            violations.append(
                {
                    "agent_id": state.agent_id,
                    "type": "km_limit",
                    "value": round(state.total_km, 3),
                    "limit": km_limit,
                }
            )
        if state.total_hours > h_limit + 1e-9:
            violations.append(
                {
                    "agent_id": state.agent_id,
                    "type": "time_limit",
                    "value": round(state.total_hours, 3),
                    "limit": h_limit,
                }
            )
    return violations


def tr_by_shoulder(routes: list[Route], dataset: RoutingDataset) -> dict[str, float]:
    task_by_id = {task.task_id: task for task in dataset.tasks}
    edge_distance = {(edge.source_id, edge.target_id): edge.distance_km for edge in dataset.graph.edges}

    first = 0.0
    second = 0.0
    for route in routes:
        task = task_by_id[route.task_ids[0]]
        dist = 0.0
        for i in range(len(route.path) - 1):
            dist += edge_distance[(route.path[i], route.path[i + 1])]
        tr = task.mass_tons * dist
        if task.container_type == "TypeO2P":
            second += tr
        else:
            first += tr
    return {"first_shoulder_ton_km": round(first, 3), "second_shoulder_ton_km": round(second, 3)}


def object_capacity_summary(dataset: RoutingDataset) -> dict[str, Any]:
    load_by_object: dict[str, float] = defaultdict(float)
    for task in dataset.tasks:
        load_by_object[task.destination_node_id] += task.mass_tons

    rows: list[dict[str, Any]] = []
    for node in dataset.graph.nodes.values():
        if not node.kind.startswith("object"):
            continue
        day_load = round(load_by_object.get(node.node_id, 0.0), 3)
        day_cap = node.object_day_capacity_tons
        year_cap = node.object_year_capacity_tons
        utilization = round(day_load / day_cap, 3) if day_cap > 0 else None
        theoretical_full_days = round(year_cap / day_load, 1) if day_load > 0 else None
        rows.append(
            {
                "object_id": node.node_id,
                "kind": node.kind,
                "day_load_tons": day_load,
                "day_capacity_tons": day_cap,
                "day_capacity_ok": (day_load <= day_cap) if day_cap > 0 else True,
                "day_utilization": utilization,
                "year_capacity_tons": year_cap,
                "theoretical_days_at_current_daily_load": theoretical_full_days,
            }
        )
    return {"objects": rows}


def build_solution_plan(routes: list[Route], dataset: RoutingDataset, states: dict[str, AgentState]) -> dict[str, Any]:
    task_by_id = {task.task_id: task for task in dataset.tasks}
    edges = {(e.source_id, e.target_id): e.distance_km for e in dataset.graph.edges}

    route_entries: list[dict[str, Any]] = []
    for route in routes:
        task = task_by_id[route.task_ids[0]]
        dist = 0.0
        for i in range(len(route.path) - 1):
            dist += edges[(route.path[i], route.path[i + 1])]
        route_entries.append(
            {
                "route_id": route.route_id,
                "agent_id": route.agent_id,
                "task_id": task.task_id,
                "container_type": task.container_type,
                "mass_tons": task.mass_tons,
                "source_node_id": task.source_node_id,
                "destination_node_id": task.destination_node_id,
                "distance_km": round(dist, 3),
                "path": list(route.path),
            }
        )

    agent_summary = {
        agent_id: {
            "vehicle_type": state.vehicle_type,
            "tasks": len(state.task_ids),
            "task_ids": state.task_ids,
            "total_km": round(state.total_km, 3),
            "task_km": round(state.task_km, 3),
            "deadhead_km": round(state.deadhead_km, 3),
            "total_hours": round(state.total_hours, 3),
            "drive_hours": round(state.drive_hours, 3),
            "service_hours": round(state.service_hours, 3),
        }
        for agent_id, state in states.items()
        if state.task_ids
    }

    return {"routes": route_entries, "agent_summary": agent_summary}


def route_order_key(route: Route) -> int:
    try:
        return int(route.route_id.split("_")[-1])
    except Exception:
        return 10**9


def group_routes_by_agent(routes: list[Route]) -> dict[str, list[Route]]:
    grouped: dict[str, list[Route]] = defaultdict(list)
    for route in routes:
        grouped[route.agent_id].append(route)
    for agent_id in grouped:
        grouped[agent_id].sort(key=route_order_key)
    return grouped


def constraints_check_summary(
    solved_dataset_valid: bool,
    limit_violations: list[dict[str, Any]],
    reachability: dict[str, Any],
    unassigned: list[str],
    mno_coverage: dict[str, Any],
) -> dict[str, Any]:
    hard_constraints_ok = solved_dataset_valid
    daily_limits_ok = len(limit_violations) == 0
    reachability_ok = (
        len(reachability.get("unreachable_tasks", [])) == 0
        and len(reachability.get("unreachable_special_nodes", [])) == 0
    )
    all_tasks_assigned = len(unassigned) == 0
    mno_coverage_ok = bool(mno_coverage.get("all_eligible_mno_covered", False))
    return {
        "hard_constraints_ok": hard_constraints_ok,
        "daily_limits_ok": daily_limits_ok,
        "reachability_ok": reachability_ok,
        "all_tasks_assigned": all_tasks_assigned,
        "mno_coverage_ok": mno_coverage_ok,
        "all_checks_ok": (
            hard_constraints_ok
            and daily_limits_ok
            and reachability_ok
            and all_tasks_assigned
            and mno_coverage_ok
        ),
    }


def task_source_coverage_summary(routes: list[Route], dataset: RoutingDataset) -> dict[str, Any]:
    task_by_id = {task.task_id: task for task in dataset.tasks}
    required_sources = {task.source_node_id for task in dataset.tasks}
    covered_sources: set[str] = set()
    covered_task_ids: set[str] = set()

    for route in routes:
        for task_id in route.task_ids:
            task = task_by_id[task_id]
            covered_task_ids.add(task_id)
            covered_sources.add(task.source_node_id)

    uncovered_sources = sorted(required_sources - covered_sources)
    uncovered_tasks = sorted(set(task_by_id) - covered_task_ids)
    return {
        "total_tasks": len(task_by_id),
        "covered_tasks": len(covered_task_ids),
        "all_tasks_covered": len(uncovered_tasks) == 0,
        "uncovered_task_ids": uncovered_tasks,
        "total_unique_task_sources": len(required_sources),
        "covered_unique_task_sources": len(covered_sources),
        "all_task_sources_covered": len(uncovered_sources) == 0,
        "uncovered_source_node_ids": uncovered_sources,
    }


def mno_scope_coverage_summary(dataset: RoutingDataset, routes: list[Route]) -> dict[str, Any]:
    task_by_id = {task.task_id: task for task in dataset.tasks}
    first_shoulder_task_types = {
        task.container_type
        for task in dataset.tasks
        if dataset.graph.nodes[task.source_node_id].kind == "mno" and task.container_type != "TypeO2P"
    }

    eligible_mno = {
        node.node_id
        for node in dataset.graph.nodes.values()
        if (
            node.kind == "mno"
            and node.daily_mass_tons > 0
            and bool(set(node.container_types) & first_shoulder_task_types)
        )
    }
    covered_mno: set[str] = set()
    for route in routes:
        for task_id in route.task_ids:
            task = task_by_id[task_id]
            if dataset.graph.nodes[task.source_node_id].kind == "mno":
                covered_mno.add(task.source_node_id)

    uncovered = sorted(eligible_mno - covered_mno)
    out_of_scope = sorted(
        node.node_id
        for node in dataset.graph.nodes.values()
        if node.kind == "mno" and node.node_id not in eligible_mno
    )

    return {
        "first_shoulder_task_types": sorted(first_shoulder_task_types),
        "eligible_mno_count": len(eligible_mno),
        "covered_eligible_mno_count": len(covered_mno & eligible_mno),
        "all_eligible_mno_covered": len(uncovered) == 0,
        "uncovered_eligible_mno_ids": uncovered,
        "out_of_scope_mno_count": len(out_of_scope),
        "out_of_scope_mno_ids": out_of_scope,
    }


def depot_activity_summary(dataset: RoutingDataset, states: dict[str, AgentState]) -> dict[str, Any]:
    depot_ids = sorted([node.node_id for node in dataset.graph.nodes.values() if node.kind == "depot"])
    active_depots = sorted(
        {
            state.depot_node
            for state in states.values()
            if state.task_ids and state.depot_node is not None
        }
    )
    inactive_depots = sorted(set(depot_ids) - set(active_depots))
    return {
        "total_depots": len(depot_ids),
        "active_depot_count": len(active_depots),
        "inactive_depot_count": len(inactive_depots),
        "active_depot_ids": active_depots,
        "inactive_depot_ids": inactive_depots,
    }


def render_solution_map(
    dataset: RoutingDataset,
    graph: nx.DiGraph,
    cache: dict[tuple[str, str], tuple[list[str], float] | None],
    routes: list[Route],
    states: dict[str, AgentState],
    output_path: Path,
) -> None:
    def offset_polyline(xs: list[float], ys: list[float], offset: float) -> tuple[list[float], list[float]]:
        if len(xs) < 2 or abs(offset) < 1e-12:
            return xs, ys
        out_x: list[float] = []
        out_y: list[float] = []
        n = len(xs)
        for i in range(n):
            if i == 0:
                dx = xs[1] - xs[0]
                dy = ys[1] - ys[0]
            elif i == n - 1:
                dx = xs[-1] - xs[-2]
                dy = ys[-1] - ys[-2]
            else:
                dx = xs[i + 1] - xs[i - 1]
                dy = ys[i + 1] - ys[i - 1]
            norm = math.hypot(dx, dy)
            if norm == 0:
                nx = 0.0
                ny = 0.0
            else:
                nx = -dy / norm
                ny = dx / norm
            out_x.append(xs[i] + offset * nx)
            out_y.append(ys[i] + offset * ny)
        return out_x, out_y

    def annotate_load_on_segment(
        xs: list[float],
        ys: list[float],
        text: str,
        color: str,
        zorder: int,
    ) -> None:
        if len(xs) < 2:
            return
        mid = len(xs) // 2
        i0 = max(0, mid - 1)
        i1 = min(len(xs) - 1, mid + 1)
        dx = xs[i1] - xs[i0]
        dy = ys[i1] - ys[i0]
        norm = math.hypot(dx, dy)
        if norm == 0:
            nx = 0.0
            ny = 0.0
        else:
            nx = -dy / norm
            ny = dx / norm
        px = xs[mid] + nx * base_offset * 2.2
        py = ys[mid] + ny * base_offset * 2.2
        ax.text(
            px,
            py,
            text,
            fontsize=6.5,
            color=color,
            zorder=zorder,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
        )

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_title("Simple Solver: Daily Agent Movements (Line labels = current load, tons)")

    for edge in dataset.graph.edges:
        source = dataset.graph.nodes[edge.source_id]
        target = dataset.graph.nodes[edge.target_id]
        ax.plot(
            [source.x, target.x],
            [source.y, target.y],
            color="#d4d8dc",
            linewidth=0.4,
            alpha=0.65,
            zorder=1,
        )

    routes_by_agent = group_routes_by_agent(routes)
    task_by_id = {task.task_id: task for task in dataset.tasks}
    source_mass_by_node: dict[str, float] = defaultdict(float)
    for task in dataset.tasks:
        source_mass_by_node[task.source_node_id] += task.mass_tons
    cmap = plt.get_cmap("tab20")
    all_nodes = list(dataset.graph.nodes.values())
    x_span = max(n.x for n in all_nodes) - min(n.x for n in all_nodes)
    y_span = max(n.y for n in all_nodes) - min(n.y for n in all_nodes)
    # Slightly larger shift makes concurrent agent paths visually separable.
    base_offset = max(min(x_span, y_span) * 0.0023, 1e-5)

    sorted_agents = sorted(routes_by_agent.keys())
    n_agents = len(sorted_agents)
    for agent_idx, agent_id in enumerate(sorted_agents):
        agent_routes = routes_by_agent[agent_id]
        color = cmap(agent_idx % 20)
        # Give each active agent its own lateral shift so polylines do not stack on top
        # of each other when multiple routes share the same road segment.
        if n_agents <= 1:
            offset = 0.0
        else:
            centered_idx = agent_idx - (n_agents - 1) / 2.0
            offset = centered_idx * base_offset

        current = states[agent_id].depot_node
        for route in agent_routes:
            route_start = route.path[0]
            route_end = route.path[-1]

            # Deadhead: current -> route source (depot to first source and between tasks).
            if current is not None and current != route_start:
                deadhead = shortest_path_cached(graph, cache, current, route_start)
                if deadhead is not None:
                    deadhead_nodes, _ = deadhead
                    dx = [dataset.graph.nodes[node_id].x for node_id in deadhead_nodes]
                    dy = [dataset.graph.nodes[node_id].y for node_id in deadhead_nodes]
                    dx, dy = offset_polyline(dx, dy, offset)
                    ax.plot(
                        dx,
                        dy,
                        color=color,
                        linewidth=1.35,
                        alpha=0.55,
                        linestyle="-",
                        zorder=2,
                    )
                    annotate_load_on_segment(dx, dy, "0t", "#555", zorder=6)

            # Service segment: source -> destination for task.
            sx = [dataset.graph.nodes[node_id].x for node_id in route.path]
            sy = [dataset.graph.nodes[node_id].y for node_id in route.path]
            sx, sy = offset_polyline(sx, sy, offset)
            ax.plot(
                sx,
                sy,
                color=color,
                linewidth=2.2,
                alpha=0.92,
                zorder=4,
            )
            task_mass = sum(task_by_id[task_id].mass_tons for task_id in route.task_ids)
            annotate_load_on_segment(sx, sy, f"{task_mass:.1f}t", color, zorder=7)

            current = route_end

        # Return deadhead: last destination -> depot.
        depot_node = states[agent_id].depot_node
        if current is not None and depot_node is not None and current != depot_node:
            back = shortest_path_cached(graph, cache, current, depot_node)
            if back is not None:
                back_nodes, _ = back
                bx = [dataset.graph.nodes[node_id].x for node_id in back_nodes]
                by = [dataset.graph.nodes[node_id].y for node_id in back_nodes]
                bx, by = offset_polyline(bx, by, offset)
                ax.plot(
                    bx,
                    by,
                    color=color,
                    linewidth=1.35,
                    alpha=0.62,
                    linestyle="-",
                    zorder=2,
                )
                annotate_load_on_segment(bx, by, "0t", "#555", zorder=6)

    active_states = [state for state in states.values() if state.task_ids]
    served_task_ids = [task_id for route in routes for task_id in route.task_ids]
    unique_served_task_ids = set(served_task_ids)
    task_coverage_pct = 100.0 * len(unique_served_task_ids) / len(dataset.tasks) if dataset.tasks else 100.0

    trip_fill_ratios: list[float] = []
    trip_unused_tons: list[float] = []
    total_trip_mass = 0.0
    total_trip_capacity = 0.0
    for route in routes:
        mass = sum(task_by_id[task_id].mass_tons for task_id in route.task_ids)
        capacity = states[route.agent_id].capacity_tons
        if capacity > 0:
            trip_fill_ratios.append(mass / capacity)
            trip_unused_tons.append(max(capacity - mass, 0.0))
            total_trip_mass += mass
            total_trip_capacity += capacity

    avg_fill_pct = 100.0 * sum(trip_fill_ratios) / len(trip_fill_ratios) if trip_fill_ratios else 0.0
    avg_underfill_pct = max(0.0, 100.0 - avg_fill_pct)
    weighted_fill_pct = 100.0 * total_trip_mass / total_trip_capacity if total_trip_capacity > 0 else 0.0
    avg_unused_tons = sum(trip_unused_tons) / len(trip_unused_tons) if trip_unused_tons else 0.0

    km_utilization = (
        [
            100.0 * state.total_km / MAX_DAILY_KM_BY_TYPE[state.vehicle_type]
            for state in active_states
            if MAX_DAILY_KM_BY_TYPE[state.vehicle_type] > 0
        ]
        if active_states
        else []
    )
    hour_utilization = (
        [
            100.0 * state.total_hours / MAX_SHIFT_HOURS_BY_TYPE[state.vehicle_type]
            for state in active_states
            if MAX_SHIFT_HOURS_BY_TYPE[state.vehicle_type] > 0
        ]
        if active_states
        else []
    )
    avg_km_util_pct = sum(km_utilization) / len(km_utilization) if km_utilization else 0.0
    avg_hour_util_pct = sum(hour_utilization) / len(hour_utilization) if hour_utilization else 0.0
    total_km = sum(state.total_km for state in active_states)
    total_deadhead_km = sum(state.deadhead_km for state in active_states)
    deadhead_share_pct = 100.0 * total_deadhead_km / total_km if total_km > 0 else 0.0
    avg_tasks_per_active = len(routes) / n_agents if n_agents > 0 else 0.0
    multi_task_agents = sum(1 for state in active_states if len(state.task_ids) > 1)
    max_tasks_by_agent = max((len(state.task_ids) for state in active_states), default=0)

    metrics_text = "\n".join(
        [
            "Metrics (day):",
            f"Task coverage: {task_coverage_pct:.1f}% ({len(unique_served_task_ids)}/{len(dataset.tasks)})",
            f"Avg load per trip: {avg_fill_pct:.1f}% (underfill {avg_underfill_pct:.1f}%)",
            f"Weighted load per trip: {weighted_fill_pct:.1f}%",
            f"Avg unused capacity: {avg_unused_tons:.2f} t/trip",
            f"Avg km utilization: {avg_km_util_pct:.1f}%",
            f"Avg shift-time utilization: {avg_hour_util_pct:.1f}%",
            f"Empty-run share: {deadhead_share_pct:.1f}% ({total_deadhead_km:.1f}/{total_km:.1f} km)",
            f"Active agents: {n_agents}, tasks: {len(routes)}",
            f"Tasks/agent avg: {avg_tasks_per_active:.2f}, max: {max_tasks_by_agent}, agents>1 task: {multi_task_agents}",
        ]
    )

    mno = [n for n in dataset.graph.nodes.values() if n.kind == "mno"]
    active_depot_agent_count: dict[str, int] = defaultdict(int)
    for state in states.values():
        if state.task_ids and state.depot_node is not None:
            active_depot_agent_count[state.depot_node] += 1

    task_destination_nodes = sorted({task.destination_node_id for task in dataset.tasks})
    if mno:
        ax.scatter(
            [n.x for n in mno],
            [n.y for n in mno],
            c="#2b7bba",
            s=22,
            alpha=0.9,
            zorder=6,
            label="Waste points (MNO)",
        )
        for node in mno:
            mass = source_mass_by_node.get(node.node_id, 0.0)
            if mass <= 0:
                continue
            ax.annotate(
                f"{mass:.1f}t",
                (node.x, node.y),
                fontsize=6.5,
                color="#0f3553",
                xytext=(3, 3),
                textcoords="offset points",
                zorder=9,
                bbox={"boxstyle": "round,pad=0.10", "facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
            )
    if task_destination_nodes:
        ax.scatter(
            [dataset.graph.nodes[node_id].x for node_id in task_destination_nodes],
            [dataset.graph.nodes[node_id].y for node_id in task_destination_nodes],
            marker="^",
            c="#2ca02c",
            edgecolors="#111",
            linewidths=0.6,
            s=170,
            zorder=8,
            label="Unload points (Object1)",
        )
    active_depot_nodes = [
        dataset.graph.nodes[node_id]
        for node_id in active_depot_agent_count
        if node_id in dataset.graph.nodes
    ]
    if active_depot_nodes:
        ax.scatter(
            [n.x for n in active_depot_nodes],
            [n.y for n in active_depot_nodes],
            c="#000000",
            marker="*",
            s=220,
            zorder=9,
            label="Start/Return depots",
        )

    ax.text(
        0.012,
        0.988,
        metrics_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        color="#111",
        bbox={"boxstyle": "square,pad=0.35", "facecolor": "white", "edgecolor": "#666", "alpha": 0.9},
        zorder=20,
    )

    ax.plot([], [], color="#333", linestyle="-", linewidth=2.0, label="Agent trajectories (different colors)")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def render_utilization(states: dict[str, AgentState], output_path: Path) -> None:
    active = [s for s in states.values() if s.task_ids]
    active.sort(key=lambda s: s.total_km, reverse=True)

    names = [s.agent_id for s in active]
    km_values = [s.total_km for s in active]
    km_limits = [MAX_DAILY_KM_BY_TYPE[s.vehicle_type] for s in active]
    h_values = [s.total_hours for s in active]
    h_limits = [MAX_SHIFT_HOURS_BY_TYPE[s.vehicle_type] for s in active]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    ax1.bar(names, km_values, color="#4C78A8", alpha=0.85, label="Used km")
    ax1.plot(names, km_limits, color="#E45756", linewidth=1.5, label="Km limit")
    ax1.set_ylabel("Kilometers")
    ax1.set_title("Daily Kilometer Utilization by Agent")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.2)

    ax2.bar(names, h_values, color="#72B7B2", alpha=0.85, label="Used hours")
    ax2.plot(names, h_limits, color="#F58518", linewidth=1.5, label="Shift limit")
    ax2.set_ylabel("Hours")
    ax2.set_title("Daily Shift-Time Utilization by Agent")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.2)

    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=90, fontsize=7)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
