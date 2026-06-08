"""
gap_vrp_solver.py
=================
Трёхэтапный оптимизирующий солвер для задачи маршрутизации сборки отходов.

Шаг 1 — Формирование задач (Task generation)
    Транспортная задача LP (scipy): оптимально распределяет потоки мусора
    от источников (MNO) к стокам (object1), минимизируя суммарное расстояние.
    Фолбэк: жадный nearest-sink если scipy недоступен.

Шаг 2 — GAP: Lagrangean Relaxation (Fisher, Jaikumar, Van Wassenhove 1986)
    Распределяет задачи по автомобилям с учётом:
    • ограничения по км (MAX_DAILY_KM_BY_TYPE)
    • ограничения по часам смены (MAX_SHIFT_HOURS_BY_TYPE)
    • типа контейнера / типа автомобиля (CONTAINER_TO_VEHICLE_TYPES)
    • ограничения центральной зоны (is_compact)
    Множители μ_j дуализируют ограничение «задача назначена ровно одной машине»,
    задача разбивается на m независимых задач о рюкзаке.

Шаг 3 — VRP: Nearest Neighbour + 2-opt (Laporte 2009)
    Для каждого автомобиля строит оптимальный маршрут по назначенным задачам,
    используя кеш кратчайших путей из NetworkX.

Метрика: суммарные тонна·км = Σ (масса задачи × длина маршрута задачи).

Совместимость: полностью использует структуры dataset.py (Route, RoutingDataset,
AgentState и т.д.) и simple_solver_components.py (граф, кеш путей, рендеринг).
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Callable

import networkx as nx

# ── Импорт инфраструктуры проекта ────────────────────────────────────────────
from .dataset import (
    AgentsFleet,
    CONTAINER_TO_VEHICLE_TYPES,
    Route,
    RoutingDataset,
    Task,
    dataset_from_dict,
)
from . import core

# ── Константы из core (переиспользуем) ───────────────────────────────────────
MAX_DAILY_KM      = core.MAX_DAILY_KM_BY_TYPE
MAX_SHIFT_HOURS   = core.MAX_SHIFT_HOURS_BY_TYPE
AVG_SPEED         = core.AVG_SPEED_KMPH_BY_TYPE
SERVICE_HOURS     = core.SERVICE_HOURS_BY_CONTAINER


# ═════════════════════════════════════════════════════════════════════════════
# Вспомогательные функции
# ═════════════════════════════════════════════════════════════════════════════

def _sp(
    graph: nx.DiGraph,
    cache: dict[tuple[str, str], tuple[list[str], float] | None],
    src: str,
    dst: str,
) -> tuple[list[str], float] | None:
    """Кешированный кратчайший путь."""
    return core.shortest_path_cached(graph, cache, src, dst)


def _dist(
    graph: nx.DiGraph,
    cache: dict[tuple[str, str], tuple[list[str], float] | None],
    src: str,
    dst: str,
) -> float:
    """Расстояние между двумя узлами, inf если путь недостижим."""
    result = _sp(graph, cache, src, dst)
    return result[1] if result is not None else float("inf")


def _progress(
    iterable,
    *,
    enabled: bool = False,
    desc: str | None = None,
    total: int | None = None,
    leave: bool = False,
):
    """Возвращает tqdm-обертку для iterable, если прогресс включен."""
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=desc, total=total, leave=leave)


def select_agents_subset(
    dataset: RoutingDataset,
    payload: dict[str, Any],
    *,
    selected_agent_ids: list[str] | tuple[str, ...] | None = None,
    max_agents: int | None = None,
) -> tuple[RoutingDataset, dict[str, Any]]:
    """
    Возвращает dataset/payload, оставляя только выбранных агентов.

    selected_agent_ids имеет приоритет. Если он не задан, используется
    ограничение по первым max_agents агентам в исходном порядке датасета.
    """
    fleet_agent_ids = list(dataset.fleet.agents.keys())

    if selected_agent_ids is not None:
        selected_ids = list(dict.fromkeys(selected_agent_ids))
        unknown_ids = [aid for aid in selected_ids if aid not in dataset.fleet.agents]
        if unknown_ids:
            raise ValueError(f"Unknown agent ids: {unknown_ids}")
    elif max_agents is not None:
        if max_agents <= 0:
            raise ValueError("max_agents must be positive")
        selected_ids = fleet_agent_ids[:max_agents]
    else:
        selected_ids = fleet_agent_ids

    if not selected_ids:
        raise ValueError("At least one agent must be selected.")

    filtered_agents = {
        aid: dataset.fleet.agents[aid]
        for aid in fleet_agent_ids
        if aid in selected_ids
    }
    filtered_routes = [
        route for route in dataset.routes
        if route.agent_id in filtered_agents
    ]

    filtered_dataset = RoutingDataset(
        graph=dataset.graph,
        fleet=AgentsFleet(agents=filtered_agents),
        tasks=list(dataset.tasks),
        routes=filtered_routes,
        metadata=deepcopy(dataset.metadata),
    )

    filtered_payload = deepcopy(payload)
    if "agents" in filtered_payload:
        filtered_payload["agents"] = [
            raw for raw in filtered_payload["agents"]
            if raw.get("agent_id") in filtered_agents
        ]

    metadata = filtered_payload.setdefault("metadata", {})
    depots = metadata.get("agent_depots")
    if isinstance(depots, dict):
        metadata["agent_depots"] = {
            aid: depot for aid, depot in depots.items()
            if aid in filtered_agents
        }
    metadata["selected_agent_ids"] = list(filtered_agents.keys())

    return filtered_dataset, filtered_payload


def _agent_resource(
    state: core.AgentState,
    tasks: list[Task],
    graph: nx.DiGraph,
    cache: dict[tuple[str, str], tuple[list[str], float] | None],
) -> tuple[float, float] | None:
    """
    Возвращает (total_km, total_hours) для агента с заданным набором задач.
    None если маршрут нельзя построить.
    """
    plan = core.evaluate_agent_task_set(state, tasks, graph, cache)
    if plan is None:
        return None
    return plan.total_km, plan.total_hours


def _is_compatible(
    dataset: RoutingDataset,
    state: core.AgentState,
    task: Task,
) -> bool:
    """Проверяет совместимость агента и задачи по типу, массе, объему и зоне."""
    return core.is_task_compatible_with_agent_state(dataset=dataset, state=state, task=task)


# ═════════════════════════════════════════════════════════════════════════════
# ШАГ 1: Формирование задач
# ═════════════════════════════════════════════════════════════════════════════

def generate_tasks_lp(
    dataset: RoutingDataset,
    graph: nx.DiGraph,
    cache: dict[tuple[str, str], tuple[list[str], float] | None],
    show_progress: bool = False,
    progress_hook: Callable[[str], None] | None = None,
) -> list[Task]:
    """
    Транспортная задача LP (scipy.optimize.linprog / HiGHS):
        min  Σ_{i,j} d_{ij} · x_{ij}
        s.t. Σ_j x_{ij} = waste_i   (весь мусор с источника вывезен)
             Σ_i x_{ij} ≤ cap_j     (дневная вместимость стока)
             x_{ij} ≥ 0

    Один источник может быть разбит между несколькими стоками (дробные задачи).
    Фолбэк на жадный nearest-sink если scipy недоступен или LP неразрешима.
    """
    def _emit(message: str) -> None:
        if progress_hook is not None:
            progress_hook(f"[GAP-step1] {message}")
        elif show_progress:
            print(f"[GAP-step1] {message}", flush=True)

    try:
        from scipy.optimize import linprog
    except ImportError:
        _emit("scipy not found, fallback to greedy")
        return generate_tasks_greedy(
            dataset,
            graph,
            cache,
            show_progress=show_progress,
            progress_hook=progress_hook,
        )

    sources = [n for n in dataset.graph.nodes.values()
               if n.kind == "mno" and n.daily_mass_tons > 0]
    sinks   = [n for n in dataset.graph.nodes.values()
               if n.kind.startswith("object")]

    if not sources or not sinks:
        _emit("no sources or sinks, fallback to greedy")
        return generate_tasks_greedy(
            dataset,
            graph,
            cache,
            show_progress=show_progress,
            progress_hook=progress_hook,
        )

    ns, nk = len(sources), len(sinks)
    total_waste = sum(s.daily_mass_tons for s in sources)
    _emit(
        f"LP setup: sources={ns}, sinks={nk}, variables={ns * nk}, "
        f"total_waste_tons={total_waste:.3f}"
    )

    # Матрица расстояний source_i → sink_j
    t_dist = time.perf_counter()
    dist_flat: list[float] = []
    source_iter = _progress(
        sources,
        enabled=show_progress,
        desc="LP: матрица расстояний",
        total=len(sources),
    )
    log_every = max(1, ns // 20) if ns else 1
    for i, src in enumerate(source_iter):
        for snk in sinks:
            dist_flat.append(_dist(graph, cache, src.node_id, snk.node_id))
        if progress_hook is not None and not show_progress:
            if ((i + 1) % log_every == 0) or (i + 1 == ns):
                _emit(
                    f"distance matrix: sources={i + 1}/{ns}, "
                    f"elapsed={time.perf_counter() - t_dist:.1f}s"
                )
    _emit(f"distance matrix built in {time.perf_counter() - t_dist:.1f}s")

    # Ограничения равенства: Σ_j x_ij = waste_i
    t_eq = time.perf_counter()
    A_eq = []
    b_eq = []
    for i in range(ns):
        row = [0.0] * (ns * nk)
        for j in range(nk):
            row[i * nk + j] = 1.0
        A_eq.append(row)
        b_eq.append(sources[i].daily_mass_tons)
    _emit(f"LP eq constraints built: {len(A_eq)} rows in {time.perf_counter() - t_eq:.1f}s")

    # Ограничения неравенства: Σ_i x_ij ≤ cap_j
    t_ub = time.perf_counter()
    A_ub = []
    b_ub = []
    for j, snk in enumerate(sinks):
        cap = snk.object_day_capacity_tons if snk.object_day_capacity_tons > 0 else total_waste
        row = [0.0] * (ns * nk)
        for i in range(ns):
            row[i * nk + j] = 1.0
        A_ub.append(row)
        b_ub.append(cap)
    _emit(f"LP ub constraints built: {len(A_ub)} rows in {time.perf_counter() - t_ub:.1f}s")

    bounds = [(0, None)] * (ns * nk)
    _emit("linprog start: method=highs")
    t_lp = time.perf_counter()
    res = linprog(dist_flat, A_ub=A_ub or None, b_ub=b_ub or None,
                  A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    _emit(
        f"linprog done in {time.perf_counter() - t_lp:.1f}s: "
        f"success={res.success}, status={getattr(res, 'status', None)}"
    )

    if not res.success:
        _emit("LP failed, fallback to greedy")
        return generate_tasks_greedy(
            dataset,
            graph,
            cache,
            show_progress=show_progress,
            progress_hook=progress_hook,
        )

    EPS = 1e-6
    t_build = time.perf_counter()
    tasks: list[Task] = []
    tid = 0
    build_iter = _progress(
        sources,
        enabled=show_progress,
        desc="LP: формирование задач",
        total=len(sources),
    )
    for i, src in enumerate(build_iter):
        for j, snk in enumerate(sinks):
            flow = res.x[i * nk + j]
            if flow > EPS:
                # Проверяем достижимость пути
                if _sp(graph, cache, src.node_id, snk.node_id) is None:
                    continue
                # Выбираем тип контейнера из совместимых с источником
                container_type = (
                    src.container_types[0] if src.container_types else "Type1"
                )
                tasks.append(Task(
                    task_id=f"LP_T{tid:04d}",
                    source_node_id=src.node_id,
                    destination_node_id=snk.node_id,
                    container_type=container_type,
                    mass_tons=round(flow, 4),
                    periodicity="daily",
                ))
                tid += 1
        if progress_hook is not None and not show_progress:
            if ((i + 1) % log_every == 0) or (i + 1 == ns):
                _emit(
                    f"build tasks: sources={i + 1}/{ns}, "
                    f"tasks={len(tasks)}, elapsed={time.perf_counter() - t_build:.1f}s"
                )
    _emit(f"step1 LP tasks built: tasks={len(tasks)}, elapsed={time.perf_counter() - t_build:.1f}s")
    return tasks if tasks else generate_tasks_greedy(
        dataset,
        graph,
        cache,
        show_progress=show_progress,
        progress_hook=progress_hook,
    )


def generate_tasks_greedy(
    dataset: RoutingDataset,
    graph: nx.DiGraph,
    cache: dict[tuple[str, str], tuple[list[str], float] | None],
    show_progress: bool = False,
    progress_hook: Callable[[str], None] | None = None,
) -> list[Task]:
    """
    Жадный фолбэк: каждый источник → ближайший достижимый сток.
    """
    def _emit(message: str) -> None:
        if progress_hook is not None:
            progress_hook(f"[GAP-step1] {message}")
        elif show_progress:
            print(f"[GAP-step1] {message}", flush=True)

    sources = [n for n in dataset.graph.nodes.values()
               if n.kind == "mno" and n.daily_mass_tons > 0]
    sinks   = [n for n in dataset.graph.nodes.values()
               if n.kind.startswith("object")]
    _emit(f"Greedy setup: sources={len(sources)}, sinks={len(sinks)}")

    tasks: list[Task] = []
    src_iter = _progress(
        sources,
        enabled=show_progress,
        desc="Greedy: формирование задач",
        total=len(sources),
    )
    log_every = max(1, len(sources) // 20) if sources else 1
    t0 = time.perf_counter()
    for tid, src in enumerate(src_iter):
        # Ближайший достижимый сток
        best_snk = None
        best_d   = float("inf")
        for snk in sinks:
            d = _dist(graph, cache, src.node_id, snk.node_id)
            if d < best_d:
                best_d, best_snk = d, snk
        if best_snk is None:
            continue
        container_type = src.container_types[0] if src.container_types else "Type1"
        tasks.append(Task(
            task_id=f"GR_T{tid:04d}",
            source_node_id=src.node_id,
            destination_node_id=best_snk.node_id,
            container_type=container_type,
            mass_tons=src.daily_mass_tons,
            periodicity="daily",
        ))
        if progress_hook is not None and not show_progress:
            if ((tid + 1) % log_every == 0) or (tid + 1 == len(sources)):
                _emit(
                    f"greedy build tasks: sources={tid + 1}/{len(sources)}, "
                    f"tasks={len(tasks)}, elapsed={time.perf_counter() - t0:.1f}s"
                )
    _emit(f"greedy tasks built: tasks={len(tasks)}, elapsed={time.perf_counter() - t0:.1f}s")
    return tasks


# ═════════════════════════════════════════════════════════════════════════════
# ШАГ 2: GAP — Lagrangean Relaxation (FJV 1986)
# ═════════════════════════════════════════════════════════════════════════════

def _cost_ij(
    task: Task,
    state: core.AgentState,
    graph: nx.DiGraph,
    cache: dict[tuple[str, str], tuple[list[str], float] | None],
) -> float:
    """
    Стоимость назначения задачи j на агента i:
    c_ij = масса × (depot→source→dest→depot) [тонна·км для одиночного рейса].
    """
    depot = state.depot_node or ""
    d = (_dist(graph, cache, depot, task.source_node_id)
         + _dist(graph, cache, task.source_node_id, task.destination_node_id)
         + _dist(graph, cache, task.destination_node_id, depot))
    return task.mass_tons * d


def _resource_ij(
    task: Task,
    state: core.AgentState,
    graph: nx.DiGraph,
    cache: dict[tuple[str, str], tuple[list[str], float] | None],
) -> tuple[float, float]:
    """
    Потребление ресурса агента при одиночном рейсе с задачей j:
    возвращает (km, hours).
    """
    depot = state.depot_node or ""
    d_repo = _dist(graph, cache, depot, task.source_node_id)
    d_serv = _dist(graph, cache, task.source_node_id, task.destination_node_id)
    d_back = _dist(graph, cache, task.destination_node_id, depot)
    total_km = d_repo + d_serv + d_back
    speed    = AVG_SPEED.get(state.vehicle_type, 24.0)
    svc_h    = SERVICE_HOURS.get(task.container_type, 0.25)
    total_h  = total_km / speed + svc_h
    return total_km, total_h


def _knapsack_2d(
    adj_costs: list[float],
    km_weights: list[float],
    h_weights: list[float],
    km_cap: float,
    h_cap: float,
    progress_hook: Any | None = None,
) -> tuple[list[int], float]:
    """
    2D 0-1 Knapsack по (км, часы) — DP.
    Шаг 1 км и 0.25 ч (ступень 15 мин).
    Минимизирует adj_costs при (km ≤ km_cap, hours ≤ h_cap).
    """
    n  = len(adj_costs)
    if n == 0 or km_cap <= 0 or h_cap <= 0:
        return [], 0.0

    KM_STEP = 1
    H_STEP  = 0.25
    W  = int(km_cap / KM_STEP) + 1
    H  = int(h_cap  / H_STEP)  + 1
    INF = float("inf")

    # dp[w][h] = минимальная стоимость при ≤w км и ≤h*H_STEP часов
    dp  = [[INF] * H for _ in range(W)]
    # parent[w][h] = (prev_w, prev_h, item_idx) для восстановления решения
    parent: list[list[tuple[int, int, int] | None]] = [[None] * H for _ in range(W)]
    dp[0][0] = 0.0

    # Чтобы прогресс был видимым, но без лишнего шума.
    hook_every = max(1, n // 5)
    for idx in range(n):
        if progress_hook is not None and (idx == 0 or (idx + 1) % hook_every == 0 or idx + 1 == n):
            progress_hook(idx + 1, n)
        w = max(1, int(math.ceil(km_weights[idx] / KM_STEP)))
        h = max(1, int(math.ceil(h_weights[idx]  / H_STEP)))
        if w >= W or h >= H:
            continue
        for cw in range(W - 1, w - 1, -1):
            for ch in range(H - 1, h - 1, -1):
                nv = dp[cw - w][ch - h] + adj_costs[idx]
                if nv < dp[cw][ch]:
                    dp[cw][ch] = nv
                    parent[cw][ch] = (cw - w, ch - h, idx)

    # Ищем ячейку с минимальной стоимостью
    best_v = INF
    best_w = 0
    best_h = 0
    for cw in range(W):
        for ch in range(H):
            if dp[cw][ch] < best_v:
                best_v = dp[cw][ch]
                best_w = cw
                best_h = ch

    # Восстановление выбранных предметов
    best_sel: list[int] = []
    cur_w, cur_h = best_w, best_h
    while True:
        step = parent[cur_w][cur_h]
        if step is None:
            break
        prev_w, prev_h, item_idx = step
        best_sel.append(item_idx)
        cur_w, cur_h = prev_w, prev_h
    best_sel.reverse()
    return best_sel, best_v


def solve_gap_lagrangean(
    tasks: list[Task],
    states: list[core.AgentState],
    dataset: RoutingDataset,
    graph: nx.DiGraph,
    cache: dict[tuple[str, str], tuple[list[str], float] | None],
    max_iter: int = 120,
    step0: float = 2.0,
    show_progress: bool = False,
    progress_hook: Callable[[str], None] | None = None,
) -> dict[str, list[Task]]:
    """
    GAP через Lagrangean Relaxation (Fisher, Jaikumar, Van Wassenhove 1986).

    Дуализируем ограничение «каждая задача назначена ровно одной машине»:
        min Σ_ij c_ij·x_ij + Σ_j μ_j·(1 - Σ_i x_ij)
    Задача распадается на m независимых 2D-knapsack по (km, hours).

    Субградиентный метод обновления μ_j:
        g_j = 1 - Σ_i x_ij
        μ_j ← max(0, μ_j + α·g_j)
        α   = step0 · (UB - LB) / ‖g‖² / (1 + it·0.04)

    Если за max_iter итераций допустимое решение не найдено — greedy repair.
    """
    m = len(states)
    n = len(tasks)
    if m == 0 or n == 0:
        return {s.agent_id: [] for s in states}

    def _emit(message: str) -> None:
        if progress_hook is not None:
            progress_hook(f"[GAP-LR] {message}")
        elif show_progress:
            print(f"[GAP-LR] {message}", flush=True)

    _emit(f"prepare matrices: agents={m}, tasks={n}, max_iter={max_iter}")

    # Предвычисляем стоимости и ресурсы только для совместимых пар
    c: list[list[float]] = []
    a_km: list[list[float]] = []
    a_h:  list[list[float]] = []
    precompute_log_every = max(1, m // 10)
    for i, state in enumerate(_progress(
        states,
        enabled=show_progress,
        desc="GAP: предвычисление стоимостей",
        total=len(states),
    )):
        ci, aki, ahi = [], [], []
        for j, task in enumerate(tasks):
            if not _is_compatible(dataset, state, task):
                ci.append(float("inf"))
                aki.append(float("inf"))
                ahi.append(float("inf"))
            else:
                cval = _cost_ij(task, state, graph, cache)
                km, h = _resource_ij(task, state, graph, cache)
                ci.append(cval)
                aki.append(km)
                ahi.append(h)
        c.append(ci)
        a_km.append(aki)
        a_h.append(ahi)
        if progress_hook is not None and not show_progress:
            if (i + 1) % precompute_log_every == 0 or (i + 1) == m:
                _emit(f"precompute costs: {i + 1}/{m} agents")
    _emit("precompute done")

    # Начальные множители: второй минимум c_ij по i (FJV 1986)
    mu: list[float] = []
    for j in range(n):
        finite_vals = sorted(v for v in (c[i][j] for i in range(m)) if v < float("inf"))
        mu.append(finite_vals[1] if len(finite_vals) > 1 else (finite_vals[0] if finite_vals else 0.0))

    best_assign: dict[int, list[int]] = {}
    UB = float("inf")

    outer_iter = _progress(
        range(max_iter),
        enabled=show_progress,
        desc="GAP: итерации",
        total=max_iter,
    )
    iter_log_every = max(1, max_iter // 10)
    for it in outer_iter:
        iter_t0 = time.perf_counter()
        if show_progress:
            print(f"[GAP] iter {it + 1}/{max_iter} started", flush=True)
        elif progress_hook is not None and (
            (it == 0) or ((it + 1) % iter_log_every == 0) or (it + 1 == max_iter)
        ):
            _emit(f"iter {it + 1}/{max_iter} started")
        assign: dict[int, list[int]] = {}
        lb = sum(mu)

        inner_iter = _progress(
            enumerate(states),
            enabled=show_progress,
            desc=f"GAP it {it + 1}/{max_iter}: агенты",
            total=len(states),
            leave=False,
        )
        agent_log_every = max(1, len(states) // 10)
        for i, state in inner_iter:
            km_cap = MAX_DAILY_KM.get(state.vehicle_type, 130.0)
            h_cap  = MAX_SHIFT_HOURS.get(state.vehicle_type, 10.0)

            # Фильтруем несовместимые задачи
            free_j  = [j for j in range(n) if a_km[i][j] < float("inf")]
            adj     = [c[i][j] - mu[j] for j in free_j]
            wts_km  = [a_km[i][j]      for j in free_j]
            wts_h   = [a_h[i][j]       for j in free_j]

            if show_progress and hasattr(inner_iter, "set_postfix"):
                inner_iter.set_postfix(
                    {
                        "agent": state.agent_id,
                        "free_tasks": len(free_j),
                        "phase": "knapsack",
                    }
                )

            agent_t0 = time.perf_counter()

            def _knapsack_progress(done: int, total: int) -> None:
                if show_progress and hasattr(inner_iter, "set_postfix"):
                    inner_iter.set_postfix(
                        {
                            "agent": state.agent_id,
                            "free_tasks": len(free_j),
                            "kp": f"{done}/{total}",
                        }
                    )

            sel_idx, val = _knapsack_2d(
                adj,
                wts_km,
                wts_h,
                km_cap,
                h_cap,
                progress_hook=_knapsack_progress if show_progress else None,
            )

            if show_progress and hasattr(inner_iter, "set_postfix"):
                inner_iter.set_postfix(
                    {
                        "agent": state.agent_id,
                        "free_tasks": len(free_j),
                        "kp": "done",
                        "t_sec": round(time.perf_counter() - agent_t0, 2),
                    }
                )
            if show_progress and (((i + 1) % agent_log_every == 0) or (i + 1 == len(states))):
                print(
                    f"[GAP] iter {it + 1}/{max_iter}, agent {i + 1}/{len(states)}, "
                    f"id={state.agent_id}, free_tasks={len(free_j)}, "
                    f"agent_sec={time.perf_counter() - agent_t0:.2f}",
                    flush=True,
                )
            elif progress_hook is not None and (
                ((it == 0) or ((it + 1) % iter_log_every == 0) or (it + 1 == max_iter))
                and (((i + 1) % agent_log_every == 0) or (i + 1 == len(states)))
            ):
                _emit(
                    f"iter {it + 1}/{max_iter}, agent {i + 1}/{len(states)}, "
                    f"id={state.agent_id}, free_tasks={len(free_j)}"
                )
            assign[i] = [free_j[k] for k in sel_idx]
            lb += val

        count = [0] * n
        for sel in assign.values():
            for j in sel:
                count[j] += 1

        # Проверка допустимости
        if all(count[j] == 1 for j in range(n)):
            cost = sum(c[i][j] for i, sel in assign.items() for j in sel
                       if c[i][j] < float("inf"))
            if cost < UB:
                UB = cost
                best_assign = {i: list(sel) for i, sel in assign.items()}

        # Субградиент
        grad  = [1 - count[j] for j in range(n)]
        norm2 = sum(g * g for g in grad)
        if norm2 < 1e-9:
            if show_progress and hasattr(outer_iter, "set_postfix"):
                outer_iter.set_postfix(
                    {
                        "LB": round(lb, 2),
                        "UB": round(UB, 2) if UB < float("inf") else "inf",
                        "assigned": f"{sum(1 for x in count if x == 1)}/{n}",
                        "status": "converged",
                    }
                )
            if show_progress:
                print(
                    f"[GAP] iter {it + 1}/{max_iter} converged: "
                    f"LB={lb:.2f}, UB={UB if UB < float('inf') else 'inf'}, "
                    f"assigned={sum(1 for x in count if x == 1)}/{n}",
                    flush=True,
                )
            elif progress_hook is not None:
                _emit(
                    f"iter {it + 1}/{max_iter} converged: "
                    f"LB={lb:.2f}, UB={UB if UB < float('inf') else 'inf'}, "
                    f"assigned={sum(1 for x in count if x == 1)}/{n}, "
                    f"iter_sec={time.perf_counter() - iter_t0:.1f}"
                )
            break

        ub_ref = UB if UB < float("inf") else lb * 1.15 + 10.0
        alpha  = step0 * max(ub_ref - lb, 0.0) / norm2 / (1.0 + it * 0.04)
        for j in range(n):
            mu[j] = max(0.0, mu[j] + alpha * grad[j])

        if show_progress and hasattr(outer_iter, "set_postfix"):
            outer_iter.set_postfix(
                {
                    "LB": round(lb, 2),
                    "UB": round(UB, 2) if UB < float("inf") else "inf",
                    "assigned": f"{sum(1 for x in count if x == 1)}/{n}",
                }
            )
        if show_progress:
            print(
                f"[GAP] iter {it + 1}/{max_iter} done: "
                f"LB={lb:.2f}, UB={UB if UB < float('inf') else 'inf'}, "
                f"assigned={sum(1 for x in count if x == 1)}/{n}",
                flush=True,
            )
        elif progress_hook is not None and (
            (it == 0) or ((it + 1) % iter_log_every == 0) or (it + 1 == max_iter)
        ):
            _emit(
                f"iter {it + 1}/{max_iter} done: "
                f"LB={lb:.2f}, UB={UB if UB < float('inf') else 'inf'}, "
                f"assigned={sum(1 for x in count if x == 1)}/{n}, "
                f"iter_sec={time.perf_counter() - iter_t0:.1f}"
            )

    if not best_assign:
        _emit("no feasible dual assignment found, run greedy repair")
        best_assign = _greedy_repair(n, m, states, dataset, c, a_km, a_h)
    _emit("gap allocation ready")

    return {
        states[i].agent_id: [tasks[j] for j in idx_list]
        for i, idx_list in best_assign.items()
    }


def _greedy_repair(
    n: int,
    m: int,
    states: list[core.AgentState],
    dataset: RoutingDataset,
    c: list[list[float]],
    a_km: list[list[float]],
    a_h:  list[list[float]],
) -> dict[int, list[int]]:
    """
    Жадный repair: задача (по убыванию waste) → агент с мин. c_ij
    при достаточном (km, hours) ресурсе.
    """
    rem_km = [MAX_DAILY_KM.get(s.vehicle_type, 130.0) for s in states]
    rem_h  = [MAX_SHIFT_HOURS.get(s.vehicle_type, 10.0) for s in states]
    asgn: dict[int, list[int]] = {i: [] for i in range(m)}
    unassigned: list[int] = []

    # Сортируем задачи по убыванию стоимости (тяжёлые — первыми)
    order = sorted(range(n), key=lambda j: -max(
        c[i][j] for i in range(m) if c[i][j] < float("inf")
    ) if any(c[i][j] < float("inf") for i in range(m)) else 0)

    for j in order:
        best_i = None
        best_c = float("inf")
        for i in range(m):
            if (a_km[i][j] < float("inf")
                    and a_km[i][j] <= rem_km[i] + 1e-6
                    and a_h[i][j]  <= rem_h[i]  + 1e-6
                    and c[i][j] < best_c):
                best_c, best_i = c[i][j], i
        if best_i is not None:
            asgn[best_i].append(j)
            rem_km[best_i] -= a_km[best_i][j]
            rem_h[best_i]  -= a_h[best_i][j]
        else:
            unassigned.append(j)

    # Нераспределённые → агент с наибольшим остатком км
    for j in unassigned:
        i = max(range(m), key=lambda i: rem_km[i])
        asgn[i].append(j)

    return asgn


# ═════════════════════════════════════════════════════════════════════════════
# ШАГ 3: VRP — Nearest Neighbour + 2-opt на реальном графе
# ═════════════════════════════════════════════════════════════════════════════

def _vrp_nn_order(
    start: str,
    targets: list[str],
    graph: nx.DiGraph,
    cache: dict[tuple[str, str], tuple[list[str], float] | None],
) -> list[str]:
    """Nearest Neighbour: строим порядок посещения точек из start."""
    unvisited = list(targets)
    order: list[str] = []
    cur = start
    while unvisited:
        nxt = min(unvisited, key=lambda x: _dist(graph, cache, cur, x))
        order.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    return order


def _vrp_2opt(
    stop_order: list[str],
    graph: nx.DiGraph,
    cache: dict[tuple[str, str], tuple[list[str], float] | None],
) -> list[str]:
    """
    2-opt на последовательности остановок (без depot на концах).
    Переворачиваем подотрезок [i..j] если это уменьшает суммарный пробег.
    """
    best = list(stop_order)
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 2, len(best)):
                d_before = (_dist(graph, cache, best[i], best[i + 1])
                            + _dist(graph, cache, best[j - 1], best[j] if j < len(best) else best[0]))
                d_after  = (_dist(graph, cache, best[i], best[j - 1])
                            + _dist(graph, cache, best[i + 1], best[j] if j < len(best) else best[0]))
                if d_after < d_before - 1e-9:
                    best[i + 1:j] = best[i + 1:j][::-1]
                    improved = True
    return best


def build_vrp_route(
    state: core.AgentState,
    assigned_tasks: list[Task],
    graph: nx.DiGraph,
    cache: dict[tuple[str, str], tuple[list[str], float] | None],
) -> core.AgentDayPlan | None:
    """
    Строит оптимальный маршрут для агента по назначенным задачам:
    1. Собирает уникальные sources и destinations
    2. NN-порядок для sources (сначала забираем весь мусор)
    3. NN-порядок для destinations (потом везём на полигоны)
    4. 2-opt улучшение порядка внутри каждой группы
    5. Собирает полный план через core.evaluate_agent_task_set

    Возвращает AgentDayPlan или None если маршрут недостижим.
    """
    if not assigned_tasks:
        return core.evaluate_agent_task_set(state, [], graph, cache)

    depot = state.depot_node or ""

    # Уникальные источники и стоки
    sources = list(dict.fromkeys(t.source_node_id      for t in assigned_tasks))
    dests   = list(dict.fromkeys(t.destination_node_id for t in assigned_tasks))

    # NN + 2-opt для порядка сбора
    src_order  = _vrp_nn_order(depot, sources, graph, cache)
    src_order  = _vrp_2opt(src_order, graph, cache)

    # NN + 2-opt для порядка доставки (начинаем от последнего source)
    last_src   = src_order[-1] if src_order else depot
    dest_order = _vrp_nn_order(last_src, dests, graph, cache)
    dest_order = _vrp_2opt(dest_order, graph, cache)

    # Переупорядочиваем задачи под найденный порядок остановок
    # Каждый source → соответствующие задачи, каждый dest → получает задачи
    stop_sequence = src_order + dest_order

    # Переупорядочиваем задачи: сначала задачи чей source встречается раньше
    src_rank  = {s: i for i, s in enumerate(src_order)}
    dest_rank = {d: i for i, d in enumerate(dest_order)}
    ordered_tasks = sorted(
        assigned_tasks,
        key=lambda t: (src_rank.get(t.source_node_id, 999),
                       dest_rank.get(t.destination_node_id, 999)),
    )

    # Используем core для финального расчёта метрик
    return core.evaluate_agent_task_set(state, ordered_tasks, graph, cache)


# ═════════════════════════════════════════════════════════════════════════════
# Iterative Repair: устраняем нарушения после VRP
# ═════════════════════════════════════════════════════════════════════════════

def iterative_repair(
    allocation: dict[str, list[Task]],
    states_map: dict[str, core.AgentState],
    dataset: RoutingDataset,
    graph: nx.DiGraph,
    cache: dict[tuple[str, str], tuple[list[str], float] | None],
    max_rounds: int = 15,
) -> dict[str, list[Task]]:
    """
    После VRP проверяем feasibility каждого агента.
    Если агент нарушает km или hours — отдаём его «самую дорогую» задачу
    агенту с наибольшим остатком ресурса. Повторяем до устранения.
    """
    alloc = {aid: list(tasks) for aid, tasks in allocation.items()}

    for _ in range(max_rounds):
        violators = []
        for aid, tasks in alloc.items():
            state = states_map[aid]
            plan = core.evaluate_agent_task_set(state, tasks, graph, cache)
            if plan is not None and not plan.feasible:
                violators.append(aid)

        if not violators:
            break

        changed = False
        for aid in violators:
            state = states_map[aid]
            tasks = alloc[aid]
            if not tasks:
                continue

            # Задача с максимальным вкладом в пробег
            donate = max(tasks, key=lambda t: _dist(graph, cache,
                         state.depot_node or "", t.source_node_id)
                         + _dist(graph, cache, t.source_node_id, t.destination_node_id)
                         + _dist(graph, cache, t.destination_node_id, state.depot_node or ""))

            # Ищем реципиента
            for rec_aid, rec_tasks in alloc.items():
                if rec_aid == aid:
                    continue
                rec_state = states_map[rec_aid]
                if not _is_compatible(dataset, rec_state, donate):
                    continue
                trial = rec_tasks + [donate]
                trial_plan = core.evaluate_agent_task_set(rec_state, trial, graph, cache)
                if trial_plan is not None and trial_plan.feasible:
                    alloc[aid].remove(donate)
                    alloc[rec_aid].append(donate)
                    changed = True
                    break
            if changed:
                break

        if not changed:
            break

    return alloc


# ═════════════════════════════════════════════════════════════════════════════
# Полный пайплайн: GAP-VRP Solver
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class SolverResult:
    """Результат одного запуска солвера."""
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


def run_gap_vrp(
    dataset: RoutingDataset,
    payload: dict[str, Any],
    graph: nx.DiGraph,
    cache: dict[tuple[str, str], tuple[list[str], float] | None],
    step1_method: str = "lp",      # "lp" | "greedy" | "dataset"
    gap_iter: int = 120,
    use_repair: bool = True,
    verbose: bool = True,
    show_progress: bool = False,
    progress_hook: Callable[[str], None] | None = None,
) -> SolverResult:
    """
    Полный трёхэтапный пайплайн GAP+VRP.

    step1_method:
        "lp"     — транспортная задача (scipy), оптимальное распределение потоков
        "greedy" — жадный nearest-sink
        "dataset" — использовать исходные dataset.tasks без генерации шага 1

    gap_iter: число итераций субградиентного метода GAP.
    use_repair: применять iterative repair после VRP.
    """
    label = f"GAP-Lagrangean + VRP(NN+2opt) [step1={step1_method}]"
    run_t0 = time.perf_counter()

    def _emit(message: str) -> None:
        if progress_hook is not None:
            progress_hook(f"[GAP-VRP] {message}")
        elif show_progress:
            print(f"[GAP-VRP] {message}", flush=True)

    if verbose:
        print(f"\n{'═'*60}")
        print(f"  {label}")
        print(f"{'═'*60}")
    else:
        _emit(f"start: {label}")

    # ── Шаг 1: Формирование задач ─────────────────────────────
    t1 = time.perf_counter()
    if verbose:
        print("\n[Шаг 1] Формирование задач...")
    else:
        _emit("step1: task generation")
    if step1_method == "dataset":
        tasks = list(dataset.tasks)
    elif step1_method == "lp":
        tasks = generate_tasks_lp(
            dataset,
            graph,
            cache,
            show_progress=show_progress,
            progress_hook=progress_hook,
        )
    else:
        tasks = generate_tasks_greedy(
            dataset,
            graph,
            cache,
            show_progress=show_progress,
            progress_hook=progress_hook,
        )
    _emit(f"step1 done in {time.perf_counter() - t1:.1f}s, tasks={len(tasks)}")

    if verbose:
        total_mass = sum(t.mass_tons for t in tasks)
        print(f"  Задач: {len(tasks)}, суммарно: {total_mass:.2f} т")

    if not tasks:
        states = core.initialize_agent_states(dataset, payload)
        return SolverResult(label, [], states, [], None, False, [], 0, 0, 0)

    # ── Инициализация состояний агентов ───────────────────────
    states = core.initialize_agent_states(dataset, payload)
    agent_list = list(states.values())

    # ── Шаг 2: GAP ────────────────────────────────────────────
    t2 = time.perf_counter()
    if verbose:
        print(f"\n[Шаг 2] GAP: Lagrangean Relaxation ({gap_iter} итераций)...")
    else:
        _emit(f"step2: GAP Lagrangean, iter={gap_iter}")
    allocation = solve_gap_lagrangean(
        tasks, agent_list, dataset, graph, cache, max_iter=gap_iter,
        show_progress=show_progress,
        progress_hook=progress_hook,
    )
    assigned_after_gap = sum(len(v) for v in allocation.values())
    _emit(
        f"step2 done in {time.perf_counter() - t2:.1f}s, "
        f"assigned_after_gap={assigned_after_gap}/{len(tasks)}"
    )

    if verbose:
        for aid, atasks in allocation.items():
            if atasks:
                print(f"  Агент {aid}: {len(atasks)} задач, "
                      f"{sum(t.mass_tons for t in atasks):.2f} т")

    # ── Шаг 3: VRP ────────────────────────────────────────────
    t3 = time.perf_counter()
    if verbose:
        print("\n[Шаг 3] VRP: NN + 2-opt...")
    else:
        _emit("step3: VRP build + validation")

    # Iterative repair если нужно
    if use_repair:
        allocation = iterative_repair(allocation, states, dataset, graph, cache)

    # Строим маршруты через VRP
    routes: list[Route] = []
    unassigned: list[str] = []
    route_counter = 1

    active_allocations = [(aid, atasks) for aid, atasks in allocation.items() if atasks]
    vrp_log_every = max(1, len(active_allocations) // 10) if active_allocations else 1
    vrp_iter = _progress(
        active_allocations,
        enabled=show_progress,
        desc="VRP: построение маршрутов",
        total=len(active_allocations),
    )
    for idx, (aid, atasks) in enumerate(vrp_iter, start=1):
        state = states[aid]

        plan = build_vrp_route(state, atasks, graph, cache)
        if plan is None or not plan.feasible:
            for t in atasks:
                unassigned.append(t.task_id)
            continue

        core.apply_plan_to_state(state, plan)

        for task in plan.ordered_tasks:
            route_id = f"GAP_ROUTE_{route_counter:04d}"
            route_counter += 1
            state.route_ids.append(route_id)
            routes.append(Route(
                route_id=route_id,
                agent_id=aid,
                path=plan.service_paths[task.task_id],
                task_ids=(task.task_id,),
            ))
        if progress_hook is not None and not show_progress:
            if (idx % vrp_log_every == 0) or (idx == len(active_allocations)):
                _emit(
                    f"step3 progress: agents={idx}/{len(active_allocations)}, "
                    f"routes={len(routes)}, unassigned={len(unassigned)}"
                )

    # ── Метрики ───────────────────────────────────────────────
    limit_violations = core.validate_daily_limits(states)
    feasible = (len(unassigned) == 0 and len(limit_violations) == 0)

    if verbose:
        print(f"\n  Назначено: {len(routes)} маршрутов")
        print(f"  Не назначено: {len(unassigned)} задач")
        print(f"  Нарушений лимитов: {len(limit_violations)}")
        print(f"  Статус: {'✓ feasible' if feasible else '✗ infeasible'}")
    else:
        _emit(
            f"step3 done in {time.perf_counter() - t3:.1f}s, "
            f"routes={len(routes)}, unassigned={len(unassigned)}, "
            f"violations={len(limit_violations)}"
        )
        _emit(f"total runtime: {time.perf_counter() - run_t0:.1f}s")

    return SolverResult(
        method_label=label,
        routes=routes,
        states=states,
        unassigned=unassigned,
        transport_work_ton_km=None,
        feasible=feasible,
        limit_violations=limit_violations,
        n_assigned=len(routes),
        n_unassigned=len(unassigned),
        active_agents=sum(1 for s in states.values() if s.task_ids),
    )

# ═════════════════════════════════════════════════════════════════════════════
# Сравнение с baseline (greedy из simple_solver_components)
# ═════════════════════════════════════════════════════════════════════════════

def run_baseline_greedy(
    dataset: RoutingDataset,
    payload: dict[str, Any],
    graph: nx.DiGraph,
    cache: dict[tuple[str, str], tuple[list[str], float] | None],
    verbose: bool = True,
) -> SolverResult:
    """Запускает оригинальный жадный солвер как baseline."""
    if verbose:
        print(f"\n{'═'*60}")
        print("  Baseline: Greedy (simple_solver_components)")
        print(f"{'═'*60}")

    routes, states, unassigned_ids = core.assign_tasks_greedy(
        dataset, graph, payload, cache
    )
    limit_violations = core.validate_daily_limits(states)
    feasible = (len(unassigned_ids) == 0 and len(limit_violations) == 0)

    if verbose:
        print(f"  Назначено: {len(routes)} маршрутов")
        print(f"  Не назначено: {len(unassigned_ids)} задач")
        print(f"  Нарушений: {len(limit_violations)}")
        print(f"  Статус: {'✓ feasible' if feasible else '✗ infeasible'}")

    return SolverResult(
        method_label="Baseline Greedy",
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


# ═════════════════════════════════════════════════════════════════════════════
# Итоговое сравнение
# ═════════════════════════════════════════════════════════════════════════════

def compare_methods(results: list[SolverResult]) -> None:
    """Печатает сводную таблицу сравнения методов."""
    print(f"\n{'═'*72}")
    print("  СРАВНЕНИЕ МЕТОДОВ  (метрика: т·км = тонны × километры маршрута)")
    print(f"{'═'*72}")
    print(f"  {'Метод':<44} {'т·км':>9}  {'нар.':>5}  {'неназн.':>8}")
    print(f"  {'─'*66}")

    baseline_tr = next(
        (r.transport_work_ton_km for r in results if "Baseline" in r.method_label),
        None,
    )
    best_tr = min(
        (r.transport_work_ton_km for r in results
         if r.transport_work_ton_km is not None and r.feasible),
        default=None,
    )

    for r in results:
        tr_str = f"{r.transport_work_ton_km:.2f}" if r.transport_work_ton_km else "—"
        viol   = str(len(r.limit_violations)) if r.limit_violations else "—"
        unasn  = str(r.n_unassigned) if r.n_unassigned else "—"
        marker = " ◄" if (r.transport_work_ton_km == best_tr and r.feasible) else ""
        diff_str = ""
        if baseline_tr and r.transport_work_ton_km and baseline_tr > 0:
            diff = (baseline_tr - r.transport_work_ton_km) / baseline_tr * 100
            sign = "−" if diff >= 0 else "+"
            diff_str = f"  {sign}{abs(diff):.1f}%"
        print(f"  {r.method_label:<44} {tr_str:>9}  {viol:>5}  {unasn:>8}{diff_str}{marker}")

    print(f"{'═'*72}")
