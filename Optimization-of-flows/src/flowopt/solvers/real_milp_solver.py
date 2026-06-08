from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import time
from typing import Any, Callable

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp

from .. import core
from ..dataset import Route, RoutingDataset, Task
from ..gap_vrp_solver import SolverResult


@dataclass
class _PairCost:
    km: float
    hours: float
    cost: float


def _overflow_score(plan: core.AgentDayPlan | None, state: core.AgentState) -> tuple[float, float]:
    if plan is None:
        return float("inf"), float("inf")
    km_limit = core.MAX_DAILY_KM_BY_TYPE.get(state.vehicle_type, 130.0)
    h_limit = core.MAX_SHIFT_HOURS_BY_TYPE.get(state.vehicle_type, 10.0)
    km_over = max(0.0, float(plan.total_km) - float(km_limit))
    h_over = max(0.0, float(plan.total_hours) - float(h_limit))
    return km_over, h_over


def _repair_agent_bundle_to_feasible(
    *,
    state: core.AgentState,
    assigned_tasks: list[Task],
    graph: nx.DiGraph,
    cache: dict[tuple[str, str], tuple[list[str], float] | None],
    single_task_costs: dict[str, _PairCost],
) -> tuple[core.AgentDayPlan | None, list[Task]]:
    """
    Try to salvage a feasible subset by iteratively dropping one task.

    Returns:
      - feasible plan for remaining tasks (or None if nothing salvageable),
      - dropped tasks (to be marked unassigned).
    """
    kept = list(assigned_tasks)
    dropped: list[Task] = []

    while kept:
        plan = core.evaluate_agent_task_set(state, kept, graph, cache)
        if plan is not None and plan.feasible:
            return plan, dropped

        best_drop_idx: int | None = None
        best_trial_plan: core.AgentDayPlan | None = None
        best_trial_score: tuple[int, float, float, float] | None = None

        for idx in range(len(kept)):
            trial_tasks = kept[:idx] + kept[idx + 1 :]
            trial_plan = core.evaluate_agent_task_set(state, trial_tasks, graph, cache)
            km_over, h_over = _overflow_score(trial_plan, state)
            total_over = km_over + h_over
            # Feasible trials rank first (flag=0), then by smallest residual overflow.
            # Tie-break by lower travel km.
            trial_score = (
                0 if (trial_plan is not None and trial_plan.feasible) else 1,
                total_over,
                km_over,
                float(trial_plan.total_km) if trial_plan is not None else float("inf"),
            )
            if best_trial_score is None or trial_score < best_trial_score:
                best_trial_score = trial_score
                best_drop_idx = idx
                best_trial_plan = trial_plan

        if best_drop_idx is None:
            # Fallback: drop the hardest task by single-task km estimate.
            best_drop_idx = max(
                range(len(kept)),
                key=lambda idx: single_task_costs.get(kept[idx].task_id, _PairCost(float("inf"), float("inf"), 0.0)).km,
            )
            best_trial_plan = None

        dropped.append(kept.pop(best_drop_idx))
        if best_trial_plan is not None and best_trial_plan.feasible:
            return best_trial_plan, dropped

    return None, dropped


def _find_best_feasible_insertion(
    *,
    task: Task,
    task_by_id: dict[str, Task],
    dataset: RoutingDataset,
    states: dict[str, core.AgentState],
    agent_ids: list[str],
    graph: nx.DiGraph,
    cache: dict[tuple[str, str], tuple[list[str], float] | None],
) -> tuple[str, core.AgentDayPlan] | None:
    best: tuple[tuple[float, float, int, str], str, core.AgentDayPlan] | None = None
    for agent_id in agent_ids:
        state = states[agent_id]
        if not core.is_task_compatible_with_agent_state(dataset=dataset, state=state, task=task):
            continue

        current_ids = list(state.task_ids)
        if task.task_id in current_ids:
            continue
        current_tasks = [task_by_id[task_id] for task_id in current_ids if task_id in task_by_id]
        trial_plan = core.evaluate_agent_task_set(state, current_tasks + [task], graph, cache)
        if trial_plan is None or not trial_plan.feasible:
            continue

        base_km = float(state.total_km) if current_ids else 0.0
        delta_km = float(trial_plan.total_km) - base_km
        score = (delta_km, float(trial_plan.total_km), len(current_ids), agent_id)
        if best is None or score < best[0]:
            best = (score, agent_id, trial_plan)
    if best is None:
        return None
    _, agent_id, plan = best
    return agent_id, plan


def _is_task_agent_compatible(dataset: RoutingDataset, state: core.AgentState, task: Task) -> bool:
    return core.is_task_compatible_with_agent_state(dataset=dataset, state=state, task=task)


def _single_task_pair_cost(
    state: core.AgentState,
    task: Task,
    graph: nx.DiGraph,
    cache: dict[tuple[str, str], tuple[list[str], float] | None],
) -> _PairCost | None:
    plan = core.evaluate_agent_task_set(state, [task], graph, cache)
    if plan is None or not plan.feasible:
        return None
    # Objective focuses on route distance; task-level ton-km is reported later from built routes.
    return _PairCost(km=plan.total_km, hours=plan.total_hours, cost=plan.total_km)


def solve_real_milp(
    *,
    dataset: RoutingDataset,
    payload: dict[str, Any],
    graph: nx.DiGraph,
    cache: dict[tuple[str, str], tuple[list[str], float] | None],
    time_limit_sec: int = 60,
    unassigned_penalty: float = 1e5,
    show_progress: bool = False,
    progress_hook: Callable[[str], None] | None = None,
) -> SolverResult:
    def _emit(message: str) -> None:
        if progress_hook is not None:
            progress_hook(message)
        elif show_progress:
            print(f"[real_milp] {message}", flush=True)

    t0 = time.perf_counter()
    tasks = list(dataset.tasks)
    states = core.initialize_agent_states(dataset, payload)
    agent_ids = list(states.keys())
    _emit(f"prepare problem: tasks={len(tasks)}, agents={len(agent_ids)}")

    if not tasks:
        return SolverResult(
            method_label="Real MILP",
            routes=[],
            states=states,
            unassigned=[],
            transport_work_ton_km=None,
            feasible=True,
            limit_violations=[],
            n_assigned=0,
            n_unassigned=0,
            active_agents=0,
        )
    if not agent_ids:
        return SolverResult(
            method_label="Real MILP",
            routes=[],
            states=states,
            unassigned=[task.task_id for task in tasks],
            transport_work_ton_km=None,
            feasible=False,
            limit_violations=[],
            n_assigned=0,
            n_unassigned=len(tasks),
            active_agents=0,
        )

    compatible_pairs: dict[tuple[int, int], _PairCost] = {}
    task_to_agents: dict[int, list[int]] = defaultdict(list)
    t_pairs = time.perf_counter()
    task_log_every = max(1, len(tasks) // 10) if tasks else 1
    for t_idx, task in enumerate(tasks):
        for a_idx, agent_id in enumerate(agent_ids):
            state = states[agent_id]
            if not _is_task_agent_compatible(dataset, state, task):
                continue
            pair = _single_task_pair_cost(state, task, graph, cache)
            if pair is None:
                continue
            compatible_pairs[(t_idx, a_idx)] = pair
            task_to_agents[t_idx].append(a_idx)
        if (t_idx + 1) % task_log_every == 0 or t_idx + 1 == len(tasks):
            _emit(
                f"compatible pairs: {t_idx + 1}/{len(tasks)} tasks processed, "
                f"pairs={len(compatible_pairs)}"
            )
    _emit(f"pair matrix ready in {time.perf_counter() - t_pairs:.1f}s")

    var_idx: dict[tuple[str, int, int] | tuple[str, int], int] = {}
    c: list[float] = []
    lb: list[float] = []
    ub: list[float] = []
    integrality: list[int] = []

    # x[t,a] binary for compatible pairs only.
    for t_idx in range(len(tasks)):
        for a_idx in task_to_agents.get(t_idx, []):
            var_idx[("x", t_idx, a_idx)] = len(c)
            c.append(compatible_pairs[(t_idx, a_idx)].cost)
            lb.append(0.0)
            ub.append(1.0)
            integrality.append(1)

    # y[t] binary: task left unassigned.
    for t_idx in range(len(tasks)):
        var_idx[("y", t_idx)] = len(c)
        c.append(unassigned_penalty)
        lb.append(0.0)
        ub.append(1.0)
        integrality.append(1)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    bl: list[float] = []
    bu: list[float] = []
    row = 0

    def add_row(coeffs: dict[int, float], lower: float, upper: float) -> None:
        nonlocal row
        for col, val in coeffs.items():
            rows.append(row)
            cols.append(col)
            vals.append(val)
        bl.append(lower)
        bu.append(upper)
        row += 1

    # Each task either assigned once or explicitly unassigned.
    for t_idx in range(len(tasks)):
        coeff: dict[int, float] = {}
        for a_idx in task_to_agents.get(t_idx, []):
            coeff[var_idx[("x", t_idx, a_idx)]] = 1.0
        coeff[var_idx[("y", t_idx)]] = coeff.get(var_idx[("y", t_idx)], 0.0) + 1.0
        add_row(coeff, 1.0, 1.0)

    # Agent daily limits (km and hours), based on conservative single-task estimates.
    for a_idx, agent_id in enumerate(agent_ids):
        state = states[agent_id]
        km_limit = core.MAX_DAILY_KM_BY_TYPE.get(state.vehicle_type, 130.0)
        h_limit = core.MAX_SHIFT_HOURS_BY_TYPE.get(state.vehicle_type, 10.0)

        coeff_km: dict[int, float] = {}
        coeff_h: dict[int, float] = {}
        for t_idx in range(len(tasks)):
            pair = compatible_pairs.get((t_idx, a_idx))
            if pair is None:
                continue
            x_col = var_idx[("x", t_idx, a_idx)]
            coeff_km[x_col] = pair.km
            coeff_h[x_col] = pair.hours

        add_row(coeff_km, -np.inf, float(km_limit))
        add_row(coeff_h, -np.inf, float(h_limit))

    if not c:
        raise RuntimeError("Real MILP has no decision variables.")

    _emit(f"build MILP matrices: vars={len(c)}, constraints={row}")
    A = sp.coo_array((vals, (rows, cols)), shape=(row, len(c)))
    bounds = Bounds(lb, ub)
    constraints = LinearConstraint(A, bl, bu)

    _emit(f"solve MILP (time_limit={time_limit_sec}s)")
    t_solve = time.perf_counter()
    res = milp(
        c=np.asarray(c, dtype=float),
        integrality=np.asarray(integrality, dtype=int),
        bounds=bounds,
        constraints=constraints,
        options={"disp": False, "time_limit": int(time_limit_sec)},
    )
    status = getattr(res, "status", None)
    _emit(f"MILP solved in {time.perf_counter() - t_solve:.1f}s (status={status})")

    if res is None or getattr(res, "x", None) is None:
        status = getattr(res, "status", None)
        message = getattr(res, "message", "MILP solver returned no solution vector")
        raise RuntimeError(f"Real MILP failed (status={status}): {message}")

    x = np.asarray(res.x)

    allocation: dict[str, list[Task]] = {aid: [] for aid in agent_ids}
    unassigned_ids: list[str] = []

    for t_idx, task in enumerate(tasks):
        assigned_agent: str | None = None
        for a_idx in task_to_agents.get(t_idx, []):
            col = var_idx[("x", t_idx, a_idx)]
            if x[col] > 0.5:
                assigned_agent = agent_ids[a_idx]
                break
        if assigned_agent is None:
            unassigned_ids.append(task.task_id)
            continue
        allocation[assigned_agent].append(task)

    _emit(
        "decode solution: "
        f"assigned_tasks={len(tasks) - len(unassigned_ids)}, unassigned={len(unassigned_ids)}"
    )
    still_unassigned: set[str] = set(unassigned_ids)
    dropped_task_ids: set[str] = set()
    final_plan_by_agent: dict[str, core.AgentDayPlan] = {}
    task_index_by_id = {task.task_id: idx for idx, task in enumerate(tasks)}
    task_by_id = {task.task_id: task for task in tasks}

    for a_idx, agent_id in enumerate(agent_ids):
        assigned_tasks = allocation[agent_id]
        state = states[agent_id]
        if not assigned_tasks:
            continue

        single_task_costs: dict[str, _PairCost] = {}
        for task in assigned_tasks:
            t_idx = task_index_by_id.get(task.task_id)
            if t_idx is None:
                continue
            pair = compatible_pairs.get((t_idx, a_idx))
            if pair is None:
                continue
            single_task_costs[task.task_id] = pair

        plan, dropped = _repair_agent_bundle_to_feasible(
            state=state,
            assigned_tasks=assigned_tasks,
            graph=graph,
            cache=cache,
            single_task_costs=single_task_costs,
        )

        if dropped:
            still_unassigned.update(task.task_id for task in dropped)
            dropped_task_ids.update(task.task_id for task in dropped)
            _emit(
                f"repair bundle: agent={agent_id}, dropped={len(dropped)}, "
                f"kept={len(assigned_tasks) - len(dropped)}"
            )

        if plan is None or not plan.feasible:
            # Could not salvage any feasible subset for this agent.
            still_unassigned.update(task.task_id for task in assigned_tasks)
            continue

        core.apply_plan_to_state(state, plan)
        final_plan_by_agent[agent_id] = plan
        for task in plan.ordered_tasks:
            still_unassigned.discard(task.task_id)

    # Second chance: try to reinsert tasks dropped by local bundle repair
    # into any currently feasible agent plan (often idle agents can absorb them).
    reinserts = 0
    for task_id in sorted(dropped_task_ids):
        if task_id not in still_unassigned:
            continue
        task = task_by_id.get(task_id)
        if task is None:
            continue
        inserted = _find_best_feasible_insertion(
            task=task,
            task_by_id=task_by_id,
            dataset=dataset,
            states=states,
            agent_ids=agent_ids,
            graph=graph,
            cache=cache,
        )
        if inserted is None:
            continue
        target_agent_id, target_plan = inserted
        core.apply_plan_to_state(states[target_agent_id], target_plan)
        final_plan_by_agent[target_agent_id] = target_plan
        still_unassigned.discard(task_id)
        reinserts += 1
    if reinserts > 0:
        _emit(f"second chance insertions: {reinserts}")

    routes: list[Route] = []
    route_counter = 1
    for agent_id in agent_ids:
        state = states[agent_id]
        state.route_ids = []
        plan = final_plan_by_agent.get(agent_id)
        if plan is None:
            continue
        for task in plan.ordered_tasks:
            route_id = f"RMILP_ROUTE_{route_counter:04d}"
            route_counter += 1
            state.route_ids.append(route_id)
            routes.append(
                Route(
                    route_id=route_id,
                    agent_id=agent_id,
                    path=plan.service_paths[task.task_id],
                    task_ids=(task.task_id,),
                )
            )

    unassigned = sorted(still_unassigned)
    limit_violations = core.validate_daily_limits(states)
    feasible = (len(unassigned) == 0 and len(limit_violations) == 0)
    _emit(
        f"finalize in {time.perf_counter() - t0:.1f}s: "
        f"routes={len(routes)}, unassigned={len(unassigned)}, "
        f"violations={len(limit_violations)}, feasible={feasible}"
    )

    return SolverResult(
        method_label="Real MILP",
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
