from __future__ import annotations

from dataclasses import dataclass
import random
import time
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp

from .common import build_single_task_route, pair_cost, summarize_checks
from .distance_oracle import DistanceOracleWithFallback
from .problem import EnrichedProblem, task_agent_compatible
from .types import AgentUsage, EnrichedSolveResult


@dataclass(frozen=True)
class EnrichedMILPConfig:
    time_limit_sec: int = 60
    max_runtime_sec: float | None = None
    unassigned_penalty: float = 1e6
    max_pairs_per_task: int = 80
    random_jitter: float = 0.0
    random_seed: int = 42
    x_cost_weight: float = 1.0
    y_cost_weight: float = 1.0
    max_unassigned_tasks: int | None = None
    zone_min_coverage_ratio: float = 0.0
    dynamic_candidate_caps: bool = False
    dynamic_min_k: int = 40
    dynamic_max_k: int = 240
    dynamic_hard_threshold: int = 25
    prioritize_hard_tasks: bool = False


class EnrichedMILPSolver:
    def __init__(self, config: EnrichedMILPConfig | None = None) -> None:
        self.config = config or EnrichedMILPConfig()

    def solve(self, *, problem: EnrichedProblem, oracle: DistanceOracleWithFallback) -> EnrichedSolveResult:
        t0 = time.perf_counter()
        rng = random.Random(self.config.random_seed)
        tasks = problem.tasks
        agents = problem.agents

        if not tasks:
            return EnrichedSolveResult(
                algorithm="enriched_milp_v2",
                feasible=True,
                routes=[],
                unassigned_task_ids=[],
                agent_usage={a.agent_id: AgentUsage(agent_id=a.agent_id) for a in agents},
                runtime_sec=time.perf_counter() - t0,
                details={"checks": summarize_checks(unassigned_count=0, overflow_km=0, overflow_hours=0)},
            )

        pair_costs: dict[tuple[int, int], tuple[float, float, float]] = {}
        task_candidates: dict[int, list[int]] = {}
        task_candidate_counts: dict[int, int] = {}
        candidate_cutoff_hit = False
        built_task_count = 0

        for ti, task in enumerate(tasks):
            if self.config.max_runtime_sec is not None and (time.perf_counter() - t0) >= float(self.config.max_runtime_sec):
                candidate_cutoff_hit = True
                task_candidate_counts[ti] = 0
                task_candidates[ti] = []
                for tj in range(ti + 1, len(tasks)):
                    task_candidate_counts[tj] = 0
                    task_candidates[tj] = []
                break
            candidates: list[tuple[float, int, float, float]] = []
            for ai, agent in enumerate(agents):
                if not task_agent_compatible(task, agent):
                    continue
                pc = pair_cost(task, agent, oracle)
                if pc is None:
                    continue
                score = pc.total_km
                if self.config.random_jitter > 0:
                    score += self.config.random_jitter * rng.random()
                candidates.append((score, ai, pc.total_km, pc.total_hours))
            task_candidate_counts[ti] = len(candidates)
            candidates.sort(key=lambda x: x[0])
            if self.config.max_pairs_per_task > 0:
                k_cap = int(self.config.max_pairs_per_task)
                if self.config.dynamic_candidate_caps:
                    cc = len(candidates)
                    if cc <= self.config.dynamic_hard_threshold:
                        k_cap = min(cc, max(k_cap, int(self.config.dynamic_max_k)))
                    else:
                        k_cap = min(k_cap, int(self.config.dynamic_min_k))
                candidates = candidates[: max(1, k_cap)]
            if not candidates:
                task_candidates[ti] = []
                continue
            task_candidates[ti] = [ai for _, ai, _, _ in candidates]
            for _, ai, km, hours in candidates:
                pair_costs[(ti, ai)] = (km, hours, km)
            built_task_count += 1

        var_index: dict[tuple[str, int, int] | tuple[str, int], int] = {}
        c: list[float] = []
        lb: list[float] = []
        ub: list[float] = []
        integrality: list[int] = []

        for ti in range(len(tasks)):
            for ai in task_candidates.get(ti, []):
                var_index[("x", ti, ai)] = len(c)
                c.append(float(self.config.x_cost_weight) * pair_costs[(ti, ai)][2])
                lb.append(0.0)
                ub.append(1.0)
                integrality.append(1)

        for ti in range(len(tasks)):
            var_index[("y", ti)] = len(c)
            penalty = float(self.config.unassigned_penalty)
            if self.config.prioritize_hard_tasks:
                cc = max(1, int(task_candidate_counts.get(ti, 1)))
                penalty = penalty * (1.0 + 1.0 / float(cc))
            c.append(float(self.config.y_cost_weight) * penalty)
            lb.append(0.0)
            ub.append(1.0)
            integrality.append(1)

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        low: list[float] = []
        up: list[float] = []
        row = 0

        def add_row(coeffs: dict[int, float], l: float, u: float) -> None:
            nonlocal row
            for ci, cv in coeffs.items():
                rows.append(row)
                cols.append(ci)
                vals.append(cv)
            low.append(l)
            up.append(u)
            row += 1

        # each task assigned to one agent or unassigned
        for ti in range(len(tasks)):
            coeff: dict[int, float] = {}
            for ai in task_candidates.get(ti, []):
                coeff[var_index[("x", ti, ai)]] = 1.0
            coeff[var_index[("y", ti)]] = 1.0
            add_row(coeff, 1.0, 1.0)

        # agent daily km/h limits
        for ai, agent in enumerate(agents):
            km_coeff: dict[int, float] = {}
            h_coeff: dict[int, float] = {}
            for ti in range(len(tasks)):
                if (ti, ai) not in pair_costs:
                    continue
                xcol = var_index[("x", ti, ai)]
                km_coeff[xcol] = pair_costs[(ti, ai)][0]
                h_coeff[xcol] = pair_costs[(ti, ai)][1]
            add_row(km_coeff, -np.inf, float(agent.max_daily_km))
            add_row(h_coeff, -np.inf, float(agent.max_shift_hours))

        # object daily mass/volume limits
        tasks_by_object: dict[str, list[int]] = {}
        for ti, task in enumerate(tasks):
            tasks_by_object.setdefault(str(task.destination_node_id), []).append(ti)

        for oid, tis in tasks_by_object.items():
            cap_mass = float(problem.object_day_capacity_tons.get(oid, 0.0) or 0.0)
            if cap_mass > 0:
                coeff: dict[int, float] = {}
                for ti in tis:
                    m = float(tasks[ti].mass_tons)
                    for ai in task_candidates.get(ti, []):
                        xcol = var_index.get(("x", ti, ai))
                        if xcol is not None:
                            coeff[xcol] = coeff.get(xcol, 0.0) + m
                add_row(coeff, -np.inf, cap_mass)

            cap_vol = float(problem.object_day_capacity_volume_m3.get(oid, 0.0) or 0.0)
            if cap_vol > 0:
                coeff = {}
                for ti in tis:
                    v = float(tasks[ti].volume_raw_m3)
                    for ai in task_candidates.get(ti, []):
                        xcol = var_index.get(("x", ti, ai))
                        if xcol is not None:
                            coeff[xcol] = coeff.get(xcol, 0.0) + v
                add_row(coeff, -np.inf, cap_vol)

        # Optional global unassigned cap (used by 2-stage lexicographic wrappers)
        if self.config.max_unassigned_tasks is not None:
            coeff: dict[int, float] = {}
            for ti in range(len(tasks)):
                coeff[var_index[("y", ti)]] = 1.0
            add_row(coeff, -np.inf, float(self.config.max_unassigned_tasks))

        # Optional per-zone minimal coverage ratio
        if self.config.zone_min_coverage_ratio > 0:
            zone_tasks: dict[int | None, list[int]] = {}
            for ti, task in enumerate(tasks):
                zone_tasks.setdefault(task.source_zone_num, []).append(ti)
            for z, tis in zone_tasks.items():
                if z is None or not tis:
                    continue
                need = int(np.ceil(float(self.config.zone_min_coverage_ratio) * len(tis)))
                if need <= 0:
                    continue
                coeff: dict[int, float] = {}
                for ti in tis:
                    for ai in task_candidates.get(ti, []):
                        xcol = var_index.get(("x", ti, ai))
                        if xcol is not None:
                            coeff[xcol] = coeff.get(xcol, 0.0) + 1.0
                add_row(coeff, float(need), np.inf)

        A = sp.coo_array((vals, (rows, cols)), shape=(row, len(c)))
        bounds = Bounds(lb, ub)
        constraints = LinearConstraint(A, low, up)

        res = milp(
            c=np.asarray(c, dtype=float),
            integrality=np.asarray(integrality, dtype=int),
            bounds=bounds,
            constraints=constraints,
            options={"disp": False, "time_limit": int(self.config.time_limit_sec)},
        )

        if res is None or getattr(res, "x", None) is None:
            return EnrichedSolveResult(
                algorithm="enriched_milp_v2",
                feasible=False,
                routes=[],
                unassigned_task_ids=[t.task_id for t in tasks],
                agent_usage={a.agent_id: AgentUsage(agent_id=a.agent_id) for a in agents},
                runtime_sec=time.perf_counter() - t0,
                details={
                    "solver_error": f"MILP failed (status={getattr(res,'status',None)}): {getattr(res,'message','')}",
                    "checks": summarize_checks(unassigned_count=len(tasks), overflow_km=0, overflow_hours=0),
                },
            )

        x = np.asarray(res.x)
        assignment: dict[int, int] = {}
        unassigned: list[str] = []

        for ti, task in enumerate(tasks):
            assigned_ai = None
            for ai in task_candidates.get(ti, []):
                if x[var_index[("x", ti, ai)]] > 0.5:
                    assigned_ai = ai
                    break
            if assigned_ai is None:
                unassigned.append(task.task_id)
            else:
                assignment[ti] = assigned_ai

        routes = []
        agent_usage = {a.agent_id: AgentUsage(agent_id=a.agent_id) for a in agents}
        overflow_km = 0
        overflow_hours = 0
        route_counter = 1

        for ti, ai in assignment.items():
            task = tasks[ti]
            agent = agents[ai]
            pc = pair_cost(task, agent, oracle)
            if pc is None:
                unassigned.append(task.task_id)
                continue
            route = build_single_task_route(
                route_id=f"EMILP_ROUTE_{route_counter:06d}",
                agent=agent,
                task=task,
                cost=pc,
            )
            route_counter += 1
            routes.append(route)

            usage = agent_usage[agent.agent_id]
            usage.tasks.append(task.task_id)
            usage.total_km += pc.total_km
            usage.total_hours += pc.total_hours
            usage.loaded_km += pc.loaded_km

        for agent in agents:
            usage = agent_usage[agent.agent_id]
            if usage.total_km > agent.max_daily_km + 1e-9:
                overflow_km += 1
            if usage.total_hours > agent.max_shift_hours + 1e-9:
                overflow_hours += 1

        checks = summarize_checks(
            unassigned_count=len(unassigned),
            overflow_km=overflow_km,
            overflow_hours=overflow_hours,
        )

        return EnrichedSolveResult(
            algorithm="enriched_milp_v2",
            feasible=bool(checks["all_checks_ok"]),
            routes=routes,
            unassigned_task_ids=sorted(set(unassigned)),
            agent_usage=agent_usage,
            runtime_sec=time.perf_counter() - t0,
            details={
                "solver_status": int(getattr(res, "status", -1)),
                "solver_message": str(getattr(res, "message", "")),
                "variables": len(c),
                "constraints": row,
                "checks": checks,
                "task_mass_by_id": problem.task_mass_by_id,
                "milp_config": {
                    "time_limit_sec": int(self.config.time_limit_sec),
                    "max_runtime_sec": self.config.max_runtime_sec,
                    "max_pairs_per_task": int(self.config.max_pairs_per_task),
                    "x_cost_weight": float(self.config.x_cost_weight),
                    "y_cost_weight": float(self.config.y_cost_weight),
                    "max_unassigned_tasks": self.config.max_unassigned_tasks,
                    "zone_min_coverage_ratio": float(self.config.zone_min_coverage_ratio),
                    "dynamic_candidate_caps": bool(self.config.dynamic_candidate_caps),
                    "prioritize_hard_tasks": bool(self.config.prioritize_hard_tasks),
                    "candidate_cutoff_hit": bool(candidate_cutoff_hit),
                    "candidate_built_tasks": int(built_task_count),
                    "candidate_total_tasks": int(len(tasks)),
                },
            },
        )
