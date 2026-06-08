from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Iterable

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp

from .common import SERVICE_HOURS_BY_CONTAINER, build_batched_route, pair_cost, summarize_checks
from .distance_oracle import DistanceOracleWithFallback
from .problem import EnrichedProblem, task_agent_compatible
from .types import AgentUsage, EnrichedAgent, EnrichedSolveResult, EnrichedTask


@dataclass(frozen=True)
class EnrichedMILPDecompConfig:
    objective: str = "tasks"  # tasks | volume
    unassigned_penalty: float = 1e5
    time_limit_sec_per_zone: int = 45
    max_pairs_per_bundle: int = 120
    bundle_mass_quantile: float = 0.35
    bundle_vol_quantile: float = 0.35
    bundle_fill_factor: float = 0.90
    bundle_max_tasks: int = 8
    max_runtime_sec: float | None = None
    verbose: bool = False


@dataclass(frozen=True)
class _Bundle:
    bundle_id: str
    agg_task: EnrichedTask
    member_tasks: tuple[EnrichedTask, ...]
    utility_tasks: float
    utility_volume: float
    utility_mass: float


def _quantile(values: Iterable[float], q: float, default: float) -> float:
    arr = np.asarray([float(x) for x in values if float(x) > 0], dtype=float)
    if arr.size == 0:
        return float(default)
    return float(np.quantile(arr, q))


def _adjust_agents(agents: list[EnrichedAgent], usage: dict[str, AgentUsage]) -> list[EnrichedAgent]:
    out: list[EnrichedAgent] = []
    for a in agents:
        u = usage.get(a.agent_id)
        used_km = float(u.total_km) if u is not None else 0.0
        used_h = float(u.total_hours) if u is not None else 0.0
        rem_km = max(0.0, a.max_daily_km - used_km)
        rem_h = max(0.0, a.max_shift_hours - used_h)
        if rem_km <= 1e-9 or rem_h <= 1e-9:
            continue
        out.append(
            EnrichedAgent(
                agent_id=a.agent_id,
                vehicle_type=a.vehicle_type,
                capacity_tons=a.capacity_tons,
                max_raw_volume_m3=a.max_raw_volume_m3,
                is_compact=a.is_compact,
                is_available=a.is_available,
                depot_node_id=a.depot_node_id,
                cap_container_types=a.cap_container_types,
                max_daily_km=rem_km,
                max_shift_hours=rem_h,
                avg_speed_kmph=a.avg_speed_kmph,
                zone_num=a.zone_num,
            )
        )
    return out


def _bundle_tasks(
    *,
    tasks: list[EnrichedTask],
    agents: list[EnrichedAgent],
    cfg: EnrichedMILPDecompConfig,
    prefix: str,
) -> list[_Bundle]:
    if not tasks:
        return []

    mass_cap = _quantile((a.capacity_tons for a in agents), cfg.bundle_mass_quantile, default=1.0)
    vol_cap = _quantile((a.max_raw_volume_m3 for a in agents), cfg.bundle_vol_quantile, default=2.0)
    mass_cap = max(0.01, mass_cap * cfg.bundle_fill_factor)
    vol_cap = max(0.01, vol_cap * cfg.bundle_fill_factor)

    by_key: dict[tuple[str, str, str, bool, int | None, tuple[str, ...]], list[EnrichedTask]] = {}
    for t in tasks:
        key = (
            t.source_node_id,
            t.destination_node_id,
            t.container_type,
            t.source_center,
            t.source_zone_num,
            tuple(sorted(t.compatible_vehicle_types)),
        )
        by_key.setdefault(key, []).append(t)

    bundles: list[_Bundle] = []
    counter = 1
    for key, group in by_key.items():
        group = sorted(group, key=lambda x: (x.mass_tons, x.volume_raw_m3), reverse=True)
        bins: list[list[EnrichedTask]] = []
        bin_mass: list[float] = []
        bin_vol: list[float] = []

        for task in group:
            placed = False
            for i in range(len(bins)):
                if len(bins[i]) >= cfg.bundle_max_tasks:
                    continue
                nm = bin_mass[i] + task.mass_tons
                nv = bin_vol[i] + task.volume_raw_m3
                if nm <= mass_cap + 1e-9 and nv <= vol_cap + 1e-9:
                    bins[i].append(task)
                    bin_mass[i] = nm
                    bin_vol[i] = nv
                    placed = True
                    break
            if not placed:
                bins.append([task])
                bin_mass.append(task.mass_tons)
                bin_vol.append(task.volume_raw_m3)

        for tasks_in_bin in bins:
            rep = tasks_in_bin[0]
            agg = EnrichedTask(
                task_id=f"{prefix}_BUNDLE_{counter:07d}",
                source_node_id=rep.source_node_id,
                destination_node_id=rep.destination_node_id,
                container_type=rep.container_type,
                mass_tons=float(sum(t.mass_tons for t in tasks_in_bin)),
                volume_raw_m3=float(sum(t.volume_raw_m3 for t in tasks_in_bin)),
                compatible_vehicle_types=rep.compatible_vehicle_types,
                source_center=rep.source_center,
                source_zone_num=rep.source_zone_num,
            )
            bundles.append(
                _Bundle(
                    bundle_id=agg.task_id,
                    agg_task=agg,
                    member_tasks=tuple(tasks_in_bin),
                    utility_tasks=float(len(tasks_in_bin)),
                    utility_volume=float(sum(t.volume_raw_m3 for t in tasks_in_bin)),
                    utility_mass=float(sum(t.mass_tons for t in tasks_in_bin)),
                )
            )
            counter += 1
    return bundles


def _solve_zone_milp(
    *,
    bundles: list[_Bundle],
    agents: list[EnrichedAgent],
    oracle: DistanceOracleWithFallback,
    cfg: EnrichedMILPDecompConfig,
) -> tuple[list[tuple[_Bundle, EnrichedAgent, float, float, float]], set[str], dict[str, str]]:
    """
    Returns:
      - chosen assignments (bundle, agent, trip_km, loaded_km, trip_hours)
      - unassigned bundle ids
      - errors by bundle id
    """
    if not bundles:
        return [], set(), {}
    if not agents:
        return [], {b.bundle_id for b in bundles}, {}

    idx_by_bundle = {b.bundle_id: i for i, b in enumerate(bundles)}

    pair_metrics: dict[tuple[int, int], tuple[float, float, float]] = {}
    candidates: dict[int, list[int]] = {}
    bundle_errors: dict[str, str] = {}

    for bi, bundle in enumerate(bundles):
        task = bundle.agg_task
        svc_per_task = SERVICE_HOURS_BY_CONTAINER.get(task.container_type, 0.25)
        rows: list[tuple[float, int, float, float, float]] = []
        for ai, agent in enumerate(agents):
            if not task_agent_compatible(task, agent):
                continue
            pc = pair_cost(task, agent, oracle)
            if pc is None:
                continue
            travel_h = pc.total_km / max(agent.avg_speed_kmph, 1e-6)
            trip_h = travel_h + svc_per_task * len(bundle.member_tasks)
            if trip_h > agent.max_shift_hours + 1e-9:
                continue
            if pc.total_km > agent.max_daily_km + 1e-9:
                continue
            rows.append((pc.total_km, ai, pc.total_km, pc.loaded_km, trip_h))
        rows.sort(key=lambda x: x[0])
        if cfg.max_pairs_per_bundle > 0:
            rows = rows[: cfg.max_pairs_per_bundle]
        if not rows:
            candidates[bi] = []
            bundle_errors[bundle.bundle_id] = "No feasible agent for bundle under km/h limits"
            continue
        candidates[bi] = [ai for _, ai, _, _, _ in rows]
        for _, ai, tk, lk, th in rows:
            pair_metrics[(bi, ai)] = (tk, lk, th)

    var_index: dict[tuple[str, int, int] | tuple[str, int], int] = {}
    c: list[float] = []
    lb: list[float] = []
    ub: list[float] = []
    integrality: list[int] = []

    for bi in range(len(bundles)):
        for ai in candidates.get(bi, []):
            var_index[("x", bi, ai)] = len(c)
            c.append(pair_metrics[(bi, ai)][0])  # prefer shorter trips
            lb.append(0.0)
            ub.append(1.0)
            integrality.append(1)

    for bi, b in enumerate(bundles):
        var_index[("y", bi)] = len(c)
        if cfg.objective == "volume":
            util = max(1e-6, b.utility_volume)
        else:
            util = max(1e-6, b.utility_tasks)
        c.append(float(cfg.unassigned_penalty) * util)
        lb.append(0.0)
        ub.append(1.0)
        integrality.append(1)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    low: list[float] = []
    up: list[float] = []
    row = 0

    def add_row(coeffs: dict[int, float], lo: float, hi: float) -> None:
        nonlocal row
        for ci, cv in coeffs.items():
            rows.append(row)
            cols.append(ci)
            vals.append(cv)
        low.append(lo)
        up.append(hi)
        row += 1

    for bi in range(len(bundles)):
        coeff: dict[int, float] = {}
        for ai in candidates.get(bi, []):
            coeff[var_index[("x", bi, ai)]] = 1.0
        coeff[var_index[("y", bi)]] = 1.0
        add_row(coeff, 1.0, 1.0)

    for ai, agent in enumerate(agents):
        km_coeff: dict[int, float] = {}
        h_coeff: dict[int, float] = {}
        for bi in range(len(bundles)):
            if (bi, ai) not in pair_metrics:
                continue
            xcol = var_index[("x", bi, ai)]
            km_coeff[xcol] = pair_metrics[(bi, ai)][0]
            h_coeff[xcol] = pair_metrics[(bi, ai)][2]
        add_row(km_coeff, -np.inf, float(agent.max_daily_km))
        add_row(h_coeff, -np.inf, float(agent.max_shift_hours))

    if not c:
        return [], {b.bundle_id for b in bundles}, bundle_errors

    A = sp.coo_array((vals, (rows, cols)), shape=(row, len(c)))
    bounds = Bounds(lb, ub)
    constraints = LinearConstraint(A, low, up)
    res = milp(
        c=np.asarray(c, dtype=float),
        integrality=np.asarray(integrality, dtype=int),
        bounds=bounds,
        constraints=constraints,
        options={"disp": False, "time_limit": int(cfg.time_limit_sec_per_zone)},
    )

    if res is None or getattr(res, "x", None) is None:
        return [], {b.bundle_id for b in bundles}, bundle_errors

    x = np.asarray(res.x, dtype=float)
    chosen: list[tuple[_Bundle, EnrichedAgent, float, float, float]] = []
    unassigned_ids: set[str] = set()

    for bi, bundle in enumerate(bundles):
        chosen_ai = None
        for ai in candidates.get(bi, []):
            if x[var_index[("x", bi, ai)]] > 0.5:
                chosen_ai = ai
                break
        if chosen_ai is None:
            unassigned_ids.add(bundle.bundle_id)
            continue
        tk, lk, th = pair_metrics[(bi, chosen_ai)]
        chosen.append((bundle, agents[chosen_ai], tk, lk, th))
    return chosen, unassigned_ids, bundle_errors


class EnrichedMILPDecompSolver:
    def __init__(self, config: EnrichedMILPDecompConfig | None = None) -> None:
        self.config = config or EnrichedMILPDecompConfig()

    def solve(self, *, problem: EnrichedProblem, oracle: DistanceOracleWithFallback) -> EnrichedSolveResult:
        t0 = time.perf_counter()
        cfg = self.config

        all_agents = list(problem.agents)
        all_tasks = list(problem.tasks)
        usage = {a.agent_id: AgentUsage(agent_id=a.agent_id) for a in all_agents}
        agent_by_id = {a.agent_id: a for a in all_agents}
        task_by_id = {t.task_id: t for t in all_tasks}

        tasks_by_zone: dict[int | None, list[EnrichedTask]] = {}
        for t in all_tasks:
            tasks_by_zone.setdefault(t.source_zone_num, []).append(t)

        agents_by_zone: dict[int | None, list[EnrichedAgent]] = {}
        for a in all_agents:
            agents_by_zone.setdefault(a.zone_num, []).append(a)

        zone_keys = sorted([z for z in tasks_by_zone.keys() if z is not None])
        if None in tasks_by_zone:
            zone_keys.append(None)

        routes = []
        route_counter = 1
        assigned_task_ids: set[str] = set()
        zone_logs: list[dict[str, object]] = []
        cutoff_hit = False

        for z in zone_keys:
            if cfg.max_runtime_sec is not None and (time.perf_counter() - t0) >= float(cfg.max_runtime_sec):
                cutoff_hit = True
                if cfg.verbose:
                    print(f"[MILP-DECOMP] global timeout reached: {cfg.max_runtime_sec}s")
                break
            zone_tasks = tasks_by_zone.get(z, [])
            if not zone_tasks:
                continue

            if z is None:
                zone_agents = _adjust_agents(all_agents, usage)
            else:
                fixed_agents = agents_by_zone.get(z, [])
                none_agents = agents_by_zone.get(None, [])
                zone_agents = _adjust_agents(fixed_agents + none_agents, usage)

            bundles = _bundle_tasks(
                tasks=zone_tasks,
                agents=zone_agents if zone_agents else all_agents,
                cfg=cfg,
                prefix=f"Z{z if z is not None else 'N'}",
            )

            chosen, unassigned_bundle_ids, _bundle_errors = _solve_zone_milp(
                bundles=bundles,
                agents=zone_agents,
                oracle=oracle,
                cfg=cfg,
            )

            bundle_by_id = {b.bundle_id: b for b in bundles}
            zone_assigned = 0
            zone_unassigned = 0
            zone_volume = 0.0

            for b, agent_adj, tk, lk, th in chosen:
                member_tasks = [task_by_id[t.task_id] for t in b.member_tasks if t.task_id in task_by_id]
                if not member_tasks:
                    continue
                route = build_batched_route(
                    route_id=f"EMD_ROUTE_{route_counter:07d}",
                    agent=agent_by_id[agent_adj.agent_id],
                    tasks=member_tasks,
                    loaded_distance_km=lk,
                    total_distance_km=tk,
                    total_hours=th,
                )
                route_counter += 1
                routes.append(route)
                zone_assigned += len(member_tasks)
                zone_volume += float(sum(t.volume_raw_m3 for t in member_tasks))
                assigned_task_ids.update(t.task_id for t in member_tasks)

                u = usage[agent_adj.agent_id]
                u.tasks.extend(t.task_id for t in member_tasks)
                u.total_km += tk
                u.total_hours += th
                u.loaded_km += lk

            for bid in unassigned_bundle_ids:
                b = bundle_by_id.get(bid)
                if b is None:
                    continue
                zone_unassigned += len(b.member_tasks)

            zone_logs.append(
                {
                    "zone": z,
                    "tasks_in_zone": len(zone_tasks),
                    "bundles": len(bundles),
                    "assigned_tasks": zone_assigned,
                    "unassigned_tasks": zone_unassigned,
                    "assigned_volume_raw_m3": round(zone_volume, 3),
                }
            )
            if cfg.verbose:
                print(
                    f"[MILP-DECOMP] zone={z} tasks={len(zone_tasks)} bundles={len(bundles)} "
                    f"assigned={zone_assigned} unassigned={zone_unassigned}"
                )

        unassigned = sorted(set(task_by_id.keys()) - assigned_task_ids)

        overflow_km = 0
        overflow_hours = 0
        for a in all_agents:
            u = usage[a.agent_id]
            if u.total_km > a.max_daily_km + 1e-9:
                overflow_km += 1
            if u.total_hours > a.max_shift_hours + 1e-9:
                overflow_hours += 1

        checks = summarize_checks(
            unassigned_count=len(unassigned),
            overflow_km=overflow_km,
            overflow_hours=overflow_hours,
        )

        assigned_volume = float(sum(task_by_id[tid].volume_raw_m3 for tid in assigned_task_ids if tid in task_by_id))
        total_volume = float(sum(t.volume_raw_m3 for t in all_tasks))
        volume_coverage_pct = 100.0 * assigned_volume / max(total_volume, 1e-9)

        return EnrichedSolveResult(
            algorithm="enriched_milp_decomp_v1",
            feasible=bool(checks["all_checks_ok"]),
            routes=routes,
            unassigned_task_ids=unassigned,
            agent_usage=usage,
            runtime_sec=time.perf_counter() - t0,
            details={
                "checks": checks,
                "task_mass_by_id": problem.task_mass_by_id,
                "objective": cfg.objective,
                "zone_logs": zone_logs,
                "assigned_volume_raw_m3": round(assigned_volume, 3),
                "total_volume_raw_m3": round(total_volume, 3),
                "volume_coverage_pct": round(volume_coverage_pct, 3),
                "config": {
                    "unassigned_penalty": cfg.unassigned_penalty,
                    "time_limit_sec_per_zone": cfg.time_limit_sec_per_zone,
                    "max_runtime_sec": cfg.max_runtime_sec,
                    "max_pairs_per_bundle": cfg.max_pairs_per_bundle,
                    "bundle_mass_quantile": cfg.bundle_mass_quantile,
                    "bundle_vol_quantile": cfg.bundle_vol_quantile,
                    "bundle_fill_factor": cfg.bundle_fill_factor,
                    "bundle_max_tasks": cfg.bundle_max_tasks,
                },
                "global_cutoff_hit": bool(cutoff_hit),
            },
        )
