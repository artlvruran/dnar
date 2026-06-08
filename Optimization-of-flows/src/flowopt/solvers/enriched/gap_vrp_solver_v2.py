from __future__ import annotations

from dataclasses import dataclass
import random
import time

from .common import SERVICE_HOURS_BY_CONTAINER, build_batched_route, pair_cost, summarize_checks
from .distance_oracle import DistanceOracleWithFallback
from .problem import EnrichedProblem, task_agent_compatible
from .types import AgentUsage, EnrichedSolveResult


@dataclass(frozen=True)
class EnrichedGapVRPConfig:
    random_seed: int = 42
    top_k_agents: int = 20
    balance_penalty: float = 0.05
    max_runtime_sec: float | None = None


class EnrichedGapVRPSolver:
    def __init__(self, config: EnrichedGapVRPConfig | None = None) -> None:
        self.config = config or EnrichedGapVRPConfig()

    def solve(self, *, problem: EnrichedProblem, oracle: DistanceOracleWithFallback) -> EnrichedSolveResult:
        t0 = time.perf_counter()
        rng = random.Random(self.config.random_seed)

        agents = list(problem.agents)
        tasks = list(problem.tasks)

        usage = {a.agent_id: AgentUsage(agent_id=a.agent_id) for a in agents}
        agent_by_id = {a.agent_id: a for a in agents}
        unassigned: list[str] = []

        # Batch key: same source/destination/container -> one physical trip can pick multiple containers.
        batches: dict[tuple[str, str, str], list[EnrichedTask]] = {}
        for task in tasks:
            key = (task.source_node_id, task.destination_node_id, task.container_type)
            batches.setdefault(key, []).append(task)
        for key, group in batches.items():
            group.sort(key=lambda t: (t.mass_tons, t.volume_raw_m3), reverse=True)

        batch_order = sorted(
            batches.keys(),
            key=lambda k: (
                1 if any(t.source_center for t in batches[k]) else 0,
                sum(t.mass_tons for t in batches[k]),
                len(batches[k]),
            ),
            reverse=True,
        )

        routes = []
        route_counter = 1
        inf = float("inf")
        object_mass_used: dict[str, float] = {}
        object_vol_used: dict[str, float] = {}
        cutoff_hit = False

        for key_idx, key in enumerate(batch_order):
            group = batches[key]
            while group:
                if self.config.max_runtime_sec is not None and (time.perf_counter() - t0) >= float(
                    self.config.max_runtime_sec
                ):
                    cutoff_hit = True
                    unassigned.extend(t.task_id for t in group)
                    for rest_key in batch_order[key_idx + 1 :]:
                        unassigned.extend(t.task_id for t in batches.get(rest_key, []))
                    group = []
                    break
                best_candidate: tuple[int, float, str, list[EnrichedTask], float, float, float] | None = None
                rep = group[0]
                service_h_per_task = SERVICE_HOURS_BY_CONTAINER.get(rep.container_type, 0.25)

                for agent in agents:
                    compat = [t for t in group if task_agent_compatible(t, agent)]
                    if not compat:
                        continue

                    pc = pair_cost(rep, agent, oracle)
                    if pc is None:
                        continue

                    au = usage[agent.agent_id]
                    remaining_km = agent.max_daily_km - au.total_km
                    remaining_h = agent.max_shift_hours - au.total_hours
                    if pc.total_km > remaining_km + 1e-9:
                        continue

                    travel_hours = pc.total_km / max(agent.avg_speed_kmph, 1e-6)
                    max_by_h = int((remaining_h - travel_hours + 1e-9) // max(service_h_per_task, 1e-6))
                    if max_by_h <= 0:
                        continue

                    cap_mass = max(agent.capacity_tons, 0.0)
                    cap_vol = agent.max_raw_volume_m3 if agent.max_raw_volume_m3 > 0 else inf
                    cap_obj_mass = float(problem.object_day_capacity_tons.get(rep.destination_node_id, 0.0) or 0.0)
                    cap_obj_vol = float(problem.object_day_capacity_volume_m3.get(rep.destination_node_id, 0.0) or 0.0)
                    obj_mass_before = object_mass_used.get(rep.destination_node_id, 0.0)
                    obj_vol_before = object_vol_used.get(rep.destination_node_id, 0.0)
                    cur_mass = 0.0
                    cur_vol = 0.0
                    picked: list[EnrichedTask] = []
                    for task in compat:
                        if len(picked) >= max_by_h:
                            break
                        nm = cur_mass + task.mass_tons
                        nv = cur_vol + task.volume_raw_m3
                        if nm <= cap_mass + 1e-9 and nv <= cap_vol + 1e-9:
                            if cap_obj_mass > 0 and (obj_mass_before + nm) > cap_obj_mass + 1e-9:
                                continue
                            if cap_obj_vol > 0 and (obj_vol_before + nv) > cap_obj_vol + 1e-9:
                                continue
                            picked.append(task)
                            cur_mass = nm
                            cur_vol = nv

                    if not picked:
                        continue

                    trip_hours = travel_hours + service_h_per_task * len(picked)
                    if trip_hours > remaining_h + 1e-9:
                        continue

                    remaining_km_safe = max(remaining_km, 1e-6)
                    remaining_h_safe = max(remaining_h, 1e-6)
                    scarcity = (pc.total_km / remaining_km_safe) + (trip_hours / remaining_h_safe)
                    balance = self.config.balance_penalty * len(au.tasks)
                    score = (pc.total_km / max(len(picked), 1)) + scarcity + balance
                    candidate = (len(picked), score, agent.agent_id, picked, pc.total_km, pc.loaded_km, trip_hours)
                    if best_candidate is None or candidate[0] > best_candidate[0] or (
                        candidate[0] == best_candidate[0] and candidate[1] < best_candidate[1]
                    ):
                        best_candidate = candidate

                if best_candidate is None:
                    unassigned.extend(t.task_id for t in group)
                    break

                top_size = best_candidate[0]
                tie: list[tuple[int, float, str, list[EnrichedTask], float, float, float]] = []
                # collect near-best to keep stochastic behavior
                for agent in agents:
                    compat = [t for t in group if task_agent_compatible(t, agent)]
                    if not compat:
                        continue
                    rep = compat[0]
                    pc = pair_cost(rep, agent, oracle)
                    if pc is None:
                        continue
                    au = usage[agent.agent_id]
                    remaining_km = agent.max_daily_km - au.total_km
                    remaining_h = agent.max_shift_hours - au.total_hours
                    if pc.total_km > remaining_km + 1e-9:
                        continue
                    travel_hours = pc.total_km / max(agent.avg_speed_kmph, 1e-6)
                    max_by_h = int((remaining_h - travel_hours + 1e-9) // max(service_h_per_task, 1e-6))
                    if max_by_h <= 0:
                        continue
                    cap_mass = max(agent.capacity_tons, 0.0)
                    cap_vol = agent.max_raw_volume_m3 if agent.max_raw_volume_m3 > 0 else inf
                    cap_obj_mass = float(problem.object_day_capacity_tons.get(rep.destination_node_id, 0.0) or 0.0)
                    cap_obj_vol = float(problem.object_day_capacity_volume_m3.get(rep.destination_node_id, 0.0) or 0.0)
                    obj_mass_before = object_mass_used.get(rep.destination_node_id, 0.0)
                    obj_vol_before = object_vol_used.get(rep.destination_node_id, 0.0)
                    cur_mass = 0.0
                    cur_vol = 0.0
                    picked: list[EnrichedTask] = []
                    for task in compat:
                        if len(picked) >= max_by_h:
                            break
                        nm = cur_mass + task.mass_tons
                        nv = cur_vol + task.volume_raw_m3
                        if nm <= cap_mass + 1e-9 and nv <= cap_vol + 1e-9:
                            if cap_obj_mass > 0 and (obj_mass_before + nm) > cap_obj_mass + 1e-9:
                                continue
                            if cap_obj_vol > 0 and (obj_vol_before + nv) > cap_obj_vol + 1e-9:
                                continue
                            picked.append(task)
                            cur_mass = nm
                            cur_vol = nv
                    if not picked or len(picked) < top_size:
                        continue
                    trip_hours = travel_hours + service_h_per_task * len(picked)
                    if trip_hours > remaining_h + 1e-9:
                        continue
                    remaining_km_safe = max(remaining_km, 1e-6)
                    remaining_h_safe = max(remaining_h, 1e-6)
                    scarcity = (pc.total_km / remaining_km_safe) + (trip_hours / remaining_h_safe)
                    balance = self.config.balance_penalty * len(au.tasks)
                    score = (pc.total_km / max(len(picked), 1)) + scarcity + balance
                    tie.append((len(picked), score, agent.agent_id, picked, pc.total_km, pc.loaded_km, trip_hours))

                tie.sort(key=lambda x: x[1])
                shortlist = tie[: max(1, min(self.config.top_k_agents, len(tie)))] if tie else [best_candidate]
                chosen = shortlist[rng.randint(0, min(2, len(shortlist) - 1))]

                _, _, aid, picked, trip_km, loaded_km, trip_hours = chosen
                agent = agent_by_id[aid]
                route = build_batched_route(
                    route_id=f"EGAP_ROUTE_{route_counter:06d}",
                    agent=agent,
                    tasks=picked,
                    loaded_distance_km=loaded_km,
                    total_distance_km=trip_km,
                    total_hours=trip_hours,
                )
                route_counter += 1
                routes.append(route)

                au = usage[aid]
                au.tasks.extend(t.task_id for t in picked)
                au.total_km += trip_km
                au.total_hours += trip_hours
                au.loaded_km += loaded_km
                for task in picked:
                    object_mass_used[task.destination_node_id] = (
                        object_mass_used.get(task.destination_node_id, 0.0) + float(task.mass_tons)
                    )
                    object_vol_used[task.destination_node_id] = (
                        object_vol_used.get(task.destination_node_id, 0.0) + float(task.volume_raw_m3)
                    )

                picked_ids = {t.task_id for t in picked}
                group = [t for t in group if t.task_id not in picked_ids]
                batches[key] = group
            if cutoff_hit:
                break

        overflow_km = 0
        overflow_hours = 0
        for agent in agents:
            au = usage[agent.agent_id]
            if au.total_km > agent.max_daily_km + 1e-9:
                overflow_km += 1
            if au.total_hours > agent.max_shift_hours + 1e-9:
                overflow_hours += 1

        checks = summarize_checks(
            unassigned_count=len(unassigned),
            overflow_km=overflow_km,
            overflow_hours=overflow_hours,
        )

        return EnrichedSolveResult(
            algorithm="enriched_batched_greedy_v1",
            feasible=bool(checks["all_checks_ok"]),
            routes=routes,
            unassigned_task_ids=sorted(set(unassigned)),
            agent_usage=usage,
            runtime_sec=time.perf_counter() - t0,
            details={
                "checks": checks,
                "task_mass_by_id": problem.task_mass_by_id,
                "top_k_agents": self.config.top_k_agents,
                "balance_penalty": self.config.balance_penalty,
                "max_runtime_sec": self.config.max_runtime_sec,
                "global_cutoff_hit": bool(cutoff_hit),
                "solver_family": "batched_greedy_assignment",
                "legacy_alias": "enriched_gap_vrp_v2",
            },
        )
