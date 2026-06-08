from __future__ import annotations

from dataclasses import dataclass
import time
from collections import defaultdict
import random

from .models import AssignmentSolution, TaskPickupLog, TripPlan, VolumeAgent, VolumeTask
from .solver_base import VolumeSolver


@dataclass(frozen=True)
class GreedyBatchConfig:
    max_runtime_sec: float = 120.0
    top_k_agents: int = 30
    top_k_destinations: int = 3
    max_tasks_in_trip: int = 32
    min_remaining_hours: float = 0.05
    verbose: bool = True
    log_every_sec: float = 5.0
    trip_log_every: int = 100
    score_mode: str = "vol_per_km"  # vol_per_km | min_km | vol_only
    stochastic_mode: bool = False
    random_seed: int = 42
    exploration_noise: float = 0.0
    deterministic_fill: bool = True
    fill_time_budget_sec: float = 20.0


class GreedyBatchVolumeSolver(VolumeSolver):
    def __init__(self, config: GreedyBatchConfig | None = None) -> None:
        self.config = config or GreedyBatchConfig()

    def solve(self, dataset: "VolumeDataset") -> AssignmentSolution:
        t0 = time.perf_counter()
        tasks = list(dataset.tasks)
        total_tasks = len(tasks)
        agents = [a for a in dataset.agents if a.is_active and a.depot_node_id is not None]

        task_by_id = {t.task_id: t for t in tasks}
        unassigned: set[str] = {t.task_id for t in tasks}

        used_km = {a.agent_id: 0.0 for a in agents}
        used_h = {a.agent_id: 0.0 for a in agents}
        object_used = {k: 0.0 for k in dataset.object_volume_caps}

        compat_agents_by_task: dict[str, list[str]] = {}
        for t in tasks:
            cands = [a.agent_id for a in agents if dataset.agent_can_take_task(t, a)]
            compat_agents_by_task[t.task_id] = cands
        scarcity_by_task: dict[str, int] = {
            tid: max(1, len(cands))
            for tid, cands in compat_agents_by_task.items()
        }

        agent_by_id = {a.agent_id: a for a in agents}
        trips: list[TripPlan] = []
        logs: list[str] = []
        rng = random.Random(self.config.random_seed)

        def _emit(msg: str) -> None:
            logs.append(msg)
            if self.config.verbose:
                print(msg, flush=True)

        _emit(
            "[volume_core_greedy_batch_v1] start: "
            f"tasks={total_tasks}, agents={len(agents)}, "
            f"max_runtime_sec={self.config.max_runtime_sec}, "
            f"top_k_agents={self.config.top_k_agents}, "
            f"top_k_destinations={self.config.top_k_destinations}, "
            f"max_tasks_in_trip={self.config.max_tasks_in_trip}, "
            f"score_mode={self.config.score_mode}, stochastic_mode={self.config.stochastic_mode}"
        )

        trip_idx = 0
        progress = True
        pass_idx = 0
        last_progress_log = t0
        last_assigned_log = 0
        total_trip_km = 0.0
        total_trip_h = 0.0
        total_trip_payload = 0.0
        total_trip_tasks = 0

        def _log_snapshot(*, now: float, pass_trips: int, pass_assigned: int, pass_agents_with_cands: int) -> None:
            assigned = total_tasks - len(unassigned)
            remaining = len(unassigned)
            elapsed = now - t0
            rate_global = assigned / max(elapsed, 1e-9)
            delta_assigned = assigned - last_assigned_log
            rate_window = delta_assigned / max(now - last_progress_log, 1e-9)
            eta = (remaining / max(rate_global, 1e-9)) if remaining > 0 else 0.0
            cov = 100.0 * assigned / max(total_tasks, 1)
            active_agents = len({tr.agent_id for tr in trips})
            avg_tasks_per_trip = total_trip_tasks / max(len(trips), 1)
            avg_km_per_trip = total_trip_km / max(len(trips), 1)
            avg_h_per_trip = total_trip_h / max(len(trips), 1)
            avg_payload_per_trip = total_trip_payload / max(len(trips), 1)
            avg_tasks_per_active_agent = assigned / max(active_agents, 1)
            avg_trips_per_active_agent = len(trips) / max(active_agents, 1)

            used_agent_ids = {tr.agent_id for tr in trips}
            rem_h_vals = []
            rem_km_vals = []
            h_util_vals = []
            km_util_vals = []
            for a in agents:
                if a.agent_id not in used_agent_ids:
                    continue
                rem_h = a.max_hours - used_h[a.agent_id]
                rem_km = a.max_daily_km - used_km[a.agent_id]
                rem_h_vals.append(rem_h)
                rem_km_vals.append(rem_km)
                if a.max_hours > 1e-9:
                    h_util_vals.append(min(1.0, max(0.0, used_h[a.agent_id] / a.max_hours)))
                if a.max_daily_km > 1e-9:
                    km_util_vals.append(min(1.0, max(0.0, used_km[a.agent_id] / a.max_daily_km)))
            if rem_h_vals:
                h_le_05 = sum(1 for x in rem_h_vals if x <= 0.5)
                h_le_10 = sum(1 for x in rem_h_vals if x <= 1.0)
                km_le_10 = sum(1 for x in rem_km_vals if x <= 10.0)
                km_le_25 = sum(1 for x in rem_km_vals if x <= 25.0)
                h_util_ge_90 = sum(1 for u in h_util_vals if u >= 0.9)
                km_util_ge_90 = sum(1 for u in km_util_vals if u >= 0.9)
                sat_line = (
                    f"used_agents={len(rem_h_vals)}, rem_h<=0.5h:{h_le_05}, rem_h<=1h:{h_le_10}, "
                    f"rem_km<=10:{km_le_10}, rem_km<=25:{km_le_25}, "
                    f"h_util>=90%:{h_util_ge_90}, km_util>=90%:{km_util_ge_90}"
                )
            else:
                sat_line = "used_agents=0"

            _emit(
                "[volume_core_greedy_batch_v1] progress: "
                f"pass={pass_idx}, assigned={assigned}/{total_tasks} ({cov:.2f}%), "
                f"unassigned={remaining}, trips={len(trips)}, active_agents={active_agents}, "
                f"pass_trips={pass_trips}, pass_assigned={pass_assigned}, pass_agents_with_cands={pass_agents_with_cands}, "
                f"rate_global={rate_global:.2f} task/s, rate_window={rate_window:.2f} task/s, eta={eta:.1f}s, elapsed={elapsed:.1f}s, "
                f"avg_tasks/trip={avg_tasks_per_trip:.2f}, avg_tasks/active_agent={avg_tasks_per_active_agent:.2f}, "
                f"avg_trips/active_agent={avg_trips_per_active_agent:.2f}, avg_km/trip={avg_km_per_trip:.2f}, "
                f"avg_h/trip={avg_h_per_trip:.2f}, avg_payload/trip={avg_payload_per_trip:.2f}, "
                f"{sat_line}"
            )

        while progress and unassigned and (time.perf_counter() - t0) < self.config.max_runtime_sec:
            progress = False
            pass_idx += 1
            pass_trips = 0
            pass_assigned = 0
            pass_agents_with_cands = 0
            pass_agents = list(agents)
            if self.config.stochastic_mode:
                rng.shuffle(pass_agents)
            for agent in pass_agents:
                if (time.perf_counter() - t0) >= self.config.max_runtime_sec:
                    break

                rem_km = agent.max_daily_km - used_km[agent.agent_id]
                rem_h = agent.max_hours - used_h[agent.agent_id]
                if rem_km <= 1e-9 or rem_h <= self.config.min_remaining_hours:
                    continue

                # candidate tasks for this agent
                cand_ids = [tid for tid in unassigned if agent.agent_id in compat_agents_by_task.get(tid, [])]
                if not cand_ids:
                    continue
                pass_agents_with_cands += 1
                cand_tasks = [task_by_id[tid] for tid in cand_ids]

                # destination priority by remaining object capacity and local task volume
                dest_score: dict[str, float] = {}
                for t in cand_tasks:
                    eff = dataset.effective_task_volume(t, agent)
                    cap = dataset.object_volume_caps.get(t.destination_node_id, 0.0)
                    rem_obj = cap - object_used.get(t.destination_node_id, 0.0)
                    if cap > 0 and rem_obj <= 1e-9:
                        continue
                    if cap > 0 and eff > rem_obj + 1e-9:
                        continue
                    sc = float(scarcity_by_task.get(t.task_id, 1))
                    scarcity_bonus = 1.0 / max(sc, 1.0)
                    dest_score[t.destination_node_id] = dest_score.get(t.destination_node_id, 0.0) + scarcity_bonus
                if not dest_score:
                    continue

                ranked_dests = sorted(dest_score.keys(), key=lambda d: dest_score[d], reverse=True)
                if self.config.stochastic_mode and len(ranked_dests) > 1:
                    k = max(1, self.config.top_k_destinations)
                    shortlist = ranked_dests[: min(len(ranked_dests), max(2 * k, 3))]
                    rng.shuffle(shortlist)
                    dests = shortlist[:k]
                else:
                    dests = ranked_dests[: max(1, self.config.top_k_destinations)]

                best_trip: tuple[float, TripPlan, list[str], float, float] | None = None
                for dst in dests:
                    trip = self._build_trip_for_destination(
                        dataset=dataset,
                        agent=agent,
                        destination=dst,
                        candidate_tasks=[t for t in cand_tasks if t.destination_node_id == dst],
                        rem_km=rem_km,
                        rem_h=rem_h,
                        object_used=object_used,
                        used_km=used_km[agent.agent_id],
                        used_h=used_h[agent.agent_id],
                        scarcity_by_task=scarcity_by_task,
                        trip_idx=trip_idx + 1,
                        rng=rng,
                    )
                    if trip is None:
                        continue
                    plan, picked, add_km, add_h = trip
                    vol = plan.payload_effective_volume_m3
                    if self.config.score_mode == "min_km":
                        score = -plan.total_km
                    elif self.config.score_mode == "vol_only":
                        score = vol
                    else:
                        score = vol / max(plan.total_km, 1e-6)
                    if self.config.stochastic_mode and self.config.exploration_noise > 0.0:
                        score += rng.uniform(-self.config.exploration_noise, self.config.exploration_noise)
                    if best_trip is None or score > best_trip[0]:
                        best_trip = (score, plan, picked, add_km, add_h)

                if best_trip is None:
                    continue

                _score, plan, picked_ids, add_km, add_h = best_trip
                trips.append(plan)
                trip_idx += 1
                pass_trips += 1
                total_trip_km += add_km
                total_trip_h += add_h
                total_trip_payload += plan.payload_effective_volume_m3
                total_trip_tasks += len(picked_ids)
                used_km[agent.agent_id] += add_km
                used_h[agent.agent_id] += add_h
                for tid in picked_ids:
                    if tid in unassigned:
                        unassigned.remove(tid)
                        pass_assigned += 1
                    t = task_by_id[tid]
                    object_used[t.destination_node_id] = object_used.get(t.destination_node_id, 0.0) + dataset.effective_task_volume(
                        t, agent
                    )
                if self.config.trip_log_every > 0 and (trip_idx % self.config.trip_log_every == 0):
                    assigned = total_tasks - len(unassigned)
                    cov = 100.0 * assigned / max(total_tasks, 1)
                    elapsed = time.perf_counter() - t0
                    _emit(
                        "[volume_core_greedy_batch_v1] trip-progress: "
                        f"pass={pass_idx}, trips={trip_idx}, assigned={assigned}/{total_tasks} ({cov:.2f}%), "
                        f"elapsed={elapsed:.1f}s"
                    )
                progress = True

                now = time.perf_counter()
                if self.config.log_every_sec > 0 and (now - last_progress_log) >= self.config.log_every_sec:
                    _log_snapshot(
                        now=now,
                        pass_trips=pass_trips,
                        pass_assigned=pass_assigned,
                        pass_agents_with_cands=pass_agents_with_cands,
                    )
                    last_assigned_log = total_tasks - len(unassigned)
                    last_progress_log = now

            now = time.perf_counter()
            if self.config.log_every_sec > 0 and (now - last_progress_log) >= self.config.log_every_sec:
                _log_snapshot(
                    now=now,
                    pass_trips=pass_trips,
                    pass_assigned=pass_assigned,
                    pass_agents_with_cands=pass_agents_with_cands,
                )
                last_assigned_log = total_tasks - len(unassigned)
                last_progress_log = now

        # Fallback repair: assign remaining tasks by single-task trips while residual capacities allow.
        if self.config.deterministic_fill and unassigned and (time.perf_counter() - t0) < self.config.max_runtime_sec:
            _emit(
                "[volume_core_greedy_batch_v1] repair-single-task: start "
                f"(unassigned={len(unassigned)})"
            )
            repair_round = 0
            fill_started = time.perf_counter()
            while unassigned and (time.perf_counter() - t0) < self.config.max_runtime_sec:
                if (time.perf_counter() - fill_started) >= max(0.0, self.config.fill_time_budget_sec):
                    _emit("[volume_core_greedy_batch_v1] repair-single-task: stop by fill_time_budget")
                    break
                repair_round += 1
                round_assigned = 0
                # Scarce tasks first.
                cand_ids = sorted(
                    list(unassigned),
                    key=lambda tid: (
                        scarcity_by_task.get(tid, 999999),
                        task_by_id[tid].source_node_id,
                        task_by_id[tid].destination_node_id,
                    ),
                )
                for tid in cand_ids:
                    if (time.perf_counter() - t0) >= self.config.max_runtime_sec:
                        break
                    if tid not in unassigned:
                        continue
                    t = task_by_id[tid]
                    best = None
                    for aid in compat_agents_by_task.get(tid, []):
                        a = agent_by_id.get(aid)
                        if a is None:
                            continue
                        rem_km = a.max_daily_km - used_km[a.agent_id]
                        rem_h = a.max_hours - used_h[a.agent_id]
                        if rem_km <= 1e-9 or rem_h <= self.config.min_remaining_hours:
                            continue
                        eff = dataset.effective_task_volume(t, a)
                        if eff > a.max_raw_volume_m3 + 1e-9:
                            continue
                        cap = dataset.object_volume_caps.get(t.destination_node_id, 0.0)
                        rem_obj = cap - object_used.get(t.destination_node_id, 0.0)
                        if cap > 0 and eff > rem_obj + 1e-9:
                            continue
                        dep = str(a.depot_node_id)
                        d1 = dataset.dist.dist(dep, t.source_node_id)
                        d2 = dataset.dist.dist(t.source_node_id, t.destination_node_id)
                        d3 = dataset.dist.dist(t.destination_node_id, dep)
                        if d1 == float("inf") or d2 == float("inf") or d3 == float("inf"):
                            continue
                        km = d1 + d2 + d3
                        h = km / max(a.avg_speed_kmph, 1e-6) + dataset.route_service_hours(t)
                        if km > rem_km + 1e-9 or h > rem_h + 1e-9:
                            continue
                        # Prefer preserving scarce resources (least residual pressure).
                        pressure = max(h / max(rem_h, 1e-9), km / max(rem_km, 1e-9))
                        score = (pressure, h, km)
                        if best is None or score < best[0]:
                            best = (score, a, km, h, eff, d1, d2, d3)
                    if best is None:
                        continue
                    _score, a, km, h, eff, d1, d2, d3 = best
                    trip_idx += 1
                    trip = TripPlan(
                        trip_id=f"TRIP_{trip_idx:07d}",
                        agent_id=a.agent_id,
                        depot_node_id=str(a.depot_node_id),
                        destination_object_id=t.destination_node_id,
                        ordered_task_ids=(t.task_id,),
                        visit_nodes=(str(a.depot_node_id), t.source_node_id, t.destination_node_id, str(a.depot_node_id)),
                        leg_distances_km=(float(d1), float(d2), float(d3)),
                        total_km=float(km),
                        total_hours=float(h),
                        payload_effective_volume_m3=float(eff),
                        task_pickups=(
                            TaskPickupLog(
                                task_id=t.task_id,
                                source_node_id=t.source_node_id,
                                effective_volume_m3=float(eff),
                                carried_distance_to_object_km=float(d2),
                            ),
                        ),
                        path_nodes_full=(),
                    )
                    trips.append(trip)
                    total_trip_km += km
                    total_trip_h += h
                    total_trip_payload += eff
                    total_trip_tasks += 1
                    used_km[a.agent_id] += km
                    used_h[a.agent_id] += h
                    object_used[t.destination_node_id] = object_used.get(t.destination_node_id, 0.0) + eff
                    unassigned.remove(tid)
                    round_assigned += 1
                _emit(
                    "[volume_core_greedy_batch_v1] repair-single-task: "
                    f"round={repair_round}, assigned={round_assigned}, remaining={len(unassigned)}"
                )
                if round_assigned == 0:
                    break

        if unassigned:
            reason = "timeout" if (time.perf_counter() - t0) >= self.config.max_runtime_sec else "no_progress"
            _emit(f"[volume_core_greedy_batch_v1] unfinished: reason={reason}, unassigned={len(unassigned)}")
        else:
            _emit("[volume_core_greedy_batch_v1] all tasks assigned")

        elapsed = float(time.perf_counter() - t0)
        assigned = total_tasks - len(unassigned)
        cov = 100.0 * assigned / max(total_tasks, 1)
        _emit(
            "[volume_core_greedy_batch_v1] done: "
            f"assigned={assigned}/{total_tasks} ({cov:.2f}%), trips={len(trips)}, "
            f"runtime_sec={elapsed:.2f}"
        )

        return AssignmentSolution(
            algorithm="volume_core_greedy_batch_v1",
            dataset_path=str(dataset.dataset_path),
            trips=tuple(trips),
            unassigned_task_ids=tuple(sorted(unassigned)),
            runtime_sec=elapsed,
            solver_logs=tuple(logs),
        )

    def _build_trip_for_destination(
        self,
        *,
        dataset: "VolumeDataset",
        agent: VolumeAgent,
        destination: str,
        candidate_tasks: list[VolumeTask],
        rem_km: float,
        rem_h: float,
        object_used: dict[str, float],
        used_km: float,
        used_h: float,
        scarcity_by_task: dict[str, int],
        trip_idx: int,
        rng: random.Random,
    ) -> tuple[TripPlan, list[str], float, float] | None:
        if not candidate_tasks:
            return None

        cap_obj = dataset.object_volume_caps.get(destination, 0.0)
        rem_obj = cap_obj - object_used.get(destination, 0.0) if cap_obj > 0 else float("inf")
        if rem_obj <= 1e-9:
            return None

        depot = str(agent.depot_node_id)
        selected: list[VolumeTask] = []
        selected_ids: list[str] = []
        total_eff_vol = 0.0
        path_km = 0.0
        cur = depot
        visited_sources: list[str] = []
        # service counted per source visit, not per individual task
        service_by_source: dict[str, float] = {}

        pool_by_source: dict[str, list[VolumeTask]] = defaultdict(list)
        for t in candidate_tasks:
            pool_by_source[t.source_node_id].append(t)
        for src in pool_by_source:
            pool_by_source[src].sort(
                key=lambda t: (
                    scarcity_by_task.get(t.task_id, 999999),
                    dataset.effective_task_volume(t, agent),
                )
            )

        source_pool = set(pool_by_source.keys())

        while source_pool and len(selected) < self.config.max_tasks_in_trip:
            best_src = None
            best_score = None
            for src in source_pool:
                src_tasks = pool_by_source.get(src, [])
                if not src_tasks:
                    continue
                d_leg = dataset.dist.dist(cur, src)
                if d_leg == float("inf"):
                    continue
                # source score: prefer close + scarce tasks
                scarcity_sum = sum(1.0 / max(float(scarcity_by_task.get(t.task_id, 1)), 1.0) for t in src_tasks[:8])
                src_score = scarcity_sum / max(d_leg, 1e-3)
                if self.config.score_mode == "min_km":
                    src_score = 1.0 / max(d_leg, 1e-3)
                elif self.config.score_mode == "vol_only":
                    src_score = float(len(src_tasks))
                if self.config.stochastic_mode and self.config.exploration_noise > 0.0:
                    src_score += rng.uniform(-self.config.exploration_noise, self.config.exploration_noise)
                if best_score is None or src_score > best_score:
                    best_score = src_score
                    best_src = src

            if best_src is None:
                break

            d_leg = dataset.dist.dist(cur, best_src)
            if d_leg == float("inf"):
                source_pool.discard(best_src)
                continue

            src_tasks = pool_by_source.get(best_src, [])
            src_sel: list[VolumeTask] = []
            src_eff = 0.0
            src_service = 0.0
            for t in src_tasks:
                if len(selected) + len(src_sel) >= self.config.max_tasks_in_trip:
                    break
                eff = dataset.effective_task_volume(t, agent)
                if total_eff_vol + src_eff + eff > agent.max_raw_volume_m3 + 1e-9:
                    continue
                if total_eff_vol + src_eff + eff > rem_obj + 1e-9:
                    continue
                # service at source is paid once, take max among picked tasks there
                trial_src_service = max(src_service, dataset.route_service_hours(t))
                trial_path_km = path_km + d_leg
                to_obj = dataset.dist.dist(best_src, destination)
                back = dataset.dist.dist(destination, depot)
                if to_obj == float("inf") or back == float("inf"):
                    continue
                trial_total_km = trial_path_km + to_obj + back
                trial_service_h = sum(service_by_source.values()) + trial_src_service
                trial_total_h = trial_total_km / max(agent.avg_speed_kmph, 1e-6) + trial_service_h
                if trial_total_km > rem_km + 1e-9 or trial_total_h > rem_h + 1e-9:
                    continue
                src_sel.append(t)
                src_eff += eff
                src_service = trial_src_service

            if not src_sel:
                source_pool.discard(best_src)
                continue

            for t in src_sel:
                selected.append(t)
                selected_ids.append(t.task_id)
            total_eff_vol += src_eff
            path_km += d_leg
            cur = best_src
            visited_sources.append(best_src)
            service_by_source[best_src] = max(service_by_source.get(best_src, 0.0), src_service)
            pool_by_source[best_src] = [x for x in src_tasks if x.task_id not in {t.task_id for t in src_sel}]
            if not pool_by_source[best_src]:
                source_pool.discard(best_src)

        if not selected:
            return None

        to_obj = dataset.dist.dist(cur, destination)
        back = dataset.dist.dist(destination, depot)
        total_km = path_km + to_obj + back
        svc_h = sum(service_by_source.values())
        total_h = total_km / max(agent.avg_speed_kmph, 1e-6) + svc_h

        visit_nodes = [depot] + visited_sources + [destination, depot]
        leg_ds: list[float] = []
        for i in range(len(visit_nodes) - 1):
            leg_ds.append(dataset.dist.dist(visit_nodes[i], visit_nodes[i + 1]))

        task_logs: list[TaskPickupLog] = []
        # carried distance by source position in the route
        source_index = {src: i for i, src in enumerate(visited_sources)}
        carry_from_source: dict[str, float] = {}
        for src in visited_sources:
            i = source_index[src]
            carry = 0.0
            for j in range(i, len(visited_sources) - 1):
                carry += dataset.dist.dist(visited_sources[j], visited_sources[j + 1])
            carry += dataset.dist.dist(visited_sources[-1], destination)
            carry_from_source[src] = float(carry)
        for t in selected:
            task_logs.append(
                TaskPickupLog(
                    task_id=t.task_id,
                    source_node_id=t.source_node_id,
                    effective_volume_m3=dataset.effective_task_volume(t, agent),
                    carried_distance_to_object_km=float(carry_from_source.get(t.source_node_id, 0.0)),
                )
            )

        # path nodes for visualization/logging
        full_nodes: list[str] = []
        for i in range(len(visit_nodes) - 1):
            p = list(dataset.dist.path(visit_nodes[i], visit_nodes[i + 1]))
            if not p:
                p = [visit_nodes[i], visit_nodes[i + 1]]
            if i == 0:
                full_nodes.extend(p)
            else:
                full_nodes.extend(p[1:])

        plan = TripPlan(
            trip_id=f"TRIP_{trip_idx:07d}",
            agent_id=agent.agent_id,
            depot_node_id=depot,
            destination_object_id=destination,
            ordered_task_ids=tuple(selected_ids),
            visit_nodes=tuple(visit_nodes),
            leg_distances_km=tuple(float(x) for x in leg_ds),
            total_km=float(total_km),
            total_hours=float(total_h),
            payload_effective_volume_m3=float(total_eff_vol),
            task_pickups=tuple(task_logs),
            path_nodes_full=tuple(full_nodes),
        )
        return plan, selected_ids, total_km, total_h
