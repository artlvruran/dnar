from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from flowopt import core
from flowopt.backend.io import load_dataset
from flowopt.solvers.enriched.common import SERVICE_HOURS_BY_CONTAINER, build_batched_route, pair_cost
from flowopt.solvers.enriched.distance_oracle import DistanceOracleWithFallback, PrecomputedDistanceOracle
from flowopt.solvers.enriched.evaluator import finalize_enriched_result
from flowopt.solvers.enriched.problem import EnrichedProblem, build_enriched_problem, task_agent_compatible
from flowopt.solvers.enriched.types import AgentUsage, EnrichedSolveResult, EnrichedTask
from flowopt.solvers.enriched.runner_v2 import solve_enriched_batched_greedy


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = (
    ROOT
    / "demo/data/object_mass_feasible_fullfleet/container_full_split_all_agents/sweeps_task_agent_5pct"
    / "dataset_real_spb_clean_full_split_by_containers_all_agents_with_distances_t031_a100.json"
)
OUT_DIR = ROOT / "experiments/local/exp11_source_policy_compare"


@dataclass(frozen=True)
class SimpleRoute:
    task_ids: list[str]
    total_km: float
    loaded_km: float
    total_hours: float


def _build_oracle(problem: EnrichedProblem) -> DistanceOracleWithFallback:
    dataset, _payload = load_dataset(problem.dataset_path)
    nx_graph = core.build_nx_graph(dataset)
    precomputed = PrecomputedDistanceOracle.from_dataset_payload(
        dataset_path=Path(problem.dataset_path),
        payload=problem.payload,
    )
    return DistanceOracleWithFallback(nx_graph=nx_graph, precomputed=precomputed)


def _policy_signature(task: EnrichedTask) -> tuple[int | None, str, bool]:
    return (task.source_zone_num, task.container_type, task.source_center)


def _trip_locked(agent: Any, tasks: list[EnrichedTask], obj_mass_left: float, obj_vol_left: float) -> SimpleRoute | None:
    if not tasks:
        return None
    seed = tasks[0]
    travel = pair_cost(seed, agent, ORACLE)
    if travel is None:
        return None
    service_h = SERVICE_HOURS_BY_CONTAINER.get(seed.container_type, 0.25)
    rem_mass = min(agent.capacity_tons, obj_mass_left)
    rem_vol = min(agent.max_raw_volume_m3 if agent.max_raw_volume_m3 > 0 else float("inf"), obj_vol_left)
    rem_hours = agent.max_shift_hours - USAGE[agent.agent_id].total_hours
    rem_km = agent.max_daily_km - USAGE[agent.agent_id].total_km
    if travel.total_km > rem_km + 1e-9:
        return None
    picked: list[str] = []
    cur_m = 0.0
    cur_v = 0.0
    for t in tasks:
        nh = travel.total_km / max(agent.avg_speed_kmph, 1e-6) + service_h * (len(picked) + 1)
        if nh > rem_hours + 1e-9:
            break
        nm = cur_m + t.mass_tons
        nv = cur_v + t.volume_raw_m3
        if nm <= rem_mass + 1e-9 and nv <= rem_vol + 1e-9:
            picked.append(t.task_id)
            cur_m = nm
            cur_v = nv
    if not picked:
        return None
    total_hours = travel.total_km / max(agent.avg_speed_kmph, 1e-6) + service_h * len(picked)
    return SimpleRoute(task_ids=picked, total_km=travel.total_km, loaded_km=travel.loaded_km, total_hours=total_hours)


def _trip_multi_source_nn(
    agent: Any,
    tasks: list[EnrichedTask],
    obj_mass_left: float,
    obj_vol_left: float,
) -> SimpleRoute | None:
    if not tasks or agent.depot_node_id is None:
        return None
    rem_mass = min(agent.capacity_tons, obj_mass_left)
    rem_vol = min(agent.max_raw_volume_m3 if agent.max_raw_volume_m3 > 0 else float("inf"), obj_vol_left)
    rem_hours = agent.max_shift_hours - USAGE[agent.agent_id].total_hours
    rem_km = agent.max_daily_km - USAGE[agent.agent_id].total_km
    if rem_mass <= 0 or rem_vol <= 0 or rem_hours <= 0 or rem_km <= 0:
        return None

    destination = tasks[0].destination_node_id
    container = tasks[0].container_type
    service_h = SERVICE_HOURS_BY_CONTAINER.get(container, 0.25)
    remaining = tasks[:]
    picked: list[EnrichedTask] = []
    cur_node = str(agent.depot_node_id)
    cur_mass = 0.0
    cur_vol = 0.0
    path_km = 0.0

    while remaining:
        best_i = -1
        best_d = float("inf")
        for i, t in enumerate(remaining):
            d = ORACLE.dist(cur_node, t.source_node_id)
            if d < best_d:
                best_d = d
                best_i = i
        if best_i < 0 or best_d == float("inf"):
            break
        t = remaining[best_i]
        nm = cur_mass + t.mass_tons
        nv = cur_vol + t.volume_raw_m3
        if nm > rem_mass + 1e-9 or nv > rem_vol + 1e-9:
            remaining.pop(best_i)
            continue
        trial_path = path_km + best_d
        to_dest = ORACLE.dist(t.source_node_id, destination)
        to_depot = ORACLE.dist(destination, str(agent.depot_node_id))
        if to_dest == float("inf") or to_depot == float("inf"):
            remaining.pop(best_i)
            continue
        trial_total_km = trial_path + to_dest + to_depot
        trial_hours = trial_total_km / max(agent.avg_speed_kmph, 1e-6) + service_h * (len(picked) + 1)
        if trial_total_km > rem_km + 1e-9 or trial_hours > rem_hours + 1e-9:
            break
        picked.append(t)
        cur_mass = nm
        cur_vol = nv
        path_km = trial_path
        cur_node = t.source_node_id
        remaining.pop(best_i)

    if not picked:
        return None
    last = picked[-1]
    to_dest = ORACLE.dist(last.source_node_id, destination)
    to_depot = ORACLE.dist(destination, str(agent.depot_node_id))
    total_km = path_km + to_dest + to_depot
    loaded_km = to_dest
    total_hours = total_km / max(agent.avg_speed_kmph, 1e-6) + service_h * len(picked)
    return SimpleRoute(
        task_ids=[t.task_id for t in picked],
        total_km=float(total_km),
        loaded_km=float(loaded_km),
        total_hours=float(total_hours),
    )


def _run_policy(problem: EnrichedProblem, policy: str, max_runtime_sec: float, seed: int) -> EnrichedSolveResult:
    global ORACLE, USAGE
    ORACLE = _build_oracle(problem)
    rng = random.Random(seed)
    tasks = list(problem.tasks)
    task_by_id = {t.task_id: t for t in tasks}
    agents = [a for a in problem.agents if a.is_available]
    USAGE = {a.agent_id: AgentUsage(agent_id=a.agent_id) for a in agents}
    agent_by_id = {a.agent_id: a for a in agents}
    obj_mass_left = {k: float(v) for k, v in problem.object_day_capacity_tons.items()}
    obj_vol_left = {k: float(v) for k, v in problem.object_day_capacity_volume_m3.items()}

    sig_agents: dict[tuple[int | None, str, bool], list[str]] = defaultdict(list)
    for a in agents:
        for c in a.cap_container_types:
            sig_agents[(a.zone_num, c, False)].append(a.agent_id)
            if a.is_compact:
                sig_agents[(a.zone_num, c, True)].append(a.agent_id)

    # group tasks by policy key
    groups: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for t in tasks:
        if policy in ("locked_current", "locked_mass_first"):
            k = (t.source_node_id, t.destination_node_id, t.container_type, t.source_zone_num, t.source_center)
        elif policy == "multi_source_nn":
            k = (t.destination_node_id, t.container_type, t.source_zone_num, t.source_center)
        else:
            raise ValueError(policy)
        groups[k].append(t.task_id)

    for g in groups.values():
        g.sort(key=lambda tid: task_by_id[tid].mass_tons, reverse=True)

    if policy == "locked_current":
        order = sorted(
            groups.keys(),
            key=lambda k: (
                1 if bool(k[-1]) else 0,
                sum(task_by_id[tid].mass_tons for tid in groups[k]),
                len(groups[k]),
            ),
            reverse=True,
        )
    elif policy == "locked_mass_first":
        order = sorted(groups.keys(), key=lambda k: sum(task_by_id[tid].mass_tons for tid in groups[k]), reverse=True)
    else:
        order = sorted(groups.keys(), key=lambda k: len(groups[k]), reverse=True)

    route_counter = 0
    routes = []
    unassigned: set[str] = set()
    t0 = time.perf_counter()

    pbar = tqdm(order, desc=f"exp11:{policy}", unit="group")
    for gk in pbar:
        if (time.perf_counter() - t0) >= max_runtime_sec:
            for remk in order[order.index(gk) :]:
                unassigned.update(groups.get(remk, []))
            break
        tids = groups[gk]
        while tids:
            if (time.perf_counter() - t0) >= max_runtime_sec:
                unassigned.update(tids)
                break
            seed_task = task_by_id[tids[0]]
            sig = _policy_signature(seed_task)
            pool = list(sig_agents.get(sig, []))
            if sig[0] is not None:
                pool += sig_agents.get((None, sig[1], sig[2]), [])
            if not pool:
                unassigned.update(tids)
                tids.clear()
                break
            # score limited agent subset
            pool = list(dict.fromkeys(pool))
            pool.sort(key=lambda aid: len(USAGE[aid].tasks))
            pool = pool[:40]

            task_objs = [task_by_id[tid] for tid in tids]
            best = None
            for aid in pool:
                agent = agent_by_id[aid]
                if policy == "multi_source_nn":
                    cand = _trip_multi_source_nn(
                        agent=agent,
                        tasks=task_objs,
                        obj_mass_left=float(obj_mass_left.get(seed_task.destination_node_id, 0.0)),
                        obj_vol_left=float(obj_vol_left.get(seed_task.destination_node_id, 0.0)),
                    )
                else:
                    cand = _trip_locked(
                        agent=agent,
                        tasks=task_objs,
                        obj_mass_left=float(obj_mass_left.get(seed_task.destination_node_id, 0.0)),
                        obj_vol_left=float(obj_vol_left.get(seed_task.destination_node_id, 0.0)),
                    )
                if cand is None:
                    continue
                score = (len(cand.task_ids), -cand.total_km, -len(USAGE[aid].tasks))
                if best is None or score > best[0]:
                    best = (score, aid, cand)

            if best is None:
                # cannot assign this group further
                unassigned.update(tids)
                tids.clear()
                break

            _, aid, cand = best
            picked_ids = set(cand.task_ids)
            picked_tasks = [task_by_id[tid] for tid in cand.task_ids]
            mass = sum(t.mass_tons for t in picked_tasks)
            vol = sum(t.volume_raw_m3 for t in picked_tasks)
            dest = picked_tasks[0].destination_node_id
            obj_mass_left[dest] = obj_mass_left.get(dest, 0.0) - mass
            obj_vol_left[dest] = obj_vol_left.get(dest, 0.0) - vol

            route_counter += 1
            route = build_batched_route(
                route_id=f"EXP11_{policy}_{route_counter:07d}",
                agent=agent_by_id[aid],
                tasks=picked_tasks,
                loaded_distance_km=cand.loaded_km,
                total_distance_km=cand.total_km,
                total_hours=cand.total_hours,
            )
            routes.append(route)
            u = USAGE[aid]
            u.tasks.extend(cand.task_ids)
            u.total_km += cand.total_km
            u.total_hours += cand.total_hours
            u.loaded_km += cand.loaded_km

            tids = [tid for tid in tids if tid not in picked_ids]
            groups[gk] = tids
            pbar.set_postfix_str(f"routes={len(routes)} unassigned~{len(unassigned)}")

    # all remaining tasks in non-empty groups => unassigned
    for tids in groups.values():
        unassigned.update(tids)

    res = EnrichedSolveResult(
        algorithm=f"exp11_{policy}",
        feasible=False,
        routes=routes,
        unassigned_task_ids=sorted(unassigned),
        agent_usage=USAGE,
        runtime_sec=time.perf_counter() - t0,
        details={"policy": policy},
    )
    res = finalize_enriched_result(problem=problem, result=res, oracle=ORACLE)
    return res


def main() -> None:
    p = argparse.ArgumentParser(description="EXP11: source-selection policy comparison in one experiment")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--max-runtime-sec", type=float, default=90.0)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()

    dataset_path = args.dataset.resolve()
    problem = build_enriched_problem(dataset_path)

    rows: list[dict[str, Any]] = []

    # 1) current production greedy baseline
    base = solve_enriched_batched_greedy(
        dataset_path=dataset_path,
        random_seed=42,
        top_k_agents=30,
        balance_penalty=0.02,
        max_runtime_sec=args.max_runtime_sec,
    )
    bd = base.as_dict()
    rows.append(
        {
            "policy": "baseline_batched_greedy_v1",
            "algorithm": bd["algorithm"],
            "feasible": bd["feasible"],
            "all_checks_ok": ((bd.get("details") or {}).get("checks") or {}).get("all_checks_ok", False),
            "assigned_tasks": bd["assigned_routes"],
            "unassigned_tasks": bd["unassigned_tasks"],
            "coverage_pct": round(100.0 * bd["assigned_routes"] / max((bd["assigned_routes"] + bd["unassigned_tasks"]), 1), 3),
            "active_agents": bd["active_agents"],
            "total_km": bd["total_km"],
            "total_hours": bd["total_hours"],
            "runtime_sec": bd["runtime_sec"],
        }
    )

    for i, pol in enumerate(["locked_current", "locked_mass_first", "multi_source_nn"], start=1):
        res = _run_policy(problem, pol, max_runtime_sec=args.max_runtime_sec, seed=100 + i)
        d = res.as_dict()
        rows.append(
            {
                "policy": pol,
                "algorithm": d["algorithm"],
                "feasible": d["feasible"],
                "all_checks_ok": ((d.get("details") or {}).get("checks") or {}).get("all_checks_ok", False),
                "assigned_tasks": d["assigned_routes"],
                "unassigned_tasks": d["unassigned_tasks"],
                "coverage_pct": round(100.0 * d["assigned_routes"] / max((d["assigned_routes"] + d["unassigned_tasks"]), 1), 3),
                "active_agents": d["active_agents"],
                "total_km": d["total_km"],
                "total_hours": d["total_hours"],
                "runtime_sec": d["runtime_sec"],
            }
        )

    df = pd.DataFrame(rows).sort_values(["coverage_pct", "runtime_sec"], ascending=[False, True]).reset_index(drop=True)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"exp11_source_policy_compare_{stamp}.csv"
    json_path = out_dir / f"exp11_source_policy_compare_{stamp}.json"
    latest_csv = out_dir / "exp11_source_policy_compare_latest.csv"

    df.to_csv(csv_path, index=False)
    df.to_csv(latest_csv, index=False)
    json_path.write_text(json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
