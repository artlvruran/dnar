"""MILP solver based on real_milp_ksenya with nearest-depot decomposition.

Tasks are partitioned into 4 disjoint sets by the principle "nearest depot"
(Euclidean distance on source coordinates). Each depot — together with its
own agents — is solved as an independent MILP sub-problem.

This is **mathematically NOT equivalent** to the monolithic ksenya solver:
a task that could optimally be served by a vehicle from another depot is
forbidden from doing so. In return the per-depot sub-problems are dramatically
smaller (fleet split ~4×, source set narrowed to its own catchment).

Day-capacities of destination objects are shared across sub-problems and are
deducted between iterations, so different depots cannot overcommit the same
object.
"""

from __future__ import annotations

from copy import deepcopy
from collections import defaultdict
from typing import Any

from .real_milp_ksenya_solver import _solve_real_milp_ksenya_core, _validate_payload


def solve_real_milp_ksenya_decomp(
    payload: dict[str, Any],
    *,
    knn_k: int | None = None,
    time_limit_sec: int = 120,
    mip_rel_gap: float = 0.05,
    verbose: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """MILP with nearest-depot decomposition.

    Parameters
    ----------
    payload
        Standard dataset payload (same shape as for ``solve_real_milp_ksenya``).
    knn_k
        If set, prune per-vehicle arcs to k nearest neighbours per node within
        each sub-MILP (force-keeping depot↔source, source→dest, dest→depot of
        every compatible task). Combine with decomposition to fit larger
        datasets in memory; ``None`` keeps the full Cartesian arc set.
    time_limit_sec
        Global time budget; split evenly across active depots (with a
        per-depot floor of 10s).
    mip_rel_gap
        MIP relative gap passed to HiGHS for each sub-problem.
    verbose
        Log group sizes and per-sub-problem progress.
    """
    _validate_payload(payload)
    return _solve_real_milp_ksenya_decomposed_public(
        payload,
        knn_k=knn_k,
        time_limit_sec=time_limit_sec,
        mip_rel_gap=mip_rel_gap,
        verbose=verbose,
    )


def _solve_real_milp_ksenya_decomposed_public(
    payload: dict[str, Any],
    *,
    knn_k: int | None,
    time_limit_sec: int,
    mip_rel_gap: float,
    verbose: bool,
    log_prefix: str = "milp_ksenya_decomp",
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = deepcopy(payload)
    nodes = payload["graph"]["nodes"]
    node_by_id = {n["node_id"]: n for n in nodes}
    depot_ids = list(payload["metadata"]["depot_node_ids"])
    agent_depots = dict(payload["metadata"]["agent_depots"])

    def _euclid(a: str, b: str) -> float:
        ax = float(node_by_id[a].get("x") or 0.0)
        ay = float(node_by_id[a].get("y") or 0.0)
        bx = float(node_by_id[b].get("x") or 0.0)
        by = float(node_by_id[b].get("y") or 0.0)
        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

    tasks_by_depot: dict[str, list[str]] = {d: [] for d in depot_ids}
    for t in payload["tasks"]:
        s = t["source_node_id"]
        nearest = min(depot_ids, key=lambda d: _euclid(s, d))
        tasks_by_depot[nearest].append(t["task_id"])

    agents_by_depot: dict[str, list[str]] = {d: [] for d in depot_ids}
    for a in payload["agents"]:
        d = agent_depots.get(a["agent_id"])
        if d in agents_by_depot:
            agents_by_depot[d].append(a["agent_id"])

    active_depots = [d for d in depot_ids if tasks_by_depot[d] and agents_by_depot[d]]
    n_active = max(1, len(active_depots))
    per_depot_limit = max(10, int(time_limit_sec) // n_active)

    merged_states: dict[str, dict[str, Any]] = {
        a["agent_id"]: {
            "agent_id": a["agent_id"],
            "vehicle_type": a["vehicle_type"],
            "capacity_tons": float(a["capacity_tons"]),
            "is_compact": bool(a["is_compact"]),
            "depot_node": agent_depots[a["agent_id"]],
            "current_node": agent_depots[a["agent_id"]],
            "task_ids": [],
            "route_ids": [],
            "total_km": 0.0,
            "total_hours": 0.0,
        }
        for a in payload["agents"]
    }
    all_routes: list[dict[str, Any]] = []
    all_unassigned: list[str] = []
    total_transport_work = 0.0
    sub_objectives: list[float] = []
    overall_success = True

    if verbose:
        sizes = ", ".join(
            f"{d}: {len(tasks_by_depot[d])}t/{len(agents_by_depot[d])}a"
            for d in depot_ids
        )
        knn_note = f", knn_k={knn_k}" if knn_k else ""
        print(f"[{log_prefix}] groups by nearest depot: {sizes}{knn_note}")
        print(
            f"[{log_prefix}] active depots: {n_active}, "
            f"per-depot time limit: {per_depot_limit}s"
        )

    for d in depot_ids:
        tids = tasks_by_depot[d]
        aids = set(agents_by_depot[d])
        if not tids:
            continue
        if not aids:
            all_unassigned.extend(tids)
            if verbose:
                print(
                    f"[{log_prefix}] depot {d}: {len(tids)} task(s) but no "
                    f"agents — all unassigned"
                )
            continue

        sub_log_prefix = f"{log_prefix}/depot={d}"
        sub_meta, sub_sol = _solve_real_milp_ksenya_core(
            payload,
            task_subset=set(tids),
            agent_subset=aids,
            knn_k=knn_k,
            time_limit_sec=per_depot_limit,
            mip_rel_gap=mip_rel_gap,
            verbose=verbose,
            log_prefix=sub_log_prefix,
        )

        if not sub_meta.get("success"):
            overall_success = False
            all_unassigned.extend(tids)
            continue

        sub_objectives.append(float(sub_meta.get("objective") or 0.0))
        all_routes.extend(sub_sol["routes"])
        for v_id, st in sub_sol["states"].items():
            merged_states[v_id] = st
        all_unassigned.extend(sub_sol["unassigned"])
        total_transport_work += float(sub_sol.get("transport_work_ton_km") or 0.0)

        assigned_this_depot = set(tids) - set(sub_sol["unassigned"])
        consumed_by_dest: dict[str, float] = defaultdict(float)
        for t in payload["tasks"]:
            if t["task_id"] in assigned_this_depot:
                consumed_by_dest[t["destination_node_id"]] += float(t["mass_tons"])
        for n in nodes:
            if n["kind"] == "object1" and n["node_id"] in consumed_by_dest:
                cur = float(n.get("object_day_capacity_tons", 0.0) or 0.0)
                n["object_day_capacity_tons"] = max(
                    0.0, cur - consumed_by_dest[n["node_id"]]
                )

    return {"success": overall_success, "objective": sum(sub_objectives)}, {
        "routes": all_routes,
        "states": merged_states,
        "unassigned": all_unassigned,
        "transport_work_ton_km": round(total_transport_work, 3),
    }
