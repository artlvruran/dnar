from __future__ import annotations

from dataclasses import dataclass
import json
import math
import time
from pathlib import Path
from typing import Any

import networkx as nx


@dataclass(frozen=True)
class VolumeTask:
    task_id: str
    source_node_id: str
    destination_node_id: str
    source_zone_num: int | None
    volume_raw_m3: float
    is_compactable: bool
    requires_compact_d: bool
    required_container_types: tuple[str, ...]


@dataclass(frozen=True)
class VolumeAgent:
    agent_id: str
    depot_node_id: str | None
    zone_num: int | None
    is_active: bool
    cap_a: bool
    cap_b: bool
    cap_c: bool
    cap_d: bool
    compaction_coeff: float
    max_raw_volume_m3: float
    max_hours: float
    max_daily_km: float
    avg_speed_kmph: float


@dataclass(frozen=True)
class VolumeOnlyRoute:
    route_id: str
    agent_id: str
    task_id: str
    path: tuple[str, str, str, str]
    total_km: float
    total_hours: float
    loaded_km: float
    effective_volume_m3: float


@dataclass(frozen=True)
class VolumeOnlyRunResult:
    algorithm: str
    dataset_path: str
    total_tasks: int
    assigned_routes: int
    unassigned_tasks: int
    task_coverage_pct: float
    active_agents: int
    total_km: float
    total_hours: float
    transport_work_volume_m3_km: float
    runtime_sec: float
    all_checks_ok: bool
    checks: dict[str, Any]
    details: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "dataset_path": self.dataset_path,
            "total_tasks": self.total_tasks,
            "assigned_routes": self.assigned_routes,
            "unassigned_tasks": self.unassigned_tasks,
            "task_coverage_pct": self.task_coverage_pct,
            "active_agents": self.active_agents,
            "total_km": self.total_km,
            "total_hours": self.total_hours,
            "transport_work_volume_m3_km": self.transport_work_volume_m3_km,
            "runtime_sec": self.runtime_sec,
            "all_checks_ok": self.all_checks_ok,
            "checks": self.checks,
            "details": self.details,
        }


def _to_int(x: Any) -> int | None:
    if x is None:
        return None
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except (TypeError, ValueError):
        return float(default)


def _task_required_types(raw_task: dict[str, Any]) -> tuple[str, ...]:
    req = raw_task.get("required_container_types")
    if isinstance(req, list) and req:
        return tuple(sorted(str(x) for x in req))
    c = str(raw_task.get("container_type", "A"))
    if c == "A+B":
        return ("A", "B")
    if c == "A+C":
        return ("A", "C")
    if c == "B+C":
        return ("B", "C")
    return (c,)


def _load_volume_only_problem(
    dataset_path: Path,
) -> tuple[list[VolumeTask], list[VolumeAgent], dict[str, float], nx.DiGraph, dict[str, Any]]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    md = payload.get("metadata") or {}
    vp = md.get("vehicle_profiles") or {}

    graph = nx.DiGraph()
    for n in (payload.get("graph") or {}).get("nodes", []) or []:
        graph.add_node(str(n.get("node_id")))
    for e in (payload.get("graph") or {}).get("edges", []) or []:
        graph.add_edge(
            str(e.get("source_id")),
            str(e.get("target_id")),
            distance_km=_to_float(e.get("distance_km"), 0.0),
        )

    object_vol_caps: dict[str, float] = {}
    for n in (payload.get("graph") or {}).get("nodes", []) or []:
        if str(n.get("kind", "")).startswith("object"):
            object_vol_caps[str(n.get("node_id"))] = _to_float(n.get("object_day_capacity_volume_m3"), 0.0)

    tasks: list[VolumeTask] = []
    for t in payload.get("tasks", []) or []:
        tid = str(t.get("task_id"))
        if not tid:
            continue
        tasks.append(
            VolumeTask(
                task_id=tid,
                source_node_id=str(t.get("source_node_id")),
                destination_node_id=str(t.get("destination_node_id")),
                source_zone_num=_to_int(t.get("source_zone_num")),
                volume_raw_m3=_to_float(t.get("volume_raw_m3"), 0.0),
                is_compactable=bool(t.get("is_compactable", False)),
                requires_compact_d=bool(t.get("requires_compact_d", False)),
                required_container_types=_task_required_types(t),
            )
        )

    agents: list[VolumeAgent] = []
    for a in payload.get("agents", []) or []:
        aid = str(a.get("agent_id"))
        if not aid:
            continue
        vt = str(a.get("vehicle_type", ""))
        pf = vp.get(vt, {}) if isinstance(vp, dict) else {}

        max_hours = _to_float(a.get("max_hours"), _to_float(pf.get("max_shift_hours"), 10.0))
        max_km = _to_float(a.get("max_daily_km"), _to_float(pf.get("max_daily_km"), 260.0))
        speed = _to_float(a.get("avg_speed_kmph"), _to_float(pf.get("avg_speed_kmph"), 35.0))
        raw_lim = _to_float(a.get("max_raw_volume_m3"), 0.0)
        if raw_lim <= 0:
            body = _to_float(a.get("body_volume_m3"), 0.0)
            comp = max(1.0, _to_float(a.get("compaction_coeff"), 1.0))
            raw_lim = body * comp if body > 0 else 0.0

        agents.append(
            VolumeAgent(
                agent_id=aid,
                depot_node_id=str(a.get("depot_node_id")) if a.get("depot_node_id") is not None else None,
                zone_num=_to_int(a.get("zone_num")),
                is_active=bool(a.get("is_active_work_1st_shoulder", False)),
                cap_a=bool(a.get("cap_container_A", False)),
                cap_b=bool(a.get("cap_container_B", False)),
                cap_c=bool(a.get("cap_container_C", False)),
                cap_d=bool(a.get("cap_container_D", False)),
                compaction_coeff=max(1.0, _to_float(a.get("compaction_coeff"), 1.0)),
                max_raw_volume_m3=max(0.0, raw_lim),
                max_hours=max(0.0, max_hours),
                max_daily_km=max(0.0, max_km),
                avg_speed_kmph=max(1e-6, speed),
            )
        )

    return tasks, agents, object_vol_caps, graph, payload


def _agent_supports_required_types(agent: VolumeAgent, required: tuple[str, ...]) -> bool:
    if not required:
        return True
    # OR semantics: at least one required type must be supported.
    for c in required:
        if c == "A" and agent.cap_a:
            return True
        if c == "B" and agent.cap_b:
            return True
        if c == "C" and agent.cap_c:
            return True
    return False


def _effective_volume(task: VolumeTask, agent: VolumeAgent) -> float:
    raw = max(0.0, float(task.volume_raw_m3))
    if task.is_compactable and agent.compaction_coeff > 1.0:
        return raw / agent.compaction_coeff
    return raw


def _compatible(task: VolumeTask, agent: VolumeAgent) -> bool:
    if not agent.is_active:
        return False
    if task.source_zone_num is not None and agent.zone_num is not None and task.source_zone_num != agent.zone_num:
        return False
    if not _agent_supports_required_types(agent, task.required_container_types):
        return False
    if task.requires_compact_d and not agent.cap_d:
        return False
    eff = _effective_volume(task, agent)
    if agent.max_raw_volume_m3 > 0 and eff > agent.max_raw_volume_m3 + 1e-9:
        return False
    return True


def solve_volume_only_greedy(
    *,
    dataset_path: Path | str,
    max_runtime_sec: float = 30.0,
    max_tasks: int | None = None,
    top_k_agents: int = 40,
) -> VolumeOnlyRunResult:
    start = time.perf_counter()
    dataset_path = Path(dataset_path).resolve()
    tasks, agents, object_vol_caps, graph, payload = _load_volume_only_problem(dataset_path)

    tasks = sorted(tasks, key=lambda t: t.volume_raw_m3, reverse=True)
    if max_tasks is not None and max_tasks > 0:
        tasks = tasks[: int(max_tasks)]

    tasks_by_id = {t.task_id: t for t in tasks}
    unassigned = {t.task_id for t in tasks}

    agent_by_id = {a.agent_id: a for a in agents if a.is_active}
    used_km = {aid: 0.0 for aid in agent_by_id}
    used_h = {aid: 0.0 for aid in agent_by_id}
    object_used = {oid: 0.0 for oid in object_vol_caps}

    dist_cache: dict[tuple[str, str], float] = {}

    def dist(u: str, v: str) -> float:
        if u == v:
            return 0.0
        key = (u, v)
        if key in dist_cache:
            return dist_cache[key]
        try:
            d = float(nx.shortest_path_length(graph, source=u, target=v, weight="distance_km"))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            d = float("inf")
        dist_cache[key] = d
        return d

    pools: dict[tuple[int | None, str, bool], list[str]] = {}

    def add_pool(z: int | None, req: str, dflag: bool, aid: str) -> None:
        pools.setdefault((z, req, dflag), []).append(aid)

    for a in agent_by_id.values():
        reqs: list[str] = []
        if a.cap_a:
            reqs.append("A")
        if a.cap_b:
            reqs.append("B")
        if a.cap_c:
            reqs.append("C")
        keys: set[str] = set()
        if "A" in reqs:
            keys.add("A")
        if "B" in reqs:
            keys.add("B")
        if "C" in reqs:
            keys.add("C")
        if "A" in reqs and "B" in reqs:
            keys.add("A+B")
        if "A" in reqs and "C" in reqs:
            keys.add("A+C")
        if "B" in reqs and "C" in reqs:
            keys.add("B+C")
        if "A" in reqs and "B" in reqs and "C" in reqs:
            keys.add("A+B+C")
        for k in keys:
            add_pool(a.zone_num, k, False, a.agent_id)
            if a.cap_d:
                add_pool(a.zone_num, k, True, a.agent_id)

    def req_keys_or(task: VolumeTask) -> list[str]:
        req = list(task.required_container_types) if task.required_container_types else ["A"]
        return [str(x) for x in req]

    routes: list[VolumeOnlyRoute] = []
    active_agents: set[str] = set()
    route_idx = 0

    for task in tasks:
        if (time.perf_counter() - start) >= max_runtime_sec:
            break
        if task.task_id not in unassigned:
            continue

        candidate_ids: list[str] = []
        for key in req_keys_or(task):
            candidate_ids += list(pools.get((task.source_zone_num, key, task.requires_compact_d), []))
            if task.source_zone_num is not None:
                candidate_ids += pools.get((None, key, task.requires_compact_d), [])
        if not candidate_ids:
            continue

        candidate_ids = list(dict.fromkeys(candidate_ids))
        candidate_ids.sort(key=lambda aid: (used_h.get(aid, 0.0), used_km.get(aid, 0.0)))
        candidate_ids = candidate_ids[: max(1, int(top_k_agents))]

        best: tuple[float, float, VolumeAgent, float, float, float] | None = None
        for aid in candidate_ids:
            agent = agent_by_id.get(aid)
            if agent is None or agent.depot_node_id is None:
                continue
            if not _compatible(task, agent):
                continue

            rem_km = agent.max_daily_km - used_km[aid]
            rem_h = agent.max_hours - used_h[aid]
            if rem_km <= 1e-9 or rem_h <= 1e-9:
                continue

            eff_vol = _effective_volume(task, agent)
            cap = object_vol_caps.get(task.destination_node_id, 0.0)
            if cap > 0 and object_used.get(task.destination_node_id, 0.0) + eff_vol > cap + 1e-9:
                continue

            d1 = dist(agent.depot_node_id, task.source_node_id)
            d2 = dist(task.source_node_id, task.destination_node_id)
            d3 = dist(task.destination_node_id, agent.depot_node_id)
            if math.isinf(d1) or math.isinf(d2) or math.isinf(d3):
                continue

            trip_km = d1 + d2 + d3
            trip_h = trip_km / max(agent.avg_speed_kmph, 1e-6)
            if trip_km > rem_km + 1e-9 or trip_h > rem_h + 1e-9:
                continue

            score = eff_vol / max(trip_km, 1e-6)
            cand = (score, -trip_km, agent, eff_vol, trip_km, trip_h)
            if best is None or (cand[0], cand[1]) > (best[0], best[1]):
                best = cand

        if best is None:
            continue

        _score, _neg_km, agent, eff_vol, trip_km, trip_h = best
        route_idx += 1
        routes.append(
            VolumeOnlyRoute(
                route_id=f"VO_ROUTE_{route_idx:07d}",
                agent_id=agent.agent_id,
                task_id=task.task_id,
                path=(agent.depot_node_id, task.source_node_id, task.destination_node_id, agent.depot_node_id),
                total_km=round(trip_km, 6),
                total_hours=round(trip_h, 6),
                loaded_km=round(dist(task.source_node_id, task.destination_node_id), 6),
                effective_volume_m3=round(eff_vol, 6),
            )
        )
        used_km[agent.agent_id] += trip_km
        used_h[agent.agent_id] += trip_h
        object_used[task.destination_node_id] = object_used.get(task.destination_node_id, 0.0) + eff_vol
        unassigned.remove(task.task_id)
        active_agents.add(agent.agent_id)

    overflow_km = sum(1 for a in agent_by_id.values() if used_km[a.agent_id] > a.max_daily_km + 1e-9)
    overflow_h = sum(1 for a in agent_by_id.values() if used_h[a.agent_id] > a.max_hours + 1e-9)
    object_overflow = 0
    for oid, cap in object_vol_caps.items():
        if cap > 0 and object_used.get(oid, 0.0) > cap + 1e-9:
            object_overflow += 1

    total_km = float(sum(r.total_km for r in routes))
    total_h = float(sum(r.total_hours for r in routes))
    tr_vol_km = float(sum(r.effective_volume_m3 * r.loaded_km for r in routes))
    assigned = len(routes)
    total_tasks = len(tasks)
    unassigned_n = len(unassigned)
    coverage = (100.0 * assigned / total_tasks) if total_tasks else 100.0

    checks = {
        "overflow_km_agents": int(overflow_km),
        "overflow_hours_agents": int(overflow_h),
        "object_volume_violations": int(object_overflow),
        "all_tasks_assigned": bool(unassigned_n == 0),
        "all_checks_ok": bool(overflow_km == 0 and overflow_h == 0 and object_overflow == 0),
    }

    details = {
        "max_runtime_sec": float(max_runtime_sec),
        "max_tasks": int(max_tasks) if max_tasks is not None else None,
        "top_k_agents": int(top_k_agents),
        "routes_preview": [
            {
                "route_id": r.route_id,
                "agent_id": r.agent_id,
                "task_id": r.task_id,
                "path": list(r.path),
                "total_km": r.total_km,
                "total_hours": r.total_hours,
                "effective_volume_m3": r.effective_volume_m3,
            }
            for r in routes[:20]
        ],
        "dataset_counts": ((payload.get("metadata") or {}).get("counts") or {}),
    }

    return VolumeOnlyRunResult(
        algorithm="volume_only_greedy_v1",
        dataset_path=str(dataset_path),
        total_tasks=int(total_tasks),
        assigned_routes=int(assigned),
        unassigned_tasks=int(unassigned_n),
        task_coverage_pct=round(float(coverage), 3),
        active_agents=int(len(active_agents)),
        total_km=round(total_km, 3),
        total_hours=round(total_h, 3),
        transport_work_volume_m3_km=round(tr_vol_km, 3),
        runtime_sec=round(time.perf_counter() - start, 3),
        all_checks_ok=bool(checks["all_checks_ok"]),
        checks=checks,
        details=details,
    )
