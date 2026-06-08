from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import networkx as nx

from .distance import DistanceEngine
from .models import (
    AssignmentSolution,
    ConstraintReport,
    EvaluationResult,
    GraphNodeLite,
    VolumeAgent,
    VolumeTask,
)


def _f(x: Any, d: float = 0.0) -> float:
    try:
        if x is None:
            return float(d)
        return float(x)
    except (TypeError, ValueError):
        return float(d)


def _i(x: Any) -> int | None:
    if x is None:
        return None
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def _req_types(raw_task: dict[str, Any]) -> tuple[str, ...]:
    req = raw_task.get("required_container_types")
    if isinstance(req, list) and req:
        return tuple(sorted(str(v) for v in req))
    c = str(raw_task.get("container_type", "A"))
    if c in {"A+B", "A+C", "B+C", "A+B+C"}:
        return tuple(c.split("+"))
    return (c,)


def _service_hours(task: VolumeTask, service_map: dict[str, float]) -> float:
    if task.container_type in service_map:
        return float(service_map[task.container_type])
    vals = [float(service_map.get(x, 0.25)) for x in task.required_container_types]
    return max(vals) if vals else 0.25


@dataclass
class VolumeDataset:
    dataset_path: Path
    payload: dict[str, Any]
    graph: nx.DiGraph
    nodes: dict[str, GraphNodeLite]
    tasks: list[VolumeTask]
    agents: list[VolumeAgent]
    object_volume_caps: dict[str, float]
    service_hours_by_container: dict[str, float]
    dist: DistanceEngine

    @classmethod
    def from_json(cls, dataset_path: Path | str) -> "VolumeDataset":
        p = Path(dataset_path).resolve()
        payload = json.loads(p.read_text(encoding="utf-8"))
        nodes_raw = (payload.get("graph") or {}).get("nodes", []) or []
        edges_raw = (payload.get("graph") or {}).get("edges", []) or []
        md = payload.get("metadata") or {}

        nodes: dict[str, GraphNodeLite] = {}
        g = nx.DiGraph()
        for n in nodes_raw:
            nid = str(n.get("node_id"))
            node = GraphNodeLite(
                node_id=nid,
                kind=str(n.get("kind", "")),
                x=_f(n.get("x"), 0.0),
                y=_f(n.get("y"), 0.0),
                object_day_capacity_volume_m3=_f(n.get("object_day_capacity_volume_m3"), 0.0),
            )
            nodes[nid] = node
            g.add_node(nid)

        for e in edges_raw:
            u = str(e.get("source_id"))
            v = str(e.get("target_id"))
            g.add_edge(u, v, distance_km=_f(e.get("distance_km"), 0.0))

        tasks: list[VolumeTask] = []
        for t in payload.get("tasks", []) or []:
            tid = str(t.get("task_id", "")).strip()
            if not tid:
                continue
            tasks.append(
                VolumeTask(
                    task_id=tid,
                    source_node_id=str(t.get("source_node_id")),
                    destination_node_id=str(t.get("destination_node_id")),
                    source_zone_num=_i(t.get("source_zone_num")),
                    volume_raw_m3=_f(t.get("volume_raw_m3"), 0.0),
                    is_compactable=bool(t.get("is_compactable", False)),
                    requires_compact_d=bool(t.get("requires_compact_d", False)),
                    required_container_types=_req_types(t),
                    container_type=str(t.get("container_type", "A")),
                )
            )

        vp = md.get("vehicle_profiles") or {}
        agents: list[VolumeAgent] = []
        for a in payload.get("agents", []) or []:
            aid = str(a.get("agent_id", "")).strip()
            if not aid:
                continue
            vt = str(a.get("vehicle_type", ""))
            pf = vp.get(vt, {}) if isinstance(vp, dict) else {}
            max_hours = _f(a.get("max_hours"), _f(pf.get("max_shift_hours"), 10.0))
            max_km = _f(a.get("max_daily_km"), _f(pf.get("max_daily_km"), 260.0))
            speed = _f(a.get("avg_speed_kmph"), _f(pf.get("avg_speed_kmph"), 35.0))
            raw_lim = _f(a.get("max_raw_volume_m3"), 0.0)
            if raw_lim <= 0:
                body = _f(a.get("body_volume_m3"), 0.0)
                comp = max(1.0, _f(a.get("compaction_coeff"), 1.0))
                raw_lim = body * comp if body > 0 else 0.0
            agents.append(
                VolumeAgent(
                    agent_id=aid,
                    depot_node_id=str(a.get("depot_node_id")) if a.get("depot_node_id") is not None else None,
                    zone_num=_i(a.get("zone_num")),
                    is_active=bool(a.get("is_active_work_1st_shoulder", False)),
                    cap_a=bool(a.get("cap_container_A", False)),
                    cap_b=bool(a.get("cap_container_B", False)),
                    cap_c=bool(a.get("cap_container_C", False)),
                    cap_d=bool(a.get("cap_container_D", False)),
                    compaction_coeff=max(1.0, _f(a.get("compaction_coeff"), 1.0)),
                    max_raw_volume_m3=max(0.0, raw_lim),
                    max_hours=max(0.0, max_hours),
                    max_daily_km=max(0.0, max_km),
                    avg_speed_kmph=max(1e-6, speed),
                )
            )

        object_caps: dict[str, float] = {}
        for nid, n in nodes.items():
            if n.kind.startswith("object"):
                object_caps[nid] = n.object_day_capacity_volume_m3

        precomp = (md.get("precomputed_distances") or {}) if isinstance(md, dict) else {}
        dist = DistanceEngine(nx_graph=g, precomputed=precomp, base_dir=p.parent)

        service_map = {str(k): _f(v, 0.25) for k, v in (md.get("service_hours_by_container") or {}).items()}

        return cls(
            dataset_path=p,
            payload=payload,
            graph=g,
            nodes=nodes,
            tasks=tasks,
            agents=agents,
            object_volume_caps=object_caps,
            service_hours_by_container=service_map,
            dist=dist,
        )

    def effective_task_volume(self, task: VolumeTask, agent: VolumeAgent) -> float:
        if task.is_compactable and agent.compaction_coeff > 1.0:
            return max(0.0, task.volume_raw_m3 / agent.compaction_coeff)
        return max(0.0, task.volume_raw_m3)

    def agent_can_take_task(self, task: VolumeTask, agent: VolumeAgent) -> bool:
        if not agent.is_active:
            return False
        if agent.max_raw_volume_m3 <= 0:
            return False
        if task.source_zone_num is not None and agent.zone_num is not None and task.source_zone_num != agent.zone_num:
            return False

        req = task.required_container_types
        # OR semantics: for multi-type task (e.g. A+C) agent must support at least one type.
        if req:
            has_any = False
            for c in req:
                if c == "A" and agent.cap_a:
                    has_any = True
                if c == "B" and agent.cap_b:
                    has_any = True
                if c == "C" and agent.cap_c:
                    has_any = True
            if not has_any:
                return False
        if task.requires_compact_d and not agent.cap_d:
            return False

        eff = self.effective_task_volume(task, agent)
        if eff > agent.max_raw_volume_m3 + 1e-9:
            return False

        if agent.depot_node_id is None:
            return False
        d1 = self.dist.dist(agent.depot_node_id, task.source_node_id)
        d2 = self.dist.dist(task.source_node_id, task.destination_node_id)
        d3 = self.dist.dist(task.destination_node_id, agent.depot_node_id)
        if d1 == float("inf") or d2 == float("inf") or d3 == float("inf"):
            return False
        trip_km = d1 + d2 + d3
        trip_h = trip_km / max(agent.avg_speed_kmph, 1e-6) + self.route_service_hours(task)
        if trip_km > agent.max_daily_km + 1e-9:
            return False
        if trip_h > agent.max_hours + 1e-9:
            return False
        return True

    def evaluate(self, solution: AssignmentSolution) -> EvaluationResult:
        task_by_id = {t.task_id: t for t in self.tasks}
        agent_by_id = {a.agent_id: a for a in self.agents}

        seen: set[str] = set()
        duplicates = 0
        unknown_tasks = 0
        unknown_agents = 0
        compat_viol = 0
        object_used: dict[str, float] = {k: 0.0 for k in self.object_volume_caps}
        agent_km: dict[str, float] = {a.agent_id: 0.0 for a in self.agents}
        agent_h: dict[str, float] = {a.agent_id: 0.0 for a in self.agents}

        total_km = 0.0
        total_hours = 0.0
        tr_v_km = 0.0

        for tr in solution.trips:
            agent = agent_by_id.get(tr.agent_id)
            if agent is None:
                unknown_agents += 1
                continue
            total_km += float(tr.total_km)
            total_hours += float(tr.total_hours)
            agent_km[agent.agent_id] += float(tr.total_km)
            agent_h[agent.agent_id] += float(tr.total_hours)
            for tp in tr.task_pickups:
                tr_v_km += float(tp.effective_volume_m3) * float(tp.carried_distance_to_object_km)

            for tid in tr.ordered_task_ids:
                task = task_by_id.get(tid)
                if task is None:
                    unknown_tasks += 1
                    continue
                if tid in seen:
                    duplicates += 1
                    continue
                seen.add(tid)
                if not self.agent_can_take_task(task, agent):
                    compat_viol += 1
                object_used[task.destination_node_id] = object_used.get(task.destination_node_id, 0.0) + self.effective_task_volume(
                    task, agent
                )

        all_task_ids = {t.task_id for t in self.tasks}
        unassigned = sorted((all_task_ids - seen) | set(solution.unassigned_task_ids))

        overflow_km = 0
        overflow_hours = 0
        for a in self.agents:
            if agent_km[a.agent_id] > a.max_daily_km + 1e-9:
                overflow_km += 1
            if agent_h[a.agent_id] > a.max_hours + 1e-9:
                overflow_hours += 1

        object_viol = 0
        for oid, cap in self.object_volume_caps.items():
            if cap > 0 and object_used.get(oid, 0.0) > cap + 1e-9:
                object_viol += 1

        checks = {
            "all_tasks_assigned": len(unassigned) == 0,
            "duplicates": int(duplicates),
            "unknown_tasks": int(unknown_tasks),
            "unknown_agents": int(unknown_agents),
            "compatibility_violations": int(compat_viol),
            "overflow_km_agents": int(overflow_km),
            "overflow_hours_agents": int(overflow_hours),
            "object_volume_violations": int(object_viol),
            "task_space_match": True,
            "metric_task_space": "dataset_reference",
        }
        all_checks_ok = (
            duplicates == 0
            and unknown_tasks == 0
            and unknown_agents == 0
            and compat_viol == 0
            and overflow_km == 0
            and overflow_hours == 0
            and object_viol == 0
            and len(unassigned) == 0
        )

        bottlenecks = {
            "unassigned_tasks": len(unassigned),
            "overflow_km_agents": overflow_km,
            "overflow_hours_agents": overflow_hours,
            "object_volume_violations": object_viol,
        }

        report = ConstraintReport(
            all_checks_ok=bool(all_checks_ok),
            feasible=bool(all_checks_ok),
            checks=checks,
            bottlenecks=bottlenecks,
            summary={
                "assigned_tasks": int(len(seen)),
                "unassigned_tasks": int(len(unassigned)),
                "total_tasks": int(len(self.tasks)),
            },
        )

        active_agents = sum(1 for aid in agent_km if agent_km[aid] > 0 or agent_h[aid] > 0)
        return EvaluationResult(
            algorithm=solution.algorithm,
            dataset_path=str(self.dataset_path),
            total_tasks=len(self.tasks),
            assigned_tasks=len(seen),
            unassigned_tasks=len(unassigned),
            task_coverage_pct=round(100.0 * len(seen) / max(len(self.tasks), 1), 3),
            active_agents=active_agents,
            trips_count=len(solution.trips),
            total_km=round(total_km, 3),
            total_hours=round(total_hours, 3),
            transport_work_volume_m3_km=round(tr_v_km, 3),
            object_volume_used_m3={k: round(v, 3) for k, v in object_used.items()},
            object_volume_capacity_m3={k: round(v, 3) for k, v in self.object_volume_caps.items()},
            constraints=report,
            runtime_sec=round(float(solution.runtime_sec), 3),
        )

    def route_service_hours(self, task: VolumeTask) -> float:
        return _service_hours(task, self.service_hours_by_container)
