#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _f(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except (TypeError, ValueError):
        return float(default)


def _b(x: Any) -> bool:
    return bool(x)


def _required_types(task: dict[str, Any]) -> tuple[str, ...]:
    req = task.get("required_container_types")
    if isinstance(req, list) and req:
        return tuple(str(v) for v in req)
    c = str(task.get("container_type", "A"))
    if c in {"A+B", "A+C", "B+C", "A+B+C"}:
        return tuple(c.split("+"))
    return (c,)


@dataclass(frozen=True)
class AgentLite:
    agent_id: str
    zone_num: int | None
    cap_a: bool
    cap_b: bool
    cap_c: bool
    cap_d: bool
    compaction_coeff: float
    max_raw_volume_m3: float


def _compatible(task: dict[str, Any], agent: AgentLite) -> bool:
    # zone match
    tz = task.get("source_zone_num")
    task_zone = int(tz) if tz is not None else None
    if task_zone is not None and agent.zone_num is not None and task_zone != agent.zone_num:
        return False

    # required_container_types: OR semantics (at least one)
    req = _required_types(task)
    if req:
        has_any = False
        for c in req:
            if c == "A" and agent.cap_a:
                has_any = True
            elif c == "B" and agent.cap_b:
                has_any = True
            elif c == "C" and agent.cap_c:
                has_any = True
        if not has_any:
            return False

    # compact-D constraint
    if _b(task.get("requires_compact_d", False)) and not agent.cap_d:
        return False

    # effective volume check
    raw = _f(task.get("volume_raw_m3"), 0.0)
    if _b(task.get("is_compactable", False)):
        eff = raw / max(agent.compaction_coeff, 1.0)
    else:
        eff = raw
    if eff > agent.max_raw_volume_m3 + 1e-9:
        return False

    return True


def _compacted_ref(task: dict[str, Any]) -> float:
    ref = task.get("volume_compacted_m3_ref")
    if ref is not None:
        return _f(ref, 0.0)
    # fallback if field absent
    raw = _f(task.get("volume_raw_m3"), 0.0)
    comp_ref = max(_f(task.get("compaction_coeff_ref"), 1.0), 1.0)
    if _b(task.get("is_compactable", False)):
        return raw / comp_ref
    return raw


def build_subset(input_path: Path, output_path: Path, summary_path: Path) -> None:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    tasks: list[dict[str, Any]] = list(payload.get("tasks", []) or [])
    agents_raw = list(payload.get("agents", []) or [])
    nodes = list((payload.get("graph") or {}).get("nodes", []) or [])

    active_agents: list[AgentLite] = []
    for a in agents_raw:
        if not _b(a.get("is_active_work_1st_shoulder", False)):
            continue
        z = a.get("zone_num")
        zone_num = int(z) if z is not None else None
        active_agents.append(
            AgentLite(
                agent_id=str(a.get("agent_id", "")),
                zone_num=zone_num,
                cap_a=_b(a.get("cap_container_A", False)),
                cap_b=_b(a.get("cap_container_B", False)),
                cap_c=_b(a.get("cap_container_C", False)),
                cap_d=_b(a.get("cap_container_D", False)),
                compaction_coeff=max(_f(a.get("compaction_coeff"), 1.0), 1.0),
                max_raw_volume_m3=max(_f(a.get("max_raw_volume_m3"), 0.0), 0.0),
            )
        )

    # Object capacities
    object_capacity: dict[str, float] = {}
    for n in nodes:
        kind = str(n.get("kind", ""))
        if kind.startswith("object"):
            object_capacity[str(n.get("node_id"))] = _f(n.get("object_day_capacity_volume_m3"), 0.0)

    original_task_count = len(tasks)
    original_raw = sum(_f(t.get("volume_raw_m3"), 0.0) for t in tasks)
    original_comp = sum(_compacted_ref(t) for t in tasks)

    # Stage 1: remove tasks with zero compatible active agents
    removed_no_compat: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    compat_count_by_task: dict[str, int] = {}
    for t in tasks:
        c = 0
        for a in active_agents:
            if _compatible(t, a):
                c += 1
        compat_count_by_task[str(t.get("task_id", ""))] = c
        if c == 0:
            removed_no_compat.append(t)
        else:
            kept.append(t)

    # Stage 2: fixed-destination object capacity repair
    by_object: dict[str, list[dict[str, Any]]] = {}
    for t in kept:
        oid = str(t.get("destination_node_id"))
        by_object.setdefault(oid, []).append(t)

    removed_overload: list[dict[str, Any]] = []
    object_util_before: dict[str, dict[str, float]] = {}
    object_util_after: dict[str, dict[str, float]] = {}

    for oid, cap in object_capacity.items():
        arr = by_object.get(oid, [])
        load_before = sum(_compacted_ref(t) for t in arr)
        object_util_before[oid] = {
            "cap_m3": cap,
            "load_compacted_ref_m3": load_before,
            "utilization_pct": (100.0 * load_before / cap) if cap > 1e-9 else 0.0,
        }
        if load_before <= cap + 1e-9:
            continue
        # remove minimal number of tasks by removing largest contributors first
        arr_sorted = sorted(arr, key=_compacted_ref, reverse=True)
        load = load_before
        keep_local: list[dict[str, Any]] = []
        for t in arr_sorted:
            if load > cap + 1e-9:
                removed_overload.append(t)
                load -= _compacted_ref(t)
            else:
                keep_local.append(t)
        # plus remaining not processed after cap reached
        # since we traversed all sorted, keep_local currently contains all after threshold
        by_object[oid] = keep_local

    # rebuild final task list preserving original order
    removed_ids = {str(t.get("task_id")) for t in removed_no_compat} | {str(t.get("task_id")) for t in removed_overload}
    final_tasks = [t for t in tasks if str(t.get("task_id")) not in removed_ids]

    # recompute post-checks
    tasks_with_0_compatible_agents = 0
    for t in final_tasks:
        has = False
        for a in active_agents:
            if _compatible(t, a):
                has = True
                break
        if not has:
            tasks_with_0_compatible_agents += 1

    load_after_by_object: dict[str, float] = {oid: 0.0 for oid in object_capacity}
    for t in final_tasks:
        oid = str(t.get("destination_node_id"))
        if oid in load_after_by_object:
            load_after_by_object[oid] += _compacted_ref(t)
    object_overload_count = 0
    for oid, cap in object_capacity.items():
        load_after = load_after_by_object.get(oid, 0.0)
        object_util_after[oid] = {
            "cap_m3": cap,
            "load_compacted_ref_m3": load_after,
            "utilization_pct": (100.0 * load_after / cap) if cap > 1e-9 else 0.0,
        }
        if load_after > cap + 1e-9:
            object_overload_count += 1

    removed_task_count = len(removed_ids)
    final_task_count = len(final_tasks)
    removed_raw = sum(_f(t.get("volume_raw_m3"), 0.0) for t in tasks if str(t.get("task_id")) in removed_ids)
    removed_comp = sum(_compacted_ref(t) for t in tasks if str(t.get("task_id")) in removed_ids)
    remaining_raw = original_raw - removed_raw
    remaining_comp = original_comp - removed_comp

    # Update dataset payload (keep structure)
    payload["tasks"] = final_tasks
    md = payload.get("metadata")
    if isinstance(md, dict):
        counts = md.get("counts")
        if isinstance(counts, dict):
            counts["tasks"] = final_task_count
        md["derived_filters"] = {
            "name": "feasible_subset_v1",
            "removed_no_compatible_agents": len(removed_no_compat),
            "removed_object_overload": len(removed_overload),
            "removed_total": removed_task_count,
        }
        md["summary"] = {
            "tasks_original": original_task_count,
            "tasks_kept": final_task_count,
            "tasks_removed": removed_task_count,
            "tasks_removed_pct": (100.0 * removed_task_count / max(original_task_count, 1)),
            "tasks_with_0_compatible_agents": tasks_with_0_compatible_agents,
            "object_overload_count": object_overload_count,
        }
    else:
        payload["metadata"] = {
            "derived_filters": {
                "name": "feasible_subset_v1",
            },
            "summary": {
                "tasks_original": original_task_count,
                "tasks_kept": final_task_count,
                "tasks_removed": removed_task_count,
                "tasks_removed_pct": (100.0 * removed_task_count / max(original_task_count, 1)),
                "tasks_with_0_compatible_agents": tasks_with_0_compatible_agents,
                "object_overload_count": object_overload_count,
            },
        }

    out_summary = {
        "input_dataset_path": str(input_path),
        "output_dataset_path": str(output_path),
        "tasks_original": original_task_count,
        "tasks_kept": final_task_count,
        "tasks_removed": removed_task_count,
        "tasks_removed_pct": (100.0 * removed_task_count / max(original_task_count, 1)),
        "removed_breakdown": {
            "no_compatible_agents": len(removed_no_compat),
            "object_overload_repair": len(removed_overload),
        },
        "volumes_m3": {
            "original_raw": original_raw,
            "original_compacted_ref": original_comp,
            "removed_raw": removed_raw,
            "removed_compacted_ref": removed_comp,
            "remaining_raw": remaining_raw,
            "remaining_compacted_ref": remaining_comp,
        },
        "object_utilization_before": object_util_before,
        "object_utilization_after": object_util_after,
        "checks": {
            "tasks_with_0_compatible_agents": tasks_with_0_compatible_agents,
            "object_overload_count": object_overload_count,
        },
    }

    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(out_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved dataset: {output_path}")
    print(f"Saved summary: {summary_path}")
    print(
        json.dumps(
            {
                "tasks_original": original_task_count,
                "tasks_kept": final_task_count,
                "tasks_removed": removed_task_count,
                "removed_pct": round(100.0 * removed_task_count / max(original_task_count, 1), 3),
                "checks": out_summary["checks"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build feasible subset for volume-only dataset by removing minimal tasks.")
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path(
            "/Users/igoreshka/Desktop/Optimization-of-flows/demo/data/object_volume_feasible_fullfleet/full_all_tasks_all_agents_stage4_volume_only/dataset_real_spb_clean_full_all_tasks_all_agents_stage4_volume_only.json"
        ),
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args()

    input_path = args.input_json.resolve()
    if args.output_json is None:
        output_path = input_path.with_name(input_path.stem + "_feasible_subset.json")
    else:
        output_path = args.output_json.resolve()

    if args.summary_json is None:
        summary_path = output_path.with_name("summary_" + output_path.stem + ".json")
    else:
        summary_path = args.summary_json.resolve()

    build_subset(input_path=input_path, output_path=output_path, summary_path=summary_path)


if __name__ == "__main__":
    main()

