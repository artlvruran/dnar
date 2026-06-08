from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from .dataset import VolumeDataset
from .models import AssignmentSolution, EvaluationResult


def save_solution_artifacts(
    *,
    dataset: VolumeDataset,
    solution: AssignmentSolution,
    evaluation: EvaluationResult,
    out_dir: Path | str,
) -> dict[str, str]:
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # 1) summary
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(evaluation.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    # 2) solver logs
    logs_path = out / "solver_logs.txt"
    logs_path.write_text("\n".join(solution.solver_logs), encoding="utf-8")

    # 3) per-trip detailed logs
    trips_rows = []
    task_rows = []
    agent_chains: dict[str, list[str]] = {}
    for tr in solution.trips:
        chain = agent_chains.setdefault(tr.agent_id, [])
        if not chain:
            chain.extend([tr.depot_node_id])
        chain.extend(list(tr.visit_nodes[1:]))

        trips_rows.append(
            {
                "trip_id": tr.trip_id,
                "agent_id": tr.agent_id,
                "destination_object_id": tr.destination_object_id,
                "tasks_count": len(tr.ordered_task_ids),
                "ordered_task_ids": list(tr.ordered_task_ids),
                "visit_nodes": list(tr.visit_nodes),
                "total_km": tr.total_km,
                "total_hours": tr.total_hours,
                "payload_effective_volume_m3": tr.payload_effective_volume_m3,
            }
        )
        for tp in tr.task_pickups:
            task_rows.append(
                {
                    "trip_id": tr.trip_id,
                    "agent_id": tr.agent_id,
                    "task_id": tp.task_id,
                    "source_node_id": tp.source_node_id,
                    "effective_volume_m3": tp.effective_volume_m3,
                    "carried_distance_to_object_km": tp.carried_distance_to_object_km,
                }
            )

    trips_path = out / "trips.json"
    tasks_path = out / "task_transport_logs.csv"
    chains_path = out / "agent_visit_sequences.json"
    trips_path.write_text(json.dumps(trips_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(task_rows).to_csv(tasks_path, index=False)
    chains_path.write_text(json.dumps(agent_chains, ensure_ascii=False, indent=2), encoding="utf-8")

    # 4) distributions
    dist_dir = out / "distributions"
    dist_dir.mkdir(exist_ok=True)

    if task_rows:
        df_task = pd.DataFrame(task_rows)
        plt.figure(figsize=(8, 4))
        plt.hist(df_task["effective_volume_m3"], bins=40)
        plt.title("Task Effective Volume Distribution")
        plt.xlabel("effective_volume_m3")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(dist_dir / "tasks_effective_volume_hist.png", dpi=140)
        plt.close()

    # fleet volume capacity distribution
    vols = [a.max_raw_volume_m3 for a in dataset.agents if a.max_raw_volume_m3 > 0]
    if vols:
        plt.figure(figsize=(8, 4))
        plt.hist(vols, bins=40)
        plt.title("Fleet Max Raw Volume Distribution")
        plt.xlabel("max_raw_volume_m3")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(dist_dir / "fleet_raw_volume_capacity_hist.png", dpi=140)
        plt.close()

    # agent time usage
    by_agent_h: dict[str, float] = {}
    by_agent_obj_visits: dict[str, int] = {}
    for tr in solution.trips:
        by_agent_h[tr.agent_id] = by_agent_h.get(tr.agent_id, 0.0) + tr.total_hours
        by_agent_obj_visits[tr.agent_id] = by_agent_obj_visits.get(tr.agent_id, 0) + 1

    if by_agent_h:
        plt.figure(figsize=(8, 4))
        plt.hist(list(by_agent_h.values()), bins=40)
        plt.title("Agent Work Hours Distribution")
        plt.xlabel("hours")
        plt.ylabel("agents")
        plt.tight_layout()
        plt.savefig(dist_dir / "agent_work_hours_hist.png", dpi=140)
        plt.close()

    # object visit distribution by agents
    if by_agent_obj_visits:
        plt.figure(figsize=(8, 4))
        plt.hist(list(by_agent_obj_visits.values()), bins=30)
        plt.title("Trips per Active Agent")
        plt.xlabel("trips_count")
        plt.ylabel("agents")
        plt.tight_layout()
        plt.savefig(dist_dir / "trips_per_agent_hist.png", dpi=140)
        plt.close()

    # 5) per-object path maps
    maps_dir = out / "object_maps"
    maps_dir.mkdir(exist_ok=True)
    flow_maps_dir = out / "object_flow_zone_maps"
    flow_maps_dir.mkdir(exist_ok=True)

    object_trips: dict[str, list] = {}
    for tr in solution.trips:
        object_trips.setdefault(tr.destination_object_id, []).append(tr)

    object_ids = [nid for nid, node in dataset.nodes.items() if node.kind.startswith("object")]
    for oid in object_ids:
        trips = object_trips.get(oid, [])
        obj = dataset.nodes.get(oid)
        if obj is None:
            continue
        plt.figure(figsize=(8, 8))
        for tr in trips:
            pts = []
            for nid in tr.path_nodes_full:
                nd = dataset.nodes.get(nid)
                if nd is None:
                    continue
                pts.append((nd.x, nd.y))
            if len(pts) >= 2:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                plt.plot(xs, ys, linewidth=0.8, alpha=0.45)

        # destination marker
        plt.scatter([obj.x], [obj.y], c="red", s=90, marker="*", label="object")

        used = evaluation.object_volume_used_m3.get(oid, 0.0)
        cap = evaluation.object_volume_capacity_m3.get(oid, 0.0)
        plt.title(f"Object {oid}: used={used:.1f} / cap={cap:.1f} m3, trips={len(trips)}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(maps_dir / f"object_{oid}.png", dpi=160)
        plt.close()

    # 6) per-object MNO->Object flow maps by zone (assigned vs unassigned)
    task_by_id = {t.task_id: t for t in dataset.tasks}
    assigned_task_ids: set[str] = set()
    for tr in solution.trips:
        assigned_task_ids.update(tr.ordered_task_ids)

    # use stable colors for zones
    zone_values = sorted({t.source_zone_num for t in dataset.tasks if t.source_zone_num is not None})
    cmap = plt.get_cmap("tab10")
    zone_color = {z: cmap(i % 10) for i, z in enumerate(zone_values)}
    default_color = (0.35, 0.35, 0.35, 1.0)

    for oid in object_ids:
        obj = dataset.nodes.get(oid)
        if obj is None:
            continue
        tasks_to_object = [t for t in dataset.tasks if t.destination_node_id == oid]
        if not tasks_to_object:
            continue

        # aggregate by (source, zone) to avoid redrawing identical line thousands of times
        grouped: dict[tuple[str, int | None], dict[str, float]] = {}
        for t in tasks_to_object:
            key = (t.source_node_id, t.source_zone_num)
            g = grouped.setdefault(
                key,
                {
                    "assigned_count": 0.0,
                    "unassigned_count": 0.0,
                    "total_count": 0.0,
                },
            )
            g["total_count"] += 1.0
            if t.task_id in assigned_task_ids:
                g["assigned_count"] += 1.0
            else:
                g["unassigned_count"] += 1.0

        plt.figure(figsize=(9, 9))

        # Draw links from source MNO nodes to object.
        for (src_id, zone), g in grouped.items():
            src = dataset.nodes.get(src_id)
            if src is None:
                continue
            color = zone_color.get(zone, default_color)
            x = [src.x, obj.x]
            y = [src.y, obj.y]

            # assigned portion: dense solid
            if g["assigned_count"] > 0:
                lw_assigned = 0.4 + min(3.2, 0.12 * g["assigned_count"])
                plt.plot(
                    x,
                    y,
                    color=color,
                    alpha=0.75,
                    linewidth=lw_assigned,
                    linestyle="-",
                )
            # unassigned portion: transparent dashed
            if g["unassigned_count"] > 0:
                lw_unassigned = 0.3 + min(2.0, 0.1 * g["unassigned_count"])
                plt.plot(
                    x,
                    y,
                    color=color,
                    alpha=0.22,
                    linewidth=lw_unassigned,
                    linestyle="--",
                )

        # Object marker
        plt.scatter([obj.x], [obj.y], c="red", s=130, marker="*", zorder=5)

        # zone-level solved share stats for this object
        zone_stats: dict[int | None, tuple[int, int]] = {}
        for t in tasks_to_object:
            z = t.source_zone_num
            a, total = zone_stats.get(z, (0, 0))
            zone_stats[z] = (a + (1 if t.task_id in assigned_task_ids else 0), total + 1)

        stats_lines = []
        for z in sorted(zone_stats, key=lambda v: (v is None, v)):
            a, total = zone_stats[z]
            share = 100.0 * a / max(total, 1)
            zlbl = "NA" if z is None else str(z)
            stats_lines.append(f"z{zlbl}: {a}/{total} ({share:.1f}%)")
        stats_text = "\n".join(stats_lines[:8])
        if len(stats_lines) > 8:
            stats_text += f"\n... (+{len(stats_lines) - 8} zones)"

        used = float(evaluation.object_volume_used_m3.get(oid, 0.0))
        cap = float(evaluation.object_volume_capacity_m3.get(oid, 0.0))
        fill_pct = (100.0 * used / cap) if cap > 1e-9 else 0.0

        plt.text(
            obj.x,
            obj.y,
            f"OBJ {oid}\nfill: {used:.1f}/{cap:.1f} ({fill_pct:.1f}%)\n{stats_text}",
            fontsize=8,
            ha="left",
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "black", "boxstyle": "round,pad=0.3"},
        )

        # legend: style + zone colors
        legend_items = [
            Line2D([0], [0], color="black", lw=1.8, linestyle="-", alpha=0.8, label="assigned"),
            Line2D([0], [0], color="black", lw=1.4, linestyle="--", alpha=0.35, label="unassigned"),
        ]
        for z in zone_values:
            legend_items.append(
                Line2D([0], [0], color=zone_color[z], lw=2.0, linestyle="-", label=f"zone {z}")
            )

        plt.legend(handles=legend_items, loc="best", fontsize=8)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"MNO -> Object flows by zone: object {oid}")
        plt.tight_layout()
        plt.savefig(flow_maps_dir / f"object_{oid}_zone_flow.png", dpi=170)
        plt.close()

    # 7) unified map with all objects and all MNO->Object links
    global_grouped: dict[tuple[str, int | None, str], dict[str, float]] = {}
    tasks_by_object: dict[str, list] = {}
    for t in dataset.tasks:
        tasks_by_object.setdefault(t.destination_node_id, []).append(t)
        key = (t.source_node_id, t.source_zone_num, t.destination_node_id)
        g = global_grouped.setdefault(
            key,
            {
                "assigned_count": 0.0,
                "unassigned_count": 0.0,
            },
        )
        if t.task_id in assigned_task_ids:
            g["assigned_count"] += 1.0
        else:
            g["unassigned_count"] += 1.0

    all_map_path = flow_maps_dir / "all_objects_zone_flow.png"
    plt.figure(figsize=(12, 11))

    for (src_id, zone, oid), g in global_grouped.items():
        src = dataset.nodes.get(src_id)
        obj = dataset.nodes.get(oid)
        if src is None or obj is None:
            continue
        color = zone_color.get(zone, default_color)
        x = [src.x, obj.x]
        y = [src.y, obj.y]
        if g["assigned_count"] > 0:
            lw_assigned = 0.2 + min(2.6, 0.04 * g["assigned_count"])
            plt.plot(x, y, color=color, alpha=0.55, linewidth=lw_assigned, linestyle="-")
        if g["unassigned_count"] > 0:
            lw_unassigned = 0.15 + min(1.6, 0.03 * g["unassigned_count"])
            plt.plot(x, y, color=color, alpha=0.15, linewidth=lw_unassigned, linestyle="--")

    # all objects + fill share labels
    for oid in object_ids:
        obj = dataset.nodes.get(oid)
        if obj is None:
            continue
        used = float(evaluation.object_volume_used_m3.get(oid, 0.0))
        cap = float(evaluation.object_volume_capacity_m3.get(oid, 0.0))
        fill_pct = (100.0 * used / cap) if cap > 1e-9 else 0.0
        to_obj = tasks_by_object.get(oid, [])
        total = len(to_obj)
        assigned = sum(1 for t in to_obj if t.task_id in assigned_task_ids)
        solved_pct = (100.0 * assigned / total) if total > 0 else 0.0

        plt.scatter([obj.x], [obj.y], c="red", s=120, marker="*", zorder=6)
        plt.text(
            obj.x,
            obj.y,
            f"OBJ {oid}\nfill {fill_pct:.1f}%\nsolved {assigned}/{total} ({solved_pct:.1f}%)",
            fontsize=7,
            ha="left",
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "black", "boxstyle": "round,pad=0.2"},
        )

    legend_items = [
        Line2D([0], [0], color="black", lw=1.8, linestyle="-", alpha=0.8, label="assigned"),
        Line2D([0], [0], color="black", lw=1.4, linestyle="--", alpha=0.35, label="unassigned"),
        Line2D([0], [0], marker="*", color="red", lw=0, markersize=10, label="object"),
    ]
    for z in zone_values:
        legend_items.append(Line2D([0], [0], color=zone_color[z], lw=2.0, linestyle="-", label=f"zone {z}"))
    plt.legend(handles=legend_items, loc="best", fontsize=8)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("All objects: MNO -> Object flows by zone (with object fill share)")
    plt.tight_layout()
    plt.savefig(all_map_path, dpi=180)
    plt.close()

    return {
        "summary": str(summary_path),
        "solver_logs": str(logs_path),
        "trips": str(trips_path),
        "task_transport": str(tasks_path),
        "agent_visit_sequences": str(chains_path),
        "object_maps_dir": str(maps_dir),
        "object_flow_zone_maps_dir": str(flow_maps_dir),
        "all_objects_flow_map": str(all_map_path),
        "distributions_dir": str(dist_dir),
    }
