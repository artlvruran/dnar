from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
TEST_TMP_ROOT = REPO_ROOT / ".test_tmp"
os.environ.setdefault("MPLCONFIGDIR", str((TEST_TMP_ROOT / "mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((TEST_TMP_ROOT / "xdg-cache").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import flowopt.core as core
import flowopt.genetic_solver_components as ga
from flowopt.backend.io import load_dataset
from flowopt.solvers import real_gap_vrp_solver, real_genetic_solver, real_milp_solver


DATASET_PATH = (
    REPO_ROOT
    / "src"
    / "data"
    / "real"
    / "day_load_profiles"
    / "u50"
    / "fast_uniform"
    / "dataset_real_spb_day_u50_fast_f10.json"
)
ARTIFACT_ROOT = REPO_ROOT / ".test_tmp" / "u50_fast_uniform"


@dataclass
class CaseArtifacts:
    algorithm: str
    out_dir: Path
    metrics_path: Path
    schedule_map_path: Path
    trajectories_path: Path
    report: dict[str, Any]


def patch_profile_limits_from_dataset(dataset_path: Path) -> None:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    profiles = payload.get("metadata", {}).get("vehicle_profiles", {})
    for vt, p in profiles.items():
        km = float(p.get("max_daily_km", core.MAX_DAILY_KM_BY_TYPE.get(vt, 130.0)))
        hh = float(p.get("max_shift_hours", core.MAX_SHIFT_HOURS_BY_TYPE.get(vt, 10.0)))
        sp = float(p.get("avg_speed_kmph", core.AVG_SPEED_KMPH_BY_TYPE.get(vt, 24.0)))
        core.MAX_DAILY_KM_BY_TYPE[vt] = km
        core.MAX_SHIFT_HOURS_BY_TYPE[vt] = hh
        core.AVG_SPEED_KMPH_BY_TYPE[vt] = sp
        ga.MAX_DAILY_KM_BY_TYPE[vt] = km
        ga.MAX_SHIFT_HOURS_BY_TYPE[vt] = hh
        ga.AVG_SPEED_KMPH_BY_TYPE[vt] = sp


def _aggregate_state_metrics(states: dict[str, core.AgentState]) -> dict[str, float | None]:
    active_states = [state for state in states.values() if state.task_ids]
    if not active_states:
        return {
            "total_km": None,
            "deadhead_km": None,
            "deadhead_share_pct": None,
            "total_hours": None,
        }
    total_km = sum(state.total_km for state in active_states)
    deadhead_km = sum(state.deadhead_km for state in active_states)
    total_hours = sum(state.total_hours for state in active_states)
    deadhead_share = (100.0 * deadhead_km / total_km) if total_km > 0 else None
    return {
        "total_km": round(total_km, 3),
        "deadhead_km": round(deadhead_km, 3),
        "deadhead_share_pct": round(deadhead_share, 3) if deadhead_share is not None else None,
        "total_hours": round(total_hours, 3),
    }


def _compute_activation_kpis(dataset, states: dict[str, core.AgentState]) -> dict[str, float | int | None]:
    total_agents = len(dataset.fleet.agents)
    active_states = [state for state in states.values() if state.task_ids]
    active_agents = len(active_states)
    active_share_pct = (100.0 * active_agents / total_agents) if total_agents > 0 else None

    task_by_id = {task.task_id: task for task in dataset.tasks}
    payload_utils: list[float] = []
    volume_utils: list[float] = []

    for state in active_states:
        if state.capacity_tons > 0:
            payload_utils.append(100.0 * getattr(state, "peak_load_tons", 0.0) / state.capacity_tons)

        raw_limit = getattr(state, "max_raw_volume_m3", 0.0)
        legs = getattr(state, "movement_legs", ())
        if raw_limit > 0 and legs:
            peak_raw_volume = 0.0
            for leg in legs:
                load_raw = 0.0
                for task_id in getattr(leg, "carrying_task_ids", ()):
                    task = task_by_id.get(task_id)
                    if task is not None:
                        load_raw += task.volume_raw_m3
                peak_raw_volume = max(peak_raw_volume, load_raw)
            volume_utils.append(100.0 * peak_raw_volume / raw_limit)

    return {
        "fleet_agents_total": total_agents,
        "active_agents": active_agents,
        "active_agents_share_pct": round(active_share_pct, 3) if active_share_pct is not None else None,
        "active_payload_peak_utilization_pct_avg": round(sum(payload_utils) / len(payload_utils), 3)
        if payload_utils else None,
        "active_payload_peak_utilization_pct_max": round(max(payload_utils), 3) if payload_utils else None,
        "active_volume_peak_utilization_pct_avg": round(sum(volume_utils) / len(volume_utils), 3)
        if volume_utils else None,
        "active_volume_peak_utilization_pct_max": round(max(volume_utils), 3) if volume_utils else None,
    }


def _build_checks(
    *,
    dataset,
    graph,
    cache,
    routes,
    limit_violations: list[dict[str, Any]],
    unassigned: list[str],
) -> dict[str, Any]:
    reachability = core.check_reachability(graph, dataset, cache)
    route_task_ids = {task_id for route in routes for task_id in route.task_ids}
    base_task_ids = {task.task_id for task in dataset.tasks}
    same_task_space = route_task_ids.issubset(base_task_ids)

    solved_dataset_valid = False
    if not unassigned and same_task_space:
        try:
            _ = core.build_solution_dataset(dataset, routes)
            solved_dataset_valid = True
        except Exception:
            solved_dataset_valid = False

    if same_task_space:
        mno_coverage = core.mno_scope_coverage_summary(dataset, routes)
    else:
        mno_coverage = {
            "all_eligible_mno_covered": len(unassigned) == 0,
            "eligible_mno_count": None,
            "covered_eligible_mno_count": None,
        }

    checks = core.constraints_check_summary(
        solved_dataset_valid=solved_dataset_valid,
        limit_violations=limit_violations,
        reachability=reachability,
        unassigned=unassigned,
        mno_coverage=mno_coverage,
    )
    checks["metric_task_space"] = "dataset_reference" if same_task_space else "solver_generated_mismatch"
    checks["task_space_match"] = bool(same_task_space)
    return checks


def _run_solver(algorithm: str):
    dataset, payload = load_dataset(DATASET_PATH)
    graph = core.build_nx_graph(dataset)
    cache: dict[tuple[str, str], tuple[list[str], float] | None] = {}

    t0 = perf_counter()
    if algorithm == "real_gap_vrp":
        result = real_gap_vrp_solver.solve_real_gap_vrp(
            dataset=dataset,
            payload=payload,
            graph=graph,
            cache=cache,
            step1_method="dataset",
            gap_iter=20,
            use_repair=True,
            show_progress=False,
            verbose=False,
        )
    elif algorithm == "real_milp":
        result = real_milp_solver.solve_real_milp(
            dataset=dataset,
            payload=payload,
            graph=graph,
            cache=cache,
            time_limit_sec=90,
            unassigned_penalty=1e7,
            show_progress=False,
            progress_hook=None,
        )
    elif algorithm == "real_genetic":
        result = real_genetic_solver.solve_real_genetic(
            dataset=dataset,
            payload=payload,
            graph=graph,
            cache=cache,
            population_size=20,
            generations=20,
            elite_size=3,
            seed=42,
            show_progress=False,
            progress_hook=None,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    elapsed = perf_counter() - t0
    return dataset, graph, cache, result, elapsed


def _build_report(algorithm: str, dataset, graph, cache, result, elapsed: float) -> dict[str, Any]:
    central = core.evaluate_solution_metrics(
        dataset=dataset,
        routes=result.routes,
        unassigned=result.unassigned,
    )
    checks = _build_checks(
        dataset=dataset,
        graph=graph,
        cache=cache,
        routes=result.routes,
        limit_violations=result.limit_violations,
        unassigned=result.unassigned,
    )
    summary = _aggregate_state_metrics(result.states)
    activation = _compute_activation_kpis(dataset, result.states)
    all_checks_ok = bool(checks.get("all_checks_ok", False))
    return {
        "algorithm": algorithm,
        "feasible": bool(result.feasible and all_checks_ok),
        "all_checks_ok": all_checks_ok,
        "assigned_routes": int(result.n_assigned),
        "unassigned_tasks": int(result.n_unassigned),
        "active_agents": int(result.active_agents),
        "transport_work_ton_km": central.get("TR"),
        "total_km": summary["total_km"],
        "deadhead_km": summary["deadhead_km"],
        "deadhead_share_pct": summary["deadhead_share_pct"],
        "total_hours": summary["total_hours"],
        "runtime_sec": round(elapsed, 3),
        "fleet_agents_total": activation["fleet_agents_total"],
        "active_agents_share_pct": activation["active_agents_share_pct"],
        "active_payload_peak_utilization_pct_avg": activation["active_payload_peak_utilization_pct_avg"],
        "active_payload_peak_utilization_pct_max": activation["active_payload_peak_utilization_pct_max"],
        "active_volume_peak_utilization_pct_avg": activation["active_volume_peak_utilization_pct_avg"],
        "active_volume_peak_utilization_pct_max": activation["active_volume_peak_utilization_pct_max"],
        "metric_task_space": central.get("metric_task_space"),
        "task_space_match": central.get("task_space_match"),
        "checks": checks,
        "central_metrics": central,
    }


def _build_agent_trajectories(dataset, result) -> list[dict[str, Any]]:
    task_by_id = {task.task_id: task for task in dataset.tasks}
    rows: list[dict[str, Any]] = []
    for agent_id, state in sorted(result.states.items()):
        if not state.task_ids:
            continue
        rows.append(
            {
                "agent_id": agent_id,
                "task_ids": list(state.task_ids),
                "route_ids": list(state.route_ids),
                "peak_load_tons": round(getattr(state, "peak_load_tons", 0.0), 3),
                "payload_peak_utilization_pct": round(
                    (100.0 * getattr(state, "peak_load_tons", 0.0) / state.capacity_tons), 3
                )
                if state.capacity_tons > 0 else None,
                "volume_peak_utilization_pct": None,
                "true_transport_work_ton_km": round(getattr(state, "true_transport_work_ton_km", 0.0), 3),
                "stop_events": [
                    {
                        "node_id": event.node_id,
                        "action": event.action,
                        "task_ids": list(event.task_ids),
                        "load_after_tons": round(event.load_after_tons, 3),
                    }
                    for event in getattr(state, "stop_events", ())
                ],
                "movement_legs": [
                    {
                        "from_node_id": leg.from_node_id,
                        "to_node_id": leg.to_node_id,
                        "distance_km": round(leg.distance_km, 3),
                        "loaded_mass_tons": round(leg.loaded_mass_tons, 3),
                        "carrying_task_ids": list(leg.carrying_task_ids),
                        "path": list(leg.path),
                    }
                    for leg in getattr(state, "movement_legs", ())
                ],
            }
        )
        raw_limit = getattr(state, "max_raw_volume_m3", 0.0)
        if raw_limit > 0 and rows[-1]["movement_legs"]:
            peak_raw_volume = 0.0
            for leg in rows[-1]["movement_legs"]:
                leg_raw = 0.0
                for task_id in leg["carrying_task_ids"]:
                    task = task_by_id.get(task_id)
                    if task is not None:
                        leg_raw += task.volume_raw_m3
                peak_raw_volume = max(peak_raw_volume, leg_raw)
            rows[-1]["volume_peak_utilization_pct"] = round(100.0 * peak_raw_volume / raw_limit, 3)
    return rows


def _render_quick_schedule(dataset, result, report: dict[str, Any], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nodes = dataset.graph.nodes

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 3.8])
    ax_meta = fig.add_subplot(gs[0, 0])
    ax_map = fig.add_subplot(gs[0, 1])

    # Left panel: key metrics.
    ax_meta.axis("off")
    lines = [
        f"algorithm: {report['algorithm']}",
        f"feasible: {report['feasible']}",
        f"all_checks_ok: {report['all_checks_ok']}",
        f"metric_task_space: {report['metric_task_space']}",
        f"fleet_agents_total: {report['fleet_agents_total']}",
        f"assigned_routes: {report['assigned_routes']}",
        f"unassigned_tasks: {report['unassigned_tasks']}",
        f"active_agents: {report['active_agents']}",
        f"active_agents_share_pct: {report['active_agents_share_pct']}",
        f"active_payload_peak_util_avg_pct: {report['active_payload_peak_utilization_pct_avg']}",
        f"active_payload_peak_util_max_pct: {report['active_payload_peak_utilization_pct_max']}",
        f"active_volume_peak_util_avg_pct: {report['active_volume_peak_utilization_pct_avg']}",
        f"active_volume_peak_util_max_pct: {report['active_volume_peak_utilization_pct_max']}",
        f"TR (ton-km): {report['transport_work_ton_km']}",
        f"total_km: {report['total_km']}",
        f"deadhead_km: {report['deadhead_km']}",
        f"deadhead_share_pct: {report['deadhead_share_pct']}",
        f"total_hours: {report['total_hours']}",
        f"runtime_sec: {report['runtime_sec']}",
    ]
    ax_meta.text(
        0.02,
        0.98,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=10,
        family="monospace",
    )

    # Fast map: no heavy edge layer, only task points + planned movement legs.
    source_xy = [(nodes[t.source_node_id].x, nodes[t.source_node_id].y) for t in dataset.tasks]
    dest_xy = [(nodes[t.destination_node_id].x, nodes[t.destination_node_id].y) for t in dataset.tasks]
    if source_xy:
        ax_map.scatter([p[0] for p in source_xy], [p[1] for p in source_xy], s=8, c="#666666", alpha=0.45, label="sources")
    if dest_xy:
        ax_map.scatter([p[0] for p in dest_xy], [p[1] for p in dest_xy], s=10, c="#228b22", alpha=0.5, marker="^", label="destinations")

    cmap = plt.get_cmap("tab20")
    active_states = [(agent_id, state) for agent_id, state in sorted(result.states.items()) if state.task_ids]
    for idx, (agent_id, state) in enumerate(active_states):
        color = cmap(idx % 20)
        for leg in getattr(state, "movement_legs", ()):
            xs = [nodes[nid].x for nid in leg.path if nid in nodes]
            ys = [nodes[nid].y for nid in leg.path if nid in nodes]
            if len(xs) < 2:
                continue
            linestyle = "-" if leg.loaded_mass_tons > 1e-9 else "--"
            alpha = 0.95 if leg.loaded_mass_tons > 1e-9 else 0.65
            ax_map.plot(xs, ys, color=color, linewidth=1.35, alpha=alpha, linestyle=linestyle)

        for event in getattr(state, "stop_events", ()):
            if event.node_id not in nodes:
                continue
            node = nodes[event.node_id]
            marker = {"pickup": "o", "unload": "^", "return_depot": "s"}.get(event.action, "x")
            ax_map.scatter([node.x], [node.y], s=24, c=[color], marker=marker, edgecolors="black", linewidths=0.25, alpha=0.9)

        if state.depot_node in nodes:
            depot = nodes[state.depot_node]
            ax_map.scatter([depot.x], [depot.y], s=50, c=[color], marker="s", edgecolors="black", linewidths=0.4)

    # Fallback for solvers without leg traces.
    if not active_states:
        routes_by_agent = core.group_routes_by_agent(result.routes)
        for idx, (agent_id, agent_routes) in enumerate(sorted(routes_by_agent.items())):
            color = cmap(idx % 20)
            for route in agent_routes:
                xs = [nodes[nid].x for nid in route.path if nid in nodes]
                ys = [nodes[nid].y for nid in route.path if nid in nodes]
                if len(xs) < 2:
                    continue
                ax_map.plot(xs, ys, color=color, linewidth=1.2, alpha=0.9)

            state = result.states.get(agent_id)
            if state is not None and state.depot_node in nodes and state.task_ids:
                depot = nodes[state.depot_node]
                ax_map.scatter([depot.x], [depot.y], s=45, c=[color], marker="s", edgecolors="black", linewidths=0.35)

    ax_map.set_title("Quick schedule map (coordinates only)")
    ax_map.set_xlabel("x")
    ax_map.set_ylabel("y")
    ax_map.grid(True, alpha=0.15)
    ax_map.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_case(algorithm: str) -> CaseArtifacts:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    patch_profile_limits_from_dataset(DATASET_PATH)
    dataset, graph, cache, result, elapsed = _run_solver(algorithm)
    report = _build_report(algorithm, dataset, graph, cache, result, elapsed)

    out_dir = ARTIFACT_ROOT / algorithm
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    schedule_map_path = out_dir / "schedule_map.png"
    trajectories_path = out_dir / "agent_trajectories.json"
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    trajectories = _build_agent_trajectories(dataset, result)
    trajectories_path.write_text(json.dumps(trajectories, ensure_ascii=False, indent=2), encoding="utf-8")
    _render_quick_schedule(dataset, result, report, schedule_map_path)

    return CaseArtifacts(
        algorithm=algorithm,
        out_dir=out_dir,
        metrics_path=metrics_path,
        schedule_map_path=schedule_map_path,
        trajectories_path=trajectories_path,
        report=report,
    )
