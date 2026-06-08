from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import random
import sys
from typing import Any


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "simple_solver.py").exists() and (candidate / "data").exists():
            return candidate
    raise RuntimeError("Repo root not found")


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "data") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "data"))

import simple_solver_components as core_solver
from dataset import dataset_from_dict


@dataclass
class DayResult:
    day_index: int
    solver_status: str
    assigned_routes: int
    unassigned_tasks: int
    active_agents: int
    transport_work_ton_km: float | None
    all_checks_ok: bool
    output_dir: Path


class DayPayloadFactory:
    """Create one-day payloads from a base sandbox dataset."""

    def __init__(self, base_payload: dict[str, Any]) -> None:
        self.base_payload = base_payload

    def build(self, day_index: int, *, seed: int, mass_noise: float) -> dict[str, Any]:
        rng = random.Random(seed + day_index * 10007)
        payload = deepcopy(self.base_payload)

        mass_by_source: dict[str, float] = defaultdict(float)
        for task in payload["tasks"]:
            base_mass = float(task["mass_tons"])
            factor = 1.0 + rng.uniform(-mass_noise, mass_noise)
            day_mass = max(0.05, base_mass * factor)
            task["mass_tons"] = round(day_mass, 3)
            mass_by_source[task["source_node_id"]] += day_mass

        for node in payload["graph"]["nodes"]:
            if node["kind"] == "mno":
                node["daily_mass_tons"] = round(mass_by_source.get(node["node_id"], 0.0), 3)

        metadata = payload.setdefault("metadata", {})
        metadata["day_index"] = day_index
        metadata["name"] = f"{metadata.get('name', 'sandbox')}_day_{day_index:03d}"
        metadata["mass_noise"] = mass_noise
        metadata["seed"] = seed
        return payload


class MultiDaySandboxRunner:
    """Notebook-focused runner with minimal API: preprocess + solve + optional plots."""

    def __init__(self, base_dataset_path: Path, output_root: Path) -> None:
        self.base_dataset_path = base_dataset_path
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)

        self.base_payload = json.loads(self.base_dataset_path.read_text(encoding="utf-8"))
        self.factory = DayPayloadFactory(self.base_payload)

    def solve_day(self, day_payload: dict[str, Any], *, render_map: bool) -> DayResult:
        day_index = int(day_payload.get("metadata", {}).get("day_index", 0))
        day_dir = self.output_root / f"day_{day_index:03d}"
        day_dir.mkdir(parents=True, exist_ok=True)

        dataset = dataset_from_dict(day_payload)
        dataset.validate()

        graph = core_solver.build_nx_graph(dataset)
        cache: dict[tuple[str, str], tuple[list[str], float] | None] = {}
        reachability = core_solver.check_reachability(graph, dataset, cache)
        routes, states, unassigned = core_solver.assign_tasks_greedy(dataset, graph, day_payload, cache)

        solver_status = "infeasible"
        limit_violations: list[dict[str, Any]] = []
        transport_work = None
        solved_dataset_valid = False

        if not unassigned:
            solved_dataset = core_solver.build_solution_dataset(dataset, routes)
            solved_dataset_valid = True
            limit_violations = core_solver.validate_daily_limits(states)
            if not limit_violations:
                solver_status = "feasible"
            transport_work = round(solved_dataset.transport_work(), 3)

        checks = core_solver.constraints_check_summary(
            solved_dataset_valid=solved_dataset_valid,
            limit_violations=limit_violations,
            reachability=reachability,
            unassigned=unassigned,
            mno_coverage=core_solver.mno_scope_coverage_summary(dataset, routes),
        )

        # Save only notebook-required artifacts (no markdown side artifacts).
        plan = core_solver.build_solution_plan(routes, dataset, states)
        (day_dir / "solution_plan.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (day_dir / "feasibility_report.json").write_text(
            json.dumps(
                {
                    "solver_status": solver_status,
                    "constraints_check_summary": checks,
                    "assigned_routes": len(routes),
                    "unassigned_tasks": unassigned,
                    "active_agents": sum(1 for state in states.values() if state.task_ids),
                    "transport_work_ton_km": transport_work,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        if render_map:
            core_solver.render_solution_map(dataset, graph, cache, routes, states, day_dir / "solution_map.png")
            core_solver.render_utilization(states, day_dir / "agent_utilization.png")

        return DayResult(
            day_index=day_index,
            solver_status=solver_status,
            assigned_routes=len(routes),
            unassigned_tasks=len(unassigned),
            active_agents=sum(1 for state in states.values() if state.task_ids),
            transport_work_ton_km=transport_work,
            all_checks_ok=bool(checks.get("all_checks_ok", False)),
            output_dir=day_dir,
        )

    def run(
        self,
        *,
        days: int,
        seed: int = 42,
        mass_noise: float = 0.08,
        render_first_day: bool = True,
    ) -> list[DayResult]:
        results: list[DayResult] = []
        for day in range(1, days + 1):
            payload = self.factory.build(day, seed=seed, mass_noise=mass_noise)
            result = self.solve_day(payload, render_map=(render_first_day and day == 1))
            results.append(result)
        return results
