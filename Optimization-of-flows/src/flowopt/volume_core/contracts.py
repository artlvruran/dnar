from __future__ import annotations

from typing import Iterable

from .models import AssignmentSolution, TripPlan


def assert_dataset_input(dataset: object) -> None:
    from .dataset import VolumeDataset

    assert isinstance(dataset, VolumeDataset), (
        "Solver input must be VolumeDataset. "
        f"Got {type(dataset).__name__}"
    )
    assert isinstance(dataset.tasks, list), "dataset.tasks must be list"
    assert isinstance(dataset.agents, list), "dataset.agents must be list"
    assert isinstance(dataset.object_volume_caps, dict), "dataset.object_volume_caps must be dict"


def assert_solution_output(solution: object, *, dataset_path: str | None = None) -> AssignmentSolution:
    assert isinstance(solution, AssignmentSolution), (
        "Solver output must be AssignmentSolution. "
        f"Got {type(solution).__name__}"
    )
    assert isinstance(solution.algorithm, str) and solution.algorithm, "solution.algorithm must be non-empty str"
    assert isinstance(solution.dataset_path, str) and solution.dataset_path, "solution.dataset_path must be non-empty str"
    assert isinstance(solution.trips, tuple), "solution.trips must be tuple[TripPlan, ...]"
    assert isinstance(solution.unassigned_task_ids, tuple), "solution.unassigned_task_ids must be tuple[str, ...]"
    assert isinstance(solution.runtime_sec, (int, float)), "solution.runtime_sec must be numeric"
    assert isinstance(solution.solver_logs, tuple), "solution.solver_logs must be tuple[str, ...]"
    if dataset_path is not None:
        assert solution.dataset_path == dataset_path, (
            "solution.dataset_path must match input dataset path: "
            f"{solution.dataset_path} != {dataset_path}"
        )
    _assert_trip_plans(solution.trips)
    return solution


def _assert_trip_plans(trips: Iterable[TripPlan]) -> None:
    for tr in trips:
        assert isinstance(tr, TripPlan), f"trip must be TripPlan, got {type(tr).__name__}"
        assert isinstance(tr.agent_id, str) and tr.agent_id, "trip.agent_id must be non-empty str"
        assert isinstance(tr.ordered_task_ids, tuple), "trip.ordered_task_ids must be tuple[str, ...]"
        assert isinstance(tr.visit_nodes, tuple) and len(tr.visit_nodes) >= 3, "trip.visit_nodes must contain at least depot-source-object-depot path"
        assert isinstance(tr.total_km, (int, float)) and tr.total_km >= 0, "trip.total_km must be >= 0"
        assert isinstance(tr.total_hours, (int, float)) and tr.total_hours >= 0, "trip.total_hours must be >= 0"
        assert isinstance(tr.payload_effective_volume_m3, (int, float)) and tr.payload_effective_volume_m3 >= 0, (
            "trip.payload_effective_volume_m3 must be >= 0"
        )

