from pathlib import Path

from volume_core import DnarFlowConfig, DnarFlowVolumeSolver, VolumeDataset


def test_legacy_mass_sandbox_dataset_is_usable_by_dnar_solver() -> None:
    dataset_path = (
        Path(__file__).resolve().parents[1]
        / "storage"
        / "syntetic_data_gap_vrp_solver"
        / "data"
        / "dataset_sandbox_type2.json"
    )
    dataset = VolumeDataset.from_json(dataset_path)

    assert len(dataset.tasks) == 18
    assert len(dataset.agents) == 28
    assert sum(a.is_active and a.depot_node_id is not None for a in dataset.agents) == 28
    assert all(task.volume_raw_m3 > 0 for task in dataset.tasks)
    assert all(cap > 0 for cap in dataset.object_volume_caps.values())

    solution = DnarFlowVolumeSolver(
        DnarFlowConfig(max_runtime_sec=30.0, verbose=False)
    ).solve_checked(dataset)
    evaluation = dataset.evaluate(solution)

    assert evaluation.assigned_tasks == evaluation.total_tasks
    assert evaluation.constraints is not None
    assert evaluation.constraints.all_checks_ok
    assert solution.solver_logs[0].startswith("[dnar_flow_policy_v1] encoded graph")
