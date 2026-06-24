from .contracts import assert_dataset_input, assert_solution_output
from .dataset import VolumeDataset
from .greedy_batch_solver import GreedyBatchConfig, GreedyBatchVolumeSolver
from .dnar_solver import DnarFlowConfig, DnarFlowVolumeSolver
from .models import AssignmentSolution, ConstraintReport, EvaluationResult
from .solver_base import VolumeSolver
from .three_algorithms import (
    VolumeGapVRPLikeSolver,
    VolumeGapVRPStochasticSolver,
    VolumeGeneticLikeSolver,
    VolumeGeneticStochasticSolver,
    VolumeMilpStochasticSolver,
    VolumeMilpLikeSolver,
)

__all__ = [
    "VolumeDataset",
    "assert_dataset_input",
    "assert_solution_output",
    "VolumeSolver",
    "GreedyBatchConfig",
    "GreedyBatchVolumeSolver",
    "DnarFlowConfig",
    "DnarFlowVolumeSolver",
    "AssignmentSolution",
    "ConstraintReport",
    "EvaluationResult",
    "save_solution_artifacts",
    "VolumeGapVRPLikeSolver",
    "VolumeGapVRPStochasticSolver",
    "VolumeMilpLikeSolver",
    "VolumeMilpStochasticSolver",
    "VolumeGeneticLikeSolver",
    "VolumeGeneticStochasticSolver",
]


def __getattr__(name: str):
    if name == "save_solution_artifacts":
        from .reporting import save_solution_artifacts

        return save_solution_artifacts
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
