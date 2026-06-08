from __future__ import annotations

from abc import ABC, abstractmethod

from .contracts import assert_dataset_input, assert_solution_output
from .models import AssignmentSolution


class VolumeSolver(ABC):
    @abstractmethod
    def solve(self, dataset: "VolumeDataset") -> AssignmentSolution:
        raise NotImplementedError

    def solve_checked(self, dataset: "VolumeDataset") -> AssignmentSolution:
        assert_dataset_input(dataset)
        solution = self.solve(dataset)
        return assert_solution_output(solution, dataset_path=str(dataset.dataset_path))
