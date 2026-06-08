from __future__ import annotations

import unittest

try:
    from .common import run_case
except ImportError:  # pragma: no cover - fallback for unittest discover by path
    from tests.integration.u50_fast_uniform.common import run_case


class TestRealGapVRPU50FastUniform(unittest.TestCase):
    def test_real_gap_vrp_end_to_end(self) -> None:
        artifacts = run_case("real_gap_vrp")
        report = artifacts.report

        self.assertTrue(report["feasible"], report)
        self.assertTrue(report["all_checks_ok"], report)
        self.assertEqual(report["metric_task_space"], "dataset_reference", report)
        self.assertEqual(report["unassigned_tasks"], 0, report)
        self.assertGreater(report["assigned_routes"], 0, report)
        self.assertIsNotNone(report["transport_work_ton_km"], report)
        self.assertLess(report["runtime_sec"], 180.0, report)

        self.assertTrue(artifacts.metrics_path.exists())
        self.assertTrue(artifacts.schedule_map_path.exists())
        self.assertTrue(artifacts.trajectories_path.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
