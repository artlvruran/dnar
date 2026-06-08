# EXP4 Report: Day Load Profiles

Generated: 2026-04-02T18:20:01

## Dataset `u50`

| dataset_tag   |   target_utilization | algorithm             | feasible   | all_checks_ok   |   assigned_routes |   unassigned_tasks |   assignment_coverage_pct |   active_agents |   runtime_sec | solver_error   |
|:--------------|---------------------:|:----------------------|:-----------|:----------------|------------------:|-------------------:|--------------------------:|----------------:|--------------:|:---------------|
| u50           |                  0.5 | real_stochastic_grasp | False      | False           |              1077 |                510 |                    67.864 |             626 |         6.076 |                |
| u50           |                  0.5 | real_stochastic_rr    | False      | False           |               542 |               1045 |                    34.152 |             311 |       302.826 |                |

## Dataset `u90`

| dataset_tag   |   target_utilization | algorithm             | feasible   | all_checks_ok   |   assigned_routes |   unassigned_tasks |   assignment_coverage_pct |   active_agents |   runtime_sec | solver_error   |
|:--------------|---------------------:|:----------------------|:-----------|:----------------|------------------:|-------------------:|--------------------------:|----------------:|--------------:|:---------------|
| u90           |                  0.9 | real_stochastic_grasp | False      | False           |              2175 |               1562 |                    58.202 |             626 |         5.438 |                |
| u90           |                  0.9 | real_stochastic_rr    | False      | False           |              1336 |               2401 |                    35.751 |             347 |       302.813 |                |

## Overall summary

| algorithm             |   runs |   feasible_runs |   avg_runtime_sec |   avg_coverage_pct |
|:----------------------|-------:|----------------:|------------------:|-------------------:|
| real_stochastic_grasp |      2 |               0 |             5.757 |            63.033  |
| real_stochastic_rr    |      2 |               0 |           302.82  |            34.9515 |