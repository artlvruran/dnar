# EXP4 Report: Day Load Profiles

Generated: 2026-04-02T18:29:57

## Dataset `u50`

| dataset_tag   |   target_utilization | algorithm             | feasible   | all_checks_ok   |   assigned_routes |   unassigned_tasks | assignment_coverage_pct   |   active_agents |   runtime_sec | solver_error                                                                                 |
|:--------------|---------------------:|:----------------------|:-----------|:----------------|------------------:|-------------------:|:--------------------------|----------------:|--------------:|:---------------------------------------------------------------------------------------------|
| u50           |                  0.5 | real_stochastic_grasp | False      | False           |                 0 |               1587 |                           |               0 |         1.023 | TypeError: run_real_stochastic_grasp() got an unexpected keyword argument '__capacity_scale' |

## Dataset `u90`

| dataset_tag   |   target_utilization | algorithm             | feasible   | all_checks_ok   |   assigned_routes |   unassigned_tasks | assignment_coverage_pct   |   active_agents |   runtime_sec | solver_error                                                                                 |
|:--------------|---------------------:|:----------------------|:-----------|:----------------|------------------:|-------------------:|:--------------------------|----------------:|--------------:|:---------------------------------------------------------------------------------------------|
| u90           |                  0.9 | real_stochastic_grasp | False      | False           |                 0 |               3737 |                           |               0 |         0.972 | TypeError: run_real_stochastic_grasp() got an unexpected keyword argument '__capacity_scale' |

## Overall summary

| algorithm             |   runs |   feasible_runs |   avg_runtime_sec |   avg_coverage_pct |
|:----------------------|-------:|----------------:|------------------:|-------------------:|
| real_stochastic_grasp |      2 |               0 |            0.9975 |                nan |