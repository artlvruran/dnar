# EXP4 Report (Recovered)

Generated: 2026-04-02T17:40:32

## Dataset `u50`

| dataset_tag   |   target_utilization | algorithm             | feasible   | all_checks_ok   |   assigned_routes |   unassigned_tasks |   assignment_coverage_pct |   active_agents |   runtime_sec | solver_error                        |
|:--------------|---------------------:|:----------------------|:-----------|:----------------|------------------:|-------------------:|--------------------------:|----------------:|--------------:|:------------------------------------|
| u50           |                  0.5 | real_gap_vrp          | False      | False           |                 0 |               1587 |                   nan     |               0 |       180.006 | TimeoutError: exceeded timeout 180s |
| u50           |                  0.5 | real_milp             | False      | False           |                 0 |               1587 |                   nan     |               0 |       180.006 | TimeoutError: exceeded timeout 180s |
| u50           |                  0.5 | real_genetic          | False      | False           |                 0 |               1587 |                   nan     |               0 |       180.004 | TimeoutError: exceeded timeout 180s |
| u50           |                  0.5 | real_stochastic_grasp | False      | False           |               512 |               1075 |                    32.262 |             282 |         4.792 |                                     |
| u50           |                  0.5 | real_stochastic_rr    | False      | False           |               509 |               1078 |                    32.073 |             283 |       122.239 |                                     |

## Dataset `u90`

| dataset_tag   |   target_utilization | algorithm             | feasible   | all_checks_ok   |   assigned_routes |   unassigned_tasks |   assignment_coverage_pct |   active_agents |   runtime_sec | solver_error                        |
|:--------------|---------------------:|:----------------------|:-----------|:----------------|------------------:|-------------------:|--------------------------:|----------------:|--------------:|:------------------------------------|
| u90           |                  0.9 | real_gap_vrp          | False      | False           |                 0 |               3737 |                   nan     |               0 |       180.004 | TimeoutError: exceeded timeout 180s |
| u90           |                  0.9 | real_milp             | False      | False           |                 0 |               3737 |                   nan     |               0 |       180.004 | TimeoutError: exceeded timeout 180s |
| u90           |                  0.9 | real_genetic          | False      | False           |                 0 |               3737 |                   nan     |               0 |       180.009 | TimeoutError: exceeded timeout 180s |
| u90           |                  0.9 | real_stochastic_grasp | False      | False           |              1243 |               2494 |                    33.262 |             315 |         6.183 |                                     |
| u90           |                  0.9 | real_stochastic_rr    | False      | False           |              1249 |               2488 |                    33.423 |             313 |       122.943 |                                     |

## Overall summary

| algorithm             |   runs |   avg_runtime_sec |   avg_coverage_pct |
|:----------------------|-------:|------------------:|-------------------:|
| real_gap_vrp          |      2 |          180.005  |            nan     |
| real_genetic          |      2 |          180.006  |            nan     |
| real_milp             |      2 |          180.005  |            nan     |
| real_stochastic_grasp |      2 |            5.4875 |             32.762 |
| real_stochastic_rr    |      2 |          122.591  |             32.748 |