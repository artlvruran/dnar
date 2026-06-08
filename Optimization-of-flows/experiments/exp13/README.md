# EXP13: Volume-Core Greedy Batch

Новый core-пайплайн под `stage4_volume_only`:

- `graph json -> VolumeDataset`
- `VolumeSolver (abstract) -> AssignmentSolution`
- `dataset.evaluate(solution) -> ConstraintReport + metrics`
- `reporting.save_solution_artifacts(...)` (логи + картинки)

## Запуск

```bash
PYTHONPATH=src python experiments/exp13/run_exp13_volume_core_batch.py \
  --dataset demo/data/object_volume_feasible_fullfleet/full_all_tasks_all_agents_stage4_volume_only/dataset_real_spb_clean_full_all_tasks_all_agents_stage4_volume_only_covered_only.json \
  --max-runtime-sec 120 \
  --top-k-agents 30 \
  --top-k-destinations 3 \
  --max-tasks-in-trip 32
```

Для smoke:

```bash
PYTHONPATH=src python experiments/exp13/run_exp13_volume_core_batch.py --task-limit 1200 --max-runtime-sec 45
```

## Выход

`experiments/local/exp13_volume_core_batch/run_<timestamp>/`:

- `summary.json`, `summary.csv`
- `solver_logs.txt`
- `trips.json`
- `task_transport_logs.csv`
- `agent_visit_sequences.json`
- `object_maps/object_<id>.png`
- `distributions/*.png`
- `index.json`
