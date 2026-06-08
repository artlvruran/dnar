# EXP6: Enriched Batched Greedy Stress Test

## Что тестируем

Стресс-тест самого быстрого enriched-алгоритма:

- solver: `enriched_batched_greedy_v1`
- legacy alias в API: `solve_enriched_gap_vrp`
- корректное имя в API: `solve_enriched_batched_greedy`

Важно: это **не** классический GAP+VRP decomposition.
Это batched greedy assignment с trip-level оценкой рейса и ограничениями на:

- совместимость контейнеров,
- zone matching,
- mass/volume,
- daily km/h limits.

## Вход

По умолчанию используются sweep-датасеты:

`demo/data/object_mass_feasible_fullfleet/container_full_split_all_agents/sweeps_task_agent_5pct`

Паттерн:

`dataset_real_spb_clean_full_split_by_containers_all_agents_with_distances_t*_a100.json`

## Запуск

```bash
PYTHONPATH=src python experiments/exp6/run_exp6_enriched_batched_stress.py
```

## Полезные параметры

```bash
PYTHONPATH=src python experiments/exp6/run_exp6_enriched_batched_stress.py \
  --timeout-sec 180 \
  --top-k-agents 20 \
  --balance-penalty 0.03 \
  --random-seed 42
```

## Выход

Артефакты сохраняются в `experiments/exp6/local`:

- `exp6_enriched_batched_stress_<timestamp>.json`
- `exp6_enriched_batched_stress_<timestamp>.csv`
- `exp6_enriched_batched_stress_latest.csv`
- `plots/coverage_runtime.png`
- `plots/assigned_unassigned.png`
- `plots/summary_with_coverage.csv`

