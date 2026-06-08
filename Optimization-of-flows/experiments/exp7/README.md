# EXP7: Enriched MILP Decomposed Stress Test

## Идея

Рядом с `EXP6` (batched greedy) запускаем более "точный" MILP-подход, но с препарированием данных:

1. Декомпозиция по зонам (`source_zone_num` / `agent.zone_num`).
2. Бандлинг задач внутри одинаковых `(source, destination, container_type, compatibility)` групп.
3. MILP на бандлах (а не на каждой задаче напрямую).

Цель: поднять покрытие задач/объема при сохранении ограничений и приемлемом времени.

## Запуск

```bash
PYTHONPATH=src python experiments/exp7/run_exp7_enriched_milp_decomp_stress.py
```

## Примеры параметров

Максимизировать число задач:

```bash
PYTHONPATH=src python experiments/exp7/run_exp7_enriched_milp_decomp_stress.py \
  --objective tasks \
  --time-limit-sec-per-zone 25 \
  --max-pairs-per-bundle 90
```

Максимизировать объем:

```bash
PYTHONPATH=src python experiments/exp7/run_exp7_enriched_milp_decomp_stress.py \
  --objective volume \
  --time-limit-sec-per-zone 25 \
  --max-pairs-per-bundle 90
```

## Выход

`experiments/exp7/local`:

- `exp7_enriched_milp_decomp_stress_<timestamp>.json`
- `exp7_enriched_milp_decomp_stress_<timestamp>.csv`
- `exp7_enriched_milp_decomp_stress_latest.csv`
- `plots/coverage_runtime.png`
- `plots/assigned_unassigned.png`
- `plots/summary_with_coverage.csv`

## Текущий статус (первый прогон)

Сравнение с `EXP6` на точках `task_pct = {1,6,11,16,21,26}` показало:

- текущий `enriched_milp_decomp_v1` уступает `enriched_batched_greedy_v1` по покрытию задач;
- и работает существенно дольше на тех же входах.

Сводка сравнения:

- `experiments/exp7/local/exp7_vs_exp6_tasks_coverage.csv`

Вывод: для практического планирования в текущем виде лучше использовать `enriched_batched_greedy_v1`,
а MILP-decomp рассматривать как задел для дальнейшей донастройки objective/декомпозиции.
