# EXP12: Stage-4 Volume-Only Fast Greedy

Быстрый baseline-эксперимент для датасета `stage4_volume_only`.

## Что делает

- учитывает ограничения по зоне, контейнерной совместимости и `requires_compact_d`;
- учитывает только объемные лимиты (без mass-лимитов):
  - лимит ТС по объему,
  - лимит объекта выгрузки по объему;
- объем задачи для конкретного ТС считается как:
  - `volume_raw_m3 / compaction_coeff`, если задача `is_compactable=True` и у ТС `compaction_coeff > 1`,
  - иначе `volume_raw_m3`.

## Запуск

```bash
PYTHONPATH=src python experiments/exp12/run_exp12_volume_only_greedy.py \
  --dataset demo/data/object_volume_feasible_fullfleet/full_all_tasks_all_agents_stage4_volume_only/dataset_real_spb_clean_full_all_tasks_all_agents_stage4_volume_only_covered_only.json \
  --max-runtime-sec 45 \
  --max-tasks 5000
```

## Выход

`experiments/local/exp12_volume_only_greedy/`:

- `exp12_volume_only_greedy_<timestamp>.csv`
- `exp12_volume_only_greedy_<timestamp>.json`
- `exp12_volume_only_greedy_latest.csv`
- `exp12_volume_only_greedy_latest.json`
