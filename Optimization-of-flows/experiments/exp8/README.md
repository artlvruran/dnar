# EXP8: Internal Batching Ablation

Цель: проверить, как solver-level batching влияет на покрытие задач (`unassigned`) и время.

Запускается единый прогон для:
- `greedy_ref` (референс)
- baseline MILP идеи
- новые гибриды `batch -> MILP` без внешнего greedy repair.

## Быстрый запуск

```bash
PYTHONPATH=src python experiments/exp8/run_exp8_batching_ablation.py
```

## Кастомный датасет

```bash
PYTHONPATH=src python experiments/exp8/run_exp8_batching_ablation.py \
  --dataset /abs/path/to/dataset.json
```

## Частичный прогон

```bash
PYTHONPATH=src python experiments/exp8/run_exp8_batching_ablation.py \
  --only greedy_ref,milp_batch_then_milp,milp_batch_cascaded
```

## Артефакты

Сохраняются в `experiments/local/exp8_batching_ablation/`:
- `exp8_batching_ablation_<timestamp>.json`
- `exp8_batching_ablation_<timestamp>.csv`
- `exp8_batching_ablation_latest.csv`
- `plots/coverage.png`
- `plots/runtime_vs_coverage.png`
