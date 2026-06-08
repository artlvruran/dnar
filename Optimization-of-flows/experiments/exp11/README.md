# EXP11: Source Policy Compare

Сравнение политик выбора source в одном эксперименте:

- `baseline_batched_greedy_v1` (текущий production baseline);
- `locked_current` (группировка по `source+dest+container`);
- `locked_mass_first` (тот же locked, но приоритет источников по массе);
- `multi_source_nn` (разрешение multi-source внутри одного `dest+container` с NN-обходом).

## Запуск

```bash
PYTHONPATH=src python experiments/exp11/run_exp11_source_policy_compare.py
```

С ограничением времени на каждую политику:

```bash
PYTHONPATH=src python experiments/exp11/run_exp11_source_policy_compare.py --max-runtime-sec 90
```

## Выход

`experiments/local/exp11_source_policy_compare/`:

- `exp11_source_policy_compare_<timestamp>.csv`
- `exp11_source_policy_compare_<timestamp>.json`
- `exp11_source_policy_compare_latest.csv`
