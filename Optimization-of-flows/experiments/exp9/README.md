# EXP9: Potential Capacity Report

Цель: оценить «потолок» покрытия задач на текущем full-датасете через простые ограничивающие модели:

- только лимиты объектов (`ObjectDayM/ObjectDayV`);
- лимиты объектов + агрегированные бюджеты флота по часам/км;
- приближенный расход км через множитель на `nearest_object_distance_km`.

## Запуск

```bash
PYTHONPATH=src python experiments/exp9/run_exp9_potential_capacity_report.py
```

Опционально задать множители км:

```bash
PYTHONPATH=src python experiments/exp9/run_exp9_potential_capacity_report.py --km-factors "1.0,2.0,2.5,3.0"
```

## Артефакты

Папка: `experiments/local/exp9_potential_capacity/`

- `exp9_potential_meta_<timestamp>.json`
- `exp9_potential_scenarios_<timestamp>.csv`
- `exp9_potential_scenarios_<timestamp>.json`
- `exp9_potential_scenarios_latest.csv`
- `exp9_potential_meta_latest.json`
- `plots/coverage_scenarios.png`
- `plots/mass_tasks_scenarios.png`
