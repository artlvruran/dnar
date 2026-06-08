# EXP10: Bundle Cost vs Route-Aware Cost

Сравнение:

- как сейчас оценивается стоимость batched route (упрощенно);
- и route-aware оценка с учетом порядка обхода source-точек внутри bundle.

## Запуск

```bash
PYTHONPATH=src python experiments/exp10/run_exp10_bundle_route_cost_compare.py
```

## Что считается

Для каждого batched route:

- `old_total_km` / `old_hours` — как вернул solver;
- `new_total_km` / `new_hours` — пересчет через nearest-neighbor:
  - `depot -> source_i -> source_j -> ... -> destination -> depot`.

Итог в summary:

- дельта по `total_km`;
- дельта по `transport_work_ton_km` (приближенно);
- сколько маршрутов действительно multi-source.

## Артефакты

Папка: `experiments/local/exp10_bundle_route_cost_compare/`

- `exp10_summary_<timestamp>.json`
- `exp10_route_compare_<timestamp>.csv`
- `exp10_summary_latest.json`
- `exp10_route_compare_latest.csv`
