# EXP4: Day Load Profiles Benchmark (`u50`, `u90`)

Runs baseline and stochastic methods on datasets from:

- `src/data/real/day_load_profiles/u50/dataset_real_spb_day_u50.json`
- `src/data/real/day_load_profiles/u90/dataset_real_spb_day_u90.json`

Methods:

- `real_gap_vrp`
- `real_milp`
- `real_genetic`
- `real_stochastic_grasp`
- `real_stochastic_rr`

Each method is wrapped with a hard timeout (`--timeout-sec`, default `600` sec).
Optional feasibility mode: `--capacity-scale N` multiplies daily `km/h` limits by `N`
(equivalent to multi-day planning capacity).

## Run

```bash
PYTHONPATH=src python experiments/exp4/run_exp4_day_load_profiles.py \
  --timeout-sec 600
```

Fast stochastic feasibility run:

```bash
PYTHONPATH=src python experiments/exp4/run_exp4_day_load_profiles.py \
  --algorithms real_stochastic_grasp \
  --capacity-scale 3.0 \
  --timeout-sec 600 \
  --stochastic-budget-sec 180 \
  --stochastic-candidate-k 48 \
  --stochastic-rcl-size 8
```

Outputs are written into `experiments/exp4/local/`.
