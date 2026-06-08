# full_enriched_clean_dataset_sweep

Run 7x7 sweep (`task_percent x agent_percent`) for datasets in:
`src/data/real/clean_full_enriched/sweep_7task_7agents`

## Run scheduling

```bash
scripts/bash_scripts/run_full_enriched_clean_dataset_sweep.sh
```

Examples:

```bash
# default: fastest solver only
scripts/bash_scripts/run_full_enriched_clean_dataset_sweep.sh \
  --algorithms real_milp

# subset + verbose progress
scripts/bash_scripts/run_full_enriched_clean_dataset_sweep.sh \
  --task-percents 5,35,100 \
  --agent-percents 50,100 \
  --show-solver-progress \
  --show-live-solver-logs

# include genetic with internal 20s cap
scripts/bash_scripts/run_full_enriched_clean_dataset_sweep.sh \
  --algorithms real_milp,real_genetic \
  --genetic-time-limit-sec 20
```

## Plot results

```bash
scripts/bash_scripts/plot_full_enriched_clean_dataset_sweep.sh
```

Outputs are stored in `scripts/local/full_enriched_clean_sweep`:

- `state/results.jsonl` - append-only state (used for resume)
- `state/results_latest.csv` - latest materialized table
- `runs/sweep_run_<timestamp>.{csv,json,md}` - per-run snapshot
- `plots/*.png` - heatmaps, 3D surfaces, line charts

## Resume behavior

On rerun, completed `(dataset, algorithm)` pairs are skipped automatically.
Use `--force` to recompute.
