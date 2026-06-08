# Exp3: greedy nearest fill per vehicle

Goal: build an approximate assignment as requested:

1. take a vehicle,
2. greedily fill it with nearest MNO tasks (shortest-path-based choice on a shortlist),
3. unload,
4. continue for the next trip / next vehicle,
5. evaluate km/h feasibility and coverage.

## Run

```bash
PYTHONPATH=src python experiments/exp3/run_exp3_greedy_fill.py \
  --dataset-path src/data/real/full_29k/dataset_real_spb_full_29k.json \
  --max-trips-per-agent 8 \
  --max-pickups-per-trip 16 \
  --shortlist-k 40 \
  --show-progress
```

## Notes

- This is an approximate heuristic for fast feasibility/coverage probing.
- Nearest candidate is selected using exact shortest-path distance on a euclidean shortlist (`shortlist-k`).
- Per-agent outputs include pickup km, unload km, return km, total km/h and limit checks.

Outputs are saved to `experiments/exp3/local/`.
