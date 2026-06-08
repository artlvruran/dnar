# Exp1: fast stochastic methods for large real dataset

This experiment compares fast heuristics with strict runtime limits per method.

## Implemented stochastic solvers

- `real_stochastic_grasp`: randomized greedy with RCL (GRASP-style), plus candidate reduction.
- `real_stochastic_rr`: ruin/recreate-lite on top of the constructive assignment.

Both methods use:

- candidate reduction (top-k compatible agents per task),
- shortest-path precompute only for source->destination service legs,
- explicit internal time budget (default 180s).

## Run

```bash
PYTHONPATH=src python experiments/exp1/run_exp1.py \
  --dataset-path src/data/real/full_29k/dataset_real_spb_full_29k.json \
  --algorithms real_stochastic_grasp,real_stochastic_rr \
  --budget-sec 180 \
  --timeout-sec 190
```

## Key references used for design

- Ropke, S., Pisinger, D. (2006). *An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery Problem with Time Windows*. Transportation Science. DOI: `10.1287/trsc.1050.0135`
- Pisinger, D., Ropke, S. (2004). *A general heuristic for vehicle routing problems* (technical report). https://di.ku.dk/forskning/Publikationer/tekniske_rapporter/tekniske-rapporter-2004/04-13.pdf
- Vidal, T. (2020/2022). *Hybrid genetic search for the CVRP: Open-source implementation and SWAP\**. arXiv: https://arxiv.org/abs/2012.10384
- Clarke, G., Wright, J.W. (1964). *Scheduling of Vehicles from a Central Depot to a Number of Delivery Points*. Operations Research. DOI: `10.1287/opre.12.4.568`

Results are saved into `experiments/exp1/local/`.
