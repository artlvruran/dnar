# Exp2: MNO batching analysis

Goal: estimate how many large tasks we get if we batch tasks inside each MNO up to truck capacity.

Modes:

- `source_container_dest`: strict, keep destination fixed.
- `source_container`: relaxed, batch by source+container (destination blended for lower-bound distance estimate).

Capacity policies per container (`A/B/C/D`), computed from compatible fleet capacities:

- `max`, `p75`, `median`, `min`.

Run:

```bash
PYTHONPATH=src python experiments/exp2/analyze_mno_batching.py \
  --dataset-path src/data/real/full_29k/dataset_real_spb_full_29k.json \
  --modes source_container_dest,source_container \
  --capacity-policies max,p75,median
```

Outputs are saved to `experiments/exp2/local/`.
