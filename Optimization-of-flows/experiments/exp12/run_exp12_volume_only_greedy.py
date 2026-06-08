from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from flowopt.solvers.volume_only import solve_volume_only_greedy


ROOT = Path(__file__).resolve().parents[2]
BASE_DATASET = (
    ROOT
    / "demo/data/object_volume_feasible_fullfleet/full_all_tasks_all_agents_stage4_volume_only"
    / "dataset_real_spb_clean_full_all_tasks_all_agents_stage4_volume_only.json"
)
COVERED_ONLY_DATASET = (
    ROOT
    / "demo/data/object_volume_feasible_fullfleet/full_all_tasks_all_agents_stage4_volume_only"
    / "dataset_real_spb_clean_full_all_tasks_all_agents_stage4_volume_only_covered_only.json"
)
DEFAULT_DATASET = COVERED_ONLY_DATASET if COVERED_ONLY_DATASET.exists() else BASE_DATASET
OUT_DIR = ROOT / "experiments/local/exp12_volume_only_greedy"


def main() -> None:
    p = argparse.ArgumentParser(description="EXP12: fast greedy baseline for stage4 volume-only dataset")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--max-runtime-sec", type=float, default=45.0)
    p.add_argument("--max-tasks", type=int, default=5000)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()

    result = solve_volume_only_greedy(
        dataset_path=args.dataset,
        max_runtime_sec=float(args.max_runtime_sec),
        max_tasks=int(args.max_tasks) if args.max_tasks and args.max_tasks > 0 else None,
    )

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = pd.DataFrame([result.as_dict()])
    csv_path = out_dir / f"exp12_volume_only_greedy_{stamp}.csv"
    json_path = out_dir / f"exp12_volume_only_greedy_{stamp}.json"
    latest_csv = out_dir / "exp12_volume_only_greedy_latest.csv"
    latest_json = out_dir / "exp12_volume_only_greedy_latest.json"

    summary.to_csv(csv_path, index=False)
    summary.to_csv(latest_csv, index=False)
    json_path.write_text(json.dumps(result.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    latest_json.write_text(json.dumps(result.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
