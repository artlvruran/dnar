from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    default_data_dir = repo_root / "demo/data/object_volume_feasible_fullfleet/full_all_tasks_all_agents_stage4_volume_only"
    unpack_script = default_data_dir / "scripts/unpack_volume_dataset.py"

    parser = argparse.ArgumentParser(description="Repo-level wrapper for unpacking volume-only dataset")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cmd = [sys.executable, str(unpack_script), "--data-dir", str(args.data_dir.resolve())]
    if args.force:
        cmd.append("--force")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

