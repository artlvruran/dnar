from __future__ import annotations

from pathlib import Path
from typing import Any

from .real_simple import DEFAULT_OUT_DATASET_PATH


def run_real_simple_smoke(
    *,
    dataset_path: Path | str = DEFAULT_OUT_DATASET_PATH,
    gap_iter: int = 25,
    step1_method: str = "lp",
) -> dict[str, Any]:
    from ..pipeline import run_dummy, run_gap_vrp

    dummy = run_dummy(dataset_path=dataset_path)
    gap = run_gap_vrp(
        dataset_path=dataset_path,
        step1_method=step1_method,
        gap_iter=gap_iter,
        use_repair=True,
        show_progress=False,
        verbose=False,
    )
    return {
        "dataset_path": str(Path(dataset_path)),
        "dummy": dummy.as_dict(),
        "gap_vrp": gap.as_dict(),
    }
