from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..dataset import RoutingDataset, dataset_from_dict


def load_payload(dataset_path: Path | str) -> dict[str, Any]:
    path = Path(dataset_path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset(
    dataset_path: Path | str,
    *,
    validate: bool = False,
) -> tuple[RoutingDataset, dict[str, Any]]:
    payload = load_payload(dataset_path)
    dataset = dataset_from_dict(payload)
    if validate:
        dataset.validate()
    else:
        # Input datasets in this project often have empty routes.
        dataset.graph.validate()
        dataset.fleet.validate()
    return dataset, payload


def save_payload(payload: dict[str, Any], output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
