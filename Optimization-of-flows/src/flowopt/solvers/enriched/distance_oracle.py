from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Any

import networkx as nx
import numpy as np


@dataclass
class PrecomputedDistanceOracle:
    matrix: np.ndarray
    node_to_index: dict[str, int]

    @classmethod
    def from_dataset_payload(cls, *, dataset_path: Path, payload: dict[str, Any]) -> "PrecomputedDistanceOracle | None":
        meta = payload.get("metadata") or {}
        info = meta.get("precomputed_distances") or {}
        matrix_file = info.get("distance_matrix_file")
        index_file = info.get("node_index_file")
        if not matrix_file or not index_file:
            return None

        candidates_base = [dataset_path.parent]
        # If dataset is a sweep copy, matrix may remain in sibling with_distances dir.
        candidates_base.append(dataset_path.parent / "with_distances")
        candidates_base.append(dataset_path.parent.parent / "with_distances")

        matrix_path = None
        index_path = None
        for base in candidates_base:
            m = base / str(matrix_file)
            i = base / str(index_file)
            if m.exists() and i.exists():
                matrix_path = m
                index_path = i
                break
        if matrix_path is None or index_path is None:
            return None

        matrix = np.load(matrix_path, mmap_mode="r")
        node_to_index: dict[str, int] = {}
        with index_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                node_id = str(row.get("node_id", "")).strip()
                idx = int(row.get("matrix_index", -1))
                if node_id and idx >= 0:
                    node_to_index[node_id] = idx

        if not node_to_index:
            return None
        return cls(matrix=matrix, node_to_index=node_to_index)

    def has_node(self, node_id: str) -> bool:
        return str(node_id) in self.node_to_index

    def dist(self, source: str, target: str) -> float:
        si = self.node_to_index.get(str(source))
        ti = self.node_to_index.get(str(target))
        if si is None or ti is None:
            return float("inf")
        return float(self.matrix[si, ti])


class DistanceOracleWithFallback:
    def __init__(self, *, nx_graph: nx.DiGraph, precomputed: PrecomputedDistanceOracle | None = None) -> None:
        self._graph = nx_graph
        self._precomputed = precomputed
        self._cache: dict[tuple[str, str], float] = {}

    def dist(self, source: str, target: str) -> float:
        key = (str(source), str(target))
        if key in self._cache:
            return self._cache[key]

        if self._precomputed is not None:
            d = self._precomputed.dist(source, target)
            if np.isfinite(d):
                self._cache[key] = float(d)
                return float(d)

        if source == target:
            self._cache[key] = 0.0
            return 0.0
        try:
            d = nx.shortest_path_length(self._graph, source=source, target=target, weight="distance_km")
        except Exception:
            d = float("inf")
        self._cache[key] = float(d)
        return float(d)
