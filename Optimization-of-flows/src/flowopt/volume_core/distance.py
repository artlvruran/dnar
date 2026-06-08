from __future__ import annotations

from pathlib import Path
from typing import Any
import csv

import networkx as nx
import numpy as np


class DistanceEngine:
    def __init__(self, *, nx_graph: nx.DiGraph, precomputed: dict[str, Any] | None = None, base_dir: Path | None = None) -> None:
        self.graph = nx_graph
        self.cache: dict[tuple[str, str], float] = {}
        self.path_cache: dict[tuple[str, str], tuple[str, ...]] = {}
        self.node_to_idx: dict[str, int] = {}
        self.matrix: np.ndarray | None = None

        if precomputed and base_dir is not None:
            mfile = precomputed.get("distance_matrix_file")
            ifile = precomputed.get("node_index_file")
            if mfile and ifile:
                mpath = base_dir / str(mfile)
                ipath = base_dir / str(ifile)
                if mpath.exists() and ipath.exists():
                    try:
                        self.matrix = np.load(mpath, mmap_mode="r")
                        with ipath.open("r", encoding="utf-8") as f:
                            rd = csv.DictReader(f)
                            for r in rd:
                                nid = str(r.get("node_id", "")).strip()
                                idx = int(r.get("matrix_index", -1))
                                if nid and idx >= 0:
                                    self.node_to_idx[nid] = idx
                    except Exception:
                        self.matrix = None
                        self.node_to_idx = {}

    def dist(self, u: str, v: str) -> float:
        u = str(u)
        v = str(v)
        if u == v:
            return 0.0
        k = (u, v)
        if k in self.cache:
            return self.cache[k]

        if self.matrix is not None and self.node_to_idx:
            ui = self.node_to_idx.get(u)
            vi = self.node_to_idx.get(v)
            if ui is not None and vi is not None:
                d = float(self.matrix[ui, vi])
                if np.isfinite(d):
                    self.cache[k] = d
                    return d

        try:
            d = float(nx.shortest_path_length(self.graph, source=u, target=v, weight="distance_km"))
        except Exception:
            d = float("inf")
        self.cache[k] = d
        return d

    def path(self, u: str, v: str) -> tuple[str, ...]:
        u = str(u)
        v = str(v)
        if u == v:
            return (u,)
        k = (u, v)
        if k in self.path_cache:
            return self.path_cache[k]
        try:
            p = tuple(str(x) for x in nx.shortest_path(self.graph, source=u, target=v, weight="distance_km"))
        except Exception:
            p = (u, v)
        self.path_cache[k] = p
        return p
