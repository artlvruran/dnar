"""MILP solver based on real_milp_ksenya with k-NN arc-filter pruning.

For every vehicle, only the ``k`` nearest outgoing and ``k`` nearest incoming
arcs are kept at each node (with depot↔source, source→dest, dest→depot of
every compatible task force-kept so single-task trips remain feasible).

This is **mathematically NOT equivalent** to the monolithic ksenya solver:
long "jumps" between distant nodes are forbidden, so some feasible routes
disappear from the search space. In return the model size shrinks from
``|N_v|² · |V|`` to roughly ``|N_v| · k · |V|``, which makes datasets that
do not fit into RAM under the full formulation tractable.
"""

from __future__ import annotations

from typing import Any

from .real_milp_ksenya_solver import _solve_real_milp_ksenya_core


def solve_real_milp_ksenya_knn(
    payload: dict[str, Any],
    *,
    knn_k: int = 10,
    time_limit_sec: int = 120,
    mip_rel_gap: float = 0.05,
    verbose: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """MILP with k-NN arc filtering, no depot-decomposition.

    Parameters
    ----------
    payload
        Standard dataset payload (same shape as for ``solve_real_milp_ksenya``).
    knn_k
        Per-node fan-in/fan-out cap on arcs kept in the model. Must be ≥ 1.
        Smaller values shrink the model further but risk pruning paths needed
        by multi-task routes; the per-task essential arcs (depot↔source,
        source→dest, dest→depot) are always force-kept regardless of ``knn_k``.
    time_limit_sec
        Time budget passed to HiGHS.
    mip_rel_gap
        MIP relative gap passed to HiGHS.
    verbose
        Log model dimensions and HiGHS progress.
    """
    if knn_k is None or int(knn_k) < 1:
        raise ValueError("knn_k must be a positive integer for the knn solver")
    return _solve_real_milp_ksenya_core(
        payload,
        knn_k=int(knn_k),
        time_limit_sec=time_limit_sec,
        mip_rel_gap=mip_rel_gap,
        verbose=verbose,
        log_prefix="milp_ksenya_knn",
    )
