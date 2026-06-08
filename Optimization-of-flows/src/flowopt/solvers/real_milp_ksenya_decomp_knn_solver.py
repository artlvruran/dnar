"""Combo MILP: nearest-depot decomposition + per-vehicle k-NN arc pruning.

Stacks two model-shrinking transforms from real_milp_ksenya:

1. Tasks are partitioned into 4 disjoint sets by "nearest depot" (Euclidean
   metric on source coordinates). Each depot solves an independent MILP over
   its own agents only.
2. Inside each sub-MILP, per-vehicle arcs are pruned to the ``k`` nearest
   outgoing / ``k`` nearest incoming neighbours per node, with task-essential
   arcs (depot↔source, source→dest, dest→depot) force-kept.

Both steps **change the math**: cross-depot assignments are forbidden and
long "jumps" through the road graph are forbidden. In return the combo is
the only formulation that fits substantial fractions of the task-sweep
dataset into 2-3 GB of RAM (see KSENYA_SOLVER_FIX_NOTES.md, Step 3).
"""

from __future__ import annotations

from typing import Any

from .real_milp_ksenya_decomp_solver import _solve_real_milp_ksenya_decomposed_public
from .real_milp_ksenya_solver import _validate_payload


def solve_real_milp_ksenya_decomp_knn(
    payload: dict[str, Any],
    *,
    knn_k: int = 5,
    time_limit_sec: int = 120,
    mip_rel_gap: float = 0.05,
    verbose: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """MILP with nearest-depot decomposition stacked on top of k-NN pruning.

    Parameters
    ----------
    payload
        Standard dataset payload.
    knn_k
        Per-node fan-in/fan-out cap on arcs kept in each sub-MILP. Must be ≥ 1.
        Recommended starting point is ``5`` for the tightest memory regime
        (see KSENYA_SOLVER_FIX_NOTES.md).
    time_limit_sec
        Global time budget; split evenly across active depots
        (per-depot floor of 10s).
    mip_rel_gap
        MIP relative gap passed to HiGHS for each sub-problem.
    verbose
        Log group sizes and per-sub-problem progress.
    """
    if knn_k is None or int(knn_k) < 1:
        raise ValueError("knn_k must be a positive integer for the decomp+knn solver")
    _validate_payload(payload)
    return _solve_real_milp_ksenya_decomposed_public(
        payload,
        knn_k=int(knn_k),
        time_limit_sec=time_limit_sec,
        mip_rel_gap=mip_rel_gap,
        verbose=verbose,
        log_prefix="milp_ksenya_decomp_knn",
    )
