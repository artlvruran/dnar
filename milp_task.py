# milp_task.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog


@dataclass
class MILPInstance:
    A: np.ndarray          # (m, n)
    b: np.ndarray          # (m,)
    c: np.ndarray          # (n,) maximize c^T x
    lb: np.ndarray         # (n,)
    ub: np.ndarray         # (n,)
    is_int: np.ndarray     # (n,) bool
    sense: str = "max"

    @property
    def m(self) -> int:
        return self.A.shape[0]

    @property
    def n(self) -> int:
        return self.A.shape[1]


@dataclass
class LPResult:
    feasible: bool
    x: Optional[np.ndarray]
    obj: float


def solve_lp_relaxation(inst: MILPInstance, fixed: np.ndarray) -> LPResult:
    n = inst.n
    bounds = []
    for j in range(n):
        lo, hi = float(inst.lb[j]), float(inst.ub[j])
        if fixed[j] == 0:
            hi = 0.0
        elif fixed[j] == 1:
            lo = 1.0
            hi = 1.0
        bounds.append((lo, hi))

    res = linprog(
        c=-inst.c,
        A_ub=inst.A,
        b_ub=inst.b,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        return LPResult(False, None, float("-inf"))
    return LPResult(True, res.x.astype(np.float64), float(inst.c @ res.x))


def fractional_vars(inst: MILPInstance, x: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    frac = np.where(
        inst.is_int & (fixed < 0) & (np.abs(x - np.round(x)) > 1e-8)
    )[0]
    return frac


def heuristic_most_fractional(inst: MILPInstance, x: np.ndarray, fixed: np.ndarray) -> Optional[int]:
    frac = fractional_vars(inst, x, fixed)
    if frac.size == 0:
        return None
    return int(frac[np.argmin(np.abs(x[frac] - 0.5))])


def heuristic_largest_obj(inst: MILPInstance, x: np.ndarray, fixed: np.ndarray) -> Optional[int]:
    frac = fractional_vars(inst, x, fixed)
    if frac.size == 0:
        return None
    return int(frac[np.argmax(np.abs(inst.c[frac]))])


def heuristic_most_fractional_reduced_cost(
    inst: MILPInstance, x: np.ndarray, fixed: np.ndarray, rc: Optional[np.ndarray] = None
) -> Optional[int]:
    frac = fractional_vars(inst, x, fixed)
    if frac.size == 0:
        return None
    if rc is None:
        return int(frac[np.argmin(np.abs(x[frac] - 0.5))])
    return int(frac[np.argmax(np.abs(rc[frac]))])


def heuristic_depth_first(
    inst: MILPInstance, x: np.ndarray, fixed: np.ndarray
) -> Optional[int]:
    frac = fractional_vars(inst, x, fixed)
    if frac.size == 0:
        return None
    return int(frac[0])


HEURISTICS = {
    0: ("most_fractional", heuristic_most_fractional),
    1: ("largest_obj", heuristic_largest_obj),
    2: ("fractional_rc", heuristic_most_fractional_reduced_cost),
    3: ("depth_first", heuristic_depth_first),
}


def child_bound(inst: MILPInstance, fixed: np.ndarray, var: int, val: int) -> float:
    child = fixed.copy()
    child[var] = val
    res = solve_lp_relaxation(inst, child)
    return res.obj if res.feasible else float("-inf")


def oracle_pick_heuristic(
    inst: MILPInstance, fixed: np.ndarray, x: np.ndarray
) -> Tuple[int, Optional[int], Dict[int, float]]:
    """
    Returns:
      best_heuristic_id,
      branching_variable chosen by that heuristic,
      score_by_heuristic (lower is better)
    Score = max(child_boundchild bounds) after branching on the chosen variable.
    """
    scores: Dict[int, float] = {}
    best_h = -1
    best_var = None
    best_score = float("inf")

    for hid, (_, hfun) in HEURISTICS.items():
        var = hfun(inst, x, fixed)
        if var is None:
            continue
        b0 = child_bound(inst, fixed, var, 0)
        b1 = child_bound(inst, fixed, var, 1)
        score = max(b0, b1)
        scores[hid] = score
        if score < best_score:
            best_score = score
            best_h = hid
            best_var = var

    return best_h, best_var, scores


def bnb_trace(inst: MILPInstance, max_steps: int = 128, node_selection: str = "best_bound"):
    n = inst.n
    open_nodes: List[np.ndarray] = [np.full(n, -1, dtype=np.int8)]

    states = []
    heuristic_y = []
    branch_y = []

    while open_nodes and len(states) < max_steps:
        evaluated = []
        for fixed in open_nodes:
            lp = solve_lp_relaxation(inst, fixed)
            evaluated.append((fixed, lp))

        feasible_nodes = [(f, lp) for f, lp in evaluated if lp.feasible]
        if not feasible_nodes:
            break

        if node_selection == "best_bound":
            idx = int(np.argmax([lp.obj for _, lp in feasible_nodes]))
        elif node_selection == "dfs":
            idx = len(feasible_nodes) - 1
        else:
            idx = 0

        fixed, lp = feasible_nodes[idx]

        for k, node in enumerate(open_nodes):
            if np.array_equal(node, fixed):
                open_nodes.pop(k)
                break

        x = lp.x
        assert x is not None

        hid, var, scores = oracle_pick_heuristic(inst, fixed, x)

        states.append(make_graph_state(inst, fixed, x, lp.obj))
        heuristic_y.append(hid)
        branch_y.append(-1 if var is None else var)

        if var is None:
            continue

        child0 = fixed.copy()
        child0[var] = 0
        child1 = fixed.copy()
        child1[var] = 1

        if solve_lp_relaxation(inst, child0).feasible:
            open_nodes.append(child0)
        if solve_lp_relaxation(inst, child1).feasible:
            open_nodes.append(child1)

    return states, np.array(heuristic_y, dtype=np.int64), np.array(branch_y, dtype=np.int64)


def make_graph_state(inst: MILPInstance, fixed: np.ndarray, x: np.ndarray, bound: float):
    n, m = inst.n, inst.m
    num_nodes = n + m + 1
    global_node = n + m

    node_fts = np.zeros((num_nodes, 8), dtype=np.float64)

    node_fts[:n, 0] = inst.is_int.astype(np.float64)
    node_fts[:n, 1] = (fixed == 0).astype(np.float64)
    node_fts[:n, 2] = (fixed == 1).astype(np.float64)
    node_fts[:n, 3] = x
    node_fts[:n, 4] = inst.c
    node_fts[:n, 5] = inst.lb
    node_fts[:n, 6] = inst.ub
    node_fts[:n, 7] = np.abs(x - np.round(x)) * inst.is_int

    Ax = inst.A @ x
    slack = inst.b - Ax
    node_fts[n:n + m, 0] = 0.0
    node_fts[n:n + m, 1] = inst.b
    node_fts[n:n + m, 2] = Ax
    node_fts[n:n + m, 3] = slack
    node_fts[n:n + m, 4] = bound

    node_fts[global_node, 0] = 1.0
    node_fts[global_node, 1] = bound
    node_fts[global_node, 2] = float(np.sum((inst.is_int & (fixed < 0)).astype(np.float64)))
    node_fts[global_node, 3] = float(np.sum(np.abs(x - np.round(x)) * inst.is_int))
    node_fts[global_node, 4] = float(np.sum(fixed >= 0))

    edge_fts = np.zeros((num_nodes, num_nodes, 1), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            a = inst.A[j, i]
            if abs(a) < 1e-12:
                continue
            c = n + j
            edge_fts[i, c, 0] = a
            edge_fts[c, i, 0] = a

    for i in range(num_nodes):
        edge_fts[i, i, 0] = 1.0

    scalars = np.zeros((num_nodes, num_nodes), dtype=np.float64)

    # self-loops carry node-level scalar information
    scalars[np.arange(num_nodes), np.arange(num_nodes)] = 0.0
    scalars[global_node, global_node] = bound
    scalars[:n, :n][np.diag_indices(n)] = np.abs(x - np.round(x)) * inst.is_int
    scalars[n:n + m, n:n + m] = 0.0

    return node_fts, edge_fts, scalars


def random_feasible_milp(
    n_vars: int = 16,
    n_cons: int = 8,
    int_ratio: float = 0.6,
    seed: Optional[int] = None,
) -> MILPInstance:
    rng = np.random.default_rng(seed)
    is_int = rng.random(n_vars) < int_ratio

    lb = np.zeros(n_vars, dtype=np.float64)
    ub = np.where(is_int, 1.0, rng.uniform(1.5, 4.0, size=n_vars))

    x_star = np.where(is_int, rng.integers(0, 2, size=n_vars), rng.uniform(lb, ub))
    A = rng.normal(size=(n_cons, n_vars))
    A = np.abs(A)
    b = A @ x_star + rng.uniform(0.5, 2.0, size=n_cons)

    c = rng.normal(size=n_vars)

    inst = MILPInstance(
        A=A,
        b=b,
        c=c,
        lb=lb,
        ub=ub,
        is_int=is_int,
    )
    inst.edge_index, inst.edge_attr = instance_edge_index(inst), instance_edge_attr(inst)[1]
    return inst

def instance_edge_index(inst: MILPInstance):
    n, m = inst.n, inst.m
    edges = []

    for i in range(n):
        for j in range(m):
            if abs(inst.A[j, i]) < 1e-12:
                continue
            c = n + j
            edges.append((i, c))
            edges.append((c, i))

    for i in range(n + m + 1):
        edges.append((i, i))

    edges = sorted(edges)  # sort by src, then dst
    edge_index = np.asarray(edges, dtype=np.int64).T
    return edge_index

def instance_edge_attr(inst: MILPInstance):
    n, m = inst.n, inst.m
    edges = []
    attrs = []

    for i in range(n):
        for j in range(m):
            a = inst.A[j, i]
            if abs(a) < 1e-12:
                continue
            c = n + j
            edges.append((i, c))
            attrs.append([a])
            edges.append((c, i))
            attrs.append([a])

    for i in range(n + m + 1):
        edges.append((i, i))
        attrs.append([1.0])

    edges, attrs = zip(*sorted(zip(edges, attrs)))
    edge_index = np.asarray(edges, dtype=np.int64).T
    edge_attr = np.asarray(attrs, dtype=np.float64)
    return edge_index, edge_attr