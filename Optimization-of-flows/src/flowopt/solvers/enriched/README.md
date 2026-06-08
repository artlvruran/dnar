# Enriched Solvers Architecture

This package follows a strict pipeline so new optimizers can be plugged in with minimal boilerplate:

1. `problem.py`
- Converts dataset JSON to a typed `EnrichedProblem`.
- Normalizes constraints input for solvers:
  - agent availability
  - zone constraints
  - container compatibility
  - compact/D constraint
  - payload and raw-volume limits
  - object daily mass/volume caps

2. `distance_oracle.py`
- Unified distance access:
  - precomputed matrix first
  - networkx fallback

3. `*_solver_*.py`
- Solver returns `EnrichedSolveResult` only (routes, unassigned list, usage, runtime).
- Solver can use its own optimization logic (MILP, greedy, stochastic, decomposition, etc.).

4. `constraints.py` + `evaluator.py`
- Centralized post-solve validation for every solver:
  - assignment uniqueness/coverage
  - compatibility checks
  - reachability checks
  - per-agent daily km/hours
  - per-object daily mass/volume
- Produces unified `checks` block and normalized unassigned list.
- Sets final `feasible` consistently across all solvers.

5. `runner_v2.py`
- Public API entrypoints.
- Builds problem + oracle.
- Runs solver.
- Always finalizes with centralized evaluator.

## Contract For New Solvers

To add a new solver:

1. Implement solver class in new `*_solver_*.py`:
- Input: `problem: EnrichedProblem`, `oracle: DistanceOracleWithFallback`
- Output: `EnrichedSolveResult`

2. Add thin wrapper in `runner_v2.py`:
- build problem
- build oracle
- run solver
- call `finalize_enriched_result(...)`

3. Export wrapper and config in `__init__.py`.

This guarantees comparable metrics and identical constraints checks for all optimization methods.

