# Discrete Neural Algorithmic Reasoning
This repository contains the code to reproduce the experiments from "Discrete Neural Algorithmic Reasoning" paper. 

## Setup
Before running the source code, make sure to install the project dependencies:
```bash
pip install -r requirements.txt
```

## Main experiments

### Algorithms
- Breadth-first search
- Depth-first search
- Minimum spanning tree (Prim's algorithm)
- Maximum Independent Set (randomized)
- Shortest paths (Dijkstra's algorithm)
- MILP branching policy (single-iteration heuristic prediction inside LP-solver loop)


### Train a single-task model
```bash
python train.py --config_path
python eval.py
```

### Hints generation
You can find hints generation procedures for each algorithm in `generate_data.py`.

### MILP mode
- `milp` uses DNAR as a **policy model** inside a branch-and-bound loop.
- External LP solver remains responsible for feasibility checks and exact math.
- DNAR predicts only one-step branching heuristic decisions (which variable to branch on).
- During dataset generation, multiple heuristics are scored per instance and the best one is selected as supervision target.
- MILP is assumed in standard form: `min c^T x`, `Ax <= b`, `x >= 0`, with integrality enforced by branching.
