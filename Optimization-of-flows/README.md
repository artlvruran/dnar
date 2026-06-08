# 🚚 Optimization of Flows

This project compares different algorithms for optimizing transportation flows and routing in urban logistics.

## Quick Start (Real Enriched Datasets)

### Unified Volume-Only Demo (recommended)

This is the current unified pipeline where all solvers use the same IO contract:

`VolumeDataset -> VolumeSolver.solve_checked() -> AssignmentSolution -> VolumeDataset.evaluate()`

#### 1) Install deps

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy pandas scipy networkx matplotlib tqdm jupyter
export PYTHONPATH=src
```

#### 2) Unpack big dataset (json.gz -> json)

```bash
python scripts/py_scripts/unpack_volume_full_dataset.py
```

This unpacks files in:

`demo/data/object_volume_feasible_fullfleet/full_all_tasks_all_agents_stage4_volume_only/`

#### 3) (Optional) Build precomputed distances

```bash
python demo/data/object_volume_feasible_fullfleet/full_all_tasks_all_agents_stage4_volume_only/scripts/add_precomputed_distances_to_dataset.py \
  --input-json demo/data/object_volume_feasible_fullfleet/full_all_tasks_all_agents_stage4_volume_only/dataset_real_spb_clean_full_all_tasks_all_agents_stage4_volume_only.json \
  --output-dir demo/data/object_volume_feasible_fullfleet/full_all_tasks_all_agents_stage4_volume_only/with_distances \
  --chunk-size 256 \
  --distance-dtype float32 \
  --save-paths false
```

#### 4) Run unified benchmark notebook

Open and run:

`demo/volume_full_unified_7algo_demo.ipynb`

It runs all methods sequentially (`greedy`, full solvers, stochastic solvers), saves per-algorithm outputs in `demo/local/...`, and builds summary with final quality score.

#### 5) Run script alternative

```bash
PYTHONPATH=src python experiments/exp13/run_exp13_volume_core_3algo.py \
  --dataset demo/data/object_volume_feasible_fullfleet/full_all_tasks_all_agents_stage4_volume_only/dataset_real_spb_clean_full_all_tasks_all_agents_stage4_volume_only.json \
  --include-stochastic
```

### Unified pipeline structure

- `src/flowopt/volume_core/dataset.py`
  - `VolumeDataset` parses JSON graph/tasks/agents.
  - validates compatibility (`agent_can_take_task`) and computes evaluation metrics.
- `src/flowopt/volume_core/solver_base.py`
  - `VolumeSolver` base class.
  - `solve_checked()` enforces unified input/output contract.
- `src/flowopt/volume_core/contracts.py`
  - strict asserts for dataset input and solver output format.
- `src/flowopt/volume_core/greedy_batch_solver.py`
  - base route-construction engine with configurable scoring and stochastic behavior.
- `src/flowopt/volume_core/three_algorithms.py`
  - algorithm wrappers (`*_like`, `*_stoch`) built on unified base.
- `src/flowopt/volume_core/models.py`
  - dataclasses: `AssignmentSolution`, `TripPlan`, `EvaluationResult`.
- `src/flowopt/volume_core/reporting.py`
  - solution export and visualization helpers.

### 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy pandas scipy networkx matplotlib tqdm jupyter
export PYTHONPATH=src
```

### 2) Build precomputed distances

Unpack base full dataset first:

```bash
gzip -dk demo/data/object_mass_feasible_fullfleet/container_full_split_all_agents/dataset_real_spb_clean_full_split_by_containers_all_agents.json.gz
```

Then build distances:

```bash
python demo/data/object_mass_feasible_fullfleet/scripts/add_precomputed_distances_to_dataset.py \
  --input-json demo/data/object_mass_feasible_fullfleet/container_full_split_all_agents/dataset_real_spb_clean_full_split_by_containers_all_agents.json \
  --output-dir demo/data/object_mass_feasible_fullfleet/container_full_split_all_agents/with_distances \
  --chunk-size 256 \
  --distance-dtype float32 \
  --save-paths false
```

### 3) Build task/agent sweeps

```bash
python demo/data/object_mass_feasible_fullfleet/scripts/build_task_agent_sweep_with_constraints.py
```

### 4) Run solver ablations

```bash
PYTHONPATH=src python experiments/exp8/run_exp8_batching_ablation.py
```

Optional full-dataset run with time limit:

```bash
PYTHONPATH=src python experiments/exp8/run_exp8_full100_2min.py
```

### 5) Where results are saved

- `experiments/local/exp8_batching_ablation/*.csv`
- `experiments/local/exp8_batching_ablation/*.json`
- `experiments/local/exp8_batching_ablation/plots/*`

We evaluate three approaches:

* **MILP (Mixed Integer Linear Programming)** — exact optimization
* **GAP + VRP heuristic** — decomposition approach
* **Genetic Algorithm (GA)** — metaheuristic method

The comparison is performed on a synthetic dataset representing transportation tasks in an urban environment.

---

## 📂 Project Structure

### 🔹 `First_step_synthetic_data/`

Contains the **MILP-based solver** for the transportation problem.

**Main files:**

* `simple_solver_milp.py` — core MILP model using linear optimization
* `simple_solver_components.py` — helper functions and cost calculations

👉 Produces **optimal routing solution** with minimal transport work.

---

### 🔹 `genetic_algo_synthetic_data/`

Implementation of the **Genetic Algorithm solver**.

**Main files:**

* `genetic_solver_min.py` — main GA runner (multi-day simulation)
* `genetic_solver_components_improved.py` — GA operators (mutation, crossover, evaluation)

👉 Produces **near-optimal solutions** using evolutionary optimization.

---

### 🔹 `synthetic_data_gap_vrp_solver/`

Contains the **GAP + VRP decomposition solver**.

**Main files:**

* `gap_vrp_solver.py` — main logic:

  * task assignment (GAP)
  * routing heuristics (VRP components)
* `dataset.py` — data structures (graph, tasks, agents, routes)

👉 First assigns tasks to agents, then builds routes heuristically.

---

### 🔹 Root files

* `dataset_sandbox_type2.json` — synthetic dataset:

  * graph (road network)
  * agents (vehicles)
  * tasks (transport demands)
  * metadata (depots)

* `README.md` — project description

---

## ⚙️ Methods Overview

### 🧮 MILP

* Guarantees optimal solution
* Minimizes total transport work (ton-km)
* Computationally expensive

---

### 🧩 GAP + VRP

* Step 1: assign tasks to agents (GAP)
* Step 2: build routes (VRP heuristics)
* Faster but approximate

---

### 🧬 Genetic Algorithm

* Population-based search
* Uses mutation and crossover
* Flexible and scalable

---

## 📊 Metrics

All methods are compared using:

* `assigned_routes` — number of completed routes
* `unassigned_tasks` — tasks not served
* `active_agents` — number of used vehicles
* `transport_work_ton_km` — main efficiency metric

---

## 🗺️ Visualization

Routes can be visualized on a map (e.g., Saint Petersburg) using `folium`.

* MILP → exact routes
* GA → heuristic routes
* GAP → assignment only (no full routing)

---

## 🚀 How to Run (Colab)

1. Upload all files to `/content`
2. Run:

   * MILP solver
   * GAP + VRP solver
   * Genetic solver
3. Compare results
4. Visualize routes on map

---

## 🧠 Conclusion

* **MILP** provides the best solution quality
* **Genetic Algorithm** gives competitive results with flexibility
* **GAP + VRP** is fast and scalable, but less precise

---

## 📌 Future Improvements

* Full VRP integration for GAP results
* Runtime comparison
* Multi-day simulation analysis
* Real-world datasets

---

## 👩‍💻 Authors

* Igor Ignashin
* Anna Komleva
* Kristina Abgaryan
* Ksenia Vydrina
