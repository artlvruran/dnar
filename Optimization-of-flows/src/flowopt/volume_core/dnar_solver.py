from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time
from typing import TYPE_CHECKING

from .greedy_batch_solver import GreedyBatchConfig, GreedyBatchVolumeSolver
from .models import AssignmentSolution
from .solver_base import VolumeSolver

if TYPE_CHECKING:  # pragma: no cover
    from .dataset import VolumeDataset


@dataclass(frozen=True)
class DnarFlowConfig:
    """Configuration for a DNAR-inspired policy wrapper for volume flows.

    The current implementation is an inference-time integration point: it builds
    discrete node/edge/scalar states from ``VolumeDataset`` and uses a small
    deterministic attention-style scorer to order tasks and agents before the
    existing checked greedy route constructor performs feasibility-preserving
    assignment and repair.
    """

    max_runtime_sec: float = 120.0
    processor_steps: int = 4
    hidden_size: int = 16
    top_k_agents: int = 30
    top_k_destinations: int = 4
    max_tasks_in_trip: int = 256
    fill_time_budget_sec: float = 20.0
    verbose: bool = True
    random_seed: int = 42


class _DnarFlowPolicy:
    """Deterministic DNAR-shaped discrete policy for flow datasets.

    This class intentionally mirrors DNAR's state-processing contract without
    requiring a trained checkpoint: tasks and agents become node states,
    feasible task-agent pairs become edge states, scalar capacity/volume values
    participate in message scores, and several processor steps refine task and
    agent priorities.  Replacing ``score_dataset`` with a trained DNAR module is
    the intended next step; the solver contract stays unchanged.
    """

    def __init__(self, config: DnarFlowConfig) -> None:
        self.config = config

    def score_dataset(self, dataset: "VolumeDataset") -> tuple[dict[str, float], dict[str, float], tuple[str, ...]]:
        tasks = list(dataset.tasks)
        agents = [a for a in dataset.agents if a.is_active and a.depot_node_id is not None]
        if not tasks or not agents:
            return {}, {}, ("[dnar_flow_policy_v1] empty task/agent side",)

        compat: dict[str, list[str]] = {}
        for task in tasks:
            compat[task.task_id] = [agent.agent_id for agent in agents if dataset.agent_can_take_task(task, agent)]

        max_volume = max([task.volume_raw_m3 for task in tasks] + [1.0])
        max_cap = max([agent.max_raw_volume_m3 for agent in agents] + [1.0])
        max_hours = max([agent.max_hours for agent in agents] + [1.0])
        max_km = max([agent.max_daily_km for agent in agents] + [1.0])

        task_state = {
            task.task_id: (task.volume_raw_m3 / max_volume) + (1.0 / max(len(compat[task.task_id]), 1))
            for task in tasks
        }
        agent_state = {
            agent.agent_id: (agent.max_raw_volume_m3 / max_cap + agent.max_hours / max_hours + agent.max_daily_km / max_km) / 3.0
            for agent in agents
        }

        agent_by_id = {agent.agent_id: agent for agent in agents}
        task_by_id = {task.task_id: task for task in tasks}
        logs = [
            "[dnar_flow_policy_v1] encoded graph: "
            f"task_nodes={len(tasks)}, agent_nodes={len(agents)}, compat_edges={sum(len(v) for v in compat.values())}, "
            f"processor_steps={self.config.processor_steps}, hidden_size={self.config.hidden_size}"
        ]

        for step in range(max(1, self.config.processor_steps)):
            next_task = dict(task_state)
            next_agent = dict(agent_state)
            for task in tasks:
                vals = []
                for aid in compat[task.task_id]:
                    agent = agent_by_id[aid]
                    eff = dataset.effective_task_volume(task, agent)
                    fit = 1.0 - min(1.0, eff / max(agent.max_raw_volume_m3, 1e-9))
                    depot = str(agent.depot_node_id)
                    route_km = (
                        dataset.dist.dist(depot, task.source_node_id)
                        + dataset.dist.dist(task.source_node_id, task.destination_node_id)
                        + dataset.dist.dist(task.destination_node_id, depot)
                    )
                    if math.isinf(route_km):
                        continue
                    proximity = 1.0 / (1.0 + route_km)
                    vals.append(0.45 * agent_state[aid] + 0.35 * fit + 0.20 * proximity)
                if vals:
                    next_task[task.task_id] = 0.5 * task_state[task.task_id] + 0.5 * max(vals)
            for agent in agents:
                linked = [task_by_id[tid] for tid, aids in compat.items() if agent.agent_id in aids]
                if linked:
                    scarcity = sum(1.0 / max(len(compat[t.task_id]), 1) for t in linked) / len(linked)
                    volume_pull = sum(dataset.effective_task_volume(t, agent) for t in linked) / max(agent.max_raw_volume_m3, 1e-9)
                    next_agent[agent.agent_id] = 0.6 * agent_state[agent.agent_id] + 0.4 * (scarcity + min(1.0, volume_pull))
            task_state, agent_state = next_task, next_agent
            logs.append(f"[dnar_flow_policy_v1] processor_step={step + 1}: updated discrete priorities")

        return task_state, agent_state, tuple(logs)


class DnarFlowVolumeSolver(VolumeSolver):
    """Volume solver that injects a DNAR-style policy before route construction."""

    def __init__(self, config: DnarFlowConfig | None = None) -> None:
        self.config = config or DnarFlowConfig()
        self.policy = _DnarFlowPolicy(self.config)

    def solve(self, dataset: "VolumeDataset") -> AssignmentSolution:
        t0 = time.perf_counter()
        task_scores, agent_scores, logs = self.policy.score_dataset(dataset)
        ranked_dataset = dataset.__class__(
            dataset_path=dataset.dataset_path,
            payload=dataset.payload,
            graph=dataset.graph,
            nodes=dataset.nodes,
            tasks=sorted(dataset.tasks, key=lambda t: (-task_scores.get(t.task_id, 0.0), t.task_id)),
            agents=sorted(dataset.agents, key=lambda a: (-agent_scores.get(a.agent_id, 0.0), a.agent_id)),
            object_volume_caps=dataset.object_volume_caps,
            service_hours_by_container=dataset.service_hours_by_container,
            dist=dataset.dist,
        )
        inner = GreedyBatchVolumeSolver(
            GreedyBatchConfig(
                max_runtime_sec=self.config.max_runtime_sec,
                top_k_agents=self.config.top_k_agents,
                top_k_destinations=self.config.top_k_destinations,
                max_tasks_in_trip=self.config.max_tasks_in_trip,
                verbose=self.config.verbose,
                log_every_sec=5.0,
                trip_log_every=100,
                score_mode="vol_per_km",
                stochastic_mode=False,
                random_seed=self.config.random_seed,
                deterministic_fill=True,
                fill_time_budget_sec=self.config.fill_time_budget_sec,
            )
        )
        solution = inner.solve(ranked_dataset)
        elapsed = time.perf_counter() - t0
        return replace(
            solution,
            algorithm="volume_dnar_flow_policy_v1",
            runtime_sec=float(elapsed),
            solver_logs=tuple(logs) + solution.solver_logs,
        )
