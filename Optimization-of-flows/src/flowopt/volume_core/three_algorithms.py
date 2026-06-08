from __future__ import annotations

from dataclasses import dataclass, replace
import random
import time

from .greedy_batch_solver import GreedyBatchConfig, GreedyBatchVolumeSolver
from .models import AssignmentSolution
from .solver_base import VolumeSolver


def _rename_solution(solution: AssignmentSolution, algorithm: str) -> AssignmentSolution:
    return replace(solution, algorithm=algorithm)


class VolumeGapVRPLikeSolver(VolumeSolver):
    def __init__(self, config: GreedyBatchConfig | None = None) -> None:
        base = config or GreedyBatchConfig()
        self._inner = GreedyBatchVolumeSolver(
            GreedyBatchConfig(
                max_runtime_sec=base.max_runtime_sec,
                top_k_agents=base.top_k_agents,
                top_k_destinations=max(4, base.top_k_destinations),
                max_tasks_in_trip=max(200, base.max_tasks_in_trip),
                min_remaining_hours=base.min_remaining_hours,
                verbose=base.verbose,
                log_every_sec=base.log_every_sec,
                trip_log_every=base.trip_log_every,
                score_mode="vol_per_km",
                stochastic_mode=False,
                random_seed=101,
                exploration_noise=0.0,
                deterministic_fill=True,
                fill_time_budget_sec=20.0,
            )
        )

    def solve(self, dataset: "VolumeDataset") -> AssignmentSolution:
        sol = self._inner.solve(dataset)
        return _rename_solution(sol, "volume_gap_vrp_like_v1")


class VolumeMilpLikeSolver(VolumeSolver):
    def __init__(self, config: GreedyBatchConfig | None = None) -> None:
        base = config or GreedyBatchConfig()
        self._inner = GreedyBatchVolumeSolver(
            GreedyBatchConfig(
                max_runtime_sec=base.max_runtime_sec,
                top_k_agents=base.top_k_agents,
                top_k_destinations=max(3, base.top_k_destinations),
                max_tasks_in_trip=max(300, base.max_tasks_in_trip),
                min_remaining_hours=base.min_remaining_hours,
                verbose=base.verbose,
                log_every_sec=base.log_every_sec,
                trip_log_every=base.trip_log_every,
                score_mode="min_km",
                stochastic_mode=False,
                random_seed=202,
                exploration_noise=0.0,
                deterministic_fill=True,
                fill_time_budget_sec=20.0,
            )
        )

    def solve(self, dataset: "VolumeDataset") -> AssignmentSolution:
        sol = self._inner.solve(dataset)
        return _rename_solution(sol, "volume_milp_like_v1")


class VolumeGeneticLikeSolver(VolumeSolver):
    def __init__(self, config: GreedyBatchConfig | None = None) -> None:
        base = config or GreedyBatchConfig()
        self._inner = GreedyBatchVolumeSolver(
            GreedyBatchConfig(
                max_runtime_sec=base.max_runtime_sec,
                top_k_agents=base.top_k_agents,
                top_k_destinations=max(4, base.top_k_destinations),
                max_tasks_in_trip=max(500, base.max_tasks_in_trip),
                min_remaining_hours=base.min_remaining_hours,
                verbose=base.verbose,
                log_every_sec=base.log_every_sec,
                trip_log_every=base.trip_log_every,
                score_mode="vol_only",
                stochastic_mode=True,
                random_seed=303,
                exploration_noise=0.02,
                deterministic_fill=True,
                fill_time_budget_sec=20.0,
            )
        )
        self._fallback = GreedyBatchVolumeSolver(
            GreedyBatchConfig(
                max_runtime_sec=base.max_runtime_sec,
                top_k_agents=base.top_k_agents,
                top_k_destinations=max(3, base.top_k_destinations),
                max_tasks_in_trip=max(500, base.max_tasks_in_trip),
                min_remaining_hours=base.min_remaining_hours,
                verbose=base.verbose,
                log_every_sec=base.log_every_sec,
                trip_log_every=base.trip_log_every,
                score_mode="vol_per_km",
                stochastic_mode=False,
                random_seed=304,
                exploration_noise=0.0,
                deterministic_fill=True,
                fill_time_budget_sec=20.0,
            )
        )

    def solve(self, dataset: "VolumeDataset") -> AssignmentSolution:
        t0 = time.perf_counter()
        sol = self._inner.solve(dataset)
        if sol.unassigned_task_ids:
            alt = self._fallback.solve(dataset)
            if len(alt.unassigned_task_ids) < len(sol.unassigned_task_ids):
                sol = alt
        elapsed = time.perf_counter() - t0
        sol = replace(sol, runtime_sec=float(elapsed))
        return _rename_solution(sol, "volume_genetic_like_v1")


def _clone_dataset_shuffled(dataset: "VolumeDataset", rng: random.Random) -> "VolumeDataset":
    # Shallow clone is enough: tasks/agents order affects deterministic greedy decisions.
    ds = dataset.__class__(
        dataset_path=dataset.dataset_path,
        payload=dataset.payload,
        graph=dataset.graph,
        nodes=dataset.nodes,
        tasks=list(dataset.tasks),
        agents=list(dataset.agents),
        object_volume_caps=dataset.object_volume_caps,
        service_hours_by_container=dataset.service_hours_by_container,
        dist=dataset.dist,
    )
    rng.shuffle(ds.tasks)
    rng.shuffle(ds.agents)
    return ds


@dataclass(frozen=True)
class StochasticSearchConfig:
    seed: int = 42
    restarts: int = 3
    cheap_restart_sec: float = 3.0
    restart_budget_frac: float = 0.12
    fill_budget_sec: float = 20.0
    force_complete: bool = False
    budget_scale: float = 0.35


class _BaseStochasticSolver(VolumeSolver):
    def __init__(
        self,
        *,
        algorithm_name: str,
        base_config: GreedyBatchConfig,
        search_config: StochasticSearchConfig | None = None,
        top_k_dest_choices: tuple[int, ...] = (3, 4, 5),
        top_k_agent_choices: tuple[int, ...] = (20, 30, 40),
        max_tasks_trip_choices: tuple[int, ...] = (250, 350, 500),
    ) -> None:
        self.algorithm_name = algorithm_name
        self.base = base_config
        self.search = search_config or StochasticSearchConfig()
        self.top_k_dest_choices = top_k_dest_choices
        self.top_k_agent_choices = top_k_agent_choices
        self.max_tasks_trip_choices = max_tasks_trip_choices

    def _pick_best(self, current: AssignmentSolution | None, candidate: AssignmentSolution) -> AssignmentSolution:
        if current is None:
            return candidate
        c_un = len(current.unassigned_task_ids)
        n_un = len(candidate.unassigned_task_ids)
        if n_un != c_un:
            return candidate if n_un < c_un else current
        c_km = sum(t.total_km for t in current.trips)
        n_km = sum(t.total_km for t in candidate.trips)
        if abs(n_km - c_km) > 1e-9:
            return candidate if n_km < c_km else current
        return candidate if candidate.runtime_sec < current.runtime_sec else current

    def solve(self, dataset: "VolumeDataset") -> AssignmentSolution:
        t0 = time.perf_counter()
        rng = random.Random(self.search.seed)
        best: AssignmentSolution | None = None
        logs: list[str] = []
        total_budget = max(1.0, float(self.base.max_runtime_sec) * max(0.05, min(1.0, float(self.search.budget_scale))))
        restarts = max(1, int(self.search.restarts))
        cheap_restart_sec = max(0.5, float(self.search.cheap_restart_sec))
        restart_pool = max(0.0, total_budget * max(0.0, min(0.9, float(self.search.restart_budget_frac))))

        # 1) Cheap randomized restarts: quick diversification only.
        spent_restart = 0.0
        for r in range(restarts):
            elapsed = time.perf_counter() - t0
            remaining = total_budget - elapsed
            if remaining <= 1.0:
                break
            allowed = restart_pool - spent_restart
            if allowed <= 0.25:
                break
            run_budget = min(cheap_restart_sec, allowed, max(0.5, remaining - 0.5))

            local_cfg = GreedyBatchConfig(
                max_runtime_sec=run_budget,
                top_k_agents=rng.choice(self.top_k_agent_choices),
                top_k_destinations=rng.choice(self.top_k_dest_choices),
                max_tasks_in_trip=rng.choice(self.max_tasks_trip_choices),
                min_remaining_hours=self.base.min_remaining_hours,
                verbose=False,
                log_every_sec=self.base.log_every_sec,
                trip_log_every=self.base.trip_log_every,
                score_mode=self.base.score_mode,
                stochastic_mode=True,
                random_seed=rng.randint(0, 1_000_000_000),
                exploration_noise=max(0.05, self.base.exploration_noise),
                deterministic_fill=False,
                fill_time_budget_sec=0.0,
            )
            solver = GreedyBatchVolumeSolver(local_cfg)
            ds = _clone_dataset_shuffled(dataset, rng)
            sol = solver.solve(ds)
            best = self._pick_best(best, sol)
            spent_restart += run_budget
            logs.append(
                f"[{self.algorithm_name}] restart={r + 1}/{restarts} "
                f"budget={run_budget:.1f}s top_k_agents={local_cfg.top_k_agents} "
                f"top_k_dest={local_cfg.top_k_destinations} max_tasks_trip={local_cfg.max_tasks_in_trip} "
                f"assigned={len(sol.trips)} trips unassigned={len(sol.unassigned_task_ids)} "
                f"runtime={sol.runtime_sec:.2f}s"
            )
            if best is not None and not best.unassigned_task_ids:
                break

        # 2) Main run: spend most of the remaining budget once.
        elapsed = time.perf_counter() - t0
        remaining = total_budget - elapsed
        if remaining > 1.0:
            reserve_fill = min(max(3.0, float(self.search.fill_budget_sec)), max(0.0, remaining * 0.4))
            main_budget = max(1.0, remaining - reserve_fill)
            main_cfg = GreedyBatchConfig(
                max_runtime_sec=main_budget,
                top_k_agents=max(self.base.top_k_agents, 30),
                top_k_destinations=max(self.base.top_k_destinations, 4),
                max_tasks_in_trip=max(self.base.max_tasks_in_trip, 500),
                min_remaining_hours=self.base.min_remaining_hours,
                verbose=False,
                log_every_sec=self.base.log_every_sec,
                trip_log_every=self.base.trip_log_every,
                score_mode=self.base.score_mode,
                stochastic_mode=False,
                random_seed=rng.randint(0, 1_000_000_000),
                exploration_noise=0.0,
                deterministic_fill=False,
                fill_time_budget_sec=0.0,
            )
            main_sol = GreedyBatchVolumeSolver(main_cfg).solve(dataset)
            best = self._pick_best(best, main_sol)
            logs.append(
                f"[{self.algorithm_name}] main_run budget={main_budget:.1f}s "
                f"assigned={len(main_sol.trips)} trips unassigned={len(main_sol.unassigned_task_ids)} "
                f"runtime={main_sol.runtime_sec:.2f}s"
            )

        assert best is not None
        # Guaranteed completion pass: if stochastic restarts leave a tiny tail,
        # run one deterministic fill pass with aggressive batching.
        if self.search.force_complete and best.unassigned_task_ids:
            elapsed = time.perf_counter() - t0
            remaining = total_budget - elapsed
            if remaining <= 1.0:
                elapsed_total = time.perf_counter() - t0
                return replace(
                    best,
                    algorithm=self.algorithm_name,
                    runtime_sec=float(elapsed_total),
                    solver_logs=tuple(logs) + tuple(best.solver_logs),
                )
            fill_cfg = GreedyBatchConfig(
                max_runtime_sec=max(1.0, min(remaining, float(self.search.fill_budget_sec))),
                top_k_agents=max(30, self.base.top_k_agents),
                top_k_destinations=max(4, self.base.top_k_destinations),
                max_tasks_in_trip=max(500, self.base.max_tasks_in_trip),
                min_remaining_hours=self.base.min_remaining_hours,
                verbose=False,
                log_every_sec=self.base.log_every_sec,
                trip_log_every=self.base.trip_log_every,
                score_mode="vol_per_km",
                stochastic_mode=False,
                random_seed=self.search.seed,
                exploration_noise=0.0,
                deterministic_fill=True,
                fill_time_budget_sec=max(1.0, min(remaining, float(self.search.fill_budget_sec))),
            )
            fill = GreedyBatchVolumeSolver(fill_cfg).solve(dataset)
            logs.append(
                f"[{self.algorithm_name}] deterministic_fill "
                f"assigned={len(fill.trips)} trips unassigned={len(fill.unassigned_task_ids)} "
                f"runtime={fill.runtime_sec:.2f}s"
            )
            best = self._pick_best(best, fill)

        elapsed_total = time.perf_counter() - t0
        return replace(
            best,
            algorithm=self.algorithm_name,
            runtime_sec=float(elapsed_total),
            solver_logs=tuple(logs) + tuple(best.solver_logs),
        )


class VolumeGapVRPStochasticSolver(_BaseStochasticSolver):
    def __init__(
        self,
        config: GreedyBatchConfig | None = None,
        search_config: StochasticSearchConfig | None = None,
    ) -> None:
        base = config or GreedyBatchConfig()
        base = replace(base, score_mode="vol_per_km", stochastic_mode=True, exploration_noise=max(0.03, base.exploration_noise))
        super().__init__(
            algorithm_name="volume_gap_vrp_stoch_v1",
            base_config=base,
            search_config=search_config
            or StochasticSearchConfig(
                seed=101, restarts=3, cheap_restart_sec=1.5, restart_budget_frac=0.12, fill_budget_sec=10.0, force_complete=False, budget_scale=0.50
            ),
            top_k_dest_choices=(3, 4, 5, 6),
            top_k_agent_choices=(20, 30, 40),
            max_tasks_trip_choices=(200, 300, 500),
        )


class VolumeMilpStochasticSolver(_BaseStochasticSolver):
    def __init__(
        self,
        config: GreedyBatchConfig | None = None,
        search_config: StochasticSearchConfig | None = None,
    ) -> None:
        base = config or GreedyBatchConfig()
        base = replace(base, score_mode="min_km", stochastic_mode=True, exploration_noise=max(0.03, base.exploration_noise))
        super().__init__(
            algorithm_name="volume_milp_stoch_v1",
            base_config=base,
            search_config=search_config
            or StochasticSearchConfig(
                seed=202, restarts=3, cheap_restart_sec=1.5, restart_budget_frac=0.12, fill_budget_sec=10.0, force_complete=False, budget_scale=0.50
            ),
            top_k_dest_choices=(2, 3, 4),
            top_k_agent_choices=(30, 40, 60),
            max_tasks_trip_choices=(250, 400, 600),
        )


class VolumeGeneticStochasticSolver(_BaseStochasticSolver):
    def __init__(
        self,
        config: GreedyBatchConfig | None = None,
        search_config: StochasticSearchConfig | None = None,
    ) -> None:
        base = config or GreedyBatchConfig()
        base = replace(base, score_mode="vol_only", stochastic_mode=True, exploration_noise=max(0.06, base.exploration_noise))
        super().__init__(
            algorithm_name="volume_genetic_stoch_v1",
            base_config=base,
            search_config=search_config
            or StochasticSearchConfig(
                seed=303, restarts=4, cheap_restart_sec=1.5, restart_budget_frac=0.15, fill_budget_sec=10.0, force_complete=False, budget_scale=0.45
            ),
            top_k_dest_choices=(2, 3, 4, 5, 6),
            top_k_agent_choices=(15, 20, 30, 40),
            max_tasks_trip_choices=(150, 250, 350, 500),
        )
