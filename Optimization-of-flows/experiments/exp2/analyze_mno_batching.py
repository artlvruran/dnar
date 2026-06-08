from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from flowopt import core
from flowopt.dataset import CONTAINER_TO_VEHICLE_TYPES, RoutingDataset, Task, dataset_from_dict


@dataclass
class BatchSummary:
    mode: str
    capacity_policy: str
    total_tasks_before: int
    total_tasks_after: int
    reduction_ratio: float
    total_mass_tons: float
    avg_batch_fill_ratio: float
    full_batches: int
    partial_batches: int
    groups_total: int
    groups_ge_capacity: int
    agents_total: int
    trips_per_agent: float
    km_lower_bound: float
    km_capacity_total: float
    km_lb_to_capacity_ratio: float
    hours_lower_bound: float
    hours_capacity_total: float
    hours_lb_to_capacity_ratio: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "capacity_policy": self.capacity_policy,
            "total_tasks_before": self.total_tasks_before,
            "total_tasks_after": self.total_tasks_after,
            "reduction_ratio": round(self.reduction_ratio, 4),
            "total_mass_tons": round(self.total_mass_tons, 3),
            "avg_batch_fill_ratio": round(self.avg_batch_fill_ratio, 4),
            "full_batches": self.full_batches,
            "partial_batches": self.partial_batches,
            "groups_total": self.groups_total,
            "groups_ge_capacity": self.groups_ge_capacity,
            "agents_total": self.agents_total,
            "trips_per_agent": round(self.trips_per_agent, 3),
            "km_lower_bound": round(self.km_lower_bound, 3),
            "km_capacity_total": round(self.km_capacity_total, 3),
            "km_lb_to_capacity_ratio": round(self.km_lb_to_capacity_ratio, 4),
            "hours_lower_bound": round(self.hours_lower_bound, 3),
            "hours_capacity_total": round(self.hours_capacity_total, 3),
            "hours_lb_to_capacity_ratio": round(self.hours_lb_to_capacity_ratio, 4),
        }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    if len(vals) == 1:
        return float(vals[0])
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(vals[lo])
    w = pos - lo
    return float(vals[lo] * (1.0 - w) + vals[hi] * w)


def _capacity_policy_values(dataset: RoutingDataset) -> dict[str, dict[str, float]]:
    agents = list(dataset.fleet.agents.values())
    out: dict[str, dict[str, float]] = {}
    for container, allowed_types in CONTAINER_TO_VEHICLE_TYPES.items():
        caps = [a.capacity_tons for a in agents if a.vehicle_type in allowed_types and a.capacity_tons > 0]
        if not caps:
            out[container] = {"max": 0.0, "p75": 0.0, "median": 0.0, "min": 0.0}
            continue
        out[container] = {
            "max": float(max(caps)),
            "p75": _percentile(caps, 0.75),
            "median": _percentile(caps, 0.50),
            "min": float(min(caps)),
        }
    return out


def _container_speed(dataset: RoutingDataset) -> dict[str, float]:
    agents = list(dataset.fleet.agents.values())
    out: dict[str, float] = {}
    for container, allowed_types in CONTAINER_TO_VEHICLE_TYPES.items():
        speeds = [
            core.AVG_SPEED_KMPH_BY_TYPE[a.vehicle_type]
            for a in agents
            if a.vehicle_type in allowed_types
        ]
        out[container] = _percentile(speeds, 0.50) if speeds else 24.0
    return out


def _distance_lookup(dataset: RoutingDataset) -> dict[tuple[str, str], float]:
    graph = core.build_nx_graph(dataset)
    rev = graph.reverse(copy=False)

    pairs = {(t.source_node_id, t.destination_node_id) for t in dataset.tasks}
    by_dst: dict[str, set[str]] = defaultdict(set)
    for src, dst in pairs:
        by_dst[dst].add(src)

    lookup: dict[tuple[str, str], float] = {}
    for dst, sources in by_dst.items():
        dist_map, _ = nx.single_source_dijkstra(rev, dst, weight="distance_km")
        for src in sources:
            if src in dist_map:
                lookup[(src, dst)] = float(dist_map[src])
    return lookup


def _aggregate_groups(tasks: list[Task], mode: str) -> dict[tuple[str, ...], dict[str, Any]]:
    groups: dict[tuple[str, ...], dict[str, Any]] = {}
    if mode not in {"source_container_dest", "source_container"}:
        raise ValueError(f"Unsupported mode: {mode}")

    for t in tasks:
        if mode == "source_container_dest":
            key = (t.source_node_id, t.container_type, t.destination_node_id)
        else:
            key = (t.source_node_id, t.container_type)

        cur = groups.get(key)
        if cur is None:
            cur = {
                "mass": 0.0,
                "container": t.container_type,
                "source": t.source_node_id,
                "dest_mass": defaultdict(float),
            }
            groups[key] = cur

        cur["mass"] += float(t.mass_tons)
        cur["dest_mass"][t.destination_node_id] += float(t.mass_tons)

    return groups


def _group_distance(group: dict[str, Any], pair_dist: dict[tuple[str, str], float]) -> float:
    source = group["source"]
    dest_mass: dict[str, float] = group["dest_mass"]
    total_mass = sum(dest_mass.values())
    if total_mass <= 0:
        return 0.0

    dist_sum = 0.0
    for dst, mass in dest_mass.items():
        d = pair_dist.get((source, dst), float("inf"))
        if math.isinf(d):
            continue
        dist_sum += d * mass
    return dist_sum / total_mass


def analyze(
    *,
    dataset: RoutingDataset,
    mode: str,
    capacity_policy: str,
    pair_dist: dict[tuple[str, str], float],
    cap_policy_values: dict[str, dict[str, float]],
    speed_by_container: dict[str, float],
) -> BatchSummary:
    groups = _aggregate_groups(dataset.tasks, mode)

    total_mass = 0.0
    total_batches = 0
    full_batches = 0
    partial_batches = 0
    groups_ge_cap = 0
    capacity_mass_sum = 0.0

    km_lb = 0.0
    hours_lb = 0.0

    for info in groups.values():
        mass = float(info["mass"])
        container = str(info["container"])
        cap = cap_policy_values.get(container, {}).get(capacity_policy, 0.0)
        if cap <= 1e-9:
            continue

        batches = int(math.ceil(mass / cap - 1e-12))
        whole = int(mass // cap)
        rem = mass - whole * cap

        total_mass += mass
        total_batches += batches
        full_batches += whole
        partial_batches += 1 if rem > 1e-9 else 0
        groups_ge_cap += 1 if mass + 1e-9 >= cap else 0
        capacity_mass_sum += batches * cap

        dist = _group_distance(info, pair_dist)
        km_lb += batches * dist

        speed = max(1e-6, float(speed_by_container.get(container, 24.0)))
        service_h = core.SERVICE_HOURS_BY_CONTAINER.get(container, 0.25)
        hours_lb += batches * (dist / speed + service_h)

    agent_states = core.initialize_agent_states(dataset, dataset.metadata if isinstance(dataset.metadata, dict) else {})
    if not agent_states:
        # fallback when metadata lacks depots; limits still from fleet
        agent_states = {
            aid: core.AgentState(
                agent_id=a.agent_id,
                vehicle_type=a.vehicle_type,
                capacity_tons=a.capacity_tons,
                is_compact=a.is_compact,
                depot_node=None,
                current_node=None,
            )
            for aid, a in dataset.fleet.agents.items()
        }

    km_capacity_total = sum(core.MAX_DAILY_KM_BY_TYPE[s.vehicle_type] for s in agent_states.values())
    hours_capacity_total = sum(core.MAX_SHIFT_HOURS_BY_TYPE[s.vehicle_type] for s in agent_states.values())

    reduction_ratio = (len(dataset.tasks) / total_batches) if total_batches > 0 else 0.0
    avg_fill = (total_mass / capacity_mass_sum) if capacity_mass_sum > 0 else 0.0
    trips_per_agent = (total_batches / len(agent_states)) if agent_states else float("inf")

    return BatchSummary(
        mode=mode,
        capacity_policy=capacity_policy,
        total_tasks_before=len(dataset.tasks),
        total_tasks_after=total_batches,
        reduction_ratio=reduction_ratio,
        total_mass_tons=total_mass,
        avg_batch_fill_ratio=avg_fill,
        full_batches=full_batches,
        partial_batches=partial_batches,
        groups_total=len(groups),
        groups_ge_capacity=groups_ge_cap,
        agents_total=len(agent_states),
        trips_per_agent=trips_per_agent,
        km_lower_bound=km_lb,
        km_capacity_total=km_capacity_total,
        km_lb_to_capacity_ratio=(km_lb / km_capacity_total) if km_capacity_total > 0 else float("inf"),
        hours_lower_bound=hours_lb,
        hours_capacity_total=hours_capacity_total,
        hours_lb_to_capacity_ratio=(hours_lb / hours_capacity_total) if hours_capacity_total > 0 else float("inf"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze MNO-level batching effect (exp2)")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("src/data/real/full_29k/dataset_real_spb_full_29k.json"),
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="source_container_dest,source_container",
        help="Comma-separated modes",
    )
    parser.add_argument(
        "--capacity-policies",
        type=str,
        default="max,p75,median",
        help="Comma-separated: max,p75,median,min",
    )
    args = parser.parse_args()

    payload = json.loads(args.dataset_path.read_text(encoding="utf-8"))
    dataset = dataset_from_dict(payload)

    pair_dist = _distance_lookup(dataset)
    cap_values = _capacity_policy_values(dataset)
    speed_by_container = _container_speed(dataset)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    policies = [p.strip() for p in args.capacity_policies.split(",") if p.strip()]

    rows: list[dict[str, Any]] = []
    by_container_rows: list[dict[str, Any]] = []

    for mode in modes:
        for policy in policies:
            summary = analyze(
                dataset=dataset,
                mode=mode,
                capacity_policy=policy,
                pair_dist=pair_dist,
                cap_policy_values=cap_values,
                speed_by_container=speed_by_container,
            )
            rows.append(summary.as_dict())

            groups = _aggregate_groups(dataset.tasks, mode)
            batch_by_container: Counter[str] = Counter()
            mass_by_container: defaultdict[str, float] = defaultdict(float)
            agents_by_container: dict[str, int] = {}

            for container, allowed_types in CONTAINER_TO_VEHICLE_TYPES.items():
                agents_by_container[container] = sum(
                    1 for a in dataset.fleet.agents.values() if a.vehicle_type in allowed_types
                )

            for info in groups.values():
                container = str(info["container"])
                cap = cap_values.get(container, {}).get(policy, 0.0)
                if cap <= 1e-9:
                    continue
                mass = float(info["mass"])
                batches = int(math.ceil(mass / cap - 1e-12))
                batch_by_container[container] += batches
                mass_by_container[container] += mass

            for container in sorted(batch_by_container):
                b = int(batch_by_container[container])
                a = int(agents_by_container.get(container, 0))
                by_container_rows.append(
                    {
                        "mode": mode,
                        "capacity_policy": policy,
                        "container": container,
                        "batched_tasks": b,
                        "compatible_agents": a,
                        "trips_per_compatible_agent": round((b / a), 3) if a > 0 else None,
                        "mass_tons": round(mass_by_container[container], 3),
                    }
                )

    df = pd.DataFrame(rows)
    print("\n=== EXP2: Batching summary ===")
    print(df.to_string(index=False))

    df_c = pd.DataFrame(by_container_rows)
    print("\n=== EXP2: By container ===")
    print(df_c.to_string(index=False))

    out_dir = Path("experiments/exp2/local")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_json = out_dir / f"exp2_batching_{stamp}.json"
    out_csv = out_dir / f"exp2_batching_{stamp}.csv"
    out_csv_c = out_dir / f"exp2_batching_by_container_{stamp}.csv"

    out_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": str(args.dataset_path.resolve()),
        "modes": modes,
        "capacity_policies": policies,
        "summary": rows,
        "by_container": by_container_rows,
        "capacity_policy_values": cap_values,
    }
    out_json.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    df.to_csv(out_csv, index=False)
    df_c.to_csv(out_csv_c, index=False)

    print(f"\nSaved JSON: {out_json.resolve()}")
    print(f"Saved CSV : {out_csv.resolve()}")
    print(f"Saved CSV : {out_csv_c.resolve()}")


if __name__ == "__main__":
    main()
