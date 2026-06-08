from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Any


CONTAINER_TO_VEHICLE_TYPES: dict[str, set[str]] = {
    "Type1": {"Type1", "Type2", "Type1-2"},
    "Type2": {"Type1", "Type2", "Type1-2"},
    "Type3": {"Type3"},
    "Type4": {"Type4"},
    "TypeRNO": {"TypeRNO"},
    "TypeO2P": {"TypeO2P"},
    "A": {"VT_A", "VT_AB", "VT_ABD", "VT_AD"},
    "B": {"VT_AB", "VT_ABD"},
    "C": {"VT_C", "VT_CD"},
    "D": {"VT_ABD", "VT_AD", "VT_CD"},
}


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    kind: str  # mno | object1 | object2
    x: float
    y: float
    center: bool = False
    container_types: tuple[str, ...] = ()
    daily_mass_tons: float = 0.0
    object_day_capacity_tons: float = 0.0
    object_year_capacity_tons: float = 0.0
    object_day_capacity_volume_m3: float = 0.0
    density_kg_m3: float = 0.0
    object_alias: str | None = None


@dataclass(frozen=True)
class GraphEdge:
    source_id: str
    target_id: str
    distance_km: float
    allowed_vehicle_types: tuple[str, ...]


@dataclass
class RoadGraph:
    nodes: dict[str, GraphNode]
    edges: list[GraphEdge]

    def validate(self) -> None:
        if not self.nodes:
            raise ValueError("Road graph must contain at least one node.")
        edge_keys: set[tuple[str, str]] = set()
        for edge in self.edges:
            if edge.source_id not in self.nodes:
                raise ValueError(f"Unknown edge source: {edge.source_id}")
            if edge.target_id not in self.nodes:
                raise ValueError(f"Unknown edge target: {edge.target_id}")
            if edge.distance_km <= 0:
                raise ValueError("Edge distance must be positive.")
            if not edge.allowed_vehicle_types:
                raise ValueError("Each edge must declare allowed vehicle types.")
            edge_key = (edge.source_id, edge.target_id)
            if edge_key in edge_keys:
                raise ValueError(f"Duplicate directed edge: {edge_key}")
            edge_keys.add(edge_key)

    def edge(self, source_id: str, target_id: str) -> GraphEdge | None:
        for edge in self.edges:
            if edge.source_id == source_id and edge.target_id == target_id:
                return edge
        return None


@dataclass(frozen=True)
class Agent:
    agent_id: str
    vehicle_type: str
    capacity_tons: float
    is_compact: bool = False
    body_volume_m3: float = 0.0
    compaction_coeff: float = 1.0
    max_raw_volume_m3: float = 0.0
    cap_container_types: tuple[str, ...] = ()

    @property
    def raw_volume_limit_m3(self) -> float:
        if self.max_raw_volume_m3 > 0:
            return self.max_raw_volume_m3
        if self.body_volume_m3 > 0 and self.compaction_coeff > 0:
            return self.body_volume_m3 * self.compaction_coeff
        return 0.0

    def supports_container(self, container_type: str) -> bool:
        if not self.cap_container_types:
            return True
        return container_type in self.cap_container_types


@dataclass
class AgentsFleet:
    agents: dict[str, Agent]

    def validate(self) -> None:
        if not self.agents:
            raise ValueError("At least one agent is required.")
        for agent in self.agents.values():
            if agent.capacity_tons <= 0:
                raise ValueError(f"Agent has non-positive capacity: {agent.agent_id}")


@dataclass(frozen=True)
class Task:
    task_id: str
    source_node_id: str
    destination_node_id: str
    container_type: str
    mass_tons: float
    periodicity: str
    mass_kg: float = 0.0
    volume_raw_m3: float = 0.0
    density_kg_m3_assumed: float = 0.0
    volume_body_m3_by_vehicle_type: tuple[tuple[str, float], ...] = ()
    compatible_vehicle_types: tuple[str, ...] = ()
    source_center: bool = False
    source_container_types: tuple[str, ...] = ()
    source_node_daily_mass_tons: float = 0.0
    destination_object_alias: str | None = None
    destination_object_day_capacity_tons: float = 0.0
    destination_object_day_capacity_volume_m3: float = 0.0
    periodicity_days: int = 1
    periodicity_weekly_frequency: float = 1.0

    def body_volume_for_vehicle(self, vehicle_type: str) -> float:
        if self.volume_body_m3_by_vehicle_type:
            for vt, value in self.volume_body_m3_by_vehicle_type:
                if vt == vehicle_type:
                    return float(value)
        return 0.0


@dataclass(frozen=True)
class Route:
    route_id: str
    agent_id: str
    path: tuple[str, ...]
    task_ids: tuple[str, ...]


@dataclass
class RoutingDataset:
    graph: RoadGraph
    fleet: AgentsFleet
    tasks: list[Task]
    routes: list[Route]
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.graph.validate()
        self.fleet.validate()

        task_by_id = {task.task_id: task for task in self.tasks}
        if len(task_by_id) != len(self.tasks):
            raise ValueError("Task IDs must be unique.")

        for task in self.tasks:
            if task.mass_tons < 0:
                raise ValueError(f"Task mass must be non-negative: {task.task_id}")
            if task.container_type not in CONTAINER_TO_VEHICLE_TYPES:
                raise ValueError(f"Unknown container type: {task.container_type}")
            if task.volume_raw_m3 < 0:
                raise ValueError(f"Task raw volume must be non-negative: {task.task_id}")
            if task.source_node_id not in self.graph.nodes:
                raise ValueError(f"Unknown task source node: {task.source_node_id}")
            if task.destination_node_id not in self.graph.nodes:
                raise ValueError(f"Unknown task destination node: {task.destination_node_id}")

            source = self.graph.nodes[task.source_node_id]
            if source.kind == "mno" and task.container_type not in source.container_types:
                raise ValueError(
                    f"Task {task.task_id} container type is incompatible with source MNO"
                )

        covered_task_ids: list[str] = []
        object_day_load: dict[str, float] = {}
        object_day_volume: dict[str, float] = {}

        for route in self.routes:
            if route.agent_id not in self.fleet.agents:
                raise ValueError(f"Unknown route agent: {route.agent_id}")
            if len(route.path) < 2:
                raise ValueError(f"Route path must have at least two nodes: {route.route_id}")

            agent = self.fleet.agents[route.agent_id]
            route_mass = 0.0
            route_raw_volume = 0.0
            route_body_volume = 0.0

            for i in range(len(route.path) - 1):
                edge = self.graph.edge(route.path[i], route.path[i + 1])
                if edge is None:
                    raise ValueError(
                        f"Route {route.route_id} uses non-existing edge "
                        f"{route.path[i]} -> {route.path[i + 1]}"
                    )
                if agent.vehicle_type not in edge.allowed_vehicle_types:
                    raise ValueError(
                        f"Route {route.route_id}: edge disallows vehicle type "
                        f"{agent.vehicle_type}"
                    )

            for task_id in route.task_ids:
                if task_id not in task_by_id:
                    raise ValueError(f"Route references unknown task ID: {task_id}")
                task = task_by_id[task_id]
                covered_task_ids.append(task_id)
                route_mass += task.mass_tons

                if task.destination_node_id != route.path[-1]:
                    raise ValueError(
                        f"Task {task.task_id} destination differs from route endpoint."
                    )

                allowed_vehicle_types = CONTAINER_TO_VEHICLE_TYPES[task.container_type]
                if agent.vehicle_type not in allowed_vehicle_types:
                    raise ValueError(
                        f"Route {route.route_id}: agent {agent.agent_id} vehicle type "
                        f"is incompatible with task {task.task_id}"
                    )
                if task.compatible_vehicle_types and agent.vehicle_type not in task.compatible_vehicle_types:
                    raise ValueError(
                        f"Route {route.route_id}: agent {agent.agent_id} is not in "
                        f"task-compatible vehicle set for task {task.task_id}"
                    )
                if not agent.supports_container(task.container_type):
                    raise ValueError(
                        f"Route {route.route_id}: agent {agent.agent_id} container capability "
                        f"is incompatible with task {task.task_id}"
                    )

                source_node = self.graph.nodes[task.source_node_id]
                if source_node.center and not agent.is_compact:
                    raise ValueError(
                        f"Route {route.route_id}: center MNO can only be served by compact agent"
                    )

                object_day_load[task.destination_node_id] = (
                    object_day_load.get(task.destination_node_id, 0.0) + task.mass_tons
                )
                object_day_volume[task.destination_node_id] = (
                    object_day_volume.get(task.destination_node_id, 0.0) + task.volume_raw_m3
                )
                route_raw_volume += task.volume_raw_m3
                body_volume = task.body_volume_for_vehicle(agent.vehicle_type)
                if body_volume <= 0 and task.volume_raw_m3 > 0 and agent.compaction_coeff > 0:
                    body_volume = task.volume_raw_m3 / agent.compaction_coeff
                route_body_volume += body_volume

            if route_mass > agent.capacity_tons:
                raise ValueError(
                    f"Route {route.route_id} exceeds agent capacity "
                    f"({route_mass:.2f} > {agent.capacity_tons:.2f})"
                )
            raw_limit = agent.raw_volume_limit_m3
            if raw_limit > 0 and route_raw_volume > raw_limit + 1e-9:
                raise ValueError(
                    f"Route {route.route_id} exceeds agent raw-volume limit "
                    f"({route_raw_volume:.3f} > {raw_limit:.3f})"
                )
            if agent.body_volume_m3 > 0 and route_body_volume > agent.body_volume_m3 + 1e-9:
                raise ValueError(
                    f"Route {route.route_id} exceeds agent body-volume limit "
                    f"({route_body_volume:.3f} > {agent.body_volume_m3:.3f})"
                )

        # C1: each task served exactly once.
        if set(covered_task_ids) != set(task_by_id.keys()):
            raise ValueError("Not all tasks are covered by routes.")
        if len(covered_task_ids) != len(task_by_id):
            raise ValueError("At least one task is served multiple times.")

        # C3: object daily capacities.
        for node_id, load in object_day_load.items():
            node = self.graph.nodes[node_id]
            if node.kind.startswith("object") and node.object_day_capacity_tons > 0:
                if load > node.object_day_capacity_tons:
                    raise ValueError(
                        f"Object daily capacity exceeded for {node_id}: "
                        f"{load:.2f} > {node.object_day_capacity_tons:.2f}"
                    )
        for node_id, volume in object_day_volume.items():
            node = self.graph.nodes[node_id]
            if node.kind.startswith("object") and node.object_day_capacity_volume_m3 > 0:
                if volume > node.object_day_capacity_volume_m3:
                    raise ValueError(
                        f"Object daily volume capacity exceeded for {node_id}: "
                        f"{volume:.3f} > {node.object_day_capacity_volume_m3:.3f}"
                    )

    def transport_work(self) -> float:
        task_by_id = {task.task_id: task for task in self.tasks}
        total_tr = 0.0
        for route in self.routes:
            route_distance = 0.0
            for i in range(len(route.path) - 1):
                edge = self.graph.edge(route.path[i], route.path[i + 1])
                if edge is None:
                    raise ValueError(
                        f"Route {route.route_id} uses unknown edge "
                        f"{route.path[i]} -> {route.path[i + 1]}"
                    )
                route_distance += edge.distance_km
            route_mass = sum(task_by_id[task_id].mass_tons for task_id in route.task_ids)
            total_tr += route_mass * route_distance
        return total_tr

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "graph": {
                "nodes": [
                    {
                        "node_id": node.node_id,
                        "kind": node.kind,
                        "x": node.x,
                        "y": node.y,
                        "center": node.center,
                        "container_types": list(node.container_types),
                        "daily_mass_tons": node.daily_mass_tons,
                        "object_day_capacity_tons": node.object_day_capacity_tons,
                        "object_year_capacity_tons": node.object_year_capacity_tons,
                        "object_day_capacity_volume_m3": node.object_day_capacity_volume_m3,
                        "density_kg_m3": node.density_kg_m3,
                        "object_alias": node.object_alias,
                    }
                    for node in self.graph.nodes.values()
                ],
                "edges": [
                    {
                        "source_id": edge.source_id,
                        "target_id": edge.target_id,
                        "distance_km": edge.distance_km,
                        "allowed_vehicle_types": list(edge.allowed_vehicle_types),
                    }
                    for edge in self.graph.edges
                ],
            },
            "agents": [
                {
                    "agent_id": agent.agent_id,
                    "vehicle_type": agent.vehicle_type,
                    "capacity_tons": agent.capacity_tons,
                    "is_compact": agent.is_compact,
                    "body_volume_m3": agent.body_volume_m3,
                    "compaction_coeff": agent.compaction_coeff,
                    "max_raw_volume_m3": agent.max_raw_volume_m3,
                    "cap_container_types": list(agent.cap_container_types),
                }
                for agent in self.fleet.agents.values()
            ],
            "tasks": [
                {
                    "task_id": task.task_id,
                    "source_node_id": task.source_node_id,
                    "destination_node_id": task.destination_node_id,
                    "container_type": task.container_type,
                    "mass_tons": task.mass_tons,
                    "periodicity": task.periodicity,
                    "mass_kg": task.mass_kg,
                    "volume_raw_m3": task.volume_raw_m3,
                    "density_kg_m3_assumed": task.density_kg_m3_assumed,
                    "volume_body_m3_by_vehicle_type": {
                        k: v for k, v in task.volume_body_m3_by_vehicle_type
                    },
                    "compatible_vehicle_types": list(task.compatible_vehicle_types),
                    "source_center": task.source_center,
                    "source_container_types": list(task.source_container_types),
                    "source_node_daily_mass_tons": task.source_node_daily_mass_tons,
                    "destination_object_alias": task.destination_object_alias,
                    "destination_object_day_capacity_tons": task.destination_object_day_capacity_tons,
                    "destination_object_day_capacity_volume_m3": task.destination_object_day_capacity_volume_m3,
                    "periodicity_days": task.periodicity_days,
                    "periodicity_weekly_frequency": task.periodicity_weekly_frequency,
                }
                for task in self.tasks
            ],
            "routes": [
                {
                    "route_id": route.route_id,
                    "agent_id": route.agent_id,
                    "path": list(route.path),
                    "task_ids": list(route.task_ids),
                }
                for route in self.routes
            ],
            "derived": {
                "transport_work_ton_km": round(self.transport_work(), 3),
            },
        }

    def save_json(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def dataset_from_dict(data: dict[str, Any]) -> RoutingDataset:
    def _f(value: Any, default: float = 0.0) -> float:
        if value is None:
            return float(default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _i(value: Any, default: int = 0) -> int:
        if value is None:
            return int(default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    def _agent_cap_types(raw: dict[str, Any]) -> tuple[str, ...]:
        explicit = raw.get("cap_container_types")
        if explicit:
            return tuple(sorted(str(x) for x in explicit))
        out: list[str] = []
        for c in ("A", "B", "C", "D"):
            if bool(raw.get(f"cap_container_{c}", False)):
                out.append(c)
        return tuple(out)

    nodes = {
        raw["node_id"]: GraphNode(
            node_id=raw["node_id"],
            kind=raw["kind"],
            x=_f(raw["x"]),
            y=_f(raw["y"]),
            center=raw.get("center", False),
            container_types=tuple(raw.get("container_types", [])),
            daily_mass_tons=_f(raw.get("daily_mass_tons", 0.0)),
            object_day_capacity_tons=_f(raw.get("object_day_capacity_tons", 0.0)),
            object_year_capacity_tons=_f(raw.get("object_year_capacity_tons", 0.0)),
            object_day_capacity_volume_m3=_f(raw.get("object_day_capacity_volume_m3", 0.0)),
            density_kg_m3=_f(raw.get("density_kg_m3", 0.0)),
            object_alias=raw.get("object_alias"),
        )
        for raw in data["graph"]["nodes"]
    }
    edges = [
        GraphEdge(
            source_id=raw["source_id"],
            target_id=raw["target_id"],
            distance_km=_f(raw["distance_km"]),
            allowed_vehicle_types=tuple(raw["allowed_vehicle_types"]),
        )
        for raw in data["graph"]["edges"]
    ]
    agents = {
        raw["agent_id"]: Agent(
            agent_id=raw["agent_id"],
            vehicle_type=raw["vehicle_type"],
            capacity_tons=_f(raw["capacity_tons"]),
            is_compact=raw.get("is_compact", False),
            body_volume_m3=_f(raw.get("body_volume_m3", 0.0)),
            compaction_coeff=_f(raw.get("compaction_coeff", 1.0), 1.0),
            max_raw_volume_m3=_f(raw.get("max_raw_volume_m3", 0.0)),
            cap_container_types=_agent_cap_types(raw),
        )
        for raw in data["agents"]
    }
    tasks = [
        Task(
            task_id=raw["task_id"],
            source_node_id=raw["source_node_id"],
            destination_node_id=raw["destination_node_id"],
            container_type=raw["container_type"],
            mass_tons=_f(raw["mass_tons"]),
            periodicity=raw["periodicity"],
            mass_kg=_f(raw.get("mass_kg", 0.0)),
            volume_raw_m3=_f(raw.get("volume_raw_m3", 0.0)),
            density_kg_m3_assumed=_f(raw.get("density_kg_m3_assumed", 0.0)),
            volume_body_m3_by_vehicle_type=tuple(
                sorted((str(k), _f(v)) for k, v in raw.get("volume_body_m3_by_vehicle_type", {}).items())
            ),
            compatible_vehicle_types=tuple(raw.get("compatible_vehicle_types", [])),
            source_center=bool(raw.get("source_center", False)),
            source_container_types=tuple(raw.get("source_container_types", [])),
            source_node_daily_mass_tons=_f(raw.get("source_node_daily_mass_tons", 0.0)),
            destination_object_alias=raw.get("destination_object_alias"),
            destination_object_day_capacity_tons=_f(raw.get("destination_object_day_capacity_tons", 0.0)),
            destination_object_day_capacity_volume_m3=_f(
                raw.get("destination_object_day_capacity_volume_m3", 0.0)
            ),
            periodicity_days=_i(raw.get("periodicity_days", 1), 1),
            periodicity_weekly_frequency=_f(raw.get("periodicity_weekly_frequency", 1.0), 1.0),
        )
        for raw in data["tasks"]
    ]
    routes = [
        Route(
            route_id=raw["route_id"],
            agent_id=raw["agent_id"],
            path=tuple(raw["path"]),
            task_ids=tuple(raw["task_ids"]),
        )
        for raw in data["routes"]
    ]
    return RoutingDataset(
        graph=RoadGraph(nodes=nodes, edges=edges),
        fleet=AgentsFleet(agents=agents),
        tasks=tasks,
        routes=routes,
        metadata=data.get("metadata", {}),
    )
