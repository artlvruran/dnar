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

        for route in self.routes:
            if route.agent_id not in self.fleet.agents:
                raise ValueError(f"Unknown route agent: {route.agent_id}")
            if len(route.path) < 2:
                raise ValueError(f"Route path must have at least two nodes: {route.route_id}")

            agent = self.fleet.agents[route.agent_id]
            route_mass = 0.0

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

                source_node = self.graph.nodes[task.source_node_id]
                if source_node.center and not agent.is_compact:
                    raise ValueError(
                        f"Route {route.route_id}: center MNO can only be served by compact agent"
                    )

                object_day_load[task.destination_node_id] = (
                    object_day_load.get(task.destination_node_id, 0.0) + task.mass_tons
                )

            if route_mass > agent.capacity_tons:
                raise ValueError(
                    f"Route {route.route_id} exceeds agent capacity "
                    f"({route_mass:.2f} > {agent.capacity_tons:.2f})"
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
    nodes = {
        raw["node_id"]: GraphNode(
            node_id=raw["node_id"],
            kind=raw["kind"],
            x=raw["x"],
            y=raw["y"],
            center=raw.get("center", False),
            container_types=tuple(raw.get("container_types", [])),
            daily_mass_tons=raw.get("daily_mass_tons", 0.0),
            object_day_capacity_tons=raw.get("object_day_capacity_tons", 0.0),
            object_year_capacity_tons=raw.get("object_year_capacity_tons", 0.0),
        )
        for raw in data["graph"]["nodes"]
    }
    edges = [
        GraphEdge(
            source_id=raw["source_id"],
            target_id=raw["target_id"],
            distance_km=raw["distance_km"],
            allowed_vehicle_types=tuple(raw["allowed_vehicle_types"]),
        )
        for raw in data["graph"]["edges"]
    ]
    agents = {
        raw["agent_id"]: Agent(
            agent_id=raw["agent_id"],
            vehicle_type=raw["vehicle_type"],
            capacity_tons=raw["capacity_tons"],
            is_compact=raw.get("is_compact", False),
        )
        for raw in data["agents"]
    }
    tasks = [
        Task(
            task_id=raw["task_id"],
            source_node_id=raw["source_node_id"],
            destination_node_id=raw["destination_node_id"],
            container_type=raw["container_type"],
            mass_tons=raw["mass_tons"],
            periodicity=raw["periodicity"],
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
