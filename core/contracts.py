"""Data contracts shared by search algorithms, tests, and GUI adapters."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


JsonValue = Any


def _json_value(value: Any) -> JsonValue:
    """Recursively convert supported values to JSON-compatible containers."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "to_dict"):
        return _json_value(value.to_dict())
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable")


def _finite_non_negative(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return converted


@dataclass(frozen=True)
class Edge:
    """A directed road segment in the adjacency graph."""

    to: str
    distance_km: float
    time_min: float
    congestion: float
    risk: float
    road_type: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.to, str) or not self.to:
            raise ValueError("Edge.to must be a non-empty node ID")
        if not isinstance(self.road_type, str) or not self.road_type:
            raise ValueError("Edge.road_type must be a non-empty string")
        object.__setattr__(
            self, "distance_km", _finite_non_negative(self.distance_km, "distance_km")
        )
        object.__setattr__(
            self, "time_min", _finite_non_negative(self.time_min, "time_min")
        )
        congestion = _finite_non_negative(self.congestion, "congestion")
        risk = _finite_non_negative(self.risk, "risk")
        if not 1 <= congestion <= 5:
            raise ValueError("congestion must be between 1 and 5")
        if not 0 <= risk <= 5:
            raise ValueError("risk must be between 0 and 5")
        object.__setattr__(self, "congestion", congestion)
        object.__setattr__(self, "risk", risk)
        if not isinstance(self.metadata, Mapping):
            raise TypeError("Edge.metadata must be a mapping")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Edge":
        """Build and validate an edge from a JSON-compatible mapping."""
        if not isinstance(data, Mapping):
            raise TypeError("Each edge must be a mapping or Edge instance")
        required = {
            "to",
            "distance_km",
            "time_min",
            "congestion",
            "risk",
            "road_type",
        }
        missing = sorted(required.difference(data))
        if missing:
            raise ValueError(f"Edge is missing required fields: {', '.join(missing)}")
        return cls(
            to=data["to"],
            distance_km=data["distance_km"],
            time_min=data["time_min"],
            congestion=data["congestion"],
            risk=data["risk"],
            road_type=data["road_type"],
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-serializable edge representation."""
        return {
            "to": self.to,
            "distance_km": self.distance_km,
            "time_min": self.time_min,
            "congestion": self.congestion,
            "risk": self.risk,
            "road_type": self.road_type,
            "metadata": _json_value(self.metadata),
        }


Graph = Mapping[str, Sequence[Edge]]
CostFunction = Callable[[str, Edge], float]
HeuristicFunction = Callable[[str, str], float]
StepCallback = Callable[["SearchStep"], None]


def graph_from_dict(data: Mapping[str, Any]) -> dict[str, tuple[Edge, ...]]:
    """Convert raw adjacency JSON/dict data into a validated directed graph.

    The input can be either a top-level adjacency mapping or an object with an
    ``adjacency`` field. Reverse edges are never inferred.
    """
    if not isinstance(data, Mapping):
        raise TypeError("Graph data must be a mapping")
    raw_adjacency = data.get("adjacency", data)
    if not isinstance(raw_adjacency, Mapping):
        raise TypeError("Graph adjacency must be a mapping")

    graph: dict[str, tuple[Edge, ...]] = {}
    for node, raw_edges in raw_adjacency.items():
        if not isinstance(node, str) or not node:
            raise ValueError("Every graph node ID must be a non-empty string")
        if isinstance(raw_edges, (str, bytes)) or not isinstance(raw_edges, Sequence):
            raise TypeError(f"Adjacency for node {node!r} must be a sequence")
        graph[node] = tuple(
            edge if isinstance(edge, Edge) else Edge.from_dict(edge)
            for edge in raw_edges
        )

    unknown_targets = sorted(
        {edge.to for edges in graph.values() for edge in edges}.difference(graph)
    )
    if unknown_targets:
        raise ValueError(
            "Every edge target must exist as a graph node; missing: "
            + ", ".join(unknown_targets)
        )
    return graph


def load_graph_json(path: str | Path) -> dict[str, tuple[Edge, ...]]:
    """Load a UTF-8 JSON graph and return its validated adjacency mapping."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    return graph_from_dict(data)


def coordinates_from_data(
    data: Mapping[str, Any],
) -> dict[str, tuple[float, float]]:
    """Extract ``(latitude, longitude)`` pairs from top-level node metadata."""
    raw_nodes = data.get("nodes")
    if not isinstance(raw_nodes, Mapping):
        raise ValueError("Graph data must contain a 'nodes' metadata mapping")

    coordinates: dict[str, tuple[float, float]] = {}
    for node, metadata in raw_nodes.items():
        if not isinstance(node, str) or not isinstance(metadata, Mapping):
            raise TypeError("Node metadata must map node IDs to mappings")
        if "latitude" not in metadata or "longitude" not in metadata:
            raise ValueError(f"Node {node!r} is missing latitude or longitude")
        latitude = float(metadata["latitude"])
        longitude = float(metadata["longitude"])
        if not math.isfinite(latitude) or not -90 <= latitude <= 90:
            raise ValueError(f"Invalid latitude for node {node!r}")
        if not math.isfinite(longitude) or not -180 <= longitude <= 180:
            raise ValueError(f"Invalid longitude for node {node!r}")
        coordinates[node] = (latitude, longitude)
    return coordinates


def coerce_graph(graph: Mapping[str, Sequence[Edge | Mapping[str, Any]]]) -> dict[
    str, tuple[Edge, ...]
]:
    """Return a validated snapshot so algorithms never mutate caller data."""
    return graph_from_dict(graph)


@dataclass
class SearchStep:
    """One GUI-friendly event captured during a graph search."""

    index: int
    event: str
    current_node: str | None
    frontier: tuple[tuple[str, float], ...] = ()
    visited: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-serializable trace event."""
        return {
            "index": self.index,
            "event": self.event,
            "current_node": self.current_node,
            "frontier": [
                {"node": node, "priority": priority}
                for node, priority in self.frontier
            ],
            "visited": list(self.visited),
            "details": _json_value(self.details),
        }


@dataclass
class SearchResult:
    """Result of a two-location graph search."""

    algorithm: str
    success: bool
    start: str
    goal: str
    path: list[str]
    total_cost: float | None
    total_distance_km: float | None
    total_time_min: float | None
    visited_order: list[str]
    expanded_nodes: int
    generated_nodes: int
    runtime_ms: float
    trace: list[SearchStep]
    message: str
    optimality: str

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-serializable search result."""
        return {
            "algorithm": self.algorithm,
            "success": self.success,
            "start": self.start,
            "goal": self.goal,
            "path": list(self.path),
            "total_cost": self.total_cost,
            "total_distance_km": self.total_distance_km,
            "total_time_min": self.total_time_min,
            "visited_order": list(self.visited_order),
            "expanded_nodes": self.expanded_nodes,
            "generated_nodes": self.generated_nodes,
            "runtime_ms": self.runtime_ms,
            "trace": [step.to_dict() for step in self.trace],
            "message": self.message,
            "optimality": self.optimality,
        }


@dataclass
class RouteSegment:
    """One shortest-path segment in a multi-location route."""

    start: str
    goal: str
    path: list[str]
    cost: float
    distance_km: float
    time_min: float

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-serializable route segment."""
        return {
            "start": self.start,
            "goal": self.goal,
            "path": list(self.path),
            "cost": self.cost,
            "distance_km": self.distance_km,
            "time_min": self.time_min,
        }


@dataclass
class MultiLocationResult:
    """Result of ordering and joining several required route locations."""

    method: str
    success: bool
    start: str
    requested_waypoints: list[str]
    visiting_order: list[str]
    full_path: list[str]
    segments: list[RouteSegment]
    total_cost: float | None
    total_distance_km: float | None
    total_time_min: float | None
    runtime_ms: float
    optimality: str
    comparison_gap_percent: float | None
    message: str

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-serializable multi-location result."""
        return {
            "method": self.method,
            "success": self.success,
            "start": self.start,
            "requested_waypoints": list(self.requested_waypoints),
            "visiting_order": list(self.visiting_order),
            "full_path": list(self.full_path),
            "segments": [segment.to_dict() for segment in self.segments],
            "total_cost": self.total_cost,
            "total_distance_km": self.total_distance_km,
            "total_time_min": self.total_time_min,
            "runtime_ms": self.runtime_ms,
            "optimality": self.optimality,
            "comparison_gap_percent": self.comparison_gap_percent,
            "message": self.message,
        }
