"""Internal validation, trace, and path helpers shared by search algorithms."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from core.contracts import (
    CostFunction,
    Edge,
    Graph,
    SearchStep,
    StepCallback,
    coerce_graph,
)


ParentMap = dict[str, tuple[str, Edge, float]]


def prepare_search(
    graph: Mapping[str, Sequence[Edge | Mapping[str, Any]]],
    start: str,
    goal: str,
    cost_fn: CostFunction,
) -> tuple[dict[str, tuple[Edge, ...]], dict[str, tuple[float, ...]]]:
    """Validate graph, endpoints, and all edge costs before a search."""
    if not callable(cost_fn):
        raise TypeError("cost_fn must be callable")
    normalized = coerce_graph(graph)
    for label, node in (("start", start), ("goal", goal)):
        if not isinstance(node, str) or not node:
            raise ValueError(f"{label} must be a non-empty node ID")
        if node not in normalized:
            raise ValueError(f"{label} node {node!r} does not exist in the graph")
    return normalized, build_cost_table(normalized, cost_fn)


def build_cost_table(
    graph: Graph, cost_fn: CostFunction
) -> dict[str, tuple[float, ...]]:
    """Evaluate and validate each directed edge cost once."""
    costs: dict[str, tuple[float, ...]] = {}
    for source, edges in graph.items():
        source_costs: list[float] = []
        for edge in edges:
            raw_cost = cost_fn(source, edge)
            if isinstance(raw_cost, bool) or not isinstance(raw_cost, (int, float)):
                raise TypeError(
                    f"Edge cost for {source!r} -> {edge.to!r} must be a number"
                )
            cost = float(raw_cost)
            if not math.isfinite(cost) or cost < 0:
                raise ValueError(
                    f"Edge cost for {source!r} -> {edge.to!r} "
                    "must be finite and non-negative"
                )
            source_costs.append(cost)
        costs[source] = tuple(source_costs)
    return costs


def validate_priority(value: Any, description: str) -> float:
    """Validate an externally supplied priority such as a heuristic value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{description} must be a number")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{description} must be finite and non-negative")
    return converted


def reconstruct_path(
    start: str, goal: str, parents: ParentMap
) -> tuple[list[str], list[tuple[str, Edge, float]]]:
    """Reconstruct a path and its exact selected edges from a parent map."""
    if start == goal:
        return [start], []
    if goal not in parents:
        return [], []

    reverse_nodes = [goal]
    reverse_edges: list[tuple[str, Edge, float]] = []
    current = goal
    while current != start:
        parent, edge, edge_cost = parents[current]
        reverse_nodes.append(parent)
        reverse_edges.append((parent, edge, edge_cost))
        current = parent
    reverse_nodes.reverse()
    reverse_edges.reverse()
    return reverse_nodes, reverse_edges


def sum_path_metrics(
    selected_edges: Sequence[tuple[str, Edge, float]],
) -> tuple[float, float, float]:
    """Sum weighted cost, physical distance, and estimated time."""
    return (
        sum(cost for _, _, cost in selected_edges),
        sum(edge.distance_km for _, edge, _ in selected_edges),
        sum(edge.time_min for _, edge, _ in selected_edges),
    )


@dataclass
class TraceEmitter:
    """Capture and optionally publish deterministic search events."""

    capture_trace: bool
    on_step: StepCallback | None
    trace: list[SearchStep] = field(default_factory=list)
    _next_index: int = 0

    def emit(
        self,
        event: str,
        current_node: str | None,
        frontier_heap: Sequence[tuple[float, int, str]],
        visited: Sequence[str],
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """Create one event, append it to trace, and invoke the callback."""
        if not self.capture_trace and self.on_step is None:
            return
        frontier = tuple(
            (node, float(priority))
            for priority, _, node in sorted(frontier_heap)
        )
        step = SearchStep(
            index=self._next_index,
            event=event,
            current_node=current_node,
            frontier=frontier,
            visited=tuple(visited),
            details=dict(details or {}),
        )
        self._next_index += 1
        if self.capture_trace:
            self.trace.append(step)
        if self.on_step is not None:
            self.on_step(step)
