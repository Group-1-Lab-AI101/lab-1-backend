"""Dijkstra shortest-path search with reusable single-source support."""

from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from algorithms._shared import (
    ParentMap,
    TraceEmitter,
    prepare_search,
    reconstruct_path,
    sum_path_metrics,
)
from core.contracts import (
    CostFunction,
    Edge,
    SearchResult,
    StepCallback,
)


@dataclass
class _DijkstraRun:
    distances: dict[str, float]
    parents: ParentMap
    visited_order: list[str]
    expanded_nodes: int
    generated_nodes: int
    runtime_ms: float
    trace: list


def _run_dijkstra(
    graph: dict[str, tuple[Edge, ...]],
    costs: dict[str, tuple[float, ...]],
    start: str,
    goal: str | None,
    capture_trace: bool,
    on_step: StepCallback | None,
) -> _DijkstraRun:
    started_at = time.perf_counter()
    tie_counter = itertools.count()
    frontier: list[tuple[float, int, str]] = [(0.0, next(tie_counter), start)]
    distances = {start: 0.0}
    parents: ParentMap = {}
    finalized: set[str] = set()
    visited_order: list[str] = []
    generated_nodes = 1
    emitter = TraceEmitter(capture_trace, on_step)

    while frontier:
        current_cost, _, current = heapq.heappop(frontier)
        emitter.emit(
            "pop",
            current,
            frontier,
            visited_order,
            {"cost": current_cost},
        )
        if current in finalized or current_cost > distances[current]:
            emitter.emit(
                "skip_stale",
                current,
                frontier,
                visited_order,
                {"cost": current_cost, "best_cost": distances[current]},
            )
            continue

        finalized.add(current)
        visited_order.append(current)
        emitter.emit(
            "expand",
            current,
            frontier,
            visited_order,
            {"cost": current_cost},
        )
        if goal is not None and current == goal:
            emitter.emit(
                "goal",
                current,
                frontier,
                visited_order,
                {"cost": current_cost},
            )
            break

        for edge_index, edge in enumerate(graph[current]):
            if edge.to in finalized:
                continue
            candidate_cost = current_cost + costs[current][edge_index]
            if candidate_cost < distances.get(edge.to, float("inf")):
                previous_cost = distances.get(edge.to)
                distances[edge.to] = candidate_cost
                parents[edge.to] = (current, edge, costs[current][edge_index])
                heapq.heappush(
                    frontier, (candidate_cost, next(tie_counter), edge.to)
                )
                generated_nodes += 1
                emitter.emit(
                    "relax",
                    current,
                    frontier,
                    visited_order,
                    {
                        "neighbor": edge.to,
                        "old_cost": previous_cost,
                        "new_cost": candidate_cost,
                    },
                )

    return _DijkstraRun(
        distances=distances,
        parents=parents,
        visited_order=visited_order,
        expanded_nodes=len(visited_order),
        generated_nodes=generated_nodes,
        runtime_ms=(time.perf_counter() - started_at) * 1000,
        trace=emitter.trace,
    )


def _to_result(
    run: _DijkstraRun,
    start: str,
    goal: str,
    *,
    trace: list | None = None,
) -> SearchResult:
    if goal not in run.distances:
        return SearchResult(
            algorithm="dijkstra",
            success=False,
            start=start,
            goal=goal,
            path=[],
            total_cost=None,
            total_distance_km=None,
            total_time_min=None,
            visited_order=list(run.visited_order),
            expanded_nodes=run.expanded_nodes,
            generated_nodes=run.generated_nodes,
            runtime_ms=run.runtime_ms,
            trace=list(run.trace if trace is None else trace),
            message=f"No route exists from {start!r} to {goal!r}.",
            optimality="not_applicable",
        )

    path, selected_edges = reconstruct_path(start, goal, run.parents)
    total_cost, total_distance, total_time = sum_path_metrics(selected_edges)
    return SearchResult(
        algorithm="dijkstra",
        success=True,
        start=start,
        goal=goal,
        path=path,
        total_cost=total_cost,
        total_distance_km=total_distance,
        total_time_min=total_time,
        visited_order=list(run.visited_order),
        expanded_nodes=run.expanded_nodes,
        generated_nodes=run.generated_nodes,
        runtime_ms=run.runtime_ms,
        trace=list(run.trace if trace is None else trace),
        message="Lowest-cost route found.",
        optimality="optimal",
    )


def dijkstra(
    graph: Mapping[str, Sequence[Edge | Mapping[str, Any]]],
    start: str,
    goal: str,
    cost_fn: CostFunction,
    *,
    capture_trace: bool = True,
    on_step: StepCallback | None = None,
) -> SearchResult:
    """Find a lowest-cost route using Dijkstra's algorithm.

    Args:
        graph: Directed adjacency mapping. Reverse edges must already exist.
        start: Existing source node ID.
        goal: Existing destination node ID.
        cost_fn: Callable returning a finite, non-negative edge cost.
        capture_trace: Whether to retain events in the returned result.
        on_step: Optional GUI callback invoked synchronously for every event.

    Returns:
        A successful optimal result, or a failure result when goal is unreachable.

    Raises:
        ValueError: If graph data, endpoint IDs, or edge costs are invalid.
        TypeError: If graph values or callback dependencies have invalid types.

    Guarantee:
        For non-negative edge costs, the returned successful path has minimum
        total cost. The input graph is never modified.
    """
    normalized, costs = prepare_search(graph, start, goal, cost_fn)
    run = _run_dijkstra(
        normalized, costs, start, goal, capture_trace, on_step
    )
    return _to_result(run, start, goal)


def dijkstra_all(
    graph: Mapping[str, Sequence[Edge | Mapping[str, Any]]],
    start: str,
    cost_fn: CostFunction,
    *,
    goals: Sequence[str] | None = None,
    capture_trace: bool = False,
    on_step: StepCallback | None = None,
) -> dict[str, SearchResult]:
    """Compute shortest paths from one source with a single Dijkstra run.

    Args:
        graph: Directed adjacency mapping.
        start: Existing source node ID.
        cost_fn: Callable returning finite, non-negative edge costs.
        goals: Optional target subset; all graph nodes are returned by default.
        capture_trace: Whether each result should include the shared full trace.
        on_step: Optional callback for each search event.

    Returns:
        Mapping from requested target IDs to success or unreachable results.

    Raises:
        ValueError: If graph, start, a requested goal, or an edge cost is invalid.
        TypeError: If graph data or the cost function has an invalid type.

    Guarantee:
        Every reachable target receives its optimal path while edge exploration
        is performed only once for the source.
    """
    if goals is None:
        target_probe = start
    else:
        if isinstance(goals, (str, bytes)):
            raise TypeError("goals must be a sequence of node IDs")
        target_probe = start
    normalized, costs = prepare_search(graph, start, target_probe, cost_fn)
    targets = list(normalized) if goals is None else list(goals)
    for goal in targets:
        if not isinstance(goal, str) or not goal:
            raise ValueError("Every goal must be a non-empty node ID")
        if goal not in normalized:
            raise ValueError(f"goal node {goal!r} does not exist in the graph")

    run = _run_dijkstra(
        normalized, costs, start, None, capture_trace, on_step
    )
    shared_trace = run.trace if capture_trace else []
    return {
        goal: _to_result(run, start, goal, trace=shared_trace)
        for goal in targets
    }
