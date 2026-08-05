"""Greedy Best-First Search with external heuristics and shared tracing."""

from __future__ import annotations

import heapq
import itertools
import time
from typing import Any, Mapping, Sequence

from algorithms._shared import (
    ParentMap,
    TraceEmitter,
    prepare_search,
    reconstruct_path,
    sum_path_metrics,
    validate_priority,
)
from core.contracts import (
    CostFunction,
    Edge,
    HeuristicFunction,
    SearchResult,
    StepCallback,
)


def greedy_best_first(
    graph: Mapping[str, Sequence[Edge | Mapping[str, Any]]],
    start: str,
    goal: str,
    heuristic_fn: HeuristicFunction,
    cost_fn: CostFunction,
    *,
    capture_trace: bool = True,
    on_step: StepCallback | None = None,
) -> SearchResult:
    """Find a heuristic-guided route without claiming cost optimality.

    Args:
        graph: Directed adjacency mapping. Reverse edges are not inferred.
        start: Existing source node ID.
        goal: Existing destination node ID.
        heuristic_fn: Callable returning finite, non-negative h(node, goal).
        cost_fn: Callable used to measure the final path's real weighted cost.
        capture_trace: Whether to retain events in the returned result.
        on_step: Optional GUI callback invoked synchronously for every event.

    Returns:
        A route result, or a failure result when goal is unreachable.

    Raises:
        ValueError: If graph data, endpoints, edge costs, or heuristic values
            are invalid.
        TypeError: If graph data or supplied functions have invalid types.

    Guarantee:
        The frontier priority is based only on h(n). A closed set guarantees
        termination on finite graphs, but the result is not generally optimal.
    """
    if not callable(heuristic_fn):
        raise TypeError("heuristic_fn must be callable")
    normalized, costs = prepare_search(graph, start, goal, cost_fn)
    started_at = time.perf_counter()
    tie_counter = itertools.count()
    start_priority = validate_priority(
        heuristic_fn(start, goal), f"Heuristic for node {start!r}"
    )
    frontier: list[tuple[float, int, str]] = [
        (start_priority, next(tie_counter), start)
    ]
    discovered = {start}
    closed: set[str] = set()
    parents: ParentMap = {}
    visited_order: list[str] = []
    generated_nodes = 1
    emitter = TraceEmitter(capture_trace, on_step)
    success = False

    while frontier:
        priority, _, current = heapq.heappop(frontier)
        emitter.emit(
            "pop",
            current,
            frontier,
            visited_order,
            {"heuristic": priority},
        )
        if current in closed:
            emitter.emit(
                "skip_stale",
                current,
                frontier,
                visited_order,
                {"heuristic": priority},
            )
            continue

        closed.add(current)
        visited_order.append(current)
        emitter.emit(
            "expand",
            current,
            frontier,
            visited_order,
            {"heuristic": priority},
        )
        if current == goal:
            success = True
            emitter.emit(
                "goal",
                current,
                frontier,
                visited_order,
                {"heuristic": priority},
            )
            break

        for edge_index, edge in enumerate(normalized[current]):
            if edge.to in closed or edge.to in discovered:
                continue
            heuristic = validate_priority(
                heuristic_fn(edge.to, goal),
                f"Heuristic for node {edge.to!r}",
            )
            discovered.add(edge.to)
            parents[edge.to] = (current, edge, costs[current][edge_index])
            heapq.heappush(frontier, (heuristic, next(tie_counter), edge.to))
            generated_nodes += 1
            emitter.emit(
                "generate",
                current,
                frontier,
                visited_order,
                {"neighbor": edge.to, "heuristic": heuristic},
            )

    runtime_ms = (time.perf_counter() - started_at) * 1000
    if not success:
        return SearchResult(
            algorithm="greedy_best_first",
            success=False,
            start=start,
            goal=goal,
            path=[],
            total_cost=None,
            total_distance_km=None,
            total_time_min=None,
            visited_order=visited_order,
            expanded_nodes=len(visited_order),
            generated_nodes=generated_nodes,
            runtime_ms=runtime_ms,
            trace=emitter.trace,
            message=f"No route exists from {start!r} to {goal!r}.",
            optimality="not_applicable",
        )

    path, selected_edges = reconstruct_path(start, goal, parents)
    total_cost, total_distance, total_time = sum_path_metrics(selected_edges)
    return SearchResult(
        algorithm="greedy_best_first",
        success=True,
        start=start,
        goal=goal,
        path=path,
        total_cost=total_cost,
        total_distance_km=total_distance,
        total_time_min=total_time,
        visited_order=visited_order,
        expanded_nodes=len(visited_order),
        generated_nodes=generated_nodes,
        runtime_ms=runtime_ms,
        trace=emitter.trace,
        message="Heuristic-guided route found; minimum cost is not guaranteed.",
        optimality="not_guaranteed",
    )
