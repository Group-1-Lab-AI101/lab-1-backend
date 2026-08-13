"""Iterative Depth-First Search with deterministic adjacency ordering."""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from algorithms._shared import (
    ParentMap,
    TraceEmitter,
    prepare_search,
    reconstruct_path,
    sum_path_metrics,
)
from core.contracts import CostFunction, Edge, SearchResult, StepCallback


def _stack_frontier(frontier: list[str]) -> list[tuple[float, int, str]]:
    next_first = list(reversed(frontier))
    return [(float(index), index, node) for index, node in enumerate(next_first)]


def depth_first_search(
    graph: Mapping[str, Sequence[Edge | Mapping[str, Any]]],
    start: str,
    goal: str,
    cost_fn: CostFunction,
    *,
    capture_trace: bool = True,
    on_step: StepCallback | None = None,
) -> SearchResult:
    """Find one directed route using iterative DFS.

    Args:
        graph: Directed adjacency mapping.
        start: Existing source node ID.
        goal: Existing destination node ID.
        cost_fn: Cost function used to measure the returned route.
        capture_trace: Whether to retain GUI events in the result.
        on_step: Optional callback invoked for each search event.

    Returns:
        The first deterministic depth-first route or an unreachable result.

    Raises:
        ValueError: If graph data, endpoints, or edge costs are invalid.
        TypeError: If graph data or the cost function has invalid types.
    """
    normalized, costs = prepare_search(graph, start, goal, cost_fn)
    started_at = time.perf_counter()
    frontier = [start]
    discovered = {start}
    parents: ParentMap = {}
    visited_order: list[str] = []
    generated_nodes = 1
    emitter = TraceEmitter(capture_trace, on_step)
    success = False

    while frontier:
        current = frontier.pop()
        emitter.emit("pop", current, _stack_frontier(frontier), visited_order)
        visited_order.append(current)
        emitter.emit("expand", current, _stack_frontier(frontier), visited_order)
        if current == goal:
            success = True
            emitter.emit("goal", current, _stack_frontier(frontier), visited_order)
            break

        indexed_edges = list(enumerate(normalized[current]))
        for edge_index, edge in reversed(indexed_edges):
            if edge.to in discovered:
                continue
            discovered.add(edge.to)
            parents[edge.to] = (current, edge, costs[current][edge_index])
            frontier.append(edge.to)
            generated_nodes += 1
            emitter.emit(
                "generate",
                current,
                _stack_frontier(frontier),
                visited_order,
                {"neighbor": edge.to},
            )

    runtime_ms = (time.perf_counter() - started_at) * 1000
    if not success:
        return SearchResult(
            algorithm="depth_first_search",
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
        algorithm="depth_first_search",
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
        message="A deterministic depth-first route was found.",
        optimality="not_guaranteed",
    )
