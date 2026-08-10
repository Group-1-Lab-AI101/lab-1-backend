"""A* graph search with external admissible-heuristic support."""

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


def a_star_search(
    graph: Mapping[str, Sequence[Edge | Mapping[str, Any]]],
    start: str,
    goal: str,
    heuristic_fn: HeuristicFunction,
    cost_fn: CostFunction,
    *,
    capture_trace: bool = True,
    on_step: StepCallback | None = None,
) -> SearchResult:
    """Find a route using A* priority `f(n) = g(n) + h(n)`.

    Args:
        graph: Directed adjacency mapping.
        start: Existing source node ID.
        goal: Existing destination node ID.
        heuristic_fn: Callable returning finite, non-negative estimates.
        cost_fn: Callable returning finite, non-negative edge costs.
        capture_trace: Whether to retain GUI events in the result.
        on_step: Optional callback invoked for each search event.

    Returns:
        A route result or an explicit unreachable result.

    Raises:
        ValueError: If graph data, endpoints, costs, or heuristics are invalid.
        TypeError: If graph data or supplied functions have invalid types.

    Guarantee:
        The result is minimum-cost when the supplied heuristic is admissible.
        Better paths can reopen a previously expanded node.
    """
    if not callable(heuristic_fn):
        raise TypeError("heuristic_fn must be callable")
    normalized, costs = prepare_search(graph, start, goal, cost_fn)
    started_at = time.perf_counter()
    tie_counter = itertools.count()
    start_h = validate_priority(
        heuristic_fn(start, goal), f"Heuristic for node {start!r}"
    )
    first_counter = next(tie_counter)
    frontier: list[tuple[float, int, str]] = [(start_h, first_counter, start)]
    queued_g = {first_counter: 0.0}
    g_score = {start: 0.0}
    expanded_best: dict[str, float] = {}
    parents: ParentMap = {}
    visited_order: list[str] = []
    generated_nodes = 1
    emitter = TraceEmitter(capture_trace, on_step)
    success = False

    while frontier:
        priority, counter, current = heapq.heappop(frontier)
        current_g = queued_g.pop(counter)
        current_details = {
            "g": current_g,
            "h": max(0.0, priority - current_g),
            "f": priority,
        }
        emitter.emit(
            "pop",
            current,
            frontier,
            visited_order,
            current_details,
        )
        if current_g > g_score[current] or current_g >= expanded_best.get(
            current, float("inf")
        ):
            emitter.emit(
                "skip_stale",
                current,
                frontier,
                visited_order,
                {"g": current_g, "best_g": g_score[current]},
            )
            continue

        expanded_best[current] = current_g
        visited_order.append(current)
        emitter.emit(
            "expand",
            current,
            frontier,
            visited_order,
            current_details,
        )
        if current == goal:
            success = True
            emitter.emit(
                "goal",
                current,
                frontier,
                visited_order,
                current_details,
            )
            break

        for edge_index, edge in enumerate(normalized[current]):
            candidate_g = current_g + costs[current][edge_index]
            if candidate_g >= g_score.get(edge.to, float("inf")):
                continue
            heuristic = validate_priority(
                heuristic_fn(edge.to, goal),
                f"Heuristic for node {edge.to!r}",
            )
            g_score[edge.to] = candidate_g
            parents[edge.to] = (current, edge, costs[current][edge_index])
            next_counter = next(tie_counter)
            queued_g[next_counter] = candidate_g
            candidate_f = candidate_g + heuristic
            heapq.heappush(frontier, (candidate_f, next_counter, edge.to))
            generated_nodes += 1
            emitter.emit(
                "relax",
                current,
                frontier,
                visited_order,
                {
                    "neighbor": edge.to,
                    "g": candidate_g,
                    "h": heuristic,
                    "f": candidate_f,
                },
            )

    runtime_ms = (time.perf_counter() - started_at) * 1000
    if not success:
        return SearchResult(
            algorithm="a_star",
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
        algorithm="a_star",
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
        message="A* route found using g(n) + h(n).",
        optimality="optimal_with_admissible_heuristic",
    )
