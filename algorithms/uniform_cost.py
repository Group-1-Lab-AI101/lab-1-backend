"""Goal-directed Uniform Cost Search with deterministic priority expansion."""

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
)
from core.contracts import CostFunction, Edge, SearchResult, StepCallback


def uniform_cost_search(
    graph: Mapping[str, Sequence[Edge | Mapping[str, Any]]],
    start: str,
    goal: str,
    cost_fn: CostFunction,
    *,
    capture_trace: bool = True,
    on_step: StepCallback | None = None,
) -> SearchResult:
    """Find a minimum-cost route with goal-directed Uniform Cost Search.

    Args:
        graph: Directed adjacency mapping.
        start: Existing source node ID.
        goal: Existing destination node ID.
        cost_fn: Callable returning finite non-negative edge costs.
        capture_trace: Whether to retain GUI events in the result.
        on_step: Optional callback invoked for each search event.

    Returns:
        A minimum-cost route or an explicit unreachable result.

    Raises:
        ValueError: If graph data, endpoints, or costs are invalid.
        TypeError: If graph data or the cost function has invalid types.

    Guarantee:
        The first goal state removed from the priority queue has minimum total
        path cost when all edge costs are non-negative.
    """
    normalized, costs = prepare_search(graph, start, goal, cost_fn)
    started_at = time.perf_counter()
    tie_counter = itertools.count()
    frontier: list[tuple[float, int, str]] = [
        (0.0, next(tie_counter), start)
    ]
    best_cost = {start: 0.0}
    closed: set[str] = set()
    parents: ParentMap = {}
    visited_order: list[str] = []
    generated_nodes = 1
    emitter = TraceEmitter(capture_trace, on_step)
    success = False

    while frontier:
        current_cost, _, current = heapq.heappop(frontier)
        emitter.emit(
            "pop",
            current,
            frontier,
            visited_order,
            {"cost": current_cost},
        )
        if current in closed or current_cost > best_cost[current]:
            emitter.emit(
                "skip_stale",
                current,
                frontier,
                visited_order,
                {"cost": current_cost, "best_cost": best_cost[current]},
            )
            continue

        closed.add(current)
        visited_order.append(current)
        emitter.emit(
            "expand",
            current,
            frontier,
            visited_order,
            {"cost": current_cost},
        )
        if current == goal:
            success = True
            emitter.emit(
                "goal",
                current,
                frontier,
                visited_order,
                {"cost": current_cost},
            )
            break

        for edge_index, edge in enumerate(normalized[current]):
            if edge.to in closed:
                continue
            candidate_cost = current_cost + costs[current][edge_index]
            if candidate_cost >= best_cost.get(edge.to, float("inf")):
                continue
            previous_cost = best_cost.get(edge.to)
            best_cost[edge.to] = candidate_cost
            parents[edge.to] = (current, edge, costs[current][edge_index])
            heapq.heappush(
                frontier,
                (candidate_cost, next(tie_counter), edge.to),
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

    runtime_ms = (time.perf_counter() - started_at) * 1000
    if not success:
        return SearchResult(
            algorithm="uniform_cost_search",
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
        algorithm="uniform_cost_search",
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
        message="Minimum-cost route found by Uniform Cost Search.",
        optimality="optimal",
    )
