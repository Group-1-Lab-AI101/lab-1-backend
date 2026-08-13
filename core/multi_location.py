"""Multi-location route ordering built on cached Dijkstra shortest paths."""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from algorithms.dijkstra import dijkstra_all
from core.contracts import (
    CostFunction,
    Edge,
    MultiLocationResult,
    RouteSegment,
    SearchResult,
    coerce_graph,
)


SUPPORTED_METHODS = {"nearest_neighbor", "exact_bruteforce"}


@dataclass
class _Request:
    graph: dict[str, tuple[Edge, ...]]
    start: str
    requested_waypoints: list[str]
    duplicate_waypoints: list[str]
    cost_fn: CostFunction
    end: str | None
    return_to_start: bool
    exact_limit: int


class _PairwiseRouter:
    """Cache one complete single-source Dijkstra run per used source."""

    def __init__(self, graph: dict[str, tuple[Edge, ...]], cost_fn: CostFunction):
        self._graph = graph
        self._cost_fn = cost_fn
        self._source_cache: dict[str, dict[str, SearchResult]] = {}

    @property
    def source_runs(self) -> int:
        """Return the number of distinct source searches performed."""
        return len(self._source_cache)

    def get(self, start: str, goal: str) -> SearchResult:
        """Return a cached shortest-path result for one ordered node pair."""
        if start not in self._source_cache:
            self._source_cache[start] = dijkstra_all(
                self._graph, start, self._cost_fn
            )
        return self._source_cache[start][goal]


def _normalize_request(
    graph: Mapping[str, Sequence[Edge | Mapping[str, Any]]],
    start: str,
    waypoints: Sequence[str],
    cost_fn: CostFunction,
    end: str | None,
    return_to_start: bool,
    exact_limit: int,
) -> _Request:
    if not callable(cost_fn):
        raise TypeError("cost_fn must be callable")
    if isinstance(waypoints, (str, bytes)) or not isinstance(waypoints, Sequence):
        raise TypeError("waypoints must be a sequence of node IDs")
    if isinstance(exact_limit, bool) or not isinstance(exact_limit, int):
        raise TypeError("exact_limit must be an integer")
    if exact_limit < 0:
        raise ValueError("exact_limit must be non-negative")
    if not isinstance(return_to_start, bool):
        raise TypeError("return_to_start must be a boolean")

    normalized_graph = coerce_graph(graph)
    if not isinstance(start, str) or not start:
        raise ValueError("start must be a non-empty node ID")
    if start not in normalized_graph:
        raise ValueError(f"start node {start!r} does not exist in the graph")
    if end is not None:
        if not isinstance(end, str) or not end:
            raise ValueError("end must be None or a non-empty node ID")
        if end not in normalized_graph:
            raise ValueError(f"end node {end!r} does not exist in the graph")
    if return_to_start and end not in (None, start):
        raise ValueError(
            "return_to_start=True requires end to be None or equal to start"
        )

    seen = {start}
    normalized_waypoints: list[str] = []
    duplicates: list[str] = []
    for waypoint in waypoints:
        if not isinstance(waypoint, str) or not waypoint:
            raise ValueError("Every waypoint must be a non-empty node ID")
        if waypoint not in normalized_graph:
            raise ValueError(
                f"waypoint node {waypoint!r} does not exist in the graph"
            )
        if waypoint in seen:
            duplicates.append(waypoint)
            continue
        seen.add(waypoint)
        normalized_waypoints.append(waypoint)

    return _Request(
        graph=normalized_graph,
        start=start,
        requested_waypoints=normalized_waypoints,
        duplicate_waypoints=duplicates,
        cost_fn=cost_fn,
        end=end,
        return_to_start=return_to_start,
        exact_limit=exact_limit,
    )


def _duplicate_note(request: _Request) -> str:
    if not request.duplicate_waypoints:
        return ""
    removed = ", ".join(repr(node) for node in request.duplicate_waypoints)
    return f" Duplicate waypoint occurrences removed: {removed}."


def _segment_from_search(result: SearchResult) -> RouteSegment:
    if (
        not result.success
        or result.total_cost is None
        or result.total_distance_km is None
        or result.total_time_min is None
    ):
        raise ValueError("Cannot create a route segment from a failed search")
    return RouteSegment(
        start=result.start,
        goal=result.goal,
        path=list(result.path),
        cost=result.total_cost,
        distance_km=result.total_distance_km,
        time_min=result.total_time_min,
    )


def _join_path(start: str, segments: Sequence[RouteSegment]) -> list[str]:
    full_path = [start]
    for segment in segments:
        if full_path[-1] != segment.path[0]:
            raise ValueError("Route segments are not contiguous")
        full_path.extend(segment.path[1:])
    return full_path


def _record_waypoint_visits(
    path: Sequence[str],
    requested_waypoints: Sequence[str],
    visiting_order: list[str],
    visited_waypoints: set[str],
) -> None:
    required = set(requested_waypoints)
    for node in path:
        if node in required and node not in visited_waypoints:
            visited_waypoints.add(node)
            visiting_order.append(node)


def _success_result(
    request: _Request,
    method: str,
    visiting_order: list[str],
    segments: list[RouteSegment],
    started_at: float,
) -> MultiLocationResult:
    return MultiLocationResult(
        method=method,
        success=True,
        start=request.start,
        requested_waypoints=list(request.requested_waypoints),
        visiting_order=visiting_order,
        full_path=_join_path(request.start, segments),
        segments=segments,
        total_cost=sum(segment.cost for segment in segments),
        total_distance_km=sum(segment.distance_km for segment in segments),
        total_time_min=sum(segment.time_min for segment in segments),
        runtime_ms=(time.perf_counter() - started_at) * 1000,
        optimality=(
            "approximate_not_guaranteed"
            if method == "nearest_neighbor"
            else "optimal_for_reduced_pairwise_problem"
        ),
        comparison_gap_percent=None,
        message="Route optimized successfully." + _duplicate_note(request),
    )


def _failure_result(
    request: _Request,
    method: str,
    started_at: float,
    message: str,
    visiting_order: list[str] | None = None,
    segments: list[RouteSegment] | None = None,
) -> MultiLocationResult:
    actual_segments = list(segments or [])
    return MultiLocationResult(
        method=method,
        success=False,
        start=request.start,
        requested_waypoints=list(request.requested_waypoints),
        visiting_order=list(visiting_order or []),
        full_path=_join_path(request.start, actual_segments),
        segments=actual_segments,
        total_cost=None,
        total_distance_km=None,
        total_time_min=None,
        runtime_ms=(time.perf_counter() - started_at) * 1000,
        optimality="not_applicable",
        comparison_gap_percent=None,
        message=message + _duplicate_note(request),
    )


def _append_required_segment(
    router: _PairwiseRouter,
    current: str,
    target: str,
    segments: list[RouteSegment],
) -> tuple[str, str | None]:
    if current == target:
        return current, None
    result = router.get(current, target)
    if not result.success:
        return current, f"Required segment is unreachable: {current!r} -> {target!r}."
    segments.append(_segment_from_search(result))
    return target, None


def _nearest_neighbor(
    request: _Request, router: _PairwiseRouter
) -> MultiLocationResult:
    started_at = time.perf_counter()
    fixed_end_is_waypoint = (
        request.end is not None and request.end in request.requested_waypoints
    )
    remaining = [
        waypoint
        for waypoint in request.requested_waypoints
        if not fixed_end_is_waypoint or waypoint != request.end
    ]
    current = request.start
    visiting_order: list[str] = []
    visited_waypoints: set[str] = set()
    segments: list[RouteSegment] = []

    while remaining:
        best_index: int | None = None
        best_result: SearchResult | None = None
        first_unreachable: tuple[str, str] | None = None
        for index, candidate in enumerate(remaining):
            result = router.get(current, candidate)
            if not result.success:
                if first_unreachable is None:
                    first_unreachable = (current, candidate)
                continue
            if best_result is None or (
                result.total_cost is not None
                and best_result.total_cost is not None
                and result.total_cost < best_result.total_cost
            ):
                best_index = index
                best_result = result

        if best_index is None or best_result is None:
            source, target = first_unreachable or (current, remaining[0])
            return _failure_result(
                request,
                "nearest_neighbor",
                started_at,
                f"Required segment is unreachable: {source!r} -> {target!r}.",
                visiting_order,
                segments,
            )

        selected = remaining[best_index]
        segment = _segment_from_search(best_result)
        segments.append(segment)
        _record_waypoint_visits(
            segment.path,
            request.requested_waypoints,
            visiting_order,
            visited_waypoints,
        )
        remaining = [
            waypoint
            for waypoint in remaining
            if waypoint not in visited_waypoints
        ]
        current = selected

    if request.end is not None:
        current, error = _append_required_segment(
            router, current, request.end, segments
        )
        if error is not None:
            return _failure_result(
                request,
                "nearest_neighbor",
                started_at,
                error,
                visiting_order,
                segments,
            )
        if segments:
            _record_waypoint_visits(
                segments[-1].path,
                request.requested_waypoints,
                visiting_order,
                visited_waypoints,
            )
        elif fixed_end_is_waypoint:
            _record_waypoint_visits(
                [request.end],
                request.requested_waypoints,
                visiting_order,
                visited_waypoints,
            )

    if request.return_to_start and current != request.start:
        current, error = _append_required_segment(
            router, current, request.start, segments
        )
        if error is not None:
            return _failure_result(
                request,
                "nearest_neighbor",
                started_at,
                error,
                visiting_order,
                segments,
            )

    return _success_result(
        request, "nearest_neighbor", visiting_order, segments, started_at
    )


def _exact_bruteforce(
    request: _Request, router: _PairwiseRouter
) -> MultiLocationResult:
    if len(request.requested_waypoints) > request.exact_limit:
        raise ValueError(
            f"exact_bruteforce supports at most {request.exact_limit} waypoints; "
            "use method='nearest_neighbor' for larger inputs"
        )

    started_at = time.perf_counter()
    fixed_end_is_waypoint = (
        request.end is not None and request.end in request.requested_waypoints
    )
    permuted_waypoints = [
        waypoint
        for waypoint in request.requested_waypoints
        if not fixed_end_is_waypoint or waypoint != request.end
    ]
    best_cost: float | None = None
    best_visiting_order: list[str] = []
    best_segments: list[RouteSegment] = []
    first_unreachable: tuple[str, str] | None = None

    for permutation in itertools.permutations(permuted_waypoints):
        visiting_order: list[str] = []
        visited_waypoints: set[str] = set()
        route_targets = [(target, False) for target in permutation]
        if request.end is not None:
            route_targets.append((request.end, True))
        if request.return_to_start and (
            not route_targets or route_targets[-1][0] != request.start
        ):
            route_targets.append((request.start, True))

        current = request.start
        segments: list[RouteSegment] = []
        feasible = True
        for target, is_required_endpoint in route_targets:
            if not is_required_endpoint and target in visited_waypoints:
                continue
            if current == target:
                _record_waypoint_visits(
                    [current],
                    request.requested_waypoints,
                    visiting_order,
                    visited_waypoints,
                )
                continue
            result = router.get(current, target)
            if not result.success:
                if first_unreachable is None:
                    first_unreachable = (current, target)
                feasible = False
                break
            segment = _segment_from_search(result)
            segments.append(segment)
            _record_waypoint_visits(
                segment.path,
                request.requested_waypoints,
                visiting_order,
                visited_waypoints,
            )
            current = target

        if not feasible or len(visited_waypoints) != len(
            request.requested_waypoints
        ):
            continue
        total_cost = sum(segment.cost for segment in segments)
        if best_cost is None or (
            total_cost < best_cost
            and not math.isclose(total_cost, best_cost, rel_tol=1e-12, abs_tol=1e-12)
        ):
            best_cost = total_cost
            best_visiting_order = visiting_order
            best_segments = segments

    if best_cost is None:
        source, target = first_unreachable or (
            request.start,
            request.requested_waypoints[0]
            if request.requested_waypoints
            else request.end or request.start,
        )
        return _failure_result(
            request,
            "exact_bruteforce",
            started_at,
            "No feasible visiting order exists; "
            f"required segment is unreachable: {source!r} -> {target!r}.",
        )

    return _success_result(
        request,
        "exact_bruteforce",
        best_visiting_order,
        best_segments,
        started_at,
    )


def _run_method(
    request: _Request, method: str, router: _PairwiseRouter
) -> MultiLocationResult:
    if method == "nearest_neighbor":
        return _nearest_neighbor(request, router)
    if method == "exact_bruteforce":
        return _exact_bruteforce(request, router)
    supported = ", ".join(sorted(SUPPORTED_METHODS))
    raise ValueError(f"Unknown method {method!r}; expected one of: {supported}")


def optimize_multi_location(
    graph: Mapping[str, Sequence[Edge | Mapping[str, Any]]],
    start: str,
    waypoints: Sequence[str],
    cost_fn: CostFunction,
    *,
    method: str = "nearest_neighbor",
    end: str | None = None,
    return_to_start: bool = False,
    exact_limit: int = 8,
) -> MultiLocationResult:
    """Order required waypoints and join cached shortest-path segments.

    Args:
        graph: Directed adjacency mapping.
        start: Existing route start node.
        waypoints: Required nodes; duplicates and ``start`` are stably removed.
        cost_fn: Callable returning finite, non-negative edge costs.
        method: ``nearest_neighbor`` or ``exact_bruteforce``.
        end: Optional fixed node to reach after all waypoints.
        return_to_start: Whether to append a final route back to ``start``.
        exact_limit: Maximum normalized waypoint count for brute force.

    Returns:
        A complete node-by-node route, or an explicit unreachable result.

    Raises:
        ValueError: If nodes, method options, costs, or limits are invalid.
        TypeError: If graph data or argument types are invalid.

    Guarantee:
        Each normalized waypoint appears once in ``visiting_order``. Pairwise
        shortest paths are cached by source and segment boundaries are not
        duplicated in ``full_path``.
    """
    request = _normalize_request(
        graph,
        start,
        waypoints,
        cost_fn,
        end,
        return_to_start,
        exact_limit,
    )
    return _run_method(request, method, _PairwiseRouter(request.graph, cost_fn))


def compare_multi_location_methods(
    graph: Mapping[str, Sequence[Edge | Mapping[str, Any]]],
    start: str,
    waypoints: Sequence[str],
    cost_fn: CostFunction,
    *,
    end: str | None = None,
    return_to_start: bool = False,
    exact_limit: int = 8,
) -> dict[str, MultiLocationResult]:
    """Run nearest-neighbor and exact methods with one shared pairwise cache.

    Args:
        graph: Directed adjacency mapping.
        start: Existing route start node.
        waypoints: Required nodes.
        cost_fn: Callable returning finite, non-negative edge costs.
        end: Optional fixed final node.
        return_to_start: Whether both methods must return to ``start``.
        exact_limit: Maximum normalized waypoint count for exact comparison.

    Returns:
        A mapping containing ``nearest_neighbor`` and ``exact_bruteforce``.
        The nearest result includes its percentage gap relative to exact.

    Raises:
        ValueError: If inputs are invalid or exact comparison exceeds the limit.
        TypeError: If graph data or argument types are invalid.

    Guarantee:
        The comparison uses the same graph, options, weights, and cached
        single-source shortest paths for both methods.
    """
    request = _normalize_request(
        graph,
        start,
        waypoints,
        cost_fn,
        end,
        return_to_start,
        exact_limit,
    )
    router = _PairwiseRouter(request.graph, cost_fn)
    nearest = _nearest_neighbor(request, router)
    exact = _exact_bruteforce(request, router)

    gap: float | None = None
    if (
        nearest.success
        and exact.success
        and nearest.total_cost is not None
        and exact.total_cost is not None
    ):
        if math.isclose(exact.total_cost, 0.0, abs_tol=1e-12):
            gap = (
                0.0
                if math.isclose(nearest.total_cost, 0.0, abs_tol=1e-12)
                else None
            )
        else:
            gap = (
                (nearest.total_cost - exact.total_cost)
                / exact.total_cost
                * 100.0
            )

    return {
        "nearest_neighbor": replace(nearest, comparison_gap_percent=gap),
        "exact_bruteforce": replace(exact, comparison_gap_percent=0.0),
    }
