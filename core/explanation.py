"""Human-readable route details and cross-algorithm comparisons."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from core.contracts import CostFunction, Edge, SearchResult


ALGORITHM_NOTES = {
    "breadth_first_search": "BFS minimizes the number of graph edges, not traffic cost.",
    "depth_first_search": "DFS returns the first depth-first route and does not guarantee route quality.",
    "uniform_cost_search": "UCS guarantees minimum total cost for non-negative edge costs.",
    "a_star": "A* guarantees minimum cost when its heuristic is admissible.",
    "dijkstra": "Dijkstra guarantees minimum total cost for non-negative edge costs.",
    "greedy_best_first": "Greedy follows the heuristic and does not guarantee minimum cost.",
}

ALGORITHM_LABELS = {
    "breadth_first_search": "BFS",
    "depth_first_search": "DFS",
    "uniform_cost_search": "UCS",
    "a_star": "A*",
    "dijkstra": "Dijkstra",
    "greedy_best_first": "Greedy Best-First",
}

CRITERION_NOTES = {
    "balanced": "balances distance, estimated time, congestion, and road risk",
    "fastest": "places the strongest emphasis on estimated travel time",
    "shortest": "places the strongest emphasis on physical distance",
    "low_congestion": "penalizes congested road segments more strongly",
    "low_risk": "penalizes road-risk factors more strongly",
    "custom": "uses the custom weights supplied by the user",
}


def route_segments(
    graph: Mapping[str, Sequence[Edge]],
    path: Sequence[str],
    cost_fn: CostFunction,
) -> list[dict[str, Any]]:
    """Describe every directed edge selected by a result path."""
    details: list[dict[str, Any]] = []
    for source, target in zip(path, path[1:]):
        edge = next(edge for edge in graph[source] if edge.to == target)
        details.append(
            {
                "from": source,
                "to": target,
                "name": edge.metadata.get("name") or "Unnamed road",
                "road_type": edge.road_type,
                "distance_km": edge.distance_km,
                "time_min": edge.time_min,
                "congestion": edge.congestion,
                "risk": edge.risk,
                "cost": cost_fn(source, edge),
                "oneway": bool(edge.metadata.get("oneway", False)),
                "geometry": [
                    list(point[:2])
                    for point in (edge.metadata.get("geometry") or [])
                ],
            }
        )
    return details


def explain_search_result(
    result: SearchResult,
    graph: Mapping[str, Sequence[Edge]],
    cost_fn: CostFunction,
    *,
    start_name: str,
    goal_name: str,
    criterion: str,
    traffic_profile: str,
    alternative_result: SearchResult | None = None,
    start_access_m: float = 0.0,
    goal_access_m: float = 0.0,
) -> dict[str, Any]:
    """Build a concise explanation grounded in final-path edge attributes."""
    if not result.success:
        return {
            "headline": f"No route was found from {start_name} to {goal_name}.",
            "reasons": [result.message],
            "optimality_note": ALGORITHM_NOTES.get(result.algorithm, ""),
            "high_congestion_segments": [],
            "alternative_comparison": None,
        }

    segments = route_segments(graph, result.path, cost_fn)
    congested = [segment for segment in segments if segment["congestion"] >= 4.0]
    named_roads: list[str] = []
    for segment in segments:
        name = str(segment["name"])
        if name != "Unnamed road" and name not in named_roads:
            named_roads.append(name)
    road_summary = ", ".join(named_roads[:3]) or "local road segments"
    reasons = [
        f"The route uses {len(segments)} road segments via {road_summary}.",
        f"The {criterion} criterion {CRITERION_NOTES.get(criterion, CRITERION_NOTES['custom'])}.",
        f"Traffic profile '{traffic_profile}' is reflected in travel time, congestion, and risk values.",
    ]
    if congested:
        reasons.append(
            f"{len(congested)} selected segment(s) have congestion level 4 or higher."
        )
    else:
        reasons.append("The selected route avoids congestion level 4 or higher.")

    if start_access_m > 0 or goal_access_m > 0:
        reasons.append(
            "Map markers are snapped to unique routable nodes "
            f"({start_access_m:.0f} m at the start and {goal_access_m:.0f} m "
            "at the destination); these access offsets are shown separately."
        )

    alternative_comparison: dict[str, Any] | None = None
    if alternative_result is not None and alternative_result.success:
        alternative_segments = route_segments(
            graph, alternative_result.path, cost_fn
        )
        alternative_congested = sum(
            segment["congestion"] >= 4.0 for segment in alternative_segments
        )
        same_path = result.path == alternative_result.path
        label = ALGORITHM_LABELS.get(
            alternative_result.algorithm, alternative_result.algorithm
        )
        if same_path:
            comparison_text = (
                f"The {label} baseline reaches the same route and metrics."
            )
        else:
            cost_delta = alternative_result.total_cost - result.total_cost
            distance_delta = (
                alternative_result.total_distance_km - result.total_distance_km
            )
            time_delta = alternative_result.total_time_min - result.total_time_min
            comparison_text = (
                f"The {label} alternative has weighted cost "
                f"{abs(cost_delta):.2f} {'higher' if cost_delta >= 0 else 'lower'}, "
                f"is {abs(distance_delta):.2f} km "
                f"{'longer' if distance_delta >= 0 else 'shorter'}, and is "
                f"{abs(time_delta):.1f} minutes "
                f"{'slower' if time_delta >= 0 else 'faster'}."
            )
        reasons.append(comparison_text)
        alternative_comparison = {
            "algorithm": alternative_result.algorithm,
            "label": label,
            "same_path": same_path,
            "summary": comparison_text,
            "total_cost": alternative_result.total_cost,
            "total_distance_km": alternative_result.total_distance_km,
            "total_time_min": alternative_result.total_time_min,
            "expanded_nodes": alternative_result.expanded_nodes,
            "high_congestion_segments": alternative_congested,
        }

    congested_by_name: dict[str, dict[str, Any]] = {}
    for segment in congested:
        name = str(segment["name"])
        grouped = congested_by_name.setdefault(
            name,
            {
                "name": name,
                "congestion": 0.0,
                "time_min": 0.0,
                "segment_count": 0,
            },
        )
        grouped["congestion"] = max(
            grouped["congestion"], segment["congestion"]
        )
        grouped["time_min"] += segment["time_min"]
        grouped["segment_count"] += 1

    return {
        "headline": (
            f"{start_name} to {goal_name}: {result.total_distance_km:.2f} km, "
            f"about {result.total_time_min:.1f} minutes."
        ),
        "reasons": reasons,
        "optimality_note": ALGORITHM_NOTES.get(result.algorithm, ""),
        "high_congestion_segments": list(congested_by_name.values()),
        "alternative_comparison": alternative_comparison,
    }


def summarize_comparison(results: Sequence[SearchResult]) -> dict[str, Any]:
    """Identify metric leaders and explain why algorithm outputs differ."""
    successful = [result for result in results if result.success]
    if not successful:
        return {
            "headline": "No compared algorithm found a route.",
            "leaders": {},
            "observations": [],
        }

    def leader(field: str) -> dict[str, Any]:
        values = [float(getattr(result, field)) for result in successful]
        best = min(values)
        return {
            "algorithms": [
                result.algorithm
                for result in successful
                if math.isclose(
                    float(getattr(result, field)),
                    best,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
            ],
            "value": best,
        }

    leaders = {
        "lowest_cost": leader("total_cost"),
        "shortest_distance": leader("total_distance_km"),
        "fastest_estimate": leader("total_time_min"),
        "fewest_expanded": leader("expanded_nodes"),
        "lowest_runtime": leader("runtime_ms"),
    }
    unique_paths = {tuple(result.path) for result in successful}
    observations = [
        f"The algorithms produced {len(unique_paths)} distinct route(s).",
        "BFS and DFS are driven by graph structure; cost-aware methods use weighted traffic values.",
        "Runtime on this small graph should be interpreted together with expanded-node count.",
    ]
    return {
        "headline": f"Compared {len(successful)} successful algorithms on one traffic scenario.",
        "leaders": leaders,
        "observations": observations,
    }
