"""Generate reproducible project evidence for the report and presentation."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Sequence

from algorithms.dijkstra import dijkstra
from core.cost import WeightedCostFunction
from core.osm_loader import apply_traffic_profile
from core.service import COST_PRESETS, RoutePlanner


ALGORITHM_CASE = ("notre_dame_cathedral", "saigon_zoo")
TRAFFIC_CASE = ("ben_thanh_market", "independence_palace")
MULTI_START = "notre_dame_cathedral"
MULTI_WAYPOINTS = (
    "ben_thanh_market",
    "nguyen_hue_walking_street",
    "bach_dang_wharf",
    "fine_arts_museum",
)


def _original_order_metrics(
    planner: RoutePlanner,
    start_id: str,
    waypoint_ids: Sequence[str],
) -> dict[str, float]:
    graph = apply_traffic_profile(planner.network.graph, "normal")
    cost_fn = WeightedCostFunction.from_dict(COST_PRESETS["balanced"])
    landmarks = planner.network.landmarks
    ordered_nodes = [landmarks[start_id].snapped_node] + [
        landmarks[item].snapped_node for item in waypoint_ids
    ]
    results = [
        dijkstra(graph, source, target, cost_fn, capture_trace=False)
        for source, target in zip(ordered_nodes, ordered_nodes[1:])
    ]
    if not all(result.success for result in results):
        raise RuntimeError("The benchmark input order contains an unreachable segment")
    return {
        "total_cost": sum(result.total_cost or 0.0 for result in results),
        "total_distance_km": sum(
            result.total_distance_km or 0.0 for result in results
        ),
        "total_time_min": sum(result.total_time_min or 0.0 for result in results),
    }


def _multi_summary(
    planner: RoutePlanner, result: dict[str, Any]
) -> dict[str, Any]:
    node_to_landmark = {
        landmark.snapped_node: landmark.id
        for landmark in planner.network.landmarks.values()
    }
    return {
        "method": result["method"],
        "visiting_order": [
            node_to_landmark[node] for node in result["visiting_order"]
        ],
        "total_cost": result["total_cost"],
        "total_distance_km": result["total_distance_km"],
        "total_time_min": result["total_time_min"],
        "runtime_ms": result["runtime_ms"],
        "comparison_gap_percent": result["comparison_gap_percent"],
        "optimality": result["optimality"],
    }


def run_benchmark(repeats: int) -> dict[str, Any]:
    """Run fixed algorithm, traffic, and multi-location scenarios."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    planner = RoutePlanner()
    algorithm_runs: dict[str, list[dict[str, Any]]] = {}
    for _ in range(repeats):
        comparison = planner.compare(*ALGORITHM_CASE)
        for payload in comparison["algorithms"]:
            algorithm_runs.setdefault(
                payload["request"]["algorithm"], []
            ).append(payload["result"])

    algorithms = []
    for algorithm, runs in algorithm_runs.items():
        representative = runs[0]
        algorithms.append(
            {
                "algorithm": algorithm,
                "total_cost": representative["total_cost"],
                "total_distance_km": representative["total_distance_km"],
                "total_time_min": representative["total_time_min"],
                "expanded_nodes": representative["expanded_nodes"],
                "generated_nodes": representative["generated_nodes"],
                "path_nodes": len(representative["path"]),
                "median_runtime_ms": statistics.median(
                    result["runtime_ms"] for result in runs
                ),
                "optimality": representative["optimality"],
            }
        )

    traffic = []
    for profile in ("normal", "rush_hour", "rainy"):
        payload = planner.search(
            *TRAFFIC_CASE,
            "dijkstra",
            traffic_profile=profile,
            capture_trace=False,
        )
        result = payload["result"]
        traffic.append(
            {
                "profile": profile,
                "total_cost": result["total_cost"],
                "total_distance_km": result["total_distance_km"],
                "total_time_min": result["total_time_min"],
                "path_nodes": len(result["path"]),
                "road_names": list(
                    dict.fromkeys(
                        segment["name"] for segment in payload["route_segments"]
                    )
                ),
            }
        )

    multi = planner.multi_route(
        MULTI_START,
        MULTI_WAYPOINTS,
        method="nearest_neighbor",
        compare_methods=True,
    )
    return {
        "network_summary": planner.network.summary,
        "repeats": repeats,
        "algorithm_case": {
            "start": ALGORITHM_CASE[0],
            "goal": ALGORITHM_CASE[1],
            "criterion": "balanced",
            "traffic_profile": "normal",
            "results": algorithms,
        },
        "traffic_case": {
            "start": TRAFFIC_CASE[0],
            "goal": TRAFFIC_CASE[1],
            "criterion": "balanced",
            "results": traffic,
        },
        "multi_case": {
            "start": MULTI_START,
            "requested_waypoints": list(MULTI_WAYPOINTS),
            "original_order": _original_order_metrics(
                planner, MULTI_START, MULTI_WAYPOINTS
            ),
            "nearest_neighbor": _multi_summary(
                planner, multi["comparison"]["nearest_neighbor"]
            ),
            "exact_bruteforce": _multi_summary(
                planner, multi["comparison"]["exact_bruteforce"]
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/benchmark_results.json"),
    )
    args = parser.parse_args()
    payload = run_benchmark(args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote benchmark results to {args.output}")


if __name__ == "__main__":
    main()
