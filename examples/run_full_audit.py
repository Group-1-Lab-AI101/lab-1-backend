"""Run exhaustive correctness and scenario checks for the final lab dataset."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from algorithms import a_star_search, dijkstra, uniform_cost_search
from core.cost import WeightedCostFunction
from core.heuristic import HaversineHeuristic
from core.osm_loader import apply_traffic_profile, load_traffic_network
from core.service import COST_PRESETS


def _cost_function(criterion: str) -> WeightedCostFunction:
    return WeightedCostFunction.from_dict(COST_PRESETS[criterion])


def _path_key(result_path: list[str]) -> tuple[str, ...]:
    return tuple(result_path)


def run_audit() -> dict[str, object]:
    """Return exhaustive optimality, reachability, and scenario evidence."""
    started = time.perf_counter()
    network = load_traffic_network()
    landmarks = list(network.landmarks.values())
    pairs = [
        (start, goal)
        for start in landmarks
        for goal in landmarks
        if start.id != goal.id
    ]
    normal_graph = apply_traffic_profile(network.graph, "normal")
    balanced_cost = _cost_function("balanced")
    distance_heuristic = HaversineHeuristic(network.coordinates)
    alpha = balanced_cost.alpha_distance

    def scaled_heuristic(node: str, goal: str) -> float:
        return alpha * distance_heuristic(node, goal)

    mismatches: list[dict[str, object]] = []
    normal_paths: dict[tuple[str, str], tuple[str, ...]] = {}
    reachable_pairs = 0
    max_optimal_cost_delta = 0.0

    for start, goal in pairs:
        source = start.snapped_node
        target = goal.snapped_node
        ucs = uniform_cost_search(
            normal_graph, source, target, balanced_cost, capture_trace=False
        )
        astar = a_star_search(
            normal_graph,
            source,
            target,
            scaled_heuristic,
            balanced_cost,
            capture_trace=False,
        )
        shortest = dijkstra(
            normal_graph, source, target, balanced_cost, capture_trace=False
        )
        if ucs.success and astar.success and shortest.success:
            reachable_pairs += 1
            costs = [ucs.total_cost, astar.total_cost, shortest.total_cost]
            assert all(cost is not None for cost in costs)
            numeric_costs = [float(cost) for cost in costs if cost is not None]
            delta = max(numeric_costs) - min(numeric_costs)
            max_optimal_cost_delta = max(max_optimal_cost_delta, delta)
            if not math.isclose(delta, 0.0, abs_tol=1e-9):
                mismatches.append(
                    {
                        "start": start.id,
                        "goal": goal.id,
                        "ucs": ucs.total_cost,
                        "astar": astar.total_cost,
                        "dijkstra": shortest.total_cost,
                    }
                )
            normal_paths[(start.id, goal.id)] = _path_key(shortest.path)
        else:
            mismatches.append(
                {
                    "start": start.id,
                    "goal": goal.id,
                    "error": "At least one optimal algorithm reported unreachable.",
                }
            )

    consistency_violations = 0
    max_consistency_excess = 0.0
    landmark_nodes = [landmark.snapped_node for landmark in landmarks]
    for preset in COST_PRESETS:
        cost_fn = _cost_function(preset)
        preset_alpha = cost_fn.alpha_distance
        for source, edges in normal_graph.items():
            for edge in edges:
                edge_cost = cost_fn(source, edge)
                for goal in landmark_nodes:
                    lhs = preset_alpha * distance_heuristic(source, goal)
                    rhs = edge_cost + preset_alpha * distance_heuristic(edge.to, goal)
                    excess = lhs - rhs
                    max_consistency_excess = max(max_consistency_excess, excess)
                    if excess > 1e-9:
                        consistency_violations += 1

    traffic_changed: set[tuple[str, str]] = set()
    for profile in ("rush_hour", "rainy"):
        profile_graph = apply_traffic_profile(network.graph, profile)
        for start, goal in pairs:
            result = dijkstra(
                profile_graph,
                start.snapped_node,
                goal.snapped_node,
                balanced_cost,
                capture_trace=False,
            )
            if _path_key(result.path) != normal_paths[(start.id, goal.id)]:
                traffic_changed.add((start.id, goal.id))

    criterion_changed: set[tuple[str, str]] = set()
    for criterion in COST_PRESETS:
        if criterion == "balanced":
            continue
        cost_fn = _cost_function(criterion)
        for start, goal in pairs:
            result = dijkstra(
                normal_graph,
                start.snapped_node,
                goal.snapped_node,
                cost_fn,
                capture_trace=False,
            )
            if _path_key(result.path) != normal_paths[(start.id, goal.id)]:
                criterion_changed.add((start.id, goal.id))

    offsets = [landmark.snapped_distance_m for landmark in landmarks]
    return {
        "landmarks": len(landmarks),
        "ordered_pairs": len(pairs),
        "reachable_pairs": reachable_pairs,
        "optimal_algorithm_mismatches": len(mismatches),
        "mismatch_examples": mismatches[:10],
        "max_optimal_cost_delta": max_optimal_cost_delta,
        "heuristic_checks": {
            "presets": len(COST_PRESETS),
            "goals": len(landmark_nodes),
            "directed_edges": sum(len(edges) for edges in normal_graph.values()),
            "consistency_violations": consistency_violations,
            "max_consistency_excess": max_consistency_excess,
        },
        "pairs_changed_by_traffic_profile": len(traffic_changed),
        "pairs_changed_by_cost_criterion": len(criterion_changed),
        "landmark_access_offsets_m": {
            "mean": sum(offsets) / len(offsets),
            "maximum": max(offsets),
            "over_100_m": sum(offset > 100.0 for offset in offsets),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/full_audit_results.json"),
        help="JSON output path",
    )
    args = parser.parse_args()
    payload = run_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
