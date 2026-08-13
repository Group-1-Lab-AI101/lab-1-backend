"""Run Dijkstra, Greedy, and multi-location examples end to end."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.dijkstra import dijkstra
from algorithms.greedy_best_first import greedy_best_first
from core.contracts import coordinates_from_data, load_graph_json
from core.cost import WeightedCostFunction
from core.heuristic import HaversineHeuristic
from core.multi_location import compare_multi_location_methods
from core.serialization import save_result_json


def print_search_step(step) -> None:
    """Example adapter that a GUI can replace with an animation update."""
    if step.event in {"expand", "goal"}:
        print(
            f"  step={step.index:02d} event={step.event:<6} "
            f"node={step.current_node}"
        )


def main() -> None:
    """Load demo data, run every Kiet-owned feature, and save JSON results."""
    graph_path = ROOT / "data" / "mock_graph.json"
    config_path = ROOT / "data" / "mock_config.json"
    output_dir = ROOT / "output" / "demo"

    with graph_path.open("r", encoding="utf-8") as stream:
        raw_graph_data = json.load(stream)
    graph = load_graph_json(graph_path)
    cost_fn = WeightedCostFunction.from_json(config_path)
    heuristic_fn = HaversineHeuristic(coordinates_from_data(raw_graph_data))

    print("Dijkstra trace:")
    dijkstra_result = dijkstra(
        graph,
        "ben_thanh_market",
        "saigon_zoo",
        cost_fn,
        capture_trace=True,
        on_step=print_search_step,
    )
    print(
        f"  path={dijkstra_result.path}\n"
        f"  cost={dijkstra_result.total_cost:.2f}\n"
    )

    greedy_result = greedy_best_first(
        graph,
        "ben_thanh_market",
        "saigon_zoo",
        heuristic_fn,
        cost_fn,
    )
    print("Greedy Best-First:")
    print(
        f"  path={greedy_result.path}\n"
        f"  cost={greedy_result.total_cost:.2f}\n"
        f"  optimality={greedy_result.optimality}\n"
    )

    comparison = compare_multi_location_methods(
        graph,
        "ben_thanh_market",
        [
            "independence_palace",
            "central_post_office",
            "saigon_zoo",
            "bach_dang_wharf",
        ],
        cost_fn,
        exact_limit=8,
    )
    print("Multi-location comparison:")
    for method, result in comparison.items():
        print(
            f"  {method}: order={result.visiting_order}, "
            f"cost={result.total_cost:.2f}, "
            f"gap={result.comparison_gap_percent}"
        )

    save_result_json(dijkstra_result, output_dir / "dijkstra.json")
    save_result_json(greedy_result, output_dir / "greedy_best_first.json")
    save_result_json(
        comparison["nearest_neighbor"], output_dir / "multi_nearest.json"
    )
    save_result_json(
        comparison["exact_bruteforce"], output_dir / "multi_exact.json"
    )
    print(f"\nJSON results written to {output_dir}")


if __name__ == "__main__":
    main()
