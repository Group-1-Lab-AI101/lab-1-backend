"""Tests for nearest-neighbor, exact ordering, and comparison helpers."""

from __future__ import annotations

import copy
import math
import unittest

from core.multi_location import (
    SUPPORTED_METHODS,
    compare_multi_location_methods,
    optimize_multi_location,
)
from tests.helpers import comparison_graph, edge, metadata_cost


class MultiLocationTests(unittest.TestCase):
    def test_supported_methods_are_nearest_neighbor_and_bruteforce(self) -> None:
        self.assertEqual(
            SUPPORTED_METHODS,
            {"nearest_neighbor", "exact_bruteforce"},
        )

    def test_nearest_neighbor_visits_every_waypoint(self) -> None:
        result = optimize_multi_location(
            comparison_graph(),
            "S",
            ["A", "B", "C"],
            metadata_cost,
        )
        self.assertTrue(result.success)
        self.assertEqual(set(result.visiting_order), {"A", "B", "C"})
        self.assertEqual(len(result.visiting_order), 3)

    def test_full_path_has_no_duplicated_segment_boundaries(self) -> None:
        result = optimize_multi_location(
            comparison_graph(), "S", ["A", "B", "C"], metadata_cost
        )
        expected = [result.segments[0].path[0]]
        for segment in result.segments:
            expected.extend(segment.path[1:])
        self.assertEqual(result.full_path, expected)
        boundary_index = 0
        for left, right in zip(result.segments, result.segments[1:]):
            boundary_index += len(left.path) - 1
            self.assertEqual(
                result.full_path[boundary_index],
                right.path[0],
            )

    def test_fixed_end_is_reached_after_waypoints(self) -> None:
        graph = comparison_graph()
        result = optimize_multi_location(
            graph, "S", ["A", "B"], metadata_cost, end="C"
        )
        self.assertTrue(result.success)
        self.assertEqual(result.full_path[-1], "C")
        self.assertEqual(set(result.visiting_order), {"A", "B"})

    def test_return_to_start(self) -> None:
        graph = comparison_graph()
        graph["A"].append(edge("S", 2))
        graph["B"].append(edge("S", 2))
        graph["C"].append(edge("S", 3))
        result = optimize_multi_location(
            graph,
            "S",
            ["A", "B"],
            metadata_cost,
            return_to_start=True,
        )
        self.assertTrue(result.success)
        self.assertEqual(result.full_path[0], "S")
        self.assertEqual(result.full_path[-1], "S")

    def test_duplicate_waypoints_are_stably_removed(self) -> None:
        result = optimize_multi_location(
            comparison_graph(),
            "S",
            ["A", "A", "S", "B", "A"],
            metadata_cost,
        )
        self.assertEqual(result.requested_waypoints, ["A", "B"])
        self.assertEqual(result.visiting_order, ["A", "B"])
        self.assertIn("removed", result.message)

    def test_unreachable_segment_returns_clear_failure(self) -> None:
        graph = {"S": [edge("A", 1)], "A": [], "B": []}
        result = optimize_multi_location(
            graph, "S", ["A", "B"], metadata_cost
        )
        self.assertFalse(result.success)
        self.assertIsNone(result.total_cost)
        self.assertIn("'A' -> 'B'", result.message)

    def test_exact_method_finds_best_order(self) -> None:
        result = optimize_multi_location(
            comparison_graph(),
            "S",
            ["A", "B", "C"],
            metadata_cost,
            method="exact_bruteforce",
        )
        self.assertTrue(result.success)
        self.assertEqual(result.visiting_order, ["B", "A", "C"])
        self.assertTrue(math.isclose(result.total_cost or -1, 6.0))
        self.assertEqual(
            result.optimality, "optimal_for_reduced_pairwise_problem"
        )

    def test_exact_limit_is_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "nearest_neighbor"):
            optimize_multi_location(
                comparison_graph(),
                "S",
                ["A", "B", "C"],
                metadata_cost,
                method="exact_bruteforce",
                exact_limit=2,
            )

    def test_comparison_gap_is_correct(self) -> None:
        comparison = compare_multi_location_methods(
            comparison_graph(), "S", ["A", "B", "C"], metadata_cost
        )
        nearest = comparison["nearest_neighbor"]
        exact = comparison["exact_bruteforce"]
        expected = ((nearest.total_cost or 0) - (exact.total_cost or 0))
        expected = expected / (exact.total_cost or 1) * 100
        self.assertTrue(
            math.isclose(
                nearest.comparison_gap_percent or -1,
                expected,
                rel_tol=1e-12,
            )
        )
        self.assertEqual(exact.comparison_gap_percent, 0.0)

    def test_pairwise_source_cache_bounds_cost_evaluations(self) -> None:
        graph = comparison_graph()
        edge_count = sum(len(edges) for edges in graph.values())
        calls = 0

        def counting_cost(source, road):
            nonlocal calls
            calls += 1
            return metadata_cost(source, road)

        compare_multi_location_methods(
            graph, "S", ["A", "B", "C"], counting_cost
        )
        self.assertLessEqual(calls, edge_count * len(graph))

    def test_totals_equal_sum_of_segments(self) -> None:
        result = optimize_multi_location(
            comparison_graph(), "S", ["A", "B", "C"], metadata_cost
        )
        self.assertTrue(
            math.isclose(
                result.total_cost or -1,
                sum(segment.cost for segment in result.segments),
            )
        )
        self.assertTrue(
            math.isclose(
                result.total_distance_km or -1,
                sum(segment.distance_km for segment in result.segments),
            )
        )
        self.assertTrue(
            math.isclose(
                result.total_time_min or -1,
                sum(segment.time_min for segment in result.segments),
            )
        )

    def test_tie_breaking_uses_waypoint_input_order(self) -> None:
        graph = {
            "S": [edge("A", 1), edge("B", 1)],
            "A": [edge("B", 1)],
            "B": [edge("A", 1)],
        }
        result = optimize_multi_location(
            graph, "S", ["B", "A"], metadata_cost
        )
        self.assertEqual(result.visiting_order, ["B", "A"])

    def test_visiting_order_follows_intermediate_waypoint_crossing(self) -> None:
        graph = {
            "S": [edge("A", 1)],
            "A": [edge("B", 0)],
            "B": [],
        }
        for method in ("nearest_neighbor", "exact_bruteforce"):
            with self.subTest(method=method):
                result = optimize_multi_location(
                    graph,
                    "S",
                    ["B", "A"],
                    metadata_cost,
                    method=method,
                )
                self.assertTrue(result.success)
                self.assertEqual(result.visiting_order, ["A", "B"])
                self.assertEqual(result.full_path, ["S", "A", "B"])
                self.assertEqual(len(result.segments), 1)

    def test_conflicting_end_and_return_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires end"):
            optimize_multi_location(
                comparison_graph(),
                "S",
                ["A"],
                metadata_cost,
                end="C",
                return_to_start=True,
            )

    def test_graph_is_not_mutated(self) -> None:
        graph = comparison_graph()
        before = copy.deepcopy(graph)
        optimize_multi_location(graph, "S", ["A", "B"], metadata_cost)
        self.assertEqual(graph, before)

    def test_empty_waypoints_produces_zero_cost_route(self) -> None:
        result = optimize_multi_location(
            comparison_graph(), "S", [], metadata_cost
        )
        self.assertTrue(result.success)
        self.assertEqual(result.full_path, ["S"])
        self.assertEqual(result.total_cost, 0)


if __name__ == "__main__":
    unittest.main()
