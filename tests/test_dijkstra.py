"""Behavioral tests for Dijkstra and its single-source helper."""

from __future__ import annotations

import copy
import math
import unittest

from algorithms.dijkstra import dijkstra, dijkstra_all
from tests.helpers import edge, metadata_cost


class DijkstraTests(unittest.TestCase):
    def test_finds_lowest_cost_path(self) -> None:
        graph = {
            "A": [edge("B", 4), edge("C", 1)],
            "B": [edge("D", 1)],
            "C": [edge("B", 1), edge("D", 5)],
            "D": [],
        }
        result = dijkstra(graph, "A", "D", metadata_cost)
        self.assertTrue(result.success)
        self.assertEqual(result.path, ["A", "C", "B", "D"])
        self.assertTrue(math.isclose(result.total_cost or -1, 3.0))
        self.assertEqual(result.optimality, "optimal")

    def test_does_not_stop_when_goal_is_only_discovered(self) -> None:
        graph = {
            "A": [edge("G", 10), edge("B", 1)],
            "B": [edge("G", 1)],
            "G": [],
        }
        result = dijkstra(graph, "A", "G", metadata_cost)
        self.assertEqual(result.path, ["A", "B", "G"])
        self.assertTrue(math.isclose(result.total_cost or -1, 2.0))
        self.assertEqual(result.visited_order[-1], "G")

    def test_start_equals_goal(self) -> None:
        result = dijkstra({"A": []}, "A", "A", metadata_cost)
        self.assertTrue(result.success)
        self.assertEqual(result.path, ["A"])
        self.assertEqual(result.total_cost, 0.0)
        self.assertEqual(result.expanded_nodes, 1)

    def test_unreachable_goal_returns_failure(self) -> None:
        graph = {"A": [edge("B", 1)], "B": [], "C": []}
        result = dijkstra(graph, "A", "C", metadata_cost)
        self.assertFalse(result.success)
        self.assertEqual(result.path, [])
        self.assertIsNone(result.total_cost)
        self.assertIn("No route", result.message)

    def test_directed_edge_is_not_reversed(self) -> None:
        graph = {"A": [edge("B", 1)], "B": []}
        self.assertTrue(dijkstra(graph, "A", "B", metadata_cost).success)
        self.assertFalse(dijkstra(graph, "B", "A", metadata_cost).success)

    def test_equal_priority_uses_insertion_order(self) -> None:
        graph = {
            "A": [edge("B", 1), edge("C", 1)],
            "B": [edge("G", 1)],
            "C": [edge("G", 1)],
            "G": [],
        }
        result = dijkstra(graph, "A", "G", metadata_cost)
        self.assertEqual(result.path, ["A", "B", "G"])
        self.assertLess(
            result.visited_order.index("B"), result.visited_order.index("C")
        )

    def test_negative_cost_is_rejected(self) -> None:
        graph = {"A": [edge("B", 1)], "B": []}
        with self.assertRaisesRegex(ValueError, "non-negative"):
            dijkstra(graph, "A", "B", lambda _source, _road: -1)

    def test_stale_heap_entry_is_skipped(self) -> None:
        graph = {
            "A": [edge("B", 10), edge("C", 1)],
            "B": [],
            "C": [edge("B", 1)],
            "D": [],
        }
        result = dijkstra(graph, "A", "D", metadata_cost)
        self.assertFalse(result.success)
        self.assertIn("skip_stale", [step.event for step in result.trace])
        self.assertEqual(result.visited_order.count("B"), 1)

    def test_path_distance_and_time_use_selected_edges(self) -> None:
        graph = {
            "A": [edge("B", 1, distance=8, time=3)],
            "B": [edge("C", 2, distance=4, time=7)],
            "C": [],
        }
        result = dijkstra(graph, "A", "C", metadata_cost)
        self.assertEqual(result.total_cost, 3.0)
        self.assertEqual(result.total_distance_km, 12.0)
        self.assertEqual(result.total_time_min, 10.0)

    def test_input_graph_is_not_mutated(self) -> None:
        graph = {"A": [edge("B", 1)], "B": []}
        before = copy.deepcopy(graph)
        dijkstra(graph, "A", "B", metadata_cost)
        self.assertEqual(graph, before)

    def test_single_source_helper_reuses_one_search(self) -> None:
        graph = {
            "A": [edge("B", 1), edge("C", 5)],
            "B": [edge("C", 1)],
            "C": [],
            "D": [],
        }
        results = dijkstra_all(graph, "A", metadata_cost)
        self.assertEqual(results["C"].path, ["A", "B", "C"])
        self.assertFalse(results["D"].success)
        self.assertEqual(results["A"].path, ["A"])

    def test_callback_runs_when_trace_capture_is_disabled(self) -> None:
        events = []
        result = dijkstra(
            {"A": [edge("B", 1)], "B": []},
            "A",
            "B",
            metadata_cost,
            capture_trace=False,
            on_step=events.append,
        )
        self.assertEqual(result.trace, [])
        self.assertGreater(len(events), 0)
        self.assertEqual(events[-1].event, "goal")

    def test_callback_exception_is_not_swallowed(self) -> None:
        def fail_callback(_step) -> None:
            raise RuntimeError("GUI callback failed")

        with self.assertRaisesRegex(RuntimeError, "GUI callback failed"):
            dijkstra(
                {"A": []},
                "A",
                "A",
                metadata_cost,
                on_step=fail_callback,
            )

    def test_missing_node_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not exist"):
            dijkstra({"A": []}, "A", "missing", metadata_cost)


if __name__ == "__main__":
    unittest.main()
