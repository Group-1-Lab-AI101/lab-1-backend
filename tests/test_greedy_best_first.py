"""Behavioral tests for Greedy Best-First Search."""

from __future__ import annotations

import math
import unittest

from algorithms.dijkstra import dijkstra
from algorithms.greedy_best_first import greedy_best_first
from tests.helpers import edge, metadata_cost


class GreedyBestFirstTests(unittest.TestCase):
    def test_finds_a_path(self) -> None:
        graph = {
            "A": [edge("B", 2), edge("C", 1)],
            "B": [edge("G", 2)],
            "C": [],
            "G": [],
        }
        heuristic = {"A": 3, "B": 1, "C": 5, "G": 0}
        result = greedy_best_first(
            graph, "A", "G", lambda node, _goal: heuristic[node], metadata_cost
        )
        self.assertTrue(result.success)
        self.assertEqual(result.path, ["A", "B", "G"])
        self.assertEqual(result.optimality, "not_guaranteed")

    def test_start_equals_goal(self) -> None:
        result = greedy_best_first(
            {"A": []}, "A", "A", lambda _node, _goal: 0, metadata_cost
        )
        self.assertEqual(result.path, ["A"])
        self.assertEqual(result.total_cost, 0.0)
        self.assertEqual(result.expanded_nodes, 1)

    def test_unreachable_goal(self) -> None:
        graph = {"A": [edge("B", 1)], "B": [], "G": []}
        result = greedy_best_first(
            graph, "A", "G", lambda _node, _goal: 0, metadata_cost
        )
        self.assertFalse(result.success)
        self.assertEqual(result.path, [])

    def test_cycle_terminates(self) -> None:
        graph = {
            "A": [edge("B", 1)],
            "B": [edge("A", 1), edge("C", 1)],
            "C": [edge("B", 1)],
            "G": [],
        }
        result = greedy_best_first(
            graph, "A", "G", lambda _node, _goal: 0, metadata_cost
        )
        self.assertFalse(result.success)
        self.assertEqual(result.expanded_nodes, 3)

    def test_negative_heuristic_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            greedy_best_first(
                {"A": [edge("G", 1)], "G": []},
                "A",
                "G",
                lambda _node, _goal: -1,
                metadata_cost,
            )

    def test_non_finite_heuristic_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            greedy_best_first(
                {"A": []},
                "A",
                "A",
                lambda _node, _goal: float("nan"),
                metadata_cost,
            )

    def test_equal_heuristic_uses_adjacency_order(self) -> None:
        graph = {
            "A": [edge("B", 1), edge("C", 1)],
            "B": [edge("G", 1)],
            "C": [edge("G", 1)],
            "G": [],
        }
        heuristic = {"A": 2, "B": 1, "C": 1, "G": 0}
        result = greedy_best_first(
            graph, "A", "G", lambda node, _goal: heuristic[node], metadata_cost
        )
        self.assertEqual(result.path, ["A", "B", "G"])

    def test_can_return_more_expensive_path_than_dijkstra(self) -> None:
        graph = {
            "A": [edge("B", 1), edge("C", 2)],
            "B": [edge("G", 100)],
            "C": [edge("G", 2)],
            "G": [],
        }
        heuristic = {"A": 3, "B": 1, "C": 2, "G": 0}
        greedy = greedy_best_first(
            graph, "A", "G", lambda node, _goal: heuristic[node], metadata_cost
        )
        optimal = dijkstra(graph, "A", "G", metadata_cost)
        self.assertEqual(greedy.path, ["A", "B", "G"])
        self.assertGreater(greedy.total_cost or 0, optimal.total_cost or 0)

    def test_total_cost_is_measured_from_final_path(self) -> None:
        graph = {
            "A": [edge("B", 7, distance=2, time=4)],
            "B": [edge("G", 3, distance=5, time=6)],
            "G": [],
        }
        heuristic = {"A": 100, "B": 50, "G": 0}
        result = greedy_best_first(
            graph, "A", "G", lambda node, _goal: heuristic[node], metadata_cost
        )
        self.assertTrue(math.isclose(result.total_cost or -1, 10.0))
        self.assertEqual(result.total_distance_km, 7.0)
        self.assertEqual(result.total_time_min, 10.0)

    def test_duplicate_frontier_node_is_not_generated_twice(self) -> None:
        graph = {
            "A": [edge("B", 1), edge("C", 1)],
            "B": [edge("D", 1)],
            "C": [edge("D", 1)],
            "D": [],
        }
        heuristic = {"A": 2, "B": 1, "C": 1, "D": 0}
        result = greedy_best_first(
            graph, "A", "D", lambda node, _goal: heuristic[node], metadata_cost
        )
        self.assertEqual(result.generated_nodes, 4)
        self.assertEqual(result.path, ["A", "B", "D"])

    def test_callback_uses_shared_step_contract(self) -> None:
        events = []
        result = greedy_best_first(
            {"A": [edge("G", 1)], "G": []},
            "A",
            "G",
            lambda _node, _goal: 0,
            metadata_cost,
            capture_trace=False,
            on_step=events.append,
        )
        self.assertEqual(result.trace, [])
        self.assertEqual(events[-1].event, "goal")
        self.assertTrue(hasattr(events[-1], "frontier"))


if __name__ == "__main__":
    unittest.main()
