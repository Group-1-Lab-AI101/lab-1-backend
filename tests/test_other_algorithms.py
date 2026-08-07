"""Tests for BFS, DFS, UCS, and A*."""

from __future__ import annotations

import math
import unittest

from algorithms import (
    a_star_search,
    breadth_first_search,
    depth_first_search,
    dijkstra,
    uniform_cost_search,
)
from tests.helpers import edge, metadata_cost


class OtherAlgorithmTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = {
            "A": [edge("B", 8), edge("C", 1)],
            "B": [edge("G", 1)],
            "C": [edge("D", 1)],
            "D": [edge("G", 1)],
            "G": [],
        }

    def test_bfs_minimizes_edge_count_not_cost(self) -> None:
        result = breadth_first_search(self.graph, "A", "G", metadata_cost)
        self.assertEqual(result.path, ["A", "B", "G"])
        self.assertEqual(result.total_cost, 9.0)
        self.assertEqual(result.optimality, "optimal_by_edge_count")

    def test_dfs_is_deterministic_and_cycle_safe(self) -> None:
        graph = dict(self.graph)
        graph["D"] = [edge("A", 1), edge("G", 1)]
        result = depth_first_search(graph, "A", "G", metadata_cost)
        self.assertTrue(result.success)
        self.assertEqual(result.path, ["A", "B", "G"])
        self.assertLessEqual(result.expanded_nodes, len(graph))

    def test_ucs_matches_dijkstra_cost(self) -> None:
        ucs = uniform_cost_search(self.graph, "A", "G", metadata_cost)
        optimal = dijkstra(self.graph, "A", "G", metadata_cost)
        self.assertEqual(ucs.path, ["A", "C", "D", "G"])
        self.assertTrue(math.isclose(ucs.total_cost or -1, optimal.total_cost or -2))
        self.assertEqual(ucs.algorithm, "uniform_cost_search")

    def test_ucs_has_its_own_goal_directed_trace(self) -> None:
        result = uniform_cost_search(self.graph, "A", "G", metadata_cost)
        events = [step.event for step in result.trace]
        self.assertIn("relax", events)
        self.assertEqual(events[-1], "goal")
        self.assertEqual(result.optimality, "optimal")

    def test_required_algorithms_handle_start_equal_goal(self) -> None:
        calls = (
            lambda: breadth_first_search(
                self.graph, "A", "A", metadata_cost
            ),
            lambda: depth_first_search(
                self.graph, "A", "A", metadata_cost
            ),
            lambda: uniform_cost_search(
                self.graph, "A", "A", metadata_cost
            ),
            lambda: a_star_search(
                self.graph,
                "A",
                "A",
                lambda _node, _goal: 0,
                metadata_cost,
            ),
        )
        for call in calls:
            with self.subTest(call=call):
                result = call()
                self.assertTrue(result.success)
                self.assertEqual(result.path, ["A"])
                self.assertEqual(result.total_cost, 0.0)

    def test_astar_uses_g_plus_h_and_finds_optimum(self) -> None:
        heuristic = {"A": 2, "B": 0, "C": 2, "D": 1, "G": 0}
        result = a_star_search(
            self.graph,
            "A",
            "G",
            lambda node, _goal: heuristic[node],
            metadata_cost,
        )
        self.assertEqual(result.path, ["A", "C", "D", "G"])
        self.assertEqual(result.total_cost, 3.0)
        self.assertIn("relax", [step.event for step in result.trace])

    def test_astar_trace_exposes_g_h_and_f(self) -> None:
        result = a_star_search(
            self.graph,
            "A",
            "G",
            lambda node, _goal: {"A": 2, "B": 0, "C": 2, "D": 1, "G": 0}[node],
            metadata_cost,
        )
        relax = next(step for step in result.trace if step.event == "relax")
        self.assertTrue({"g", "h", "f"}.issubset(relax.details))

    def test_all_algorithms_return_unreachable_result(self) -> None:
        graph = {"A": [], "G": []}
        calls = (
            lambda: breadth_first_search(graph, "A", "G", metadata_cost),
            lambda: depth_first_search(graph, "A", "G", metadata_cost),
            lambda: uniform_cost_search(graph, "A", "G", metadata_cost),
            lambda: a_star_search(
                graph, "A", "G", lambda _node, _goal: 0, metadata_cost
            ),
        )
        for call in calls:
            with self.subTest(call=call):
                self.assertFalse(call().success)


if __name__ == "__main__":
    unittest.main()
