"""Foundation, JSON adapter, and serialization tests."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from algorithms.dijkstra import dijkstra
from core.contracts import (
    Edge,
    coordinates_from_data,
    graph_from_dict,
    load_graph_json,
)
from core.cost import WeightedCostFunction
from core.heuristic import HaversineHeuristic, zero_heuristic
from core.serialization import save_result_json
from tests.helpers import edge, metadata_cost


ROOT = Path(__file__).resolve().parents[1]


class FoundationTests(unittest.TestCase):
    def test_mock_graph_and_config_load(self) -> None:
        graph = load_graph_json(ROOT / "data" / "mock_graph.json")
        cost_fn = WeightedCostFunction.from_json(
            ROOT / "data" / "mock_config.json"
        )
        self.assertGreaterEqual(len(graph), 8)
        self.assertGreater(cost_fn("ben_thanh_market", graph["ben_thanh_market"][0]), 0)

    def test_graph_adapter_does_not_infer_reverse_edges(self) -> None:
        graph = graph_from_dict(
            {
                "A": [
                    {
                        "to": "B",
                        "distance_km": 1,
                        "time_min": 2,
                        "congestion": 1,
                        "risk": 0,
                        "road_type": "test",
                    }
                ],
                "B": [],
            }
        )
        self.assertEqual(len(graph["A"]), 1)
        self.assertEqual(graph["B"], ())

    def test_edge_validation_rejects_invalid_congestion(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 1 and 5"):
            Edge("B", 1, 1, 6, 0, "test")

    def test_weight_validation_rejects_negative_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            WeightedCostFunction(alpha_distance=-1)

    def test_haversine_and_zero_heuristics(self) -> None:
        heuristic = HaversineHeuristic(
            {"A": (10.0, 106.0), "B": (10.01, 106.01)}
        )
        self.assertGreater(heuristic("A", "B"), 0)
        self.assertTrue(math.isclose(heuristic("A", "A"), 0.0, abs_tol=1e-12))
        self.assertEqual(zero_heuristic("A", "B"), 0.0)

    def test_coordinates_are_extracted_from_node_metadata(self) -> None:
        with (ROOT / "data" / "mock_graph.json").open(
            "r", encoding="utf-8"
        ) as stream:
            raw_data = json.load(stream)
        coordinates = coordinates_from_data(raw_data)
        self.assertIn("ben_thanh_market", coordinates)
        self.assertEqual(len(coordinates["ben_thanh_market"]), 2)

    def test_result_to_dict_is_json_serializable(self) -> None:
        result = dijkstra(
            {"A": [edge("B", 1)], "B": []}, "A", "B", metadata_cost
        )
        encoded = json.dumps(result.to_dict(), ensure_ascii=False)
        self.assertIn('"algorithm": "dijkstra"', encoded)
        self.assertIsInstance(result.to_dict()["trace"], list)

    def test_save_result_json_writes_utf8_file(self) -> None:
        destination = ROOT / "tests" / "_result_test.json"
        self.addCleanup(lambda: destination.unlink(missing_ok=True))
        result = dijkstra({"A": []}, "A", "A", metadata_cost)
        written = save_result_json(result, destination)
        self.assertEqual(written, destination)
        with destination.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        self.assertEqual(data["path"], ["A"])


if __name__ == "__main__":
    unittest.main()
