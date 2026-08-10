"""Integration tests for OSM loading, service output, REST, and WebSocket."""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app.main import app
from core.osm_loader import (
    MAX_LANDMARK_ROAD_OFFSET_M,
    apply_traffic_profile,
    load_traffic_network,
)
from core.service import RoutePlanner


class OsmNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.network = load_traffic_network()

    def test_real_graph_exceeds_course_minimum(self) -> None:
        self.assertGreaterEqual(len(self.network.graph), 20)
        self.assertGreaterEqual(
            sum(len(edges) for edges in self.network.graph.values()), 30
        )
        self.assertGreaterEqual(len(self.network.landmarks), 20)

    def test_every_landmark_is_snapped_to_routable_node(self) -> None:
        for landmark in self.network.landmarks.values():
            self.assertIn(landmark.snapped_node, self.network.graph)

    def test_landmarks_use_unique_routable_nodes(self) -> None:
        snapped_nodes = [
            landmark.snapped_node for landmark in self.network.landmarks.values()
        ]
        self.assertEqual(len(snapped_nodes), len(set(snapped_nodes)))

    def test_landmark_access_points_are_close_to_the_driving_graph(self) -> None:
        for landmark in self.network.landmarks.values():
            with self.subTest(landmark=landmark.id):
                self.assertLessEqual(
                    landmark.snapped_distance_m,
                    MAX_LANDMARK_ROAD_OFFSET_M,
                )
                self.assertTrue(landmark.access_label)
                self.assertTrue(landmark.access_road)

    def test_exported_vietnamese_road_names_are_normalized(self) -> None:
        names = [
            edge.metadata.get("name")
            for edges in self.network.graph.values()
            for edge in edges
            if edge.metadata.get("name")
        ]
        access_roads = [
            landmark.access_road for landmark in self.network.landmarks.values()
        ]
        self.assertFalse(any("Ð" in name for name in [*names, *access_roads]))
        self.assertFalse(any(name.startswith("[") for name in names))
        self.assertTrue(any(" / " in name for name in names))

    def test_traffic_profile_returns_copy_and_changes_metrics(self) -> None:
        source = next(node for node, edges in self.network.graph.items() if edges)
        base_edge = self.network.graph[source][0]
        rush_graph = apply_traffic_profile(self.network.graph, "rush_hour")
        self.assertGreater(rush_graph[source][0].time_min, base_edge.time_min)
        self.assertEqual(self.network.graph[source][0], base_edge)

    def test_network_and_route_geojson_are_valid(self) -> None:
        roads = self.network.roads_geojson()
        self.assertEqual(roads["type"], "FeatureCollection")
        self.assertGreater(len(roads["features"]), 30)


class RoutePlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.planner = RoutePlanner()

    def test_all_algorithms_run_on_osm_graph(self) -> None:
        for algorithm in ("bfs", "dfs", "ucs", "astar", "dijkstra", "greedy"):
            with self.subTest(algorithm=algorithm):
                payload = self.planner.search(
                    "notre_dame_cathedral",
                    "central_post_office",
                    algorithm,
                    capture_trace=False,
                )
                self.assertTrue(payload["result"]["success"])
                self.assertGreaterEqual(
                    len(payload["route_geojson"]["geometry"]["coordinates"]), 1
                )

    def test_comparison_reports_tied_cost_optimal_algorithms(self) -> None:
        comparison = self.planner.compare(
            "notre_dame_cathedral", "saigon_zoo"
        )
        leaders = comparison["summary"]["leaders"]["lowest_cost"]["algorithms"]
        self.assertIn("uniform_cost_search", leaders)
        self.assertIn("dijkstra", leaders)

    def test_single_search_explains_a_distinct_alternative(self) -> None:
        payload = self.planner.search(
            "notre_dame_cathedral",
            "saigon_zoo",
            "dijkstra",
            capture_trace=False,
        )
        alternative = payload["alternative"]
        self.assertIsNotNone(alternative)
        self.assertNotEqual(
            payload["result"]["path"], alternative["result"]["path"]
        )
        explanation = payload["explanation"]
        self.assertIsNotNone(explanation["alternative_comparison"])
        self.assertIn("alternative has weighted cost", explanation["reasons"][-1])

    def test_multi_route_returns_actual_landmark_order(self) -> None:
        payload = self.planner.multi_route(
            "notre_dame_cathedral",
            ["central_post_office", "saigon_opera_house", "bach_dang_wharf"],
            method="exact_bruteforce",
            compare_methods=True,
        )
        self.assertTrue(payload["result"]["success"])
        self.assertEqual(len(payload["visiting_landmarks"]), 3)
        landmark_by_node = {
            landmark.snapped_node: landmark.id
            for landmark in self.planner.network.landmarks.values()
        }
        self.assertEqual(
            [item["id"] for item in payload["visiting_landmarks"]],
            [
                landmark_by_node[node]
                for node in payload["result"]["visiting_order"]
            ],
        )
        self.assertIsNotNone(payload["comparison"])
        self.assertEqual(
            len(payload["route_segments"]),
            len(payload["result"]["full_path"]) - 1,
        )
        self.assertTrue(
            all(segment["geometry"] for segment in payload["route_segments"])
        )

    def test_single_route_segments_include_map_geometry(self) -> None:
        payload = self.planner.search(
            "notre_dame_cathedral",
            "saigon_zoo",
            "dijkstra",
            capture_trace=False,
        )
        self.assertTrue(payload["route_segments"])
        self.assertTrue(
            all(segment["geometry"] for segment in payload["route_segments"])
        )

    def test_held_karp_service_route_is_supported(self) -> None:
        payload = self.planner.multi_route(
            "fine_arts_museum",
            [
                "ben_thanh_market",
                "nguyen_hue_walking_street",
                "bach_dang_wharf",
            ],
            method="held_karp",
            compare_methods=True,
        )
        self.assertTrue(payload["result"]["success"])
        self.assertEqual(payload["result"]["method"], "held_karp")
        self.assertEqual(
            payload["result"]["optimality"],
            "optimal_for_reduced_pairwise_problem",
        )
        self.assertIsNotNone(payload["comparison"])


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_health_and_bootstrap(self) -> None:
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertGreaterEqual(health.json()["routable_nodes"], 20)
        bootstrap = self.client.get("/api/bootstrap")
        self.assertEqual(bootstrap.status_code, 200)
        self.assertGreaterEqual(len(bootstrap.json()["landmarks"]), 20)

    def test_search_compare_and_multi_endpoints(self) -> None:
        base = {
            "start": "notre_dame_cathedral",
            "goal": "central_post_office",
        }
        search = self.client.post(
            "/api/search", json={**base, "algorithm": "dijkstra"}
        )
        self.assertEqual(search.status_code, 200)
        self.assertTrue(search.json()["result"]["success"])
        comparison = self.client.post("/api/compare", json=base)
        self.assertEqual(comparison.status_code, 200)
        self.assertEqual(len(comparison.json()["algorithms"]), 6)
        multi = self.client.post(
            "/api/multi-route",
            json={
                "start": "notre_dame_cathedral",
                "waypoints": ["central_post_office", "saigon_opera_house"],
            },
        )
        self.assertEqual(multi.status_code, 200)
        self.assertTrue(multi.json()["result"]["success"])

    def test_invalid_landmark_returns_422(self) -> None:
        response = self.client.post(
            "/api/search",
            json={"start": "missing", "goal": "saigon_zoo", "algorithm": "bfs"},
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("Unknown landmark", response.json()["detail"])

    def test_websocket_streams_steps_and_complete_payload(self) -> None:
        message_types: list[str] = []
        with self.client.websocket_connect("/ws/search") as websocket:
            websocket.send_json(
                {
                    "start": "notre_dame_cathedral",
                    "goal": "central_post_office",
                    "algorithm": "astar",
                }
            )
            while True:
                message = websocket.receive_json()
                message_types.append(message["type"])
                if message["type"] == "complete":
                    self.assertTrue(message["payload"]["result"]["success"])
                    break
        self.assertIn("step", message_types)
        self.assertEqual(message_types[0], "started")

    def test_websocket_bounds_large_graph_animation_payload(self) -> None:
        steps: list[dict] = []
        completed_payload: dict | None = None
        with self.client.websocket_connect("/ws/search") as websocket:
            websocket.send_json(
                {
                    "start": "vinh_nghiem_pagoda",
                    "goal": "bach_dang_wharf",
                    "algorithm": "dijkstra",
                }
            )
            while True:
                message = websocket.receive_json()
                if message["type"] == "step":
                    steps.append(message["step"])
                if message["type"] == "complete":
                    self.assertTrue(message["payload"]["result"]["success"])
                    completed_payload = message["payload"]
                    break
        self.assertGreater(len(steps), 1)
        self.assertIsNotNone(completed_payload)
        self.assertTrue(all(step["event"] == "expand" for step in steps))
        self.assertEqual(
            len(steps), completed_payload["result"]["expanded_nodes"]
        )
        self.assertTrue(
            all(step["details"]["sample_rate"] == 1 for step in steps)
        )
        self.assertEqual(
            [step["details"]["visited_count"] for step in steps],
            list(range(1, len(steps) + 1)),
        )
        self.assertTrue(all(step["visited"] == [] for step in steps))
        self.assertTrue(
            all(step["visited_delta"] == [step["current_node"]] for step in steps)
        )
        self.assertTrue(
            all(step["details"]["visited_encoding"] == "delta" for step in steps)
        )
        self.assertTrue(all(len(step["frontier"]) <= 80 for step in steps))
        self.assertTrue(
            all("visited_count" in step["details"] for step in steps)
        )


if __name__ == "__main__":
    unittest.main()
