#!/usr/bin/env python3
"""Full destination-pair test matrix for every backend function (lab-1).

Runs each two-location function against every ordered (start, goal) landmark
pair (24 x 23 = 552 pairs) on the real OSM graph, plus unit-level checks for
the remaining functions. Results are written as JSON; build_testing_md.py
converts them into TESTING.md.

Usage:
    python scripts/run_test_matrix.py [--out /tmp/lab1_matrix.json]
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from algorithms import (
    a_star_search,
    breadth_first_search,
    depth_first_search,
    dijkstra,
    dijkstra_all,
    greedy_best_first,
    uniform_cost_search,
)
from core.contracts import (
    Edge,
    SearchStep,
    coerce_graph,
    coordinates_from_data,
    graph_from_dict,
    load_graph_json,
)
from core.cost import WeightedCostFunction
from core.explanation import (
    explain_search_result,
    route_segments,
    summarize_comparison,
)
from core.heuristic import HaversineHeuristic, zero_heuristic
from core.multi_location import (
    SUPPORTED_METHODS,
    compare_multi_location_methods,
    optimize_multi_location,
)
from core.osm_loader import (
    MAX_LANDMARK_ROAD_OFFSET_M,
    TRAFFIC_PROFILES,
    apply_traffic_profile,
    load_traffic_network,
)
from core.serialization import save_result_json
from core.service import COST_PRESETS, RoutePlanner

RAW_ALGORITHMS = {
    "bfs": breadth_first_search,
    "dfs": depth_first_search,
    "ucs": uniform_cost_search,
    "dijkstra": dijkstra,
    "greedy": greedy_best_first,
    "astar": a_star_search,
}
OPTIMAL_ALGORITHMS = {"ucs", "dijkstra", "astar"}


def round4(value):
    return None if value is None else round(float(value), 4)


def keyed(prefix, name):
    return name if not prefix else f"{prefix}_{name}"


def result_row(prefix, result):
    def get(key):
        if isinstance(result, dict):
            return result.get(key)
        return getattr(result, key)

    return {
        keyed(prefix, "success"): get("success"),
        keyed(prefix, "path_len"): len(get("path") or []),
        keyed(prefix, "cost"): round4(get("total_cost")),
        keyed(prefix, "dist_km"): round4(get("total_distance_km")),
        keyed(prefix, "time_min"): round4(get("total_time_min")),
        keyed(prefix, "expanded"): get("expanded_nodes"),
        keyed(prefix, "generated"): get("generated_nodes"),
        keyed(prefix, "runtime_ms"): round4(get("runtime_ms")),
        keyed(prefix, "optimality"): get("optimality"),
        keyed(prefix, "message"): get("message"),
    }


def run_matrix(network, planner):
    """Run raw algorithms, planner.search, and planner.compare on all pairs."""
    landmarks = list(network.landmarks.values())
    graph_normal = apply_traffic_profile(network.graph, "normal")
    cost_fn = WeightedCostFunction.from_dict(COST_PRESETS["balanced"])
    heuristic = HaversineHeuristic(network.coordinates)

    pairs = []
    for start in landmarks:
        for goal in landmarks:
            if start.id == goal.id:
                continue
            pairs.append(
                {
                    "start": start.id,
                    "goal": goal.id,
                    "start_node": start.snapped_node,
                    "goal_node": goal.snapped_node,
                    "raw": {},
                    "search": {},
                    "compare": {},
                }
            )

    failures = []
    for index, pair in enumerate(pairs, start=1):
        s_node, g_node = pair["start_node"], pair["goal_node"]
        checks = {"path_edges_valid": True, "optimal_agree": True, "search_matches_raw": True}

        for name, algorithm in RAW_ALGORITHMS.items():
            alpha = cost_fn.alpha_distance if name == "astar" else 0.0
            row = {"success": None, "error": None}
            try:
                if name == "greedy":
                    result = algorithm(graph_normal, s_node, g_node, heuristic, cost_fn, capture_trace=False)
                elif name == "astar":
                    result = algorithm(
                        graph_normal, s_node, g_node,
                        lambda node, goal: alpha * heuristic(node, goal),
                        cost_fn, capture_trace=False,
                    )
                else:
                    result = algorithm(graph_normal, s_node, g_node, cost_fn, capture_trace=False)
                row = result_row("", result)
                row["path"] = list(result.path)
                if result.success:
                    for source, target in zip(result.path, result.path[1:]):
                        if not any(edge.to == target for edge in graph_normal[source]):
                            checks["path_edges_valid"] = False
                            failures.append(
                                f"{name}: invalid path edge {source!r} -> {target!r}"
                            )
                            break
            except Exception as error:
                row = {"success": None, "error": f"{type(error).__name__}: {error}"}
                failures.append(f"{name}({pair['start']}->{pair['goal']}): {row['error']}")
            pair["raw"][name] = row

        costs = [
            pair["raw"][name]["cost"]
            for name in OPTIMAL_ALGORITHMS
            if pair["raw"][name]["success"]
        ]
        if costs and len({round(c, 6) for c in costs}) > 1:
            checks["optimal_agree"] = False
            failures.append(
                f"optimal algorithms disagree on cost for {pair['start']}->{pair['goal']}: {costs}"
            )

        for name in RAW_ALGORITHMS:
            row = {"success": None, "error": None}
            try:
                payload = planner.search(
                    pair["start"], pair["goal"], name, capture_trace=False
                )
                result = payload["result"]
                row = result_row("", result)
                raw = pair["raw"][name]
                if raw["success"] and (
                    raw["path"] != result["path"] or raw["cost"] != row["cost"]
                ):
                    checks["search_matches_raw"] = False
                    failures.append(
                        f"planner.search({name}) diverges from raw for "
                        f"{pair['start']}->{pair['goal']}"
                    )
                row["request"] = payload["request"]
                row["explanation_headline"] = payload["explanation"]["headline"]
                row["alternative_algorithm"] = (
                    payload["alternative"]["algorithm"]
                    if payload["alternative"]
                    else None
                )
            except Exception as error:
                row = {"success": None, "error": f"{type(error).__name__}: {error}"}
                failures.append(f"search({name}, {pair['start']}->{pair['goal']}): {row['error']}")
            pair["search"][name] = row

        try:
            comparison = planner.compare(pair["start"], pair["goal"])
            pair["compare"]["summary"] = comparison["summary"]
            pair["compare"]["algorithms"] = {
                item["result"]["algorithm"]: result_row("", item["result"])
                for item in comparison["algorithms"]
            }
        except Exception as error:
            pair["compare"]["error"] = f"{type(error).__name__}: {error}"
            failures.append(f"compare({pair['start']}->{pair['goal']}): {pair['compare']['error']}")

        pair["checks"] = checks
        if index % 100 == 0 or index == len(pairs):
            print(f"  pairs {index}/{len(pairs)} done", flush=True)

    return pairs, failures


def run_dijkstra_all(network):
    """dijkstra_all: every landmark source, all other landmarks as goals."""
    graph_normal = apply_traffic_profile(network.graph, "normal")
    cost_fn = WeightedCostFunction.from_dict(COST_PRESETS["balanced"])
    rows = []
    for start in network.landmarks.values():
        goals = [
            goal.snapped_node
            for goal in network.landmarks.values()
            if goal.id != start.id
        ]
        start_time = time.perf_counter()
        results = dijkstra_all(graph_normal, start.snapped_node, cost_fn, goals=goals, capture_trace=False)
        runtime = (time.perf_counter() - start_time) * 1000
        for goal in network.landmarks.values():
            if goal.id == start.id:
                continue
            result = results[goal.snapped_node]
            rows.append(
                {
                    "start": start.id,
                    "goal": goal.id,
                    **result_row("", result),
                    "source_runtime_ms": round4(runtime),
                }
            )
    return rows


def run_multi_location(network):
    """optimize_multi_location, compare_multi_location_methods, multi_route."""
    graph_normal = apply_traffic_profile(network.graph, "normal")
    cost_fn = WeightedCostFunction.from_dict(COST_PRESETS["balanced"])
    ids = [landmark.id for landmark in network.landmarks.values()]
    nodes = [landmark.snapped_node for landmark in network.landmarks.values()]
    start_node = nodes[0]

    waypoint_sets = {
        "trio": nodes[0:3],
        "five": nodes[0:5],
        "eight": nodes[0:8],
        "twelve": nodes[0:12],
        "all_except_start": nodes[1:],
    }
    rows = []
    for set_name, waypoints in waypoint_sets.items():
        for method in SUPPORTED_METHODS:
            if method == "exact_bruteforce" and len(waypoints) > 8:
                continue
            row = {
                "set": set_name,
                "start": ids[0],
                "waypoint_count": len(waypoints),
                "method": method,
            }
            try:
                result = optimize_multi_location(
                    graph_normal, start_node, waypoints, cost_fn, method=method
                )
                row.update(
                    {
                        "success": result.success,
                        "visiting_order": list(result.visiting_order),
                        "full_path_len": len(result.full_path),
                        "cost": round4(result.total_cost),
                        "dist_km": round4(result.total_distance_km),
                        "time_min": round4(result.total_time_min),
                        "runtime_ms": round4(result.runtime_ms),
                        "optimality": result.optimality,
                        "message": result.message,
                    }
                )
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
            rows.append(row)

    compare_row = {
        "set": "five",
        "start": ids[0],
        "method": "compare_multi_location_methods",
    }
    try:
        comparison = compare_multi_location_methods(
            graph_normal, start_node, nodes[0:5], cost_fn
        )
        compare_row["results"] = {
            name: {
                "success": result.success,
                "cost": round4(result.total_cost),
                "gap_percent": (
                    None if result.comparison_gap_percent is None
                    else round4(result.comparison_gap_percent)
                ),
                "visiting_order": list(result.visiting_order),
                "runtime_ms": round4(result.runtime_ms),
            }
            for name, result in comparison.items()
        }
    except Exception as error:
        compare_row["error"] = f"{type(error).__name__}: {error}"
    rows.append(compare_row)

    for method in ("nearest_neighbor", "exact_bruteforce"):
        facade_row = {"facade": "planner.multi_route", "start": ids[0], "method": method}
        try:
            payload = RoutePlanner().multi_route(ids[0], ids[1:5], method=method)
            facade_row["result"] = {
                "success": payload["result"]["success"],
                "visiting_order": list(payload["result"]["visiting_order"]),
                "cost": round4(payload["result"]["total_cost"]),
                "dist_km": round4(payload["result"]["total_distance_km"]),
                "time_min": round4(payload["result"]["total_time_min"]),
            }
            facade_row["explanation_headline"] = payload["explanation"]["headline"]
        except Exception as error:
            facade_row["error"] = f"{type(error).__name__}: {error}"
        rows.append(facade_row)

    return rows


def run_unit_tests(network):
    """Unit-level checks for the remaining backend functions."""
    rows = []
    ids = [landmark.id for landmark in network.landmarks.values()]
    s_node = network.landmarks[ids[0]].snapped_node
    g_node = network.landmarks[ids[1]].snapped_node
    graph_normal = apply_traffic_profile(network.graph, "normal")
    cost_fn = WeightedCostFunction.from_dict(COST_PRESETS["balanced"])
    heuristic = HaversineHeuristic(network.coordinates)

    def check(name, fn, expected_success=True):
        row = {"function": name, "expected": "pass" if expected_success else "raise"}
        try:
            outcome = fn()
            row["outcome"] = "pass" if expected_success else "fail_no_raise"
            row["detail"] = outcome if isinstance(outcome, str) else json.dumps(outcome, default=str)
        except Exception as error:
            row["outcome"] = "pass" if not expected_success else "fail"
            row["detail"] = f"{type(error).__name__}: {error}"
        rows.append(row)

    check("Edge.from_dict(valid)", lambda: Edge.from_dict({
        "to": "n2", "distance_km": 1.0, "time_min": 2.0,
        "congestion": 3, "risk": 1, "road_type": "primary",
    }).to_dict())
    check("Edge.from_dict(missing field)", lambda: Edge.from_dict({
        "to": "n2", "distance_km": 1.0,
    }), expected_success=False)
    check("Edge(congestion out of range)", lambda: Edge(
        to="n2", distance_km=1.0, time_min=1.0, congestion=6, risk=0, road_type="x"
    ), expected_success=False)
    check("Edge(negative distance)", lambda: Edge(
        to="n2", distance_km=-1.0, time_min=1.0, congestion=1, risk=0, road_type="x"
    ), expected_success=False)
    check("graph_from_dict(valid)", lambda: graph_from_dict({
        "n1": [{"to": "n2", "distance_km": 1.0, "time_min": 1.0, "congestion": 1, "risk": 0, "road_type": "x"}],
        "n2": [],
    }))
    check("graph_from_dict(unknown target)", lambda: graph_from_dict({
        "n1": [{"to": "ghost", "distance_km": 1.0, "time_min": 1.0, "congestion": 1, "risk": 0, "road_type": "x"}],
    }), expected_success=False)
    check("graph_from_dict(empty)", lambda: graph_from_dict({}))
    check("coerce_graph(real graph)", lambda: len(coerce_graph(graph_normal)))
    check("load_graph_json(mock graph)", lambda: len(load_graph_json(BACKEND_ROOT / "data" / "mock_graph.json")))
    check("coordinates_from_data(valid)", lambda: len(coordinates_from_data({
        "nodes": {"a": {"latitude": 10.7, "longitude": 106.6}},
    })))
    check("coordinates_from_data(missing lon)", lambda: coordinates_from_data({
        "nodes": {"a": {"latitude": 10.7}},
    }), expected_success=False)
    check("coordinates_from_data(bad lat)", lambda: coordinates_from_data({
        "nodes": {"a": {"latitude": 95.0, "longitude": 106.6}},
    }), expected_success=False)

    for preset_name, preset in COST_PRESETS.items():
        check(f"WeightedCostFunction.from_dict({preset_name})", lambda p=preset: WeightedCostFunction.from_dict(p).to_dict())
    check("WeightedCostFunction(call on edge)", lambda: round4(WeightedCostFunction.from_dict(
        COST_PRESETS["balanced"]
    )(s_node, next(iter(graph_normal[s_node])))))
    check("WeightedCostFunction(negative alpha)", lambda: WeightedCostFunction(alpha_distance=-1.0), expected_success=False)
    check("WeightedCostFunction(alpha=bool)", lambda: WeightedCostFunction(alpha_distance=True), expected_success=False)
    check("WeightedCostFunction(alpha=NaN)", lambda: WeightedCostFunction(alpha_distance=math.nan), expected_success=False)
    check("WeightedCostFunction.from_json", lambda: WeightedCostFunction.from_json(BACKEND_ROOT / "data" / "mock_config.json").to_dict())
    check("WeightedCostFunction.from_dict(empty)", lambda: WeightedCostFunction.from_dict({}).to_dict())

    check("zero_heuristic", lambda: zero_heuristic(s_node, g_node))
    check("HaversineHeuristic(valid pairs)", lambda: (
        f"nodes={len(network.coordinates)} min={min(heuristic(a, b) for a in network.coordinates for b in network.coordinates if a != b):.3f} km"
    ))
    check("HaversineHeuristic(invalid lat)", lambda: HaversineHeuristic({"n": (91.0, 0.0)}), expected_success=False)
    check("HaversineHeuristic(missing node)", lambda: HaversineHeuristic(network.coordinates)("ghost", "n1"), expected_success=False)

    dijkstra_result = dijkstra(graph_normal, s_node, g_node, cost_fn, capture_trace=False)
    check("route_segments(success)", lambda: len(route_segments(graph_normal, dijkstra_result.path, cost_fn)))
    check("explain_search_result(success)", lambda: explain_search_result(
        dijkstra_result, graph_normal, cost_fn,
        start_name=ids[0], goal_name=ids[1], criterion="balanced",
        traffic_profile="normal",
        start_access_m=0.0, goal_access_m=0.0,
    )["headline"])
    check("summarize_comparison(all success)", lambda: summarize_comparison([dijkstra_result])["headline"])
    check("summarize_comparison(empty)", lambda: summarize_comparison([])["headline"])

    check("apply_traffic_profile(normal)", lambda: len(apply_traffic_profile(network.graph, "normal")))
    check("apply_traffic_profile(rush_hour)", lambda: len(apply_traffic_profile(network.graph, "rush_hour")))
    check("apply_traffic_profile(rainy)", lambda: len(apply_traffic_profile(network.graph, "rainy")))
    check("apply_traffic_profile(unknown)", lambda: apply_traffic_profile(network.graph, "ice_storm"), expected_success=False)
    check("TRAFFIC_PROFILES", lambda: sorted(TRAFFIC_PROFILES))
    check("load_traffic_network()", lambda: f"nodes={len(network.graph)} edges={sum(len(e) for e in network.graph.values())} landmarks={len(network.landmarks)}")
    check("MAX_LANDMARK_ROAD_OFFSET_M", lambda: MAX_LANDMARK_ROAD_OFFSET_M)

    search_step = SearchStep(index=0, event="expand", current_node=s_node)
    check("SearchStep.to_dict()", lambda: search_step.to_dict())
    check("SearchResult round-trip via save_result_json", lambda: str(save_result_json(
        dijkstra_result, Path("/tmp/opencode/lab1_unit_result.json")
    )))

    check("optimize_multi_location(unknown waypoint)", lambda: optimize_multi_location(
        graph_normal, s_node, ["ghost"], cost_fn
    ), expected_success=False)
    check("optimize_multi_location(str waypoints)", lambda: optimize_multi_location(
        graph_normal, s_node, "n1", cost_fn
    ), expected_success=False)
    check("optimize_multi_location(return_to_start+end)", lambda: optimize_multi_location(
        graph_normal, s_node, [g_node], cost_fn,
        end=g_node, return_to_start=True,
    ), expected_success=False)
    check("optimize_multi_location(unknown method)", lambda: optimize_multi_location(
        graph_normal, s_node, [g_node], cost_fn, method="random"
    ), expected_success=False)
    check("optimize_multi_location(exact too large)", lambda: optimize_multi_location(
        graph_normal, s_node, [network.landmarks[i].snapped_node for i in ids[1:10]], cost_fn,
        method="exact_bruteforce",
    ), expected_success=False)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/opencode/lab1_matrix.json")
    args = parser.parse_args()

    started_at = time.perf_counter()
    print("Loading traffic network ...")
    network = load_traffic_network()
    planner = RoutePlanner()

    print("Running 552-pair x 6-algorithm matrix ...")
    pairs, failures = run_matrix(network, planner)
    print("Running dijkstra_all ...")
    dijkstra_rows = run_dijkstra_all(network)
    print("Running multi-location tests ...")
    multi_rows = run_multi_location(network)
    print("Running unit tests ...")
    unit_rows = run_unit_tests(network)

    unit_failures = [
        row for row in unit_rows
        if (row["expected"] == "raise" and row["outcome"] != "pass")
        or (row["expected"] == "pass" and row["outcome"] != "pass")
    ]
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pair_count": len(pairs),
        "algorithm_names": sorted(RAW_ALGORITHMS),
        "landmark_ids": [landmark.id for landmark in network.landmarks.values()],
        "network": {
            "nodes": len(network.graph),
            "edges": sum(len(edges) for edges in network.graph.values()),
            "landmarks": len(network.landmarks),
        },
        "pairs": pairs,
        "dijkstra_all": dijkstra_rows,
        "multi_location": multi_rows,
        "unit_tests": unit_rows,
        "failures": failures,
        "unit_failures": unit_failures,
        "total_runtime_s": round(time.perf_counter() - started_at, 1),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Failures: {len(failures)} matrix + {len(unit_failures)} unit")
    for failure in failures[:20] + unit_failures[:20]:
        print("  -", failure)


if __name__ == "__main__":
    main()
