"""Shared contracts and route-optimization utilities."""

from core.contracts import (
    Edge,
    Graph,
    MultiLocationResult,
    RouteSegment,
    SearchResult,
    SearchStep,
    coordinates_from_data,
    graph_from_dict,
    load_graph_json,
)
from core.cost import WeightedCostFunction
from core.heuristic import HaversineHeuristic, zero_heuristic
from core.serialization import save_result_json

__all__ = [
    "Edge",
    "Graph",
    "HaversineHeuristic",
    "MultiLocationResult",
    "RouteSegment",
    "SearchResult",
    "SearchStep",
    "WeightedCostFunction",
    "coordinates_from_data",
    "graph_from_dict",
    "load_graph_json",
    "save_result_json",
    "zero_heuristic",
]
