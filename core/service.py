"""Application service joining datasets, algorithms, and API-friendly output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from algorithms import (
    a_star_search,
    breadth_first_search,
    depth_first_search,
    dijkstra,
    greedy_best_first,
    uniform_cost_search,
)
from core.contracts import Edge, SearchResult, SearchStep, StepCallback
from core.cost import WeightedCostFunction
from core.explanation import (
    explain_search_result,
    route_segments,
    summarize_comparison,
)
from core.heuristic import HaversineHeuristic
from core.multi_location import (
    compare_multi_location_methods,
    optimize_multi_location,
)
from core.osm_loader import (
    TRAFFIC_PROFILES,
    Landmark,
    TrafficNetwork,
    apply_traffic_profile,
    load_traffic_network,
)


COST_PRESETS = {
    "balanced": {
        "alpha_distance": 1.0,
        "beta_time": 0.4,
        "gamma_congestion": 0.08,
        "delta_risk": 0.12,
    },
    "fastest": {
        "alpha_distance": 0.2,
        "beta_time": 1.2,
        "gamma_congestion": 0.04,
        "delta_risk": 0.08,
    },
    "shortest": {
        "alpha_distance": 1.5,
        "beta_time": 0.05,
        "gamma_congestion": 0.01,
        "delta_risk": 0.03,
    },
    "low_congestion": {
        "alpha_distance": 0.7,
        "beta_time": 0.3,
        "gamma_congestion": 0.25,
        "delta_risk": 0.15,
    },
    "low_risk": {
        "alpha_distance": 0.5,
        "beta_time": 0.3,
        "gamma_congestion": 0.08,
        "delta_risk": 0.8,
    },
}

ALGORITHM_METADATA = {
    "bfs": {
        "label": "Breadth-First Search",
        "description": "Fewest graph edges",
        "guarantee": "Optimal only by edge count",
    },
    "dfs": {
        "label": "Depth-First Search",
        "description": "First depth-first route",
        "guarantee": "No route-quality guarantee",
    },
    "ucs": {
        "label": "Uniform Cost Search",
        "description": "Lowest weighted traffic cost",
        "guarantee": "Optimal with non-negative costs",
    },
    "astar": {
        "label": "A* Search",
        "description": "Cost plus straight-line lower bound",
        "guarantee": "Optimal with admissible heuristic",
    },
    "dijkstra": {
        "label": "Dijkstra",
        "description": "Lowest weighted traffic cost",
        "guarantee": "Optimal with non-negative costs",
    },
    "greedy": {
        "label": "Greedy Best-First",
        "description": "Straight-line heuristic only",
        "guarantee": "No cost-optimality guarantee",
    },
}


@dataclass(frozen=True)
class SearchContext:
    """Validated graph, cost, heuristic, and landmark request state."""

    graph: dict[str, tuple[Edge, ...]]
    cost_fn: WeightedCostFunction
    heuristic: HaversineHeuristic
    start: Landmark
    goal: Landmark
    criterion: str
    traffic_profile: str


class RoutePlanner:
    """Facade used by HTTP routes, WebSockets, tests, and future GUI adapters."""

    def __init__(self, network: TrafficNetwork | None = None):
        self.network = network or load_traffic_network()

    def bootstrap(self) -> dict[str, Any]:
        """Return static configuration required to initialize the frontend."""
        return {
            "landmarks": [
                landmark.to_dict() for landmark in self.network.landmarks.values()
            ],
            "algorithms": ALGORITHM_METADATA,
            "cost_presets": COST_PRESETS,
            "traffic_profiles": TRAFFIC_PROFILES,
            "network_summary": self.network.summary,
            "node_coordinates": {
                node: [latitude, longitude]
                for node, (latitude, longitude) in self.network.coordinates.items()
            },
            "boundary": self.network.boundary_geojson,
        }

    def roads(self) -> dict[str, Any]:
        """Return routable road geometry for the map background."""
        return self.network.roads_geojson()

    def _landmark(self, landmark_id: str) -> Landmark:
        try:
            return self.network.landmarks[landmark_id]
        except KeyError as error:
            raise ValueError(f"Unknown landmark ID: {landmark_id!r}") from error

    def _weights(
        self,
        criterion: str,
        custom_weights: Mapping[str, float] | None,
    ) -> tuple[str, WeightedCostFunction]:
        if custom_weights is not None:
            return "custom", WeightedCostFunction.from_dict(custom_weights)
        if criterion not in COST_PRESETS:
            supported = ", ".join(COST_PRESETS)
            raise ValueError(f"Unknown criterion {criterion!r}; expected: {supported}")
        return criterion, WeightedCostFunction.from_dict(COST_PRESETS[criterion])

    def _context(
        self,
        start_id: str,
        goal_id: str,
        criterion: str,
        traffic_profile: str,
        custom_weights: Mapping[str, float] | None,
    ) -> SearchContext:
        start = self._landmark(start_id)
        goal = self._landmark(goal_id)
        actual_criterion, cost_fn = self._weights(criterion, custom_weights)
        graph = apply_traffic_profile(self.network.graph, traffic_profile)
        heuristic = HaversineHeuristic(self.network.coordinates)
        return SearchContext(
            graph=graph,
            cost_fn=cost_fn,
            heuristic=heuristic,
            start=start,
            goal=goal,
            criterion=actual_criterion,
            traffic_profile=traffic_profile,
        )

    def _run_algorithm(
        self,
        algorithm: str,
        context: SearchContext,
        capture_trace: bool,
        on_step: StepCallback | None,
    ) -> SearchResult:
        start_node = context.start.snapped_node
        goal_node = context.goal.snapped_node
        common = {
            "capture_trace": capture_trace,
            "on_step": on_step,
        }
        if algorithm == "bfs":
            return breadth_first_search(
                context.graph, start_node, goal_node, context.cost_fn, **common
            )
        if algorithm == "dfs":
            return depth_first_search(
                context.graph, start_node, goal_node, context.cost_fn, **common
            )
        if algorithm == "ucs":
            return uniform_cost_search(
                context.graph, start_node, goal_node, context.cost_fn, **common
            )
        if algorithm == "dijkstra":
            return dijkstra(
                context.graph, start_node, goal_node, context.cost_fn, **common
            )
        if algorithm == "greedy":
            return greedy_best_first(
                context.graph,
                start_node,
                goal_node,
                context.heuristic,
                context.cost_fn,
                **common,
            )
        if algorithm == "astar":
            alpha = context.cost_fn.alpha_distance

            def admissible_heuristic(node: str, goal: str) -> float:
                return alpha * context.heuristic(node, goal)

            return a_star_search(
                context.graph,
                start_node,
                goal_node,
                admissible_heuristic,
                context.cost_fn,
                **common,
            )
        supported = ", ".join(ALGORITHM_METADATA)
        raise ValueError(f"Unknown algorithm {algorithm!r}; expected: {supported}")

    def search(
        self,
        start_id: str,
        goal_id: str,
        algorithm: str,
        *,
        criterion: str = "balanced",
        traffic_profile: str = "normal",
        custom_weights: Mapping[str, float] | None = None,
        capture_trace: bool = True,
        on_step: Callable[[SearchStep], None] | None = None,
    ) -> dict[str, Any]:
        """Run one algorithm and return route geometry, metrics, and explanation."""
        context = self._context(
            start_id, goal_id, criterion, traffic_profile, custom_weights
        )
        result = self._run_algorithm(
            algorithm, context, capture_trace, on_step
        )
        alternative = self._find_alternative(algorithm, result, context)
        return self._search_payload(
            result,
            algorithm,
            context,
            alternative=alternative,
        )

    def _find_alternative(
        self,
        algorithm: str,
        result: SearchResult,
        context: SearchContext,
    ) -> tuple[str, SearchResult] | None:
        """Return a successful distinct route for human-readable comparison."""
        if not result.success:
            return None
        if algorithm in {"dijkstra", "ucs", "astar"}:
            candidates = ("bfs", "greedy", "dfs")
        else:
            candidates = ("dijkstra", "astar", "bfs", "greedy", "dfs")
        same_path: tuple[str, SearchResult] | None = None
        for candidate in candidates:
            if candidate == algorithm:
                continue
            candidate_result = self._run_algorithm(
                candidate, context, False, None
            )
            if not candidate_result.success:
                continue
            if same_path is None:
                same_path = (candidate, candidate_result)
            if candidate_result.path != result.path:
                return candidate, candidate_result
        return same_path

    def _search_payload(
        self,
        result: SearchResult,
        algorithm: str,
        context: SearchContext,
        *,
        alternative: tuple[str, SearchResult] | None = None,
    ) -> dict[str, Any]:
        segments = (
            route_segments(context.graph, result.path, context.cost_fn)
            if result.success
            else []
        )
        alternative_name, alternative_result = (
            alternative if alternative is not None else (None, None)
        )
        return {
            "request": {
                "start": context.start.to_dict(),
                "goal": context.goal.to_dict(),
                "algorithm": algorithm,
                "criterion": context.criterion,
                "traffic_profile": context.traffic_profile,
                "weights": context.cost_fn.to_dict(),
            },
            "result": result.to_dict(),
            "route_geojson": self.network.route_geojson(result.path),
            "route_segments": segments,
            "alternative": (
                {
                    "algorithm": alternative_name,
                    "result": alternative_result.to_dict(),
                    "route_geojson": self.network.route_geojson(
                        alternative_result.path
                    ),
                }
                if alternative_result is not None
                else None
            ),
            "explanation": explain_search_result(
                result,
                context.graph,
                context.cost_fn,
                start_name=context.start.name,
                goal_name=context.goal.name,
                criterion=context.criterion,
                traffic_profile=context.traffic_profile,
                alternative_result=alternative_result,
                start_access_m=context.start.snapped_distance_m,
                goal_access_m=context.goal.snapped_distance_m,
            ),
        }

    def compare(
        self,
        start_id: str,
        goal_id: str,
        *,
        criterion: str = "balanced",
        traffic_profile: str = "normal",
        custom_weights: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        """Run all six algorithms under one identical routing scenario."""
        context = self._context(
            start_id, goal_id, criterion, traffic_profile, custom_weights
        )
        payloads = [
            self._search_payload(
                self._run_algorithm(name, context, False, None), name, context
            )
            for name in ALGORITHM_METADATA
        ]
        results = [
            SearchResult(
                algorithm=payload["result"]["algorithm"],
                success=payload["result"]["success"],
                start=payload["result"]["start"],
                goal=payload["result"]["goal"],
                path=payload["result"]["path"],
                total_cost=payload["result"]["total_cost"],
                total_distance_km=payload["result"]["total_distance_km"],
                total_time_min=payload["result"]["total_time_min"],
                visited_order=payload["result"]["visited_order"],
                expanded_nodes=payload["result"]["expanded_nodes"],
                generated_nodes=payload["result"]["generated_nodes"],
                runtime_ms=payload["result"]["runtime_ms"],
                trace=[],
                message=payload["result"]["message"],
                optimality=payload["result"]["optimality"],
            )
            for payload in payloads
        ]
        return {
            "request": payloads[0]["request"],
            "algorithms": payloads,
            "summary": summarize_comparison(results),
        }

    def multi_route(
        self,
        start_id: str,
        waypoint_ids: Sequence[str],
        *,
        method: str = "nearest_neighbor",
        end_id: str | None = None,
        return_to_start: bool = False,
        criterion: str = "balanced",
        traffic_profile: str = "normal",
        custom_weights: Mapping[str, float] | None = None,
        exact_limit: int = 8,
        compare_methods: bool = False,
    ) -> dict[str, Any]:
        """Optimize a multi-landmark visit and return map-ready output."""
        start = self._landmark(start_id)
        waypoints = [self._landmark(item) for item in waypoint_ids]
        end = self._landmark(end_id) if end_id is not None else None
        actual_criterion, cost_fn = self._weights(criterion, custom_weights)
        graph = apply_traffic_profile(self.network.graph, traffic_profile)
        waypoint_nodes = [item.snapped_node for item in waypoints]
        common = {
            "end": end.snapped_node if end else None,
            "return_to_start": return_to_start,
            "exact_limit": exact_limit,
        }
        if compare_methods:
            comparison = compare_multi_location_methods(
                graph, start.snapped_node, waypoint_nodes, cost_fn, **common
            )
            result = (
                comparison[method]
                if method in comparison
                else optimize_multi_location(
                    graph,
                    start.snapped_node,
                    waypoint_nodes,
                    cost_fn,
                    method=method,
                    **common,
                )
            )
            comparison_payload = {
                name: item.to_dict() for name, item in comparison.items()
            }
        else:
            result = optimize_multi_location(
                graph,
                start.snapped_node,
                waypoint_nodes,
                cost_fn,
                method=method,
                **common,
            )
            comparison_payload = None

        landmark_by_node = {
            landmark.snapped_node: landmark for landmark in waypoints
        }
        visiting_landmarks = [
            landmark_by_node[node].to_dict()
            for node in result.visiting_order
            if node in landmark_by_node
        ]
        segments = (
            route_segments(graph, result.full_path, cost_fn)
            if result.success
            else []
        )
        return {
            "request": {
                "start": start.to_dict(),
                "waypoints": [item.to_dict() for item in waypoints],
                "end": end.to_dict() if end else None,
                "method": method,
                "return_to_start": return_to_start,
                "criterion": actual_criterion,
                "traffic_profile": traffic_profile,
                "weights": cost_fn.to_dict(),
            },
            "result": result.to_dict(),
            "visiting_landmarks": visiting_landmarks,
            "route_geojson": self.network.route_geojson(result.full_path),
            "route_segments": segments,
            "comparison": comparison_payload,
            "explanation": {
                "headline": (
                    f"Visited {len(visiting_landmarks)} landmark(s) using {method}."
                    if result.success
                    else result.message
                ),
                "optimality_note": result.optimality,
                "duplicate_policy": "Duplicate waypoints are removed in input order.",
            },
        }
