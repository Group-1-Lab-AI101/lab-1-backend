"""Validated request models for route-planning API endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


AlgorithmName = Literal["bfs", "dfs", "ucs", "astar", "dijkstra", "greedy"]
CriterionName = Literal[
    "balanced", "fastest", "shortest", "low_congestion", "low_risk"
]
TrafficProfileName = Literal["normal", "rush_hour", "rainy"]
MultiMethodName = Literal["nearest_neighbor", "exact_bruteforce"]


class CostWeights(BaseModel):
    """Optional non-negative custom cost coefficients."""

    model_config = ConfigDict(extra="forbid")

    alpha_distance: float = Field(ge=0)
    beta_time: float = Field(ge=0)
    gamma_congestion: float = Field(ge=0)
    delta_risk: float = Field(ge=0)


class SearchRequest(BaseModel):
    """Request for one two-location search run."""

    model_config = ConfigDict(extra="forbid")

    start: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    algorithm: AlgorithmName = "dijkstra"
    criterion: CriterionName = "balanced"
    traffic_profile: TrafficProfileName = "normal"
    custom_weights: CostWeights | None = None
    capture_trace: bool = True


class CompareRequest(BaseModel):
    """Request to compare every supported two-location algorithm."""

    model_config = ConfigDict(extra="forbid")

    start: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    criterion: CriterionName = "balanced"
    traffic_profile: TrafficProfileName = "normal"
    custom_weights: CostWeights | None = None


class MultiRouteRequest(BaseModel):
    """Request for ordered multi-landmark route optimization."""

    model_config = ConfigDict(extra="forbid")

    start: str = Field(min_length=1)
    waypoints: list[str] = Field(min_length=1)
    method: MultiMethodName = "nearest_neighbor"
    end: str | None = None
    return_to_start: bool = False
    criterion: CriterionName = "balanced"
    traffic_profile: TrafficProfileName = "normal"
    custom_weights: CostWeights | None = None
    exact_limit: int = Field(default=8, ge=1, le=9)
    compare_methods: bool = False
