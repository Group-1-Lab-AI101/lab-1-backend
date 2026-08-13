"""External heuristic functions for Greedy Best-First Search."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


def zero_heuristic(node: str, goal: str) -> float:
    """Return zero for every node-goal pair."""
    del node, goal
    return 0.0


@dataclass(frozen=True)
class HaversineHeuristic:
    """Callable straight-line distance heuristic measured in kilometers."""

    coordinates: Mapping[str, tuple[float, float]]

    def __post_init__(self) -> None:
        normalized: dict[str, tuple[float, float]] = {}
        for node, pair in self.coordinates.items():
            if (
                not isinstance(node, str)
                or len(pair) != 2
                or not all(isinstance(value, (int, float)) for value in pair)
            ):
                raise TypeError("Coordinates must map node IDs to (lat, lon) pairs")
            latitude, longitude = float(pair[0]), float(pair[1])
            if not math.isfinite(latitude) or not -90 <= latitude <= 90:
                raise ValueError(f"Invalid latitude for node {node!r}")
            if not math.isfinite(longitude) or not -180 <= longitude <= 180:
                raise ValueError(f"Invalid longitude for node {node!r}")
            normalized[node] = (latitude, longitude)
        object.__setattr__(self, "coordinates", normalized)

    def __call__(self, node: str, goal: str) -> float:
        """Return great-circle distance from ``node`` to ``goal`` in kilometers."""
        try:
            latitude_1, longitude_1 = self.coordinates[node]
            latitude_2, longitude_2 = self.coordinates[goal]
        except KeyError as error:
            raise KeyError(f"Missing coordinates for node {error.args[0]!r}") from error

        lat_1 = math.radians(latitude_1)
        lat_2 = math.radians(latitude_2)
        delta_lat = lat_2 - lat_1
        delta_lon = math.radians(longitude_2 - longitude_1)
        haversine = (
            math.sin(delta_lat / 2) ** 2
            + math.cos(lat_1) * math.cos(lat_2) * math.sin(delta_lon / 2) ** 2
        )
        central_angle = 2 * math.atan2(
            math.sqrt(haversine), math.sqrt(max(0.0, 1 - haversine))
        )
        return 6371.0088 * central_angle
