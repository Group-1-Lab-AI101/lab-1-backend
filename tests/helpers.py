"""Small deterministic graph fixtures used by multiple test modules."""

from __future__ import annotations

from core.contracts import Edge


def edge(
    target: str,
    cost: float,
    *,
    distance: float | None = None,
    time: float | None = None,
) -> Edge:
    """Create a valid edge whose test cost is stored in metadata."""
    return Edge(
        to=target,
        distance_km=cost if distance is None else distance,
        time_min=cost if time is None else time,
        congestion=1,
        risk=0,
        road_type="test",
        metadata={"cost": cost},
    )


def metadata_cost(_source: str, road: Edge) -> float:
    """Read the synthetic test cost attached to an edge."""
    return float(road.metadata["cost"])


def comparison_graph() -> dict[str, list[Edge]]:
    """Return a graph where nearest-neighbor is worse than exact ordering."""
    return {
        "S": [edge("A", 2), edge("B", 2), edge("C", 3)],
        "A": [edge("B", 2), edge("C", 2)],
        "B": [edge("A", 2), edge("C", 100)],
        "C": [edge("A", 2), edge("B", 100)],
    }
