"""Configurable edge-cost functions for Vietnamese traffic routing."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from core.contracts import Edge


@dataclass(frozen=True)
class WeightedCostFunction:
    """Compute a non-negative weighted traffic cost for a road segment."""

    alpha_distance: float = 1.0
    beta_time: float = 1.0
    gamma_congestion: float = 1.0
    delta_risk: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "alpha_distance",
            "beta_time",
            "gamma_congestion",
            "delta_risk",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, float(value))

    def __call__(self, from_node: str, edge: Edge) -> float:
        """Return weighted cost; ``from_node`` is available for custom adapters."""
        del from_node
        cost = (
            self.alpha_distance * edge.distance_km
            + self.beta_time * edge.time_min
            + self.gamma_congestion * edge.congestion
            + self.delta_risk * edge.risk
        )
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("Weighted edge cost must be finite and non-negative")
        return cost

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WeightedCostFunction":
        """Create weights from a mapping or its nested ``cost_weights`` field."""
        raw_weights = data.get("cost_weights", data)
        if not isinstance(raw_weights, Mapping):
            raise TypeError("Cost weights must be a mapping")
        return cls(
            alpha_distance=raw_weights.get("alpha_distance", 1.0),
            beta_time=raw_weights.get("beta_time", 1.0),
            gamma_congestion=raw_weights.get("gamma_congestion", 1.0),
            delta_risk=raw_weights.get("delta_risk", 1.0),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "WeightedCostFunction":
        """Load a UTF-8 JSON config and create a weighted cost function."""
        with Path(path).open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, float]:
        """Return JSON-serializable weight configuration."""
        return {
            "alpha_distance": self.alpha_distance,
            "beta_time": self.beta_time,
            "gamma_congestion": self.gamma_congestion,
            "delta_risk": self.delta_risk,
        }
