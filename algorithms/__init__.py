"""Search algorithms used by the complete route-planning project."""

from algorithms.a_star import a_star_search
from algorithms.breadth_first import breadth_first_search
from algorithms.depth_first import depth_first_search
from algorithms.uniform_cost import uniform_cost_search

__all__ = [
    "a_star_search",
    "breadth_first_search",
    "depth_first_search",
    "uniform_cost_search",
]
