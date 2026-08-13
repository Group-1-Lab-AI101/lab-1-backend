# Thai Kiet Search Module

## 1. Scope

This module implements only Thai Kiet's assigned work:

- Dijkstra's Algorithm.
- Greedy Best-First Search.
- Multi-location route optimization with nearest-neighbor and exact brute force.

It does not implement BFS, DFS, UCS, A*, route explanation, or GUI logic. The
public contracts and `on_step` callback are intentionally small so those parts
can integrate later without changing the algorithms.

The implementation uses Python 3.10+ and the standard library only. NetworkX is
not used to perform any search.

## 2. Project Structure

```text
algorithms/
  _shared.py
  dijkstra.py
  greedy_best_first.py
core/
  contracts.py
  cost.py
  heuristic.py
  multi_location.py
  serialization.py
data/
  mock_graph.json
  mock_config.json
tests/
  test_dijkstra.py
  test_greedy_best_first.py
  test_multi_location.py
  test_serialization.py
examples/
  run_kiet_demo.py
```

## 3. Setup and Commands

No third-party package is required.

```powershell
cd C:\Users\Admin\Documents\AI01\lab-1-backend
python -m unittest discover -s tests -v
python examples\run_kiet_demo.py
```

The demo writes JSON output under `output/demo/`.

## 4. Graph Input Contract

Algorithms receive a directed adjacency mapping:

```python
Graph = Mapping[str, Sequence[Edge]]
```

Each node ID is a stable ASCII string. A display name belongs in node metadata,
not in the algorithm key. Each `Edge` contains:

| Field | Meaning |
| --- | --- |
| `to` | Existing destination node ID |
| `distance_km` | Non-negative physical distance |
| `time_min` | Non-negative estimated travel time |
| `congestion` | Traffic level from 1 to 5 |
| `risk` | Risk penalty from 0 to 5 |
| `road_type` | Road classification |
| `metadata` | Optional JSON-compatible attributes |

The loader does not infer reverse edges. If JSON contains only `A -> B`, the
road is directed. The data owner must add `B -> A` when a road is two-way.

`graph_from_dict()` accepts either a raw adjacency object or a document with an
`adjacency` field. `load_graph_json()` reads the same schema from UTF-8 JSON.
Both validate every edge target and return a snapshot. Algorithms never mutate
the caller's graph.

## 5. Cost and Heuristic Contracts

The cost function has this contract:

```python
CostFunction = Callable[[str, Edge], float]
```

It must return a finite, non-negative value. `WeightedCostFunction` implements:

```text
cost = alpha_distance * distance_km
     + beta_time * time_min
     + gamma_congestion * congestion
     + delta_risk * risk
```

Weights are configurable through `mock_config.json`; they are not embedded in
Dijkstra or Greedy.

The heuristic contract is:

```python
HeuristicFunction = Callable[[str, str], float]
```

It must also return a finite, non-negative value. `zero_heuristic` always
returns 0. `HaversineHeuristic` uses node latitude/longitude metadata and
returns straight-line distance in kilometers. Greedy receives the heuristic
from its caller.

For A* integration, the group must decide whether the final heuristic matches
the cost units and whether it is admissible and consistent. Greedy does not
require admissibility, but heuristic quality strongly affects its route.

## 6. Dijkstra API

```python
from algorithms.dijkstra import dijkstra
from core.contracts import load_graph_json
from core.cost import WeightedCostFunction

graph = load_graph_json("data/mock_graph.json")
cost_fn = WeightedCostFunction.from_json("data/mock_config.json")

result = dijkstra(
    graph,
    "ben_thanh_market",
    "saigon_zoo",
    cost_fn,
    capture_trace=True,
)
```

Dijkstra uses a `heapq` priority queue and an insertion counter for
deterministic ties. It finalizes a goal only when that goal is popped with its
best known cost. Stale heap entries are skipped. Parent edges are retained so
cost, distance, and time are all calculated from the final path.

`dijkstra_all(graph, start, cost_fn)` runs one single-source search and returns
a `SearchResult` for every node. Multi-location routing uses this API to avoid
performing one complete search for every individual target pair.

### Dijkstra Guarantees and Complexity

With finite non-negative edge costs:

- It is complete for a reachable goal in a finite graph.
- It returns a minimum-cost route.
- Time complexity is `O((V + E) log V)` with a binary heap.
- Space complexity is `O(V + E)` including the input and search state.

If a requested goal is unreachable, the function returns `success=False`,
`path=[]`, and `total_cost=None`. Invalid nodes and negative/non-finite costs
raise a clear exception.

### Dijkstra versus UCS

Both algorithms can use the same priority rule for a single goal. The module
keeps Dijkstra separate from the team's UCS implementation because its public
role is different:

- UCS is usually presented as goal-directed state-space search.
- This Dijkstra module also provides reusable single-source shortest paths.
- Multi-location optimization caches those single-source results to build a
  reduced pairwise routing problem.

No UCS source file is copied or reimplemented here.

## 7. Greedy Best-First API

```python
from algorithms.greedy_best_first import greedy_best_first
from core.contracts import coordinates_from_data
from core.heuristic import HaversineHeuristic

heuristic_fn = HaversineHeuristic(coordinates)
result = greedy_best_first(
    graph,
    "ben_thanh_market",
    "saigon_zoo",
    heuristic_fn,
    cost_fn,
)
```

Greedy priority is only `h(n)`. It does not add `g(n)` to the priority. The
algorithm still stores the selected parent edges and uses `cost_fn` to measure
the returned route.

On a finite graph, the closed set prevents cycles from causing an infinite
loop. Worst-case time and space are `O(V + E)` plus heap operations, while
actual behavior depends heavily on the heuristic. A successful result has
`optimality="not_guaranteed"`.

Example where Greedy is more expensive than Dijkstra:

```text
A -> B cost 1, B -> G cost 100, h(B)=1
A -> C cost 2, C -> G cost   2, h(C)=2
```

Greedy expands B and then G, returning cost 101. Dijkstra returns `A -> C -> G`
with cost 4. This behavior is tested automatically.

## 8. Multi-location API

```python
from core.multi_location import optimize_multi_location

result = optimize_multi_location(
    graph,
    "ben_thanh_market",
    ["independence_palace", "central_post_office", "saigon_zoo"],
    cost_fn,
    method="nearest_neighbor",
    end="bach_dang_wharf",
)
```

Duplicate waypoints and occurrences of `start` are removed stably. A fixed
`end` is visited after all free-order waypoints. `return_to_start=True` adds a
final segment back to start and cannot be combined with a different fixed end.

If a shortest-path segment crosses another requested waypoint, that waypoint
is recorded at its real first position in `visiting_order` and is not scheduled
again. Segment boundaries are joined once in `full_path`.

### Nearest Neighbor

At each step, the method chooses the unvisited waypoint with the smallest
Dijkstra cost from the current node. Input waypoint order breaks equal-cost
ties. It is fast and understandable, but approximate:

```text
optimality = "approximate_not_guaranteed"
```

Its main advantage is practical scaling. Its main disadvantage is that a
locally cheap next stop can produce an expensive later segment.

### Exact Brute Force

`method="exact_bruteforce"` enumerates waypoint permutations and evaluates
them using cached pairwise Dijkstra routes. It is exact only for this reduced
pairwise problem under the supplied graph and cost function:

```text
optimality = "optimal_for_reduced_pairwise_problem"
```

It is not described as a universally optimal physical TSP solver. Runtime is
factorial in waypoint count, so `exact_limit` defaults to 8. Larger requests
raise an exception that recommends nearest-neighbor.

### Method Comparison

```python
from core.multi_location import compare_multi_location_methods

comparison = compare_multi_location_methods(
    graph, start, waypoints, cost_fn, exact_limit=8
)
nearest = comparison["nearest_neighbor"]
exact = comparison["exact_bruteforce"]
```

For a non-zero exact cost, the nearest result reports:

```text
gap_percent = (nearest_cost - exact_cost) / exact_cost * 100
```

If both costs are zero, the gap is 0. If exact cost is zero while nearest cost
is positive, the percentage is undefined and stored as `None`.

## 9. Result Fields

`SearchResult` contains:

| Field | Meaning |
| --- | --- |
| `algorithm` | Stable algorithm name |
| `success` | Whether a route was found |
| `start`, `goal` | Requested endpoints |
| `path` | Complete node-by-node path |
| `total_cost` | Weighted cost of final path |
| `total_distance_km` | Physical distance of final path |
| `total_time_min` | Estimated time of final path |
| `visited_order` | Finalized or expanded node order |
| `expanded_nodes` | Number of expanded nodes |
| `generated_nodes` | Number of heap insertions |
| `runtime_ms` | Runtime measured by `perf_counter()` |
| `trace` | GUI-ready `SearchStep` events |
| `message` | Human-readable status |
| `optimality` | Correct guarantee label |

`MultiLocationResult` adds the normalized requested waypoints, actual visiting
order, full joined path, route segments, aggregate metrics, method label, and
optional comparison gap.

Every result, segment, and trace step has `to_dict()`. `save_result_json()`
writes UTF-8 JSON with nested objects fully converted.

## 10. GUI Integration

Both two-location algorithms use the same callback contract:

```python
def update_animation(step):
    current = step.current_node
    frontier = step.frontier
    visited = step.visited
    event = step.event

result = dijkstra(
    graph,
    start,
    goal,
    cost_fn,
    capture_trace=False,
    on_step=update_animation,
)
```

The callback is synchronous. GUI code should place events onto its own UI
queue when the search runs in a worker thread. Callback exceptions propagate
to the caller instead of being silently swallowed. Algorithm modules do not
import Tkinter, Matplotlib, FastAPI, or frontend code.

## 11. Demo Data

`mock_graph.json` has 10 nodes, directed roads, competing routes, congestion
differences, geographic coordinates, and one unreachable test node. It is only
an integration fixture.

The course specification requires the group's final simulated dataset to have
at least 20 nodes and 30 edges, or to use a clearly described simplified real
or hybrid dataset. The mock file must not be presented as that final dataset.

## 12. Current Limits and Group Decisions Still Needed

- Final graph schema must be aligned with the graph/dataset owner's output.
- Final cost weights require group justification and traffic experiments.
- Final heuristic and its admissibility/consistency claims require agreement.
- Runtime metrics are appropriate for comparison, but very small graphs may
  need repeated runs for stable report tables.
- Exact brute force is intentionally limited to small waypoint sets.
- Real-time traffic, turn restrictions, and map matching are outside this
  module.

Run the full suite after any shared-schema integration:

```powershell
python -m unittest discover -s tests -v
```
