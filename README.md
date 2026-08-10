# Saigon Route Lab Backend

FastAPI backend and pure-Python search engine for a multi-landmark tourist
route planner in Ho Chi Minh City.

## Included

- OSM-derived directed road graph: 481 routable nodes and 995 directed edges.
- 24 landmarks snapped to the road graph.
- BFS, DFS, Uniform Cost Search, A*, Dijkstra, and Greedy Best-First Search.
- Weighted distance, time, congestion, and risk cost model.
- Normal, rush-hour, and rainy traffic profiles.
- Nearest-neighbor and exact brute-force multi-location routing.
- REST endpoints, WebSocket search animation, GeoJSON output, explanations,
  and algorithm comparison summaries.

## Run locally

Python 3.10 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API documentation.

## Verify

```powershell
python -m unittest discover -s tests -v
python examples\run_kiet_demo.py
python examples\run_full_audit.py
```

The test suite covers contracts, graph loading, all search algorithms,
multi-location behavior, traffic profiles, REST routes, validation errors, and
WebSocket streaming. See [docs/KIET_MODULE_README.md](docs/KIET_MODULE_README.md)
for the original Kiet module contract and integration notes.
