"""FastAPI REST and WebSocket entry point for the complete lab project."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.websockets import WebSocketState

from app.models import CompareRequest, MultiRouteRequest, SearchRequest
from core.service import RoutePlanner


planner = RoutePlanner()
app = FastAPI(
    title="Saigon Route Lab API",
    description="Search algorithms for a multi-landmark Ho Chi Minh City route planner.",
    version="1.0.0",
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ValueError)
async def value_error_handler(_request, error: ValueError) -> JSONResponse:
    """Convert domain validation errors into consistent HTTP responses."""
    return JSONResponse(status_code=422, content={"detail": str(error)})


@app.get("/")
def root() -> dict[str, Any]:
    """Return API identity and documentation links."""
    return {
        "name": "Saigon Route Lab API",
        "status": "ready",
        "docs": "/docs",
        "health": "/api/health",
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Return health status plus loaded graph dimensions."""
    summary = planner.network.summary
    return {
        "status": "ok",
        "routable_nodes": summary["routable_nodes"],
        "routable_edges": summary["routable_edges"],
        "landmarks": summary["landmarks"],
    }


@app.get("/api/bootstrap")
def bootstrap() -> dict[str, Any]:
    """Return landmarks, controls, coordinates, boundary, and summary."""
    return planner.bootstrap()


@app.get("/api/network")
def network() -> dict[str, Any]:
    """Return routable roads as GeoJSON for Leaflet rendering."""
    return planner.roads()


@app.post("/api/search")
def search(request: SearchRequest) -> dict[str, Any]:
    """Run one algorithm for a start-goal landmark pair."""
    return planner.search(
        request.start,
        request.goal,
        request.algorithm,
        criterion=request.criterion,
        traffic_profile=request.traffic_profile,
        custom_weights=(
            request.custom_weights.model_dump() if request.custom_weights else None
        ),
        capture_trace=request.capture_trace,
    )


@app.post("/api/compare")
def compare(request: CompareRequest) -> dict[str, Any]:
    """Compare all six algorithms under one scenario."""
    return planner.compare(
        request.start,
        request.goal,
        criterion=request.criterion,
        traffic_profile=request.traffic_profile,
        custom_weights=(
            request.custom_weights.model_dump() if request.custom_weights else None
        ),
    )


@app.post("/api/multi-route")
def multi_route(request: MultiRouteRequest) -> dict[str, Any]:
    """Optimize the visiting order for several landmarks."""
    return planner.multi_route(
        request.start,
        request.waypoints,
        method=request.method,
        end_id=request.end,
        return_to_start=request.return_to_start,
        criterion=request.criterion,
        traffic_profile=request.traffic_profile,
        custom_weights=(
            request.custom_weights.model_dump() if request.custom_weights else None
        ),
        exact_limit=request.exact_limit,
        compare_methods=request.compare_methods,
    )


@app.websocket("/ws/search")
async def search_websocket(websocket: WebSocket) -> None:
    """Stream search steps followed by one complete route payload."""
    await websocket.accept()
    try:
        raw_request = await websocket.receive_json()
        request = SearchRequest.model_validate(raw_request)
        await websocket.send_json({"type": "started", "algorithm": request.algorithm})
        event_loop = asyncio.get_running_loop()
        step_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        def publish_step(step) -> None:
            event_loop.call_soon_threadsafe(
                step_queue.put_nowait,
                {"type": "step", "step": step.to_dict()},
            )

        worker = asyncio.create_task(
            asyncio.to_thread(
                planner.search,
                request.start,
                request.goal,
                request.algorithm,
                criterion=request.criterion,
                traffic_profile=request.traffic_profile,
                custom_weights=(
                    request.custom_weights.model_dump()
                    if request.custom_weights
                    else None
                ),
                capture_trace=False,
                on_step=publish_step,
            )
        )
        while not worker.done() or not step_queue.empty():
            try:
                event = await asyncio.wait_for(step_queue.get(), timeout=0.05)
                await websocket.send_json(event)
            except TimeoutError:
                continue
        payload = await worker
        await websocket.send_json({"type": "complete", "payload": payload})
    except ValidationError as error:
        await websocket.send_json(
            {"type": "error", "detail": error.errors(include_url=False)}
        )
    except ValueError as error:
        await websocket.send_json({"type": "error", "detail": str(error)})
    except WebSocketDisconnect:
        return
    finally:
        if (
            websocket.client_state is not WebSocketState.DISCONNECTED
            and websocket.application_state is not WebSocketState.DISCONNECTED
        ):
            await websocket.close()
