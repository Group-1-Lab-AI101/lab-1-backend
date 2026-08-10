"""Generate one continuous OSM driving graph plus landmark access nodes.

Install the data-generation dependencies first with:

    python -m pip install -r requirements-data.txt

The runtime backend does not depend on OSMnx. This script downloads a current
driving network that covers every landmark, projects each routing access point
onto the nearest road edge, splits that edge at the projected point, and writes
the compact GeoJSON files consumed by ``core.osm_loader``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import osmnx as ox
from shapely.geometry import LineString, Point, mapping
from shapely.ops import substring


ROOT = Path(__file__).resolve().parents[1]
LANDMARKS_PATH = ROOT / "data" / "landmarks.json"
OVERRIDES_PATH = ROOT / "data" / "landmark_access_overrides.json"
OUTPUT_DIR = ROOT / "data" / "osm"
DEFAULT_BUFFER_M = 600.0


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "item"):
        return _json_value(value.item())
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    return str(value)


def _edge_line(graph, source, target, data) -> LineString:
    geometry = data.get("geometry")
    if geometry is None:
        geometry = LineString(
            [
                (graph.nodes[source]["x"], graph.nodes[source]["y"]),
                (graph.nodes[target]["x"], graph.nodes[target]["y"]),
            ]
        )
    source_point = Point(graph.nodes[source]["x"], graph.nodes[source]["y"])
    coordinates = list(geometry.coords)
    if source_point.distance(Point(coordinates[-1])) < source_point.distance(
        Point(coordinates[0])
    ):
        coordinates.reverse()
    return LineString(coordinates)


def _split_edge(graph, source, target, key, access_node, ratio) -> None:
    data = copy.deepcopy(graph.edges[source, target, key])
    line = _edge_line(graph, source, target, data)
    first_line = substring(line, 0.0, ratio, normalized=True)
    second_line = substring(line, ratio, 1.0, normalized=True)
    if first_line.geom_type != "LineString" or second_line.geom_type != "LineString":
        return
    original_length = max(float(data.get("length") or 0.0), 0.001)
    graph.remove_edge(source, target, key)
    first = copy.deepcopy(data)
    first["length"] = original_length * ratio
    first["geometry"] = first_line
    second = copy.deepcopy(data)
    second["length"] = original_length * (1.0 - ratio)
    second["geometry"] = second_line
    graph.add_edge(source, access_node, **first)
    graph.add_edge(access_node, target, **second)


def _insert_access_node(graph, landmark, override) -> dict[str, Any]:
    query_latitude = float(override.get("latitude", landmark["latitude"]))
    query_longitude = float(override.get("longitude", landmark["longitude"]))
    projected = ox.projection.project_graph(graph)
    projected_crs = projected.graph["crs"]
    query_point = gpd.GeoSeries(
        [Point(query_longitude, query_latitude)], crs="EPSG:4326"
    ).to_crs(projected_crs).iloc[0]
    source, target, key = ox.distance.nearest_edges(
        projected, query_point.x, query_point.y
    )
    selected_data = projected.edges[source, target, key]
    selected_line = _edge_line(projected, source, target, selected_data)
    projected_distance = selected_line.project(query_point)
    road_point = selected_line.interpolate(projected_distance)
    road_wgs84 = gpd.GeoSeries([road_point], crs=projected_crs).to_crs(
        "EPSG:4326"
    ).iloc[0]
    road_longitude = float(road_wgs84.x)
    road_latitude = float(road_wgs84.y)
    access_node = f"landmark_access:{landmark['id']}"
    graph.add_node(
        access_node,
        x=road_longitude,
        y=road_latitude,
        street_count=2,
        landmark_access=landmark["id"],
    )

    split_candidates = []
    for candidate_source, candidate_target, candidate_key, candidate_data in list(
        projected.edges(keys=True, data=True)
    ):
        if {candidate_source, candidate_target} != {source, target}:
            continue
        candidate_line = _edge_line(
            projected, candidate_source, candidate_target, candidate_data
        )
        if candidate_line.distance(road_point) <= 0.75:
            distance = candidate_line.project(road_point)
            if 0.5 < distance < candidate_line.length - 0.5:
                split_candidates.append(
                    (
                        candidate_source,
                        candidate_target,
                        candidate_key,
                        distance / candidate_line.length,
                    )
                )
    for candidate in split_candidates:
        if graph.has_edge(candidate[0], candidate[1], candidate[2]):
            _split_edge(
                graph,
                candidate[0],
                candidate[1],
                candidate[2],
                access_node,
                candidate[3],
            )

    if not split_candidates:
        nearest_endpoint = min(
            (source, target),
            key=lambda node: Point(
                projected.nodes[node]["x"], projected.nodes[node]["y"]
            ).distance(road_point),
        )
        graph.remove_node(access_node)
        access_node = nearest_endpoint
        road_longitude = float(graph.nodes[access_node]["x"])
        road_latitude = float(graph.nodes[access_node]["y"])

    if override:
        use_road_projection = bool(override.get("use_road_projection", False))
        routing_latitude = road_latitude if use_road_projection else query_latitude
        routing_longitude = road_longitude if use_road_projection else query_longitude
        access_kind = str(override.get("kind", "entrance"))
        access_label = str(override.get("label", "Entrance"))
        access_source = str(override.get("source", "Curated access point"))
    else:
        routing_latitude = road_latitude
        routing_longitude = road_longitude
        access_kind = "nearest_road"
        access_label = "Nearest road access"
        access_source = "Generated from the nearest drivable OSM edge"
    road_name = selected_data.get("name")
    if isinstance(road_name, list):
        road_name = " / ".join(map(str, road_name))
    return {
        **landmark,
        "routing_latitude": round(routing_latitude, 7),
        "routing_longitude": round(routing_longitude, 7),
        "access_kind": access_kind,
        "access_label": access_label,
        "access_source": access_source,
        "access_road": str(road_name or "Unnamed road"),
    }


def _road_feature(graph, source, target, data) -> dict[str, Any]:
    line = _edge_line(graph, source, target, data)
    return {
        "type": "Feature",
        "properties": {
            "osmid": _json_value(data.get("osmid")),
            "highway": _json_value(data.get("highway")),
            "oneway": bool(data.get("oneway", False)),
            "reversed": _json_value(data.get("reversed", False)),
            "length": float(data.get("length") or 0.0),
            "lanes": _json_value(data.get("lanes")),
            "maxspeed": _json_value(data.get("maxspeed")),
            "name": _json_value(data.get("name")),
            "junction": _json_value(data.get("junction")),
            "access": _json_value(data.get("access")),
            "bridge": _json_value(data.get("bridge")),
            "relation_id": "continuous_landmark_bbox",
        },
        "geometry": mapping(line),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def generate(buffer_m: float) -> None:
    landmarks = json.loads(LANDMARKS_PATH.read_text(encoding="utf-8"))
    overrides = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
    latitudes = [float(item["latitude"]) for item in landmarks]
    longitudes = [float(item["longitude"]) for item in landmarks]
    middle_latitude = sum(latitudes) / len(latitudes)
    latitude_padding = buffer_m / 110_540.0
    longitude_padding = buffer_m / (
        111_320.0 * math.cos(math.radians(middle_latitude))
    )
    bbox = (
        min(longitudes) - longitude_padding,
        min(latitudes) - latitude_padding,
        max(longitudes) + longitude_padding,
        max(latitudes) + latitude_padding,
    )
    ox.settings.use_cache = True
    ox.settings.requests_timeout = 180
    graph = ox.graph.graph_from_bbox(
        bbox,
        network_type="drive",
        simplify=True,
        retain_all=False,
        truncate_by_edge=True,
    )
    enriched_landmarks = [
        _insert_access_node(graph, item, overrides.get(item["id"], {}))
        for item in landmarks
    ]

    road_features = [
        _road_feature(graph, source, target, data)
        for source, target, _key, data in graph.edges(keys=True, data=True)
    ]
    intersection_features = [
        {
            "type": "Feature",
            "properties": {
                "osmid": _json_value(node),
                "street_count": _json_value(data.get("street_count")),
                "landmark_access": _json_value(data.get("landmark_access")),
            },
            "geometry": {
                "type": "Point",
                "coordinates": [float(data["x"]), float(data["y"])],
            },
        }
        for node, data in graph.nodes(data=True)
    ]
    left, bottom, right, top = bbox
    boundary = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "Continuous landmark coverage",
                    "buffer_m": buffer_m,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [left, bottom],
                            [right, bottom],
                            [right, top],
                            [left, top],
                            [left, bottom],
                        ]
                    ],
                },
            }
        ],
    }
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "osmnx_version": ox.__version__,
        "network_type": "drive",
        "coverage": "continuous_landmark_bbox",
        "bbox": [left, bottom, right, top],
        "buffer_m": buffer_m,
        "total_roads": len(road_features),
        "total_intersections": len(intersection_features),
        "total_road_length_km": round(
            sum(feature["properties"]["length"] for feature in road_features)
            / 1000.0,
            3,
        ),
        "landmark_access_overrides": len(overrides),
        "source": "OpenStreetMap contributors via OSMnx",
        "errors": [],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(
        OUTPUT_DIR / "roads.geojson",
        {"type": "FeatureCollection", "features": road_features},
    )
    _write_json(
        OUTPUT_DIR / "intersections.geojson",
        {"type": "FeatureCollection", "features": intersection_features},
    )
    _write_json(OUTPUT_DIR / "boundary.geojson", boundary)
    _write_json(OUTPUT_DIR / "summary.json", summary)
    _write_json(LANDMARKS_PATH, enriched_landmarks)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buffer-m", type=float, default=DEFAULT_BUFFER_M)
    arguments = parser.parse_args()
    if arguments.buffer_m < 100:
        parser.error("--buffer-m must be at least 100")
    generate(arguments.buffer_m)


if __name__ == "__main__":
    main()
