"""Load OSM GeoJSON into the shared directed traffic-graph contract."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from core.contracts import Edge, Graph


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "osm"
DEFAULT_LANDMARKS_PATH = Path(__file__).resolve().parents[1] / "data" / "landmarks.json"

ROAD_SPEED_KMH = {
    "motorway": 70.0,
    "trunk": 55.0,
    "primary": 40.0,
    "primary_link": 30.0,
    "secondary": 35.0,
    "secondary_link": 28.0,
    "tertiary": 30.0,
    "tertiary_link": 25.0,
    "residential": 22.0,
    "living_street": 12.0,
    "service": 15.0,
    "unclassified": 24.0,
}

ROAD_CONGESTION = {
    "motorway": 2.5,
    "trunk": 3.0,
    "primary": 4.0,
    "primary_link": 3.5,
    "secondary": 3.2,
    "secondary_link": 3.0,
    "tertiary": 2.7,
    "tertiary_link": 2.5,
    "residential": 2.0,
    "living_street": 1.5,
    "service": 1.5,
    "unclassified": 2.2,
}

ROAD_RISK = {
    "motorway": 0.2,
    "trunk": 0.2,
    "primary": 0.3,
    "primary_link": 0.5,
    "secondary": 0.4,
    "secondary_link": 0.6,
    "tertiary": 0.5,
    "tertiary_link": 0.7,
    "residential": 0.7,
    "living_street": 0.8,
    "service": 1.0,
    "unclassified": 0.8,
}

TRAFFIC_PROFILES = {
    "normal": "Typical daytime traffic",
    "rush_hour": "Heavier congestion and slower travel on main roads",
    "rainy": "Longer travel time and higher road-risk penalty",
}


@dataclass(frozen=True)
class Landmark:
    """A user-facing place snapped to one routable OSM graph node."""

    id: str
    name: str
    category: str
    description: str
    latitude: float
    longitude: float
    snapped_node: str
    snapped_distance_m: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible landmark representation."""
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "snapped_node": self.snapped_node,
            "snapped_distance_m": self.snapped_distance_m,
        }


@dataclass
class TrafficNetwork:
    """In-memory OSM road graph plus geometry and landmark adapters."""

    graph: dict[str, tuple[Edge, ...]]
    coordinates: dict[str, tuple[float, float]]
    landmarks: dict[str, Landmark]
    boundary_geojson: dict[str, Any]
    summary: dict[str, Any]
    edge_lookup: dict[tuple[str, str], Edge]

    def route_coordinates(self, path: Sequence[str]) -> list[list[float]]:
        """Return joined `[longitude, latitude]` coordinates for a node path."""
        if not path:
            return []
        if len(path) == 1:
            latitude, longitude = self.coordinates[path[0]]
            return [[longitude, latitude]]

        joined: list[list[float]] = []
        for source, target in zip(path, path[1:]):
            edge = self.edge_lookup[(source, target)]
            raw_geometry = edge.metadata.get("geometry", [])
            geometry = [list(point[:2]) for point in raw_geometry]
            if not geometry:
                source_lat, source_lon = self.coordinates[source]
                target_lat, target_lon = self.coordinates[target]
                geometry = [[source_lon, source_lat], [target_lon, target_lat]]
            if joined and joined[-1] == geometry[0]:
                joined.extend(geometry[1:])
            else:
                joined.extend(geometry)
        return joined

    def route_geojson(self, path: Sequence[str]) -> dict[str, Any]:
        """Return a GeoJSON feature for a complete route path."""
        return {
            "type": "Feature",
            "properties": {"node_count": len(path)},
            "geometry": {
                "type": "LineString",
                "coordinates": self.route_coordinates(path),
            },
        }

    def roads_geojson(self) -> dict[str, Any]:
        """Return the routable strongly connected road network as GeoJSON."""
        features: list[dict[str, Any]] = []
        for source, edges in self.graph.items():
            for edge in edges:
                geometry = edge.metadata.get("geometry")
                if not geometry:
                    source_lat, source_lon = self.coordinates[source]
                    target_lat, target_lon = self.coordinates[edge.to]
                    geometry = [[source_lon, source_lat], [target_lon, target_lat]]
                features.append(
                    {
                        "type": "Feature",
                        "properties": {
                            "source": source,
                            "target": edge.to,
                            "name": edge.metadata.get("name"),
                            "road_type": edge.road_type,
                            "distance_km": edge.distance_km,
                            "time_min": edge.time_min,
                            "congestion": edge.congestion,
                            "risk": edge.risk,
                            "oneway": edge.metadata.get("oneway", False),
                        },
                        "geometry": {
                            "type": "LineString",
                            "coordinates": geometry,
                        },
                    }
                )
        return {"type": "FeatureCollection", "features": features}


def _road_type(value: Any) -> str:
    if isinstance(value, list):
        value = value[0] if value else "unclassified"
    road_type = str(value or "unclassified")
    return road_type if road_type in ROAD_SPEED_KMH else "unclassified"


def _parse_speed(value: Any, road_type: str) -> float:
    if isinstance(value, list):
        value = value[0] if value else None
    match = re.search(r"\d+(?:\.\d+)?", str(value or ""))
    if match:
        parsed = float(match.group())
        if 5 <= parsed <= 100:
            return parsed
    return ROAD_SPEED_KMH[road_type]


def _node_id(longitude: float, latitude: float) -> str:
    lat = f"{latitude:.7f}".replace("-", "m").replace(".", "_")
    lon = f"{longitude:.7f}".replace("-", "m").replace(".", "_")
    return f"road_{lat}_{lon}"


def _haversine_km(
    latitude_1: float,
    longitude_1: float,
    latitude_2: float,
    longitude_2: float,
) -> float:
    lat_1 = math.radians(latitude_1)
    lat_2 = math.radians(latitude_2)
    delta_lat = lat_2 - lat_1
    delta_lon = math.radians(longitude_2 - longitude_1)
    haversine = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat_1) * math.cos(lat_2) * math.sin(delta_lon / 2) ** 2
    )
    return 6371.0088 * 2 * math.atan2(
        math.sqrt(haversine), math.sqrt(max(0.0, 1 - haversine))
    )


def _strongly_connected_components(
    adjacency: Mapping[str, Mapping[str, Edge]],
) -> list[set[str]]:
    """Find SCCs with iterative Kosaraju traversal, largest first."""
    seen: set[str] = set()
    finish_order: list[str] = []
    for start in adjacency:
        if start in seen:
            continue
        stack: list[tuple[str, bool]] = [(start, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                finish_order.append(node)
                continue
            if node in seen:
                continue
            seen.add(node)
            stack.append((node, True))
            for neighbor in reversed(list(adjacency[node])):
                if neighbor not in seen:
                    stack.append((neighbor, False))

    reverse: dict[str, list[str]] = {node: [] for node in adjacency}
    for source, targets in adjacency.items():
        for target in targets:
            reverse[target].append(source)

    assigned: set[str] = set()
    components: list[set[str]] = []
    for start in reversed(finish_order):
        if start in assigned:
            continue
        component: set[str] = set()
        stack = [start]
        assigned.add(start)
        while stack:
            node = stack.pop()
            component.add(node)
            for neighbor in reverse[node]:
                if neighbor not in assigned:
                    assigned.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return sorted(components, key=len, reverse=True)


def _connect_major_components(
    adjacency: dict[str, dict[str, Edge]],
    coordinates: Mapping[str, tuple[float, float]],
) -> None:
    """Bridge separately clipped neighboring OSM regions with short links."""
    major_components = [
        component
        for component in _strongly_connected_components(adjacency)
        if len(component) >= 20
    ]
    if len(major_components) < 2:
        return

    connected = set(major_components[0])
    for component in major_components[1:]:
        source, target, distance_km = min(
            (
                (
                    source_node,
                    target_node,
                    _haversine_km(
                        coordinates[source_node][0],
                        coordinates[source_node][1],
                        coordinates[target_node][0],
                        coordinates[target_node][1],
                    ),
                )
                for source_node in connected
                for target_node in component
            ),
            key=lambda item: item[2],
        )
        source_lat, source_lon = coordinates[source]
        target_lat, target_lon = coordinates[target]
        geometry = [[source_lon, source_lat], [target_lon, target_lat]]
        safe_distance = max(distance_km, 0.001)
        for from_node, to_node, line in (
            (source, target, geometry),
            (target, source, list(reversed(geometry))),
        ):
            adjacency[from_node][to_node] = Edge(
                to=to_node,
                distance_km=safe_distance,
                time_min=safe_distance / 20.0 * 60.0,
                congestion=2.0,
                risk=0.8,
                road_type="unclassified",
                metadata={
                    "name": "Boundary connector",
                    "oneway": False,
                    "geometry": line,
                    "simulated_connector": True,
                    "base_time_min": safe_distance / 20.0 * 60.0,
                    "base_congestion": 2.0,
                    "base_risk": 0.8,
                },
            )
        connected.update(component)


def _build_graph(
    roads: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, tuple[Edge, ...]], dict[str, tuple[float, float]]]:
    adjacency: dict[str, dict[str, Edge]] = {}
    coordinates: dict[str, tuple[float, float]] = {}

    for feature in roads:
        geometry = feature.get("geometry", {})
        raw_coordinates = geometry.get("coordinates", [])
        if geometry.get("type") != "LineString" or len(raw_coordinates) < 2:
            continue
        source_lon, source_lat = map(float, raw_coordinates[0][:2])
        target_lon, target_lat = map(float, raw_coordinates[-1][:2])
        source = _node_id(source_lon, source_lat)
        target = _node_id(target_lon, target_lat)
        if source == target:
            continue
        coordinates[source] = (source_lat, source_lon)
        coordinates[target] = (target_lat, target_lon)
        adjacency.setdefault(source, {})
        adjacency.setdefault(target, {})

        properties = feature.get("properties", {})
        road_type = _road_type(properties.get("highway"))
        distance_km = max(float(properties.get("length") or 0.0) / 1000.0, 0.001)
        speed_kmh = _parse_speed(properties.get("maxspeed"), road_type)
        time_min = distance_km / speed_kmh * 60.0
        edge = Edge(
            to=target,
            distance_km=distance_km,
            time_min=time_min,
            congestion=ROAD_CONGESTION[road_type],
            risk=ROAD_RISK[road_type],
            road_type=road_type,
            metadata={
                "name": properties.get("name"),
                "osmid": properties.get("osmid"),
                "oneway": bool(properties.get("oneway", False)),
                "lanes": properties.get("lanes"),
                "maxspeed_kmh": speed_kmh,
                "relation_id": properties.get("relation_id"),
                "geometry": [list(point[:2]) for point in raw_coordinates],
                "base_time_min": time_min,
                "base_congestion": ROAD_CONGESTION[road_type],
                "base_risk": ROAD_RISK[road_type],
            },
        )
        previous = adjacency[source].get(target)
        if previous is None or edge.time_min < previous.time_min:
            adjacency[source][target] = edge

    _connect_major_components(adjacency, coordinates)
    routable_nodes = _strongly_connected_components(adjacency)[0]
    filtered_graph = {
        source: tuple(
            edge for target, edge in targets.items() if target in routable_nodes
        )
        for source, targets in adjacency.items()
        if source in routable_nodes
    }
    filtered_coordinates = {
        node: coordinate
        for node, coordinate in coordinates.items()
        if node in routable_nodes
    }
    return filtered_graph, filtered_coordinates


def _load_landmarks(
    path: Path, coordinates: Mapping[str, tuple[float, float]]
) -> dict[str, Landmark]:
    with path.open("r", encoding="utf-8") as stream:
        raw_landmarks = json.load(stream)
    landmarks: dict[str, Landmark] = {}
    used_nodes: set[str] = set()
    for raw in raw_landmarks:
        latitude = float(raw["latitude"])
        longitude = float(raw["longitude"])
        if len(used_nodes) >= len(coordinates):
            raise ValueError("There are more landmarks than routable graph nodes")
        snapped_node, distance_km = min(
            (
                (
                    node,
                    _haversine_km(latitude, longitude, node_lat, node_lon),
                )
                for node, (node_lat, node_lon) in coordinates.items()
                if node not in used_nodes
            ),
            key=lambda item: item[1],
        )
        landmark = Landmark(
            id=str(raw["id"]),
            name=str(raw["name"]),
            category=str(raw["category"]),
            description=str(raw["description"]),
            latitude=latitude,
            longitude=longitude,
            snapped_node=snapped_node,
            snapped_distance_m=round(distance_km * 1000.0, 1),
        )
        if landmark.id in landmarks:
            raise ValueError(f"Duplicate landmark ID: {landmark.id!r}")
        landmarks[landmark.id] = landmark
        used_nodes.add(snapped_node)
    return landmarks


@lru_cache(maxsize=1)
def load_traffic_network(
    data_dir: Path = DEFAULT_DATA_DIR,
    landmarks_path: Path = DEFAULT_LANDMARKS_PATH,
) -> TrafficNetwork:
    """Load and cache the routable OSM graph and landmark adapters."""
    with (data_dir / "roads.geojson").open("r", encoding="utf-8") as stream:
        roads_geojson = json.load(stream)
    with (data_dir / "boundary.geojson").open("r", encoding="utf-8") as stream:
        boundary_geojson = json.load(stream)
    with (data_dir / "summary.json").open("r", encoding="utf-8") as stream:
        summary = json.load(stream)

    graph, coordinates = _build_graph(roads_geojson.get("features", []))
    if len(graph) < 20 or sum(len(edges) for edges in graph.values()) < 30:
        raise ValueError("OSM graph does not satisfy the 20-node/30-edge requirement")
    landmarks = _load_landmarks(landmarks_path, coordinates)
    edge_lookup = {
        (source, edge.to): edge
        for source, edges in graph.items()
        for edge in edges
    }
    summary = dict(summary)
    summary.update(
        {
            "routable_nodes": len(graph),
            "routable_edges": len(edge_lookup),
            "landmarks": len(landmarks),
            "unique_landmark_nodes": len(
                {landmark.snapped_node for landmark in landmarks.values()}
            ),
            "simulated_connectors": sum(
                1
                for edge in edge_lookup.values()
                if edge.metadata.get("simulated_connector")
            ),
            "source": "OpenStreetMap via the group's osmnx-tools repository",
        }
    )
    return TrafficNetwork(
        graph=graph,
        coordinates=coordinates,
        landmarks=landmarks,
        boundary_geojson=boundary_geojson,
        summary=summary,
        edge_lookup=edge_lookup,
    )


def apply_traffic_profile(graph: Graph, profile: str) -> dict[str, tuple[Edge, ...]]:
    """Return an immutable-style graph copy adjusted for a traffic profile."""
    if profile not in TRAFFIC_PROFILES:
        supported = ", ".join(sorted(TRAFFIC_PROFILES))
        raise ValueError(f"Unknown traffic profile {profile!r}; expected: {supported}")
    if profile == "normal":
        return {node: tuple(edges) for node, edges in graph.items()}

    adjusted: dict[str, tuple[Edge, ...]] = {}
    for source, edges in graph.items():
        transformed: list[Edge] = []
        for edge in edges:
            if profile == "rush_hour":
                main_road_factor = 0.18 if edge.road_type in {
                    "primary",
                    "primary_link",
                    "secondary",
                    "secondary_link",
                } else 0.05
                time_multiplier = 1.25 + main_road_factor * edge.congestion
                congestion = min(5.0, edge.congestion + 1.0)
                risk = edge.risk
            else:
                time_multiplier = 1.25 + 0.05 * edge.risk
                congestion = min(5.0, edge.congestion + 0.4)
                risk = min(5.0, edge.risk + 1.2)
            transformed.append(
                Edge(
                    to=edge.to,
                    distance_km=edge.distance_km,
                    time_min=edge.time_min * time_multiplier,
                    congestion=congestion,
                    risk=risk,
                    road_type=edge.road_type,
                    metadata={**edge.metadata, "traffic_profile": profile},
                )
            )
        adjusted[source] = tuple(transformed)
    return adjusted
