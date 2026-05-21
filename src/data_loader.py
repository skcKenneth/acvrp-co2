"""
data_loader.py
==============

Functions for retrieving the road-network graph from OpenStreetMap and
loading customer / depot information from a CSV file.

The road graph is cached locally so that experiments do not repeatedly
hit the Overpass API.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# osmnx is heavy and pulls in geospatial deps; import lazily inside the
# functions that need it so that `Customer` etc. remain usable without it.

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class Customer:
    """A single delivery node (depot if index == 0)."""

    index: int          # Position in the customer list; depot has index 0
    name: str
    lat: float
    lon: float
    demand: int         # Integer demand units; 0 for the depot


def download_road_graph(
    centre_lat: float,
    centre_lon: float,
    radius_m: int,
    network_type: str = "drive",
) -> nx.MultiDiGraph:
    """
    Download (or load from cache) a drivable street network centred on
    a lat/lon point.

    OSMnx returns a NetworkX MultiDiGraph whose edges carry attributes
    such as 'length' (metres) and 'oneway' (bool). Because the graph is
    *directed*, edges (u, v) and (v, u) need not both exist, which is
    exactly what we need to study asymmetry.

    Parameters
    ----------
    centre_lat, centre_lon : float
        Centre point of the bounding box in decimal degrees.
    radius_m : int
        Radius of the bounding box in metres.
    network_type : str
        OSMnx network filter. 'drive' is the standard choice for road
        routing.

    Returns
    -------
    networkx.MultiDiGraph
        The (cached) road network graph.
    """
    import osmnx as ox  # lazy import

    cache_file = CACHE_DIR / f"graph_{centre_lat:.4f}_{centre_lon:.4f}_{radius_m}.graphml"

    if cache_file.exists():
        graph = ox.load_graphml(cache_file)
        return graph

    # Newer OSMnx versions deprecated graph_from_point's distance argument
    # in favour of dist; we use the modern call signature.
    graph = ox.graph_from_point(
        center_point=(centre_lat, centre_lon),
        dist=radius_m,
        network_type=network_type,
        simplify=True,
    )
    ox.save_graphml(graph, cache_file)
    return graph


def load_customers(csv_path: str | os.PathLike) -> List[Customer]:
    """
    Load customer/depot definitions from a CSV file.

    The CSV must contain columns: name, latitude, longitude, demand.
    The first row is treated as the depot (demand should be 0).

    Returns
    -------
    list[Customer]
        Ordered list of customers; the first element is the depot.
    """
    df = pd.read_csv(csv_path)
    required = {"name", "latitude", "longitude", "demand"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Customer CSV is missing required columns: {missing}")

    customers = [
        Customer(
            index=i,
            name=row["name"],
            lat=float(row["latitude"]),
            lon=float(row["longitude"]),
            demand=int(row["demand"]),
        )
        for i, row in df.iterrows()
    ]

    # Sanity check: depot demand must be zero.
    if customers[0].demand != 0:
        raise ValueError(
            f"Depot (first row) must have demand=0; got {customers[0].demand}."
        )
    return customers


def _depot_strongly_connected_component(
    graph: nx.MultiDiGraph,
    depot_node: int,
) -> set[int]:
    """Nodes in the same directed strongly-connected component as the depot."""
    for component in nx.strongly_connected_components(graph):
        if depot_node in component:
            return component
    raise ValueError(f"Depot OSM node {depot_node} is not present in the graph.")


def _nearest_nodes_in_subset(
    graph: nx.MultiDiGraph,
    lons: Sequence[float],
    lats: Sequence[float],
    candidate_nodes: Sequence[int],
) -> List[int]:
    """Nearest-node snap restricted to ``candidate_nodes`` (graph x/y = lon/lat)."""
    if not candidate_nodes:
        raise ValueError("candidate_nodes must not be empty.")

    xs = np.array([graph.nodes[n]["x"] for n in candidate_nodes], dtype=float)
    ys = np.array([graph.nodes[n]["y"] for n in candidate_nodes], dtype=float)
    tree = cKDTree(np.column_stack([xs, ys]))
    points = np.column_stack(
        [np.asarray(lons, dtype=float), np.asarray(lats, dtype=float)]
    )
    _, indices = tree.query(points)
    return [candidate_nodes[int(i)] for i in np.atleast_1d(indices)]


def snap_customers_to_nodes(
    graph: nx.MultiDiGraph,
    customers: List[Customer],
    *,
    depot_index: int = 0,
    warn_on_resnap: bool = True,
) -> List[int]:
    """
    Map each customer's (lat, lon) to its nearest OSM node id.

    On one-way road networks the geographically nearest node can lie
    outside the depot's strongly-connected component, which makes some
    customer-to-customer legs unreachable and breaks AR fuel / CO2 totals.
    Customers snapped outside that component are re-snapped to the
    nearest node that *is* reachable from (and can reach) the depot.
    """
    import osmnx as ox  # lazy import

    lons = [c.lon for c in customers]
    lats = [c.lat for c in customers]
    node_ids = list(ox.nearest_nodes(graph, X=lons, Y=lats))

    depot_node = node_ids[depot_index]
    reachable = _depot_strongly_connected_component(graph, depot_node)
    unreachable_indices = [i for i, nid in enumerate(node_ids) if nid not in reachable]

    if unreachable_indices:
        replacement = _nearest_nodes_in_subset(
            graph,
            [lons[i] for i in unreachable_indices],
            [lats[i] for i in unreachable_indices],
            list(reachable),
        )
        for i, new_node in zip(unreachable_indices, replacement):
            if warn_on_resnap:
                print(
                    f"  Warning: {customers[i].name} was on an unreachable "
                    f"road node; re-snapped to the nearest depot-reachable node."
                )
            node_ids[i] = new_node

    return node_ids
