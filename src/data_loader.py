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
from typing import List

import networkx as nx
import pandas as pd

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


def snap_customers_to_nodes(
    graph: nx.MultiDiGraph,
    customers: List[Customer],
) -> List[int]:
    """
    Map each customer's (lat, lon) to its nearest OSM node id.

    OSMnx's nearest_nodes uses a k-d tree under the hood, so this is
    efficient even for hundreds of points.
    """
    import osmnx as ox  # lazy import

    lons = [c.lon for c in customers]
    lats = [c.lat for c in customers]
    node_ids = ox.nearest_nodes(graph, X=lons, Y=lats)
    return list(node_ids)
