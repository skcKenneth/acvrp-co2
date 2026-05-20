"""
distance_matrix.py
==================

Constructs the four distance matrices that anchor the experimental
design:

    SE  -- Symmetric Euclidean (Haversine over a sphere)
    SM  -- Symmetric Manhattan (L1 on local equirectangular projection)
    SR  -- Symmetric Road (averaged true road distance in both ways)
    AR  -- Asymmetric Road (true directed shortest-path distance)

All matrices are returned as NumPy 2-D arrays in METRES so that downstream
emissions and CVRP code can treat them uniformly.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import networkx as nx
import numpy as np

from .data_loader import Customer


# Mean Earth radius (m) -- WGS84 IUGG value rounded to four sig figs.
EARTH_RADIUS_M = 6_371_000.0


# ---------------------------------------------------------------------------
# Symmetric matrices: closed-form formulas, no graph needed
# ---------------------------------------------------------------------------

def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance between two lat/lon points using the Haversine
    formula. Output is in metres.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    return EARTH_RADIUS_M * c


def build_euclidean_matrix(customers: List[Customer]) -> np.ndarray:
    """Symmetric great-circle distance matrix in metres."""
    n = len(customers)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_distance_m(
                customers[i].lat, customers[i].lon,
                customers[j].lat, customers[j].lon,
            )
            D[i, j] = D[j, i] = d
    return D


def build_manhattan_matrix(customers: List[Customer]) -> np.ndarray:
    """
    Symmetric Manhattan distance matrix.

    We project the lat/lon points to a local equirectangular plane
    centred at the mean latitude so that one degree of latitude and one
    degree of longitude correspond to the appropriate metric distances.
    """
    n = len(customers)
    lats = np.array([c.lat for c in customers])
    lons = np.array([c.lon for c in customers])

    mean_lat_rad = math.radians(lats.mean())
    # Metres per degree at this latitude
    m_per_deg_lat = 111_132.954 - 559.822 * math.cos(2 * mean_lat_rad)
    m_per_deg_lon = 111_412.84 * math.cos(mean_lat_rad)

    x = lons * m_per_deg_lon
    y = lats * m_per_deg_lat

    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(x[i] - x[j]) + abs(y[i] - y[j])
            D[i, j] = D[j, i] = d
    return D


# ---------------------------------------------------------------------------
# Road-network matrices: true shortest paths on the directed graph
# ---------------------------------------------------------------------------

def _shortest_path_length_safe(
    graph: nx.MultiDiGraph,
    source: int,
    target: int,
) -> float:
    """
    Dijkstra shortest path length using the 'length' edge attribute.
    Returns +inf if no directed path exists (which can happen on
    one-way networks).
    """
    if source == target:
        return 0.0
    try:
        return nx.shortest_path_length(graph, source, target, weight="length")
    except nx.NetworkXNoPath:
        return math.inf


def build_asymmetric_road_matrix(
    graph: nx.MultiDiGraph,
    node_ids: List[int],
) -> np.ndarray:
    """
    True directed shortest-path matrix on the road graph (in metres).

    Entry [i, j] is the length of the shortest directed path from
    `node_ids[i]` to `node_ids[j]`. Because the graph respects one-way
    streets, this matrix is in general asymmetric.

    For n nodes this calls n single-source Dijkstra computations; an
    O(n * (E + V log V)) total cost, which is well within budget for
    n <= a few hundred.
    """
    n = len(node_ids)
    D = np.zeros((n, n), dtype=float)
    for i, src in enumerate(node_ids):
        # single_source_dijkstra_path_length is much faster than calling
        # shortest_path_length n times because it reuses the heap.
        lengths = nx.single_source_dijkstra_path_length(graph, src, weight="length")
        for j, dst in enumerate(node_ids):
            if i == j:
                continue
            D[i, j] = lengths.get(dst, math.inf)
    return D


def symmetrise_road_matrix(asym: np.ndarray) -> np.ndarray:
    """
    Return a symmetric matrix obtained by averaging the two directions:

        SR[i, j] = SR[j, i] = (AR[i, j] + AR[j, i]) / 2

    Infinite entries in either direction propagate so that the
    symmetrised value is also infinite (no reachable round-trip).
    """
    sym = 0.5 * (asym + asym.T)
    # Where either direction was infinite, the average will already be
    # +inf because inf + finite = inf in NumPy float semantics.
    return sym


def build_all_matrices(
    customers: List[Customer],
    graph: nx.MultiDiGraph,
    node_ids: List[int],
) -> dict[str, np.ndarray]:
    """
    Convenience function that returns all four variants in a dict
    keyed by 'SE', 'SM', 'SR', 'AR'.
    """
    AR = build_asymmetric_road_matrix(graph, node_ids)
    SR = symmetrise_road_matrix(AR)
    SE = build_euclidean_matrix(customers)
    SM = build_manhattan_matrix(customers)
    return {"SE": SE, "SM": SM, "SR": SR, "AR": AR}


def asymmetry_index(asym: np.ndarray) -> float:
    """
    Scalar measure of how asymmetric an n x n matrix is.

    We use the ratio of the sum of |A_ij - A_ji| to the sum of
    (A_ij + A_ji), excluding the diagonal. The result lies in [0, 1];
    0 means the matrix is symmetric, 1 means maximally asymmetric.
    """
    n = asym.shape[0]
    num = 0.0
    den = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if math.isfinite(asym[i, j]) and math.isfinite(asym[j, i]):
                num += abs(asym[i, j] - asym[j, i])
                den += asym[i, j] + asym[j, i]
    return num / den if den > 0 else 0.0
