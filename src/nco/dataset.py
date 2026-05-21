"""
nco/dataset.py
==============

Generation pipelines for the training and evaluation distributions.

We support two regimes:

1. **Synthetic** instances built from random points in the unit square,
   then perturbed into an *asymmetric* matrix by multiplying half of
   the off-diagonal entries by a random factor in [1.0, 1.4]. This
   gives a controllable proxy for "one-way streets" without OSM.

2. **Real OSM** instances built by sampling random customer points
   inside a bounding box around a city centre, snapping them to the
   nearest drivable nodes, and computing all-pairs shortest paths on
   the directed road graph.

Both generators return ``CVRPInstance`` objects ready to be batched.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterator, List, Optional

import networkx as nx
import numpy as np

from ..emissions_model import EmissionsParams, arc_fuel_litres, fuel_to_co2_kg
from .instance import CVRPInstance


# ---------------------------------------------------------------------------
# Synthetic generation -------------------------------------------------------
# ---------------------------------------------------------------------------

@dataclass
class SyntheticConfig:
    num_customers: int = 20
    capacity: int = 30
    demand_low: int = 1
    demand_high: int = 9
    asymmetry_factor_max: float = 1.4  # 1.0 = symmetric, 1.4 = up to 40% asymmetry
    depot_at_centre: bool = True


def make_synthetic_instance(
    cfg: SyntheticConfig,
    params: EmissionsParams,
    rng: np.random.Generator,
) -> CVRPInstance:
    """
    Build a single synthetic ACVRP instance with the configured asymmetry.

    Coordinates are uniform in the unit square scaled to a 5 km area so
    that distances live in metres consistent with the OSM pipeline.
    """
    n = cfg.num_customers + 1
    SCALE_M = 5000.0  # 5 km square

    coords = rng.uniform(0, 1, size=(n, 2)) * SCALE_M
    if cfg.depot_at_centre:
        coords[0] = SCALE_M / 2

    # Symmetric base distance matrix from Euclidean coords
    diff = coords[:, None, :] - coords[None, :, :]
    base = np.sqrt((diff ** 2).sum(axis=-1))

    # Inject asymmetry: for each unordered pair (i, j), pick one of the
    # two directions at random and scale it up by a factor in [1, fmax].
    mat = base.copy()
    for i in range(n):
        for j in range(i + 1, n):
            factor = 1.0 + (cfg.asymmetry_factor_max - 1.0) * rng.random()
            if rng.random() < 0.5:
                mat[i, j] *= factor
            else:
                mat[j, i] *= factor

    demands = np.zeros(n, dtype=np.int64)
    demands[1:] = rng.integers(cfg.demand_low, cfg.demand_high + 1, size=n - 1)

    return _build_instance_from_distance(
        coords, demands, mat, cfg.capacity, params, "synthetic"
    )


def synthetic_dataset(
    cfg: SyntheticConfig,
    params: EmissionsParams,
    num_instances: int,
    seed: int = 0,
) -> List[CVRPInstance]:
    """Generate a list of synthetic instances with a fixed seed."""
    rng = np.random.default_rng(seed)
    return [make_synthetic_instance(cfg, params, rng) for _ in range(num_instances)]


# ---------------------------------------------------------------------------
# OSM-derived generation -----------------------------------------------------
# ---------------------------------------------------------------------------

def make_osm_instance(
    graph: nx.MultiDiGraph,
    num_customers: int,
    capacity: int,
    params: EmissionsParams,
    rng: np.random.Generator,
    city_name: str = "osm",
    demand_low: int = 1,
    demand_high: int = 9,
) -> CVRPInstance:
    """
    Sample `num_customers` random nodes from a road graph and build an
    ACVRP instance from the all-pairs directed shortest-path distances.

    The depot is sampled from the most central nodes (defined here as
    the nearest node to the graph's centroid in lat/lon).
    """
    n = num_customers + 1
    all_nodes = list(graph.nodes(data=True))

    # Depot: most central node by lat/lon centroid
    mean_y = np.mean([d["y"] for _, d in all_nodes])
    mean_x = np.mean([d["x"] for _, d in all_nodes])
    depot_id = min(all_nodes, key=lambda nd: (nd[1]["y"] - mean_y) ** 2 + (nd[1]["x"] - mean_x) ** 2)[0]

    sampled = rng.choice(
        [nid for nid, _ in all_nodes if nid != depot_id],
        size=num_customers,
        replace=False,
    )
    node_ids = [depot_id] + list(sampled)

    # Coordinates for visualisation
    coords = np.array([[graph.nodes[nid]["y"], graph.nodes[nid]["x"]] for nid in node_ids])

    # Directed all-pairs shortest path on edge 'length' (metres)
    mat = np.zeros((n, n), dtype=float)
    for i, src in enumerate(node_ids):
        lengths = nx.single_source_dijkstra_path_length(graph, src, weight="length")
        for j, dst in enumerate(node_ids):
            if i == j:
                continue
            mat[i, j] = lengths.get(dst, math.inf)

    # If any node is unreachable from another, redraw — easier than
    # patching infinities through the entire pipeline.
    if not np.isfinite(mat).all():
        # Retry once with a different sample; in practice this is rare
        # for a well-connected urban graph.
        return make_osm_instance(
            graph, num_customers, capacity, params, rng,
            city_name, demand_low, demand_high,
        )

    demands = np.zeros(n, dtype=np.int64)
    demands[1:] = rng.integers(demand_low, demand_high + 1, size=n - 1)

    return _build_instance_from_distance(
        coords, demands, mat, capacity, params, city_name,
    )


def osm_dataset(
    graph: nx.MultiDiGraph,
    num_customers: int,
    capacity: int,
    params: EmissionsParams,
    num_instances: int,
    city_name: str = "osm",
    seed: int = 0,
) -> List[CVRPInstance]:
    """Generate multiple ACVRP instances from a single road graph."""
    rng = np.random.default_rng(seed)
    return [
        make_osm_instance(
            graph, num_customers, capacity, params, rng, city_name=city_name
        )
        for _ in range(num_instances)
    ]


# ---------------------------------------------------------------------------
# Helper: derive edge features from a distance matrix
# ---------------------------------------------------------------------------

def sanitize_distance_matrix(
    dist: np.ndarray,
    *,
    penalty: float | None = None,
) -> np.ndarray:
    """
    Replace non-finite off-diagonal entries with a large finite penalty.

    OSM road matrices can contain ``inf`` for unreachable directed arcs.
    Classical solvers treat those arcs as prohibitively expensive; the NCO
    pipeline needs finite values so edge-feature normalisation and CUDA
    sampling remain well-defined.
    """
    out = np.asarray(dist, dtype=np.float64, order="C").copy()
    n = out.shape[0]
    off_diag = ~np.eye(n, dtype=bool)
    bad = off_diag & ~np.isfinite(out)
    if not bad.any():
        return out.astype(np.float32)

    finite_off = off_diag & np.isfinite(out)
    if penalty is None:
        if finite_off.any():
            penalty = float(np.max(out[finite_off]) * 10.0)
        else:
            penalty = 1e7
    out[bad] = penalty
    return out.astype(np.float32)


def _build_instance_from_distance(
    coords: np.ndarray,
    demands: np.ndarray,
    dist: np.ndarray,
    capacity: int,
    params: EmissionsParams,
    city_name: str,
) -> CVRPInstance:
    """
    Convert (coords, demands, distance) into a fully populated
    CVRPInstance by computing time / fuel / CO2 from the linear
    emissions model at empty payload.

    We use "empty payload" for the edge features because the *true*
    payload depends on the route the model eventually builds, so it is
    not knowable at the encoder stage. The decoder reasons about
    payload implicitly through its capacity state.
    """
    dist = sanitize_distance_matrix(dist)
    speed = params.avg_speed_mps
    time = dist / max(speed, 1e-9)

    fuel = np.zeros_like(dist)
    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            if i == j:
                continue
            fuel[i, j] = arc_fuel_litres(float(dist[i, j]), payload_kg=0.0, params=params)

    co2 = fuel_to_co2_kg(fuel, params)

    return CVRPInstance(
        locations=coords.astype(np.float32),
        demands=demands.astype(np.int64),
        distance=dist.astype(np.float32),
        time=time.astype(np.float32),
        fuel_per_arc=fuel.astype(np.float32),
        co2_per_arc=co2.astype(np.float32),
        capacity=int(capacity),
        depot_index=0,
        city_name=city_name,
    )


# ---------------------------------------------------------------------------
# Iterator wrappers for training
# ---------------------------------------------------------------------------

class StreamingSyntheticDataset:
    """
    Infinite iterator yielding fresh synthetic instances on demand.

    Used during training so that the model never sees the same instance
    twice; this avoids the overfitting trap that all NCO models share.
    """

    def __init__(
        self,
        cfg: SyntheticConfig,
        params: EmissionsParams,
        seed: Optional[int] = None,
    ):
        self.cfg = cfg
        self.params = params
        self.rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[CVRPInstance]:
        while True:
            yield make_synthetic_instance(self.cfg, self.params, self.rng)

    def sample_batch(self, batch_size: int) -> List[CVRPInstance]:
        return [make_synthetic_instance(self.cfg, self.params, self.rng)
                for _ in range(batch_size)]
