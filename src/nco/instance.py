"""
nco/instance.py
===============

Container for a single ACVRP instance and helpers to assemble batches of
instances into the tensors that the neural model consumes.

An instance contains
    locations    : (n, 2) float       lat/lon (used only for visualisation)
    demands      : (n,)   int         demand of each node (depot demand = 0)
    distance     : (n, n) float       asymmetric distance matrix (metres)
    time         : (n, n) float       arc-traversal time     (seconds)
    fuel_per_arc : (n, n) float       per-arc fuel litres at empty payload
    co2_per_arc  : (n, n) float       per-arc CO2 kg at empty payload
    capacity     : int                vehicle capacity
    depot_index  : int                index of the depot node (default 0)

Edge features (`time`, `fuel_per_arc`, `co2_per_arc`) are fed to the
encoder as additional input dimensions so the network can learn to
trade off distance against emissions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch  # type: ignore


@dataclass
class CVRPInstance:
    locations: np.ndarray          # (n, 2)
    demands: np.ndarray            # (n,)
    distance: np.ndarray           # (n, n)
    time: np.ndarray               # (n, n)
    fuel_per_arc: np.ndarray       # (n, n)  fuel litres at *empty* load
    co2_per_arc: np.ndarray        # (n, n)  CO2 kg at *empty* load
    capacity: int
    depot_index: int = 0
    city_name: str = "synthetic"

    @property
    def num_nodes(self) -> int:
        return int(self.demands.shape[0])

    @property
    def num_customers(self) -> int:
        return self.num_nodes - 1


@dataclass
class BatchedInstances:
    """A mini-batch of instances, padded to a common size."""

    locations: "torch.Tensor"      # (B, N, 2)
    demands: "torch.Tensor"        # (B, N)         int
    distance: "torch.Tensor"       # (B, N, N)      float
    edge_features: "torch.Tensor"  # (B, N, N, F)   float; F = 4 below
    capacity: "torch.Tensor"       # (B,)           int
    depot_index: "torch.Tensor"    # (B,)           int

    @property
    def batch_size(self) -> int:
        return int(self.locations.shape[0])

    @property
    def num_nodes(self) -> int:
        return int(self.locations.shape[1])

    def to(self, device) -> "BatchedInstances":
        return BatchedInstances(
            locations=self.locations.to(device),
            demands=self.demands.to(device),
            distance=self.distance.to(device),
            edge_features=self.edge_features.to(device),
            capacity=self.capacity.to(device),
            depot_index=self.depot_index.to(device),
        )


# The four edge-feature channels we expose to the encoder.
# Order matters: kept in sync with the model code below.
EDGE_FEATURE_NAMES: List[str] = [
    "distance_norm",   # distance / max(distance) per instance
    "time_norm",       # time / max(time) per instance
    "fuel_norm",       # fuel per arc at empty load, normalised
    "co2_norm",        # co2 per arc at empty load, normalised
]


def _safe_channel_norm(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Per-instance max-normalisation using only finite values."""
    finite = np.isfinite(x)
    out = np.zeros_like(x, dtype=np.float64)
    if finite.any():
        scale = float(np.max(x[finite])) + eps
        out[finite] = x[finite] / scale
    return out.astype(np.float32)


def collate_instances(instances: Sequence[CVRPInstance]) -> BatchedInstances:
    """
    Pad-stack a sequence of instances into a single batch.

    For simplicity we require all instances in a batch to share the same
    node count (this is the standard NCO assumption for a fixed-size
    training distribution). Asserting it up front catches accidental
    mixing.
    """
    import torch  # lazy import: we only need torch to call this function.

    n = instances[0].num_nodes
    for inst in instances[1:]:
        if inst.num_nodes != n:
            raise ValueError(
                "All instances in a batch must have the same node count; "
                f"got {n} and {inst.num_nodes}."
            )

    locs = torch.tensor(
        np.stack([inst.locations for inst in instances]), dtype=torch.float32
    )
    dem = torch.tensor(
        np.stack([inst.demands for inst in instances]), dtype=torch.long
    )
    dist = torch.tensor(
        np.stack([inst.distance for inst in instances]), dtype=torch.float32
    )

    # --- Build (B, N, N, F) edge-feature tensor -------------------------
    feats = []
    for inst in instances:
        d = inst.distance
        t = inst.time
        f = inst.fuel_per_arc
        c = inst.co2_per_arc

        # Per-instance max-normalisation (finite values only).
        d_n = _safe_channel_norm(d)
        t_n = _safe_channel_norm(t)
        f_n = _safe_channel_norm(f)
        c_n = _safe_channel_norm(c)

        feats.append(np.stack([d_n, t_n, f_n, c_n], axis=-1))
    edge_feats = torch.tensor(np.stack(feats), dtype=torch.float32)

    caps = torch.tensor(
        [inst.capacity for inst in instances], dtype=torch.long
    )
    depots = torch.tensor(
        [inst.depot_index for inst in instances], dtype=torch.long
    )

    return BatchedInstances(
        locations=locs,
        demands=dem,
        distance=dist,
        edge_features=edge_feats,
        capacity=caps,
        depot_index=depots,
    )
