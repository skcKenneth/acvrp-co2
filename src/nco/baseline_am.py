"""
nco/baseline_am.py
==================

Vanilla Attention Model (Kool, Hoof, Welling, 2019) baseline.

The full MatNet-CVRP architecture in this project consumes both node
coordinates *and* per-arc edge features (distance, time, fuel, CO2).
The baseline AM is the standard reference architecture from the
literature: it ignores edge features entirely and embeds each node
purely from (x, y) coordinates plus demand.

We include this baseline for two reasons:

1. To answer RQ3: how badly does a model designed for symmetric
   Euclidean CVRP perform on asymmetric road-network instances? If the
   gap is large, that justifies our bidirectional edge-attention
   approach.

2. To act as a sanity floor: any meaningful architectural addition
   should produce a model that beats this on the AR (asymmetric)
   ground-truth metric.

Architecturally, the baseline reuses the same capacity-aware decoder as
the main model but swaps out the encoder for a coord-only multi-head
self-attention transformer (no edge bias).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import CapacityAwareDecoder, make_mask
from .instance import BatchedInstances
from .model import Rollout


class CoordAttentionEncoder(nn.Module):
    """
    Standard Transformer encoder over node coordinates only.

    This mirrors the encoder used by Kool et al. 2019 and POMO (Kwon
    et al. 2020). No edge features are consumed, so the encoder has no
    way to detect (or react to) asymmetry in the underlying graph.
    """

    def __init__(
        self,
        node_feature_dim: int = 3,    # (x, y, demand_norm)
        embed_dim: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        ffn_dim: int = 256,
    ):
        super().__init__()
        if embed_dim % n_heads != 0:
            raise ValueError("embed_dim must be divisible by n_heads")
        self.embed_dim = embed_dim
        self.n_heads = n_heads

        self.input_proj = nn.Linear(node_feature_dim, embed_dim)
        self.blocks = nn.ModuleList([
            _SelfAttnBlock(embed_dim, n_heads, ffn_dim)
            for _ in range(n_layers)
        ])

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(node_features)
        for block in self.blocks:
            h = block(h)
        return h


class _SelfAttnBlock(nn.Module):
    """A single MHA + FFN block with skip connections and LayerNorm."""

    def __init__(self, embed_dim: int, n_heads: int, ffn_dim: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, n_heads, batch_first=True,
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + a)
        x = self.ln2(x + self.ffn(x))
        return x


class CoordOnlyACVRPPolicy(nn.Module):
    """
    Baseline policy that mimics the vanilla AM/POMO approach.

    Identical in shape to `ACVRPPolicy` but uses `CoordAttentionEncoder`
    instead of `BidirectionalEncoder`, so the model receives no signal
    about arc asymmetry. The decoder still respects capacity through
    masking and tour distance is computed under the ground-truth
    asymmetric distance matrix (otherwise we'd be cheating by training
    against a fictional symmetric metric).
    """

    def __init__(
        self,
        node_feature_dim: int = 3,
        embed_dim: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        ffn_dim: int = 256,
        tanh_clip: float = 10.0,
        cost_mode: str = "distance",
        co2_weight: float = 0.5,
        co2_scale: float = 1000.0,
    ):
        super().__init__()
        if cost_mode not in {"distance", "co2", "blend"}:
            raise ValueError(
                f"cost_mode must be 'distance', 'co2', or 'blend'; got {cost_mode!r}."
            )
        # See note in ACVRPPolicy: persist constructor args for checkpoint
        # introspection. Note this class has no edge_feature_dim.
        self.node_feature_dim = node_feature_dim
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ffn_dim = ffn_dim
        self.tanh_clip = tanh_clip
        self.cost_mode = cost_mode
        self.co2_weight = float(co2_weight)
        self.co2_scale = float(co2_scale)

        self.encoder = CoordAttentionEncoder(
            node_feature_dim=node_feature_dim,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            ffn_dim=ffn_dim,
        )
        self.decoder = CapacityAwareDecoder(
            embed_dim=embed_dim,
            n_heads=n_heads,
            tanh_clip=tanh_clip,
        )

    @staticmethod
    def _node_features(batch: BatchedInstances) -> torch.Tensor:
        """Same node-feature construction as the main model."""
        locs = batch.locations
        mins = locs.min(dim=1, keepdim=True).values
        maxs = locs.max(dim=1, keepdim=True).values
        scale = (maxs - mins).clamp_min(1e-6)
        loc_norm = (locs - mins) / scale
        dem_norm = (
            batch.demands.float() / batch.capacity.float().unsqueeze(-1)
        ).unsqueeze(-1)
        return torch.cat([loc_norm, dem_norm], dim=-1)

    def _rollout(
        self,
        batch: BatchedInstances,
        greedy: bool,
        start_action: Optional[torch.Tensor] = None,
    ) -> Rollout:
        # Re-use the same rollout logic as ACVRPPolicy but with the
        # coord-only encoder. We deliberately duplicate the loop here
        # (rather than refactor ACVRPPolicy) to keep each policy class
        # self-contained and easy to read.
        device = batch.locations.device
        B, N = batch.batch_size, batch.num_nodes

        node_feats = self._node_features(batch)
        node_embeds = self.encoder(node_feats)
        graph_embed = node_embeds.mean(dim=1)

        current = batch.depot_index.clone()
        visited = torch.zeros(B, N, dtype=torch.bool, device=device)
        visited.scatter_(1, batch.depot_index.unsqueeze(-1), True)
        remaining = batch.capacity.clone().long()

        distance_acc = torch.zeros(B, device=device)
        co2_acc = torch.zeros(B, device=device)
        log_prob_sum = torch.zeros(B, device=device)
        actions = [current.clone()]

        # Read absolute CO2 directly from the batch (set by
        # collate_instances). This was previously reconstructed by
        # dividing two normalised channels, which collapsed to 1.0
        # everywhere and silently turned the CO2 reward into distance.
        co2_per_arc_kg = batch.co2_per_arc

        max_steps = 2 * N
        for step in range(max_steps):
            mask = make_mask(
                visited, batch.demands, remaining, current, batch.depot_index
            )
            logits = self.decoder.step(
                node_embeds=node_embeds,
                graph_embed=graph_embed,
                current_node=current,
                remaining_cap=remaining,
                capacity=batch.capacity,
                mask=mask,
            )

            if step == 0 and start_action is not None:
                # POMO exploration: forced first action; detach its
                # log-probability so the gradient does not flow through
                # this non-policy choice (matches Kwon et al. 2020).
                next_node = start_action
                step_lp = torch.log_softmax(logits, dim=-1).gather(
                    1, next_node.unsqueeze(-1)
                ).squeeze(-1).detach()
            else:
                if greedy:
                    next_node = logits.argmax(dim=-1)
                else:
                    probs = torch.softmax(logits, dim=-1)
                    next_node = torch.multinomial(probs, num_samples=1).squeeze(-1)
                step_lp = torch.log_softmax(logits, dim=-1).gather(
                    1, next_node.unsqueeze(-1)
                ).squeeze(-1)

            d_row = batch.distance.gather(
                1, current.view(B, 1, 1).expand(B, 1, N)
            ).squeeze(1)
            arc_dist = d_row.gather(1, next_node.unsqueeze(-1)).squeeze(-1)
            co2_row = co2_per_arc_kg.gather(
                1, current.view(B, 1, 1).expand(B, 1, N)
            ).squeeze(1)
            arc_co2 = co2_row.gather(1, next_node.unsqueeze(-1)).squeeze(-1)

            distance_acc = distance_acc + arc_dist
            co2_acc = co2_acc + arc_co2
            log_prob_sum = log_prob_sum + step_lp

            # Update state. scatter() already sets the bit; no OR needed.
            visited = visited.scatter(1, next_node.unsqueeze(-1), True)
            served = next_node != batch.depot_index
            picked_demand = batch.demands.gather(1, next_node.unsqueeze(-1)).squeeze(-1)
            remaining = torch.where(
                served, remaining - picked_demand, batch.capacity.clone().long()
            )
            current = next_node
            actions.append(current.clone())

            arange_n = torch.arange(N, device=device)
            not_depot_col = arange_n.unsqueeze(0) != batch.depot_index.unsqueeze(-1)
            customers_done = (visited | ~not_depot_col).all(dim=-1)
            at_depot = current == batch.depot_index
            if (customers_done & at_depot).all():
                break

        if self.cost_mode == "distance":
            cost = distance_acc
        elif self.cost_mode == "co2":
            cost = co2_acc * self.co2_scale
        else:
            cost = (1.0 - self.co2_weight) * distance_acc \
                 + self.co2_weight * self.co2_scale * co2_acc

        return Rollout(
            actions=torch.stack(actions, dim=1),
            log_probs=log_prob_sum,
            cost=cost,
            distance_m=distance_acc,
            co2_kg=co2_acc,
        )

    # Public mirror of ACVRPPolicy
    def sample(self, batch):  return self._rollout(batch, greedy=False)
    def greedy(self, batch):  return self._rollout(batch, greedy=True)

    def pomo_sample(self, batch: BatchedInstances, n_starts: int) -> Rollout:
        device = batch.locations.device
        B, N = batch.batch_size, batch.num_nodes
        n_starts = min(n_starts, N - 1)
        customers = torch.arange(1, N, device=device).expand(B, N - 1)
        perm = torch.argsort(torch.rand(B, N - 1, device=device), dim=-1)
        starts = customers.gather(1, perm)[:, :n_starts]

        def tile(t: torch.Tensor) -> torch.Tensor:
            return t.repeat_interleave(n_starts, dim=0)

        tiled = BatchedInstances(
            locations=tile(batch.locations),
            demands=tile(batch.demands),
            distance=tile(batch.distance),
            co2_per_arc=tile(batch.co2_per_arc),
            edge_features=tile(batch.edge_features),
            capacity=tile(batch.capacity),
            depot_index=tile(batch.depot_index),
        )
        first_actions = starts.reshape(-1)
        return self._rollout(tiled, greedy=False, start_action=first_actions)
