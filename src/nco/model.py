"""
nco/model.py
============

Glues the encoder and decoder into a complete ACVRP policy.

Two inference modes are supported:

    sample(): for training, stochastic rollouts producing
              log-probabilities to feed REINFORCE.
    greedy(): for evaluation, argmax decoding.

The model also exposes a `pomo_sample(n_starts)` convenience method
that returns N parallel rollouts starting from N different first
customers. Their mean cost serves as a low-variance baseline that
works on asymmetric data, unlike the rotation-based instance
augmentation used in vanilla POMO for Euclidean problems.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .decoder import CapacityAwareDecoder, make_mask
from .encoder import BidirectionalEncoder
from .instance import BatchedInstances


@dataclass
class Rollout:
    """
    Container for the output of a rollout (one route per batch element).

    `cost` is the scalar that REINFORCE optimises — its definition is
    controlled by the policy's `cost_mode` setting and may be a weighted
    combination of distance and CO2. The raw per-arc components are also
    exposed so downstream evaluation can compute distance / fuel / CO2
    separately without re-running the model.
    """

    actions: torch.Tensor          # (B, T) long
    log_probs: torch.Tensor        # (B,) sum log-prob of the trajectory
    cost: torch.Tensor             # (B,) scalar used as REINFORCE return
    distance_m: torch.Tensor       # (B,) tour distance in metres
    co2_kg: torch.Tensor           # (B,) tour CO2 emissions in kilograms


class ACVRPPolicy(nn.Module):
    """End-to-end ACVRP solver."""

    def __init__(
        self,
        node_feature_dim: int = 3,    # (x, y, demand_norm)
        edge_feature_dim: int = 4,    # (dist, time, fuel, co2) normalised
        embed_dim: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        ffn_dim: int = 256,
        tanh_clip: float = 10.0,
        cost_mode: str = "distance",  # "distance" | "co2" | "blend"
        co2_weight: float = 0.5,      # used only when cost_mode == "blend"
        co2_scale: float = 1000.0,    # multiplies kg CO2 in the blended cost
    ):
        super().__init__()
        if cost_mode not in {"distance", "co2", "blend"}:
            raise ValueError(
                f"cost_mode must be 'distance', 'co2', or 'blend'; got {cost_mode!r}."
            )
        # Store all constructor args as attributes so that save_checkpoint
        # in trainer.py can recover them and rebuild the same architecture
        # at load time. This avoids the silent size-mismatch failure that
        # occurs when defaults change between training and inference.
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ffn_dim = ffn_dim
        self.tanh_clip = tanh_clip
        self.cost_mode = cost_mode
        self.co2_weight = float(co2_weight)
        self.co2_scale = float(co2_scale)

        self.encoder = BidirectionalEncoder(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
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

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    @staticmethod
    def _node_features(batch: BatchedInstances) -> torch.Tensor:
        """Build (B, N, 3) node-feature tensor from the batch."""
        locs = batch.locations
        # Normalise lat/lon to [0, 1] per instance so the model is
        # geographically agnostic.
        mins = locs.min(dim=1, keepdim=True).values
        maxs = locs.max(dim=1, keepdim=True).values
        scale = (maxs - mins).clamp_min(1e-6)
        loc_norm = (locs - mins) / scale  # (B, N, 2)

        dem_norm = (
            batch.demands.float() / batch.capacity.float().unsqueeze(-1)
        ).unsqueeze(-1)  # (B, N, 1)
        return torch.cat([loc_norm, dem_norm], dim=-1)

    # ------------------------------------------------------------------
    # Rollout core
    # ------------------------------------------------------------------

    def _rollout(
        self,
        batch: BatchedInstances,
        greedy: bool,
        start_action: Optional[torch.Tensor] = None,
    ) -> Rollout:
        """
        Run a single rollout per batch element.

        Parameters
        ----------
        batch        : BatchedInstances
        greedy       : if True, take argmax at each step; else sample.
        start_action : (B,) long, optional. If provided, used as the
                       first action *after* the depot. This is how POMO
                       diversifies its rollouts.
        """
        device = batch.locations.device
        B = batch.batch_size
        N = batch.num_nodes

        node_feats = self._node_features(batch)
        node_embeds = self.encoder(node_feats, batch.edge_features)  # (B, N, D)
        graph_embed = node_embeds.mean(dim=1)                        # (B, D)

        # --- State -----------------------------------------------------
        current = batch.depot_index.clone()                # (B,)
        visited = torch.zeros(B, N, dtype=torch.bool, device=device)
        # Mark depot as "visited" so the customer-mask doesn't pick it.
        # The depot is re-enabled through the explicit `is_depot` term
        # in make_mask().
        visited.scatter_(1, batch.depot_index.unsqueeze(-1), True)
        remaining = batch.capacity.clone().long()

        distance_acc = torch.zeros(B, device=device)
        co2_acc = torch.zeros(B, device=device)
        log_prob_sum = torch.zeros(B, device=device)
        actions = [current.clone()]

        # Absolute per-arc CO2 in kg, as stored on the batch by
        # collate_instances. This is the true emissions tensor; the
        # normalised edge-feature channels are inputs to the encoder
        # only and must NOT be used for reward accumulation.
        co2_per_arc_kg = batch.co2_per_arc                 # (B, N, N) kg

        # Max horizon: each customer visit + up to (n-1) returns to
        # depot for refills. 2N is a safe upper bound.
        max_steps = 2 * N
        forced_first = start_action

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

            if step == 0 and forced_first is not None:
                # The first action under POMO is forced to a randomly-chosen
                # customer; it is an *exploration* device, not a policy
                # decision. Following Kwon et al. 2020 we exclude its
                # gradient contribution by detaching its log-probability.
                next_node = forced_first
                log_probs = torch.log_softmax(logits, dim=-1)
                step_lp = log_probs.gather(1, next_node.unsqueeze(-1)).squeeze(-1).detach()
            else:
                if greedy:
                    next_node = logits.argmax(dim=-1)
                else:
                    probs = torch.softmax(logits, dim=-1)
                    next_node = torch.multinomial(probs, num_samples=1).squeeze(-1)
                log_probs = torch.log_softmax(logits, dim=-1)
                step_lp = log_probs.gather(1, next_node.unsqueeze(-1)).squeeze(-1)

            # Accumulate distance and CO2 under the ground-truth matrices.
            d_row = batch.distance.gather(
                1, current.view(B, 1, 1).expand(B, 1, N)
            ).squeeze(1)                                            # (B, N)
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
            # Subtract demand if we visited a customer; reset capacity at depot.
            served = next_node != batch.depot_index
            picked_demand = batch.demands.gather(1, next_node.unsqueeze(-1)).squeeze(-1)
            remaining = torch.where(
                served, remaining - picked_demand, batch.capacity.clone().long()
            )
            current = next_node
            actions.append(current.clone())

            # Termination: all customers visited and vehicle back at depot.
            # We exclude *only* the depot column when checking; this
            # works regardless of where the depot sits in the index.
            arange_n = torch.arange(N, device=device)
            not_depot_col = arange_n.unsqueeze(0) != batch.depot_index.unsqueeze(-1)
            customers_done = (visited | ~not_depot_col).all(dim=-1)
            at_depot = current == batch.depot_index
            if (customers_done & at_depot).all():
                break

        # ---- Compose the scalar `cost` according to cost_mode ----------
        if self.cost_mode == "distance":
            cost = distance_acc
        elif self.cost_mode == "co2":
            cost = co2_acc * self.co2_scale
        else:  # "blend"
            cost = (1.0 - self.co2_weight) * distance_acc \
                 + self.co2_weight * self.co2_scale * co2_acc

        return Rollout(
            actions=torch.stack(actions, dim=1),
            log_probs=log_prob_sum,
            cost=cost,
            distance_m=distance_acc,
            co2_kg=co2_acc,
        )

    # ------------------------------------------------------------------
    # Public inference modes
    # ------------------------------------------------------------------

    def sample(self, batch: BatchedInstances) -> Rollout:
        """Stochastic rollout (used in training)."""
        return self._rollout(batch, greedy=False)

    def greedy(self, batch: BatchedInstances) -> Rollout:
        """Greedy/argmax rollout (used in evaluation)."""
        return self._rollout(batch, greedy=True)

    def pomo_sample(
        self,
        batch: BatchedInstances,
        n_starts: int,
    ) -> Rollout:
        """
        POMO-style multi-start sampling: run `n_starts` parallel
        rollouts per instance, each starting from a different customer.

        Returns a flattened rollout of shape (B * n_starts, ...). The
        first action in each is the chosen start customer; subsequent
        actions follow the policy stochastically.
        """
        device = batch.locations.device
        B, N = batch.batch_size, batch.num_nodes

        n_starts = min(n_starts, N - 1)
        # Pick n_starts distinct customer indices per instance.
        customers = torch.arange(1, N, device=device).expand(B, N - 1)
        # Shuffle each row then take first n_starts entries.
        perm = torch.argsort(torch.rand(B, N - 1, device=device), dim=-1)
        starts = customers.gather(1, perm)[:, :n_starts]    # (B, n_starts)

        # Tile the batch so we can launch all starts in parallel
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
