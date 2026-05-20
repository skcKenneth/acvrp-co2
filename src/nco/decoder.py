"""
nco/decoder.py
==============

Capacity-aware autoregressive decoder for ACVRP.

State maintained across decoding steps:
    - current_node      (B,)         index of the vehicle's location
    - remaining_cap     (B,)         current vehicle capacity (resets at depot)
    - visited           (B, N) bool  which customers have been served

At each step:
    1. Build a "context" query from the current node embedding,
       remaining capacity, and a global graph embedding.
    2. Compute attention scores against every node embedding.
    3. Mask out nodes that are visited, or whose demand exceeds the
       remaining capacity. The depot is always selectable (vehicle
       returns to depot to recharge).
    4. Apply a temperature-scaled tanh "clipping" before the softmax to
       stabilise training, as in Bello et al. 2017 / Kool et al. 2019.
    5. Sample or argmax to pick the next node; update state.

A trajectory ends when all customers are visited and the vehicle
returns to the depot.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CapacityAwareDecoder(nn.Module):
    """
    Pointer-style decoder with capacity masking.

    Implementation notes
    --------------------
    - We compute keys/values from the node embeddings once per
      decoding step (they don't change), but the query depends on
      state, so it's rebuilt every step.
    - Multi-head intermediate attention followed by a single-head
      pointer attention is the standard Attention-Model recipe.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 8,
        tanh_clip: float = 10.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.tanh_clip = tanh_clip

        # Context = [graph_mean, current_node_embed, remaining_cap_scalar]
        self.W_context = nn.Linear(2 * embed_dim + 1, embed_dim, bias=False)

        # Multi-head intermediate attention (Q from context, K/V from nodes)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

        # Single-head pointer attention
        self.W_q_ptr = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k_ptr = nn.Linear(embed_dim, embed_dim, bias=False)

    def _intermediate_attention(
        self,
        context: torch.Tensor,         # (B, D)
        node_embeds: torch.Tensor,     # (B, N, D)
        mask: torch.Tensor,            # (B, N) bool; True == valid
    ) -> torch.Tensor:                 # (B, D)
        B, N, D = node_embeds.shape
        H = self.n_heads
        d_k = self.head_dim

        Q = self.W_q(context).view(B, H, 1, d_k)
        K = self.W_k(node_embeds).view(B, N, H, d_k).permute(0, 2, 1, 3)
        V = self.W_v(node_embeds).view(B, N, H, d_k).permute(0, 2, 1, 3)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (B, H, 1, N)
        scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)               # (B, H, 1, d_k)
        out = out.permute(0, 2, 1, 3).reshape(B, 1, H * d_k).squeeze(1)
        return self.W_o(out)

    def _pointer_logits(
        self,
        refined_context: torch.Tensor,   # (B, D)
        node_embeds: torch.Tensor,       # (B, N, D)
        mask: torch.Tensor,              # (B, N) bool; True == valid
    ) -> torch.Tensor:                   # (B, N)
        Q = self.W_q_ptr(refined_context).unsqueeze(1)   # (B, 1, D)
        K = self.W_k_ptr(node_embeds)                    # (B, N, D)
        scores = torch.matmul(Q, K.transpose(-2, -1)).squeeze(1) / math.sqrt(self.embed_dim)
        scores = self.tanh_clip * torch.tanh(scores)
        scores = scores.masked_fill(~mask, float("-inf"))
        return scores

    def step(
        self,
        node_embeds: torch.Tensor,        # (B, N, D)
        graph_embed: torch.Tensor,        # (B, D)
        current_node: torch.Tensor,       # (B,) long
        remaining_cap: torch.Tensor,      # (B,) float
        capacity: torch.Tensor,           # (B,) float -- for normalisation
        mask: torch.Tensor,               # (B, N) bool; True == valid
    ) -> torch.Tensor:                    # (B, N) logits
        """
        Compute the logits over next nodes given the current state.
        """
        B, _, D = node_embeds.shape
        cur_embed = node_embeds.gather(
            1, current_node.view(B, 1, 1).expand(-1, 1, D)
        ).squeeze(1)
        cap_scalar = (remaining_cap.float() / capacity.float()).unsqueeze(-1)
        context = torch.cat([graph_embed, cur_embed, cap_scalar], dim=-1)
        context = self.W_context(context)
        refined = self._intermediate_attention(context, node_embeds, mask)
        return self._pointer_logits(refined, node_embeds, mask)


def make_mask(
    visited: torch.Tensor,        # (B, N) bool
    demands: torch.Tensor,        # (B, N) long
    remaining_cap: torch.Tensor,  # (B,) long
    current_node: torch.Tensor,   # (B,) long
    depot_index: torch.Tensor,    # (B,) long
) -> torch.Tensor:                # (B, N) bool; True == selectable
    """
    Build the action mask for the next decoding step.

    Selectable nodes are:
      * customers that are not yet visited and whose demand fits
        in the remaining capacity, OR
      * the depot, but only if the vehicle is not already at it.

    If no customer is selectable, we always allow the depot so the
    rollout can terminate cleanly.
    """
    B, N = visited.shape

    not_visited = ~visited
    fits = demands <= remaining_cap.unsqueeze(-1)

    # Depot column = arange == depot_index (per batch element)
    arange = torch.arange(N, device=visited.device).unsqueeze(0).expand(B, N)
    is_depot = arange == depot_index.unsqueeze(-1)

    # Customers are selectable iff unvisited AND demand fits
    cust_ok = not_visited & fits & ~is_depot

    # Depot is selectable iff the vehicle is not currently at it.
    at_depot = (current_node == depot_index)
    depot_ok = is_depot & ~at_depot.unsqueeze(-1)

    mask = cust_ok | depot_ok

    # Failsafe: if no node is selectable for some batch element (every
    # customer's demand exceeds remaining capacity AND the vehicle is at
    # the depot, which can happen with very tight capacity), force the
    # depot back on so the search doesn't crash.
    no_choice = ~mask.any(dim=-1)
    if no_choice.any():
        force_depot = arange == depot_index.unsqueeze(-1)
        mask = mask | (force_depot & no_choice.unsqueeze(-1))
    return mask
