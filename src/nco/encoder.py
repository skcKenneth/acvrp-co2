"""
nco/encoder.py
==============

Bidirectional edge-attention encoder for asymmetric ACVRP.

The standard Attention Model (Kool, Hoof, Welling, 2019) and POMO
(Kwon et al., 2020) encode each customer purely from its 2D
coordinates. That makes geometric sense but cannot handle a problem
where the relationship between two customers depends on which way you
drive between them.

We adapt the dual-graph idea from MatNet (Kwon et al., 2021) to ACVRP:
for every node i we form two context vectors,

    h_i^{in}  = attention over all j != i using arc (j -> i) features
    h_i^{out} = attention over all j != i using arc (i -> j) features

and concatenate them with the node-local features (location, demand)
before projecting to the final embedding dimension.

This is the *minimum* architectural change required for the encoder to
distinguish d_ij from d_ji. Heavier alternatives (full transformer
over all O(n^2) edges) are computationally expensive and offer
diminishing returns at the problem sizes we care about (N <= 100).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _multi_head(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """(B, N, H*D) -> (B, H, N, D)"""
    B, N, HD = x.shape
    D = HD // n_heads
    return x.view(B, N, n_heads, D).permute(0, 2, 1, 3).contiguous()


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    """(B, H, N, D) -> (B, N, H*D)"""
    B, H, N, D = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(B, N, H * D)


class EdgeAwareAttention(nn.Module):
    """
    Single multi-head attention layer where the score between query i
    and key j is computed from the standard Q.K^T term *plus* a
    bias derived from the edge features e_ij.

    score(i, j) = (Q_i K_j^T) / sqrt(d_k) + W_e e_ij
    """

    def __init__(self, embed_dim: int, n_heads: int, edge_dim: int):
        super().__init__()
        if embed_dim % n_heads != 0:
            raise ValueError("embed_dim must be divisible by n_heads")
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        # One scalar bias per head, projected from the edge-feature vector
        self.W_e = nn.Linear(edge_dim, n_heads, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        nodes: torch.Tensor,          # (B, N, D)
        edge_features: torch.Tensor,  # (B, N, N, F) where edge_features[i, j] is the i->j arc
    ) -> torch.Tensor:                # (B, N, D)
        Q = _multi_head(self.W_q(nodes), self.n_heads)   # (B, H, N, d_k)
        K = _multi_head(self.W_k(nodes), self.n_heads)
        V = _multi_head(self.W_v(nodes), self.n_heads)

        # (B, H, N_i, N_j)  --  attention scores per head
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Edge bias: (B, N_i, N_j, F) -> (B, H, N_i, N_j)
        edge_bias = self.W_e(edge_features).permute(0, 3, 1, 2)
        scores = scores + edge_bias

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)         # (B, H, N, d_k)
        out = _merge_heads(out)             # (B, N, D)
        return self.W_o(out)


class EncoderBlock(nn.Module):
    """One transformer-style block: edge-aware attention + FFN + skip."""

    def __init__(self, embed_dim: int, n_heads: int, edge_dim: int, ffn_dim: int):
        super().__init__()
        self.attn = EdgeAwareAttention(embed_dim, n_heads, edge_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        nodes: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        nodes = self.norm1(nodes + self.attn(nodes, edge_features))
        nodes = self.norm2(nodes + self.ffn(nodes))
        return nodes


class BidirectionalEncoder(nn.Module):
    """
    Full encoder that processes the graph in both arc directions.

    Forward path: a tower of EncoderBlocks on the "outgoing" view
    (edge_features[i, j] = arc i -> j) and a parallel tower on the
    transposed "incoming" view, then a fusion MLP that combines both
    representations with the local node features.

    Inputs
    ------
    node_features : (B, N, F_node)    e.g. [x, y, demand_norm]
    edge_features : (B, N, N, F_edge) e.g. [d_norm, t_norm, fuel_norm, co2_norm]
    """

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        embed_dim: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        ffn_dim: int = 256,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.node_proj = nn.Linear(node_feature_dim, embed_dim)
        # We re-use the same parameter set for both directions; the
        # encoder sees a different edge-feature tensor (transposed in
        # the "incoming" pass), which is what makes the two views
        # different.
        self.blocks = nn.ModuleList([
            EncoderBlock(embed_dim, n_heads, edge_feature_dim, ffn_dim)
            for _ in range(n_layers)
        ])
        self.fuse = nn.Sequential(
            nn.Linear(2 * embed_dim + node_feature_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        # Outgoing pass: arc i -> j; edge_features[b, i, j, :] is the i->j arc
        out = self.node_proj(node_features)
        for block in self.blocks:
            out = block(out, edge_features)

        # Incoming pass: arc j -> i; we transpose the first two arc dims
        in_edge = edge_features.transpose(1, 2).contiguous()
        inp = self.node_proj(node_features)
        for block in self.blocks:
            inp = block(inp, in_edge)

        # Fuse the two embeddings with the raw node features.
        fused = self.fuse(torch.cat([out, inp, node_features], dim=-1))
        return fused
