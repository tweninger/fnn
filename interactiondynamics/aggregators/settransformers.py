from __future__ import annotations
from dataclasses import dataclass
import time
from typing import Optional

import torch
import torch.nn as nn

from interactiondynamics.core.events import EventBatch
from interactiondynamics.core.interfaces import Aggregator, ModelState


class MAB(nn.Module):
    """
    Multihead Attention Block: LayerNorm + MHA + residual + FF + residual
    Q attends to K/V.
    """
    def __init__(self, dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        # Q: [B, nQ, d], K: [B, nK, d]
        x = self.ln1(Q)
        k = self.ln1(K)
        h, _ = self.attn(x, k, k, need_weights=False)
        Q = Q + self.dropout(h)

        y = self.ln2(Q)
        Q = Q + self.dropout(self.ff(y))
        return Q


class PMA(nn.Module):
    """
    Pooling by Multihead Attention with learnable seed vectors.
    """
    def __init__(self, dim: int, num_heads: int, ff_dim: int, num_seeds: int = 1, dropout: float = 0.0):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, num_seeds, dim) * 0.02)
        self.mab = MAB(dim=dim, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: [B, K, d]
        B = X.size(0)
        S = self.seed.expand(B, -1, -1)  # [B, num_seeds, d]
        return self.mab(S, X)            # [B, num_seeds, d]


class SetTransformerAggregator(Aggregator):
    """
    Per-node set attention aggregator (binned):
      - gather up to max_events_per_node incident events for each node
      - run (optional) self-attention layers
      - pool via PMA (1 seed) -> [N, d]
    """

    def __init__(
        self,
        msg_dim: int,
        num_heads: int = 4,
        ff_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        add_to_dst: bool = True,
        max_events_per_node: int = 32,
    ):
        super().__init__()
        self.msg_dim = int(msg_dim)
        self.add_to_dst = bool(add_to_dst)
        self.max_events_per_node = int(max_events_per_node)

        self.self_blocks = nn.ModuleList(
            [MAB(dim=self.msg_dim, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.pma = PMA(dim=self.msg_dim, num_heads=num_heads, ff_dim=ff_dim, num_seeds=1, dropout=dropout)

    @torch.no_grad()
    def _build_padded_sets(
        self,
        event_embeddings: torch.Tensor,  # [M, d]
        events: EventBatch,
        num_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = event_embeddings.device
        dtype = event_embeddings.dtype
        M, d = event_embeddings.shape
        K = self.max_events_per_node

        # Collect endpoints (src, and maybe dst) as a single list
        src = events.src.to(device=device, dtype=torch.long)
        idx = torch.arange(M, device=device, dtype=torch.long)

        if self.add_to_dst:
            dst = events.dst.to(device=device, dtype=torch.long)
            nodes = torch.cat([src, dst], dim=0)          # [2M]
            eidx  = torch.cat([idx, idx], dim=0)          # [2M]
        else:
            nodes = src                                   # [M]
            eidx  = idx

        # Compute per-node occurrence number (slot) for each endpoint.
        # slot = cumsum(one_hot(node)) - 1, but we do it with scatter_add on a counter.
        # We simulate a running counter by sorting by node, then using arange within group.

        # Sort endpoints by node
        perm = torch.argsort(nodes)
        nodes_s = nodes[perm]
        eidx_s  = eidx[perm]

        # Compute slot within each node group: 0,1,2,... for consecutive equal nodes
        # group_start is True where node changes
        group_start = torch.ones_like(nodes_s, dtype=torch.bool)
        group_start[1:] = nodes_s[1:] != nodes_s[:-1]
        # group ids via cumulative sum of starts
        gid = torch.cumsum(group_start.to(torch.int64), dim=0) - 1  # [E]
        # position within group = arange - first_index_of_group
        ar = torch.arange(nodes_s.numel(), device=device, dtype=torch.int64)
        n_groups = int(gid.max().item()) + 1 if gid.numel() > 0 else 0
        first = torch.zeros((n_groups,), device=device, dtype=torch.int64)        
        first.scatter_(0, gid[group_start], ar[group_start])
        slot = ar - first[gid]  # 0..deg-1 per node

        # Keep only first K per node
        keep = slot < K
        nodes_k = nodes_s[keep]                 # [E']
        slot_k  = slot[keep].to(torch.long)     # [E']
        eidx_k  = eidx_s[keep]                  # [E']

        # Build padded [N,K,d] and mask [N,K]
        X = torch.zeros((num_nodes, K, d), device=device, dtype=dtype)
        mask = torch.zeros((num_nodes, K), device=device, dtype=torch.bool)

        X[nodes_k, slot_k] = event_embeddings[eidx_k]
        mask[nodes_k, slot_k] = True
        return X, mask


    def forward(
        self,
        state: ModelState | None,
        event_embeddings: torch.Tensor,   # [M, d]
        events: EventBatch,
        num_nodes: int,
    ) -> torch.Tensor:
                
        # Build padded per-node sets
        X, mask = self._build_padded_sets(event_embeddings, events, num_nodes)  # X: [N,K,d]

        # Zero-out padded positions explicitly (already zero) and run self-attn blocks.
        # NOTE: nn.MultiheadAttention supports key_padding_mask, but we kept MAB simple.
        # Because padded positions are zeros and LN/FF can leak, we'll re-mask after blocks.
        # X: [N,K,d], mask: [N,K]
        active = mask.any(dim=1)               # [N] active gating
        if not torch.any(active):
            return torch.zeros((num_nodes, self.msg_dim), device=X.device, dtype=X.dtype)
            

        X_a = X[active]                        # [A,K,d]
        mask_a = mask[active]                  # [A,K]


        for blk in self.self_blocks:
            X_a = blk(X_a, X_a)
            X_a = X_a * mask_a.unsqueeze(-1)

        pooled_a = self.pma(X_a).squeeze(1)    # [A,d]           

        out = torch.zeros((num_nodes, self.msg_dim), device=X.device, dtype=X.dtype)
        out[active] = pooled_a
        return out  
