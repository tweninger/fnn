from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from interactiondynamics.core.events import EventBatch
from interactiondynamics.core.interfaces import Aggregator, ModelState


class HopfieldAggregator(Aggregator):
    """
    Hopfield-style associative retrieval aggregator.

    For each node v:
      - build a padded set of incident event embeddings X_v: [K, d_msg]
      - query q_v from state.node[v]
      - retrieve m_v = softmax(beta * qK^T) V

    Returns:
      messages: [N, d_msg]
    """

    def __init__(
        self,
        node_dim: int,
        msg_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        beta: float = 1.0,
        steps: int = 1,
        max_events_per_node: int = 32,
        add_to_dst: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.node_dim = int(node_dim)
        self.msg_dim = int(msg_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.beta = float(beta)
        self.steps = int(steps)
        self.max_events_per_node = int(max_events_per_node)
        self.add_to_dst = bool(add_to_dst)

        # Projections
        self.q_proj = nn.Linear(self.node_dim, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(self.msg_dim, self.hidden_dim, bias=False)
        self.v_proj = nn.Linear(self.msg_dim, self.hidden_dim, bias=False)
        self.out_proj = nn.Linear(self.hidden_dim, self.msg_dim, bias=False)

        self.drop = nn.Dropout(dropout)

    def _build_padded_sets(
        self,
        event_embeddings: torch.Tensor,  # [M, d_msg]
        events: EventBatch,
        num_nodes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build X: [N, K, d_msg] and mask: [N, K] for incident events per node.

        Keeps the MOST RECENT K events per node (by within-bin order),
        using a vectorized group-by on sorted node ids.
        """
        device = event_embeddings.device
        dtype = event_embeddings.dtype
        M, d = event_embeddings.shape
        K = self.max_events_per_node

        src = events.src.to(device=device, dtype=torch.long)
        idx = torch.arange(M, device=device, dtype=torch.long)

        if self.add_to_dst:
            dst = events.dst.to(device=device, dtype=torch.long)
            nodes = torch.cat([src, dst], dim=0)  # [2M]
            eidx  = torch.cat([idx, idx], dim=0)  # [2M] (event id within bin)
        else:
            nodes = src
            eidx  = idx

        # Group by node id
        perm = torch.argsort(nodes * max(M, 1) + eidx, stable=True)
        nodes_s = nodes[perm]
        eidx_s  = eidx[perm]

        # Identify group boundaries (each group is a node)
        group_start = torch.ones_like(nodes_s, dtype=torch.bool)
        group_start[1:] = nodes_s[1:] != nodes_s[:-1]
        gid = torch.cumsum(group_start.to(torch.int64), dim=0) - 1  # [len(nodes_s)]
        num_groups = int(gid.max().item() + 1) if nodes_s.numel() > 0 else 0

        # ar = position within nodes_s
        ar = torch.arange(nodes_s.numel(), device=device, dtype=torch.int64)

        # first index of each group (node) in nodes_s
        first = torch.zeros(num_groups, device=device, dtype=torch.int64)
        first.scatter_(0, gid[group_start], ar[group_start])

        # count per group
        ones = torch.ones_like(gid, dtype=torch.int64)
        count = torch.zeros(num_groups, device=device, dtype=torch.int64)
        count.scatter_add_(0, gid, ones)

        # last index per group
        last = first + count - 1  # [num_groups]

        # position-from-end: 0 means most recent (last), 1 means second last, ...
        pos_from_end = last[gid] - ar  # [len(nodes_s)]

        keep = pos_from_end < K
        if not torch.any(keep):
            X = torch.zeros((num_nodes, K, d), device=device, dtype=dtype)
            mask = torch.zeros((num_nodes, K), device=device, dtype=torch.bool)
            return X, mask

        nodes_k = nodes_s[keep]               # [E_kept]
        eidx_k  = eidx_s[keep]                # [E_kept]
        # map to slots 0..K-1 with oldest->0, newest->K-1 (nice for debugging)
        slot_k = (K - 1 - pos_from_end[keep]).to(torch.long)  # [E_kept]

        X = torch.zeros((num_nodes, K, d), device=device, dtype=dtype)
        mask = torch.zeros((num_nodes, K), device=device, dtype=torch.bool)
        # Keep this gather/scatter differentiable so the event encoder trains.
        X[nodes_k, slot_k] = event_embeddings[eidx_k]
        mask[nodes_k, slot_k] = True
        return X, mask

    def forward(
        self,
        state: Optional[ModelState],
        event_embeddings: torch.Tensor,   # [M, d_msg]
        events: EventBatch,
        num_nodes: int,
    ) -> torch.Tensor:
        assert state is not None and state.node is not None, "HopfieldAggregator requires state.node"
        H = state.node  # [N, node_dim]
        device = event_embeddings.device
        dtype = event_embeddings.dtype

        X, mask = self._build_padded_sets(event_embeddings, events, num_nodes)  # [N,K,d], [N,K]
        active = mask.any(dim=1)  # [N]

        out = torch.zeros((num_nodes, self.msg_dim), device=device, dtype=dtype)
        if not torch.any(active):
            return out

        # Active-only compute
        X_a = X[active]            # [A,K,d_msg]
        mask_a = mask[active]      # [A,K]
        H_a = H[active].to(device=device, dtype=dtype)  # [A,node_dim]

        # Project to multi-head Q,K,V
        # Q: [A, heads, 1, head_dim]
        q = self.q_proj(H_a).view(-1, self.num_heads, self.head_dim).unsqueeze(2)

        # K,V: [A, heads, K, head_dim]
        k = self.k_proj(X_a).view(-1, X_a.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(X_a).view(-1, X_a.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        # Mask: True for valid positions -> we want to mask invalid
        attn_mask = ~mask_a  # [A,K]
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # [A,1,1,K] broadcast over heads

        # Retrieval steps (usually 1 is enough)
        for _ in range(max(1, self.steps)):
            # q: [A,H,1,D], k: [A,H,K,D]  -> scores: [A,H,1,K]
            scale = (self.head_dim ** -0.5)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [A,H,1,K]
            scores = scores * self.beta  # optional; set beta=1.0 to start

            scores = scores.masked_fill(attn_mask, float("-inf"))

            w = torch.softmax(scores, dim=-1)  # [A,heads,1,K]
            w = self.drop(w)

            # retrieved: [A,heads,1,head_dim]
            r = torch.matmul(w, v)  # [A,heads,1,head_dim]

            q = r  # Refine the retrieval query, keeping stored patterns fixed.

        r = r.squeeze(2)  # [A,heads,head_dim]
        r = r.reshape(-1, self.hidden_dim)  # [A, hidden_dim]
        r = self.out_proj(r)  # [A, msg_dim]

        out[active] = r
        return out
