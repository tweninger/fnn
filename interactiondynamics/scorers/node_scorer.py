from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from interactiondynamics.core.interfaces import ModelState, NodeScoringHead
from interactiondynamics.scorers.common import build_mlp, gather_node_features, require_node_state


class MLPNodeScorer(NodeScoringHead):
    """
    score(i) = MLP([h_i, (Lh)_i, z_i])

    If a graph operator `L` is present in `state.aux["L"]`, the head also reads
    the local operator response `(Lh)_i`. Otherwise it falls back to zeros.
    Returns one score per node.
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        node_features: Optional[nn.Module] = None,
        node_feature_dim: int = 0,
        out_dim: int = 1,
    ):
        super().__init__()
        self.node_dim = int(node_dim)
        self.node_features = node_features
        self.node_feature_dim = int(node_feature_dim)
        self.out_dim = int(out_dim)

        in_dim = (2 * self.node_dim) + self.node_feature_dim
        self.mlp = build_mlp(in_dim, hidden_dim, out_dim=self.out_dim, dropout=dropout)

    def forward(self, state: Optional[ModelState]) -> torch.Tensor:
        h = require_node_state(state, who="MLPNodeScorer")
        if state is not None and state.aux is not None:
            operator = state.aux.get("L", None)
        else:
            operator = None

        if operator is not None:
            l_h = torch.sparse.mm(operator, h)
        else:
            l_h = torch.zeros_like(h)

        pieces = [h, l_h]

        node_idx = torch.arange(h.size(0), device=h.device, dtype=torch.long)
        pieces.extend(gather_node_features(self.node_features, h.device, node_idx))

        x = torch.cat(pieces, dim=-1)
        out = self.mlp(x)
        if self.out_dim == 1:
            return out.squeeze(-1)
        return out
