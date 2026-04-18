from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

from core.interfaces import ModelState, ScoringHead
from core.events import EventBatch

class MLPNodePredictor(nn.Module):
    """
    Predict one vector per node from the current latent node state.

    Input:
        state.node: [N, node_dim]

    Output:
        pred: [N, out_dim]
    """

    def __init__(
        self,
        node_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.node_dim = int(node_dim)
        self.out_dim = int(out_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.out_dim),
        )

    def forward(self, state: Optional[ModelState]) -> torch.Tensor:
        assert state is not None and state.node is not None, \
            "MLPNodePredictor requires state.node"
        return self.mlp(state.node)
    
    
class LinearNodePredictor(nn.Module):
    def __init__(self, node_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(node_dim, out_dim)

    def forward(self, state: Optional[ModelState]) -> torch.Tensor:
        assert state is not None and state.node is not None, \
            "LinearNodePredictor requires state.node"
        return self.linear(state.node)