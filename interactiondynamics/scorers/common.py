from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from interactiondynamics.core.interfaces import ModelState


def require_node_state(
    state: Optional[ModelState],
    *,
    who: str,
) -> torch.Tensor:
    assert state is not None and state.node is not None, f"{who} requires state.node."
    return state.node


def build_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    dropout: float,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class ScalarTimeEncoder(nn.Module):
    def __init__(self, time_emb_dim: int):
        super().__init__()
        self.time_emb_dim = int(time_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(1, self.time_emb_dim),
            nn.ReLU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


def gather_node_features(
    node_features: Optional[nn.Module],
    device: torch.device,
    *index_tensors: torch.Tensor,
) -> list[torch.Tensor]:
    if node_features is None:
        return []
    z = node_features(device)
    return [z[idx] for idx in index_tensors]
