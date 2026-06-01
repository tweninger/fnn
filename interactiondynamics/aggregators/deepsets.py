import torch
from typing import Optional
import torch.nn as nn
from interactiondynamics.core.events import EventBatch
from interactiondynamics.core.interfaces import Aggregator, ModelState

class DeepSetsAggregator(Aggregator):
    """
    DeepSets aggregation of per-event embeddings into per-node messages:
        m_v = rho( sum_{e incident to v} phi(x_e) )

    Notes:
      - "incident" here can mean src-only or src+dst.
      - Output dim can differ from msg_dim if you want (out_dim).
    """

    def __init__(
        self,
        msg_dim: int,
        out_dim: Optional[int] = None,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        add_to_dst: bool = True,
        reduce: str = "sum",  # or "mean"
    ):
        super().__init__()
        self.msg_dim = int(msg_dim)
        self.out_dim = int(out_dim) if out_dim is not None else int(msg_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.add_to_dst = bool(add_to_dst)

        if reduce not in ("sum", "mean"):
            raise ValueError(f"reduce must be 'sum' or 'mean', got {reduce}")
        self.reduce = reduce

        # phi: per-event transform
        self.phi = nn.Sequential(
            nn.Linear(self.msg_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.out_dim),
        )

        # rho: post-pooling transform
        self.rho = nn.Sequential(
            nn.Linear(self.out_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.out_dim),
        )

    def forward(
        self,
        state: ModelState | None,
        event_embeddings: torch.Tensor,   # [M, msg_dim]
        events: EventBatch,
        num_nodes: int,
    ) -> torch.Tensor:
        device = event_embeddings.device
        dtype = event_embeddings.dtype

        # 1) phi(x_e)
        z = self.phi(event_embeddings)  # [M, out_dim]

        # 2) scatter sum to nodes
        messages = torch.zeros((num_nodes, self.out_dim), device=device, dtype=dtype)

        src = events.src.to(device=device, dtype=torch.long)
        messages.index_add_(0, src, z)

        if self.add_to_dst:
            dst = events.dst.to(device=device, dtype=torch.long)
            messages.index_add_(0, dst, z)

        if self.reduce == "mean":
            counts = torch.zeros((num_nodes,), device=device, dtype=dtype)
            ones = torch.ones((z.size(0),), device=device, dtype=dtype)
            counts.index_add_(0, src, ones)
            if self.add_to_dst:
                counts.index_add_(0, dst, ones)
            messages = messages / counts.clamp_min(1.0).unsqueeze(-1)

        # 3) rho( pooled )
        return self.rho(messages)  # [N, out_dim]

