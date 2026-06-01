from typing import Optional
import torch
import torch.nn as nn

from interactiondynamics.core.events import EventBatch
from interactiondynamics.core.interfaces import ModelState, ScoringHead


class DotProductScorer(ScoringHead):
    """
    score(u, v) = <h_u, h_v>
    """

    def forward(self, state: Optional[ModelState], candidate_events: EventBatch) -> torch.Tensor:
        assert state is not None and state.node is not None, "DotProductScorer requires state.node."

        H = state.node  # (N, d)
        src = candidate_events.src.to(device=H.device, dtype=torch.long)
        dst = candidate_events.dst.to(device=H.device, dtype=torch.long)

        h_src = H[src]  # (M, d)
        h_dst = H[dst]  # (M, d)
        return (h_src * h_dst).sum(dim=-1)  # (M,)

class MLPEdgeScorer(ScoringHead):
    """
    score(u,v,e,t) = MLP([h_u, h_v, e, phi(t)])
    """

    def __init__(
        self,
        node_dim: int,
        event_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_time: bool = False,
        time_emb_dim: int = 32,
    ):
        super().__init__()
        self.node_dim = int(node_dim)
        self.event_dim = int(event_dim)

        self.use_time = bool(use_time)
        self.time_emb_dim = int(time_emb_dim)

        self.time_mlp: Optional[nn.Module] = None
        extra = 0
        if self.use_time:
            self.time_mlp = nn.Sequential(
                nn.Linear(1, self.time_emb_dim),
                nn.ReLU(),
                nn.Linear(self.time_emb_dim, self.time_emb_dim),
            )
            extra = self.time_emb_dim

        in_dim = 2 * self.node_dim + self.event_dim + extra

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        ) 

    def forward(self, state: Optional[ModelState], candidate_events: EventBatch) -> torch.Tensor:
        assert state is not None and state.node is not None, "MLPEdgeScorer requires state.node."
        H = state.node  # (N, d)

        src = candidate_events.src.to(device=H.device, dtype=torch.long)
        dst = candidate_events.dst.to(device=H.device, dtype=torch.long)
        h_src = H[src]
        h_dst = H[dst]

        # print("DEBUG scorer src shape", candidate_events.src.shape, "dst shape", candidate_events.dst.shape)
        # print("DEBUG first 10 src", candidate_events.src[:10].tolist())
        # print("DEBUG first 10 dst", candidate_events.dst[:10].tolist())   

        pieces = [h_src, h_dst]

        # edge / event features
        if self.event_dim > 0:
            if candidate_events.features is None:
                e = torch.zeros((src.numel(), self.event_dim), device=H.device, dtype=H.dtype)
            else:
                assert candidate_events.features.dim() == 2
                assert candidate_events.features.size(1) == self.event_dim, \
                    f"features dim {candidate_events.features.size(1)} != event_dim {self.event_dim}"
                e = candidate_events.features.to(device=H.device, dtype=H.dtype)
            pieces.append(e)

        # time features
        if self.use_time:
            assert candidate_events.t is not None, "use_time=True but candidate_events.t is None"
            t = candidate_events.t.to(device=H.device, dtype=H.dtype).view(-1, 1)
            assert self.time_mlp is not None
            pieces.append(self.time_mlp(t))
        
        x = torch.cat(pieces, dim=-1)
        return self.mlp(x).squeeze(-1)
