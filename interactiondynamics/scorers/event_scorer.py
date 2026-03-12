from typing import Optional
import torch
import torch.nn as nn

from core.interfaces import ModelState, ScoringHead
from core.events import EventBatch

# given curr node states, how likely is this cadidate event/edge? :O
# after fancy updating, does src -> dest look plausible rn?
#... give me a number for that candidate
# from ranking.py: model sees real and fake dest, scorer gives a score to each one, and training wants real dest to score highest


# yay basic baseline
class DotProductScorer(ScoringHead):
    """
    score(u, v) = <h_u, h_v> 
    """

    def forward(self, state: Optional[ModelState], candidate_events: EventBatch) -> torch.Tensor:
        assert state is not None and state.node is not None, "DotProductScorer requires state.node."

        H = state.node  # (N, d)
        src = candidate_events.src.to(device=H.device, dtype=torch.long)
        dst = candidate_events.dst.to(device=H.device, dtype=torch.long)

        # take src node embedding and dest node embedding and compute dot product!
        # aka get hidden state of src and dest, multiply elementwise and sum
        # if two node states align well, high score!! ... its just similarity based scoring
        h_src = H[src]  # (M, d)
        h_dst = H[dst]  # (M, d)
        return (h_src * h_dst).sum(dim=-1)  # (M,)

# flexible scorerrrr yay!
# formula thing
# ^^ src node state and dest node state, maybe include event features, maybe include time features, concatenate!
# ... then feed into MLP and output one score
# NN learns the scoring rule
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
        # hi barbie! src and dest states
        h_src = H[src]
        h_dst = H[dst]

        # print("DEBUG scorer src shape", candidate_events.src.shape, "dst shape", candidate_events.dst.shape)
        # print("DEBUG first 10 src", candidate_events.src[:10].tolist())
        # print("DEBUG first 10 dst", candidate_events.dst[:10].tolist())   

        # base input is src and dest embeddings yes yes
        pieces = [h_src, h_dst]

        # edge / event features
        if self.event_dim > 0: # includes actual event features, like event attributes
            if candidate_events.features is None:
                e = torch.zeros((src.numel(), self.event_dim), device=H.device, dtype=H.dtype)
            else:
                assert candidate_events.features.dim() == 2
                assert candidate_events.features.size(1) == self.event_dim, \
                    f"features dim {candidate_events.features.size(1)} != event_dim {self.event_dim}"
                e = candidate_events.features.to(device=H.device, dtype=H.dtype)
            pieces.append(e)

        # time features
        if self.use_time: # take timestamp t and include that in MLP
            assert candidate_events.t is not None, "use_time=True but candidate_events.t is None"
            t = candidate_events.t.to(device=H.device, dtype=H.dtype).view(-1, 1)
            assert self.time_mlp is not None
            pieces.append(self.time_mlp(t))
        
        # concatenate and score -- gives one scalar score per candidate event
        x = torch.cat(pieces, dim=-1)
        return self.mlp(x).squeeze(-1)
