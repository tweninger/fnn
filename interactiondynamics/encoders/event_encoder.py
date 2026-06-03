# encoders/event_encoder.py
from typing import Optional
import torch
import torch.nn as nn

from core.interfaces import EventEncoder, ModelState
from core.events import EventBatch 

class TGNEventEncoder(EventEncoder):
    """
    (state, events) -> per-event message vectors
    """
    def __init__(
        self,
        node_dim: int,
        event_dim: int,
        msg_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        use_time_features: bool = False,
        time_emb_dim: int = 32,
    ):
        super().__init__()
        self.node_dim = int(node_dim)
        self.event_dim = int(event_dim)
        self.use_time_features = bool(use_time_features)

        # Time embedding (same style as encoder)
        self.time_mlp: Optional[nn.Module] = None
        time_in = 0
        if self.use_time_features:
            self.time_mlp = nn.Sequential(
                nn.Linear(1, time_emb_dim),
                nn.ReLU(), #                look! tiny time embeddingggg so small
                nn.Linear(time_emb_dim, time_emb_dim),
            )
            time_in = int(time_emb_dim)

        in_dim = 2 * self.node_dim + self.event_dim + time_in

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, msg_dim),
        )

    def forward(self, state: Optional[ModelState], events: EventBatch) -> torch.Tensor:
        assert state is not None and state.node is not None, "Need state.node for TGNEventEncoder."

        # for each event now we have src and dest node state
        h_src = state.node[events.src]
        h_dst = state.node[events.dst]

        # build input to encoder: always uses src and dest hidden state... and maybe event/time features
        pieces = [h_src, h_dst]

        # if model expects event features, then event batch must contain them + those features get appended into input ^^
        if self.event_dim > 0:
            assert events.features is not None, "event_dim>0 but events.features is None"
            assert events.features.size(-1) == self.event_dim, \
                f"events.features dim {events.features.size(-1)} != event_dim {self.event_dim}" 
            pieces.append(events.features) # append event features to input

        #turn timestamp into vector and include it too
        if self.use_time_features:
            assert events.t is not None
            t = events.t.to(h_src.dtype).view(-1, 1)

            if self.time_mlp is None:
                pieces.append(t)                 # scalar time
            else:
                pieces.append(self.time_mlp(t))  # learned time embedding... TIME EMBEDDING bc models don't like one raw scalar value

        # encoder input essentially src state, dest state, event features, maybe time features
        x = torch.cat(pieces, dim=-1)
        return self.mlp(x)  # (M, msg_dim)
        # ^^ hi barbie
        # feed that long vector into a NN!
        # with linear layer, ReLU, dropout, linear layer -> message vector of size msg_dim which goes to aggre!
      