from dataclasses import dataclass
from typing import Optional, cast
import torch


@dataclass
class EventBatch:
    """
    A batch of interaction events.

    This is the ONLY observed object in the system.
    No edges, no adjacency, no graph.

    Attributes
    ----------
    src : LongTensor [M]
        Source node indices.
    dst : LongTensor [M]
        Destination node indices.
    features : Optional[Tensor] [M, d_e]
        Optional per-event features.
    t : Optional[LongTensor] [M]
        Time associated with each event.
        Can be a discrete bin index or continuous timestamp.
    """

    src: torch.LongTensor
    dst: torch.LongTensor
    features: Optional[torch.Tensor] = None
    t: Optional[torch.LongTensor] = None
    
    node_targets: Optional[torch.Tensor] = None   # [N, d_y]
    node_mask: Optional[torch.Tensor] = None      # [N] bool, optional

    def to(self, device):
        self.src = cast(torch.LongTensor, self.src.to(device))
        self.dst = cast(torch.LongTensor, self.dst.to(device))
        if self.features is not None:
            self.features = self.features.to(device)
        if self.t is not None:
            self.t = cast(torch.LongTensor, self.t.to(device))
        if self.node_targets is not None:
            self.node_targets = self.node_targets.to(device)
        if self.node_mask is not None:
            self.node_mask = self.node_mask.to(device)
        return self

    @property
    def num_events(self) -> int:
        return int(self.src.numel())