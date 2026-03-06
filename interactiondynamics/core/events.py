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

    def to(self, device):
        """Move all tensors to a device."""
        self.src = cast(torch.LongTensor, self.src.to(device))
        self.dst = cast(torch.LongTensor, self.dst.to(device))
        if self.features is not None:
            self.features = self.features.to(device)
        if self.t is not None:
            self.t = cast(torch.LongTensor, self.t.to(device))
        return self

    @property
    def num_events(self) -> int:
        return int(self.src.numel())