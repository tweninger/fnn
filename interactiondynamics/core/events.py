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
    episode : Optional[LongTensor] [M]
        Identifier for the independently simulated trajectory containing an
        event.  This is sequence bookkeeping, not an observed interaction
        feature: trainers use it to reset recurrent state at a trajectory
        boundary.
    is_external : Optional[BoolTensor] [M]
        Marks an observed intervention, such as the raindrop that starts an
        episode. It is part of the event record, not hidden simulator state.
    """
    src: torch.LongTensor
    dst: torch.LongTensor
    features: Optional[torch.Tensor] = None
    t: Optional[torch.LongTensor] = None
    episode: Optional[torch.LongTensor] = None
    is_external: Optional[torch.BoolTensor] = None

    def to(self, device):
        """Move all tensors to a device."""
        self.src = cast(torch.LongTensor, self.src.to(device))
        self.dst = cast(torch.LongTensor, self.dst.to(device))
        if self.features is not None:
            self.features = self.features.to(device)
        if self.t is not None:
            self.t = cast(torch.LongTensor, self.t.to(device))
        if self.episode is not None:
            self.episode = cast(torch.LongTensor, self.episode.to(device))
        if self.is_external is not None:
            self.is_external = cast(torch.BoolTensor, self.is_external.to(device))
        return self

    @property
    def num_events(self) -> int:
        return int(self.src.numel())
