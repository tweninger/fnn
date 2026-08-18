from dataclasses import dataclass
from typing import Iterable, Optional, cast
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
    batch: Optional[torch.LongTensor] = None

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
        if self.batch is not None:
            self.batch = cast(torch.LongTensor, self.batch.to(device))
        return self

    @property
    def num_events(self) -> int:
        return int(self.src.numel())


def pack_independent_episode_bins(
    bins: Iterable[EventBatch],
    *,
    num_nodes: int,
) -> list[EventBatch]:
    """Pack equal-length independent episodes by their local timestep.

    Node IDs are offset into one disjoint packed node space. ``batch`` retains
    the originating episode for operations (notably negative sampling) whose
    candidate set must remain within a physical system.  The event content and
    temporal order of every episode are otherwise unchanged.

    A single episode is returned untouched, preserving the legacy path.
    """
    sequence = list(bins)
    if not sequence:
        return []
    if any(events.episode is None for events in sequence):
        return sequence

    episodes: list[list[EventBatch]] = []
    current: list[EventBatch] = []
    current_id: Optional[int] = None
    for events in sequence:
        episode_id = int(events.episode[0].item())
        if current_id is None or episode_id == current_id:
            current.append(events)
        else:
            episodes.append(current)
            current = [events]
        current_id = episode_id
    if current:
        episodes.append(current)
    if len(episodes) <= 1:
        return sequence
    lengths = {len(episode) for episode in episodes}
    if len(lengths) != 1:
        # Unequal episodes are uncommon for the synthetic physical suite. Do
        # not silently change their sampling semantics; use the old stream.
        return sequence

    packed: list[EventBatch] = []
    for local_step in range(len(episodes[0])):
        chunks = [episode[local_step] for episode in episodes]
        src = torch.cat([events.src + batch_id * num_nodes for batch_id, events in enumerate(chunks)])
        dst = torch.cat([events.dst + batch_id * num_nodes for batch_id, events in enumerate(chunks)])
        batch = torch.cat([
            torch.full_like(events.src, batch_id) for batch_id, events in enumerate(chunks)
        ])
        features = (
            None
            if chunks[0].features is None
            else torch.cat([events.features for events in chunks])
        )
        # Empty physical bins retain one scalar time/episode marker for the
        # serial stream. In a packed event tensor that marker has no matching
        # event row, so exclude it rather than misaligning metadata.
        times = None if chunks[0].t is None else torch.cat([
            events.t[: events.num_events] for events in chunks
        ])
        episode = torch.cat([
            events.episode[: events.num_events] for events in chunks
        ])
        external = (
            None
            if chunks[0].is_external is None
            else torch.cat([events.is_external for events in chunks])
        )
        packed.append(EventBatch(
            src=src,
            dst=dst,
            features=features,
            t=times,
            episode=episode,
            is_external=external,
            batch=batch,
        ))
    return packed
