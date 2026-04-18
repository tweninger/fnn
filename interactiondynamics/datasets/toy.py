# data/toy.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, cast
import torch

from core.events import EventBatch
from datasets import DataSpec, EventStreamDataset


@dataclass
class ToyShiftConfig:
    name: str = "toy_shift"
    num_nodes: int = 200
    num_bins: int = 200
    events_per_bin: int = 256

    # If shift is not None => stationary mapping dst=(src+shift)%N
    shift: Optional[int] = 7

    # If shift is None => time-varying shift; optionally expose time as features (dim=1)
    time_as_feature: bool = True

    # device for batches (you can also just do events.to(device) in train loop)
    device: Optional[torch.device] = None


class ToyShiftDataset(EventStreamDataset):
    def __init__(self, cfg: ToyShiftConfig):
        self.cfg = cfg
        self._event_dim = 1 if (cfg.shift is None and cfg.time_as_feature) else 0

    def spec(self) -> DataSpec:
        return DataSpec(
            name=self.cfg.name,
            num_nodes=self.cfg.num_nodes,
            event_dim=self._event_dim,
            num_bins=self.cfg.num_bins,
            num_events=self.cfg.num_bins * self.cfg.events_per_bin,
            extra={"shift": self.cfg.shift, "time_as_feature": self.cfg.time_as_feature},
        )

    def bins(self, split: str = "train") -> Iterable[EventBatch]:
        # For toys we can ignore split or later add a split policy.
        return _ToyShiftStream(self.cfg, self._event_dim)


@dataclass
class _ToyShiftStream(Iterable[EventBatch]):
    cfg: ToyShiftConfig
    event_dim: int

    def __iter__(self) -> Iterator[EventBatch]:
        device = self.cfg.device
        N = self.cfg.num_nodes
        for b in range(self.cfg.num_bins):
            src = torch.randint(0, N, (self.cfg.events_per_bin,), dtype=torch.long)
            if self.cfg.shift is not None:
                dst = (src + int(self.cfg.shift)) % N
                feats = None
            else:
                # time-varying shift
                shift = 1 + (b % max(1, N - 1))
                dst = (src + shift) % N
                feats = None
                if self.event_dim > 0:
                    feats = torch.full((src.numel(), 1), float(b) / max(1, self.cfg.num_bins - 1), dtype=torch.float32)

            eb = EventBatch(
                src=cast(torch.LongTensor, src),
                dst=cast(torch.LongTensor, dst),
                t=cast(torch.LongTensor, torch.full((src.numel(),), b, dtype=torch.long)),
                features=feats,
            )
            if device is not None:
                eb = eb.to(device)
            yield eb
