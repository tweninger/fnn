# data/interfaces.py

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional
import torch

from interactiondynamics.core.events import EventBatch


@dataclass(frozen=True)
class DataSpec:
    """
    Minimum metadata needed to build models + training harness.
    """
    name: str
    num_nodes: int
    event_dim: int

    # Optional but often handy
    num_events: Optional[int] = None
    num_bins: Optional[int] = None
    extra: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class EdgeTargetBatch:
    events: EventBatch
    targets: torch.Tensor


class EventStreamDataset(ABC):
    """
    Base class for datasets that yield binned event streams.

    Key requirements:
      - spec() is available immediately (for model construction)
      - bins(split) returns a RE-ITERABLE Iterable[EventBatch]
    """

    @abstractmethod
    def spec(self) -> DataSpec:
        pass

    @abstractmethod
    def bins(self, split: str = "train") -> Iterable[EventBatch]:
        """
        split in {"train","val","test"}; implementations decide how to split.
        Must be re-iterable (safe to loop multiple epochs).
        """
        pass

    def node_targets(self, split: str = "train") -> Optional[Iterable[torch.Tensor]]:
        """
        Optional per-bin node-level supervision aligned with `bins(split)`.
        Returns tensors of shape [num_nodes] or [num_nodes, d].
        """
        return None

    def edge_targets(self, split: str = "train") -> Optional[Iterable[EdgeTargetBatch]]:
        """
        Optional per-bin edge-level supervision aligned with `bins(split)`.
        Each batch provides candidate edges plus scalar targets for those edges.
        """
        return None
