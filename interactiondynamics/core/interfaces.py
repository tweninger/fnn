from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn

from .events import EventBatch

class InteractionModel(nn.Module, ABC):
    """
    Top-level interface for all models in the codebase.

    Every model (DeepSets, LSTM, TGN, Hopfield, HNN, LNN, IFT)
    must implement this interface.

    State may be None (stateless models).
    """

    @abstractmethod
    def init_state(
        self,
        batch_size: int,
        num_nodes: int,
        device: torch.device,
    ) -> Optional[Any]:
        """
        Initialize the latent state.

        Stateless models MUST return None.
        """
        pass

    @abstractmethod
    def step(
        self,
        state: Optional[Any],
        events: EventBatch,
        drive: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[Any], Dict[str, Any]]:
        """
        Advance the system by one discrete time bin.

        Returns
        -------
        next_state : Optional[Any]
            Updated latent state.
        aux : Dict[str, Any]
            Diagnostics (energies, losses, etc.).
        """
        pass

    @abstractmethod
    def score(
        self,
        state: Optional[Any],
        candidate_events: EventBatch,
    ) -> torch.Tensor:
        """
        Score candidate events for ranking / likelihood.

        Returns
        -------
        scores : Tensor [num_candidates]
        """
        pass


@dataclass
class ModelState:
    """
    Container for all persistent latent state.

    This allows different models to store different
    things without changing the training loop.
    """

    # Per-node memory (TGN / IFT / Hopfield)
    node: Optional[torch.Tensor] = None        # (N, d_h)

    # Optional previous node state (Lagrangian-style)
    node_prev: Optional[torch.Tensor] = None   # (N, d_h)

    # Optional sparse dyad cache or other memory
    aux: Optional[Dict[str, Any]] = None

    def detach_(self) -> "ModelState":
        if self.node is not None:
            self.node = self.node.detach()
        if self.node_prev is not None:
            self.node_prev = self.node_prev.detach()
        if self.aux is not None:
            # detach any tensor values in aux
            self.aux = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in self.aux.items()}
        return self
    
    def clone(self, detach: bool = False) -> "ModelState":
        def _copy(x):
            if x is None:
                return None
            if torch.is_tensor(x):
                y = x.clone()
                return y.detach() if detach else y
            if isinstance(x, dict):
                return {k: _copy(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                t = [_copy(v) for v in x]
                return type(x)(t)
            return x

        return ModelState(
            node=_copy(self.node), # type: ignore
            node_prev=_copy(self.node_prev), # type: ignore
            aux=_copy(self.aux), # type: ignore
        )


class EventEncoder(nn.Module, ABC):
    """
    Maps (state, events) -> per-event embeddings.
    """

    @abstractmethod
    def forward(
        self,
        state: Optional[ModelState],
        events: EventBatch,
    ) -> torch.Tensor:
        """
        Returns
        -------
        event_embeddings : Tensor [M, d_event]
        """
        pass


class Aggregator(nn.Module, ABC):
    """
    Aggregates per-event embeddings into per-entity messages.
    """

    @abstractmethod
    def forward(
        self,
        state: Optional[ModelState],
        event_embeddings: torch.Tensor,
        events: EventBatch,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Returns
        -------
        messages : Tensor [num_nodes, d_msg]
        """
        pass


class UpdateLaw(nn.Module, ABC):
    """
    Defines the discrete-time law of motion.

    This is the ONLY place where 'physics' or 'dynamics' live.
    """

    @abstractmethod
    def init_state(
        self,
        batch_size: int,
        num_nodes: int,
        device: torch.device,
    ) -> Optional[ModelState]:
        pass

    @abstractmethod
    def forward(
        self,
        state: Optional[ModelState],
        messages: torch.Tensor,
        drive: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[ModelState], Dict[str, Any]]:
        """
        Returns
        -------
        next_state : Optional[ModelState]
        aux : Dict[str, Any]
        """
        pass

class ScoringHead(nn.Module, ABC):
    """
    Maps state to event scores.
    """

    @abstractmethod
    def forward(
        self,
        state: Optional[ModelState],
        candidate_events: EventBatch,
    ) -> torch.Tensor:
        pass


class NodeScoringHead(nn.Module, ABC):
    """
    Maps state to per-node scores.
    """

    @abstractmethod
    def forward(
        self,
        state: Optional[ModelState],
    ) -> torch.Tensor:
        pass

class ComposedInteractionModel(InteractionModel):
    """
    Canonical composition:
        EventEncoder -> Aggregator -> UpdateLaw -> ScoringHead
    """

    def __init__(
        self,
        encoder: EventEncoder,
        aggregator: Aggregator,
        update: UpdateLaw,
        scorer: ScoringHead,
        num_nodes: int,
        node_scorer: Optional[NodeScoringHead] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        self.update = update
        self.scorer = scorer
        self.node_scorer = node_scorer
        self.num_nodes = num_nodes

    def init_state(self, batch_size, num_nodes, device):
        return self.update.init_state(batch_size, num_nodes, device)

    def step(self, state, events, drive=None):
        event_emb = self.encoder(state, events)
        messages = self.aggregator(
            state, event_emb, events, self.num_nodes
        )
        next_state, aux = self.update(state, messages, drive)
        return next_state, aux

    def score(self, state, candidate_events):
        return self.scorer(state, candidate_events)

    def score_nodes(self, state):
        if self.node_scorer is None:
            raise RuntimeError("Model was built without a node_scorer")
        return self.node_scorer(state)
