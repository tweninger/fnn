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
        event_feature_dim: int = 0,
        event_feature_hidden: int = 128,
    ):
        super().__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        self.update = update
        self.scorer = scorer
        self.node_scorer = node_scorer
        self.num_nodes = num_nodes
        self.event_feature_dim = int(event_feature_dim)
        self.event_feature_decoder: Optional[nn.Module]
        if self.event_feature_dim > 0:
            self.event_feature_decoder = nn.Sequential(
                nn.Linear(2 * self._node_dim(), event_feature_hidden),
                nn.ReLU(),
                nn.Linear(event_feature_hidden, self.event_feature_dim),
            )
        else:
            self.event_feature_decoder = None
        # Runner-filled, training-split calibration for optional physical
        # force prediction.  These are not event inputs.
        calibration_dim = max(self.event_feature_dim, 1)
        self.register_buffer("event_feature_target_std", torch.ones(calibration_dim))
        self.register_buffer("event_feature_active_threshold", torch.tensor(0.0))
        self.register_buffer("event_feature_magnitude_q90", torch.tensor(1.0))
        self.event_feature_magnitude_weight = 2.0

    def _node_dim(self) -> int:
        """Infer the shared state width from the scoring head's configuration."""
        node_dim = getattr(self.scorer, "node_dim", None)
        if node_dim is None:
            raise ValueError("Event feature decoding requires a scorer with node_dim.")
        return int(node_dim)

    def init_state(self, batch_size, num_nodes, device):
        return self.update.init_state(batch_size, num_nodes, device)

    def step(self, state, events, drive=None):
        event_emb = self.encoder(state, events)
        effective_num_nodes = (
            self.num_nodes
            if state is None or state.node is None
            else int(state.node.size(0))
        )
        messages = self.aggregator(
            state, event_emb, events, effective_num_nodes
        )
        agg_aux = None
        if state is not None and state.aux is not None:
            # Aggregators may stash persistent operator state (for example an
            # EMA-smoothed Laplacian) on the incoming state before the update
            # law runs. Preserve that state across update laws that allocate a
            # fresh ModelState instead of mutating/cloning the old one.
            agg_aux = dict(state.aux)
        next_state, aux = self.update(state, messages, drive)
        if next_state is not None and agg_aux is not None:
            next_aux = {} if next_state.aux is None else dict(next_state.aux)
            merged_aux = dict(agg_aux)
            merged_aux.update(next_aux)
            next_state.aux = merged_aux
        return next_state, aux

    def score(self, state, candidate_events):
        return self.scorer(state, candidate_events)

    def score_nodes(self, state):
        if self.node_scorer is None:
            raise RuntimeError("Model was built without a node_scorer")
        return self.node_scorer(state)

    def predict_event_features(self, state, events: EventBatch) -> torch.Tensor:
        """Predict a physical force vector from state and an event pair only."""
        if self.event_feature_decoder is None:
            raise RuntimeError("Model was built without an event-feature decoder")
        if state is None or state.node is None:
            raise ValueError("Event feature decoding requires node state.")
        h = state.node
        src = events.src.to(device=h.device, dtype=torch.long)
        dst = events.dst.to(device=h.device, dtype=torch.long)
        return self.event_feature_decoder(torch.cat([h[src], h[dst]], dim=-1))

    @torch.no_grad()
    def configure_event_feature_objective(
        self,
        *,
        target_std: torch.Tensor,
        active_threshold: float,
        magnitude_q90: float,
        magnitude_weight: float,
    ) -> None:
        self.event_feature_target_std.copy_(target_std.to(self.event_feature_target_std).clamp_min(1e-8))
        self.event_feature_active_threshold.fill_(float(active_threshold))
        self.event_feature_magnitude_q90.fill_(max(float(magnitude_q90), 1e-8))
        self.event_feature_magnitude_weight = float(magnitude_weight)
