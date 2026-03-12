from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn

from .events import EventBatch

# template for any interaction model in the repo using pytorch model class and abstract base class (states required funcitons)
class InteractionModel(nn.Module, ABC):
    """
    Top-level interface for all models in the codebase.

    Every model (DeepSets, LSTM, TGN, Hopfield, HNN, LNN, IFT)
    must implement this interface.

    State may be None (stateless models).

    ok anyways...
    interface/architecture file! not actual math implementation
    what model is supposed to do, what a state object looks like, what encoder/aggr/update/scorer are
    how those pieces fit together

    RECIPE template not actual recipe yum
    """
    # when training starts, how do you initialize the latent memory/state? aka what memory do i start with
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

    # given curr state and one batch/bin of events, advance the system one step
    # called in train.py duh
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

    # connects to ranking_loss_and_metrics
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

# container for hidden state/memory
@dataclass
class ModelState:
    """
    Container for all persistent latent state.

    This allows different models to store different
    things without changing the training loop.
    """

    # Per-node memory (TGN / IFT / Hopfield)
    node: Optional[torch.Tensor] = None        # (N, d_h) nodes, hidden dimension.. main hidden state thats updated

    # Optional previous node state (Lagrangian-style)
    node_prev: Optional[torch.Tensor] = None   # (N, d_h)

    # Optional sparse dyad cache or other memory
    aux: Optional[Dict[str, Any]] = None


    # detaches all tensors in the state from the computation graph!!! bye bye fors sequential models
    def detach_(self) -> "ModelState":
        if self.node is not None:
            self.node = self.node.detach()
        if self.node_prev is not None:
            self.node_prev = self.node_prev.detach()
        if self.aux is not None:
            # detach any tensor values in aux
            self.aux = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in self.aux.items()}
        return self
    
    # copy state ok
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

# curr state + raw events -> event embeddings
class EventEncoder(nn.Module, ABC):
    """
    Maps (state, events) -> per-event embeddings.
    """
    # one vector per event
    @abstractmethod
    def forward(
        self,
        state: Optional[ModelState],
        events: EventBatch,
    ) -> torch.Tensor:
        """
        Returns
        -------
        event_embeddings : Tensor [M, d_event] # number of events in bin + embedding dimension
        """
        pass

# combine hehe
class Aggregator(nn.Module, ABC):
    """
    Aggregates per-event embeddings into per-entity messages.
    """
    # take all event level vectors and compress/aggre into one message per node
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

# update is state evolution rule... V CENTRAL OKAY...
class UpdateLaw(nn.Module, ABC):
    """
    Defines the discrete-time law of motion.

    This is the ONLY place where 'physics' or 'dynamics' live.
    """

    # initialize model state
    @abstractmethod
    def init_state(
        self,
        batch_size: int,
        num_nodes: int,
        device: torch.device,
    ) -> Optional[ModelState]:
        pass

    # actual evolution step - returns updated model state + diagnostics .. messages in -> new node mem/state out
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

# prediction head
class ScoringHead(nn.Module, ABC):
    """
    Maps state to event scores.
    """
    # return score.. curr state + candidate event info -> scores
    @abstractmethod
    def forward(
        self,
        state: Optional[ModelState],
        candidate_events: EventBatch,
    ) -> torch.Tensor:
        pass

# WOWOWOW real concrete model implementation that follows interactionModel interface
# pulls everything together
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
    ):
        super().__init__() 
        self.encoder = encoder
        self.aggregator = aggregator
        self.update = update
        self.scorer = scorer
        self.num_nodes = num_nodes

    def init_state(self, batch_size, num_nodes, device): # hidden state is owned by the update modeule... diff update laws might want diff state structure
        return self.update.init_state(batch_size, num_nodes, device)

    # pipelineeee for forward time
    def step(self, state, events, drive=None):
        event_emb = self.encoder(state, events) # encode raw evenets into event embeddings
        messages = self.aggregator( # this stuff aggregates event embeddings into per node messages
            state, event_emb, events, self.num_nodes
        )
        next_state, aux = self.update(state, messages, drive) # update hidden state with those messages
        return next_state, aux
    # prediction is delegated to scorer (step - state evolution, score - event ranking/prediction)
    def score(self, state, candidate_events):
        return self.scorer(state, candidate_events)
