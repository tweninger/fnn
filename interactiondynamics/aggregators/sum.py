import torch
from interactiondynamics.core.events import EventBatch
from interactiondynamics.core.interfaces import Aggregator, ModelState

class SumAggregator(Aggregator):
    """
    Sum aggregation of per-event message vectors into per-node messages.

    Minimal TGN-style aggregator:
        m_v = sum_{events incident to v} m_event
    """

    def __init__(self, add_to_dst: bool = True):
        super().__init__()
        self.add_to_dst = add_to_dst

    def forward(
        self,
        state: ModelState | None,
        event_embeddings: torch.Tensor,   # [M, d_msg]
        events: EventBatch,
        num_nodes: int,
    ) -> torch.Tensor:
        device = event_embeddings.device
        dtype = event_embeddings.dtype
        msg_dim = event_embeddings.size(-1)

        messages = torch.zeros((num_nodes, msg_dim), device=device, dtype=dtype)

        # Make indices safe for index_add_
        src = events.src.to(device=device, dtype=torch.long)
        messages.index_add_(0, src, event_embeddings)

        if self.add_to_dst:
            dst = events.dst.to(device=device, dtype=torch.long)
            messages.index_add_(0, dst, event_embeddings)

        return messages  # [N, d_msg]
