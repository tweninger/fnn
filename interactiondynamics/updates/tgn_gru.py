from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn

from interactiondynamics.core.interfaces import ModelState, UpdateLaw


class TGNGRUUpdate(UpdateLaw):
    """
    TGN-style node memory update using a GRUCell.

    Minimal (binned-time) version:
    - every step updates ALL nodes using messages (N, msg_dim)
    - state.node stores the node memory (N, node_dim)
    """

    def __init__(self, node_dim: int, msg_dim: int):
        super().__init__()
        self.node_dim = int(node_dim)
        self.msg_dim = int(msg_dim)
        self.gru = nn.GRUCell(self.msg_dim, self.node_dim)

    def init_state(
        self,
        batch_size: int,
        num_nodes: int,
        device: torch.device,
    ) -> Optional[ModelState]:
        node = torch.randn((int(batch_size) * num_nodes, self.node_dim), device=device) * 0.02
        return ModelState(node=node, aux={"batch_size": int(batch_size), "nodes_per_graph": int(num_nodes)})

    def forward(
        self,
        state: Optional[ModelState],
        messages: torch.Tensor,                  # (N, msg_dim)
        drive: Optional[torch.Tensor] = None,    # ignored for now
    ) -> Tuple[Optional[ModelState], Dict]:

        assert state is not None and state.node is not None, \
            "TGNGRUUpdate requires state.node."
        h = state.node  # (N, node_dim)

        # Safety checks (these catch subtle wiring bugs early)
        assert messages.dim() == 2, "messages must be [N, msg_dim]"
        assert messages.size(0) == h.size(0), "messages N must match state.node N"
        assert messages.size(1) == self.msg_dim, \
            f"messages dim {messages.size(1)} != msg_dim {self.msg_dim}"

        # GRUCell: (input, hidden) -> next_hidden
        h_next = self.gru(messages, h)

        next_state = ModelState(node=h_next)
        return next_state, {}
