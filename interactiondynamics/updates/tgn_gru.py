from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn

from core.interfaces import UpdateLaw, ModelState
# hashtag classic temporal graph baseline

#GRU!!!
# GRU is a gated recurrent unit silly
# NN memory cell that decides how much of the old mem to keep, and how much new info to write in
# TGN - temporal graph networks
# ^^ TGNs commonly use a memory module updated with recurrent cells like GRUs in response to interaction messages
class TGNGRUUpdate(UpdateLaw):
    """
    TGN-style node memory update using a GRUCell.

    Minimal (binned-time) version:
    - every step updates ALL nodes using messages (N, msg_dim)
    - state.node stores the node memory (N, node_dim)

    aka... take current node memory and the new message, and update memory with a GRU
    """

    def __init__(self, node_dim: int, msg_dim: int):
        super().__init__()
        self.node_dim = int(node_dim)
        self.msg_dim = int(msg_dim)
        self.gru = nn.GRUCell(self.msg_dim, self.node_dim)

    # wow haven't seen this before
    def init_state(
        self,
        batch_size: int,
        num_nodes: int,
        device: torch.device,
    ) -> Optional[ModelState]:
        # batch_size unused in this minimal version (single global node memory)
        node = torch.zeros((num_nodes, self.node_dim), device=device)
        return ModelState(node=node)

    # each step, take the aggregated messages and use GRUCell to update every node's hidden state
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
        # use a GRUCell to combine the old state and the new input into a new state
        # aka standard recurrent memory update
        h_next = self.gru(messages, h)

        next_state = ModelState(node=h_next)
        return next_state, {} # new state yay
