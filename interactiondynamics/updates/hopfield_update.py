from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.interfaces import UpdateLaw, ModelState


class HopfieldUpdate(UpdateLaw):
    """
    Hopfield-style associative memory update.

    Minimal (binned-time) version:
    - every step updates ALL nodes using messages (N, msg_dim)
    - state.node stores the node memory/patterns (N, node_dim)

    Retrieval: multi-head attention over current node states as patterns.
      Q = Wq(messages)
      K = Wk(state.node)
      V = Wv(state.node)
      retrieved = softmax(beta * QK^T / sqrt(dk)) V

    Update: gated residual mixing
      proposal = MLP([h, retrieved, msg])
      g = sigmoid(Wg([h, msg]))  (or fixed alpha)
      h_next = (1-g)*h + g*proposal
    """
# hi constructor
    def __init__(
        self,
        node_dim: int,
        msg_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4, # multi-head attention style split of the hidden state! hi barbie
        beta: float = 1.0, # controls sharpness of retrieval attention (bigger beta = more peaked/selective retireval)
        steps: int = 1, # how many times to do retrieval-update loop (so rn, one retrieval/update pass)
        dropout: float = 0.0,
        use_layernorm: bool = True,
        gate: str = "sigmoid",     # "sigmoid" or "fixed" aka learned per dimension gate or constant mixing weight
        fixed_alpha: float = 0.5,  # used if gate == "fixed"
    ):
        super().__init__()
        self.node_dim = int(node_dim)
        self.msg_dim = int(msg_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.beta = float(beta)
        self.steps = int(steps)
        self.dropout = float(dropout)
        self.gate = str(gate)
        self.fixed_alpha = float(fixed_alpha)

        assert self.steps >= 1, "steps must be >= 1"
        assert self.num_heads >= 1, "num_heads must be >= 1"
        if self.node_dim % self.num_heads != 0:
            raise ValueError(f"node_dim={self.node_dim} must be divisible by num_heads={self.num_heads}")
        self.dk = self.node_dim // self.num_heads

        # Projections
        # standard attention-style machinery hi barbie
        # q comes from messages, k and v come from current node states
        # the message is the query, and the current node memories are the keys/values
        # aka retrieval setup
        self.q_proj = nn.Linear(self.msg_dim, self.node_dim, bias=False)
        self.k_proj = nn.Linear(self.node_dim, self.node_dim, bias=False)
        self.v_proj = nn.Linear(self.node_dim, self.node_dim, bias=False)
        self.out_proj = nn.Linear(self.node_dim, self.node_dim, bias=False)

        # Mixing network for proposal
        self.mix = nn.Sequential(
            nn.Linear(self.node_dim + self.node_dim + self.msg_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, self.node_dim),
        )

        # Gate
        if self.gate == "sigmoid":
            self.gate_net = nn.Sequential(
                nn.Linear(self.node_dim + self.msg_dim, self.node_dim),
                nn.Sigmoid(),
            )
        elif self.gate == "fixed":
            self.gate_net = None
        else:
            raise ValueError("gate must be 'sigmoid' or 'fixed'")

        self.ln = nn.LayerNorm(self.node_dim) if use_layernorm else None

    # wow surprise initial node memory is all zeros, no q p split like HNN, just one hidden vector per node
    def init_state(
        self,
        batch_size: int,
        num_nodes: int,
        device: torch.device,
    ) -> Optional[ModelState]:
        node = torch.zeros((num_nodes, self.node_dim), device=device)
        return ModelState(node=node)

    #split and merge... helper functions for multi-head attention
    # reshape from one big vector into smaller heads and back again.. attention bookkeeping
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (N, D) -> (H, N, dk)
        N, D = x.shape
        return x.view(N, self.num_heads, self.dk).transpose(0, 1).contiguous()

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (H, N, dk) -> (N, D)
        H, N, dk = x.shape
        return x.transpose(0, 1).contiguous().view(N, H * dk)

    # hi barbie
    def forward(
        self,
        state: Optional[ModelState],
        messages: torch.Tensor,                  # (N, msg_dim)
        drive: Optional[torch.Tensor] = None,    # ignored for now
    ) -> Tuple[Optional[ModelState], Dict]:

        assert state is not None and state.node is not None, \
            "HopfieldUpdate requires state.node."
        h = state.node  # (N, node_dim) current memory

        # Safety checks (match TGNGRUUpdate style)
        assert messages.dim() == 2, "messages must be [N, msg_dim]"
        assert messages.size(0) == h.size(0), "messages N must match state.node N"
        assert messages.size(1) == self.msg_dim, \
            f"messages dim {messages.size(1)} != msg_dim {self.msg_dim}"

        # Optional pre-norm
        h0 = self.ln(h) if self.ln is not None else h # -> stabilizes values a bit before retrieval, layer normalizaiton hi baribe

        # Patterns
        # build memory patterns!
        # keys and vals come from curr node states, so node states themselves are the thing being searched over
        # this is why its associative memory flavored
        K = self._split_heads(self.k_proj(h0))  # (H, N, dk)
        V = self._split_heads(self.v_proj(h0))  # (H, N, dk)

        h_cur = h
        retrieved = None

        #iterative retrieval/update loop.. many rounds of retrieval if you want
        for _ in range(self.steps):
            Q = self._split_heads(self.q_proj(messages))  # (H, N, dk) ... incoming messages for each node turns into query vector
            # ^^ what mem pattern should this message retrieve

            # scores: (H, N, N)
            # just attention. aka for each node/message, compare query to all mem patterns
            # compute relevance scores
            # softmax them into weights
            #... yeah lol aka which node state memories matter most for this message
            scores = torch.einsum("hnd,hmd->hnm", Q, K) / (self.dk ** 0.5)
            scores = self.beta * scores

            A = torch.softmax(scores, dim=-1)
            A = F.dropout(A, p=self.dropout, training=self.training)

            # weighted sum of memory vals.. result = one retrieved memory vector per node
            # HOPFIELD RETRIEVAL PART
            retrieved_h = torch.einsum("hnm,hmd->hnd", A, V)  # (H, N, dk)
            retrieved = self.out_proj(self._merge_heads(retrieved_h))  # (N, node_dim)

            # who is proposing
            # mixes curr node state, retrieved memory, and incoming message -> proposed new state! 
            proposal = self.mix(torch.cat([h_cur, retrieved, messages], dim=-1))  # (N, node_dim)

            if self.gate_net is not None and self.gate == "sigmoid":
                g = self.gate_net(torch.cat([h_cur, messages], dim=-1))  # (N, node_dim)             # learned gate
            else:
                g = torch.full_like(h_cur, self.fixed_alpha) # fixed gate

            h_cur = (1.0 - g) * h_cur + g * proposal # update! blend old memory + new proposal so node keeps some of what it already knew
            # GRU vibes but with hopfield retrieval inside
            if self.ln is not None:
                h_cur = self.ln(h_cur) # hi norm

        next_state = ModelState(node=h_cur) # next node mem is updated h_cur
        aux: Dict[str, torch.Tensor] = {} # retrieved norm + node norm
        if retrieved is not None:
            aux["retrieved_norm"] = retrieved.norm(dim=-1).mean().detach()
        aux["node_norm"] = h_cur.norm(dim=-1).mean().detach()

        return next_state, aux
