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

    def __init__(
        self,
        node_dim: int,
        msg_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        beta: float = 1.0,
        steps: int = 1,
        dropout: float = 0.0,
        use_layernorm: bool = True,
        gate: str = "sigmoid",     # "sigmoid" or "fixed"
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

    def init_state(
        self,
        batch_size: int,
        num_nodes: int,
        device: torch.device,
    ) -> Optional[ModelState]:
        node = torch.zeros((num_nodes, self.node_dim), device=device)
        return ModelState(node=node)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (N, D) -> (H, N, dk)
        N, D = x.shape
        return x.view(N, self.num_heads, self.dk).transpose(0, 1).contiguous()

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (H, N, dk) -> (N, D)
        H, N, dk = x.shape
        return x.transpose(0, 1).contiguous().view(N, H * dk)

    def forward(
        self,
        state: Optional[ModelState],
        messages: torch.Tensor,                  # (N, msg_dim)
        drive: Optional[torch.Tensor] = None,    # ignored for now
    ) -> Tuple[Optional[ModelState], Dict]:

        assert state is not None and state.node is not None, \
            "HopfieldUpdate requires state.node."
        h = state.node  # (N, node_dim)

        # Safety checks (match TGNGRUUpdate style)
        assert messages.dim() == 2, "messages must be [N, msg_dim]"
        assert messages.size(0) == h.size(0), "messages N must match state.node N"
        assert messages.size(1) == self.msg_dim, \
            f"messages dim {messages.size(1)} != msg_dim {self.msg_dim}"

        # Optional pre-norm
        h0 = self.ln(h) if self.ln is not None else h

        # Patterns
        K = self._split_heads(self.k_proj(h0))  # (H, N, dk)
        V = self._split_heads(self.v_proj(h0))  # (H, N, dk)

        h_cur = h
        retrieved = None

        for _ in range(self.steps):
            Q = self._split_heads(self.q_proj(messages))  # (H, N, dk)

            # scores: (H, N, N)
            scores = torch.einsum("hnd,hmd->hnm", Q, K) / (self.dk ** 0.5)
            scores = self.beta * scores

            A = torch.softmax(scores, dim=-1)
            A = F.dropout(A, p=self.dropout, training=self.training)

            retrieved_h = torch.einsum("hnm,hmd->hnd", A, V)  # (H, N, dk)
            retrieved = self.out_proj(self._merge_heads(retrieved_h))  # (N, node_dim)

            proposal = self.mix(torch.cat([h_cur, retrieved, messages], dim=-1))  # (N, node_dim)

            if self.gate_net is not None and self.gate == "sigmoid":
                g = self.gate_net(torch.cat([h_cur, messages], dim=-1))  # (N, node_dim)
            else:
                g = torch.full_like(h_cur, self.fixed_alpha)

            h_cur = (1.0 - g) * h_cur + g * proposal

            if self.ln is not None:
                h_cur = self.ln(h_cur)

        next_state = ModelState(node=h_cur)
        aux: Dict[str, torch.Tensor] = {}
        if retrieved is not None:
            aux["retrieved_norm"] = retrieved.norm(dim=-1).mean().detach()
        aux["node_norm"] = h_cur.norm(dim=-1).mean().detach()

        return next_state, aux
