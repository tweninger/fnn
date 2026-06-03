# updates/lnn.py
from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from interactiondynamics.core.interfaces import ModelState, UpdateLaw


def _mlp(in_dim: int, hidden: int, layers: int, out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    mods = []
    d = in_dim
    for _ in range(max(1, layers)):
        mods.append(nn.Linear(d, hidden))
        mods.append(nn.SiLU())
        if dropout and dropout > 0:
            mods.append(nn.Dropout(dropout))
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


class LNNUpdate(UpdateLaw):
    """
    Discrete-time Lagrangian-style update.

    State:
      node      := q      (N, d)
      node_prev := q_prev (N, d)

    Velocity estimate:
      qdot = (q - q_prev) / dt

    Learned potential:
      V = V_theta(q, messages, drive)  (scalar per node, summed)

    Dynamics (unit mass):
      qddot = - dV/dq

    Integration:
      qdot_next = (1 - damping)*qdot + qddot * dt
      q_next    = q + qdot_next * dt
    """

    def __init__(
        self,
        node_dim: int,
        msg_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dt: float = 1.0,
        damping: float = 0.0,
        dropout: float = 0.0,
        drive_dim: int = 0,  # set >0 if you actually pass a per-node drive vector
    ):
        super().__init__()
        self.node_dim = int(node_dim)
        self.msg_dim = int(msg_dim)
        self.dt = float(dt)
        self.damping = float(damping)

        in_dim = self.node_dim + self.msg_dim + int(drive_dim)
        # output 1 scalar per node => potential energy contribution
        self.V = _mlp(in_dim, hidden_dim, num_layers, out_dim=1, dropout=dropout)

    def init_state(
        self,
        batch_size: int,
        num_nodes: int,
        device: torch.device,
    ) -> Optional[ModelState]:
        q = torch.randn(num_nodes, self.node_dim, device=device) * 0.02
        q_prev = q.clone()
        return ModelState(node=q, node_prev=q_prev, aux={})

    def forward(self, state, messages, drive=None):
        assert state is not None
        assert state.node is not None and state.node_prev is not None

        q_raw = state.node
        q_prev = state.node_prev
        dt = self.dt

        # Always compute the physics with grads enabled (even inside torch.no_grad eval)
        with torch.enable_grad():
            # Make q a leaf requiring grad
            q = q_raw.detach().requires_grad_(True)

            qdot = (q - q_prev) / dt

            if drive is not None:
                drive_in = drive.unsqueeze(-1) if drive.dim() == 1 else drive
                inp = torch.cat([q, messages, drive_in], dim=-1)
            else:
                inp = torch.cat([q, messages], dim=-1)

            V_per = self.V(inp).squeeze(-1)
            V_tot = V_per.sum()

            (dV_dq,) = torch.autograd.grad(
                V_tot, q,
                create_graph=torch.is_grad_enabled(),   # False in eval's enable_grad? Actually True here.
                retain_graph=False,
                allow_unused=False,
            )

            qddot = -dV_dq

            if self.damping != 0.0:
                qdot_next = (1.0 - self.damping) * qdot + qddot * dt
            else:
                qdot_next = qdot + qddot * dt

            q_next = q + qdot_next * dt

        # Store next state (q_next may carry a graph during training; eval it's fine)
        next_state = state.clone(detach=False)
        next_state.node_prev = q_raw
        next_state.node = q_next.detach()

        aux = {
            "V_tot": V_tot.detach(),
            "qdot_norm": qdot.norm(dim=-1).mean().detach(),
            "qddot_norm": qddot.norm(dim=-1).mean().detach(),
        }
        return next_state, aux
