# updates/hnn.py
from __future__ import annotations
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

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


class HNNUpdate(UpdateLaw):
    """
    Hamiltonian Neural Network update.

    State interpretation:
      node: (N, node_dim) = concat([q, p]) where q,p each have dim d = node_dim//2.

    Learned Hamiltonian:
      H_theta(q,p,messages,drive) -> scalar per node, summed across nodes.

    Hamilton's equations:
      dq/dt =  dH/dp
      dp/dt = -dH/dq

    Integration (symplectic Euler):
      p_{t+1} = p_t + dt * dp/dt
      q_{t+1} = q_t + dt * dq/dt evaluated at (q_t, p_{t+1})
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
        drive_dim: int = 0,
    ):
        super().__init__()
        self.node_dim = int(node_dim)
        self.msg_dim = int(msg_dim)
        self.dt = float(dt)
        self.damping = float(damping)

        if self.node_dim % 2 != 0:
            raise ValueError(f"HNN requires even node_dim (got {self.node_dim})")
        self.d = self.node_dim // 2

        in_dim = (2 * self.d) + self.msg_dim + int(drive_dim)
        self.H = _mlp(in_dim, hidden_dim, num_layers, out_dim=1, dropout=dropout)

    def init_state(
        self,
        batch_size: int,
        num_nodes: int,
        device: torch.device,
    ) -> Optional[ModelState]:
        # state.node holds [q,p]
        qp = torch.zeros(num_nodes, 2 * self.d, device=device)
        return ModelState(node=qp, node_prev=None, aux={})

    def forward(
        self,
        state: Optional[ModelState],
        messages: torch.Tensor,
        drive: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[ModelState], Dict[str, torch.Tensor]]:
        assert state is not None, "HNNUpdate is stateful; expected non-None state"
        assert state.node is not None

        qp_raw = state.node
        dt = self.dt

        # HNN needs autograd even during eval (your evaluate_stream_sliced uses torch.no_grad).
        # So: locally re-enable grads for the physics part.
        with torch.enable_grad():
            qp = qp_raw.detach().requires_grad_(True)
            q, p = qp[:, : self.d], qp[:, self.d :]

            if drive is not None:
                drive_in = drive.unsqueeze(-1) if drive.dim() == 1 else drive
                inp = torch.cat([q, p, messages, drive_in], dim=-1)
            else:
                inp = torch.cat([q, p, messages], dim=-1)

            H_per = self.H(inp).squeeze(-1)   # (N,)
            H_tot = H_per.sum()

            create_graph = self.training
            retain = create_graph

            (dH_dqp,) = torch.autograd.grad(
                H_tot, qp,
                create_graph=create_graph,
                retain_graph=retain,
                allow_unused=False,
            )                 

            dH_dq = dH_dqp[:, : self.d]
            dH_dp = dH_dqp[:, self.d :]

            dqdt = dH_dp
            dpdt = -dH_dq

            # Optional damping on momentum
            if self.damping != 0.0:
                dpdt = dpdt - self.damping * p

            # ---- symplectic Euler (kick-drift) ----
            # 1) kick: update momentum
            p_next = p + dt * dpdt

            # 2) drift: recompute dq/dt at (q_t, p_{t+1})
            qp2 = torch.cat([q, p_next], dim=-1).detach().requires_grad_(True)

            q2, p2 = qp2[:, : self.d], qp2[:, self.d :]

            if drive is not None:
                drive_in = drive.unsqueeze(-1) if drive.dim() == 1 else drive
                inp2 = torch.cat([q2, p2, messages, drive_in], dim=-1)
            else:
                inp2 = torch.cat([q2, p2, messages], dim=-1)

            H_per2 = self.H(inp2).squeeze(-1)
            H_tot2 = H_per2.sum()


            (dH_dqp2,) = torch.autograd.grad(
                H_tot2, qp2,
                create_graph=create_graph,
                retain_graph=retain,
                allow_unused=False,
            )            

            dqdt_next = dH_dqp2[:, self.d :]   # dH/dp evaluated at p_next

            # 3) update position using dq/dt at new momentum
            q_next = q + dt * dqdt_next

            qp_next = torch.cat([q_next, p_next], dim=-1)


        next_state = state.clone(detach=False)
        next_state.node = qp_next.detach()

        aux: Dict[str, torch.Tensor] = {
            "H_tot": H_tot.detach(),
            "q_norm": q.norm(dim=-1).mean().detach(),
            "p_norm": p.norm(dim=-1).mean().detach(),
            "dqdt_norm": dqdt.norm(dim=-1).mean().detach(),
            "dpdt_norm": dpdt.norm(dim=-1).mean().detach(),
            "dqdt_next_norm": dqdt_next.norm(dim=-1).mean().detach(),
        }
        return next_state, aux
