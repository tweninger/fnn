# updates/ift_update.py
from __future__ import annotations
import math
from pyexpat.errors import messages
from pyexpat.errors import messages
from typing import Dict, Optional, Tuple, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.interfaces import UpdateLaw, ModelState

KappaParam = Literal["exp", "softplus"]
class IFTDiffusionUpdate(UpdateLaw):
    """
    h_{t+1} = h_t + dt * ( -gamma*h_t - kappa*(L_t h_t) + W(messages) )

    L_t is induced from the bin events and stashed in state.aux["L"].
    """
    def __init__(
        self,
        node_dim: int,
        msg_dim: int,
        dt: float = 0.05,
        gamma: float = 0.0,
        kappa: float = 1.0,
        learn_kappa: bool = True,
        kappa_param: KappaParam = "softplus",
        kappa_cap: bool = False,
        kappa_max: float | None = None,
    ):
        super().__init__()
        self.node_dim = int(node_dim)
        self.msg_dim = int(msg_dim)
        self.dt = float(dt)
        self.gamma = float(gamma)
        self.learn_kappa = bool(learn_kappa)
        self.kappa_param: KappaParam = kappa_param
        self.kappa_cap = bool(kappa_cap)
        self.kappa_max = kappa_max


        self.msg_proj = nn.Linear(self.msg_dim, self.node_dim)

        if self.learn_kappa:
            # Unconstrained scalar parameter; transformed to positive in forward()
            if self.kappa_param == "exp":
                init_raw = float(kappa)
            else:
                # softplus(raw)=kappa  =>  raw=inv_softplus(kappa)
                # inv_softplus(y) = log(exp(y)-1)
                k0 = float(kappa)
                k0 = max(k0, 1e-8)  
                init_raw = float(torch.log(torch.expm1(torch.tensor(k0))).item())
            self.kappa_raw = nn.Parameter(torch.tensor(init_raw))
        else:
            # Constant scalar tensor that follows device moves
            self.register_buffer("kappa_const", torch.tensor(float(kappa)))

                
    def _kappa(self, h: torch.Tensor) -> torch.Tensor:
        """Return positive scalar kappa as a 0-dim tensor on h's device/dtype."""
        if self.learn_kappa:
            if self.kappa_param == "exp":
                k = torch.exp(self.kappa_raw)
            elif self.kappa_param == "softplus":
                k = F.softplus(self.kappa_raw)
            else:
                raise ValueError(f"unknown kappa_param={self.kappa_param}")
        else:
            k = torch.as_tensor(self.kappa_const, device=h.device, dtype=h.dtype)

        if self.kappa_cap:
            if self.kappa_max is None:
                raise ValueError("kappa_cap=True requires kappa_max not None")  
            k = torch.clamp(k, 0.0, float(self.kappa_max))  

        return k.to(device=h.device, dtype=h.dtype).reshape(())


    def init_state(self, batch_size: int, num_nodes: int, device: torch.device) -> Optional[ModelState]:
        h = torch.zeros((num_nodes, self.node_dim), device=device)
        return ModelState(node=h, aux={})

    def forward(
        self,
        state: Optional[ModelState],
        messages: torch.Tensor,                 # (N, msg_dim)
        drive: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[ModelState], Dict]:
        assert state is not None and state.node is not None, "IFTDiffusionUpdate requires state.node"
        h = state.node

        assert torch.isfinite(h).all()
        assert torch.isfinite(messages).all()
        # print("DEBUG IFT h max", float(h.abs().max().item()),
        #    "messages max", float(messages.abs().max().item()))        

        assert messages.abs().sum().item() > 0, "IFT messages are all zeros (unexpected)"

        kappa = self._kappa(h)

        L = None
        if state.aux is not None:
            L = state.aux.get("L", None)

        if L is None:
            Lh = torch.zeros_like(h)
        else:
            Lh = torch.sparse.mm(L, h)

        # ratio = (Lh.norm() / (h.norm() + 1e-12)).item()
        # print("[IFT] ||Lh||/||h||", ratio)

        # if L is not None:
        #     assert L.is_sparse
        #     Lc = L.coalesce()
        #     vals = Lc.values()
        #     print("[IFT] L nnz", vals.numel(),
        #         "max|L|", float(vals.abs().max().item()),
        #         "dtype", vals.dtype)            

        inj = self.msg_proj(messages)

        inj_max = 1.0  # tune: 0.5–2.0
        inj_norm = inj.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        inj = inj * (inj_max / inj_norm).clamp(max=1.0)    

        # print("DEBUG inj max", float(inj.abs().max().item()),
        #    "inj std", float(inj.std().item()))

        dh = (-self.gamma * h) - (kappa * Lh) + inj
        h_next = h + self.dt * dh    

        next_state = ModelState(
            node=h_next,
            aux=dict(state.aux) if state.aux is not None else {}
        )

        def _stat(name, x):
            mx = float(x.abs().max().item())
            fin = bool(torch.isfinite(x).all().item())
            # print(f"[IFT] {name}: finite={fin} max|.|={mx}")

        _stat("h", h)
        _stat("messages", messages)
        _stat("inj", inj)
        _stat("Lh", Lh)
        _stat("dh", dh)
        _stat("h_next", h_next)

        # H0 = state.node
        # H1 = next_state.node
        # dH = H1 - H0        

        # print("||H||", float(H0.norm()), "||dH||", float(dH.norm()),
        #    "ratio ||dH||/||H||", float(dH.norm() / (H0.norm() + 1e-12)))
                
        # with torch.no_grad():
        #     print("||messages||", float(messages.norm()))        

        aux = {
            "kappa": kappa.detach(),
            "h_norm": h_next.norm(dim=-1).mean().detach(),
            "inj_norm": inj.norm(dim=-1).mean().detach(),
            "Lh_norm": Lh.norm(dim=-1).mean().detach(),
        }
        return next_state, aux
