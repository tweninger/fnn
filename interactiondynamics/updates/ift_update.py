# updates/ift_update.py
from __future__ import annotations
import math
from pyexpat.errors import messages
from pyexpat.errors import messages
from typing import Dict, Optional, Tuple, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

        # messages come in as msg_dim, but node states live in node_dim, so this maps messages into state space
        self.msg_proj = nn.Linear(self.msg_dim, self.node_dim)

        # kappa hello!!!
        # controls strength of the diffusion term
        # fixed or learnable... if learnable? forces it stay positive w exp/softplus
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

   # we want diffusion strength to be positive, so don't learn kappa directly...
   # learn a raw number then transform to a postive one very sneaky

   # helper just says compute pos scaler kappa + maybe cap it if requested             
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

        if self.kappa_cap: # maybe clamp it for stability... 
            if self.kappa_max is None:
                raise ValueError("kappa_cap=True requires kappa_max not None")  
            k = torch.clamp(k, 0.0, float(self.kappa_max))  

        return k.to(device=h.device, dtype=h.dtype).reshape(())

    # again the initial node state is zeros... like hopfield!
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
        h = state.node # current node memory hello

        # safety check, no NaNs/infs
        assert torch.isfinite(h).all()
        assert torch.isfinite(messages).all()
        # print("DEBUG IFT h max", float(h.abs().max().item()),
        #    "messages max", float(messages.abs().max().item()))        

        assert messages.abs().sum().item() > 0, "IFT messages are all zeros (unexpected)"

        kappa = self._kappa(h) # get kappa, diffusion strength is now a scalar

        # where updater looks for the graph operator
        L = None
        if state.aux is not None:
            L = state.aux.get("L", None)

        if L is None:
            Lh = torch.zeros_like(h) # L is missing? no diffusion
        else:
            Lh = torch.sparse.mm(L, h) # if it exists tho? apply sprase graph operator L to the current node states
        # ^^ central interaction term!!
        # ^^ each node's new state is influenced by how its current state differs/relates across the local graph induced..
        # by the current event bin. this is "field/diffusion" part

        # ratio = (Lh.norm() / (h.norm() + 1e-12)).item()
        # print("[IFT] ||Lh||/||h||", ratio)

        # if L is not None:
        #     assert L.is_sparse
        #     Lc = L.coalesce()
        #     vals = Lc.values()
        #     print("[IFT] L nnz", vals.numel(),
        #         "max|L|", float(vals.abs().max().item()),
        #         "dtype", vals.dtype)            


        # makes message injection live in node-state space
        inj = self.msg_proj(messages)
        # clip it injection norm a bit why not.. don't let it get too big
        #inj_max = 3.0  # tune: 0.5–2.0
        inj = inj.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        #inj = inj * (inj_max / inj_norm).clamp(max=1.0)    

        # print("DEBUG inj max", float(inj.abs().max().item()),
        #    "inj std", float(inj.std().item()))

        # compute derivative like update... big point of the updater
        # gamme part - decay/damping/forgetting 
        # kappa part - diffusion/interaction across the graph operator
        # inj part - new information entering from current messages
        # aka change in hidden state = decay + graph interaciton + incoming signal
        dh = (-self.gamma * h) - (kappa * Lh) + inj 
        h_next = h + self.dt * dh    # euler step... simple discrete time update: take curr state and add a timestep sized change
        # ^^ simpler than HNN symplectic update 

        #DEBUGGGINGGGG
        # if state.aux is None:
        #     state.aux = {}

        # step_idx = int(state.aux.get("debug_step", 0))
        
        # # print first few, then every 1000
        # if step_idx < 5 or step_idx % 1500 == 0:
        #     inj_raw = self.msg_proj(messages)
        #     inj_norm_raw = inj_raw.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        #     inj = inj_raw * (inj_max / inj_norm_raw).clamp(max=1.0)

        #     row_norms = inj_raw.norm(dim=-1).detach().cpu().numpy()
        #     print(
        #         f"p50={np.percentile(row_norms,50):.3f} | "
        #         f"p90={np.percentile(row_norms,90):.3f} | "
        #         f"p95={np.percentile(row_norms,95):.3f} | "
        #         f"p99={np.percentile(row_norms,99):.3f} | "
        #         f"max={row_norms.max():.3f}"
        #     )

        #     with torch.no_grad():
        #         print(
        #             f"[INJ CHECK step={step_idx}] "
        #             f"raw_total={float(inj_raw.norm().item()):.6f} | "
        #             f"clipped_total={float(inj.norm().item()):.6f} | "
        #             f"raw_mean_node={float(inj_norm_raw.mean().item()):.6f} | "
        #             f"frac_clipped={float((inj_norm_raw > inj_max).float().mean().item()):.6f}"
        #         )
        #     with torch.no_grad():
        #         print(
        #             f"[IFT DEBUG] "
        #             f"kappa={float(kappa.item()):.6f} | "
        #             f"||h||={float(h.norm().item()):.6f} | "
        #             f"||messages||={float(messages.norm().item()):.6f} | "
        #             f"||inj||={float(inj.norm().item()):.6f} | "
        #             f"||Lh||={float(Lh.norm().item()):.6f} | "
        #             f"||dh||={float(dh.norm().item()):.6f} | "
        #             f"||h_next||={float(h_next.norm().item()):.6f}"
        #         )

                #     print(
                #         f"[IFT DEBUG] "
                #         f"max|h|={float(h.abs().max().item()):.6f} | "
                #         f"max|messages|={float(messages.abs().max().item()):.6f} | "
                #         f"max|inj|={float(inj.abs().max().item()):.6f} | "
                #         f"max|Lh|={float(Lh.abs().max().item()):.6f} | "
                #         f"max|dh|={float(dh.abs().max().item()):.6f} | "
                #         f"max|h_next|={float(h_next.abs().max().item()):.6f}"
                #     )
                # decay_term = -self.gamma * h
                # diff_term = -(kappa * Lh)
                # inj_term = inj

                # with torch.no_grad():
                #     print(
                #         f"[IFT TERMS] "
                #         f"||decay||={float(decay_term.norm().item()):.6f} | "
                #         f"||diff||={float(diff_term.norm().item()):.6f} | "
                #         f"||inj||={float(inj_term.norm().item()):.6f}"
                #     )

                #state.aux["debug_step"] = step_idx + 1
        
        next_state = ModelState( # return next state, new node memory is h_next + aux info, so L and metadata can continue being carried around
            node=h_next,
            aux=dict(state.aux) if state.aux is not None else {}
        )

        # #still part of ^^ inj_message debugging
        # if next_state.aux is None:
        #     next_state.aux = {}
        # next_state.aux["debug_step"] = step_idx + 1


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

        # stores useful diagnostics like..
        # diffusion strength, state size, message inejction size, graph-diffusion term size
        # for debugging and sweep analysis 
        aux = {
            "kappa": kappa.detach(),
            "h_norm": h_next.norm(dim=-1).mean().detach(),
            "inj_norm": inj.norm(dim=-1).mean().detach(),
            "Lh_norm": Lh.norm(dim=-1).mean().detach(),
        }
        return next_state, aux
