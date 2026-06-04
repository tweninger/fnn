from typing import Optional
import torch
import torch.nn as nn

from interactiondynamics.core.events import EventBatch
from interactiondynamics.core.interfaces import ModelState, ScoringHead


class DotProductScorer(ScoringHead):
    """
    score(u, v) = <h_u, h_v>
    """

    def forward(self, state: Optional[ModelState], candidate_events: EventBatch) -> torch.Tensor:
        assert state is not None and state.node is not None, "DotProductScorer requires state.node."

        H = state.node  # (N, d)
        src = candidate_events.src.to(device=H.device, dtype=torch.long)
        dst = candidate_events.dst.to(device=H.device, dtype=torch.long)

        h_src = H[src]  # (M, d)
        h_dst = H[dst]  # (M, d)
        return (h_src * h_dst).sum(dim=-1)  # (M,)

class MLPEdgeScorer(ScoringHead):
    """
    score(u,v,e,t) = MLP([h_u, h_v, e, phi(t)])
    """

    def __init__(
        self,
        node_dim: int,
        event_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_time: bool = False,
        time_emb_dim: int = 32,
    ):
        super().__init__()
        self.node_dim = int(node_dim)
        self.event_dim = int(event_dim)

        self.use_time = bool(use_time)
        self.time_emb_dim = int(time_emb_dim)

        self.time_mlp: Optional[nn.Module] = None
        extra = 0
        if self.use_time:
            self.time_mlp = nn.Sequential(
                nn.Linear(1, self.time_emb_dim),
                nn.ReLU(),
                nn.Linear(self.time_emb_dim, self.time_emb_dim),
            )
            extra = self.time_emb_dim

        in_dim = 2 * self.node_dim + self.event_dim + extra

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        ) 

    def forward(self, state: Optional[ModelState], candidate_events: EventBatch) -> torch.Tensor:
        assert state is not None and state.node is not None, "MLPEdgeScorer requires state.node."
        H = state.node  # (N, d)

        src = candidate_events.src.to(device=H.device, dtype=torch.long)
        dst = candidate_events.dst.to(device=H.device, dtype=torch.long)
        h_src = H[src]
        h_dst = H[dst]

        # print("DEBUG scorer src shape", candidate_events.src.shape, "dst shape", candidate_events.dst.shape)
        # print("DEBUG first 10 src", candidate_events.src[:10].tolist())
        # print("DEBUG first 10 dst", candidate_events.dst[:10].tolist())   

        pieces = [h_src, h_dst]

        # edge / event features
        if self.event_dim > 0:
            if candidate_events.features is None:
                e = torch.zeros((src.numel(), self.event_dim), device=H.device, dtype=H.dtype)
            else:
                assert candidate_events.features.dim() == 2
                assert candidate_events.features.size(1) == self.event_dim, \
                    f"features dim {candidate_events.features.size(1)} != event_dim {self.event_dim}"
                e = candidate_events.features.to(device=H.device, dtype=H.dtype)
            pieces.append(e)

        # time features
        if self.use_time:
            assert candidate_events.t is not None, "use_time=True but candidate_events.t is None"
            t = candidate_events.t.to(device=H.device, dtype=H.dtype).view(-1, 1)
            assert self.time_mlp is not None
            pieces.append(self.time_mlp(t))
        
        x = torch.cat(pieces, dim=-1)
        return self.mlp(x).squeeze(-1)


class IFTSecondOrderLinearHVForceScorer(ScoringHead):
    """
    Diagnostic second-order readout:
      delta_pred = Linear([position_t, velocity_t, force_t])
      y_pred = y_t + delta_pred

    The training/eval pipeline reconstructs `y_pred` by adding `delta_pred`
    back to the previous target when `prediction_mode=state_plus_delta`.
    """

    def __init__(
        self,
        *,
        init_mode: str = "zero",
        init_scale: float = 0.01,
        near_ar1_coeffs: Optional[tuple[float, float, float, float]] = None,
        oracle_coeffs: Optional[tuple[float, float, float]] = None,
        trainable: bool = True,
    ):
        super().__init__()
        self.linear = nn.Linear(3, 1)
        if init_mode == "zero" and oracle_coeffs is not None:
            init_mode = "oracle"
        self.init_mode = str(init_mode)
        self.init_scale = float(init_scale)
        self._oracle_coeffs = None if oracle_coeffs is None else (
            float(oracle_coeffs[0] + oracle_coeffs[1] - 1.0),
            float(-oracle_coeffs[1]),
            float(oracle_coeffs[2]),
            0.0,
        )
        self._initialize_weights(near_ar1_coeffs=near_ar1_coeffs)
        if not trainable:
            for param in self.linear.parameters():
                param.requires_grad_(False)

    def _initialize_weights(
        self,
        *,
        near_ar1_coeffs: Optional[tuple[float, float, float, float]],
    ) -> None:
        with torch.no_grad():
            if self.init_mode == "small_random":
                nn.init.normal_(self.linear.weight, mean=0.0, std=self.init_scale)
                nn.init.normal_(self.linear.bias, mean=0.0, std=self.init_scale)
                return

            self.linear.weight.zero_()
            self.linear.bias.zero_()
            if self.init_mode == "near_ar1":
                if near_ar1_coeffs is None:
                    raise ValueError("near_ar1 init requires near_ar1_coeffs")
                self.linear.weight[0, 0] = float(near_ar1_coeffs[0])
                self.linear.weight[0, 1] = float(near_ar1_coeffs[1])
                self.linear.weight[0, 2] = float(near_ar1_coeffs[2])
                self.linear.bias[0] = float(near_ar1_coeffs[3])
                return
            if self.init_mode == "oracle":
                if self._oracle_coeffs is None:
                    raise ValueError("oracle init requires oracle_coeffs")
                self.linear.weight[0, 0] = self._oracle_coeffs[0]
                self.linear.weight[0, 1] = self._oracle_coeffs[1]
                self.linear.weight[0, 2] = self._oracle_coeffs[2]
                self.linear.bias[0] = self._oracle_coeffs[3]
                return
            if self.init_mode != "zero":
                raise ValueError(f"Unsupported init_mode={self.init_mode!r}")

    def coefficient_dict(self) -> dict[str, float]:
        return {
            "w_y": float(self.linear.weight[0, 0].detach().item()),
            "w_v": float(self.linear.weight[0, 1].detach().item()),
            "w_drive": float(self.linear.weight[0, 2].detach().item()),
            "bias": float(self.linear.bias[0].detach().item()),
        }

    def oracle_coefficient_dict(self) -> Optional[dict[str, float]]:
        if self._oracle_coeffs is None:
            return None
        return {
            "w_y_oracle": self._oracle_coeffs[0],
            "w_v_oracle": self._oracle_coeffs[1],
            "w_drive_oracle": self._oracle_coeffs[2],
            "bias_oracle": self._oracle_coeffs[3],
        }

    def forward(self, state: Optional[ModelState], candidate_events: EventBatch) -> torch.Tensor:
        assert state is not None and state.aux is not None, "IFTSecondOrderLinearHVForceScorer requires state.aux."
        pos = state.aux.get("ift_readout_position_scalar")
        vel = state.aux.get("ift_readout_velocity_scalar")
        force = state.aux.get("ift_readout_force_scalar")
        assert torch.is_tensor(pos) and torch.is_tensor(vel) and torch.is_tensor(force), (
            "IFTSecondOrderLinearHVForceScorer requires readout scalars in state.aux"
        )
        dst = candidate_events.dst.to(device=pos.device, dtype=torch.long)
        features = torch.stack([pos[dst], vel[dst], force[dst]], dim=-1)
        return self.linear(features).squeeze(-1)
