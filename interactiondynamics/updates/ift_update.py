# updates/ift_update.py
from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from interactiondynamics.core.interfaces import ModelState, UpdateLaw

KappaParam = Literal["exp", "softplus"]
IFTForcingMode = Literal[
    "generic_mlp",
    "linear_event",
    "gated_linear_event",
    "direct_scalar",
    "gated_direct_scalar",
]
IFTVelocityInitMode = Literal["zero", "learned", "finite_difference"]


def _inv_softplus(value: float) -> float:
    clamped = max(float(value), 1e-8)
    return float(torch.log(torch.expm1(torch.tensor(clamped))).item())


def _laplacian_stats(
    L: Optional[torch.Tensor],
    num_nodes: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, torch.Tensor]:
    zero = torch.zeros((), device=device, dtype=dtype)
    if L is None:
        return zero, zero, 0.0, zero, zero

    Lc = L.coalesce()
    vals = Lc.values().to(dtype=dtype)
    idx = Lc.indices()
    nnz = float(vals.numel())
    density = torch.tensor(nnz / float(max(num_nodes * num_nodes, 1)), device=device, dtype=dtype)
    diag_mask = idx[0] == idx[1]
    offdiag_mask = ~diag_mask
    diag_mean = vals[diag_mask].mean() if torch.any(diag_mask) else zero
    offdiag_abs_mean = vals[offdiag_mask].abs().mean() if torch.any(offdiag_mask) else zero
    return density, diag_mean, nnz, offdiag_abs_mean, vals


class IFTForcingEncoder(nn.Module):
    def __init__(
        self,
        *,
        msg_dim: int,
        node_dim: int,
        event_dim: int,
        mode: IFTForcingMode = "generic_mlp",
        drive_feature_idx: Optional[int] = None,
        force_scale_init: float = 1.0,
        force_learn_scale: bool = True,
        force_target_dim: Optional[int] = 0,
    ):
        super().__init__()
        self.msg_dim = int(msg_dim)
        self.node_dim = int(node_dim)
        self.event_dim = int(event_dim)
        self.mode: IFTForcingMode = mode
        self.drive_feature_idx = None if drive_feature_idx is None else int(drive_feature_idx)
        self.force_target_dim = None if force_target_dim is None else int(force_target_dim)

        if self.mode not in {
            "generic_mlp",
            "linear_event",
            "gated_linear_event",
            "direct_scalar",
            "gated_direct_scalar",
        }:
            raise ValueError(f"unknown forcing mode={self.mode}")
        if self.mode != "generic_mlp" and self.event_dim <= 0:
            raise ValueError("structured IFT forcing requires event_dim > 0")
        if self.drive_feature_idx is not None and not 0 <= self.drive_feature_idx < max(self.event_dim, 1):
            raise ValueError("ift_drive_feature_idx is out of range for event_dim")
        if self.force_target_dim is not None and not 0 <= self.force_target_dim < self.node_dim:
            raise ValueError("ift_force_target_dim must be in [0, node_dim)")

        self.msg_proj = nn.Linear(self.msg_dim, self.node_dim)
        self.event_linear = nn.Linear(self.event_dim, self.node_dim, bias=False) if self.event_dim > 0 else None
        self.gate_mlp = None
        if self.mode in {"gated_direct_scalar", "gated_linear_event"}:
            hidden = max(8, self.event_dim)
            self.gate_mlp = nn.Sequential(
                nn.Linear(self.event_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 1),
            )

        if self.force_target_dim is None:
            self.force_vector = nn.Parameter(torch.randn(self.node_dim) * 0.02)
        else:
            basis = torch.zeros((self.node_dim,), dtype=torch.float32)
            basis[self.force_target_dim] = 1.0
            self.register_buffer("force_basis", basis)
            self.force_vector = None

        if force_learn_scale:
            self.force_scale = nn.Parameter(torch.tensor(float(force_scale_init), dtype=torch.float32))
        else:
            self.register_buffer("force_scale", torch.tensor(float(force_scale_init), dtype=torch.float32))

    def _structured_scalar_force(self, raw_features: torch.Tensor) -> torch.Tensor:
        if self.drive_feature_idx is None:
            raise ValueError("direct scalar IFT forcing requires ift_drive_feature_idx")

        drive_scalar = raw_features[:, self.drive_feature_idx : self.drive_feature_idx + 1]
        scale = cast(torch.Tensor, self.force_scale).to(
            device=raw_features.device,
            dtype=raw_features.dtype,
        )
        if self.force_target_dim is None:
            assert self.force_vector is not None
            vector = self.force_vector.to(device=raw_features.device, dtype=raw_features.dtype)
        else:
            vector = cast(torch.Tensor, self.force_basis).to(
                device=raw_features.device,
                dtype=raw_features.dtype,
            )
        return drive_scalar * scale * vector[None, :]

    def forward(
        self,
        messages: torch.Tensor,
        state: Optional[ModelState],
    ) -> torch.Tensor:
        if self.mode == "generic_mlp":
            return self.msg_proj(messages)

        raw_features: Optional[torch.Tensor] = None
        if state is not None and state.aux is not None:
            force_features = state.aux.get("ift_force_features")
            if torch.is_tensor(force_features):
                raw_features = force_features.to(device=messages.device, dtype=messages.dtype)
            elif self.mode in {"direct_scalar", "gated_direct_scalar"}:
                direct_drive = state.aux.get("ift_direct_drive")
                if torch.is_tensor(direct_drive):
                    raw_features = direct_drive.to(device=messages.device, dtype=messages.dtype)

        if raw_features is None:
            width = max(self.event_dim, 1)
            raw_features = torch.zeros((messages.size(0), width), device=messages.device, dtype=messages.dtype)

        if self.mode in {"linear_event", "gated_linear_event"}:
            assert self.event_linear is not None
            scale = self.force_scale.to(device=messages.device, dtype=messages.dtype)
            force = scale * self.event_linear(raw_features[:, : self.event_dim])
            if self.mode == "gated_linear_event":
                assert self.gate_mlp is not None
                gate = torch.sigmoid(self.gate_mlp(raw_features[:, : self.event_dim]))
                force = gate * force
            return force

        force = self._structured_scalar_force(raw_features)
        if self.mode == "gated_direct_scalar":
            assert self.gate_mlp is not None
            gate = torch.sigmoid(self.gate_mlp(raw_features[:, : self.event_dim]))
            force = gate * force
        return force


class IFTDiffusionUpdate(UpdateLaw):
    """
    First-order IFT update:
      h_{t+1} = h_t + dt * (-gamma * h_t - kappa * L_t h_t + force_t)
    """

    def __init__(
        self,
        node_dim: int,
        msg_dim: int,
        *,
        event_dim: int = 0,
        dt: float = 0.05,
        gamma: float = 0.0,
        kappa: float = 1.0,
        learn_kappa: bool = True,
        kappa_param: KappaParam = "softplus",
        kappa_cap: bool = False,
        kappa_max: float | None = None,
        zero_injection: bool = False,
        inj_clip: float | None = 1.0,
        direct_drive: bool = False,
        forcing_mode: IFTForcingMode = "generic_mlp",
        drive_feature_idx: Optional[int] = None,
        force_scale_init: float = 1.0,
        force_learn_scale: bool = True,
        force_target_dim: Optional[int] = 0,
    ):
        super().__init__()
        self.node_dim = int(node_dim)
        self.msg_dim = int(msg_dim)
        self.event_dim = int(event_dim)
        self.dt = float(dt)
        self.gamma = float(gamma)
        self.learn_kappa = bool(learn_kappa)
        self.kappa_param: KappaParam = kappa_param
        self.kappa_cap = bool(kappa_cap)
        self.kappa_max = kappa_max
        self.zero_injection = bool(zero_injection)
        self.inj_clip = None if inj_clip is None else float(inj_clip)
        self.direct_drive = bool(direct_drive)
        self.forcing_mode: IFTForcingMode = forcing_mode

        self.force_encoder = IFTForcingEncoder(
            msg_dim=self.msg_dim,
            node_dim=self.node_dim,
            event_dim=self.event_dim,
            mode=self.forcing_mode,
            drive_feature_idx=drive_feature_idx,
            force_scale_init=force_scale_init,
            force_learn_scale=force_learn_scale,
            force_target_dim=force_target_dim,
        )

        if self.learn_kappa:
            init_raw = float(kappa) if self.kappa_param == "exp" else _inv_softplus(kappa)
            self.kappa_raw = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))
        else:
            self.register_buffer("kappa_const", torch.tensor(float(kappa), dtype=torch.float32))

    def _kappa(self, ref: torch.Tensor) -> torch.Tensor:
        if self.learn_kappa:
            if self.kappa_param == "exp":
                k = torch.exp(self.kappa_raw)
            elif self.kappa_param == "softplus":
                k = F.softplus(self.kappa_raw)
            else:
                raise ValueError(f"unknown kappa_param={self.kappa_param}")
        else:
            k = torch.as_tensor(self.kappa_const, device=ref.device, dtype=ref.dtype)

        if self.kappa_cap:
            if self.kappa_max is None:
                raise ValueError("kappa_cap=True requires kappa_max not None")
            k = torch.clamp(k, 0.0, float(self.kappa_max))
        return k.to(device=ref.device, dtype=ref.dtype).reshape(())

    def init_state(self, batch_size: int, num_nodes: int, device: torch.device) -> Optional[ModelState]:
        h = torch.randn((num_nodes, self.node_dim), device=device) * 0.02
        return ModelState(node=h, aux={})

    def _force(self, state: Optional[ModelState], messages: torch.Tensor) -> torch.Tensor:
        if self.direct_drive and state is not None and state.aux is not None and "ift_direct_drive" in state.aux:
            direct_drive = state.aux["ift_direct_drive"].to(device=messages.device, dtype=messages.dtype)
            scale = cast(torch.Tensor, self.force_encoder.force_scale).to(
                device=messages.device,
                dtype=messages.dtype,
            )
            if self.force_encoder.force_target_dim is None:
                assert self.force_encoder.force_vector is not None
                vector = self.force_encoder.force_vector.to(device=messages.device, dtype=messages.dtype)
            else:
                vector = cast(torch.Tensor, self.force_encoder.force_basis).to(
                    device=messages.device,
                    dtype=messages.dtype,
                )
            return direct_drive * scale * vector[None, :]
        return self.force_encoder(messages, state)

    def forward(
        self,
        state: Optional[ModelState],
        messages: torch.Tensor,
        drive: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[ModelState], Dict]:
        assert state is not None and state.node is not None, "IFTDiffusionUpdate requires state.node"
        h = state.node

        kappa = self._kappa(h)
        L = None if state.aux is None else state.aux.get("L")
        if L is None:
            Lh = torch.zeros_like(h)
        else:
            Lh = torch.sparse.mm(L, h)

        force = self._force(state, messages)
        if self.inj_clip is not None:
            force_norm = force.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            force = force * (float(self.inj_clip) / force_norm).clamp(max=1.0)
        if self.zero_injection:
            force = torch.zeros_like(force)

        diffusion = kappa * Lh
        decay = self.gamma * h
        dh = (-decay) - diffusion + force
        h_next = h + self.dt * dh
        next_state = ModelState(node=h_next, aux=dict(state.aux) if state.aux is not None else {})

        h_norm = h.norm(dim=-1).mean()
        Lh_norm = Lh.norm(dim=-1).mean()
        force_norm = force.norm(dim=-1).mean()
        dh_norm = dh.norm(dim=-1).mean()
        diffusion_term_norm = diffusion.norm(dim=-1).mean()
        decay_term_norm = decay.norm(dim=-1).mean()
        relative_diffusion = diffusion_term_norm / (force_norm + 1e-8)
        relative_update = (h_next - h).norm(dim=-1).mean() / (h.norm(dim=-1).mean() + 1e-8)
        L_density, L_diag_mean, L_nnz, L_offdiag_abs_mean, _ = _laplacian_stats(
            L, h.size(0), h.device, h.dtype
        )

        aux = {
            "kappa": kappa.detach(),
            "learned_kappa": kappa.detach(),
            "gamma": h.new_tensor(self.gamma).detach(),
            "dt": h.new_tensor(self.dt).detach(),
            "h_norm": h_norm.detach(),
            "Lh_norm": Lh_norm.detach(),
            "force_norm": force_norm.detach(),
            "inj_norm": force_norm.detach(),
            "dh_norm": dh_norm.detach(),
            "diffusion_term_norm": diffusion_term_norm.detach(),
            "injection_term_norm": force_norm.detach(),
            "decay_term_norm": decay_term_norm.detach(),
            "relative_diffusion": relative_diffusion.detach(),
            "relative_update": relative_update.detach(),
            "L_nnz": h.new_tensor(L_nnz).detach(),
            "L_density": L_density.detach(),
            "L_diag_mean": L_diag_mean.detach(),
            "L_offdiag_abs_mean": L_offdiag_abs_mean.detach(),
        }
        return next_state, aux


class IFTSecondOrderUpdate(UpdateLaw):
    """
    Second-order IFT update with explicit velocity memory.

      v_{t+1} = alpha * v_t - dt * (gamma * v_t + kappa * L_t h_t) + dt * force_t
      h_{t+1} = h_t + dt * v_{t+1}
    """

    def __init__(
        self,
        node_dim: int,
        msg_dim: int,
        *,
        event_dim: int = 0,
        alpha: float = 1.0,
        dt: float = 0.1,
        gamma: float = 0.0,
        kappa: float = 1.0,
        learn_params: bool = True,
        zero_injection: bool = False,
        inj_clip: float | None = 1.0,
        direct_drive: bool = False,
        forcing_mode: IFTForcingMode = "generic_mlp",
        drive_feature_idx: Optional[int] = None,
        force_scale_init: float = 1.0,
        force_learn_scale: bool = True,
        force_target_dim: Optional[int] = 0,
        velocity_init_mode: IFTVelocityInitMode = "finite_difference",
        velocity_supervision: bool = True,
        velocity_loss_weight: float = 0.01,
        internal_velocity_loss_weight: float = 0.0,
        readout_mode: str = "default",
        velocity_teacher_forcing: bool = False,
    ):
        super().__init__()
        self.node_dim = int(node_dim)
        self.msg_dim = int(msg_dim)
        self.event_dim = int(event_dim)
        self.learn_params = bool(learn_params)
        self.zero_injection = bool(zero_injection)
        self.inj_clip = None if inj_clip is None else float(inj_clip)
        self.direct_drive = bool(direct_drive)
        self.forcing_mode: IFTForcingMode = forcing_mode
        self.velocity_init_mode: IFTVelocityInitMode = velocity_init_mode
        self.velocity_supervision = bool(velocity_supervision)
        self.velocity_loss_weight = float(velocity_loss_weight)
        self.internal_velocity_loss_weight = float(internal_velocity_loss_weight)
        self.readout_mode = str(readout_mode)
        self.velocity_teacher_forcing = bool(velocity_teacher_forcing)

        self.force_encoder = IFTForcingEncoder(
            msg_dim=self.msg_dim,
            node_dim=self.node_dim,
            event_dim=self.event_dim,
            mode=self.forcing_mode,
            drive_feature_idx=drive_feature_idx,
            force_scale_init=force_scale_init,
            force_learn_scale=force_learn_scale,
            force_target_dim=force_target_dim,
        )
        self.velocity_encoder = nn.Linear(1, self.node_dim)
        self.velocity_decoder = nn.Linear(self.node_dim, 1)
        self.velocity_init = nn.Parameter(torch.zeros((self.node_dim,), dtype=torch.float32))

        if self.learn_params:
            self.alpha_param = nn.Parameter(torch.tensor(float(alpha), dtype=torch.float32))
            self.dt_raw = nn.Parameter(torch.tensor(_inv_softplus(dt), dtype=torch.float32))
            self.gamma_raw = nn.Parameter(torch.tensor(_inv_softplus(max(gamma, 1e-6)), dtype=torch.float32))
            self.kappa_raw = nn.Parameter(torch.tensor(_inv_softplus(kappa), dtype=torch.float32))
        else:
            self.register_buffer("alpha_const", torch.tensor(float(alpha), dtype=torch.float32))
            self.register_buffer("dt_const", torch.tensor(float(dt), dtype=torch.float32))
            self.register_buffer("gamma_const", torch.tensor(float(gamma), dtype=torch.float32))
            self.register_buffer("kappa_const", torch.tensor(float(kappa), dtype=torch.float32))

    def _alpha(self, ref: torch.Tensor) -> torch.Tensor:
        alpha = cast(torch.Tensor, self.alpha_param if self.learn_params else self.alpha_const)
        return alpha.to(device=ref.device, dtype=ref.dtype).reshape(())

    def _positive(self, name: str, ref: torch.Tensor) -> torch.Tensor:
        if self.learn_params:
            raw = getattr(self, f"{name}_raw")
            value = F.softplus(raw)
        else:
            value = getattr(self, f"{name}_const")
        return torch.as_tensor(value, device=ref.device, dtype=ref.dtype).reshape(())

    def init_state(self, batch_size: int, num_nodes: int, device: torch.device) -> Optional[ModelState]:
        h = torch.randn((num_nodes, self.node_dim), device=device) * 0.02
        return ModelState(node=h, node_prev=None, aux={"ift_velocity_initialized": False})

    def _force(self, state: Optional[ModelState], messages: torch.Tensor) -> torch.Tensor:
        if self.direct_drive and state is not None and state.aux is not None and "ift_direct_drive" in state.aux:
            direct_drive = state.aux["ift_direct_drive"].to(device=messages.device, dtype=messages.dtype)
            scale = cast(torch.Tensor, self.force_encoder.force_scale).to(
                device=messages.device,
                dtype=messages.dtype,
            )
            if self.force_encoder.force_target_dim is None:
                assert self.force_encoder.force_vector is not None
                vector = self.force_encoder.force_vector.to(device=messages.device, dtype=messages.dtype)
            else:
                vector = cast(torch.Tensor, self.force_encoder.force_basis).to(
                    device=messages.device,
                    dtype=messages.dtype,
                )
            return direct_drive * scale * vector[None, :]
        return self.force_encoder(messages, state)

    def _initial_velocity(self, state: Optional[ModelState], h: torch.Tensor) -> torch.Tensor:
        if self.velocity_init_mode == "zero":
            return torch.zeros_like(h)
        if self.velocity_init_mode == "learned":
            init = self.velocity_init.to(device=h.device, dtype=h.dtype)
            return init.unsqueeze(0).expand_as(h)

        if state is not None and state.aux is not None:
            observed = state.aux.get("ift_state_observed_target")
            prev_observed = state.aux.get("ift_prev_observed_target")
            if torch.is_tensor(observed) and torch.is_tensor(prev_observed):
                curr_obs = observed.to(device=h.device, dtype=h.dtype)
                prev_obs = prev_observed.to(device=h.device, dtype=h.dtype)
                if curr_obs.dim() == 1:
                    curr_obs = curr_obs.unsqueeze(-1)
                if prev_obs.dim() == 1:
                    prev_obs = prev_obs.unsqueeze(-1)
                if curr_obs.shape[0] == h.shape[0] and prev_obs.shape == curr_obs.shape:
                    obs_delta = curr_obs[:, :1] - prev_obs[:, :1]
                    return self.velocity_encoder(obs_delta)

            prev_h = state.aux.get("ift_prev_h")
            if torch.is_tensor(prev_h):
                prev_h_t = prev_h.to(device=h.device, dtype=h.dtype)
                if prev_h_t.shape == h.shape:
                    return h - prev_h_t
        return torch.zeros_like(h)

    def _scalar_target_feature(
        self,
        state: Optional[ModelState],
        key: str,
        h: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if state is None or state.aux is None:
            return None
        value = state.aux.get(key)
        if not torch.is_tensor(value):
            return None
        value_t = value.to(device=h.device, dtype=h.dtype)
        if value_t.dim() == 2 and value_t.size(-1) == 1:
            value_t = value_t.squeeze(-1)
        if value_t.dim() != 1 or value_t.size(0) != h.size(0):
            return None
        return value_t

    def _force_scalar(self, state: Optional[ModelState], h: torch.Tensor) -> torch.Tensor:
        if state is not None and state.aux is not None:
            force_features = state.aux.get("ift_force_features")
            if torch.is_tensor(force_features):
                ff = force_features.to(device=h.device, dtype=h.dtype)
                if ff.dim() == 2 and ff.size(0) == h.size(0) and ff.size(1) > 0:
                    idx = self.force_encoder.drive_feature_idx or 0
                    idx = max(0, min(idx, ff.size(1) - 1))
                    scalar = ff[:, idx]
                    if ff.size(1) > idx + 1:
                        gate_col = ff[:, idx + 1]
                        if torch.all((gate_col == 0) | (gate_col == 1)):
                            scalar = scalar * gate_col
                    return scalar
            direct_drive = state.aux.get("ift_direct_drive")
            if torch.is_tensor(direct_drive):
                dd = direct_drive.to(device=h.device, dtype=h.dtype)
                if dd.dim() == 2 and dd.size(-1) == 1:
                    return dd.squeeze(-1)
        return torch.zeros((h.size(0),), device=h.device, dtype=h.dtype)

    def forward(
        self,
        state: Optional[ModelState],
        messages: torch.Tensor,
        drive: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[ModelState], Dict]:
        assert state is not None and state.node is not None, "IFTSecondOrderUpdate requires state.node"
        h = state.node
        velocity_initialized = bool(state.aux.get("ift_velocity_initialized", False)) if state.aux else False
        if state.node_prev is None or not velocity_initialized:
            v = self._initial_velocity(state, h)
        else:
            v = state.node_prev

        readout_position = self._scalar_target_feature(state, "ift_state_observed_target", h)
        if readout_position is None:
            readout_position = self._scalar_target_feature(state, "ift_position_scalar", h)
        if readout_position is None:
            readout_position = h[:, 0]

        prior_velocity_scalar = self._scalar_target_feature(state, "ift_velocity_scalar", h)
        true_velocity = None
        prev_observed = self._scalar_target_feature(state, "ift_prev_observed_target", h)
        if readout_position is not None and prev_observed is not None:
            true_velocity = readout_position - prev_observed
        if prior_velocity_scalar is None:
            if state.node_prev is not None and state.node_prev.shape == h.shape:
                prior_velocity_scalar = state.node_prev[:, 0]
            else:
                prior_velocity_scalar = torch.zeros((h.size(0),), device=h.device, dtype=h.dtype)
        readout_force = self._force_scalar(state, h)

        alpha = self._alpha(h)
        dt = self._positive("dt", h)
        gamma = self._positive("gamma", h)
        kappa = self._positive("kappa", h)

        L = None if state.aux is None else state.aux.get("L")
        if L is None:
            Lh = torch.zeros_like(h)
        else:
            Lh = torch.sparse.mm(L, h)

        force = self._force(state, messages)
        if self.inj_clip is not None:
            force_norm = force.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            force = force * (float(self.inj_clip) / force_norm).clamp(max=1.0)
        if self.zero_injection:
            force = torch.zeros_like(force)

        diffusion = kappa * Lh
        damping = gamma * v
        v_next = alpha * v - dt * (damping + diffusion) + dt * force
        h_next = h + dt * v_next
        decoded_velocity = self.velocity_decoder(v_next).squeeze(-1)
        internal_velocity_scalar = v_next[:, 0]
        readout_velocity = (
            true_velocity
            if self.velocity_teacher_forcing and true_velocity is not None
            else internal_velocity_scalar
        )
        next_state = ModelState(
            node=h_next,
            node_prev=v_next,
            aux=dict(state.aux) if state.aux is not None else {},
        )
        assert next_state.aux is not None
        next_state.aux["ift_velocity_initialized"] = True
        next_state.aux["ift_prev_h"] = h.detach()
        next_state.aux["ift_position_scalar"] = h_next[:, 0]
        next_state.aux["ift_velocity_scalar"] = internal_velocity_scalar
        next_state.aux["ift_force_scalar"] = readout_force
        next_state.aux["ift_readout_position_scalar"] = readout_position
        next_state.aux["ift_readout_velocity_scalar"] = readout_velocity
        next_state.aux["ift_readout_force_scalar"] = readout_force
        next_state.aux["ift_prior_velocity_scalar"] = prior_velocity_scalar

        h_norm = h.norm(dim=-1).mean()
        v_norm = v.norm(dim=-1).mean()
        Lh_norm = Lh.norm(dim=-1).mean()
        force_norm = force.norm(dim=-1).mean()
        diffusion_term_norm = diffusion.norm(dim=-1).mean()
        velocity_term_norm = (alpha * v).norm(dim=-1).mean()
        force_term_norm = (dt * force).norm(dim=-1).mean()
        velocity_fraction = velocity_term_norm / (velocity_term_norm + force_term_norm + 1e-8)
        force_fraction = force_term_norm / (velocity_term_norm + force_term_norm + 1e-8)
        relative_diffusion = diffusion_term_norm / (force_norm + 1e-8)
        relative_update = (h_next - h).norm(dim=-1).mean() / (h.norm(dim=-1).mean() + 1e-8)
        L_density, L_diag_mean, L_nnz, L_offdiag_abs_mean, _ = _laplacian_stats(
            L, h.size(0), h.device, h.dtype
        )

        aux = {
            "kappa": kappa.detach(),
            "learned_kappa": kappa.detach(),
            "gamma": gamma.detach(),
            "dt": dt.detach(),
            "alpha": alpha.detach(),
            "h_norm": h_norm.detach(),
            "v_norm": v_norm.detach(),
            "Lh_norm": Lh_norm.detach(),
            "force_norm": force_norm.detach(),
            "inj_norm": force_norm.detach(),
            "diffusion_term_norm": diffusion_term_norm.detach(),
            "injection_term_norm": force_norm.detach(),
            "relative_diffusion": relative_diffusion.detach(),
            "relative_update": relative_update.detach(),
            "velocity_fraction": velocity_fraction.detach(),
            "force_fraction": force_fraction.detach(),
            "L_nnz": h.new_tensor(L_nnz).detach(),
            "L_density": L_density.detach(),
            "L_diag_mean": L_diag_mean.detach(),
            "L_offdiag_abs_mean": L_offdiag_abs_mean.detach(),
            "decoded_velocity": decoded_velocity,
            "internal_velocity_scalar": internal_velocity_scalar,
            "stored_velocity_scalar": readout_velocity.detach(),
            "true_velocity_scalar": (
                true_velocity.detach()
                if true_velocity is not None
                else torch.zeros((h.size(0),), device=h.device, dtype=h.dtype)
            ),
            "position_scalar": readout_position.detach(),
            "force_scalar": readout_force.detach(),
        }
        return next_state, aux
