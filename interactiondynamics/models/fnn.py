"""Event-only force-field neural network.

The model never receives an adjacency matrix, node field, velocity, drive, or
generator parameter.  Its only dynamic input is a bin of observed directed
force events ``(source, target, force_vector)``.  A persistent learned operator
gates those forces before they update a latent second-order node field.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from interactiondynamics.core.events import EventBatch
from interactiondynamics.core.interfaces import InteractionModel, ModelState


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-5)
    return math.log(math.expm1(value))


class FieldNeuralNetwork(InteractionModel):
    """A learned first- or second-order field driven by observed forces."""

    def __init__(
        self,
        *,
        num_nodes: int,
        force_dim: int,
        state_dim: int,
        gamma_init: float,
        omega_init: float,
        dt: float,
        topology_init: float = 0.0,
        order: int = 2,
        force_decoder: str = "mlp",
        learn_physical_params: bool = False,
        force_scale_init: float = 1.0,
    ) -> None:
        super().__init__()
        if force_dim != state_dim:
            raise ValueError(
                "FieldNeuralNetwork uses physical force vectors directly, so "
                f"force_dim ({force_dim}) must equal state_dim ({state_dim})."
            )
        self.num_nodes = int(num_nodes)
        self.force_dim = int(force_dim)
        self.state_dim = int(state_dim)
        self.dt = float(dt)
        if order not in {1, 2}:
            raise ValueError(f"FieldNeuralNetwork order must be 1 or 2, got {order}.")
        self.order = int(order)
        self.learn_physical_params = bool(learn_physical_params)
        if force_decoder not in {"mlp", "linear", "field_difference"}:
            raise ValueError(
                "force_decoder must be one of {'mlp', 'linear', 'field_difference'}, "
                f"got {force_decoder!r}."
            )
        self.force_decoder_mode = force_decoder

        # One persistent symmetric gate per pair.  It is never formed from
        # current event adjacency, and its scale is fixed to [0, 1] so it
        # cannot trade an arbitrary coupling constant against force magnitude.
        logits = torch.full((num_nodes, num_nodes), float(topology_init))
        logits.fill_diagonal_(-12.0)
        self.topology_logits = nn.Parameter(logits)

        gamma_raw = torch.tensor(_inverse_softplus(gamma_init))
        omega_raw = torch.tensor(_inverse_softplus(omega_init))
        if self.learn_physical_params:
            self.gamma_raw = nn.Parameter(gamma_raw)
            self.omega_raw = nn.Parameter(omega_raw)
        else:
            # Buffers follow device placement and checkpoints but receive no
            # optimizer updates.  This is the default predictive setting.
            self.register_buffer("gamma_raw", gamma_raw)
            self.register_buffer("omega_raw", omega_raw)
        if force_decoder == "mlp":
            self.force_decoder: nn.Module | None = nn.Sequential(
                nn.Linear(4 * state_dim, 2 * state_dim),
                nn.Tanh(),
                nn.Linear(2 * state_dim, force_dim),
            )
        elif force_decoder == "linear":
            # No bias: a quiescent latent field should predict zero force.
            self.force_decoder = nn.Linear(4 * state_dim, force_dim, bias=False)
        else:
            self.force_decoder = None
            # Shared positive readout scale.  This is deliberately the only
            # flexibility beyond the latent field difference itself.
            force_scale_raw = torch.tensor(_inverse_softplus(force_scale_init))
            if self.learn_physical_params:
                self.force_scale_raw = nn.Parameter(force_scale_raw)
            else:
                self.register_buffer("force_scale_raw", force_scale_raw)
        # Filled from training targets by the runner.  Buffers make the
        # calibration device-safe and ensure checkpoints retain it.
        self.register_buffer("event_feature_target_std", torch.ones(force_dim))
        self.register_buffer("event_feature_active_threshold", torch.tensor(0.0))
        self.register_buffer("event_feature_magnitude_q90", torch.tensor(1.0))
        self.event_feature_magnitude_weight = 2.0

    @torch.no_grad()
    def configure_event_feature_objective(
        self,
        *,
        target_std: torch.Tensor,
        active_threshold: float,
        magnitude_q90: float,
        magnitude_weight: float,
    ) -> None:
        """Install training-split-only calibration for the force objective."""
        self.event_feature_target_std.copy_(target_std.to(self.event_feature_target_std).clamp_min(1e-8))
        self.event_feature_active_threshold.fill_(float(active_threshold))
        self.event_feature_magnitude_q90.fill_(max(float(magnitude_q90), 1e-8))
        self.event_feature_magnitude_weight = float(magnitude_weight)

    def topology_gate(self) -> torch.Tensor:
        logits = 0.5 * (self.topology_logits + self.topology_logits.T)
        gate = torch.sigmoid(logits)
        return gate * (1.0 - torch.eye(self.num_nodes, device=gate.device, dtype=gate.dtype))

    def physical_parameters(self) -> dict[str, torch.Tensor]:
        params = {"gamma": F.softplus(self.gamma_raw), "omega": F.softplus(self.omega_raw)}
        if self.force_decoder_mode == "field_difference":
            params["force_scale"] = F.softplus(self.force_scale_raw)
        return params

    def init_state(self, batch_size: int, num_nodes: int, device: torch.device) -> ModelState:
        if int(num_nodes) != self.num_nodes:
            raise ValueError(f"Model was built for {self.num_nodes} nodes, got {num_nodes}.")
        batch_size = int(batch_size)
        h = torch.zeros((batch_size * self.num_nodes, self.state_dim), device=device)
        v = torch.zeros_like(h)
        return ModelState(node=h, node_prev=v, aux={"batch_size": batch_size})

    def step(
        self,
        state: ModelState | None,
        events: EventBatch,
        drive: torch.Tensor | None = None,
    ) -> Tuple[ModelState, Dict[str, Any]]:
        if state is None or state.node is None or state.node_prev is None:
            raise ValueError("FieldNeuralNetwork requires an initialized state.")
        if events.features is None or events.features.shape[-1] != self.force_dim:
            raise ValueError(
                "FieldNeuralNetwork expects each event feature to be the observed "
                f"physical force vector with dimension {self.force_dim}."
            )
        h, v = state.node, state.node_prev
        gate = self.topology_gate()
        force = events.features.to(device=h.device, dtype=h.dtype)
        is_drop = (
            events.is_external.to(device=h.device, dtype=torch.bool)
            if events.is_external is not None
            else events.src == events.dst
        )
        # Packed episode batches use disjoint node-index blocks. The learned
        # physical operator is shared, so map those IDs back to local nodes.
        src_local = events.src.remainder(self.num_nodes)
        dst_local = events.dst.remainder(self.num_nodes)
        weighted_force = gate[src_local, dst_local].unsqueeze(-1) * force
        # An external event is an observed raindrop impulse. It bypasses the
        # learned inter-node operator; no hidden drive signal is available.
        weighted_force = torch.where(is_drop.unsqueeze(-1), force, weighted_force)
        incoming = torch.zeros_like(h)
        incoming.index_add_(0, events.dst, weighted_force)
        params = self.physical_parameters()
        if self.order == 1:
            h_next = (1.0 - params["gamma"] * self.dt) * h + self.dt * incoming
            v_next = torch.zeros_like(h_next)
        else:
            v_next = (1.0 - params["gamma"] * self.dt) * v + self.dt * (
                incoming - params["omega"].square() * h
            )
            h_next = h + self.dt * v_next
        next_state = ModelState(
            node=h_next,
            node_prev=v_next,
            aux={"batch_size": int((state.aux or {}).get("batch_size", 1))},
        )
        aux: Dict[str, Any] = {
            "gamma": params["gamma"],
            "omega": params["omega"],
            "dt": self.dt,
            "topology_gate_mean": gate.mean(),
        }
        return next_state, aux

    def _pair_features(self, state: ModelState, events: EventBatch) -> torch.Tensor:
        assert state.node is not None and state.node_prev is not None
        return torch.cat(
            [
                state.node[events.src],
                state.node[events.dst],
                state.node_prev[events.src],
                state.node_prev[events.dst],
            ],
            dim=-1,
        )

    def score(self, state: ModelState | None, candidate_events: EventBatch) -> torch.Tensor:
        if state is None:
            raise ValueError("FieldNeuralNetwork requires a state for next-event scoring.")
        # Event force values are intentionally ignored for destination ranking:
        # they are targets to predict, never clues that identify a candidate.
        return self.topology_logits[
            candidate_events.src.remainder(self.num_nodes),
            candidate_events.dst.remainder(self.num_nodes),
        ]

    def predict_event_features(self, state: ModelState | None, events: EventBatch) -> torch.Tensor:
        if state is None:
            raise ValueError("FieldNeuralNetwork requires a state for force prediction.")
        if self.force_decoder_mode == "field_difference":
            assert state.node is not None
            scale = F.softplus(self.force_scale_raw)
            return scale * (state.node[events.src] - state.node[events.dst])
        assert self.force_decoder is not None
        return self.force_decoder(self._pair_features(state, events))

    @torch.no_grad()
    def recovery_metrics(self, truth_adjacency: torch.Tensor, truth_params: dict[str, float]) -> dict[str, float]:
        """Evaluation-only comparisons against synthetic hidden truth."""
        gate = self.topology_gate().detach().flatten()
        truth = truth_adjacency.to(device=gate.device, dtype=gate.dtype).flatten()
        mask = ~torch.eye(self.num_nodes, device=gate.device, dtype=torch.bool).flatten()
        gate, truth = gate[mask], truth[mask]
        # Pairwise AUC without adding a sklearn dependency.
        pos, neg = gate[truth > 0.5], gate[truth <= 0.5]
        auc = float(((pos[:, None] > neg[None, :]).float() + 0.5 * (pos[:, None] == neg[None, :]).float()).mean().item())
        predicted = gate >= 0.5
        labels = truth > 0.5
        tp = int((predicted & labels).sum().item())
        fp = int((predicted & ~labels).sum().item())
        fn = int((~predicted & labels).sum().item())
        f1 = 0.0 if 2 * tp + fp + fn == 0 else (2.0 * tp) / (2 * tp + fp + fn)
        params = self.physical_parameters()
        learned_gamma = float(params["gamma"].item())
        metrics = {
            "topology_auc": auc,
            "topology_f1": f1,
            "learned_gamma": learned_gamma,
            "true_gamma": float(truth_params["gamma"]),
            "gamma_abs_error": abs(learned_gamma - float(truth_params["gamma"])),
        }
        if "omega" in truth_params:
            learned_omega = float(params["omega"].item())
            metrics["learned_omega"] = learned_omega
            metrics["true_omega"] = float(truth_params["omega"])
            metrics["omega_abs_error"] = abs(learned_omega - float(truth_params["omega"]))
        if "force_scale" in params:
            learned_force_scale = float(params["force_scale"].item())
            metrics["learned_force_scale"] = learned_force_scale
            if "force_scale" in truth_params:
                metrics["true_force_scale"] = float(truth_params["force_scale"])
                metrics["force_scale_abs_error"] = abs(
                    learned_force_scale - float(truth_params["force_scale"])
                )
        return metrics
