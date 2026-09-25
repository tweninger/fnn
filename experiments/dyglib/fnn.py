"""FNN memory backbone for the pinned upstream DyGLib training loop."""
import math
import numpy as np
import torch
from torch import nn

from interactiondynamics.models.fnn import FieldNeuralNetwork


class FieldMemory(nn.Module):
    def __init__(self, num_nodes, state_dim=1, track_time=False):
        super().__init__()
        self.register_buffer("h", torch.zeros(num_nodes, state_dim))
        self.register_buffer("v", torch.zeros(num_nodes, state_dim))
        if track_time:
            self.register_buffer("last_update_time", torch.tensor(float("nan"), dtype=torch.float64))
        self.node_raw_messages = None  # Upstream checkpoints pending observations separately.

    def __init_memory_bank__(self):
        self.h = torch.zeros_like(self.h)
        self.v = torch.zeros_like(self.v)
        self.node_raw_messages = None
        if hasattr(self, "modal"):
            self.modal = torch.zeros_like(self.modal)
        if hasattr(self, "last_update_time"):
            self.last_update_time.fill_(float("nan"))

    def detach_memory_bank(self):
        self.h = self.h.detach()
        self.v = self.v.detach()
        if hasattr(self, "modal"):
            self.modal = self.modal.detach()

    def backup_memory_bank(self):
        pending = self.node_raw_messages
        result = (self.h.detach().clone(), self.v.detach().clone(),
                None if pending is None else tuple(x.detach().clone() for x in pending))
        if hasattr(self, "modal"):
            result += (self.modal.detach().clone(),)
        if hasattr(self, "last_update_time"):
            result += (self.last_update_time.detach().clone(),)
        return result

    def reload_memory_bank(self, backup):
        h, v, pending = backup[:3]
        self.h = h.detach().clone().to(self.h.device)
        self.v = v.detach().clone().to(self.v.device)
        self.node_raw_messages = None if pending is None else tuple(x.clone().to(self.h.device) for x in pending)
        offset = 3
        if hasattr(self, "modal"):
            self.modal = backup[offset].detach().clone().to(self.h.device)
            offset += 1
        if hasattr(self, "last_update_time"):
            self.last_update_time = backup[offset].detach().clone().to(self.h.device)


class FNN(nn.Module):
    def __init__(self, node_raw_features, edge_raw_features, neighbor_sampler, fnn_state_dim=1,
                 fnn_order=2, fnn_spectral_rank=0, fnn_propagate=0, fnn_clock="event",
                 fnn_time_cap=10.0, fnn_ablation="none", fnn_fixed_gate_value=0.5,
                 fnn_gamma_init=0.15, fnn_omega_init=0.8,
                 fnn_input_scale_init=1.0, **kwargs):
        super().__init__()
        if not isinstance(fnn_state_dim, int) or fnn_state_dim < 1:
            raise ValueError("fnn_state_dim must be a positive integer")
        self.state_dim = fnn_state_dim
        if fnn_order not in {1, 2}:
            raise ValueError("fnn_order must be 1 or 2")
        self.order = int(fnn_order)
        if not isinstance(fnn_spectral_rank, int) or fnn_spectral_rank < 0:
            raise ValueError("fnn_spectral_rank must be a nonnegative integer")
        if not isinstance(fnn_propagate, int) or isinstance(fnn_propagate, bool) or fnn_propagate < 0:
            raise ValueError("fnn_propagate must be a nonnegative integer")
        self.propagation_hops = fnn_propagate
        if fnn_clock not in {
            "event", "event_exact", "event_exact_unit", "normalized", "normalized_exact",
            "normalized_substep", "normalized_substep_unit",
        }:
            raise ValueError(
                "fnn_clock must be 'event', 'event_exact', 'event_exact_unit', "
                "'normalized', 'normalized_exact', 'normalized_substep', or "
                "'normalized_substep_unit'"
            )
        if fnn_time_cap <= 0:
            raise ValueError("fnn_time_cap must be positive")
        self.clock = fnn_clock
        self.time_cap = float(fnn_time_cap)
        valid_ablations = {
            "none", "fixed_topology", "fixed_gates", "fixed_physical",
            "fixed_gates_physical", "fixed_dynamics", "fixed_input_scale",
            "fixed_gamma", "fixed_omega",
        }
        if fnn_ablation not in valid_ablations:
            raise ValueError(f"fnn_ablation must be one of {sorted(valid_ablations)}")
        if not 0.0 < fnn_fixed_gate_value < 1.0:
            raise ValueError("fnn_fixed_gate_value must be strictly between 0 and 1")
        if fnn_gamma_init <= 0 or fnn_omega_init <= 0 or fnn_input_scale_init <= 0:
            raise ValueError("FNN physical initializations must be positive")
        self.ablation = fnn_ablation
        self.fixed_gate_value = float(fnn_fixed_gate_value)
        self.gamma_init = float(fnn_gamma_init)
        self.omega_init = float(fnn_omega_init)
        self.input_scale_init = float(fnn_input_scale_init)
        if fnn_propagate and fnn_spectral_rank:
            raise ValueError("Choose sparse input propagation or spectral propagation, not both")
        if self.order == 1 and fnn_spectral_rank:
            raise ValueError("First-order DyGLib FNN does not support spectral propagation")
        if self.order == 1 and fnn_clock != "event":
            raise ValueError("First-order DyGLib FNN currently requires fnn_clock=event")
        n, width = node_raw_features.shape
        self.field = FieldNeuralNetwork(
            num_nodes=n, force_dim=fnn_state_dim, state_dim=fnn_state_dim,
            gamma_init=self.gamma_init, omega_init=self.omega_init, dt=0.1,
            topology_mode="observed_sparse", learn_gamma=True, learn_omega=True,
            learn_input_force_scale=True, input_force_scale_init=self.input_scale_init,
            order=self.order,
        )
        if fnn_state_dim > 1:
            # Different time scales break channel symmetry from initialization.
            scales = torch.logspace(-0.3, 0.3, fnn_state_dim)
            self.field.gamma_raw = nn.Parameter(torch.log(torch.expm1(self.gamma_init * scales)))
            self.field.omega_raw = nn.Parameter(torch.log(torch.expm1(self.omega_init * scales)))
            self.drive_vector = nn.Parameter(torch.ones(fnn_state_dim) / fnn_state_dim**0.5)
        # Candidate support comes only from the training neighbor sampler.
        # DyGLib stores neighbors in both directions; neither test pairs nor
        # test features are consulted when creating learnable gates.
        src, dst = [], []
        for node, neighbors in enumerate(neighbor_sampler.nodes_neighbor_ids):
            src.extend([node] * len(neighbors))
            dst.extend(neighbors.tolist())
        self.field.set_sparse_topology_candidates(torch.tensor(src, dtype=torch.long),
                                                  torch.tensor(dst, dtype=torch.long))
        if fnn_ablation == "fixed_topology":
            # DyGLib candidates are exactly the directed pairs observed in the
            # training graph. Treat those as binary edges and all unknown
            # validation/test pairs as nonedges, matching the synthetic oracle.
            with torch.no_grad():
                self.field.sparse_topology_logits.fill_(30.0)
            self.field._topology_init = -30.0
            self.field.sparse_topology_logits.requires_grad_(False)
        elif fnn_ablation in {"fixed_gates", "fixed_gates_physical"}:
            # Use one constant gate for train-observed and unseen pairs.
            # A gate of 0.5 corresponds to logit zero.
            logit = math.log(self.fixed_gate_value / (1.0 - self.fixed_gate_value))
            with torch.no_grad():
                self.field.sparse_topology_logits.fill_(logit)
            self.field._topology_init = logit
            self.field.sparse_topology_logits.requires_grad_(False)
        if fnn_ablation in {"fixed_physical", "fixed_gates_physical"}:
            for name in ("gamma_raw", "omega_raw", "input_force_scale_raw", "force_scale_raw", "dt_raw"):
                parameter = getattr(self.field, name, None)
                if isinstance(parameter, nn.Parameter):
                    parameter.requires_grad_(False)
        elif fnn_ablation in {"fixed_gamma", "fixed_omega"}:
            name = "gamma_raw" if fnn_ablation == "fixed_gamma" else "omega_raw"
            parameter = getattr(self.field, name, None)
            if isinstance(parameter, nn.Parameter):
                parameter.requires_grad_(False)
        elif fnn_ablation == "fixed_dynamics":
            # Isolate decay/oscillation dynamics while retaining learnable forcing amplitude.
            for name in ("gamma_raw", "omega_raw"):
                parameter = getattr(self.field, name, None)
                if isinstance(parameter, nn.Parameter):
                    parameter.requires_grad_(False)
        elif fnn_ablation == "fixed_input_scale":
            # Isolate forcing-amplitude calibration while retaining learnable dynamics.
            parameter = getattr(self.field, "input_force_scale_raw", None)
            if isinstance(parameter, nn.Parameter):
                parameter.requires_grad_(False)
        self.memory_bank = FieldMemory(n, fnn_state_dim, track_time=fnn_clock != "event")
        if fnn_clock != "event":
            observed_times = [times for times in neighbor_sampler.nodes_neighbor_times if len(times)]
            unique_times = np.unique(np.concatenate(observed_times)) if observed_times else np.array([0.0])
            positive_gaps = np.diff(unique_times)
            positive_gaps = positive_gaps[positive_gaps > 0]
            time_scale = float(np.median(positive_gaps)) if len(positive_gaps) else 1.0
            self.register_buffer("time_scale", torch.tensor(time_scale, dtype=torch.float64))
        if fnn_propagate:
            keys = self.field.sparse_candidate_keys
            indices = torch.where(keys // n != keys % n)[0]
            self.register_buffer("spread_indices", indices, persistent=False)
            self.register_buffer("spread_sources", keys[indices] // n, persistent=False)
            self.register_buffer("spread_targets", keys[indices] % n, persistent=False)
            self.spread_raw = nn.Parameter(torch.logit(torch.tensor(0.1)))
        if fnn_spectral_rank:
            from experiments.dyglib.spectral import train_basis
            basis, eigenvalues = train_basis(n, self.field.sparse_candidate_keys, fnn_spectral_rank)
            self.register_buffer("basis", basis)
            self.register_buffer("eigenvalues", eigenvalues)
            self.kappa_raw = nn.Parameter(torch.log(torch.expm1(torch.tensor(0.1))))
            self.memory_bank.register_buffer("modal", torch.zeros(basis.shape[1], fnn_state_dim, 2))
        self.projection = nn.Linear(2 * fnn_state_dim, width)

    def extra_repr(self):
        if self.clock == "event":
            return (f"order={self.order}, clock=event, ablation={self.ablation}, fixed_gate_value={self.fixed_gate_value:g}, "
                    f"gamma_init={self.gamma_init:g}, omega_init={self.omega_init:g}, "
                    f"input_scale_init={self.input_scale_init:g}")
        return (f"clock={self.clock}, time_scale={self.time_scale.item():g}, "
                f"time_cap={self.time_cap:g}, ablation={self.ablation}, "
                f"fixed_gate_value={self.fixed_gate_value:g}, gamma_init={self.gamma_init:g}, "
                f"omega_init={self.omega_init:g}, input_scale_init={self.input_scale_init:g}")

    def transformed_decay_parameters(self):
        """Raw leaves whose positive transforms, rather than raws, are regularized."""
        parameters = []
        names = ["gamma_raw", "input_force_scale_raw", "force_scale_raw", "dt_raw"]
        if self.order == 2:
            names.insert(1, "omega_raw")
        for name in names:
            value = getattr(self.field, name, None)
            if isinstance(value, nn.Parameter) and value.requires_grad:
                parameters.append(value)
        if hasattr(self, "kappa_raw") and self.kappa_raw.requires_grad:
            parameters.append(self.kappa_raw)
        return parameters

    def transformed_weight_decay(self, coefficient):
        """L2 penalty in positive physical space, whose minimum is actually zero."""
        zero = self.projection.weight.new_zeros(())
        if not coefficient:
            return zero
        params = self.field.physical_parameters()
        terms = []
        physical_pairs = [
            ("gamma_raw", "gamma"),
            ("input_force_scale_raw", "input_force_scale"),
            ("force_scale_raw", "force_scale"),
            ("dt_raw", "dt"),
        ]
        if self.order == 2:
            physical_pairs.insert(1, ("omega_raw", "omega"))
        for raw_name, physical_name in physical_pairs:
            raw = getattr(self.field, raw_name, None)
            if isinstance(raw, nn.Parameter) and raw.requires_grad and physical_name in params:
                terms.append(params[physical_name].square().sum())
        if hasattr(self, "kappa_raw") and self.kappa_raw.requires_grad:
            terms.append(torch.nn.functional.softplus(self.kappa_raw).square().sum())
        return zero if not terms else 0.5 * float(coefficient) * torch.stack(terms).sum()

    def _advance_modes(self, src, dst, group, steps):
        """Compose the SAME semi-implicit event steps in each spatial mode."""
        p = self.field.physical_parameters()
        dt = p["dt"]
        stiffness = p["omega"].reshape(1, -1).square() + \
            torch.nn.functional.softplus(self.kappa_raw) * self.eigenvalues[:, None]
        damp = (1 - p["gamma"].reshape(1, -1)*dt).expand_as(stiffness)
        matrix = torch.stack((torch.stack((1-dt.square()*stiffness, dt*damp), -1),
                              torch.stack((-dt*stiffness, damp), -1)), -2)
        powers = torch.eye(2, device=matrix.device, dtype=matrix.dtype).expand(steps+1, *matrix.shape)
        exponents = torch.arange(steps+1, device=matrix.device)
        base = matrix
        for bit in range(steps.bit_length()):
            powers = torch.where(((exponents >> bit)&1).bool()[:, None, None, None, None], powers @ base, powers)
            base = base @ base
        modal = torch.einsum("rcij,rcj->rci", powers[steps], self.memory_bank.modal)
        impulse = torch.stack((dt.square(), dt))
        amplitude = p["input_force_scale"] * self.field._topology_logits_for(src, dst).sigmoid()
        drive = self.drive_vector if self.state_dim > 1 else matrix.new_ones(1)
        # Chunk event responses; there is no sequential timestamp loop.
        for start in range(0, len(src), 64):
            stop = start+64
            response = powers[steps-1-group[start:stop]] @ impulse
            weights = self.basis[dst[start:stop]] * amplitude[start:stop, None]
            modal = modal + torch.einsum("brci,br,c->rci", response, weights, drive)
        self.memory_bank.modal = modal

    def _node_states(self, src, dst):
        bank = self.memory_bank
        h_src, v_src, h_dst, v_dst = bank.h[src], bank.v[src], bank.h[dst], bank.v[dst]
        if hasattr(self, "basis"):
            # Local state retains omitted modes and isolated/new-node inputs.
            # Replace retained modes rather than double-counting their drives.
            reference = torch.stack((self.basis.T @ bank.h, self.basis.T @ bank.v), -1)
            correction = bank.modal-reference
            source = torch.einsum("br,rci->bci", self.basis[src], correction)
            destination = torch.einsum("br,rci->bci", self.basis[dst], correction)
            h_src, v_src = h_src+source[..., 0], v_src+source[..., 1]
            h_dst, v_dst = h_dst+destination[..., 0], v_dst+destination[..., 1]
        return torch.cat((h_src, v_src), -1), torch.cat((h_dst, v_dst), -1)

    def _spread_inputs(self, dst, group, amplitude):
        """Conservative K-hop input diffusion; aggregate by timestamp and node."""
        n = self.memory_bank.h.shape[0]

        def coalesce(nodes, groups, values):
            keys, inverse = torch.unique(groups*n + nodes, return_inverse=True)
            return keys % n, keys // n, values.new_zeros(len(keys)).index_add(0, inverse, values)

        dst, group, amplitude = coalesce(dst, group, amplitude)
        kept_nodes, kept_groups, kept_values = [], [], []
        for _ in range(self.propagation_hops):
            nodes, groups, retained, targets, remote = self._spread_one_hop(dst, group, amplitude)
            kept_nodes.append(nodes)
            kept_groups.append(group)
            kept_values.append(retained)
            dst, group, amplitude = coalesce(targets, groups, remote)
        kept_nodes.append(dst)
        kept_groups.append(group)
        kept_values.append(amplitude)
        return coalesce(torch.cat(kept_nodes), torch.cat(kept_groups), torch.cat(kept_values))

    def _spread_one_hop(self, dst, group, amplitude):
        starts = torch.searchsorted(self.spread_sources, dst)
        ends = torch.searchsorted(self.spread_sources, dst, right=True)
        counts = ends - starts
        event = torch.repeat_interleave(torch.arange(len(dst), device=dst.device), counts)
        offsets = torch.repeat_interleave(counts.cumsum(0) - counts, counts)
        edge = torch.repeat_interleave(starts, counts) + torch.arange(len(event), device=dst.device) - offsets
        weights = self.field.sparse_topology_logits[self.spread_indices[edge]].sigmoid()
        totals = amplitude.new_zeros(len(dst)).index_add(0, event, weights)
        fraction = self.spread_raw.sigmoid() * (totals > 0)
        remote = amplitude[event] * fraction[event] * weights / totals[event].clamp_min(torch.finfo(weights.dtype).tiny)
        return dst, group[event], amplitude * (1 - fraction), self.spread_targets[edge], remote

    def _continuous_transition(self, elapsed, stiffness=None):
        """Exact damped-oscillator transition for a normalized elapsed-time gap."""
        params = self.field.physical_parameters()
        gamma = params["gamma"].reshape(-1)
        if stiffness is None:
            stiffness = params["omega"].reshape(-1).square()
        duration = params["dt"] * elapsed.to(dtype=gamma.dtype)
        zeros = torch.zeros_like(stiffness)
        ones = torch.ones_like(stiffness)
        if stiffness.ndim == 1:
            damping = gamma
        else:
            damping = gamma.unsqueeze(0).expand_as(stiffness)
        generator = torch.stack((torch.stack((zeros, ones), -1),
                                 torch.stack((-stiffness, -damping), -1)), -2)
        return torch.matrix_exp(generator * duration)

    def _substep_transition(self, elapsed, stiffness=None):
        """Compose original semi-implicit steps over a normalized unit-clock gap."""
        params = self.field.physical_parameters()
        gamma = params["gamma"].reshape(-1)
        if stiffness is None:
            stiffness = params["omega"].reshape(-1).square()
        damping = gamma if stiffness.ndim == 1 else gamma.unsqueeze(0).expand_as(stiffness)

        def step_matrix(step):
            step = step.to(device=stiffness.device, dtype=stiffness.dtype)
            damp = 1 - damping * step
            return torch.stack((
                torch.stack((1 - step.square() * stiffness, step * damp), -1),
                torch.stack((-step * stiffness, damp), -1),
            ), -2)

        base_dt = params["dt"].reshape(()).to(dtype=elapsed.dtype)
        # elapsed derives only from fixed timestamps, so selecting an integer
        # number of substeps does not cut a learnable gradient path.
        full_steps = int(torch.floor(elapsed / base_dt + 1e-5).item())
        remainder = (elapsed - full_steps * base_dt).clamp_min(0)
        identity = torch.eye(2, device=stiffness.device, dtype=stiffness.dtype)
        transition = identity.expand(*stiffness.shape, 2, 2)
        if full_steps:
            transition = torch.linalg.matrix_power(step_matrix(params["dt"]), full_steps)
        if float(remainder) > 1e-7:
            transition = step_matrix(remainder) @ transition
        return transition

    def _advance_normalized(self, src, dst, times, target_time=None,
                            fixed_transitions=None):
        """Advance with median-gap-normalized elapsed time and bounded quiet gaps."""
        bank = self.memory_bank
        unique, group = torch.unique(times, sorted=True, return_inverse=True)
        params = self.field.physical_parameters()
        drive = torch.stack((params["dt"].square(), params["dt"]))
        if len(times):
            amplitude = params["input_force_scale"] * self.field._topology_logits_for(src, dst).sigmoid()
            if hasattr(self, "spread_raw"):
                dst, group, amplitude = self._spread_inputs(dst, group, amplitude)
        state = torch.stack((bank.h, bank.v), -1)
        modal = bank.modal if hasattr(self, "basis") else None
        if modal is not None:
            stiffness = params["omega"].reshape(1, -1).square() + \
                torch.nn.functional.softplus(self.kappa_raw) * self.eigenvalues[:, None]
        fixed_transition = fixed_modal_transition = None
        if self.clock in {"event_exact", "event_exact_unit"}:
            if fixed_transitions is not None:
                fixed_transition, fixed_modal_transition = fixed_transitions
            else:
                one_step = state.new_tensor(
                    1.0 if self.clock == "event_exact" else 1.0 / float(params["dt"])
                )
                fixed_transition = self._continuous_transition(one_step)
                if modal is not None:
                    fixed_modal_transition = self._continuous_transition(one_step, stiffness)

        last = bank.last_update_time

        def evolve(current_time, state, modal, last):
            if torch.isnan(last):
                return state, modal, current_time
            raw_gap = current_time - last
            if bool(raw_gap < 0):
                raise ValueError("FNN timestamps must be nondecreasing")
            elapsed = (raw_gap > 0).to(raw_gap.dtype) if self.clock in {
                "event_exact", "event_exact_unit"
            } else \
                (raw_gap / self.time_scale).clamp(max=self.time_cap)
            if bool(elapsed > 0):
                if self.clock in {"normalized_substep", "normalized_substep_unit"}:
                    # Match normalized_exact by scaling the normalized gap by dt.
                    substep_elapsed = params["dt"] * elapsed \
                        if self.clock == "normalized_substep" else elapsed
                    transition = self._substep_transition(substep_elapsed)
                else:
                    transition = fixed_transition if fixed_transition is not None else \
                        self._continuous_transition(elapsed)
                state = torch.einsum("cij,ncj->nci", transition, state)
                if modal is not None:
                    if self.clock in {"normalized_substep", "normalized_substep_unit"}:
                        modal_transition = self._substep_transition(substep_elapsed, stiffness)
                    else:
                        modal_transition = fixed_modal_transition if fixed_modal_transition is not None else \
                            self._continuous_transition(elapsed, stiffness)
                    modal = torch.einsum("rcij,rcj->rci", modal_transition, modal)
            return state, modal, current_time

        channel_drive = self.drive_vector if self.state_dim > 1 else state.new_ones(1)
        for index, timestamp in enumerate(unique):
            state, modal, last = evolve(timestamp, state, modal, last)
            mask = group == index
            event_drive = amplitude[mask, None, None] * channel_drive[None, :, None] * drive
            state = state.index_add(0, dst[mask], event_drive)
            if modal is not None:
                weights = self.basis[dst[mask]] * amplitude[mask, None]
                modal = modal + torch.einsum("br,bc,ci->rci", weights,
                                             channel_drive.expand(int(mask.sum()), -1), drive)
        if target_time is not None:
            state, modal, last = evolve(target_time, state, modal, last)
        bank.h, bank.v = state[..., 0], state[..., 1]
        if modal is not None:
            bank.modal = modal
        bank.last_update_time = last.detach().clone()

    def advance(self, src, dst, times, target_time=None, fixed_transitions=None):
        """Exact batched composition of the field's linear timestamp updates.

        x_T = M**T x_0 + sum_k M**(T-1-k) B incoming_k.
        Binary matrix powers avoid a Python loop over timestamps or nodes.
        Parameters stay constant within the batch, as in the upstream optimizer.
        """
        if self.clock != "event":
            if len(times) or target_time is not None:
                self._advance_normalized(
                    src, dst, times, target_time,
                    fixed_transitions=fixed_transitions,
                )
            return
        if not len(times):
            return
        unique, group = torch.unique(times, sorted=True, return_inverse=True)
        steps = len(unique)
        if hasattr(self, "basis"):
            self._advance_modes(src, dst, group, steps)
        params = self.field.physical_parameters()
        dt, gamma, omega = params["dt"], params["gamma"], params["omega"]
        gamma, omega = gamma.reshape(-1), omega.reshape(-1)
        damp = 1 - gamma * dt
        bank = self.memory_bank
        gates = torch.sigmoid(self.field._topology_logits_for(src, dst))
        amplitude = params["input_force_scale"] * gates
        if hasattr(self, "spread_raw"):
            dst, group, amplitude = self._spread_inputs(dst, group, amplitude)
        if self.order == 1:
            exponents = torch.arange(steps + 1, device=damp.device)
            powers = damp[None, :].pow(exponents[:, None])
            evolved = bank.h * powers[steps]
            response = powers[steps - 1 - group] * dt * amplitude[:, None]
            if self.state_dim > 1:
                response = response * self.drive_vector[None, :]
            bank.h = evolved.index_add(0, dst, response)
            bank.v = torch.zeros_like(bank.h)
            return
        matrix = torch.stack([torch.stack([1 - dt.square() * omega.square(), dt * damp], dim=-1),
                              torch.stack([-dt * omega.square(), damp], dim=-1)], dim=-2)
        powers = torch.eye(2, device=matrix.device, dtype=matrix.dtype).expand(steps + 1, self.state_dim, 2, 2)
        exponents = torch.arange(steps + 1, device=matrix.device)
        base = matrix
        for bit in range(steps.bit_length()):
            powers = torch.where(((exponents >> bit) & 1).bool()[:, None, None, None], powers @ base, powers)
            base = base @ base
        initial = torch.stack([bank.h, bank.v], dim=-1)
        evolved = torch.einsum("cij,ncj->nci", powers[steps], initial)
        drive = torch.stack([dt.square(), dt])
        response = powers[steps - 1 - group] @ drive
        response = response * amplitude[:, None, None]
        if self.state_dim > 1:
            response = response * self.drive_vector[None, :, None]
        evolved = evolved.index_add(0, dst, response)
        bank.h, bank.v = evolved[..., 0], evolved[..., 1]

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids, dst_node_ids,
                                                node_interact_times, edge_ids=None,
                                                edges_are_positive=False, **kwargs):
        device = self.memory_bank.h.device
        times = torch.as_tensor(node_interact_times, dtype=torch.float64, device=device)
        bank = self.memory_bank
        pending = bank.node_raw_messages
        if pending is not None and len(times):
            src, dst, past_times = pending
            # All queries in this batch see only observations strictly before
            # the earliest query. Never ingest current positives before scoring.
            ready = past_times < times.min()
            self.advance(src[ready], dst[ready], past_times[ready],
                         target_time=times.min() if self.clock != "event" else None)
            bank.node_raw_messages = tuple(x[~ready] for x in pending) if (~ready).any() else None
        elif self.clock != "event" and len(times):
            empty = torch.empty(0, dtype=torch.long, device=device)
            self.advance(empty, empty, times.new_empty(0), target_time=times.min())
        src = torch.as_tensor(src_node_ids, dtype=torch.long, device=device)
        dst = torch.as_tensor(dst_node_ids, dtype=torch.long, device=device)
        # Project only queried nodes, not the entire graph.
        src_state, dst_state = self._node_states(src, dst)
        src_embedding = self.projection(src_state)
        dst_embedding = self.projection(dst_state)
        if edges_are_positive:
            new = (src.detach(), dst.detach(), times.detach())
            old = bank.node_raw_messages
            bank.node_raw_messages = new if old is None else tuple(torch.cat([a, b]) for a, b in zip(old, new))
        return src_embedding, dst_embedding

    def compute_link_prediction_batch(self, src_node_ids, dst_node_ids, neg_src_node_ids,
                                      neg_dst_node_ids, node_interact_times):
        """Score one batch causally, advancing and ingesting one timestamp at a time."""
        if self.clock not in {"event_exact", "event_exact_unit", "normalized_exact",
                              "normalized_substep", "normalized_substep_unit"}:
            raise ValueError(
                "timestamp-grouped scoring requires an exact-event or grouped normalized clock"
            )
        device = self.memory_bank.h.device
        src = torch.as_tensor(src_node_ids, dtype=torch.long, device=device)
        dst = torch.as_tensor(dst_node_ids, dtype=torch.long, device=device)
        neg_src = torch.as_tensor(neg_src_node_ids, dtype=torch.long, device=device)
        neg_dst = torch.as_tensor(neg_dst_node_ids, dtype=torch.long, device=device)
        times = torch.as_tensor(node_interact_times, dtype=torch.float64, device=device)
        unique_times = torch.unique(times, sorted=True)
        fixed_transitions = None
        if self.clock in {"event_exact", "event_exact_unit"}:
            params = self.field.physical_parameters()
            one_step = self.memory_bank.h.new_tensor(
                1.0 if self.clock == "event_exact" else 1.0 / float(params["dt"])
            )
            fixed_transition = self._continuous_transition(one_step)
            fixed_modal_transition = None
            if hasattr(self, "basis"):
                stiffness = params["omega"].reshape(1, -1).square() + \
                    torch.nn.functional.softplus(self.kappa_raw) * self.eigenvalues[:, None]
                fixed_modal_transition = self._continuous_transition(one_step, stiffness)
            fixed_transitions = (fixed_transition, fixed_modal_transition)
        indices, positive_sources, positive_destinations = [], [], []
        negative_sources, negative_destinations = [], []

        for query_time in unique_times:
            mask = times == query_time
            group_indices = torch.where(mask)[0]
            pending = self.memory_bank.node_raw_messages
            if pending is None:
                empty = torch.empty(0, dtype=torch.long, device=device)
                self.advance(
                    empty, empty, times.new_empty(0), target_time=query_time,
                    fixed_transitions=fixed_transitions,
                )
            else:
                past_src, past_dst, past_times = pending
                ready = past_times < query_time
                self.advance(
                    past_src[ready], past_dst[ready], past_times[ready],
                    target_time=query_time, fixed_transitions=fixed_transitions,
                )
                self.memory_bank.node_raw_messages = (
                    tuple(value[~ready] for value in pending) if (~ready).any() else None)

            pos_src_state, pos_dst_state = self._node_states(src[mask], dst[mask])
            neg_src_state, neg_dst_state = self._node_states(neg_src[mask], neg_dst[mask])
            indices.append(group_indices)
            positive_sources.append(self.projection(pos_src_state))
            positive_destinations.append(self.projection(pos_dst_state))
            negative_sources.append(self.projection(neg_src_state))
            negative_destinations.append(self.projection(neg_dst_state))

            new = (src[mask].detach(), dst[mask].detach(), times[mask].detach())
            old = self.memory_bank.node_raw_messages
            self.memory_bank.node_raw_messages = (
                new if old is None else tuple(torch.cat((a, b)) for a, b in zip(old, new)))

        order = torch.cat(indices)
        restore_order = torch.argsort(order)
        return tuple(torch.cat(parts)[restore_order] for parts in
                     (positive_sources, positive_destinations,
                      negative_sources, negative_destinations))


def MemoryModel(*args, model_name, **kwargs):
    """Dispatch only FNN here; all native memory backbones remain upstream."""
    if model_name == "FNN":
        return FNN(*args, **kwargs)
    kwargs.pop("fnn_state_dim", None)
    kwargs.pop("fnn_order", None)
    kwargs.pop("fnn_spectral_rank", None)
    kwargs.pop("fnn_propagate", None)
    kwargs.pop("fnn_clock", None)
    kwargs.pop("fnn_time_cap", None)
    kwargs.pop("fnn_ablation", None)
    kwargs.pop("fnn_fixed_gate_value", None)
    kwargs.pop("fnn_gamma_init", None)
    kwargs.pop("fnn_omega_init", None)
    kwargs.pop("fnn_input_scale_init", None)
    from models.MemoryModel import MemoryModel as NativeMemoryModel
    return NativeMemoryModel(*args, model_name=model_name, **kwargs)
