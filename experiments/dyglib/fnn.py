"""FNN memory backbone for the pinned upstream DyGLib training loop."""
import torch
from torch import nn

from interactiondynamics.models.fnn import FieldNeuralNetwork


class FieldMemory(nn.Module):
    def __init__(self, num_nodes, state_dim=1):
        super().__init__()
        self.register_buffer("h", torch.zeros(num_nodes, state_dim))
        self.register_buffer("v", torch.zeros(num_nodes, state_dim))
        self.node_raw_messages = None  # Upstream checkpoints pending observations separately.

    def __init_memory_bank__(self):
        self.h = torch.zeros_like(self.h)
        self.v = torch.zeros_like(self.v)
        self.node_raw_messages = None
        if hasattr(self, "modal"):
            self.modal = torch.zeros_like(self.modal)

    def detach_memory_bank(self):
        self.h = self.h.detach()
        self.v = self.v.detach()
        if hasattr(self, "modal"):
            self.modal = self.modal.detach()

    def backup_memory_bank(self):
        pending = self.node_raw_messages
        result = (self.h.detach().clone(), self.v.detach().clone(),
                None if pending is None else tuple(x.detach().clone() for x in pending))
        return result + (self.modal.detach().clone(),) if hasattr(self, "modal") else result

    def reload_memory_bank(self, backup):
        h, v, pending = backup[:3]
        self.h = h.detach().clone().to(self.h.device)
        self.v = v.detach().clone().to(self.v.device)
        self.node_raw_messages = None if pending is None else tuple(x.clone().to(self.h.device) for x in pending)
        if hasattr(self, "modal"):
            self.modal = backup[3].detach().clone().to(self.h.device)


class FNN(nn.Module):
    def __init__(self, node_raw_features, edge_raw_features, neighbor_sampler, fnn_state_dim=1,
                 fnn_spectral_rank=0, fnn_propagate=0, **kwargs):
        super().__init__()
        if not isinstance(fnn_state_dim, int) or fnn_state_dim < 1:
            raise ValueError("fnn_state_dim must be a positive integer")
        self.state_dim = fnn_state_dim
        if not isinstance(fnn_spectral_rank, int) or fnn_spectral_rank < 0:
            raise ValueError("fnn_spectral_rank must be a nonnegative integer")
        if not isinstance(fnn_propagate, int) or isinstance(fnn_propagate, bool) or fnn_propagate < 0:
            raise ValueError("fnn_propagate must be a nonnegative integer")
        self.propagation_hops = fnn_propagate
        if fnn_propagate and fnn_spectral_rank:
            raise ValueError("Choose sparse input propagation or spectral propagation, not both")
        n, width = node_raw_features.shape
        self.field = FieldNeuralNetwork(
            num_nodes=n, force_dim=fnn_state_dim, state_dim=fnn_state_dim, gamma_init=0.15,
            omega_init=0.8, dt=0.1, topology_mode="observed_sparse",
            learn_gamma=True, learn_omega=True, learn_input_force_scale=True,
        )
        if fnn_state_dim > 1:
            # Different time scales break channel symmetry from initialization.
            scales = torch.logspace(-0.3, 0.3, fnn_state_dim)
            self.field.gamma_raw = nn.Parameter(torch.log(torch.expm1(0.15 * scales)))
            self.field.omega_raw = nn.Parameter(torch.log(torch.expm1(0.8 * scales)))
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
        self.memory_bank = FieldMemory(n, fnn_state_dim)
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

    def advance(self, src, dst, times):
        """Exact batched composition of the field's linear timestamp updates.

        x_T = M**T x_0 + sum_k M**(T-1-k) B incoming_k.
        Binary matrix powers avoid a Python loop over timestamps or nodes.
        Parameters stay constant within the batch, as in the upstream optimizer.
        """
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
        matrix = torch.stack([torch.stack([1 - dt.square() * omega.square(), dt * damp], dim=-1),
                              torch.stack([-dt * omega.square(), damp], dim=-1)], dim=-2)
        powers = torch.eye(2, device=matrix.device, dtype=matrix.dtype).expand(steps + 1, self.state_dim, 2, 2)
        exponents = torch.arange(steps + 1, device=matrix.device)
        base = matrix
        for bit in range(steps.bit_length()):
            powers = torch.where(((exponents >> bit) & 1).bool()[:, None, None, None], powers @ base, powers)
            base = base @ base
        bank = self.memory_bank
        initial = torch.stack([bank.h, bank.v], dim=-1)
        evolved = torch.einsum("cij,ncj->nci", powers[steps], initial)
        drive = torch.stack([dt.square(), dt])
        gates = torch.sigmoid(self.field._topology_logits_for(src, dst))
        amplitude = params["input_force_scale"] * gates
        if hasattr(self, "spread_raw"):
            dst, group, amplitude = self._spread_inputs(dst, group, amplitude)
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
            self.advance(src[ready], dst[ready], past_times[ready])
            bank.node_raw_messages = tuple(x[~ready] for x in pending) if (~ready).any() else None
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


def MemoryModel(*args, model_name, **kwargs):
    """Dispatch only FNN here; all native memory backbones remain upstream."""
    if model_name == "FNN":
        return FNN(*args, **kwargs)
    kwargs.pop("fnn_state_dim", None)
    kwargs.pop("fnn_spectral_rank", None)
    kwargs.pop("fnn_propagate", None)
    from models.MemoryModel import MemoryModel as NativeMemoryModel
    return NativeMemoryModel(*args, model_name=model_name, **kwargs)
