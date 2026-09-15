"""FNN memory backbone for the pinned upstream DyGLib training loop."""
import math
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

    def detach_memory_bank(self):
        self.h = self.h.detach()
        self.v = self.v.detach()

    def backup_memory_bank(self):
        pending = self.node_raw_messages
        return (self.h.detach().clone(), self.v.detach().clone(),
                None if pending is None else tuple(x.detach().clone() for x in pending))

    def reload_memory_bank(self, backup):
        h, v, pending = backup
        self.h = h.detach().clone().to(self.h.device)
        self.v = v.detach().clone().to(self.v.device)
        self.node_raw_messages = None if pending is None else tuple(x.clone().to(self.h.device) for x in pending)


class FNN(nn.Module):
    def __init__(self, node_raw_features, edge_raw_features, neighbor_sampler, fnn_state_dim=1,
                 fnn_coupling=0.0, **kwargs):
        super().__init__()
        if not isinstance(fnn_state_dim, int) or fnn_state_dim < 1:
            raise ValueError("fnn_state_dim must be a positive integer")
        self.state_dim = fnn_state_dim
        if not math.isfinite(fnn_coupling) or fnn_coupling < 0:
            raise ValueError("fnn_coupling must be finite and nonnegative")
        # Zero disables coupling entirely, preserving legacy checkpoint keys.
        if fnn_coupling > 0:
            value = torch.tensor(float(fnn_coupling))
            self.kappa_raw = nn.Parameter(value + torch.log(-torch.expm1(-value)))
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
        self.projection = nn.Linear(2 * fnn_state_dim, width)

    def advance(self, src, dst, times):
        """Exact batched composition of the field's linear timestamp updates.

        x_T = M**T x_0 + sum_k M**(T-1-k) B incoming_k.
        Binary matrix powers avoid a Python loop over timestamps or nodes.
        Parameters stay constant within the batch, as in the upstream optimizer.
        """
        if not len(times):
            return
        if hasattr(self, "kappa_raw"):
            self._advance_coupled(src, dst, times, torch.nn.functional.softplus(self.kappa_raw))
            return
        unique, group = torch.unique(times, sorted=True, return_inverse=True)
        steps = len(unique)
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
        response = powers[steps - 1 - group] @ drive
        gates = torch.sigmoid(self.field._topology_logits_for(src, dst))
        response = response * (params["input_force_scale"] * gates)[:, None, None]
        if self.state_dim > 1:
            response = response * self.drive_vector[None, :, None]
        evolved = evolved.index_add(0, dst, response)
        bank.h, bank.v = evolved[..., 0], evolved[..., 1]

    def _advance_coupled(self, src, dst, times, kappa):
        """Sequential event-clock steps with sparse, incoming-normalized exchange.

        All candidate edges couple fields each step, including edges without
        an event at that timestamp. Parameters and the sparse operator are
        shared across steps, while gradients flow through the full recurrence.
        """
        if not len(times):
            return
        bank = self.memory_bank
        params = self.field.physical_parameters()
        dt, gamma, omega = params["dt"], params["gamma"], params["omega"]
        n = bank.h.shape[0]
        keys = self.field.sparse_candidate_keys
        edge_src, edge_dst = keys // n, keys % n
        weights = torch.sigmoid(self.field.sparse_topology_logits)
        # Self-edges have zero exchange and do not dilute neighbor coupling.
        weights = weights * (edge_src != edge_dst)
        degree = weights.new_zeros(n).index_add(0, edge_dst, weights)
        normalized = weights / (degree[edge_dst] + 1e-8)
        operator = torch.sparse_coo_tensor(torch.stack([edge_dst, edge_src]), normalized,
                                           (n, n)).coalesce()
        row_mass = degree / (degree + 1e-8)
        amplitude = params["input_force_scale"] * torch.sigmoid(self.field._topology_logits_for(src, dst))
        drive = self.drive_vector if self.state_dim > 1 else bank.h.new_ones(1)
        # Group once, avoiding a scan over every event for every timestamp.
        order = torch.argsort(times, stable=True)
        _, counts = torch.unique_consecutive(times[order], return_counts=True)
        offset = 0
        h, v = bank.h, bank.v
        for count in counts.tolist():
            events = order[offset:offset + count]
            offset += count
            incoming = torch.zeros_like(h).index_add(0, dst[events], amplitude[events, None] * drive)
            exchange = torch.sparse.mm(operator, h) - row_mass[:, None] * h
            v = (1 - gamma * dt) * v + dt * (incoming + kappa * exchange - omega.square() * h)
            h = h + dt * v
        bank.h, bank.v = h, v

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
        src_embedding = self.projection(torch.cat([bank.h[src], bank.v[src]], dim=-1))
        dst_embedding = self.projection(torch.cat([bank.h[dst], bank.v[dst]], dim=-1))
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
    kwargs.pop("fnn_coupling", None)
    from models.MemoryModel import MemoryModel as NativeMemoryModel
    return NativeMemoryModel(*args, model_name=model_name, **kwargs)
