"""FNN memory backbone for the pinned upstream DyGLib training loop."""
import torch
from torch import nn

from interactiondynamics.models.fnn import FieldNeuralNetwork


class FieldMemory(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.register_buffer("h", torch.zeros(num_nodes, 1))
        self.register_buffer("v", torch.zeros(num_nodes, 1))
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
    def __init__(self, node_raw_features, edge_raw_features, neighbor_sampler, **kwargs):
        super().__init__()
        n, width = node_raw_features.shape
        self.field = FieldNeuralNetwork(
            num_nodes=n, force_dim=1, state_dim=1, gamma_init=0.15,
            omega_init=0.8, dt=0.1, topology_mode="observed_sparse",
            learn_gamma=True, learn_omega=True, learn_input_force_scale=True,
        )
        # Candidate support comes only from the training neighbor sampler.
        # DyGLib stores neighbors in both directions; neither test pairs nor
        # test features are consulted when creating learnable gates.
        src, dst = [], []
        for node, neighbors in enumerate(neighbor_sampler.nodes_neighbor_ids):
            src.extend([node] * len(neighbors))
            dst.extend(neighbors.tolist())
        self.field.set_sparse_topology_candidates(torch.tensor(src, dtype=torch.long),
                                                  torch.tensor(dst, dtype=torch.long))
        self.memory_bank = FieldMemory(n)
        self.projection = nn.Linear(2, width)

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
        params = self.field.physical_parameters()
        dt, gamma, omega = params["dt"], params["gamma"], params["omega"]
        damp = 1 - gamma * dt
        matrix = torch.stack([torch.stack([1 - dt.square() * omega.square(), dt * damp]),
                              torch.stack([-dt * omega.square(), damp])])
        powers = torch.eye(2, device=matrix.device, dtype=matrix.dtype).expand(steps + 1, 2, 2)
        exponents = torch.arange(steps + 1, device=matrix.device)
        base = matrix
        for bit in range(steps.bit_length()):
            powers = torch.where(((exponents >> bit) & 1).bool()[:, None, None], powers @ base, powers)
            base = base @ base
        bank = self.memory_bank
        initial = torch.cat([bank.h, bank.v], dim=-1)
        evolved = initial @ powers[steps].T
        drive = torch.stack([dt.square(), dt])
        response = powers[steps - 1 - group] @ drive
        gates = torch.sigmoid(self.field._topology_logits_for(src, dst))
        response = response * (params["input_force_scale"] * gates)[:, None]
        evolved = evolved.index_add(0, dst, response)
        bank.h, bank.v = evolved[:, :1], evolved[:, 1:]

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
    from models.MemoryModel import MemoryModel as NativeMemoryModel
    return NativeMemoryModel(*args, model_name=model_name, **kwargs)
