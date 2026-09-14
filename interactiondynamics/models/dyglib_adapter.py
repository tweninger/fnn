"""DyGLib backbones under the existing predict-before-observe bin protocol.

This adapts architectures, not DyGLib's published evaluation protocol. Node 0
is reserved for upstream padding. Only step() adds history; score() is read-only.
"""
import numpy as np
import torch
from torch import nn

from interactiondynamics.core.interfaces import ModelState
from .dyglib_vendor.neighbor_sampler import NeighborSampler
from .dyglib_vendor.EdgeBank import edge_bank_unlimited_memory, predict_link_probabilities


def _detach(value):
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, dict):
        return {k: _detach(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_detach(v) for v in value)
    return value


class HistoryState(ModelState):
    def detach_(self):
        self.node = _detach(self.node)
        self.aux = _detach(self.aux)
        return self

    def clone(self, detach=False):
        copy = super().clone(detach=detach)
        return HistoryState(node=copy.node, aux=copy.aux)


class EdgeBankAdapter(nn.Module):
    parameter_free = True
    requires_unpacked_episodes = True

    def __init__(self, num_nodes, event_dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.event_dim = event_dim
        self.register_buffer("anchor", torch.zeros(1))

    def init_state(self, batch_size, num_nodes, device):
        if batch_size != 1 or num_nodes != self.num_nodes:
            raise ValueError("DyGLib adapters require one graph per state (no episode packing).")
        return HistoryState(node=torch.zeros(num_nodes, 1, device=device), aux={
            "src": torch.empty(0, dtype=torch.long, device=device),
            "dst": torch.empty(0, dtype=torch.long, device=device),
            "time": torch.empty(0, dtype=torch.float64, device=device),
            "features": torch.empty(0, self.event_dim, device=device),
        })

    def step(self, state, events, drive=None):
        if not events.num_events:
            return state, {}
        if events.t is None:
            raise ValueError("DyGLib adapters require event timestamps.")
        aux = dict(state.aux)
        features = events.features
        if features is None:
            features = torch.zeros(events.num_events, self.event_dim, device=events.src.device)
        for key, value in (("src", events.src), ("dst", events.dst),
                           ("time", events.t.double()), ("features", features)):
            aux[key] = torch.cat([aux[key], value], dim=0)
        return HistoryState(node=state.node, aux=aux), {}

    def score(self, state, candidate_events):
        a = state.aux
        memory = edge_bank_unlimited_memory(a["src"].cpu().numpy(), a["dst"].cpu().numpy())
        scores = predict_link_probabilities(memory, (candidate_events.src.cpu().numpy(), candidate_events.dst.cpu().numpy()))
        return torch.as_tensor(scores, dtype=self.anchor.dtype, device=self.anchor.device)


class DyGLibAdapter(EdgeBankAdapter):
    parameter_free = False

    def __init__(self, num_nodes, event_dim, cfg):
        super().__init__(num_nodes, event_dim)
        self.kind = cfg.temporal_model
        self.num_neighbors = cfg.temporal_num_neighbors
        self.node_dim = cfg.node_dim
        nodes = np.zeros((num_nodes + 1, cfg.node_dim), dtype=np.float32)
        edges = np.zeros((1, event_dim), dtype=np.float32)
        sampler = NeighborSampler([[] for _ in range(num_nodes + 1)], sample_neighbor_strategy="recent")
        common = dict(node_raw_features=nodes, edge_raw_features=edges,
                      neighbor_sampler=sampler, time_feat_dim=cfg.time_emb_dim, dropout=cfg.dropout)
        if self.kind == "graphmixer":
            from .dyglib_vendor.GraphMixer import GraphMixer
            self.backbone = GraphMixer(**common, num_tokens=self.num_neighbors)
        elif self.kind == "dygformer":
            from .dyglib_vendor.DyGFormer import DyGFormer
            self.backbone = DyGFormer(**common, channel_embedding_dim=cfg.node_dim,
                                     max_input_sequence_length=cfg.temporal_history_length)
        elif self.kind in {"tgn", "jodie"}:
            from .dyglib_vendor.MemoryModel import MemoryModel
            self.backbone = MemoryModel(**common, model_name=self.kind.upper())
        else:
            raise ValueError(self.kind)
        from .dyglib_vendor.modules import MergeLayer
        self.readout = MergeLayer(cfg.node_dim, cfg.node_dim, cfg.scorer_hidden, 1)
        self.event_feature_decoder = nn.Sequential(nn.Linear(2 * cfg.node_dim, cfg.scorer_hidden),
                                                  nn.ReLU(), nn.Linear(cfg.scorer_hidden, event_dim))
        self.event_feature_loss_weight = cfg.event_feature_loss_weight
        self.event_feature_magnitude_weight = cfg.event_feature_magnitude_weight
        self.register_buffer("event_feature_target_std", torch.ones(event_dim))
        self.register_buffer("event_feature_active_threshold", torch.tensor(0.))
        self.register_buffer("event_feature_magnitude_q90", torch.tensor(1.))
        self._history_cache = None

    def init_state(self, batch_size, num_nodes, device):
        state = super().init_state(batch_size, num_nodes, device)
        state.node = torch.zeros(num_nodes, self.node_dim, device=device)
        if self.kind in {"tgn", "jodie"}:
            bank = self.backbone.memory_bank
            bank.detach_memory_bank()
            bank.__init_memory_bank__()
            state.aux["memory"] = bank.backup_memory_bank()
        return state

    def _prepare(self, state):
        # Upstream feature tables/device strings are not registered buffers.
        device = self.anchor.device
        b = self.backbone
        b.device = str(device)
        b.node_raw_features = b.node_raw_features.to(device)
        a = state.aux
        if self._history_cache is not a["src"]:
            adj = [[] for _ in range(self.num_nodes + 1)]
            for eid, (src, dst, time) in enumerate(zip(a["src"].tolist(), a["dst"].tolist(), a["time"].tolist()), 1):
                adj[src + 1].append((dst + 1, eid, time))
                adj[dst + 1].append((src + 1, eid, time))
            self._sampler = NeighborSampler(adj, sample_neighbor_strategy="recent")
            self._history_cache = a["src"]
        b.edge_raw_features = torch.cat([self.anchor.new_zeros(1, self.event_dim), a["features"]])
        if self.kind in {"tgn", "jodie"}:
            b.embedding_module.device = str(device)
            b.embedding_module.node_raw_features = b.node_raw_features
            b.embedding_module.edge_raw_features = b.edge_raw_features
            b.embedding_module.neighbor_sampler = self._sampler
            b.memory_bank.detach_memory_bank()
            b.memory_bank.reload_memory_bank(a["memory"])
        else:
            b.neighbor_sampler = self._sampler
            if self.kind == "dygformer":
                b.neighbor_co_occurrence_encoder.device = str(device)

    def _embeddings(self, state, events, observe=False, edge_ids=None):
        self._prepare(state)
        if events.t is None:
            raise ValueError("DyGLib queries require timestamps.")
        kwargs = dict(src_node_ids=events.src.detach().cpu().numpy() + 1,
                      dst_node_ids=events.dst.detach().cpu().numpy() + 1,
                      node_interact_times=events.t.detach().cpu().numpy())
        if self.kind in {"tgn", "jodie"}:
            kwargs.update(edge_ids=edge_ids, edges_are_positive=observe, num_neighbors=self.num_neighbors)
        elif self.kind == "graphmixer":
            kwargs.update(num_neighbors=self.num_neighbors)
        return self.backbone.compute_src_dst_node_temporal_embeddings(**kwargs)

    def step(self, state, events, drive=None):
        next_state, aux = super().step(state, events, drive)
        if self.kind in {"tgn", "jodie"} and events.num_events:
            start = state.aux["src"].numel() + 1
            self._embeddings(next_state, events, observe=True,
                             edge_ids=np.arange(start, start + events.num_events))
            next_state.aux["memory"] = self.backbone.memory_bank.backup_memory_bank()
        return next_state, aux

    def score(self, state, candidate_events):
        src, dst = self._embeddings(state, candidate_events)
        return self.readout(src, dst).squeeze(-1)

    def predict_event_features(self, state, events):
        src, dst = self._embeddings(state, events)
        return self.event_feature_decoder(torch.cat([src, dst], dim=-1))

    def configure_event_feature_objective(self, target_std, active_threshold, magnitude_q90, magnitude_weight):
        self.event_feature_target_std.copy_(target_std.to(self.anchor).clamp_min(1e-8))
        self.event_feature_active_threshold.fill_(active_threshold)
        self.event_feature_magnitude_q90.fill_(max(magnitude_q90, 1e-8))
        self.event_feature_magnitude_weight = magnitude_weight
