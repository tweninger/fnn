# aggregators/ift_operator.py
from __future__ import annotations
from typing import Optional
import torch

from interactiondynamics.core.events import EventBatch
from interactiondynamics.core.interfaces import Aggregator, ModelState


class IFTLaplacianAggregator(Aggregator):
    """
    Produces per-node messages by summing event embeddings to dst (and optionally src),
    and stashes a Laplacian L in state.aux["L"].

    In `current_bin` mode, L is induced only by the current event bin.
    In `ema` mode, the underlying adjacency is smoothed over time as:
        A_ema = beta * A_prev + (1 - beta) * A_current
    """

    def __init__(
        self,
        add_to_dst: bool = True,
        add_to_src: bool = False,
        make_undirected: bool = True,
        laplacian_mode: str = "ema",
        ema_beta: float = 0.9,
        message_reduce: str = "sum",
        force_reduce: str = "sum",
        disable_laplacian: bool = False,
        randomize_laplacian: bool = False,
        identity_laplacian: bool = False,
        zero_messages: bool = False,
        direct_drive: bool = False,
    ):
        super().__init__()
        self.add_to_dst = bool(add_to_dst)
        self.add_to_src = bool(add_to_src)
        self.make_undirected = bool(make_undirected)
        if laplacian_mode not in {"current_bin", "ema", "fixed_ring"}:
            raise ValueError(f"unknown laplacian_mode={laplacian_mode}")
        if not 0.0 <= float(ema_beta) < 1.0:
            raise ValueError("ema_beta must lie in [0, 1)")
        if message_reduce not in {"mean", "sum"}:
            raise ValueError(f"unknown message_reduce={message_reduce}")
        if force_reduce not in {"mean", "sum"}:
            raise ValueError(f"unknown force_reduce={force_reduce}")
        self.laplacian_mode = str(laplacian_mode)
        self.ema_beta = float(ema_beta)
        self.message_reduce = str(message_reduce)
        self.force_reduce = str(force_reduce)
        self.disable_laplacian = bool(disable_laplacian)
        self.randomize_laplacian = bool(randomize_laplacian)
        self.identity_laplacian = bool(identity_laplacian)
        self.zero_messages = bool(zero_messages)
        self.direct_drive = bool(direct_drive)

    @torch.no_grad()
    def _adjacency_from_events(self, src: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
        device = src.device
        M = int(src.numel())
        if M == 0:
            idx = torch.empty((2, 0), dtype=torch.long, device=device)
            vals = torch.empty((0,), dtype=torch.float32, device=device)
            return torch.sparse_coo_tensor(idx, vals, (num_nodes, num_nodes), device=device).coalesce()

        w = torch.ones((M,), dtype=torch.float32, device=device)

        i = src
        j = dst
        v = w
        if self.make_undirected:
            i = torch.cat([src, dst], dim=0)
            j = torch.cat([dst, src], dim=0)
            v = torch.cat([w, w], dim=0)

        return torch.sparse_coo_tensor(
            torch.stack([i, j], dim=0),
            v,
            (num_nodes, num_nodes),
            device=device,
        ).coalesce()

    @torch.no_grad()
    def _ring_adjacency(self, num_nodes: int, device: torch.device) -> torch.Tensor:
        if num_nodes <= 1:
            idx = torch.empty((2, 0), dtype=torch.long, device=device)
            vals = torch.empty((0,), dtype=torch.float32, device=device)
            return torch.sparse_coo_tensor(idx, vals, (num_nodes, num_nodes), device=device).coalesce()

        src = torch.arange(num_nodes, device=device, dtype=torch.long)
        dst_fwd = (src + 1) % num_nodes
        dst_bwd = (src - 1) % num_nodes
        idx = torch.stack(
            [
                torch.cat([src, src], dim=0),
                torch.cat([dst_fwd, dst_bwd], dim=0),
            ],
            dim=0,
        )
        vals = torch.ones((idx.size(1),), dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(idx, vals, (num_nodes, num_nodes), device=device).coalesce()

    @torch.no_grad()
    def _ema_adjacency(
        self,
        prev_adj: Optional[torch.Tensor],
        curr_adj: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        if prev_adj is None:
            return curr_adj.detach().coalesce()

        prev = prev_adj.detach().coalesce()
        # Detach the current-bin adjacency before folding it into the running EMA
        # so graph history never participates in backward through time.
        curr = curr_adj.detach().coalesce()
        beta = self.ema_beta

        idx = torch.cat([prev.indices(), curr.indices()], dim=1)
        vals = torch.cat([beta * prev.values(), (1.0 - beta) * curr.values()], dim=0)
        out = torch.sparse_coo_tensor(idx, vals, (num_nodes, num_nodes), device=curr.device).coalesce()
        if out._nnz() == 0:
            return out
        keep = out.values().abs() > 0.0
        if torch.all(keep):
            return out
        return torch.sparse_coo_tensor(
            out.indices()[:, keep],
            out.values()[keep],
            (num_nodes, num_nodes),
            device=curr.device,
        ).coalesce()

    @torch.no_grad()
    def _random_adjacency_like(self, A: torch.Tensor, num_nodes: int) -> torch.Tensor:
        device = A.device
        nnz = int(A._nnz())
        if nnz == 0 or num_nodes <= 1:
            idx = torch.empty((2, 0), dtype=torch.long, device=device)
            vals = torch.empty((0,), dtype=torch.float32, device=device)
            return torch.sparse_coo_tensor(idx, vals, (num_nodes, num_nodes), device=device).coalesce()

        target = nnz
        max_edges = max(num_nodes * max(num_nodes - 1, 0), 1)
        sample_n = min(max(target * 2, 1), max_edges)
        src = torch.randint(num_nodes, (sample_n,), device=device)
        dst = torch.randint(num_nodes, (sample_n,), device=device)
        keep = src != dst
        if not torch.any(keep):
            return self._ring_adjacency(num_nodes, device)
        src = src[keep]
        dst = dst[keep]
        pair_ids = torch.unique(src * num_nodes + dst, sorted=False)
        pair_ids = pair_ids[:target]
        rand_src = pair_ids // num_nodes
        rand_dst = pair_ids % num_nodes
        vals = torch.ones((pair_ids.numel(),), dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(
            torch.stack([rand_src, rand_dst], dim=0),
            vals,
            (num_nodes, num_nodes),
            device=device,
        ).coalesce()

    @torch.no_grad()
    def _laplacian_from_adjacency(self, A: torch.Tensor, num_nodes: int) -> torch.Tensor:
        device = A.device
        deg = torch.sparse.sum(A, dim=1).to_dense()  # (N,)
        deg_inv_sqrt = torch.rsqrt(deg.clamp_min(1.0))  # (N,)

        Ai, Aj = A.indices()
        Av = A.values()
        Av_norm = Av * deg_inv_sqrt[Ai] * deg_inv_sqrt[Aj]
        A_norm = torch.sparse_coo_tensor(
            torch.stack([Ai, Aj], dim=0),
            Av_norm,
            (num_nodes, num_nodes),
            device=device
        ).coalesce()

        diag = torch.arange(num_nodes, device=device)
        I = torch.sparse_coo_tensor(
            torch.stack([diag, diag], dim=0),
            torch.ones((num_nodes,), dtype=torch.float32, device=device),
            (num_nodes, num_nodes),
            device=device
        ).coalesce()

        L = (I - A_norm).coalesce()
        return L

    @torch.no_grad()
    def _identity_laplacian(self, num_nodes: int, device: torch.device) -> torch.Tensor:
        diag = torch.arange(num_nodes, device=device, dtype=torch.long)
        vals = torch.ones((num_nodes,), dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(
            torch.stack([diag, diag], dim=0),
            vals,
            (num_nodes, num_nodes),
            device=device,
        ).coalesce()

    @torch.no_grad()
    def _zero_laplacian(self, num_nodes: int, device: torch.device) -> torch.Tensor:
        idx = torch.empty((2, 0), dtype=torch.long, device=device)
        vals = torch.empty((0,), dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(idx, vals, (num_nodes, num_nodes), device=device).coalesce()

    @torch.no_grad()
    def _reduce_to_nodes(
        self,
        values: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        num_nodes: int,
        *,
        reduce: str,
    ) -> torch.Tensor:
        out = torch.zeros((num_nodes, values.size(1)), device=values.device, dtype=values.dtype)
        if values.size(0) == 0:
            return out

        if self.add_to_dst:
            out.index_add_(0, dst, values)
        if self.add_to_src:
            out.index_add_(0, src, values)

        if reduce == "mean":
            counts = torch.zeros((num_nodes,), device=values.device, dtype=values.dtype)
            ones = torch.ones((values.size(0),), device=values.device, dtype=values.dtype)
            if self.add_to_dst:
                counts.index_add_(0, dst, ones)
            if self.add_to_src:
                counts.index_add_(0, src, ones)
            out = out / counts.clamp_min(1.0).unsqueeze(-1)
        return out

    @torch.no_grad()
    def _force_feature_summary(
        self,
        events: EventBatch,
        src: torch.Tensor,
        dst: torch.Tensor,
        num_nodes: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if events.features is None or events.features.numel() == 0:
            return None
        features = events.features.to(device)
        if features.dim() != 2:
            return None
        return self._reduce_to_nodes(features, src, dst, num_nodes, reduce=self.force_reduce)

    @torch.no_grad()
    def _direct_drive_summary(self, events: EventBatch, num_nodes: int, device: torch.device) -> Optional[torch.Tensor]:
        if not self.direct_drive or events.features is None or events.features.numel() == 0:
            return None

        features = events.features.to(device)
        if features.dim() != 2 or features.size(0) == 0:
            return None

        drive = features[:, :1]
        if features.size(1) >= 2:
            drive = drive * features[:, 1:2]

        dst = events.dst.to(device=device, dtype=torch.long)
        out = torch.zeros((num_nodes, 1), device=device, dtype=features.dtype)
        out.index_add_(0, dst, drive)
        return out


    def forward(
        self,
        state: Optional[ModelState],
        event_embeddings: torch.Tensor,  # (M, d_msg)
        events: EventBatch,
        num_nodes: int,
    ) -> torch.Tensor:
        device = event_embeddings.device
        M, d = event_embeddings.shape

        msg = torch.zeros((num_nodes, d), device=device)

        src = events.src.to(device)
        dst = events.dst.to(device)

        if self.add_to_dst and M > 0:
            msg.index_add_(0, dst, event_embeddings)
        if self.add_to_src and M > 0:
            msg.index_add_(0, src, event_embeddings)

        if self.message_reduce == "mean" and (self.add_to_dst or self.add_to_src) and M > 0:
            # count how many embeddings were added into each node
            cnt = torch.zeros((num_nodes,), device=device, dtype=event_embeddings.dtype)

            if self.add_to_dst:
                cnt.index_add_(0, dst, torch.ones((M,), device=device, dtype=event_embeddings.dtype))
            if self.add_to_src:
                cnt.index_add_(0, src, torch.ones((M,), device=device, dtype=event_embeddings.dtype))

            # avoid divide-by-zero; broadcast to (N,1)
            msg = msg / cnt.clamp_min(1.0).unsqueeze(-1)     

        if self.zero_messages:
            msg.zero_()

        if state is not None:
            if state.aux is None:
                state.aux = {}
            force_features = self._force_feature_summary(events, src, dst, num_nodes, device)
            if force_features is not None:
                state.aux["ift_force_features"] = force_features.detach()
            else:
                state.aux.pop("ift_force_features", None)
            if self.direct_drive:
                direct_drive = self._direct_drive_summary(events, num_nodes, device)
                if direct_drive is not None:
                    state.aux["ift_direct_drive"] = direct_drive.detach()
                else:
                    state.aux.pop("ift_direct_drive", None)

            if self.laplacian_mode == "fixed_ring":
                A_effective = self._ring_adjacency(num_nodes=num_nodes, device=device)
                state.aux.pop("A_ema", None)
            else:
                A_current = self._adjacency_from_events(src, dst, num_nodes=num_nodes)
                if self.laplacian_mode == "ema":
                    A_prev = state.aux.get("A_ema", None)
                    A_effective = self._ema_adjacency(A_prev, A_current, num_nodes=num_nodes)
                    state.aux["A_ema"] = A_effective.detach()
                else:
                    A_effective = A_current
                    state.aux.pop("A_ema", None)

            if self.randomize_laplacian:
                A_effective = self._random_adjacency_like(A_effective, num_nodes=num_nodes)
                state.aux.pop("A_ema", None)

            if self.disable_laplacian:
                L = self._zero_laplacian(num_nodes=num_nodes, device=device)
            elif self.identity_laplacian:
                L = self._identity_laplacian(num_nodes=num_nodes, device=device)
            else:
                L = self._laplacian_from_adjacency(A_effective, num_nodes=num_nodes)

            state.aux["L"] = L
            state.aux["ift_laplacian_mode"] = self.laplacian_mode
            state.aux["ift_message_reduce"] = self.message_reduce
            state.aux["ift_force_reduce"] = self.force_reduce
            state.aux["ift_disable_laplacian"] = self.disable_laplacian
            state.aux["ift_randomize_laplacian"] = self.randomize_laplacian
            state.aux["ift_identity_laplacian"] = self.identity_laplacian
            state.aux["ift_zero_messages"] = self.zero_messages
            state.aux["ift_direct_drive_enabled"] = self.direct_drive

            if events.t is not None:
                state.aux["L_bin_t_min"] = int(events.t.min().item())
                state.aux["L_bin_t_max"] = int(events.t.max().item())

        return msg
