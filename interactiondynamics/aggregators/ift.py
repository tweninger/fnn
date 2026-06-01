# aggregators/ift_operator.py
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

from interactiondynamics.core.events import EventBatch
from interactiondynamics.core.interfaces import Aggregator, ModelState


class IFTLaplacianAggregator(Aggregator):
    """
    Produces per-node messages by summing event embeddings to dst (and optionally src),
    and stashes a bin-local Laplacian L in state.aux["L"].

    L is induced only by the event stream in the current bin (no persistent graph).
    """

    def __init__(self, add_to_dst: bool = True, add_to_src: bool = False, make_undirected: bool = True):
        super().__init__()
        self.add_to_dst = bool(add_to_dst)
        self.add_to_src = bool(add_to_src)
        self.make_undirected = bool(make_undirected)

    @torch.no_grad()
    def _laplacian_from_events(self, src: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
        device = src.device
        M = int(src.numel())
        if M == 0:
            # L = I (or 0). Use 0 to be "no operator" in empty bins.
            idx = torch.empty((2, 0), dtype=torch.long, device=device)
            vals = torch.empty((0,), dtype=torch.float32, device=device)
            return torch.sparse_coo_tensor(idx, vals, (num_nodes, num_nodes), device=device).coalesce()

        # edge weights = 1 per event
        w = torch.ones((M,), dtype=torch.float32, device=device)

        i = src
        j = dst
        v = w
        if self.make_undirected:
            i = torch.cat([src, dst], dim=0)
            j = torch.cat([dst, src], dim=0)
            v = torch.cat([w, w], dim=0)

        # adjacency A
        A = torch.sparse_coo_tensor(torch.stack([i, j], dim=0), v, (num_nodes, num_nodes), device=device).coalesce()

        # degree (avoid div-by-zero)
        deg = torch.sparse.sum(A, dim=1).to_dense()  # (N,)
        deg_inv_sqrt = torch.rsqrt(deg.clamp_min(1.0))  # (N,)

        # normalized adjacency: A_norm = D^{-1/2} A D^{-1/2}
        Ai, Aj = A.indices()
        Av = A.values()
        Av_norm = Av * deg_inv_sqrt[Ai] * deg_inv_sqrt[Aj]
        A_norm = torch.sparse_coo_tensor(
            torch.stack([Ai, Aj], dim=0),
            Av_norm,
            (num_nodes, num_nodes),
            device=device
        ).coalesce()

        # normalized Laplacian: L = I - A_norm
        diag = torch.arange(num_nodes, device=device)
        I = torch.sparse_coo_tensor(
            torch.stack([diag, diag], dim=0),
            torch.ones((num_nodes,), dtype=torch.float32, device=device),
            (num_nodes, num_nodes),
            device=device
        ).coalesce()

        L = (I - A_norm).coalesce()
        return L


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

        if (self.add_to_dst or self.add_to_src) and M > 0:
            # count how many embeddings were added into each node
            cnt = torch.zeros((num_nodes,), device=device, dtype=event_embeddings.dtype)

            if self.add_to_dst:
                cnt.index_add_(0, dst, torch.ones((M,), device=device, dtype=event_embeddings.dtype))
            if self.add_to_src:
                cnt.index_add_(0, src, torch.ones((M,), device=device, dtype=event_embeddings.dtype))

            # avoid divide-by-zero; broadcast to (N,1)
            msg = msg / cnt.clamp_min(1.0).unsqueeze(-1)     
                   

        if state is not None:
            if state.aux is None:
                state.aux = {}
            state.aux["L"] = self._laplacian_from_events(src, dst, num_nodes=num_nodes)

            if events.t is not None:
                state.aux["L_bin_t_min"] = int(events.t.min().item())
                state.aux["L_bin_t_max"] = int(events.t.max().item())

        return msg
