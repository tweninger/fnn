from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, cast, List

import torch

from core.events import EventBatch
from data.interfaces import DataSpec, EventStreamDataset


@dataclass
class NBodyConfig:
    name: str = "nbody_fullgraph_2d"

    # number of particles / nodes
    num_nodes: int = 32

    # number of time bins
    num_bins: int = 2000

    # simple dynamics params
    dt: float = 0.02
    damping: float = 0.995
    interaction_strength: float = 1.0
    softening: float = 0.1

    # initialization
    init_pos_scale: float = 1.0
    init_vel_scale: float = 0.1

    # train/val/test split by time
    split_fracs: tuple[float, float, float] = (0.7, 0.15, 0.15)

    seed: int = 0
    device: Optional[torch.device] = None


class NBodyDataset(EventStreamDataset):
    """
    Continuous synthetic 2D interacting-particle temporal dataset.

    Hidden state per node:
      - 2D position
      - 2D velocity

    At each timestep:
      - compute all-pairs distance-based forces
      - update positions/velocities
      - emit a full directed graph over all ordered pairs i != j

    Event features:
      [dx, dy, dvx, dvy, dist, strength]
    """

    def __init__(self, cfg: NBodyConfig):
        self.cfg = cfg
        self._event_dim = 6

        self._build()
        self._split()

    def _build(self) -> None:
        g = torch.Generator().manual_seed(self.cfg.seed)

        N = self.cfg.num_nodes
        T = self.cfg.num_bins

        dt = self.cfg.dt
        damping = self.cfg.damping
        strength_scale = self.cfg.interaction_strength
        eps = self.cfg.softening

        # positions and velocities in 2D
        x = self.cfg.init_pos_scale * torch.randn(N, 2, generator=g, dtype=torch.float32)
        v = self.cfg.init_vel_scale * torch.randn(N, 2, generator=g, dtype=torch.float32)

        self.src_bins: List[torch.Tensor] = []
        self.dst_bins: List[torch.Tensor] = []
        self.t_bins: List[torch.Tensor] = []
        self.feat_bins: List[torch.Tensor] = []

        for b in range(T):
            # pairwise relative positions: rel[i, j] = x[j] - x[i]
            rel = x.unsqueeze(1) - x.unsqueeze(0)   # x[i] - x[j]
            rel = -rel                              # now x[j] - x[i]

            # softened squared distances
            dist2 = (rel ** 2).sum(dim=-1) + eps * eps
            dist = torch.sqrt(dist2)

            # interaction law:
            # force_ij ~ (x_j - x_i) / ||x_j - x_i||^3
            # with softening for stability
            inv_dist3 = 1.0 / (dist2 * dist)
            inv_dist3[torch.arange(N), torch.arange(N)] = 0.0

            force_pair = strength_scale * rel * inv_dist3.unsqueeze(-1)  # [N, N, 2]

            # net force on each node: sum_j force_ij
            net_force = force_pair.sum(dim=1)  # [N, 2]

            # build full directed event graph
            src_list: List[int] = []
            dst_list: List[int] = []
            feat_list: List[List[float]] = []

            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue

                    rel_ij = x[j] - x[i]
                    rel_vij = v[j] - v[i]

                    # use same softened geometry as above
                    dist2_ij = torch.dot(rel_ij, rel_ij) + eps * eps
                    dist_ij = torch.sqrt(dist2_ij)

                    # scalar interaction strength ~ 1 / r^2
                    strength_ij = strength_scale / dist2_ij

                    feat = [
                        float(rel_ij[0]),
                        float(rel_ij[1]),
                        float(rel_vij[0]),
                        float(rel_vij[1]),
                        float(dist_ij),
                        float(strength_ij),
                    ]

                    src_list.append(i)
                    dst_list.append(j)
                    feat_list.append(feat)

            src = torch.tensor(src_list, dtype=torch.long)
            dst = torch.tensor(dst_list, dtype=torch.long)
            feats = torch.tensor(feat_list, dtype=torch.float32)
            t = torch.full((src.numel(),), b, dtype=torch.long)

            self.src_bins.append(src)
            self.dst_bins.append(dst)
            self.t_bins.append(t)
            self.feat_bins.append(feats)

            # update system state
            a = net_force
            v = damping * (v + dt * a)
            x = x + dt * v

        self._num_bins = len(self.src_bins)
        self._num_events = sum(s.numel() for s in self.src_bins)

    def _split(self) -> None:
        T = self._num_bins
        f_tr, f_va, f_te = self.cfg.split_fracs

        assert abs((f_tr + f_va + f_te) - 1.0) < 1e-6, "split_fracs must sum to 1.0"

        n_tr = int(T * f_tr)
        n_va = int(T * f_va)

        self.split_bins = {
            "train": list(range(0, n_tr)),
            "val": list(range(n_tr, n_tr + n_va)),
            "test": list(range(n_tr + n_va, T)),
        }

    def spec(self) -> DataSpec:
        return DataSpec(
            name=self.cfg.name,
            num_nodes=self.cfg.num_nodes,
            event_dim=self._event_dim,
            num_events=self._num_events,
            num_bins=self._num_bins,
            extra={
                "dt": self.cfg.dt,
                "damping": self.cfg.damping,
                "interaction_strength": self.cfg.interaction_strength,
                "softening": self.cfg.softening,
                "seed": self.cfg.seed,
            },
        )

    def bins(self, split: str = "train") -> Iterable[EventBatch]:
        assert split in self.split_bins, f"unknown split={split}"
        return _NBodyStream(
            self.src_bins,
            self.dst_bins,
            self.t_bins,
            self.feat_bins,
            self.split_bins[split],
            self.cfg.device,
        )


class _NBodyStream(Iterable[EventBatch]):
    def __init__(self, src, dst, t, feat, idxs, device):
        self.src = src
        self.dst = dst
        self.t = t
        self.feat = feat
        self.idxs = idxs
        self.device = device

    def __iter__(self) -> Iterator[EventBatch]:
        for i in self.idxs:
            eb = EventBatch(
                src=cast(torch.LongTensor, self.src[i]),
                dst=cast(torch.LongTensor, self.dst[i]),
                t=cast(torch.LongTensor, self.t[i]),
                features=self.feat[i],
            )
            if self.device is not None:
                eb = eb.to(self.device)
            yield eb