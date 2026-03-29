from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, cast, List

import math
import torch

from core.events import EventBatch
from data.interfaces import DataSpec, EventStreamDataset

"""
- Its spring_mass.py... but a cycle instead of a chain
- Main changes:
    + nodes start on a circle
    + spring force is now a vector (for chain its scalar foce because its on a line). Now the spring can pull in both x and y directions
"""

@dataclass
class SpringRing2DConfig:
    name: str = "spring_ring_2d"

    num_nodes: int = 64
    num_bins: int = 4000

    # simulation parameters
    dt: float = 0.05 # lower for more stability
    spring_k: float = 2.0 # increase for stronger events
    damping: float = 0.995

    # geometry
    ring_radius: float = 5.0

    # random perturbations
    init_pos_noise: float = 0.30
    init_vel_noise: float = 0.25

    # only emit events when spring force magnitude exceeds threshold
    force_threshold: float = 0.015 # lower for more events
    bidirectional: bool = True

    # train/val/test split by time
    split_fracs: tuple = (0.7, 0.15, 0.15)

    seed: int = 0
    device: Optional[torch.device] = None


class SpringRing2DDataset(EventStreamDataset):
    def __init__(self, cfg: SpringRing2DConfig):
        self.cfg = cfg

        # features:
        # [dx, dy, dvx, dvy, dist, extension, fx, fy]
        self._event_dim = 8

        self._build()
        self._split()

    def _build(self):
        g = torch.Generator().manual_seed(self.cfg.seed)

        N = self.cfg.num_nodes
        T = self.cfg.num_bins

        dt = self.cfg.dt
        k = self.cfg.spring_k
        damping = self.cfg.damping
        radius = self.cfg.ring_radius
        thr = self.cfg.force_threshold
        bidir = self.cfg.bidirectional

        # --------------------------------------------------
        # 1) Initialize positions on a circle in 2D
        # --------------------------------------------------
        angles = torch.linspace(0, 2 * math.pi, steps=N + 1, dtype=torch.float32)[:-1]

        x = torch.stack(
            [
                radius * torch.cos(angles),
                radius * torch.sin(angles),
            ],
            dim=1,
        )  # shape [N, 2]

        # add a little positional noise
        x = x + self.cfg.init_pos_noise * torch.randn(N, 2, generator=g)

        # small random initial velocities in 2D
        v = self.cfg.init_vel_noise * torch.randn(N, 2, generator=g)

        # --------------------------------------------------
        # 2) Rest length = distance between neighboring points
        #    on the ideal ring
        # --------------------------------------------------
        neighbor_angle = 2 * math.pi / N
        rest = 2 * radius * math.sin(neighbor_angle / 2)

        self.src_bins: List[torch.Tensor] = []
        self.dst_bins: List[torch.Tensor] = []
        self.t_bins: List[torch.Tensor] = []
        self.feat_bins: List[torch.Tensor] = []

        # --------------------------------------------------
        # 3) Simulate through time
        # --------------------------------------------------
        for b in range(T):
            net_force = torch.zeros(N, 2, dtype=torch.float32)

            src_list, dst_list, feat_list = [], [], []

            # ring topology: i connected to (i+1)%N
            # dpos: where j is relative to i
            # dist: how far apart they are
            # direction: which way the spring points
            # extension: stretched vs compressed
            # force_vec: gives actual 2D force
            for i in range(N):
                j = (i + 1) % N

                # vector from i to j
                dpos = x[j] - x[i]         # shape [2]
                dvel = v[j] - v[i]         # shape [2]

                dist = torch.norm(dpos) + 1e-8
                direction = dpos / dist

                extension = dist - rest

                # Hooke's law in vector form:
                # force points along the spring direction
                force_vec = k * extension * direction

                net_force[i] += force_vec
                net_force[j] -= force_vec

                force_mag = torch.norm(force_vec)

                if force_mag > thr:
                    feat = [
                        float(dpos[0]),
                        float(dpos[1]),
                        float(dvel[0]),
                        float(dvel[1]),
                        float(dist),
                        float(extension),
                        float(force_vec[0]),
                        float(force_vec[1]),
                    ]

                    src_list.append(i)
                    dst_list.append(j)
                    feat_list.append(feat)

                    if bidir:
                        feat_rev = [
                            float(-dpos[0]),
                            float(-dpos[1]),
                            float(-dvel[0]),
                            float(-dvel[1]),
                            float(dist),          # distance stays positive
                            float(extension),     # extension stays the same
                            float(-force_vec[0]),
                            float(-force_vec[1]),
                        ]
                        src_list.append(j)
                        dst_list.append(i)
                        feat_list.append(feat_rev)

            if len(src_list) > 0:
                src = torch.tensor(src_list, dtype=torch.long)
                dst = torch.tensor(dst_list, dtype=torch.long)
                feats = torch.tensor(feat_list, dtype=torch.float32)
                t = torch.full((len(src_list),), b, dtype=torch.long)

                self.src_bins.append(src)
                self.dst_bins.append(dst)
                self.t_bins.append(t)
                self.feat_bins.append(feats)

            # --------------------------------------------------
            # 4) Update dynamics
            # --------------------------------------------------
            a = net_force                    # assume unit mass
            v = damping * (v + dt * a)
            x = x + dt * v

        self._num_bins = len(self.src_bins)
        self._num_events = sum(s.numel() for s in self.src_bins)

    def _split(self):
        T = self._num_bins
        f_tr, f_va, f_te = self.cfg.split_fracs

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
        )

    def bins(self, split: str = "train") -> Iterable[EventBatch]:
        return _Stream(
            self.src_bins,
            self.dst_bins,
            self.t_bins,
            self.feat_bins,
            self.split_bins[split],
            self.cfg.device,
        )


class _Stream(Iterable[EventBatch]):
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