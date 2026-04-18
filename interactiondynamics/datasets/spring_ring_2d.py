from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, cast, List, Sequence

import math
import torch

from core.events import EventBatch
from datasets.interfaces import DataSpec, EventStreamDataset


@dataclass
class SpringRing2DConfig:
    name: str = "spring_ring_2d"

    num_nodes: int = 32
    num_bins: int = 256

    # simulation parameters
    dt: float = 0.05
    spring_k: float = 1.0
    damping: float = 0.995

    # geometry
    ring_radius: float = 5.0

    # random perturbations
    init_pos_noise: float = 0.05
    init_vel_noise: float = 0.05

    # only emit events when spring force magnitude exceeds threshold
    force_threshold: float = 0.015
    bidirectional: bool = True

    # train/val/test split by time
    split_fracs: tuple = (0.7, 0.15, 0.15)

    seed: int = 0
    device: Optional[torch.device] = None

    standardize_node_targets: bool = True


# Keep whichever features you want uncommented.
_EDGE_FEATURE_NAMES: Sequence[str] = (
    # "dx",
    # "dy",
    # "dvx",
    # "dvy",
    # "dist",
    # "extension",
    # "fx",
    # "fy",
)


class SpringRing2DDataset(EventStreamDataset):
    def __init__(self, cfg: SpringRing2DConfig):
        self.cfg = cfg

        # event feature dimension
        self._event_dim = len(_EDGE_FEATURE_NAMES)

        self._build()
        self._split()

        self.dv_mean: Optional[torch.Tensor] = None   # shape [2]
        self.dv_std: Optional[torch.Tensor] = None    # shape [2]

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
        )  # [N, 2]

        x = x + self.cfg.init_pos_noise * torch.randn(N, 2, generator=g)
        v = self.cfg.init_vel_noise * torch.randn(N, 2, generator=g)

        # --------------------------------------------------
        # 2) Rest length between neighboring points
        # --------------------------------------------------
        neighbor_angle = 2 * math.pi / N
        rest = 2 * radius * math.sin(neighbor_angle / 2)

        self.src_bins: List[torch.Tensor] = []
        self.dst_bins: List[torch.Tensor] = []
        self.t_bins: List[torch.Tensor] = []
        self.feat_bins: List[torch.Tensor] = []
        self.node_target_bins: List[torch.Tensor] = []

        # --------------------------------------------------
        # 3) Simulate through time
        # --------------------------------------------------
        for b in range(T):
            net_force = torch.zeros(N, 2, dtype=torch.float32)

            src_list: List[int] = []
            dst_list: List[int] = []
            feat_list: List[List[float]] = []

            # ring topology: i connected to (i+1)%N
            for i in range(N):
                j = (i + 1) % N

                dpos = x[j] - x[i]   # [2]
                dvel = v[j] - v[i]   # [2]

                dist = torch.norm(dpos) + 1e-8
                direction = dpos / dist
                extension = dist - rest

                # Hooke's law
                force_vec = k * extension * direction

                net_force[i] += force_vec
                net_force[j] -= force_vec

                force_mag = torch.norm(force_vec)

                if force_mag > thr:
                    feat = [
                        # float(dpos[0]),
                        # float(dpos[1]),
                        # float(dvel[0]),
                        # float(dvel[1]),
                        # float(dist),
                        # float(extension),
                        # float(force_vec[0]),
                        # float(force_vec[1]),
                    ]

                    src_list.append(i)
                    dst_list.append(j)
                    feat_list.append(feat)

                    if bidir:
                        feat_rev = [
                            #  float(-dpos[0]),
                            #  float(-dpos[1]),
                            #  float(-dvel[0]),
                            #  float(-dvel[1]),
                            #  float(dist),
                            #  float(extension),
                            # float(-force_vec[0]),
                            # float(-force_vec[1]),
                         ]

                        src_list.append(j)
                        dst_list.append(i)
                        feat_list.append(feat_rev)

            # --------------------------------------------------
            # 4) Compute next state and node targets
            # --------------------------------------------------
            v_prev = v.clone()

            a = net_force                      # assume unit mass
            v_next = damping * (v + dt * a)
            x_next = x + dt * v_next

            # next-step delta velocity target: [N, 2]
            node_targets = (v_next - v_prev).clone()

            # store this bin only if it has events
            if len(src_list) > 0:
                src = torch.tensor(src_list, dtype=torch.long)
                dst = torch.tensor(dst_list, dtype=torch.long)

                if self._event_dim == 0:
                    feats = torch.empty((len(src_list), 0), dtype=torch.float32)
                else:
                    feats = torch.tensor(feat_list, dtype=torch.float32)

                t = torch.full((len(src_list),), b, dtype=torch.long)

                self.src_bins.append(src)
                self.dst_bins.append(dst)
                self.t_bins.append(t)
                self.feat_bins.append(feats)
                self.node_target_bins.append(node_targets)

            # advance system
            x = x_next
            v = v_next

        self._num_bins = len(self.src_bins)
        self._num_events = sum(s.numel() for s in self.src_bins)


        if self.cfg.standardize_node_targets and len(self.node_target_bins) > 0:
            T = len(self.node_target_bins)
            f_tr, f_va, f_te = self.cfg.split_fracs
            n_tr = int(T * f_tr)

            # concatenate only training-bin targets: each is [N, 2]
            train_targets = torch.cat(
                [self.node_target_bins[i] for i in range(n_tr)],
                dim=0,   # [n_tr * N, 2]
            )

            self.dv_mean = train_targets.mean(dim=0)                     # [2]
            self.dv_std = train_targets.std(dim=0).clamp_min(1e-8)      # [2]

            # standardize every stored bin target using train stats
            for i in range(len(self.node_target_bins)):
                self.node_target_bins[i] = (self.node_target_bins[i] - self.dv_mean) / self.dv_std


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
            extra={
                "node_target_dim": 2,
                "node_target_names": ["dvx_std", "dvy_std"] if self.cfg.standardize_node_targets else ["dvx", "dvy"],
            }
        )

    def bins(self, split: str = "train") -> Iterable[EventBatch]:
        return _Stream(
            self.src_bins,
            self.dst_bins,
            self.t_bins,
            self.feat_bins,
            self.node_target_bins,
            self.split_bins[split],
            self.cfg.device,
        )


class _Stream(Iterable[EventBatch]):
    def __init__(self, src, dst, t, feat, node_targets, idxs, device):
        self.src = src
        self.dst = dst
        self.t = t
        self.feat = feat
        self.node_targets = node_targets
        self.idxs = idxs
        self.device = device

    def __iter__(self) -> Iterator[EventBatch]:
        for i in self.idxs:
            eb = EventBatch(
                src=cast(torch.LongTensor, self.src[i]),
                dst=cast(torch.LongTensor, self.dst[i]),
                t=cast(torch.LongTensor, self.t[i]),
                features=self.feat[i],
                node_targets=self.node_targets[i],
            )
            if self.device is not None:
                eb = eb.to(self.device)
            yield eb