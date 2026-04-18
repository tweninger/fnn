
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, cast

import math
import torch

from core.events import EventBatch
from datasets import DataSpec, EventStreamDataset

"""
2D spring-web benchmark adapted to the repo's EventStreamDataset interface.

Design split
------------
There are two separate layers in this dataset:

1) Physical spring graph (topology)
   - ring       : only the cycle backbone
   - knn        : each node connected to its k nearest neighbors on the ideal ring
   - radius     : each node connected to all nodes within a distance radius on the ideal ring
   - all_pairs  : fully connected spring web

   This determines the *true latent physics*: which springs actually exert forces.

2) Event emission rule
   - all_neighbors : emit every physical spring interaction every bin
   - thresholded   : emit only spring interactions whose salience exceeds a cutoff

   This determines what the model gets to observe as events.
   It does NOT change the underlying physics.

Node targets
------------
This file supports several node-level regression targets:

- delta_v   : v[t+h] - v[t]
- dv        : immediate one-step local delta-v from the current bin under the discrete update
- accel     : instantaneous acceleration a[t]
- delta_x   : x[t+h] - x[t]
- v_future  : v[t+h]
- x_future  : x[t+h]

where h = target_horizon for the future-based targets.
"""

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

@dataclass
class SpringWeb2DConfig:
    name: str = "spring_web_2d"

    # size / time
    num_nodes: int = 32
    num_bins: int = 256

    # simulation parameters
    dt: float = 0.05
    spring_k: float = 1.0
    damping: float = 0.995

    # geometry
    ring_radius: float = 5.0

    # topology = true physical spring graph
    topology: str = "knn"                 # {"ring", "knn", "radius", "all_pairs"}
    topology_k: int = 4                   # used for knn
    topology_radius: Optional[float] = None   # used for radius
    include_ring_edges: bool = True       # keep cycle backbone even when adding extra web edges

    # random perturbations
    init_pos_noise: float = 0.05
    init_vel_noise: float = 0.05

    # event emission
    event_mode: str = "thresholded"       # {"all_neighbors", "thresholded"}
    interaction_threshold: float = 0.015
    threshold_metric: str = "force_mag"   # {"force_mag", "extension", "distance", "rel_speed"}
    threshold_use_absolute: bool = True
    threshold_keep_one_if_empty: bool = True

    bidirectional: bool = True

    # node target configuration
    target_type: str = "delta_v"          # {"delta_v","dv","accel","delta_x","v_future","x_future"}
    target_horizon: int = 1
    standardize_node_targets: bool = True

    # split by time
    split_fracs: Tuple[float, float, float] = (0.7, 0.15, 0.15)

    # rng / device
    seed: int = 0
    device: Optional[torch.device] = None


# Keep whichever features you want uncommented.
_EDGE_FEATURE_NAMES: Sequence[str] = (
    # "dx",
    # "dy",
    # "dvx",
    # "dvy",
    # "dist",
    #"extension",
    #  "fx",
    #  "fy",
)


# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------

class SpringWeb2DDataset(EventStreamDataset):
    def __init__(self, cfg: SpringWeb2DConfig):
        self.cfg = cfg

        self.target_mean: Optional[torch.Tensor] = None
        self.target_std: Optional[torch.Tensor] = None

        self._event_dim = len(_EDGE_FEATURE_NAMES)

        self._build()
        self._split()

    # -------------------------------------------------------------------------
    # helpers: geometry / topology
    # -------------------------------------------------------------------------

    def _ideal_ring_positions(self, n: int, radius: float) -> torch.Tensor:
        angles = torch.linspace(0.0, 2.0 * math.pi, steps=n + 1, dtype=torch.float32)[:-1]
        return torch.stack(
            [
                radius * torch.cos(angles),
                radius * torch.sin(angles),
            ],
            dim=1,
        )  # [N,2]

    def _build_spring_pairs(self, x0: torch.Tensor) -> List[Tuple[int, int, float]]:
        """
        Build undirected spring pairs (i, j, rest_length) from the ideal ring geometry.

        Important:
        - topology determines the TRUE physical spring graph
        - rest lengths are computed from the ideal geometry before adding noise
        """
        n = x0.size(0)
        pair_set: set[Tuple[int, int]] = set()

        if self.cfg.include_ring_edges or self.cfg.topology == "ring":
            for i in range(n):
                j = (i + 1) % n
                pair_set.add((min(i, j), max(i, j)))

        topo = self.cfg.topology

        if topo == "ring":
            pass

        elif topo == "knn":
            dist_mat = torch.cdist(x0, x0)  # [N,N]
            k = max(1, min(int(self.cfg.topology_k), n - 1))
            for i in range(n):
                nbrs = torch.argsort(dist_mat[i])[1 : k + 1]
                for j in nbrs.tolist():
                    pair_set.add((min(i, j), max(i, j)))

        elif topo == "radius":
            if self.cfg.topology_radius is None:
                raise ValueError("topology='radius' requires cfg.topology_radius")
            r = float(self.cfg.topology_radius)
            dist_mat = torch.cdist(x0, x0)
            for i in range(n):
                for j in range(i + 1, n):
                    if float(dist_mat[i, j].item()) <= r:
                        pair_set.add((i, j))

        elif topo == "all_pairs":
            for i in range(n):
                for j in range(i + 1, n):
                    pair_set.add((i, j))

        else:
            raise ValueError(
                f"unknown topology={topo!r}; expected one of "
                "{'ring','knn','radius','all_pairs'}"
            )

        out: List[Tuple[int, int, float]] = []
        for i, j in sorted(pair_set):
            rest = float(torch.norm(x0[j] - x0[i]).item())
            out.append((i, j, rest))
        return out

    # -------------------------------------------------------------------------
    # helpers: event features / thresholding
    # -------------------------------------------------------------------------

    def _record_to_features(self, rec: Dict[str, torch.Tensor], reverse: bool = False) -> List[float]:
        dpos = -rec["dpos"] if reverse else rec["dpos"]
        dvel = -rec["dvel"] if reverse else rec["dvel"]
        force_vec = -rec["force_vec"] if reverse else rec["force_vec"]

        feature_map = {
            "dx": float(dpos[0].item()),
            "dy": float(dpos[1].item()),
            "dvx": float(dvel[0].item()),
            "dvy": float(dvel[1].item()),
            "dist": float(rec["dist"].item()),
            "extension": float(rec["extension"].item()),
            "fx": float(force_vec[0].item()),
            "fy": float(force_vec[1].item()),
        }
        return [feature_map[name] for name in _EDGE_FEATURE_NAMES]

    def _interaction_strength(self, rec: Dict[str, torch.Tensor]) -> float:
        metric = self.cfg.threshold_metric

        if metric == "force_mag":
            val = float(rec["force_mag"].item())
        elif metric == "extension":
            val = float(rec["extension"].item())
        elif metric == "distance":
            val = float(rec["dist"].item())
        elif metric == "rel_speed":
            val = float(rec["rel_speed"].item())
        else:
            raise ValueError(
                f"unknown threshold_metric={metric!r}; expected one of "
                "{'force_mag','extension','distance','rel_speed'}"
            )

        return abs(val) if self.cfg.threshold_use_absolute else val

    def _filter_records(self, records_all: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        if self.cfg.event_mode == "all_neighbors":
            return records_all

        if self.cfg.event_mode != "thresholded":
            raise ValueError(
                f"unknown event_mode={self.cfg.event_mode!r}; expected 'all_neighbors' or 'thresholded'"
            )

        thr = float(self.cfg.interaction_threshold)
        kept = [r for r in records_all if self._interaction_strength(r) >= thr]

        if (not kept) and self.cfg.threshold_keep_one_if_empty and records_all:
            kept = [max(records_all, key=self._interaction_strength)]

        return kept

    # -------------------------------------------------------------------------
    # helpers: node targets
    # -------------------------------------------------------------------------

    def _target_names(self, target_type: str, standardized: bool) -> List[str]:
        if target_type in ("delta_v", "dv", "accel", "v_future"):
            base = ["vx", "vy"]
        elif target_type in ("delta_x", "x_future"):
            base = ["x", "y"]
        else:
            raise ValueError(f"unknown target_type={target_type!r}")

        prefix = {
            "delta_v": "delta_",
            "dv": "dv_",
            "accel": "a_",
            "delta_x": "delta_",
            "v_future": "future_",
            "x_future": "future_",
        }[target_type]

        names = [f"{prefix}{b}" for b in base]
        if standardized:
            names = [f"{n}_std" for n in names]
        return names

    def _compute_target(
        self,
        b: int,
        x_hist: Sequence[torch.Tensor],
        v_hist: Sequence[torch.Tensor],
        a_hist: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        h = int(self.cfg.target_horizon)
        target_type = self.cfg.target_type

        if h < 1:
            raise ValueError("target_horizon must be >= 1")

        if target_type == "delta_v":
            return (v_hist[b + h] - v_hist[b]).clone()

        if target_type == "dv":
            # immediate local one-step delta-v under the current discrete update
            v_one = self.cfg.damping * (v_hist[b] + self.cfg.dt * a_hist[b])
            return (v_one - v_hist[b]).clone()

        if target_type == "accel":
            return a_hist[b].clone()

        if target_type == "delta_x":
            return (x_hist[b + h] - x_hist[b]).clone()

        if target_type == "v_future":
            return v_hist[b + h].clone()

        if target_type == "x_future":
            return x_hist[b + h].clone()

        raise ValueError(
            f"unknown target_type={target_type!r}; expected one of "
            "{'delta_v','dv','accel','delta_x','v_future','x_future'}"
        )

    # -------------------------------------------------------------------------
    # build
    # -------------------------------------------------------------------------

    def _build(self):
        g = torch.Generator().manual_seed(self.cfg.seed)

        n = self.cfg.num_nodes
        T = self.cfg.num_bins
        dt = self.cfg.dt
        k = self.cfg.spring_k
        damping = self.cfg.damping
        radius = self.cfg.ring_radius
        bidir = self.cfg.bidirectional
        h = int(self.cfg.target_horizon)

        if h < 1:
            raise ValueError("target_horizon must be >= 1")

        # ideal geometry defines topology + spring rest lengths
        x0 = self._ideal_ring_positions(n, radius)
        spring_pairs = self._build_spring_pairs(x0)

        # noisy actual initial state
        x = x0 + self.cfg.init_pos_noise * torch.randn(n, 2, generator=g)
        v = self.cfg.init_vel_noise * torch.randn(n, 2, generator=g)

        # full latent history so we can build future-based targets cleanly
        x_hist: List[torch.Tensor] = [x.clone()]
        v_hist: List[torch.Tensor] = [v.clone()]
        a_hist: List[torch.Tensor] = []

        # temporary storage keyed by original bin index
        raw_src_bins: List[torch.Tensor] = []
        raw_dst_bins: List[torch.Tensor] = []
        raw_t_bins: List[torch.Tensor] = []
        raw_feat_bins: List[torch.Tensor] = []
        raw_has_events: List[bool] = []

        for b in range(T):
            net_force = torch.zeros(n, 2, dtype=torch.float32)
            records_all: List[Dict[str, torch.Tensor]] = []

            for i, j, rest in spring_pairs:
                dpos = x[j] - x[i]      # [2]
                dvel = v[j] - v[i]      # [2]

                dist = torch.norm(dpos).clamp_min(1e-8)
                direction = dpos / dist
                extension = dist - rest

                # Hooke's law in vector form
                force_vec = k * extension * direction

                net_force[i] += force_vec
                net_force[j] -= force_vec

                records_all.append(
                    {
                        "i": torch.tensor(i, dtype=torch.long),
                        "j": torch.tensor(j, dtype=torch.long),
                        "dpos": dpos,
                        "dvel": dvel,
                        "dist": dist,
                        "extension": extension,
                        "force_vec": force_vec,
                        "force_mag": torch.norm(force_vec),
                        "rel_speed": torch.norm(dvel),
                    }
                )

            records = self._filter_records(records_all)

            src_list: List[int] = []
            dst_list: List[int] = []
            feat_list: List[List[float]] = []

            for rec in records:
                i = int(rec["i"].item())
                j = int(rec["j"].item())

                src_list.append(i)
                dst_list.append(j)
                feat_list.append(self._record_to_features(rec, reverse=False))

                if bidir:
                    src_list.append(j)
                    dst_list.append(i)
                    feat_list.append(self._record_to_features(rec, reverse=True))

            if len(src_list) > 0:
                src = torch.tensor(src_list, dtype=torch.long)
                dst = torch.tensor(dst_list, dtype=torch.long)
                if self._event_dim == 0:
                    feats = torch.empty((len(src_list), 0), dtype=torch.float32)
                else:
                    feats = torch.tensor(feat_list, dtype=torch.float32)
                t = torch.full((len(src_list),), b, dtype=torch.long)

                raw_src_bins.append(src)
                raw_dst_bins.append(dst)
                raw_t_bins.append(t)
                raw_feat_bins.append(feats)
                raw_has_events.append(True)
            else:
                raw_src_bins.append(torch.empty((0,), dtype=torch.long))
                raw_dst_bins.append(torch.empty((0,), dtype=torch.long))
                raw_t_bins.append(torch.empty((0,), dtype=torch.long))
                raw_feat_bins.append(torch.empty((0, self._event_dim), dtype=torch.float32))
                raw_has_events.append(False)

            # advance dynamics
            a = net_force                      # unit mass
            v_next = damping * (v + dt * a)
            x_next = x + dt * v_next

            a_hist.append(a.clone())
            x = x_next
            v = v_next
            x_hist.append(x.clone())
            v_hist.append(v.clone())

        # final filtered storage with targets attached
        self.src_bins: List[torch.Tensor] = []
        self.dst_bins: List[torch.Tensor] = []
        self.t_bins: List[torch.Tensor] = []
        self.feat_bins: List[torch.Tensor] = []
        self.node_target_bins: List[torch.Tensor] = []

        for b in range(T):
            if not raw_has_events[b]:
                continue
            if b + h > T:
                continue

            node_targets = self._compute_target(b, x_hist=x_hist, v_hist=v_hist, a_hist=a_hist)

            self.src_bins.append(raw_src_bins[b])
            self.dst_bins.append(raw_dst_bins[b])
            self.t_bins.append(raw_t_bins[b])
            self.feat_bins.append(raw_feat_bins[b])
            self.node_target_bins.append(node_targets)

        self._num_bins = len(self.src_bins)
        self._num_events = sum(int(s.numel()) for s in self.src_bins)

        if self.cfg.standardize_node_targets and len(self.node_target_bins) > 0:
            T_stored = len(self.node_target_bins)
            f_tr, _, _ = self.cfg.split_fracs
            n_tr = max(1, int(T_stored * f_tr))

            train_targets = torch.cat(
                [self.node_target_bins[i] for i in range(n_tr)],
                dim=0,   # [n_tr * N, d_target]
            )

            self.target_mean = train_targets.mean(dim=0)
            self.target_std = train_targets.std(dim=0).clamp_min(1e-8)

            for i in range(len(self.node_target_bins)):
                self.node_target_bins[i] = (self.node_target_bins[i] - self.target_mean) / self.target_std

    def _split(self):
        T = self._num_bins
        f_tr, f_va, _ = self.cfg.split_fracs

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
                "feature_names": list(_EDGE_FEATURE_NAMES),
                "physics": "2D spring-mass web on noisy ring geometry",
                "topology": self.cfg.topology,
                "topology_k": int(self.cfg.topology_k),
                "topology_radius": self.cfg.topology_radius,
                "include_ring_edges": bool(self.cfg.include_ring_edges),
                "event_mode": self.cfg.event_mode,
                "interaction_threshold": float(self.cfg.interaction_threshold),
                "threshold_metric": self.cfg.threshold_metric,
                "threshold_use_absolute": bool(self.cfg.threshold_use_absolute),
                "target_type": self.cfg.target_type,
                "target_horizon": int(self.cfg.target_horizon),
                "standardize_node_targets": bool(self.cfg.standardize_node_targets),
                "node_target_dim": 2,
                "node_target_names": self._target_names(
                    target_type=self.cfg.target_type,
                    standardized=self.cfg.standardize_node_targets,
                ),
            },
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


# -----------------------------------------------------------------------------
# VARIANT HELPERS
# -----------------------------------------------------------------------------

SPRING_WEB_K_OPTIONS: Tuple[int, ...] = (2, 4, 6)
SPRING_WEB_RADIUS_OPTIONS: Tuple[float, ...] = (2.0, 3.0, 4.0)
SPRING_WEB_THRESHOLD_VALUES: Tuple[float, ...] = (0.01, 0.05, 0.10)
SPRING_WEB_THRESHOLD_METRICS: Tuple[str, ...] = ("force_mag", "extension", "distance", "rel_speed")
SPRING_WEB_TARGET_TYPES: Tuple[str, ...] = ("delta_v", "dv", "accel", "delta_x", "v_future", "x_future")
SPRING_WEB_TARGET_HORIZONS: Tuple[int, ...] = (1, 2, 4)


def _tag(value: object) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def _topology_variants(
    base_cfg: SpringWeb2DConfig,
    *,
    topologies: Sequence[str],
    knn_values: Sequence[int],
    radius_values: Sequence[float],
    include_ring_edges_options: Sequence[bool],
) -> List[SpringWeb2DConfig]:
    out: List[SpringWeb2DConfig] = []

    for topo in topologies:
        if topo == "ring":
            out.append(
                replace(
                    base_cfg,
                    name=f"{base_cfg.name}_topo-ring",
                    topology="ring",
                    include_ring_edges=True,
                )
            )

        elif topo == "knn":
            for k in knn_values:
                for include_ring in include_ring_edges_options:
                    out.append(
                        replace(
                            base_cfg,
                            name=f"{base_cfg.name}_topo-knn{k}_ring{int(include_ring)}",
                            topology="knn",
                            topology_k=int(k),
                            include_ring_edges=bool(include_ring),
                        )
                    )

        elif topo == "radius":
            for r in radius_values:
                for include_ring in include_ring_edges_options:
                    out.append(
                        replace(
                            base_cfg,
                            name=f"{base_cfg.name}_topo-radius{_tag(r)}_ring{int(include_ring)}",
                            topology="radius",
                            topology_radius=float(r),
                            include_ring_edges=bool(include_ring),
                        )
                    )

        elif topo == "all_pairs":
            for include_ring in include_ring_edges_options:
                out.append(
                    replace(
                        base_cfg,
                        name=f"{base_cfg.name}_topo-allpairs_ring{int(include_ring)}",
                        topology="all_pairs",
                        include_ring_edges=bool(include_ring),
                    )
                )

        else:
            raise ValueError(
                f"unknown topology={topo!r}; expected one of "
                "{'ring','knn','radius','all_pairs'}"
            )

    return out


def make_spring_web_variants(
    base_cfg: SpringWeb2DConfig,
    *,
    topologies: Sequence[str] = ("ring", "knn", "radius", "all_pairs"),
    knn_values: Sequence[int] = SPRING_WEB_K_OPTIONS,
    radius_values: Sequence[float] = SPRING_WEB_RADIUS_OPTIONS,
    include_ring_edges_options: Sequence[bool] = (True,),

    event_modes: Sequence[str] = ("all_neighbors", "thresholded"),
    threshold_metrics: Sequence[str] = SPRING_WEB_THRESHOLD_METRICS,
    threshold_use_absolute_options: Sequence[bool] = (True,),
    threshold_values: Sequence[float] = SPRING_WEB_THRESHOLD_VALUES,

    target_types: Sequence[str] = ("delta_v",),
    target_horizons: Sequence[int] = (1,),
    standardize_node_targets_options: Sequence[bool] = (True,),
) -> Dict[str, SpringWeb2DDataset]:
    """
    Build a sweep of datasets over:
      - physical spring topology
      - event emission mode / threshold metric / threshold value
      - node target type / target horizon / target standardization
    """
    topo_cfgs = _topology_variants(
        base_cfg,
        topologies=topologies,
        knn_values=knn_values,
        radius_values=radius_values,
        include_ring_edges_options=include_ring_edges_options,
    )

    out: Dict[str, SpringWeb2DDataset] = {}

    for topo_cfg in topo_cfgs:
        for target_type in target_types:
            for horizon in target_horizons:
                for std_targets in standardize_node_targets_options:
                    for event_mode in event_modes:
                        if event_mode == "all_neighbors":
                            cfg = replace(
                                topo_cfg,
                                name=(
                                    f"{topo_cfg.name}"
                                    f"_events-all"
                                    f"_target-{target_type}"
                                    f"_h{int(horizon)}"
                                    f"_ystd{int(std_targets)}"
                                ),
                                event_mode="all_neighbors",
                                target_type=str(target_type),
                                target_horizon=int(horizon),
                                standardize_node_targets=bool(std_targets),
                            )
                            out[cfg.name] = SpringWeb2DDataset(cfg)

                        elif event_mode == "thresholded":
                            for metric in threshold_metrics:
                                for use_abs in threshold_use_absolute_options:
                                    for thr in threshold_values:
                                        cfg = replace(
                                            topo_cfg,
                                            name=(
                                                f"{topo_cfg.name}"
                                                f"_events-thr"
                                                f"_metric-{metric}"
                                                f"_abs{int(use_abs)}"
                                                f"_thr-{_tag(thr)}"
                                                f"_target-{target_type}"
                                                f"_h{int(horizon)}"
                                                f"_ystd{int(std_targets)}"
                                            ),
                                            event_mode="thresholded",
                                            threshold_metric=str(metric),
                                            threshold_use_absolute=bool(use_abs),
                                            interaction_threshold=float(thr),
                                            target_type=str(target_type),
                                            target_horizon=int(horizon),
                                            standardize_node_targets=bool(std_targets),
                                        )
                                        out[cfg.name] = SpringWeb2DDataset(cfg)
                        else:
                            raise ValueError(
                                f"unknown event_mode={event_mode!r}; expected one of "
                                "{'all_neighbors','thresholded'}"
                            )

    return out
