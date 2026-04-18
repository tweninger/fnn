from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, cast

import numpy as np
import torch

from core.events import EventBatch
from datasets.interfaces import DataSpec, EventStreamDataset

"""
MD22 / MD17-style molecular trajectory dataset adapted to the repo's
EventStreamDataset interface.

----------------------
Supports three event-extraction modes:
1) distance         : keep directed pair j->i if current distance(i,j) <= distance_threshold
2) distance_change  : keep directed pair j->i if |d_t(i,j) - d_{t-1}(i,j)| >= distance_change_threshold
3) knn              : for each receiver i, keep the k nearest senders j

Design notes
------------
- each MD frame becomes one discrete time bin
- one long trajectory is split by TIME, not by random frame shuffle
- we keep the representation directed (sender -> receiver) so it drops directly
  into the same EventBatch format as JODIE / your other binned physics datasets
- for molecules, the underlying proximity relation is symmetric, but storing both
  i->j and j->i as separate directed events is a normal and convenient graph trick
"""


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

@dataclass
class MD22BinnedConfig:
    name: str = "md22_binned"

    # path to one MD17 / rMD17 / MD22 .npz file
    npz_path: str = "/home/akapociu/ift/interactiondynamics/data/MD_DATA/naphthalene.npz"

    # optional subsampling 
    frame_stride: int = 100
    max_frames: Optional[int] = None

    # split one long trajectory by TIME
    split_fracs: Tuple[float, float, float] = (0.7, 0.15, 0.15)

    # event extraction mode
    #   distance        : distance <= distance_threshold
    #   distance_change : |distance_t - distance_{t-1}| >= distance_change_threshold
    #   knn             : sender is among receiver's k nearest neighbors aka keep k nearest senders for that reciever based on curr_dist
    event_mode: str = "distance"
    # parameters used by the different event modes
    distance_threshold: float = 5
    distance_change_threshold: float = 0.1
    distance_change_use_absolute: bool = True
    knn_k: int = 4 # edges per receiver atom (so for e.g. 87x4=348 edges per bin)

    # if you want to keep bins non-empty even after filtering
    min_edges_per_bin: int = 1

    # optional observation noise added AFTER loading the trajectory
    observation_noise_pos: float = 0.02
    observation_noise_force: float = 0.1

    # reproducibility for observation noise
    seed: int = 0

    # optional torch device for EventBatch tensors
    device: Optional[torch.device] = None


MD22_DISTANCE_OPTIONS: Tuple[float, float, float] = (2.0, 2.5, 3.0)
MD22_DISTANCE_CHANGE_OPTIONS: Tuple[float, float, float] = (0.01, 0.05, 0.10)
MD22_KNN_OPTIONS: Tuple[int, int, int] = (2, 4, 8)


# -----------------------------------------------------------------------------
# FEATURES
# -----------------------------------------------------------------------------

_EDGE_FEATURE_NAMES: Sequence[str] = (
    # atom identities
    # "recv_z",
    # "send_z",

    # receiver local geometry / force
    # "recv_px",
    # "recv_py",
    # "recv_pz",
    # "recv_fx",
    # "recv_fy",
    # "recv_fz",

    # sender local geometry / force
    # "send_px",
    # "send_py",
    # "send_pz",
    # "send_fx",
    # "send_fy",
    # "send_fz",

    # pairwise relation
    #  "rel_px",
    #  "rel_py",
    #  "rel_pz",
    # "distance",
    # "distance_delta",

    # force relation
    # "rel_fx",
    # "rel_fy",
    # "rel_fz",
    # "recv_force_mag",
    # "send_force_mag",
    # "rel_force_mag",

    # one global scalar label/summary for the frame
    # "energy",
)


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def _load_md_npz(npz_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    data = np.load(npz_path)

    if "R" not in data:
        raise KeyError(f"{npz_path} is missing 'R' positions")
    if "F" not in data:
        raise KeyError(f"{npz_path} is missing 'F' forces")
    if "E" not in data:
        raise KeyError(f"{npz_path} is missing 'E' energies")

    z_key = "z" if "z" in data else "Z" if "Z" in data else None
    if z_key is None:
        raise KeyError(f"{npz_path} is missing atom types ('z' or 'Z')")

    R = np.asarray(data["R"], dtype=np.float64)
    F = np.asarray(data["F"], dtype=np.float64)
    E = np.asarray(data["E"], dtype=np.float64)
    z = np.asarray(data[z_key], dtype=np.int64)

    E = np.asarray(data["E"], dtype=np.float64)

    if E.ndim == 2 and E.shape[1] == 1:
        E = E[:, 0]
    elif E.ndim == 2 and E.shape[0] == 1:
        E = E[0]

    if E.ndim != 1 or E.shape[0] != R.shape[0]:
        raise ValueError(f"Expected E shape [T], got {E.shape} for T={R.shape[0]}")
    if R.ndim != 3 or R.shape[-1] != 3:
        raise ValueError(f"Expected R to have shape [T, N, 3], got {R.shape}")
    if F.shape != R.shape:
        raise ValueError(f"Expected F to match R shape; got R={R.shape}, F={F.shape}")
    if E.ndim != 1 or E.shape[0] != R.shape[0]:
        raise ValueError(f"Expected E shape [T], got {E.shape} for T={R.shape[0]}")
    if z.ndim != 1 or z.shape[0] != R.shape[1]:
        raise ValueError(f"Expected z shape [N], got {z.shape} for N={R.shape[1]}")

    meta = {}
    for k in ("r_unit", "e_unit", "perms", "md5"):
        if k in data:
            meta[k] = data[k]

    return R, z, F, E, meta


def _pairwise_distances(coords: np.ndarray) -> np.ndarray:
    # coords: [N, 3] -> distances: [N, N]
    return np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)


def _pair_record(
    *,
    coords: np.ndarray,
    forces: np.ndarray,
    z: np.ndarray,
    energy: float,
    sender: int,
    receiver: int,
    curr_dist: float,
    prev_dist: float,
) -> dict:
    recv_pos = coords[receiver]
    send_pos = coords[sender]
    recv_force = forces[receiver]
    send_force = forces[sender]

    rel_pos = send_pos - recv_pos
    rel_force = send_force - recv_force

    recv_force_mag = float(np.linalg.norm(recv_force))
    send_force_mag = float(np.linalg.norm(send_force))
    rel_force_mag = float(np.linalg.norm(rel_force))

    distance_delta = float(curr_dist - prev_dist)

    feats = np.array(
        [
            # float(z[receiver]),
            # float(z[sender]),
            # recv_pos[0],
            # recv_pos[1],
            # recv_pos[2],
            # recv_force[0],
            # recv_force[1],
            # recv_force[2],
            # send_pos[0],
            # send_pos[1],
            # send_pos[2],
            # send_force[0],
            # send_force[1],
            # send_force[2],
            # rel_pos[0],
            # rel_pos[1],
            # rel_pos[2],
            # curr_dist,
            # distance_delta,
            # rel_force[0],
            # rel_force[1],
            # rel_force[2],
            # recv_force_mag,
            # send_force_mag,
            # rel_force_mag,
            # float(energy),
        ],
        dtype=np.float32,
    )

    return {
        "src": sender,
        "dst": receiver,
        "distance": float(curr_dist),
        "distance_delta": distance_delta,
        "distance_delta_abs": float(abs(distance_delta)),
        "features": feats,
    }


def _all_pair_records(
    *,
    coords: np.ndarray,
    forces: np.ndarray,
    z: np.ndarray,
    energy: float,
    curr_dists: np.ndarray,
    prev_dists: np.ndarray,
) -> List[dict]:
    records: List[dict] = []
    num_atoms = coords.shape[0]
    for receiver in range(num_atoms):
        for sender in range(num_atoms):
            if sender == receiver:
                continue
            records.append(
                _pair_record(
                    coords=coords,
                    forces=forces,
                    z=z,
                    energy=energy,
                    sender=sender,
                    receiver=receiver,
                    curr_dist=float(curr_dists[receiver, sender]),
                    prev_dist=float(prev_dists[receiver, sender]),
                )
            )
    return records


def _select_records(records: List[dict], curr_dists: np.ndarray, cfg: MD22BinnedConfig) -> List[dict]:
    mode = str(cfg.event_mode)

    if mode == "distance":
        chosen = [r for r in records if r["distance"] <= float(cfg.distance_threshold)]
        if len(chosen) < max(0, int(cfg.min_edges_per_bin)):
            order = np.argsort([r["distance"] for r in records])
            need = min(len(records), max(1, int(cfg.min_edges_per_bin)))
            chosen = [records[int(i)] for i in order[:need]]
        return chosen

    if mode == "distance_change":
        if cfg.distance_change_use_absolute:
            scores = np.array([r["distance_delta_abs"] for r in records], dtype=np.float64)
        else:
            scores = np.array([r["distance_delta"] for r in records], dtype=np.float64)

        chosen = [r for r, s in zip(records, scores) if s >= float(cfg.distance_change_threshold)]
        if len(chosen) < max(0, int(cfg.min_edges_per_bin)):
            order = np.argsort(-scores)
            need = min(len(records), max(1, int(cfg.min_edges_per_bin)))
            chosen = [records[int(i)] for i in order[:need]]
        return chosen

    if mode == "knn":
        num_atoms = curr_dists.shape[0]
        k = max(1, min(int(cfg.knn_k), num_atoms - 1))
        by_pair = {(int(r["src"]), int(r["dst"])): r for r in records}

        chosen: List[dict] = []
        for receiver in range(num_atoms):
            order = np.argsort(curr_dists[receiver])
            kept = 0
            for sender in order:
                sender = int(sender)
                if sender == receiver:
                    continue
                chosen.append(by_pair[(sender, receiver)])
                kept += 1
                if kept >= k:
                    break

        if len(chosen) < max(0, int(cfg.min_edges_per_bin)):
            order = np.argsort([r["distance"] for r in records])
            need = min(len(records), max(1, int(cfg.min_edges_per_bin)))
            chosen = [records[int(i)] for i in order[:need]]
        return chosen

    raise ValueError(f"unknown event_mode={cfg.event_mode}")


def _frame_to_event_batch(
    *,
    coords: np.ndarray,
    forces: np.ndarray,
    z: np.ndarray,
    energy: float,
    t_idx: int,
    curr_dists: np.ndarray,
    prev_dists: np.ndarray,
    cfg: MD22BinnedConfig,
) -> EventBatch:
    records = _all_pair_records(
        coords=coords,
        forces=forces,
        z=z,
        energy=energy,
        curr_dists=curr_dists,
        prev_dists=prev_dists,
    )
    chosen = _select_records(records, curr_dists, cfg)

    src = torch.tensor([r["src"] for r in chosen], dtype=torch.long, device=cfg.device)
    dst = torch.tensor([r["dst"] for r in chosen], dtype=torch.long, device=cfg.device)
    feats = torch.tensor(
        np.stack([r["features"] for r in chosen], axis=0),
        dtype=torch.float32,
        device=cfg.device,
    )
    t = torch.full((src.numel(),), int(t_idx), dtype=torch.long, device=cfg.device)

    # node regression targets: [px, py, pz]
    node_targets = torch.tensor(
        coords,
        dtype=torch.float32,
        device=cfg.device,
    )  # [N, 3]

    node_mask = torch.ones(
        (coords.shape[0],),
        dtype=torch.bool,
        device=cfg.device,
    )

    eb = EventBatch(
        src=cast(torch.LongTensor, src),
        dst=cast(torch.LongTensor, dst),
        features=feats,
        t=cast(torch.LongTensor, t),
        node_targets=node_targets,
        node_mask=node_mask,
    )
    if cfg.device is not None:
        eb = eb.to(cfg.device)
    return eb


# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------

class MD22BinnedDataset(EventStreamDataset):
    def __init__(self, cfg: MD22BinnedConfig):
        if not cfg.npz_path:
            raise ValueError("cfg.npz_path must point to an MD17/rMD17/MD22-style .npz file")

        self.cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)
        self._event_dim = len(_EDGE_FEATURE_NAMES)

        R, z, F, E, meta = _load_md_npz(cfg.npz_path)

        stride = max(1, int(cfg.frame_stride))
        R = R[::stride].copy()
        F = F[::stride].copy()
        E = E[::stride].copy()

        if cfg.max_frames is not None:
            keep = max(2, int(cfg.max_frames))
            R = R[:keep].copy()
            F = F[:keep].copy()
            E = E[:keep].copy()

        if cfg.observation_noise_pos > 0:
            R += self._rng.normal(loc=0.0, scale=cfg.observation_noise_pos, size=R.shape)
        if cfg.observation_noise_force > 0:
            F += self._rng.normal(loc=0.0, scale=cfg.observation_noise_force, size=F.shape)

        self._R = R
        self._z = z
        self._F = F
        self._E = E
        self._num_nodes = int(z.shape[0])
        self._num_bins = int(R.shape[0])
        self._meta = meta
        self._dataset_name = Path(cfg.npz_path).stem

        self._all_bins: List[EventBatch] = []
        prev_dists: Optional[np.ndarray] = None

        for t_idx in range(self._num_bins):
            coords = R[t_idx]
            forces = F[t_idx]
            energy = float(E[t_idx])
            curr_dists = _pairwise_distances(coords)
            if prev_dists is None:
                prev_dists = curr_dists

            self._all_bins.append(
                _frame_to_event_batch(
                    coords=coords,
                    forces=forces,
                    z=z,
                    energy=energy,
                    t_idx=t_idx,
                    curr_dists=curr_dists,
                    prev_dists=prev_dists,
                    cfg=cfg,
                )
            )
            prev_dists = curr_dists

        f_tr, f_va, f_te = cfg.split_fracs
        assert abs((f_tr + f_va + f_te) - 1.0) < 1e-6, "split_fracs must sum to 1.0"
        tr_end = int(self._num_bins * f_tr)
        va_end = tr_end + int(self._num_bins * f_va)

        # remove 1st bin if distance_change, as there's no prev frame to compare with
        start_idx = 1 if cfg.event_mode == "distance_change" else 0

        self._split_ranges = {
            "train": (start_idx, max(0, tr_end - 1)),
            "val": (tr_end, max(tr_end, va_end - 1)),
            "test": (va_end, self._num_bins - 1),
        }

    def spec(self) -> DataSpec:
        extra = {
            "source_file": self.cfg.npz_path,
            "molecule": self._dataset_name,
            "num_atoms": self._num_nodes,
            "split_fracs": self.cfg.split_fracs,
            "event_mode": str(self.cfg.event_mode),
            "distance_threshold": float(self.cfg.distance_threshold),
            "distance_change_threshold": float(self.cfg.distance_change_threshold),
            "distance_change_use_absolute": bool(self.cfg.distance_change_use_absolute),
            "knn_k": int(self.cfg.knn_k),
            "frame_stride": int(self.cfg.frame_stride),
            "max_frames": None if self.cfg.max_frames is None else int(self.cfg.max_frames),
            "feature_names": list(_EDGE_FEATURE_NAMES),
            "observation_noise_pos": float(self.cfg.observation_noise_pos),
            "observation_noise_force": float(self.cfg.observation_noise_force),
            "node_target_dim": 3,
            "node_target_names": ["px", "py", "pz"],
        }
        if "r_unit" in self._meta:
            extra["r_unit"] = str(self._meta["r_unit"])
        if "e_unit" in self._meta:
            extra["e_unit"] = str(self._meta["e_unit"])
        if "perms" in self._meta:
            extra["num_symmetry_perms"] = int(np.asarray(self._meta["perms"]).shape[0])

        return DataSpec(
            name=self.cfg.name,
            num_nodes=self._num_nodes,
            event_dim=self._event_dim,
            num_events=int(sum(int(b.src.numel()) for b in self._all_bins)),
            num_bins=int(len(self._all_bins)),
            extra=extra,
        )

    def bins(self, split: str = "train") -> Iterable[EventBatch]:
        assert split in self._split_ranges, f"unknown split={split}"
        b0, b1 = self._split_ranges[split]
        return _PrecomputedStream(self._all_bins, b0=b0, b1=b1)


@dataclass
class _PrecomputedStream(Iterable[EventBatch]):
    bins_all: Sequence[EventBatch]
    b0: int
    b1: int

    def __iter__(self) -> Iterator[EventBatch]:
        for b in range(self.b0, self.b1 + 1):
            yield self.bins_all[b]


# -----------------------------------------------------------------------------
# VARIANT HELPERS
# -----------------------------------------------------------------------------

def make_md22_distance_datasets(
    base_cfg: MD22BinnedConfig,
    thresholds: Sequence[float] = MD22_DISTANCE_OPTIONS,
) -> dict[str, MD22BinnedDataset]:
    out: dict[str, MD22BinnedDataset] = {}
    stem = Path(base_cfg.npz_path).stem or "md22"
    for thr in thresholds:
        tag = str(thr).replace(".", "p")
        cfg = replace(
            base_cfg,
            name=f"md22_{stem}_distance_thr{tag}",
            event_mode="distance",
            distance_threshold=float(thr),
        )
        out[cfg.name] = MD22BinnedDataset(cfg)
    return out


def make_md22_distance_change_datasets(
    base_cfg: MD22BinnedConfig,
    thresholds: Sequence[float] = MD22_DISTANCE_CHANGE_OPTIONS,
) -> dict[str, MD22BinnedDataset]:
    out: dict[str, MD22BinnedDataset] = {}
    stem = Path(base_cfg.npz_path).stem or "md22"
    for thr in thresholds:
        tag = str(thr).replace(".", "p")
        cfg = replace(
            base_cfg,
            name=f"md22_{stem}_distancechange_thr{tag}",
            event_mode="distance_change",
            distance_change_threshold=float(thr),
        )
        out[cfg.name] = MD22BinnedDataset(cfg)
    return out


def make_md22_knn_datasets(
    base_cfg: MD22BinnedConfig,
    ks: Sequence[int] = MD22_KNN_OPTIONS,
) -> dict[str, MD22BinnedDataset]:
    out: dict[str, MD22BinnedDataset] = {}
    stem = Path(base_cfg.npz_path).stem or "md22"
    for k in ks:
        cfg = replace(
            base_cfg,
            name=f"md22_{stem}_knn_k{int(k)}",
            event_mode="knn",
            knn_k=int(k),
        )
        out[cfg.name] = MD22BinnedDataset(cfg)
    return out


def make_all_md22_mode_datasets(base_cfg: MD22BinnedConfig) -> dict[str, MD22BinnedDataset]:
    out: dict[str, MD22BinnedDataset] = {}
    out.update(make_md22_distance_datasets(base_cfg))
    out.update(make_md22_distance_change_datasets(base_cfg))
    out.update(make_md22_knn_datasets(base_cfg))
    return out


# -----------------------------------------------------------------------------
# SMOKE TEST / EXAMPLE
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_cfg = MD22BinnedConfig(
        npz_path="/home/akapociu/ift/interactiondynamics/data/stachyose.npz",
        frame_stride=10,
        max_frames=512,
        observation_noise_pos=0.0,
        observation_noise_force=0.0,
        min_edges_per_bin=1,
        device=device,
    )

    datasets = make_all_md22_mode_datasets(base_cfg)

    for name, ds in datasets.items():
        spec = ds.spec()
        print("\n", name)
        print(spec)
        first_batch = next(iter(ds.bins("train")))
        print("first batch events:", int(first_batch.src.numel()))
        if first_batch.features is not None:
            print("first batch feature shape:", tuple(first_batch.features.shape))
