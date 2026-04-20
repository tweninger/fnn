from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, cast

import numpy as np
import torch

from core.events import EventBatch
from datasets.interfaces import DataSpec, EventStreamDataset
from dataclasses import dataclass, replace
from dataclasses import replace
from utils.dataset_names import make_threshold_dataset_name
"""
Charged-particle N-body benchmark adapted to the repo's EventStreamDataset interface.

Design philosophy
-----------------
This dataset mirrors the structure used in the wave / three-body files:

1) One latent physical system evolves continuously through time.
2) We split ONE long trajectory by time into train/val/test.
3) At each saved timestep, we convert the full latent particle state into a
   set of local directed interaction events.
4) The event emission policy can be dense (all_pairs) or sparse (e.g. distance
   threshold, force threshold, top-k strongest).

Underlying physics
------------------
- 2D charged particles moving inside a reflecting box
- charges sampled from {-1, 0, +1}
- pairwise Coulomb-like interactions with softening for numerical stability
- simple leapfrog-ish integration, inspired by common N-body benchmark code

Observed events
---------------
For each directed pair sender -> receiver we can expose features like:
- receiver/sender charges
- local positions / velocities
- relative displacement / velocity
- distance
- signed pair charge product
- pairwise force contribution on the receiver from the sender
- total system energy (optional global summary)

Event emission rules
--------------------
- all_pairs: emit every directed pair i <- j, j != i
- distance_threshold: emit only pairs closer than a cutoff
- force_threshold: emit only pairs whose pairwise force magnitude exceeds cutoff
- top_k: emit the globally strongest k directed interactions in a bin
"""


# -----------------------------------------------------------------------------
# PART 1: LOW-LEVEL PHYSICS HELPERS
# -----------------------------------------------------------------------------


def _l2_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise squared distances between rows of a and rows of b."""
    a_norm = (a ** 2).sum(axis=1, keepdims=True)
    b_norm = (b ** 2).sum(axis=1, keepdims=True).T
    return a_norm + b_norm - 2.0 * (a @ b.T)



def _clamp_to_box(loc: np.ndarray, vel: np.ndarray, box_size: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Reflect particles elastically at the box boundaries.

    loc: [2, N]
    vel: [2, N]
    """
    over = loc > box_size
    loc[over] = 2.0 * box_size - loc[over]
    vel[over] = -np.abs(vel[over])

    under = loc < -box_size
    loc[under] = -2.0 * box_size - loc[under]
    vel[under] = np.abs(vel[under])
    return loc, vel



def _sample_charges(
    rng: np.random.Generator,
    num_nodes: int,
    charge_types: Sequence[float],
    charge_probs: Sequence[float],
) -> np.ndarray:
    charges = rng.choice(np.asarray(charge_types, dtype=np.float64), size=(num_nodes, 1), p=np.asarray(charge_probs, dtype=np.float64))
    return charges.astype(np.float64)



def _sample_initial_state(cfg: "ChargedParticlesBinnedConfig", rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample one initial state.

    Returns
    -------
    loc : [2, N]
    vel : [2, N]
    charges : [N, 1]
    """
    charges = _sample_charges(rng, cfg.num_nodes, cfg.charge_types, cfg.charge_probs)

    loc = rng.normal(loc=0.0, scale=cfg.loc_std, size=(2, cfg.num_nodes)).astype(np.float64)
    vel = rng.normal(loc=0.0, scale=1.0, size=(2, cfg.num_nodes)).astype(np.float64)

    v_norm = np.sqrt((vel ** 2).sum(axis=0, keepdims=True))
    v_norm = np.clip(v_norm, 1e-12, None)
    vel = vel * (cfg.vel_norm / v_norm)

    loc, vel = _clamp_to_box(loc, vel, cfg.box_size)
    return loc, vel, charges



def _pair_force_matrix(
    loc: np.ndarray,
    charges: np.ndarray,
    interaction_strength: float,
    softening: float,
    max_force: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute directed pairwise forces on receiver i from sender j.

    Parameters
    ----------
    loc : [2, N]
    charges : [N, 1]

    Returns
    -------
    rel : [N, N, 2]
        rel[i, j] = x_j - x_i
    force_pair : [N, N, 2]
        force contribution ON receiver i FROM sender j
    dist : [N, N]
        softened pairwise distances
    """
    pos = loc.T  # [N,2]
    rel = pos[None, :, :] - pos[:, None, :]  # [i,j,:] = x_j - x_i

    dist2 = (rel ** 2).sum(axis=-1) + float(softening) ** 2
    dist = np.sqrt(dist2)

    qq = charges @ charges.T  # [N,N]

    # Coulomb-like force on receiver i from sender j:
    #   F_{j->i} = k * q_i q_j * (x_i - x_j) / ||x_i-x_j||^3
    # With rel = x_j - x_i, that is:
    #   F_{j->i} = -k * q_i q_j * rel / ||rel||^3
    inv_dist3 = 1.0 / (dist2 * dist)
    np.fill_diagonal(inv_dist3, 0.0)

    force_pair = -interaction_strength * qq[:, :, None] * rel * inv_dist3[:, :, None]
    force_pair = np.clip(force_pair, -max_force, max_force)
    return rel, force_pair, dist



def _kinetic_energy(vel: np.ndarray) -> float:
    return float(0.5 * np.sum(vel ** 2))



def _potential_energy(
    loc: np.ndarray,
    charges: np.ndarray,
    interaction_strength: float,
    softening: float,
) -> float:
    pos = loc.T
    dist = np.sqrt(_l2_dist(pos, pos) + float(softening) ** 2)
    qq = charges @ charges.T

    total = 0.0
    n = pos.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            total += interaction_strength * float(qq[i, j]) / float(dist[i, j])
    return total



def _total_energy(
    loc: np.ndarray,
    vel: np.ndarray,
    charges: np.ndarray,
    interaction_strength: float,
    softening: float,
) -> float:
    return _kinetic_energy(vel) + _potential_energy(loc, charges, interaction_strength, softening)



def simulate_charged_trajectory(
    cfg: "ChargedParticlesBinnedConfig",
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate one long charged-particle trajectory.

    Returns
    -------
    loc_traj : [T, N, 2]
    vel_traj : [T, N, 2]
    charges  : [N, 1]
    t_eval   : [T]
    """
    loc, vel, charges = _sample_initial_state(cfg, rng)

    micro_dt = float(cfg.micro_dt)
    max_force = float(cfg.max_force_clip)
    sample_every = max(1, int(cfg.steps_per_bin))

    loc_bins: List[np.ndarray] = []
    vel_bins: List[np.ndarray] = []
    t_eval: List[float] = []

    t_cur = 0.0

    for b in range(cfg.num_bins):
        loc_bins.append(loc.T.copy())
        vel_bins.append(vel.T.copy())
        t_eval.append(t_cur)

        # Advance by sample_every microsteps between saved bins.
        for _ in range(sample_every):
            _, force_pair, _ = _pair_force_matrix(
                loc,
                charges,
                interaction_strength=cfg.interaction_strength,
                softening=cfg.softening,
                max_force=max_force,
            )
            net_force = force_pair.sum(axis=1).T  # [2,N]

            vel = vel + micro_dt * net_force
            loc = loc + micro_dt * vel
            loc, vel = _clamp_to_box(loc, vel, cfg.box_size)
            t_cur += micro_dt

    loc_traj = np.stack(loc_bins, axis=0)  # [T,N,2]
    vel_traj = np.stack(vel_bins, axis=0)  # [T,N,2]
    return loc_traj, vel_traj, charges, np.asarray(t_eval, dtype=np.float64)


# -----------------------------------------------------------------------------
# PART 2: DATASET CONFIGURATION
# -----------------------------------------------------------------------------


@dataclass
class ChargedParticlesBinnedConfig:
    name: str = "charged_particles"

    # DATASET LENGTH / SPLIT
    num_nodes: int = 32
    num_bins: int = 256
    split_fracs: Tuple[float, float, float] = (0.7, 0.15, 0.15)

    # SIMULATION
    box_size: float = 4.0
    loc_std: float = 1.0
    vel_norm: float = 0.5
    interaction_strength: float = 1.0
    softening: float = 0.10
    micro_dt: float = 1e-3
    steps_per_bin: int = 250
    max_force_clip: float = 100.0  # similar spirit to NRI's force clipping

    # CHARGE SAMPLING
    charge_types: Tuple[float, float, float] = (-1.0, 0.0, 1.0)
    charge_probs: Tuple[float, float, float] = (0.4, 0.2, 0.4)

    # OBSERVATION NOISE (added after latent rollout)
    observation_noise_loc: float = 0.0
    observation_noise_vel: float = 0.0

    distance_threshold_jitter_std: float = 0.01
    force_threshold_jitter_std: float = 0.01

    obs_edge_keep_prob: float = 1.0     # 1.0 = keep all selected edges

    # EVENT EMISSION
    # all_pairs: every directed pair i <- j, j != i
    # distance_threshold: keep if distance <= threshold
    # force_threshold: keep if pair_force_mag >= threshold
    # top_k: keep strongest k directed pairwise forces in the bin
    interaction_rule: str = "all_pairs"
    distance_threshold: float = 2.5
    force_threshold: float = 0.20
    top_k: int = 64
    min_edges_per_bin: int = 1

    # DEVICE / RNG
    seed: int = 0
    device: Optional[torch.device] = None


# -----------------------------------------------------------------------------
# PART 3: TURN ONE CONTINUOUS PARTICLE STATE INTO LOCAL INTERACTION EVENTS
# -----------------------------------------------------------------------------


_EDGE_FEATURE_NAMES: Sequence[str] = (
    # "recv_charge",
    # "send_charge",
    # "recv_px",
    # "recv_py",
    # "recv_vx",
    # "recv_vy",
    # "send_px",
    # "send_py",
    # "send_vx",
    # "send_vy",
    # "rel_px",
    # "rel_py",
    #"rel_vx",
    #"rel_vy",
    # "distance",
    # "charge_product",
    # # "force_x",
    # # "force_y",
    # "force_mag",
    # "energy",
)



def _pair_record(
    loc: np.ndarray,
    vel: np.ndarray,
    charges: np.ndarray,
    sender: int,
    receiver: int,
    cfg: ChargedParticlesBinnedConfig,
    global_energy: float,
) -> dict:
    recv_pos = loc[receiver]
    send_pos = loc[sender]
    recv_vel = vel[receiver]
    send_vel = vel[sender]
    recv_charge = float(charges[receiver, 0])
    send_charge = float(charges[sender, 0])

    rel_pos = send_pos - recv_pos
    rel_vel = send_vel - recv_vel

    dist2 = float(np.dot(rel_pos, rel_pos) + cfg.softening ** 2)
    distance = float(np.sqrt(dist2))
    inv_dist3 = 1.0 / (dist2 * distance)

    charge_product = recv_charge * send_charge
    force_vec = -cfg.interaction_strength * charge_product * rel_pos * inv_dist3
    force_vec = np.clip(force_vec, -cfg.max_force_clip, cfg.max_force_clip)
    force_mag = float(np.linalg.norm(force_vec))

    feats = np.array(
        [
            # recv_charge,
            # send_charge,
            # float(recv_pos[0]),
            # float(recv_pos[1]),
            # float(recv_vel[0]),
            # float(recv_vel[1]),
            # float(send_pos[0]),
            # float(send_pos[1]),
            # float(send_vel[0]),
            # float(send_vel[1]),
            # float(rel_pos[0]),
            # float(rel_pos[1]),
            # float(rel_vel[0]),
            # float(rel_vel[1]),
            # distance,
            # charge_product,
            # # float(force_vec[0]),
            # # float(force_vec[1]),
            # force_mag,
            # global_energy,
        ],
        dtype=np.float32,
    )

    return {
        "src": sender,
        "dst": receiver,
        "distance": distance,
        "force_mag": force_mag,
        "charge_product": charge_product,
        "features": feats,
    }



def _select_edges(
    records: List[dict],
    cfg: ChargedParticlesBinnedConfig,
    rng: np.random.Generator | None = None,
) -> List[dict]:
    if rng is None:
        rng = np.random.default_rng(cfg.seed)

    # --- jittered thresholds ---
    dist_thr = float(cfg.distance_threshold)
    if cfg.distance_threshold_jitter_std > 0.0:
        dist_thr += float(rng.normal(0.0, cfg.distance_threshold_jitter_std))
        dist_thr = max(0.0, dist_thr)

    force_thr = float(cfg.force_threshold)
    if cfg.force_threshold_jitter_std > 0.0:
        force_thr += float(rng.normal(0.0, cfg.force_threshold_jitter_std))
        force_thr = max(0.0, force_thr)

    # --- deterministic selection first ---
    if cfg.interaction_rule == "all_pairs":
        chosen = list(records)

    elif cfg.interaction_rule == "distance_threshold":
        chosen = [r for r in records if r["distance"] <= dist_thr]

    elif cfg.interaction_rule == "force_threshold":
        chosen = [r for r in records if r["force_mag"] >= force_thr]

    elif cfg.interaction_rule == "top_k":
        k = max(1, int(cfg.top_k))
        order = np.argsort([-r["force_mag"] for r in records])
        chosen = [records[int(i)] for i in order[:k]]

    else:
        raise ValueError(
            f"unknown interaction_rule={cfg.interaction_rule!r}; expected one of "
            "{'all_pairs','distance_threshold','force_threshold','top_k'}"
        )

    # --- edge dropout after selection ---
    keep_prob = float(cfg.obs_edge_keep_prob)
    if keep_prob < 1.0 and len(chosen) > 0:
        kept = [r for r in chosen if rng.random() < keep_prob]
        chosen = kept

    # --- fallback so a bin never goes empty if you don't want that ---
    if len(chosen) < max(0, int(cfg.min_edges_per_bin)):
        fallback_k = min(len(records), max(1, int(cfg.min_edges_per_bin)))
        order = np.argsort([-r["force_mag"] for r in records])
        chosen = [records[int(i)] for i in order[:fallback_k]]

    return chosen


def _state_to_event_batch(
    loc: np.ndarray,
    vel: np.ndarray,
    charges: np.ndarray,
    t_idx: int,
    cfg: ChargedParticlesBinnedConfig,
) -> EventBatch:
    records: List[dict] = []
    num_nodes = loc.shape[0]

    global_energy = _total_energy(
        loc.T,
        vel.T,
        charges,
        interaction_strength=cfg.interaction_strength,
        softening=cfg.softening,
    )

    for receiver in range(num_nodes):
        for sender in range(num_nodes):
            if sender == receiver:
                continue
            records.append(
                _pair_record(
                    loc,
                    vel,
                    charges,
                    sender=sender,
                    receiver=receiver,
                    cfg=cfg,
                    global_energy=global_energy,
                )
            )

    rng = np.random.default_rng(cfg.seed + t_idx)
    chosen = _select_edges(records, cfg, rng=rng)

    src = torch.tensor([r["src"] for r in chosen], dtype=torch.long, device=cfg.device)
    dst = torch.tensor([r["dst"] for r in chosen], dtype=torch.long, device=cfg.device)
    feats = torch.tensor(
        np.stack([r["features"] for r in chosen], axis=0),
        dtype=torch.float32,
        device=cfg.device,
    )
    t = torch.full((src.numel(),), int(t_idx), dtype=torch.long, device=cfg.device)

    # compute per-node net force from the current latent state
    # _pair_force_matrix expects loc shape [2, N], so transpose loc from [N, 2] -> [2, N]
    _, force_pair, _ = _pair_force_matrix(
        loc.T,
        charges,
        interaction_strength=cfg.interaction_strength,
        softening=cfg.softening,
        max_force=cfg.max_force_clip,
    )

    # force_pair: [N, N, 2]
    # sum over senders j to get total force on each receiver i
    net_force = force_pair.sum(axis=1)   # [N, 2]
    delta_v = cfg.micro_dt * net_force
    
    # node regression targets: [fx, fy] or dvx dvy
    node_targets = torch.tensor(
        delta_v, # net_force
        dtype=torch.float32,
        device=cfg.device,
    )
    node_mask = torch.ones(
        (num_nodes,),
        dtype=torch.bool,
        device=cfg.device,
    )

    eb = EventBatch(
        src=cast(torch.LongTensor, src),
        dst=cast(torch.LongTensor, dst),
        t=cast(torch.LongTensor, t),
        features=feats,
        node_targets=node_targets,
        node_mask=node_mask,
    )
    if cfg.device is not None:
        eb = eb.to(cfg.device)
    return eb


# -----------------------------------------------------------------------------
# PART 4: DATASET OBJECT THAT THE REST OF THE REPO CAN USE
# -----------------------------------------------------------------------------


class ChargedParticlesBinnedDataset(EventStreamDataset):
    def __init__(self, cfg: ChargedParticlesBinnedConfig):
        self.cfg = cfg
        self.device = cfg.device
        self._rng = np.random.default_rng(cfg.seed)
        self._num_nodes = int(cfg.num_nodes)
        self._event_dim = len(_EDGE_FEATURE_NAMES)

        loc_traj, vel_traj, charges, self._t_eval = simulate_charged_trajectory(cfg, self._rng)

        if cfg.observation_noise_loc > 0.0:
            loc_traj = loc_traj + self._rng.normal(loc=0.0, scale=cfg.observation_noise_loc, size=loc_traj.shape)
        if cfg.observation_noise_vel > 0.0:
            vel_traj = vel_traj + self._rng.normal(loc=0.0, scale=cfg.observation_noise_vel, size=vel_traj.shape)

        self._loc_traj = loc_traj.astype(np.float64, copy=True)
        self._vel_traj = vel_traj.astype(np.float64, copy=True)
        self._charges = charges.astype(np.float64, copy=True)

        self._all_bins: List[EventBatch] = [
            _state_to_event_batch(
                self._loc_traj[t_idx],
                self._vel_traj[t_idx],
                self._charges,
                t_idx=t_idx,
                cfg=cfg,
            )
            for t_idx in range(cfg.num_bins)
        ]

        f_tr, f_va, f_te = cfg.split_fracs
        assert abs((f_tr + f_va + f_te) - 1.0) < 1e-6, "split_fracs must sum to 1.0"
        tr_end = int(cfg.num_bins * f_tr)
        va_end = tr_end + int(cfg.num_bins * f_va)

        self._split_ranges = {
            "train": (0, max(0, tr_end - 1)),
            "val": (tr_end, max(tr_end, va_end - 1)),
            "test": (va_end, cfg.num_bins - 1),
        }

    def spec(self) -> DataSpec:
        dt = 0.0
        if len(self._t_eval) >= 2:
            dt = float(self._t_eval[1] - self._t_eval[0])

        return DataSpec(
            name=self.cfg.name,
            num_nodes=self._num_nodes,
            event_dim=self._event_dim,
            num_events=int(sum(int(b.src.numel()) for b in self._all_bins)),
            num_bins=int(len(self._all_bins)),
            extra={
                "physics": "2D charged particles in a reflecting box with Coulomb-like interactions",
                "state_layout": "[px, py, vx, vy] per particle plus fixed scalar charge",
                "split_fracs": self.cfg.split_fracs,
                "dt": dt,
                "micro_dt": float(self.cfg.micro_dt),
                "steps_per_bin": int(self.cfg.steps_per_bin),
                "box_size": float(self.cfg.box_size),
                "interaction_strength": float(self.cfg.interaction_strength),
                "softening": float(self.cfg.softening),
                "max_force_clip": float(self.cfg.max_force_clip),
                "charge_types": list(float(x) for x in self.cfg.charge_types),
                "charge_probs": list(float(x) for x in self.cfg.charge_probs),
                "interaction_rule": self.cfg.interaction_rule,
                "distance_threshold": float(self.cfg.distance_threshold),
                "force_threshold": float(self.cfg.force_threshold),
                "top_k": int(self.cfg.top_k),
                "feature_names": list(_EDGE_FEATURE_NAMES),
                "observation_noise_loc": float(self.cfg.observation_noise_loc),
                "observation_noise_vel": float(self.cfg.observation_noise_vel),
                "node_target_dim": 2,
                "node_target_names": ["dvx", "dvy"],
            },
        )

    def bins(self, split: str = "train") -> Iterable[EventBatch]:
        assert split in self._split_ranges, f"unknown split={split}"
        b0, b1 = self._split_ranges[split]
        return _PrecomputedStream(self._all_bins, b0=b0, b1=b1)

    def trajectory(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return latent observed trajectories.

        Returns
        -------
        loc_traj : [T,N,2]
        vel_traj : [T,N,2]
        charges  : [N,1]
        t_eval   : [T]
        """
        return (
            self._loc_traj.copy(),
            self._vel_traj.copy(),
            self._charges.copy(),
            self._t_eval.copy(),
        )


@dataclass
class _PrecomputedStream(Iterable[EventBatch]):
    bins_all: Sequence[EventBatch]
    b0: int
    b1: int

    def __iter__(self) -> Iterator[EventBatch]:
        for b in range(self.b0, self.b1 + 1):
            yield self.bins_all[b]
def _tag(value: object) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def _charged_variant_name(cfg: ChargedParticlesBinnedConfig) -> str:
    parts = [
        cfg.name,
        f"rule-{cfg.interaction_rule}",
        f"keep-{_tag(cfg.obs_edge_keep_prob)}",
        f"minedges-{int(cfg.min_edges_per_bin)}",
    ]

    if cfg.interaction_rule == "distance_threshold":
        parts.extend([
            f"dthr-{_tag(cfg.distance_threshold)}",
            f"djitter-{_tag(cfg.distance_threshold_jitter_std)}",
        ])
    elif cfg.interaction_rule == "force_threshold":
        parts.extend([
            f"fthr-{_tag(cfg.force_threshold)}",
            f"fjitter-{_tag(cfg.force_threshold_jitter_std)}",
        ])
    elif cfg.interaction_rule == "top_k":
        parts.append(f"topk-{int(cfg.top_k)}")

    return "__".join(parts)


def make_charged_particle_threshold_variants(
    base_cfg: ChargedParticlesBinnedConfig,
    *,
    threshold_metric: str,
    threshold_values: Sequence[float],
) -> dict[str, ChargedParticlesBinnedDataset]:
    out: dict[str, ChargedParticlesBinnedDataset] = {}

    for thr in threshold_values:
        if threshold_metric == "force_threshold":
            cfg = replace(
                base_cfg,
                interaction_rule="force_threshold",
                force_threshold=float(thr),
                name=make_threshold_dataset_name(
                    base_cfg.name,
                    threshold_metric,
                    thr,
                ),
            )
        elif threshold_metric == "distance_threshold":
            cfg = replace(
                base_cfg,
                interaction_rule="distance_threshold",
                distance_threshold=float(thr),
                name=make_threshold_dataset_name(
                    base_cfg.name,
                    threshold_metric,
                    thr,
                ),
            )
        else:
            raise ValueError(f"Unknown threshold_metric: {threshold_metric}")

        out[cfg.name] = ChargedParticlesBinnedDataset(cfg)

    return out

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = ChargedParticlesBinnedConfig(
        num_nodes=8,
        num_bins=32,
        interaction_rule="distance_threshold",
        distance_threshold=2.0,
        observation_noise_loc=0.01,
        observation_noise_vel=0.01,
        device=device,
    )
    ds = ChargedParticlesBinnedDataset(cfg)
    print(ds.spec())
    for i, batch in zip(range(3), ds.bins("train")):
        print(f"bin={i} num_events={batch.src.numel()} src={batch.src[:8].tolist()} dst={batch.dst[:8].tolist()}")
        if batch.features is not None:
            print("features shape:", tuple(batch.features.shape))
