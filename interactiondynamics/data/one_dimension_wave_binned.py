
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, cast

import numpy as np
import scipy.integrate
import torch

from core.events import EventBatch
from data.interfaces import DataSpec, EventStreamDataset

"""
LNN-inspired 1D wave equation benchmark adapted to this repo's EventStreamDataset interface.

Core idea
---------
This file keeps the *underlying* 1D wave / spring-lattice physics, but exposes it
in an event-stream style that fits the current TGN/IFT training harness.

Important conceptual choice
---------------------------
For this system there are really two separate layers:

1) Physics graph
   - nodes are grid sites on a 1D periodic lattice (a ring)
   - each node is physically coupled to its immediate left/right neighbors
   - that coupling is always present in the true dense physics

2) Event emission rule
   - dense / "all_neighbors": emit every physical neighbor interaction every bin
   - thresholded: emit only neighbor interactions whose salience exceeds a cutoff

This is exactly analogous to the 3-body distinction between:
- faithful all-pairs physics
- thresholded sparse eventization

Physics summary
---------------
State at node i:
    q_i : displacement
    v_i : velocity

Continuous-time first-order dynamics:
    dq_i/dt = v_i
    dv_i/dt = c^2 * q_xx[i] - damping * v_i

Discrete nearest-neighbor form:
    dv_i/dt = c^2 * ((q_{i-1}-q_i) + (q_{i+1}-q_i)) / dx^2 - damping * v_i

So the natural pairwise edge message from sender j -> receiver i is:
    m_{j->i} = c^2 * (q_j - q_i) / dx^2

This means:
- neighbor edges carry the coupling
- the apparent "-2 q_i" center term is just what you get after summing the two
  pairwise spring-like neighbor contributions
- we do NOT need an explicit self-event to make the physics interpretable
"""

solve_ivp = scipy.integrate.solve_ivp


# -----------------------------------------------------------------------------
# PART 1: LOW-LEVEL PHYSICS HELPERS
# -----------------------------------------------------------------------------

def periodic_laplacian(q: np.ndarray, dx: float) -> np.ndarray:
    """Periodic 1D second-difference Laplacian."""
    q_plus = np.roll(q, -1)
    q_minus = np.roll(q, +1)
    return (q_plus - 2.0 * q + q_minus) / (dx ** 2)


def wave_rhs(
    _t: float,
    state: np.ndarray,
    *,
    num_nodes: int,
    dx: float,
    wave_speed: float,
    damping: float,
) -> np.ndarray:
    """
    First-order wave equation:
        dq/dt = v
        dv/dt = c^2 q_xx - damping * v

    state layout:
        [q_0, ..., q_{N-1}, v_0, ..., v_{N-1}]
    """
    q = state[:num_nodes]
    v = state[num_nodes:]

    q_xx = periodic_laplacian(q, dx=dx)
    dqdt = v
    dvdt = (wave_speed ** 2) * q_xx - damping * v
    return np.concatenate([dqdt, dvdt], axis=0)


def _smooth_periodic_noise(
    rng: np.random.Generator,
    num_nodes: int,
    num_rolls: int,
    *,
    weight_temperature: float = 10.0,
) -> np.ndarray:
    """
    Build a smooth periodic random field by blending circularly shifted white noise.
    Kept close in spirit to the LNN notebook's random smooth field construction.
    """
    noise = rng.normal(size=(num_nodes,)).astype(np.float64)

    num_rolls = max(1, int(num_rolls))
    center = num_rolls / 2.0
    weights = np.array(
        [2.0 ** (-((center - i) ** 2) / weight_temperature) for i in range(num_rolls)],
        dtype=np.float64,
    )

    field = np.zeros((num_nodes,), dtype=np.float64)
    for i, w in enumerate(weights):
        field += w * np.roll(noise, i)

    std = float(field.std())
    if std > 1e-12:
        field = field / std
    field = field - field.mean()
    return field


def sample_initial_state(
    cfg: "WaveEquationBinnedConfig",
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one smooth initial displacement field q and velocity field v."""
    num_rolls = cfg.smooth_num_rolls
    if num_rolls is None:
        num_rolls = max(3, cfg.num_nodes // 10)

    q0 = cfg.init_q_scale * _smooth_periodic_noise(
        rng,
        cfg.num_nodes,
        num_rolls=num_rolls,
        weight_temperature=cfg.smooth_weight_temperature,
    )
    v0 = cfg.init_v_scale * _smooth_periodic_noise(
        rng,
        cfg.num_nodes,
        num_rolls=num_rolls,
        weight_temperature=cfg.smooth_weight_temperature,
    )
    return np.concatenate([q0, v0], axis=0)


def integrate_wave_trajectory(
    state0: np.ndarray,
    cfg: "WaveEquationBinnedConfig",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrate one long continuous trajectory.

    Returns
    -------
    states : ndarray [T, 2N]
        Stacked states over time.
    t_eval : ndarray [T]
        Sample times.
    """
    dx = cfg.domain_length / float(cfg.num_nodes)
    t_eval = np.linspace(cfg.t_span[0], cfg.t_span[1], cfg.num_bins, dtype=np.float64)

    path = solve_ivp(
        fun=lambda t, y: wave_rhs(
            t,
            y,
            num_nodes=cfg.num_nodes,
            dx=dx,
            wave_speed=cfg.wave_speed,
            damping=cfg.damping,
        ),
        t_span=cfg.t_span,
        y0=state0.astype(np.float64),
        t_eval=t_eval,
        rtol=cfg.rtol,
        atol=cfg.atol,
    )

    states = path.y.T.copy()
    return states, t_eval


def discrete_total_energy(q: np.ndarray, v: np.ndarray, wave_speed: float, dx: float) -> float:
    """
    Discrete total energy on a periodic 1D grid:
        E = 1/2 * sum_i [ v_i^2 + c^2 * ((q_{i+1}-q_i)/dx)^2 ] * dx
    """
    grad = (np.roll(q, -1) - q) / dx
    density = 0.5 * (v ** 2 + (wave_speed ** 2) * (grad ** 2))
    return float(np.sum(density) * dx)


def local_energy_density(q: np.ndarray, v: np.ndarray, i: int, wave_speed: float, dx: float) -> float:
    """Symmetric local energy density around node i."""
    q_left = q[(i - 1) % q.shape[0]]
    q_i = q[i]
    q_right = q[(i + 1) % q.shape[0]]

    left_grad = (q_i - q_left) / dx
    right_grad = (q_right - q_i) / dx

    return float(
        0.5 * v[i] ** 2
        + 0.25 * (wave_speed ** 2) * left_grad ** 2
        + 0.25 * (wave_speed ** 2) * right_grad ** 2
    )


# -----------------------------------------------------------------------------
# PART 2: DATASET CONFIGURATION
# -----------------------------------------------------------------------------

@dataclass
class WaveEquationBinnedConfig:
    name: str = "wave_equation_binned"

    # GRID / TIME
    num_nodes: int = 64
    num_bins: int = 400
    domain_length: float = 1.0
    t_span: Tuple[float, float] = (0.0, 20.0)
    split_fracs: Tuple[float, float, float] = (0.7, 0.15, 0.15)

    # PHYSICS
    wave_speed: float = 1.0
    damping: float = 0.0

    # INITIAL CONDITION
    init_q_scale: float = 0.5
    init_v_scale: float = 0.25
    smooth_num_rolls: Optional[int] = None
    smooth_weight_temperature: float = 10.0
    seed: int = 0

    # OBSERVATION NOISE (added after solving the clean dynamics)
    observation_noise_q: float = 0.0
    observation_noise_v: float = 0.0

    # EVENT EMISSION
    # "all_neighbors" = faithful dense physics graph emission
    # "thresholded"   = sparse eventization layer on top of the same physics graph
    event_mode: str = "thresholded"
    interaction_threshold: float = 658.964
    threshold_metric: str = "pair_accel"   # {"pair_accel","rel_q","rel_v","pair_grad"}
    threshold_use_absolute: bool = True
    threshold_keep_one_if_empty: bool = True

    # ODE SOLVER
    rtol: float = 1e-9
    atol: float = 1e-9

    # Optional torch device for EventBatch tensors
    device: Optional[torch.device] = None


# -----------------------------------------------------------------------------
# PART 3: TURN ONE CONTINUOUS FIELD STATE INTO LOCAL INTERACTION EVENTS
# -----------------------------------------------------------------------------

_EDGE_FEATURE_NAMES: Sequence[str] = (
    # geometry on the ring
    "recv_x",
    "send_x",

    # receiver local state
    "recv_q",
    "recv_v",

    # sender local state
    "send_q",
    "send_v",

    # pairwise relative state
    "rel_q",
    "rel_v",

    # local coupling bookkeeping
    "direction",              # -1 for left neighbor, +1 for right neighbor
    "dx",
    "pair_grad",              # (q_j - q_i) / dx
    "pair_accel_contrib",     # c^2 * (q_j - q_i) / dx^2

    # optional summary features
    "recv_local_energy",
    "global_energy",
)


def _pair_record(
    q: np.ndarray,
    v: np.ndarray,
    sender: int,
    receiver: int,
    cfg: WaveEquationBinnedConfig,
    x_positions: np.ndarray,
    global_e: float,
) -> dict:
    """
    Build one directed nearest-neighbor interaction record: sender -> receiver.

    Pairwise interpretation:
        m_{j->i} = c^2 * (q_j - q_i) / dx^2

    Summing the two incoming neighbor contributions to i gives the coupling part of
    the discrete wave update. Damping remains a local node effect and is *not*
    emitted as a separate self-event in this version.
    """
    num_nodes = q.shape[0]
    dx = cfg.domain_length / float(cfg.num_nodes)

    recv_q = float(q[receiver])
    recv_v = float(v[receiver])
    send_q = float(q[sender])
    send_v = float(v[sender])

    rel_q = send_q - recv_q
    rel_v = send_v - recv_v

    if sender == (receiver - 1) % num_nodes:
        direction = -1.0
    elif sender == (receiver + 1) % num_nodes:
        direction = +1.0
    else:
        raise ValueError("wave event sender must be an immediate periodic neighbor of receiver")

    pair_grad = rel_q / dx
    pair_accel_contrib = (cfg.wave_speed ** 2) * rel_q / (dx ** 2)
    recv_local_e = local_energy_density(q, v, receiver, cfg.wave_speed, dx)

    feats = np.array(
        [
            x_positions[receiver],
            x_positions[sender],
            recv_q,
            recv_v,
            send_q,
            send_v,
            rel_q,
            rel_v,
            direction,
            dx,
            pair_grad,
            pair_accel_contrib,
            recv_local_e,
            global_e,
        ],
        dtype=np.float32,
    )

    return {
        "src": sender,
        "dst": receiver,
        "features": feats,
        "rel_q": rel_q,
        "rel_v": rel_v,
        "pair_grad": pair_grad,
        "pair_accel_contrib": pair_accel_contrib,
    }


def _interaction_strength(record: dict, cfg: WaveEquationBinnedConfig) -> float:
    """
    Scalar used for thresholded event emission.

    This does NOT change the underlying physics graph.
    It only decides whether a physically valid neighbor coupling is emitted as an
    observed event in the sparse thresholded variant.
    """
    metric = cfg.threshold_metric
    if metric == "pair_accel":
        val = float(record["pair_accel_contrib"])
    elif metric == "rel_q":
        val = float(record["rel_q"])
    elif metric == "rel_v":
        val = float(record["rel_v"])
    elif metric == "pair_grad":
        val = float(record["pair_grad"])
    else:
        raise ValueError(
            f"unknown threshold_metric={metric!r}; expected one of "
            "{'pair_accel','rel_q','rel_v','pair_grad'}"
        )
    return abs(val) if cfg.threshold_use_absolute else val


def _filter_records(records_all: List[dict], cfg: WaveEquationBinnedConfig) -> List[dict]:
    """
    Apply the event emission policy on top of the fixed nearest-neighbor physics graph.
    """
    if cfg.event_mode == "all_neighbors":
        return records_all

    if cfg.event_mode != "thresholded":
        raise ValueError(
            f"unknown event_mode={cfg.event_mode!r}; expected 'all_neighbors' or 'thresholded'"
        )

    thr = float(cfg.interaction_threshold)
    kept = [r for r in records_all if _interaction_strength(r, cfg) >= thr]

    if (not kept) and cfg.threshold_keep_one_if_empty and records_all:
        kept = [max(records_all, key=lambda r: _interaction_strength(r, cfg))]

    return kept


def _state_to_event_batch(
    q: np.ndarray,
    v: np.ndarray,
    t_idx: int,
    cfg: WaveEquationBinnedConfig,
) -> EventBatch:
    """
    Convert one full field snapshot at time index t_idx into one EventBatch.

    Underlying dense physics graph:
        each node i has exactly two incoming nearest-neighbor couplings
            (i-1) -> i
            (i+1) -> i

    Event emission options:
        - all_neighbors: emit all 2N directed couplings every bin
        - thresholded: emit only couplings whose salience exceeds the cutoff
    """
    records_all: List[dict] = []
    num_nodes = q.shape[0]
    x_positions = np.linspace(
        0.0,
        cfg.domain_length,
        num_nodes,
        endpoint=False,
        dtype=np.float64,
    )
    global_e = discrete_total_energy(q, v, cfg.wave_speed, cfg.domain_length / float(cfg.num_nodes))

    for receiver in range(num_nodes):
        left = (receiver - 1) % num_nodes
        right = (receiver + 1) % num_nodes

        records_all.append(
            _pair_record(
                q,
                v,
                sender=left,
                receiver=receiver,
                cfg=cfg,
                x_positions=x_positions,
                global_e=global_e,
            )
        )
        records_all.append(
            _pair_record(
                q,
                v,
                sender=right,
                receiver=receiver,
                cfg=cfg,
                x_positions=x_positions,
                global_e=global_e,
            )
        )

    records = _filter_records(records_all, cfg)

    src = torch.tensor([r["src"] for r in records], dtype=torch.long, device=cfg.device)
    dst = torch.tensor([r["dst"] for r in records], dtype=torch.long, device=cfg.device)
    feats = torch.tensor(
        np.stack([r["features"] for r in records], axis=0),
        dtype=torch.float32,
        device=cfg.device,
    )
    t = torch.full((src.numel(),), int(t_idx), dtype=torch.long, device=cfg.device)

    eb = EventBatch(
        src=cast(torch.LongTensor, src),
        dst=cast(torch.LongTensor, dst),
        t=cast(torch.LongTensor, t),
        features=feats,
    )
    if cfg.device is not None:
        eb = eb.to(cfg.device)
    return eb


# -----------------------------------------------------------------------------
# PART 4: DATASET OBJECT THAT THE REST OF THE REPO CAN USE
# -----------------------------------------------------------------------------

class WaveEquationBinnedDataset(EventStreamDataset):
    def __init__(self, cfg: WaveEquationBinnedConfig):
        self.cfg = cfg
        self.device = cfg.device
        self._rng = np.random.default_rng(cfg.seed)

        self._num_nodes = int(cfg.num_nodes)
        self._event_dim = len(_EDGE_FEATURE_NAMES)

        state0 = sample_initial_state(cfg, self._rng)
        states, self._t_eval = integrate_wave_trajectory(state0, cfg)

        q_traj = states[:, : cfg.num_nodes].copy()
        v_traj = states[:, cfg.num_nodes :].copy()

        if cfg.observation_noise_q > 0.0:
            q_traj += self._rng.normal(
                loc=0.0,
                scale=cfg.observation_noise_q,
                size=q_traj.shape,
            )
        if cfg.observation_noise_v > 0.0:
            v_traj += self._rng.normal(
                loc=0.0,
                scale=cfg.observation_noise_v,
                size=v_traj.shape,
            )

        self._q_traj = q_traj
        self._v_traj = v_traj

        self._all_bins: List[EventBatch] = [
            _state_to_event_batch(q_traj[t_idx], v_traj[t_idx], t_idx=t_idx, cfg=cfg)
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
        dx = self.cfg.domain_length / float(self.cfg.num_nodes)
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
                "physics": "1D periodic wave equation in first-order form",
                "interaction_view": "pairwise nearest-neighbor spring-like coupling on a ring",
                "state_layout": "[q_0..q_{N-1}, v_0..v_{N-1}]",
                "split_fracs": self.cfg.split_fracs,
                "t_span": tuple(float(x) for x in self.cfg.t_span),
                "dt": dt,
                "dx": dx,
                "domain_length": float(self.cfg.domain_length),
                "wave_speed": float(self.cfg.wave_speed),
                "damping": float(self.cfg.damping),
                "feature_names": list(_EDGE_FEATURE_NAMES),
                "event_mode": self.cfg.event_mode,
                "interaction_threshold": float(self.cfg.interaction_threshold),
                "threshold_metric": self.cfg.threshold_metric,
                "threshold_use_absolute": bool(self.cfg.threshold_use_absolute),
                "threshold_keep_one_if_empty": bool(self.cfg.threshold_keep_one_if_empty),
                "observation_noise_q": float(self.cfg.observation_noise_q),
                "observation_noise_v": float(self.cfg.observation_noise_v),
            },
        )

    def bins(self, split: str = "train") -> Iterable[EventBatch]:
        assert split in self._split_ranges, f"unknown split={split}"
        b0, b1 = self._split_ranges[split]
        return _PrecomputedStream(self._all_bins, b0=b0, b1=b1)

    def trajectory(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return q_traj [T,N], v_traj [T,N], t_eval [T]."""
        return self._q_traj.copy(), self._v_traj.copy(), self._t_eval.copy()


@dataclass
class _PrecomputedStream(Iterable[EventBatch]):
    bins_all: Sequence[EventBatch]
    b0: int
    b1: int

    def __iter__(self) -> Iterator[EventBatch]:
        for b in range(self.b0, self.b1 + 1):
            yield self.bins_all[b]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dense_cfg = WaveEquationBinnedConfig(
        num_nodes=16,
        num_bins=32,
        domain_length=1.0,
        t_span=(0.0, 5.0),
        wave_speed=1.0,
        damping=0.05,
        event_mode="all_neighbors",
        device=device,
    )
    dense_ds = WaveEquationBinnedDataset(dense_cfg)
    print("DENSE:", dense_ds.spec())
    for i, batch in zip(range(2), dense_ds.bins("train")):
        print(f"dense bin={i} num_events={batch.src.numel()}")

    thr_cfg = WaveEquationBinnedConfig(
        num_nodes=16,
        num_bins=32,
        domain_length=1.0,
        t_span=(0.0, 5.0),
        wave_speed=1.0,
        damping=0.05,
        event_mode="thresholded",
        interaction_threshold=1.0,
        threshold_metric="pair_accel",
        threshold_keep_one_if_empty=True,
        device=device,
    )
    thr_ds = WaveEquationBinnedDataset(thr_cfg)
    print("THRESHOLDED:", thr_ds.spec())
    for i, batch in zip(range(2), thr_ds.bins("train")):
        print(f"thresholded bin={i} num_events={batch.src.numel()}")
