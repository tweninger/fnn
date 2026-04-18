from __future__ import annotations

from dataclasses import dataclass, replace
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

Physics summary
---------------
State at node i:
    q_i : displacement
    v_i : velocity

Continuous-time first-order dynamics:
    dq_i/dt = v_i
    dv_i/dt = c^2 q_xx[i] - damping * v_i

Discrete nearest-neighbor form:
    dv_i/dt = c^2 * ((q_{i-1}-q_i) + (q_{i+1}-q_i)) / dx^2 - damping * v_i
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
    num_nodes: int = 32
    num_bins: int = 256
    domain_length: float = 1.0
    t_span: Tuple[float, float] = (0.0, 20.0)
    split_fracs: Tuple[float, float, float] = (0.7, 0.15, 0.15)

    # PHYSICS
    wave_speed: float = 0.5
    damping: float = 0.1

    # INITIAL CONDITION
    init_q_scale: float = 0.3
    init_v_scale: float = 0.1
    smooth_num_rolls: Optional[int] = None
    smooth_weight_temperature: float = 10.0
    seed: int = 0

    # OBSERVATION NOISE
    observation_noise_q: float = 0.0
    observation_noise_v: float = 0.0

    # EVENT EMISSION
    event_mode: str = "thresholded"  # {"all_neighbors", "thresholded"}
    interaction_threshold: float = 19.0334
    threshold_metric: str = "pair_accel"   # {"pair_accel","rel_q","rel_v","pair_grad"}
    threshold_use_absolute: bool = False
    threshold_keep_one_if_empty: bool = True

    # ODE SOLVER
    rtol: float = 1e-9
    atol: float = 1e-9

    # Optional torch device
    device: Optional[torch.device] = None

    # Node targets
    standardize_node_targets: bool = True
    target_horizon: int = 1
    target_name: str = "delta_v"
    # choices:
    #   "delta_v" : v(t+h) - v(t)
    #   "dv"      : dt * (c^2 q_xx(t+h) - damping * v(t+h))
    #   "v"       : v(t+h)
    #   "q"       : q(t+h)
    #   "delta_q" : q(t+h) - q(t)
    #   "q_xx"    : q_xx(t+h)


# -----------------------------------------------------------------------------
# PART 3: TURN ONE CONTINUOUS FIELD STATE INTO LOCAL INTERACTION EVENTS
# -----------------------------------------------------------------------------

_EDGE_FEATURE_NAMES: Sequence[str] = (
    # Uncomment whatever you want to expose as edge features.
    # "rel_q",
    # "rel_v",
    # "direction",
    # "dx",
    # "pair_grad",
    # "pair_accel_contrib",
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
            # rel_q,
            # rel_v,
            # direction,
            # dx,
            # pair_grad,
            # pair_accel_contrib,
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


def _compute_node_target(
    *,
    q: np.ndarray,
    v: np.ndarray,
    q_target: np.ndarray,
    v_target: np.ndarray,
    cfg: "WaveEquationBinnedConfig",
) -> tuple[np.ndarray, str]:
    """
    Return per-node regression target and its base name.
    Output shape: [N]
    """
    dx = cfg.domain_length / float(cfg.num_nodes)
    dt = (cfg.t_span[1] - cfg.t_span[0]) / float(cfg.num_bins - 1)

    if cfg.target_name == "delta_v":
        y = v_target - v
        name = "delta_v"

    elif cfg.target_name == "dv":
        q_xx_tgt = periodic_laplacian(q_target, dx=dx)
        accel_tgt = (cfg.wave_speed ** 2) * q_xx_tgt - cfg.damping * v_target
        y = dt * accel_tgt
        name = "dv"

    elif cfg.target_name == "v":
        y = v_target
        name = "v"

    elif cfg.target_name == "q":
        y = q_target
        name = "q"

    elif cfg.target_name == "delta_q":
        y = q_target - q
        name = "delta_q"

    elif cfg.target_name == "q_xx":
        y = periodic_laplacian(q_target, dx=dx)
        name = "q_xx"

    else:
        raise ValueError(
            f"unknown target_name={cfg.target_name!r}; expected one of "
            "{'delta_v','dv','v','q','delta_q','q_xx'}"
        )

    return y.astype(np.float32, copy=False), name


def _state_to_event_batch(
    q: np.ndarray,
    v: np.ndarray,
    t_idx: int,
    cfg: WaveEquationBinnedConfig,
    target_mean: Optional[float] = None,
    target_std: Optional[float] = None,
    q_target: Optional[np.ndarray] = None,
    v_target: Optional[np.ndarray] = None,
) -> EventBatch:
    """
    Convert one full field snapshot at time index t_idx into one EventBatch.
    """
    records_all: List[dict] = []
    num_nodes = q.shape[0]
    dx = cfg.domain_length / float(cfg.num_nodes)

    x_positions = np.linspace(
        0.0,
        cfg.domain_length,
        num_nodes,
        endpoint=False,
        dtype=np.float64,
    )
    global_e = discrete_total_energy(q, v, cfg.wave_speed, dx)

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

    q_tgt = q if q_target is None else q_target
    v_tgt = v if v_target is None else v_target

    target_values, _ = _compute_node_target(
        q=q,
        v=v,
        q_target=q_tgt,
        v_target=v_tgt,
        cfg=cfg,
    )

    if target_mean is not None and target_std is not None:
        target_values = (target_values - target_mean) / target_std

    node_targets = torch.tensor(
        target_values[:, None],
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

        num_bins = q_traj.shape[0]

        # Keep your original ordering:
        # standardization stats are computed from the clean trajectory first.
        n_train = int(cfg.split_fracs[0] * num_bins)
        train_end = n_train

        self.target_mean = None
        self.target_std = None

        if cfg.standardize_node_targets:
            train_targets = []
            usable_train_bins = max(0, train_end - cfg.target_horizon)

            for t_idx in range(usable_train_bins):
                tgt_idx = t_idx + cfg.target_horizon

                y, _ = _compute_node_target(
                    q=q_traj[t_idx],
                    v=v_traj[t_idx],
                    q_target=q_traj[tgt_idx],
                    v_target=v_traj[tgt_idx],
                    cfg=cfg,
                )
                train_targets.append(y.reshape(-1))

            train_targets = np.concatenate(train_targets, axis=0)
            self.target_mean = float(train_targets.mean())
            self.target_std = float(max(train_targets.std(), 1e-8))

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

        target_mean = self.target_mean if cfg.standardize_node_targets else None
        target_std = self.target_std if cfg.standardize_node_targets else None

        self._all_bins: List[EventBatch] = []

        usable_bins = num_bins - cfg.target_horizon
        for t_idx in range(usable_bins):
            tgt_idx = t_idx + cfg.target_horizon

            eb = _state_to_event_batch(
                q_traj[t_idx],
                v_traj[t_idx],
                t_idx=t_idx,
                cfg=cfg,
                q_target=q_traj[tgt_idx],
                v_target=v_traj[tgt_idx],
                target_mean=target_mean,
                target_std=target_std,
            )
            self._all_bins.append(eb)

        f_tr, f_va, f_te = cfg.split_fracs
        assert abs((f_tr + f_va + f_te) - 1.0) < 1e-6, "split_fracs must sum to 1.0"

        num_effective_bins = len(self._all_bins)
        tr_end = int(num_effective_bins * f_tr)
        va_end = tr_end + int(num_effective_bins * f_va)

        self._split_ranges = {
            "train": (0, max(0, tr_end - 1)),
            "val": (tr_end, max(tr_end, va_end - 1)),
            "test": (va_end, num_effective_bins - 1),
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
                "node_target_dim": 1,
                "node_target_names": [
                    f"{self.cfg.target_name}_std"
                    if self.cfg.standardize_node_targets
                    else self.cfg.target_name
                ],
                "target_name": self.cfg.target_name,
                "target_horizon": int(self.cfg.target_horizon),
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

def _wave_variant_name(
    cfg: WaveEquationBinnedConfig,
) -> str:
    parts = [
        cfg.name,
        f"mode-{cfg.event_mode}",
        f"target-{cfg.target_name}",
        f"h-{cfg.target_horizon}",
        f"std-{cfg.standardize_node_targets}",
    ]

    if cfg.event_mode == "thresholded":
        parts.extend([
            f"metric-{cfg.threshold_metric}",
            f"thr-{cfg.interaction_threshold}",
            f"abs-{cfg.threshold_use_absolute}",
        ])

    return "__".join(parts)


def make_wave_variants(
    base_cfg: WaveEquationBinnedConfig,
    *,
    event_modes: Sequence[str] = ("thresholded",),
    threshold_metrics: Sequence[str] = ("pair_accel",),
    interaction_thresholds: Sequence[float] = (19.0334,),
    threshold_use_absolute_options: Sequence[bool] = (False,),
    standardize_node_targets_options: Sequence[bool] = (True,),
    target_names: Sequence[str] = ("dv",),
    target_horizons: Sequence[int] = (1,),
) -> dict[str, WaveEquationBinnedDataset]:
    """
    Build a dict of named WaveEquationBinnedDataset variants.

    Notes
    -----
    - For event_mode='all_neighbors', threshold-specific knobs are ignored.
    - For event_mode='thresholded', we sweep threshold metric / value / abs flag.
    """
    datasets: dict[str, WaveEquationBinnedDataset] = {}

    for event_mode in event_modes:
        for target_name in target_names:
            for target_horizon in target_horizons:
                for standardize in standardize_node_targets_options:

                    if event_mode == "all_neighbors":
                        cfg = replace(
                            base_cfg,
                            event_mode="all_neighbors",
                            target_name=target_name,
                            target_horizon=target_horizon,
                            standardize_node_targets=standardize,
                        )
                        cfg = replace(cfg, name=_wave_variant_name(cfg))
                        datasets[cfg.name] = WaveEquationBinnedDataset(cfg)
                        continue

                    if event_mode != "thresholded":
                        raise ValueError(
                            f"unknown wave event_mode={event_mode!r}; expected "
                            "{'all_neighbors', 'thresholded'}"
                        )

                    for threshold_metric in threshold_metrics:
                        for interaction_threshold in interaction_thresholds:
                            for use_abs in threshold_use_absolute_options:
                                cfg = replace(
                                    base_cfg,
                                    event_mode="thresholded",
                                    threshold_metric=threshold_metric,
                                    interaction_threshold=float(interaction_threshold),
                                    threshold_use_absolute=bool(use_abs),
                                    target_name=target_name,
                                    target_horizon=target_horizon,
                                    standardize_node_targets=standardize,
                                )
                                cfg = replace(cfg, name=_wave_variant_name(cfg))
                                datasets[cfg.name] = WaveEquationBinnedDataset(cfg)

    return datasets


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
        target_name="delta_v",
        device=device,
    )
    dense_ds = WaveEquationBinnedDataset(dense_cfg)
    print("DENSE:", dense_ds.spec())
    for i, batch in zip(range(2), dense_ds.bins("train")):
        print(f"dense bin={i} num_events={batch.src.numel()}")

    dv_cfg = WaveEquationBinnedConfig(
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
        target_name="dv",
        device=device,
    )
    dv_ds = WaveEquationBinnedDataset(dv_cfg)
    print("DV TARGET:", dv_ds.spec())
    for i, batch in zip(range(2), dv_ds.bins("train")):
        print(f"dv bin={i} num_events={batch.src.numel()}")