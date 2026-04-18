from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, cast

import numpy as np
import scipy.integrate # -> library for scientific computing (integration, optimization, linear algebra, stats, math bs)
import torch

from core.events import EventBatch
from datasets.interfaces import DataSpec, EventStreamDataset

"""
HNN paper's three-body gravitational dataset adapted to the repo's EventStreamDataset interface.

Design goals
------------
1) Keep the underlying HNN-style 3-body physics intact:
   - same equal-mass setup
   - same pairwise gravitational acceleration law (G=1)
   - same near-circular random initialization idea
   - continuous ODE integration with scipy.solve_ivp

2) Expose the dynamics in the temporal interaction format expected by this codebase:
   - each solver sample becomes one discrete time bin
   - each active directed pairwise influence j -> i becomes one event in that bin
   - event features encode the current local physics (relative displacement, velocity,
     force-like acceleration contribution, etc.)

3) Avoid a subtle mismatch with the current training harness:
   - the train loop treats each split as one continuous stream with persistent memory
   - therefore this dataset uses ONE long orbit and splits it by time
   - it does NOT stitch together multiple independent trajectories inside one split,
     because that would create fake cross-trajectory transitions.
"""

solve_ivp = scipy.integrate.solve_ivp # -> SciPy's ODE solver (a function that solves differential equations)

# -----------------------------------------------------------------------------
# PART 1: LOW LEVEL PHYSICS HELPERS (HNN-style 3-body physics)
# -----------------------------------------------------------------------------

# Rotate 2D vector by angle theta
# Set up 3 bodies by rotating the first body twice lol
def rotate2d(p: np.ndarray, theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return (R @ p.reshape(2, 1)).squeeze(-1)


# Initial physical state for the 3-body system
def random_config(
    rng: np.random.Generator,
    nu: float = 2e-1,
    min_radius: float = 0.9,
    max_radius: float = 1.2,
) -> np.ndarray:
    """
    Output shape: [3 bodies, 5 values per body]
    Each body's row is:
        [mass, px, py, vx, vy]

    Meaning of those columns:
    - mass: body mass
    - px, py: current position coordinates
    - vx, vy: current velocity coordinates

    Physics idea:
    - start with 3 equal masses
    - place them roughly symmetrically around the origin
    - choose velocities that would make a circular-ish orbit
    - then change (perturb whatever) velocities a bit so the motion is not perfectly circular (ideal is the word they always use)

    Same as OG HNN code!!
    """
    state = np.zeros((3, 5), dtype=np.float64)
    state[:, 0] = 1.0  # equal masses

    # Pick random direction for body 1, normalize to random radius
    p1 = 2.0 * rng.random(2) - 1.0
    r = float(rng.uniform(min_radius, max_radius))
    p1 *= r / np.sqrt(np.sum(p1**2))

    # Rotate them so its symmetrical 3-body setup
    p2 = rotate2d(p1, theta=2 * np.pi / 3)
    p3 = rotate2d(p2, theta=2 * np.pi / 3)

    #Start with velocity perpendicular to position for orbital motion (we are moving IN A CIRCLE), aka point body sideways
    v1 = rotate2d(p1, theta=np.pi / 2)

    # Scale velocity magnitude based on radius so it's orbit-like
    # Shout out copying because I don't know math
    v1 = v1 / (r ** 1.5)
    v1 = v1 * np.sqrt(np.sin(np.pi / 3) / (2 * np.cos(np.pi / 6) ** 2))

    # ROTATE!!! to get corresponding velocities for the other 2
    v2 = rotate2d(v1, theta=2 * np.pi / 3)
    v3 = rotate2d(v2, theta=2 * np.pi / 3)

    # make the circular orbits slightly chaotic
    # aka add noise b/c perfectly circular orbits are boring
    v1 *= 1.0 + nu * (2.0 * rng.random(2) - 1.0)
    v2 *= 1.0 + nu * (2.0 * rng.random(2) - 1.0)
    v3 *= 1.0 + nu * (2.0 * rng.random(2) - 1.0)

    state[0, 1:3], state[0, 3:5] = p1, v1
    state[1, 1:3], state[1, 3:5] = p2, v2
    state[2, 1:3], state[2, 3:5] = p3, v3
    return state


# Get net gravitational acceleration for each body
# aka for each body, how hard are the other bodies pulling on it right now
# position changes according to velocity, velocity changes according to acceleration
# so if we want to know how the system moves next, we need acceleration
def get_accelerations(state: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
    """
    Input:
        state shape = [num_bodies, 5]
        columns     = [mass, px, py, vx, vy]

    Output:
        accelerations shape = [num_bodies, 2]
        columns             = [ax, ay]

    Physics meaning:
    - For each receiver body i:
        * look at every other sender body j (where is everybody)
        * compute displacement (sender position - receiver position) 
        * compute distance (with our friend Euclidian, how far is everybody)
        * compute gravitational contribution ~ mj * displacement / distance^3 (how strongly gravity pulls from each one)
        * sum all sender contributions

    THIS IS PAIRWISE INFLUENCE IDEA!! But in math!

    -> The total acceleration of one body is the SUM of influences from the others. <-
    """
    net_accs: List[np.ndarray] = []

    for i in range(state.shape[0]):
        # ALL other bodies are potential senders acting on receiver i
        other_bodies = np.concatenate([state[:i, :], state[i + 1 :, :]], axis=0)

        #vector from receiver i to each sender j
        displacements = other_bodies[:, 1:3] - state[i, 1:3]

        # hi Euclidean distances abs value of xj-xi
        # hi middle school math!
        distances = np.sqrt((displacements**2).sum(1, keepdims=True))

        #sender masses mj
        masses = other_bodies[:, 0:1]

        # pairwise acceleration contriubtions from each sender j to receiver i
        # epsilon is tiny stabilizer to avoide divide by zero explosions nightmare
        pointwise_accs = masses * displacements / (distances**3 + epsilon)

        # net acceleration on receiever i = sum of all incoming pairwise influences
        net_accs.append(pointwise_accs.sum(0, keepdims=True))

    return np.concatenate(net_accs, axis=0)


# Solver uses get_acceleration() (how velocity changes) to generate the orbit over time
# SIMULATE THE MOTION
def update(_t: float, flat_state: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
    """
    ODE right-hand side: given the current state, return its time derivative.

    This is the thing the solver calls over and over -> solver: numerical took that given the rule for how the system changes,
    I'll step it forward through time.
    Give current state & equation for how the state changes -> Get what the state should be later, and later and later ...

    Input is FLAT because scipy's solver wants one long 1D vector.
    We reshape it back to [num_bodies, 5] so it is readable.

    Mathematical meaning:
    - d(px)/dt = vx
    - d(py)/dt = vy
    - d(vx)/dt = ax
    - d(vy)/dt = ay

    """

    state = flat_state.reshape(-1, 5)
    deriv = np.zeros_like(state)

    # position derivatives are just current velocities 
    deriv[:, 1:3] = state[:, 3:5]  # dx/dt = vx, dy/dt = vy

    # velocity derivatives are the accelerations caused by gravity
    deriv[:, 3:5] = get_accelerations(state, epsilon=epsilon)

    return deriv.reshape(-1)

# start from one initial 3-body setup, then simulate how it moves over time
# collects the movie frames/states ... simulates it!!!! yes
def get_orbit(
    state: np.ndarray,
    t_points: int,
    t_span: Tuple[float, float],
    epsilon: float = 0.0,
    rtol: float = 1e-9,
    atol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    
    """
    Numerically integrate ONE continuous orbit.

    Inputs:
    - state   : initial state at time t_span[0]
    - t_points: how many sample times we want back
    - t_span  : start/end time, e.g. (0, 40)

    Outputs:
    - orbit: shape [bodies, properties, time]
    - t_eval: the actual sampled times

    What this is doing in plain English:
    - create initial state
    - call the solver/hand the physics ODE to scipy
    - ask it to solve the motion continuously over time
    - sample the system at evenly spaced times

    This is the step that turns "initial condition + laws of motion" into an actual trajectory.
    """

    t_eval = np.linspace(t_span[0], t_span[1], t_points, dtype=np.float64)
    path = solve_ivp(
        fun=lambda t, y: update(t, y, epsilon=epsilon),
        t_span=t_span,
        y0=state.flatten(),
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )

    # Solver returns flattened states over time: reshape back to [body, property, time]
    orbit = path.y.reshape(state.shape[0], 5, t_points)
    return orbit, t_eval

# compute gravitational potential energy of the whole 3-body system
# for each unordered pair ((i,j) but not (j,i)), potential energy is ~ -mimj / r ij (negative sign b/c gravity is attractive, aka when bodies are close together, the system is in "lower energy config" aka you have to add energy to pull them apart)
# like more negative means you need more energy to push it (more negative also means they're closer together)
# mimj is the two masses and rij is the distance between them
# divide by distance because gravitational interaction gets weaker when objects are farther apart
def potential_energy(state: np.ndarray) -> float:
    tot = 0.0
    for i in range(state.shape[0]):
        for j in range(i + 1, state.shape[0]):
            r = float(np.linalg.norm(state[i, 1:3] - state[j, 1:3]))
            tot += state[i, 0] * state[j, 0] / r
    return -tot

# compute total kinetic energy = sum over bodies of .5 times m times abs val of v^2
def kinetic_energy(state: np.ndarray) -> float:
    return float((0.5 * state[:, 0] * (state[:, 3:5] ** 2).sum(axis=1)).sum())

# total mechanical energy = kinetic + potential
# in idea conservative dynamics, total energy should stay roughly constant (nice physics summary signal)
def total_energy(state: np.ndarray) -> float:
    return potential_energy(state) + kinetic_energy(state)


# -----------------------------------------------------------------------------
# PART 2: Dataset configuration
# Knobs you turn when building the dataset
# Like controlling physics, observation process, how we turn continuous pairwise influences into discrete events...
# -----------------------------------------------------------------------------

@dataclass
class ThreeBodyBinnedConfig:
    name: str = "three_body_binned"

    # DATASET LENGTH + TIME SPAN
    # One long orbit, split by time into train/val/test to match the current harness.
    num_bins: int = 1024
    t_span: Tuple[float, float] = (0.0, 40.0)
    split_fracs: Tuple[float, float, float] = (0.7, 0.15, 0.15)

    # PHYSICS / INITIALIZATION
    orbit_noise: float = 2e-1 # how much randomness to inject into initial velocities
    min_radius: float = 0.9 # random initial radius lower bound
    max_radius: float = 1.2 # random initial radius upper bound
    epsilon: float = 1e-9 # tiny numerical stabilizer for 1 / distance^3
    seed: int = 0 # RNG seed / reproducability

    # OBSERVATION NOISE
    # added after solving true dynamics, aka underlying orbit is clean, only our measured view gets noisy
    observation_noise_pos: float = 0.0
    observation_noise_vel: float = 0.0

    # INTERACTION EXTRACTION
    # Continuous gravity is always on, but we need discrete events so...
    # How continuous pairwise influences become discrete events.
    # all_pairs         : emit all directed pairwise interactions every bin
    # force_threshold   : emit j->i only if ||a_ij|| >= force_threshold
    # distance_threshold: emit j->i only if ||x_j - x_i|| <= distance_threshold
    # top_k             : keep the globally strongest k directed influences per bin
    interaction_rule: str = "all_pairs"
    force_threshold: float = 0.20
    distance_threshold: float = 1.75
    top_k: int = 3
    min_edges_per_bin: int = 1

    # Optional torch device for EventBatch tensors
    device: Optional[torch.device] = None


# -----------------------------------------------------------------------------
# PART 3: TURN ONE CONTINUOUS STATE INTO LOCAL PAIRWISE INTERACTIONS (Event conversion helpers)
# -----------------------------------------------------------------------------

_EDGE_FEATURE_NAMES: Sequence[str] = (

    # receiver body identity / local state
    # "recv_mass",
    # "send_mass",
    # "recv_px",
    # "recv_py",
    # "recv_vx",
    # "recv_vy",

    # sender body local state
    # "send_px",
    # "send_py",
    # "send_vx",
    # "send_vy",

    # relationship features between sender and receiver
    # "rel_px",
    # "rel_py",
    # "rel_vx",
    # "rel_vy",
    # "distance",

    # force like / influence features
    # "acc_x",
    # "acc_y",
    # "acc_mag",

    # one global summary feature
    # "energy",
)


def _pair_record(state: np.ndarray, sender: int, receiver: int, epsilon: float) -> dict:
    """
    Build ONE directed pairwise interaction record: sender -> receiver.

    Example:
        sender=2, receiver=0
    means:
        "how is body 2 influencing body 0 right now?"

    This is the core local representation.
    HNN would have hidden this inside a big global vector.
    We make it explicit as an edge/event.
    """
    recv = state[receiver]
    send = state[sender]

    recv_mass = float(recv[0])
    send_mass = float(send[0])
    recv_pos = recv[1:3]
    send_pos = send[1:3]
    recv_vel = recv[3:5]
    send_vel = send[3:5]

    # relative quantities are usually what matters for pairwise interaction
    rel_pos = send_pos - recv_pos
    rel_vel = send_vel - recv_vel

    # pairwise gravittaional acceleration contribution from sender to receiver
    # NOT full net acceleration on receiver... just one sender's piece
    distance = float(np.linalg.norm(rel_pos))
    acc_vec = send_mass * rel_pos / (distance**3 + epsilon)
    acc_mag = float(np.linalg.norm(acc_vec))

    feats = np.array(
        [
            # recv_mass,
            # send_mass,
            # recv_pos[0],
            # recv_pos[1],
            # recv_vel[0],
            # recv_vel[1],
            # send_pos[0],
            # send_pos[1],
            # send_vel[0],
            # send_vel[1],
            # rel_pos[0],
            # rel_pos[1],
            # rel_vel[0],
            # rel_vel[1],
            # distance,
            # acc_vec[0],
            # acc_vec[1],
            # acc_mag,
            # total_energy(state),
        ],
        dtype=np.float32,
    )

    return {
        "src": sender,
        "dst": receiver,
        "distance": distance,
        "acc_mag": acc_mag,
        "features": feats,
    }


def _select_edges(records: List[dict], cfg: ThreeBodyBinnedConfig) -> List[dict]:
    if cfg.interaction_rule == "all_pairs":
        chosen = list(records)
    elif cfg.interaction_rule == "force_threshold":
        chosen = [r for r in records if r["acc_mag"] >= cfg.force_threshold]
    elif cfg.interaction_rule == "distance_threshold":
        chosen = [r for r in records if r["distance"] <= cfg.distance_threshold]
    elif cfg.interaction_rule == "top_k":
        k = max(1, int(cfg.top_k))
        order = np.argsort([-r["acc_mag"] for r in records])
        chosen = [records[int(i)] for i in order[:k]]
    else:
        raise ValueError(f"unknown interaction_rule={cfg.interaction_rule}")

    if len(chosen) < max(0, int(cfg.min_edges_per_bin)):
        order = np.argsort([-r["acc_mag"] for r in records])
        fallback_k = min(len(records), max(1, int(cfg.min_edges_per_bin)))
        chosen = [records[int(i)] for i in order[:fallback_k]]

    return chosen

# Convert ONE continuous full-system state at time index t_idx into ONE EventBatch.
def _state_to_event_batch(
    state: np.ndarray,
    t_idx: int,
    cfg: ThreeBodyBinnedConfig,
) -> EventBatch:
    records: List[dict] = []
    num_bodies = state.shape[0]

    # For 3 bodies, this generates 6 directed candidate interactions:
    # 0->1, 1->0, 0->2, 2->0, 1->2, 2->1
    for receiver in range(num_bodies):
        for sender in range(num_bodies):
            if sender == receiver:
                continue
            records.append(_pair_record(state, sender=sender, receiver=receiver, epsilon=cfg.epsilon))

    chosen = _select_edges(records, cfg)

    # Python records -> torch tensors expected by EventBatch
    src = torch.tensor([r["src"] for r in chosen], dtype=torch.long, device=cfg.device)
    dst = torch.tensor([r["dst"] for r in chosen], dtype=torch.long, device=cfg.device)
    feats = torch.tensor(np.stack([r["features"] for r in chosen], axis=0), dtype=torch.float32, device=cfg.device)
    node_targets = torch.tensor(
        state[:, 1:5],   # [px, py, vx, vy]
        dtype=torch.float32,
        device=cfg.device,
    )
    node_mask = torch.ones((state.shape[0],), dtype=torch.bool, device=cfg.device)

    # All events in this batch share the same discrete time index
    t = torch.full((src.numel(),), int(t_idx), dtype=torch.long, device=cfg.device)

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
# PART 4: DATASET OBJECT THAT THE REST OF THE REPO CAN USE (dataset class)
# -----------------------------------------------------------------------------

class ThreeBodyBinnedDataset(EventStreamDataset):
    def __init__(self, cfg: ThreeBodyBinnedConfig):
        self.cfg = cfg
        self.device= cfg.device
        self._rng = np.random.default_rng(cfg.seed)
        self._event_dim = len(_EDGE_FEATURE_NAMES)
        self._num_nodes = 3

        # sample one initial physical configuration
        state0 = random_config(
            rng=self._rng,
            nu=cfg.orbit_noise,
            min_radius=cfg.min_radius,
            max_radius=cfg.max_radius,
        )

        # solve one long continuous orbit from that initial condition
        orbit, self._t_eval = get_orbit(
            state=state0,
            t_points=cfg.num_bins,
            t_span=cfg.t_span,
            epsilon=cfg.epsilon,
        )

        # orbit shape: [bodies, properties, time]
        states = orbit.transpose(2, 0, 1).copy()  # [time, bodies, properties]

        # optionally add observation noise AFTER solving the dynamics
        if cfg.observation_noise_pos > 0:
            states[:, :, 1:3] += self._rng.normal(
                loc=0.0,
                scale=cfg.observation_noise_pos,
                size=states[:, :, 1:3].shape,
            )
        if cfg.observation_noise_vel > 0:
            states[:, :, 3:5] += self._rng.normal(
                loc=0.0,
                scale=cfg.observation_noise_vel,
                size=states[:, :, 3:5].shape,
            )

        # turn each time slice into one interaction bin
        self._all_bins: List[EventBatch] = [
            _state_to_event_batch(states[t_idx], t_idx=t_idx, cfg=cfg)
            for t_idx in range(states.shape[0])
        ]

        # split the one long stream by TIME, not by independent trajectories
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
        dt = 0.0 # -> how far apart time bins are (simulation span/# of bins) rn bins are uniform steps
        if len(self._t_eval) >= 2:
            dt = float(self._t_eval[1] - self._t_eval[0])

        return DataSpec(
            name=self.cfg.name,
            num_nodes=self._num_nodes,
            event_dim=self._event_dim,
            num_events=int(sum(int(b.src.numel()) for b in self._all_bins)),
            num_bins=int(len(self._all_bins)),
            extra={
                "physics": "HNN-style equal-mass 3-body gravity, G=1",
                "state_layout": "[mass, px, py, vx, vy] per body",
                "split_fracs": self.cfg.split_fracs,
                "t_span": tuple(float(x) for x in self.cfg.t_span),
                "dt": dt,
                "interaction_rule": self.cfg.interaction_rule,
                "force_threshold": float(self.cfg.force_threshold),
                "distance_threshold": float(self.cfg.distance_threshold),
                "top_k": int(self.cfg.top_k),
                "feature_names": list(_EDGE_FEATURE_NAMES),
                "observation_noise_pos": float(self.cfg.observation_noise_pos),
                "observation_noise_vel": float(self.cfg.observation_noise_vel),
                "node_target_dim": 4,
                "node_target_names": ["px", "py", "vx", "vy"],
            },
        )

    # returns iterable over EventBatch bins for requested split -> what training loop consumes
    # aka whole time-ordered stream of bins (anything python can iterate thru hello)
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


# smoke test
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = ThreeBodyBinnedConfig(
        num_bins=32,
        interaction_rule="force_threshold",
        force_threshold=0.25,
        observation_noise_pos=0.0,
        observation_noise_vel=0.0,
        device=device,
    )
    ds = ThreeBodyBinnedDataset(cfg)
    print(ds.spec())
    for i, batch in zip(range(3), ds.bins("train")):
        print(f"bin={i} num_events={batch.src.numel()} src={batch.src.tolist()} dst={batch.dst.tolist()}")
        if batch.features is not None:
            print("features shape:", tuple(batch.features.shape))
