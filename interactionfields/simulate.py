from __future__ import annotations

from collections import deque
from typing import Any, Callable, Dict, List, Tuple, Optional, Literal, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, diags, issparse

__all__ = [
    "run_simulator",
    "SIMULATORS",
    "simulate_faucet_on_graph",
    "simulate_waves_on_graph",
    # new
    "simulate_field_dynamics",
    "simulate_sis",
    "simulate_threshold",
    "simulate_voter",
    "simulate_transport",
    "simulate_hawkes_edges",
]

# =============================================================================
# Private helpers
# =============================================================================

def _as_square_csr(adj) -> csr_matrix:
    """Return a square CSR adjacency (converts from any SciPy sparse type)."""
    if not issparse(adj):
        raise TypeError("adj must be a scipy.sparse matrix")
    if adj.shape[0] != adj.shape[1]:
        raise ValueError("adj must be square")
    return adj if isinstance(adj, csr_matrix) else adj.tocsr()

def _csr_degree(adj: csr_matrix) -> np.ndarray:
    """Degree vector deg = A @ 1 (float64)."""
    return np.asarray(adj.sum(axis=1)).ravel().astype(np.float64)

def _bfs_hops_csr(adj: csr_matrix, src: int) -> np.ndarray:
    """Hop distances (0,1,2,...) from src on an unweighted CSR graph."""
    assert adj.shape is not None
    N = adj.shape[0]
    dist = np.full(N, np.iinfo(np.int32).max, dtype=np.int32)
    dist[src] = 0
    q = deque([src])
    indptr, indices = adj.indptr, adj.indices
    while q:
        u = q.popleft()
        start, end = indptr[u], indptr[u + 1]
        for v in indices[start:end]:
            if dist[v] == np.iinfo(np.int32).max:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def _undirected_edge_list(adj: csr_matrix) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return each undirected edge exactly once as two arrays (u,v) with u < v.
    Works for symmetric and non-symmetric adjacency (uses row lists).
    """
    indptr = adj.indptr
    indices = adj.indices
    assert adj.shape is not None
    N = adj.shape[0]
    uu, vv = [], []
    for u in range(N):
        nbrs = indices[indptr[u]:indptr[u + 1]]
        keep = nbrs > u  # ensure u < v
        if np.any(keep):
            vs = nbrs[keep]
            uu.extend([u] * len(vs))
            vv.extend(vs.tolist())
    return np.asarray(uu, dtype=np.int64), np.asarray(vv, dtype=np.int64)

def _edges_to_bin(u: np.ndarray, v: np.ndarray, N: int) -> csr_matrix:
    """
    Build a binarized CSR adjacency for one time bin from parallel endpoints u,v.
    Duplicates are coalesced to 1.
    """
    if u.size == 0:
        return csr_matrix((N, N), dtype=np.uint8)
    data = np.ones(u.size, dtype=np.uint8)
    A = csr_matrix((data, (u.astype(np.int32), v.astype(np.int32))), shape=(N, N), dtype=np.uint8)
    A.data[:] = 1
    return A

def _laplacian_mv(adj: csr_matrix, deg: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Compute (D - A) @ x without forming L explicitly (uses CSR SpMV).
    x may be shape (N,) or (N, K). Returns same shape as x.
    """
    y = adj @ x
    return deg[:, None] * x - y if x.ndim == 2 else deg * x - np.asarray(y).ravel()

def _normalized_laplacian(adj: csr_matrix) -> csr_matrix:
    """
    L_sym = I - D^{-1/2} A D^{-1/2}, eigenvalues in [0, 2].
    For isolated nodes, we define D^{-1/2}=0 (standard).
    """
    assert adj.shape is not None
    N = adj.shape[0]
    deg = _csr_degree(adj)
    with np.errstate(divide='ignore'):
        dmh = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    Dmh = diags(dmh, 0, format='csr')
    return (diags(np.ones(N), 0, format='csr') - (Dmh @ adj @ Dmh)).tocsr()

def _estimate_lmax_power(L: csr_matrix, iters: int = 40, seed: int = 0) -> float:
    """
    Power iteration estimate of largest eigenvalue (Rayleigh quotient).
    Returns at least a tiny floor to avoid division by zero.
    """
    assert L.shape is not None
    N = L.shape[0]
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(N)
    x /= np.linalg.norm(x) + 1e-12
    lam = 0.0
    for _ in range(max(1, iters)):
        y = L @ x
        lam = float(np.dot(x, y))
        yn = np.linalg.norm(y)
        if yn < 1e-12:
            break
        x = y / yn
    return max(lam, 1e-12)

def _ricker_pulse(t: int, t0: int, f: float) -> float:
    """
    Zero-mean Ricker pulse at discrete time t, centered at t0, with "frequency" f (steps^-1).
    """
    tau = float(t - t0)
    a = np.pi * f * tau
    return (1.0 - 2.0 * a * a) * np.exp(-a * a)

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

# =============================================================================
# Public: simulator registry (like graph builders)
# =============================================================================

SimulatorBins = List[csr_matrix]
SimulatorMeta = List[Dict[str, Any]]
SimulatorReturn = Tuple[SimulatorBins, np.ndarray, SimulatorMeta]

SIMULATORS: Dict[str, Callable[..., SimulatorReturn]] = {}


def _empty_states(N: int) -> np.ndarray:
    return np.zeros((0, N), dtype=np.float64)


def _empty_meta_list(t_bins: int) -> SimulatorMeta:
    return [{} for _ in range(t_bins)]


def run_simulator(kind: str, /, **kwargs) -> SimulatorReturn:
    """Dispatch to a registered graph-time simulator with a fixed return shape."""
    if kind not in SIMULATORS:
        raise ValueError(f"Unknown simulator '{kind}'. Available: {sorted(SIMULATORS)}")
    res = SIMULATORS[kind](**kwargs)
    if not isinstance(res, tuple) or len(res) != 3:
        raise TypeError(f"Simulator '{kind}' must return (bins, H, meta_list)")
    return res

# =============================================================================
# Simulator 1: Faucet-driven outward ring
# =============================================================================

def simulate_faucet_on_graph(
    adj: csr_matrix,
    t_bins: int,
    *,
    center_idx: int,
    faucet_period: int = 40,
    speed_hops_per_step: float = 1.0,
    sigma_hops: float = 1.05,
    p_max: float = 0.98,
    return_states: bool = False,
    mass: float = 0.1,
    gamma: float = 0.02,
    c: float = 1.0,
    dt: float = 1.0,
    faucet_kick: float = 2.0,
    nu: float = 0.02,
    ricker_cycles_per_period: float = 1.0,
    kill_margin: float = 1.0,
    kill_tau: float = 1.0,
    seed: int = 0,
    debug_every: int = 0,
) -> SimulatorReturn:
    adj = _as_square_csr(adj)
    rng = np.random.default_rng(seed)
    assert adj.shape is not None
    N = int(adj.shape[0])

    dist = _bfs_hops_csr(adj, center_idx)
    reachable = dist < np.iinfo(np.int32).max
    maxhop = int(dist[reachable].max()) if reachable.any() else 0

    und_u, und_v = _undirected_edge_list(adj)
    E = und_u.size

    d_u = dist[und_u]; d_v = dist[und_v]
    shell_edge = (np.abs(d_u - d_v) == 1)
    out_u = np.where(d_u < d_v, und_u, und_v).astype(np.int32)
    out_v = np.where(d_u < d_v, und_v, und_u).astype(np.int32)
    avg_d = 0.5 * (d_u + d_v).astype(np.float64)

    Ls: Optional[NDArray[np.float64]] = None
    h: Optional[NDArray[np.float64]] = None
    v_half: Optional[NDArray[np.float64]] = None
    pulses = np.arange(0, t_bins, max(1, faucet_period), dtype=int)
    def global_fade(r: float) -> float:
        over = max(0.0, r - (maxhop - kill_margin))
        return float(np.exp(- (over / max(1e-8, kill_tau)) ** 2))

    H_list: List[NDArray[np.float64]] = []
    if return_states:
        h = np.zeros(N, dtype=np.float64)
        v_half = np.zeros(N, dtype=np.float64)

        # normalized L scaled to ~spectral radius 1
        L = _normalized_laplacian(adj)
        lam_max = _estimate_lmax_power(L)
        Ls = (L * (1.0 / lam_max)).tocsr()
        sigma = float(c) * float(dt)
        if sigma > 1.5:
            print(f"[warn] c*dt={sigma:.3f} is high; consider c*dt ≤ 1.5 for stability")

    csr_bins: List[csr_matrix] = []

    for t in range(t_bins):
        if pulses.size and t >= pulses[0]:
            act = pulses[pulses <= t]
            radii = speed_hops_per_step * (t - act).astype(np.float64)
        else:
            radii = np.empty((0,), dtype=np.float64)

        p = np.zeros(E, dtype=np.float64)
        if radii.size and shell_edge.any():
            idx = np.nonzero(shell_edge)[0]
            ad = avg_d[idx]
            for r in radii:
                band = np.exp(-0.5 * ((ad - r) / max(1e-8, sigma_hops)) ** 2)
                p_r = p_max * band * global_fade(r)
                p[idx] = np.maximum(p[idx], p_r)

        active = rng.random(E) < p
        if not np.any(active):
            csr_bins.append(csr_matrix((N, N), dtype=np.uint8))
        else:
            uu = out_u[active]; vv = out_v[active]
            csr_bins.append(_edges_to_bin(uu, vv, N))

        if return_states:
            s = np.zeros(N, dtype=np.float64)
            f = float(ricker_cycles_per_period) / max(1.0, float(faucet_period))
            g_t = 0.0
            if pulses.size:
                w = int(4 * faucet_period)
                for t0 in pulses:
                    dτ = t - t0
                    if -w <= dτ <= w:
                        g_t += _ricker_pulse(t, int(t0), f)
            s[center_idx] = faucet_kick * g_t

            assert Ls is not None and h is not None and v_half is not None
            a = -(float(c) * float(c)) * (Ls @ h)
            a += -(float(mass) * float(mass)) * h
            a += -float(gamma) * v_half
            a += -float(nu) * (Ls @ v_half)
            a += s
            v_half = v_half + float(dt) * a
            h = h + float(dt) * v_half

            if debug_every and (t % debug_every == 0):
                e_lap = np.linalg.norm(Ls @ h)
                print(f"[t={t:04d}] ||h||={np.linalg.norm(h):.3g}  ||v||={np.linalg.norm(v_half):.3g}  "
                      f"c^2||Lh||={(float(c) * float(c)) * e_lap:.3g}")

            H_list.append(h.copy())

    H = np.stack(H_list, axis=0) if return_states else _empty_states(N)
    meta_list = _empty_meta_list(t_bins)
    return csr_bins, H, meta_list

SIMULATORS["faucet"] = simulate_faucet_on_graph

# =============================================================================
# Simulator 2: Simple damped "waves-on-graph" with logistic activation
# =============================================================================

def simulate_waves_on_graph(
    adj: csr_matrix,
    t_bins: int,
    *,
    c2: float = 0.4,
    m2: float = 0.2,
    gamma: float = 0.3,
    noise: float = 0.02,
    scale: float = 3.0,
    grad_threshold: float = 0.2,
    dt: float | None = None,
    seed: int = 0,
) -> SimulatorReturn:
    adj = _as_square_csr(adj)
    assert adj.shape is not None
    N = adj.shape[0]
    rng = np.random.default_rng(seed)
    deg = _csr_degree(adj)

    if dt is None:
        dt = 0.25

    und_u, und_v = _undirected_edge_list(adj)
    E = und_u.size

    h: NDArray[np.float64] = np.asarray(rng.normal(0.0, 0.1, size=N), dtype=np.float64)
    v: NDArray[np.float64] = np.zeros(N, dtype=np.float64)

    csr_bins: List[csr_matrix] = []

    for _ in range(t_bins):
        lap_h = _laplacian_mv(adj, deg, h)
        v += dt * (c2 * lap_h - m2 * h - gamma * v) + noise * rng.normal(size=N)
        h += dt * v

        if E == 0:
            csr_bins.append(csr_matrix((N, N), dtype=np.uint8))
            continue

        grad = np.abs(h[und_u] - h[und_v])
        p = 1.0 / (1.0 + np.exp(-scale * (grad - grad_threshold))) * 1e-2
        active = rng.random(E) < p

        if not np.any(active):
            csr_bins.append(csr_matrix((N, N), dtype=np.uint8))
            continue

        au = und_u[active].astype(np.int32)
        av = und_v[active].astype(np.int32)
        flips = rng.random(au.size) < 0.5
        u = np.where(flips, au, av)
        v2 = np.where(flips, av, au)

        csr_bins.append(_edges_to_bin(u, v2, N))

    H = _empty_states(N)
    meta_list = _empty_meta_list(t_bins)
    return csr_bins, H, meta_list

SIMULATORS["waves"] = simulate_waves_on_graph


ForcingKind = Literal["none", "impulse", "ricker_train", "sin", "chirp", "multi_sin", "moving_ricker"]
ReactionKind = Literal["none", "allen_cahn", "fisher_kpp"]

def _reaction(x: np.ndarray, kind: ReactionKind, *, ac_mu: float = 1.0, fk_r: float = 1.0) -> np.ndarray:
    if kind == "none":
        return np.zeros_like(x)
    if kind == "allen_cahn":
        # u_t = ... + mu*(u - u^3)
        return ac_mu * (x - x**3)
    if kind == "fisher_kpp":
        # u_t = ... + r*u*(1-u)
        return fk_r * x * (1.0 - x)
    raise ValueError(f"Unknown reaction kind: {kind}")

def _make_forcing_fn(
    kind: ForcingKind,
    *,
    N: int,
    center_idx: int = 0,
    # generic amplitude knobs
    amp: float = 1.0,
    # impulse / ricker
    t0: int = 10,
    ricker_f: float = 1.0/40.0,   # cycles per step
    period: int = 40,
    # sinusoid / chirp
    f0: float = 1.0/80.0,
    f1: float = 1.0/10.0,
    phase: float = 0.0,
    # multi-source
    centers: Optional[Sequence[int]] = None,
    phases: Optional[Sequence[float]] = None,
    # moving source
    move_mode: Literal["random_walk","path"] = "random_walk",
    path: Optional[Sequence[int]] = None,
    stay_prob: float = 0.25,
    seed: int = 0,
    adj: Optional[csr_matrix] = None,
) -> Callable[[int], Tuple[np.ndarray, Dict]]:
    """
    Returns forcing(t) -> (s: (N,), meta: dict) so callers can record the “source location”.
    """
    rng = np.random.default_rng(seed)

    centers_list = list(centers) if centers is not None else [int(center_idx)]
    if phases is None:
        phases_list = [0.0 for _ in centers_list]
    else:
        phases_list = list(phases)
        if len(phases_list) != len(centers_list):
            raise ValueError("phases must match centers length")

    # moving source state
    if kind == "moving_ricker":
        if move_mode == "path":
            if path is None or len(path) == 0:
                raise ValueError("moving_ricker with move_mode='path' requires non-empty path")
            path = [int(x) for x in path]
            def source_at(t: int) -> int:
                return path[min(t, len(path)-1)]
        else:
            if adj is None:
                raise ValueError("moving_ricker with random_walk requires adj")
            adj_csr = _as_square_csr(adj)
            cur = int(center_idx)
            indptr, indices = adj_csr.indptr, adj_csr.indices
            def step_rw() -> int:
                nonlocal cur
                if rng.random() < stay_prob:
                    return cur
                nbrs = indices[indptr[cur]:indptr[cur+1]]
                if nbrs.size == 0:
                    return cur
                cur = int(rng.choice(nbrs))
                return cur
            def source_at(t: int) -> int:
                return step_rw()
    else:
        def source_at(t: int) -> int:
            return int(center_idx)

    def forcing(t: int) -> Tuple[np.ndarray, Dict]:
        s = np.zeros(N, dtype=np.float64)
        meta: Dict = {}

        if kind == "none":
            return s, meta

        if kind == "impulse":
            # one Ricker at t0
            g = _ricker_pulse(t, int(t0), float(ricker_f))
            s[int(center_idx)] = float(amp) * g
            meta["source"] = int(center_idx)
            return s, meta

        if kind == "ricker_train":
            # Ricker pulses every `period`
            g = 0.0
            for tt in range(0, t+1, max(1, int(period))):
                g += _ricker_pulse(t, int(tt), float(ricker_f))
            s[int(center_idx)] = float(amp) * g
            meta["source"] = int(center_idx)
            return s, meta

        if kind == "sin":
            g = np.sin(2*np.pi*float(f0)*t + float(phase))
            s[int(center_idx)] = float(amp) * g
            meta["source"] = int(center_idx)
            return s, meta

        if kind == "chirp":
            # linear chirp in cycles/step: f(t)=f0 + (f1-f0)*t/(T-1)
            # phase = 2π * sum f
            # use continuous approx: phase(t)=2π*(f0 t + 0.5 (f1-f0)t^2/(T-1))
            # (good enough for visuals)
            # NOTE: we don’t know T here, so assume “t_bins-like” by using f1 ramp over 400 if you don’t override.
            # You can pass f0,f1 tuned for your t_bins.
            Tnom = 400.0
            ft = float(f0) + (float(f1) - float(f0)) * (t / max(1.0, Tnom-1.0))
            ph = 2*np.pi*(float(f0)*t + 0.5*(float(f1)-float(f0))*(t*t)/max(1.0, Tnom-1.0)) + float(phase)
            g = np.sin(ph)
            s[int(center_idx)] = float(amp) * g
            meta["source"] = int(center_idx)
            meta["f_t"] = ft
            return s, meta

        if kind == "multi_sin":
            # multiple fixed sources (optionally phased)
            for c, ph in zip(centers_list, phases_list):
                g = np.sin(2*np.pi*float(f0)*t + float(ph))
                s[int(c)] += float(amp) * g
            meta["sources"] = [int(c) for c in centers_list]
            return s, meta

        if kind == "moving_ricker":
            src = int(source_at(t))
            g = _ricker_pulse(t, int(t0), float(ricker_f))
            s[src] = float(amp) * g
            meta["source"] = src
            return s, meta

        raise ValueError(f"Unknown forcing kind: {kind}")

    return forcing

def _edges_from_field(
    und_u: np.ndarray,
    und_v: np.ndarray,
    x: np.ndarray,
    rng: np.random.Generator,
    *,
    mode: Literal["grad", "flux"] = "grad",
    scale: float = 6.0,
    thresh: float = 0.2,
    sparsity: float = 1e-2,
    min_edges: int = 0,          # NEW: ensure at least this many directed edges
) -> csr_matrix:
    """
    Turn a node field x into sparse directed events on edges.
    - grad: |x_u - x_v| drives probability
    - flux:  |x_u - x_v| drives probability; direction is downhill
    """
    N = x.size
    E = und_u.size
    if E == 0:
        return csr_matrix((N, N), dtype=np.uint8)

    min_edges = int(min_edges)
    min_edges = max(0, min(min_edges, E))

    if mode == "grad":
        g = np.abs(x[und_u] - x[und_v])
        p = _sigmoid(scale * (g - thresh)) * float(sparsity)
        active = rng.random(E) < p

        if not np.any(active) and min_edges > 0:
            # fallback: pick top edges by p
            idx = np.argpartition(p, -min_edges)[-min_edges:]
            active = np.zeros(E, dtype=bool)
            active[idx] = True

        if not np.any(active):
            return csr_matrix((N, N), dtype=np.uint8)

        au = und_u[active].astype(np.int32)
        av = und_v[active].astype(np.int32)
        flips = rng.random(au.size) < 0.5
        u = np.where(flips, au, av)
        v = np.where(flips, av, au)
        return _edges_to_bin(u, v, N)

    if mode == "flux":
        duv = x[und_u] - x[und_v]
        mag = np.abs(duv)
        p = _sigmoid(scale * (mag - thresh)) * float(sparsity)
        active = rng.random(E) < p

        if not np.any(active) and min_edges > 0:
            idx = np.argpartition(p, -min_edges)[-min_edges:]
            active = np.zeros(E, dtype=bool)
            active[idx] = True

        if not np.any(active):
            return csr_matrix((N, N), dtype=np.uint8)

        a_u = und_u[active].astype(np.int32)
        a_v = und_v[active].astype(np.int32)
        duv_a = duv[active]
        u = np.where(duv_a >= 0.0, a_u, a_v)
        v = np.where(duv_a >= 0.0, a_v, a_u)
        return _edges_to_bin(u, v, N)

    raise ValueError("edge mode must be 'grad' or 'flux'")

def simulate_field_dynamics(
    adj: csr_matrix,
    t_bins: int,
    *,
    # wave / diffusion / damping core
    dyn: Literal["wave", "diffusion"] = "wave",
    use_normalized_laplacian: bool = True,
    c2: float = 0.8,              # coupling strength (wave stiffness or diffusion rate)
    m2: float = 0.05,             # mass/restoring (wave only; can be 0)
    gamma: float = 0.15,          # damping (wave: on velocity; diffusion: on state)
    nu: float = 0.0,              # optional viscosity: +nu * (-L v) or diffusion smoothing
    dt: float = 0.25,
    noise: float = 0.0,           # Gaussian drive
    # forcing + reaction
    forcing_kind: ForcingKind = "none",
    forcing_kwargs: Optional[Dict] = None,
    reaction_kind: ReactionKind = "none",
    reaction_kwargs: Optional[Dict] = None,
    # events from field
    edge_from: Literal["grad","flux"] = "grad",
    edge_scale: float = 6.0,
    edge_thresh: float = 0.2,
    edge_sparsity: float = 1e-2,
    # init
    init_scale: float = 0.1,
    return_states: bool = True,
    seed: int = 0,
) -> SimulatorReturn:
    """
    Generic node-field simulator that can cover:
      - impulse / chirp / multi-source / moving source (via forcing_kind)
      - Allen–Cahn / Fisher–KPP (via reaction_kind)
      - wave-like or diffusion-like integration
    Emits sparse directed edge events derived from the field each step.
    Returns: (event_bins, H[T,N], forcing_meta_list[T]).
    """
    adj = _as_square_csr(adj)
    rng = np.random.default_rng(seed)
    assert adj.shape is not None
    N = adj.shape[0]
    deg = _csr_degree(adj)

    if forcing_kwargs is None:
        forcing_kwargs = {}
    if reaction_kwargs is None:
        reaction_kwargs = {}

    # Laplacian operator
    if use_normalized_laplacian:
        L = _normalized_laplacian(adj)
        lam_max = _estimate_lmax_power(L, seed=seed)
        Lop = L * (1.0 / lam_max)
        # for diffusion we want something like -L (since L is PSD)
        def apply_L(x: np.ndarray) -> np.ndarray:
            return (Lop @ x)
    else:
        def apply_L(x: np.ndarray) -> np.ndarray:
            return _laplacian_mv(adj, deg, x)

    und_u, und_v = _undirected_edge_list(adj)

    forcing_fn = _make_forcing_fn(
        forcing_kind,
        N=N,
        adj=adj,
        seed=seed,
        **forcing_kwargs,
    )

    # init
    x: NDArray[np.float64] = np.asarray(rng.normal(0.0, float(init_scale), size=N), dtype=np.float64)
    v: NDArray[np.float64] = np.zeros(N, dtype=np.float64)  # only used for wave

    bins: List[csr_matrix] = []
    H_list: List[np.ndarray] = []
    meta_list: SimulatorMeta = []

    for t in range(t_bins):
        s, meta = forcing_fn(t)
        meta_list.append(meta)

        r = _reaction(x, reaction_kind, **reaction_kwargs)

        if dyn == "wave":
            # x'' = c2*(-L x) - m2*x - gamma*v + nu*(-L v) + r + s + noise
            ax = -float(c2) * apply_L(x) - float(m2) * x - float(gamma) * v
            if nu != 0.0:
                ax += -float(nu) * apply_L(v)
            ax += r + s
            if noise != 0.0:
                ax += float(noise) * rng.normal(size=N)
            v = v + float(dt) * ax
            x = x + float(dt) * v

        elif dyn == "diffusion":
            # x' = -c2*L x - gamma*x + r + s + noise  (note: L is PSD so -L diffuses)
            dx = -float(c2) * apply_L(x) - float(gamma) * x + r + s
            if nu != 0.0:
                # extra smoothing term (same form)
                dx += -float(nu) * apply_L(x)
            if noise != 0.0:
                dx += float(noise) * rng.normal(size=N)
            x = x + float(dt) * dx
        else:
            raise ValueError("dyn must be 'wave' or 'diffusion'")

        bins.append(_edges_from_field(
            und_u, und_v, x, rng,
            mode="flux",
            scale=edge_scale,
            thresh=edge_thresh,
            sparsity=edge_sparsity,
            min_edges=max(1, int(0.02 * N)),  # let's try 2% of N as a starting point
        ))
        if return_states:
            H_list.append(x.copy())

    H = np.stack(H_list, axis=0) if return_states else _empty_states(N)
    return bins, H, meta_list

# Convenience names (register a few common “do the thing” presets)
def _field_preset(**preset_kw):
    def _sim(**kw):
        return simulate_field_dynamics(**{**preset_kw, **kw})
    return _sim

# 1) Impulse
SIMULATORS["impulse"] = _field_preset(
    dyn="wave",
    forcing_kind="impulse",
    forcing_kwargs={"center_idx": 0, "t0": 10, "amp": 2.0, "ricker_f": 1.0/25.0},
    reaction_kind="none",
)

# 2) Chirp
SIMULATORS["chirp"] = _field_preset(
    dyn="wave",
    forcing_kind="chirp",
    forcing_kwargs={"center_idx": 0, "amp": 1.0, "f0": 1.0/120.0, "f1": 1.0/12.0},
    reaction_kind="none",
)

# 3) Multi-source sinusoid
SIMULATORS["multi_source"] = _field_preset(
    dyn="wave",
    forcing_kind="multi_sin",
    forcing_kwargs={"centers": [0], "amp": 1.0, "f0": 1.0/50.0, "phases": [0.0]},
    reaction_kind="none",
)

# 4) Moving source (random walk) with Ricker
SIMULATORS["moving_source"] = _field_preset(
    dyn="wave",
    forcing_kind="moving_ricker",
    forcing_kwargs={"center_idx": 0, "t0": 10, "amp": 2.0, "ricker_f": 1.0/25.0, "move_mode": "random_walk", "stay_prob": 0.25},
    reaction_kind="none",
)

# 5) Allen–Cahn (reaction–diffusion)
SIMULATORS["allen_cahn"] = _field_preset(
    dyn="diffusion",
    forcing_kind="none",
    reaction_kind="allen_cahn",
    reaction_kwargs={"ac_mu": 1.0},
    c2=0.8, gamma=0.02, dt=0.25,
)

# 6) Fisher–KPP (reaction–diffusion / traveling fronts)
SIMULATORS["fisher_kpp"] = _field_preset(
    dyn="diffusion",
    forcing_kind="none",
    reaction_kind="fisher_kpp",
    reaction_kwargs={"fk_r": 0.8},
    c2=0.7, gamma=0.02, dt=0.25,
)

# =============================================================================
# Discrete contagion / threshold / voter
# =============================================================================

def simulate_sis(
    adj: csr_matrix,
    t_bins: int,
    *,
    beta: float = 0.08,         # infection probability per infected neighbor per step (approx)
    mu: float = 0.05,           # recovery probability per step
    init_infected: float = 0.02,
    seed: int = 0,
    return_states: bool = True,
) -> SimulatorReturn:
    """
    SIS on nodes. Edge events are transmissions u->v when u infected and v becomes infected.
    State H is infection indicator (0/1).
    """
    adj = _as_square_csr(adj)
    rng = np.random.default_rng(seed)
    assert adj.shape is not None
    N = adj.shape[0]
    indptr, indices = adj.indptr, adj.indices

    infected: NDArray[np.bool_] = np.asarray(rng.random(N) < float(init_infected), dtype=bool)
    bins: List[csr_matrix] = []
    H_list: List[np.ndarray] = []

    for _ in range(t_bins):
        # recoveries
        rec = infected & np.asarray(rng.random(N) < float(mu), dtype=bool)
        infected[rec] = False

        # infections driven by infected neighbors
        new_inf_u: List[int] = []
        new_inf_v: List[int] = []

        # For each susceptible v, compute infection chance from infected neighbors:
        # p = 1 - (1-beta)^(k_inf)
        # (fast-ish in Python for moderate N; good enough for these demos)
        sus = np.where(~infected)[0]
        for v in sus:
            nbrs = indices[indptr[v]:indptr[v+1]]
            if nbrs.size == 0:
                continue
            inf_nbrs = nbrs[infected[nbrs]]
            k = inf_nbrs.size
            if k == 0:
                continue
            p = 1.0 - (1.0 - float(beta))**k
            if rng.random() < p:
                infected[v] = True
                # pick one “causal” infected neighbor for an event edge
                u = int(rng.choice(inf_nbrs))
                new_inf_u.append(u)
                new_inf_v.append(int(v))

        bins.append(_edges_to_bin(np.asarray(new_inf_u, np.int32), np.asarray(new_inf_v, np.int32), N))
        if return_states:
            H_list.append(infected.astype(np.float64))

    H = np.stack(H_list, axis=0) if return_states else _empty_states(N)
    meta_list = _empty_meta_list(t_bins)
    return bins, H, meta_list

SIMULATORS["sis"] = simulate_sis

def simulate_threshold(
    adj: csr_matrix,
    t_bins: int,
    *,
    theta: float = 0.25,        # fraction of neighbors required to activate
    mu_off: float = 0.01,       # optional deactivation prob (keeps motion)
    init_on: float = 0.02,
    seed: int = 0,
    return_states: bool = True,
) -> SimulatorReturn:
    """
    Complex contagion / threshold cascade.
    Edge events are (u->v) when v activates and u is one of the active neighbors (picked).
    """
    adj = _as_square_csr(adj)
    rng = np.random.default_rng(seed)
    assert adj.shape is not None
    N = adj.shape[0]
    deg = _csr_degree(adj)
    indptr, indices = adj.indptr, adj.indices

    on: NDArray[np.bool_] = np.asarray(rng.random(N) < float(init_on), dtype=bool)
    bins: List[csr_matrix] = []
    H_list: List[np.ndarray] = []

    for _ in range(t_bins):
        # optional random off
        offmask = on & np.asarray(rng.random(N) < float(mu_off), dtype=bool)
        on[offmask] = False

        new_u: List[int] = []
        new_v: List[int] = []

        # compute fraction active among neighbors
        cand = np.where(~on)[0]
        for v in cand:
            nbrs = indices[indptr[v]:indptr[v+1]]
            if nbrs.size == 0:
                continue
            frac = float(on[nbrs].sum()) / max(1.0, float(nbrs.size))
            if frac >= float(theta):
                on[v] = True
                act_nbrs = nbrs[on[nbrs]]
                if act_nbrs.size:
                    u = int(rng.choice(act_nbrs))
                    new_u.append(u); new_v.append(int(v))

        bins.append(_edges_to_bin(np.asarray(new_u, np.int32), np.asarray(new_v, np.int32), N))
        if return_states:
            H_list.append(on.astype(np.float64))

    H = np.stack(H_list, axis=0) if return_states else _empty_states(N)
    meta_list = _empty_meta_list(t_bins)
    return bins, H, meta_list

SIMULATORS["threshold"] = simulate_threshold

def simulate_voter(
    adj: csr_matrix,
    t_bins: int,
    *,
    init_p: float = 0.5,
    seed: int = 0,
    return_states: bool = True,
) -> SimulatorReturn:
    """
    Voter dynamics. Each step, each node copies a random neighbor's state (if any).
    Edge events record the copy edge (u->v).
    """
    adj = _as_square_csr(adj)
    rng = np.random.default_rng(seed)
    assert adj.shape is not None
    N = adj.shape[0]
    indptr, indices = adj.indptr, adj.indices

    x: NDArray[np.int8] = np.asarray((rng.random(N) < float(init_p)), dtype=np.int8)
    bins: List[csr_matrix] = []
    H_list: List[np.ndarray] = []

    for _ in range(t_bins):
        uu: List[int] = []
        vv: List[int] = []
        x_new = x.copy()
        for v in range(N):
            nbrs = indices[indptr[v]:indptr[v+1]]
            if nbrs.size == 0:
                continue
            u = int(rng.choice(nbrs))
            if x_new[v] != x[u]:
                x_new[v] = x[u]
                uu.append(u); vv.append(v)
        x = x_new
        bins.append(_edges_to_bin(np.asarray(uu, np.int32), np.asarray(vv, np.int32), N))
        if return_states:
            H_list.append(x.astype(np.float64)*2.0 - 1.0)  # map {0,1}->{-1,+1}

    H = np.stack(H_list, axis=0) if return_states else _empty_states(N)
    meta_list = _empty_meta_list(t_bins)
    return bins, H, meta_list

SIMULATORS["voter"] = simulate_voter

# =============================================================================
# Transport / flow style dynamics (conservation-ish)
# =============================================================================

def simulate_transport(
    adj: csr_matrix,
    t_bins: int,
    *,
    alpha: float = 0.25,         # step size (diffusive)
    noise: float = 0.0,
    seed: int = 0,
    init_scale: float = 1.0,
    edge_scale: float = 6.0,
    edge_thresh: float = 0.1,
    edge_sparsity: float = 1e-2,
    return_states: bool = True,
) -> SimulatorReturn:
    """
    Simple mass transport: x <- x - alpha*Lx  (diffusion-like; approximately conserved without damping).
    Edge events derived from flux (downhill direction).
    """
    adj = _as_square_csr(adj)
    rng = np.random.default_rng(seed)
    assert adj.shape is not None
    N = adj.shape[0]
    deg = _csr_degree(adj)
    und_u, und_v = _undirected_edge_list(adj)

    x: NDArray[np.float64] = np.asarray(rng.normal(0.0, float(init_scale), size=N), dtype=np.float64)

    bins: List[csr_matrix] = []
    H_list: List[np.ndarray] = []

    for _ in range(t_bins):
        Lx = _laplacian_mv(adj, deg, x)
        x = x - float(alpha) * Lx
        if noise != 0.0:
            x += float(noise) * rng.normal(size=N)

        bins.append(_edges_from_field(
            und_u, und_v, x, rng,
            mode="flux",
            scale=edge_scale,
            thresh=edge_thresh,
            sparsity=edge_sparsity,
        ))
        if return_states:
            H_list.append(x.copy())

    H = np.stack(H_list, axis=0) if return_states else _empty_states(N)
    meta_list = _empty_meta_list(t_bins)
    return bins, H, meta_list

SIMULATORS["transport"] = simulate_transport

# =============================================================================
# Edge Hawkes-like bursts (point-process native)
# =============================================================================

def simulate_hawkes_edges(
    adj: csr_matrix,
    t_bins: int,
    *,
    base_rate: float = 1e-4,
    alpha: float = 0.8,       # self-excitation amount per event
    beta: float = 0.25,       # decay per step (lambda <- (1-beta)*lambda + ...)
    neighbor_coupling: float = 0.15,  # excitation spreads to edges sharing a node
    seed: int = 0,
) -> SimulatorReturn:
    """
    Very simple discrete-time Hawkes on *undirected* edges, sampled each bin.
    - Each undirected edge e has intensity lam_e.
    - If e fires, lam_e increases by alpha and neighboring edges get +neighbor_coupling*alpha.
    Emits directed events by random orientation.
    """
    adj = _as_square_csr(adj)
    rng = np.random.default_rng(seed)
    assert adj.shape is not None
    N = adj.shape[0]
    und_u, und_v = _undirected_edge_list(adj)
    E = und_u.size

    if E == 0:
        bins = [csr_matrix((N, N), dtype=np.uint8) for _ in range(t_bins)]
        return bins, _empty_states(N), _empty_meta_list(t_bins)

    # build incidence: edges incident to node
    incident: List[List[int]] = [[] for _ in range(N)]
    for e in range(E):
        incident[int(und_u[e])].append(e)
        incident[int(und_v[e])].append(e)

    lam = np.full(E, float(base_rate), dtype=np.float64)
    bins: List[csr_matrix] = []

    for _ in range(t_bins):
        # sample events per edge
        p = np.clip(lam, 0.0, 0.25)  # cap for sanity; for visuals only
        fire = rng.random(E) < p
        fired_idx = np.where(fire)[0]

        # build bin
        if fired_idx.size == 0:
            bins.append(csr_matrix((N, N), dtype=np.uint8))
        else:
            a_u = und_u[fired_idx].astype(np.int32)
            a_v = und_v[fired_idx].astype(np.int32)
            flips = rng.random(fired_idx.size) < 0.5
            u = np.where(flips, a_u, a_v)
            v = np.where(flips, a_v, a_u)
            bins.append(_edges_to_bin(u, v, N))

        # decay + excite
        lam *= (1.0 - float(beta))
        if fired_idx.size:
            for e in fired_idx.tolist():
                lam[e] += float(alpha)
                u0 = int(und_u[e]); v0 = int(und_v[e])
                # spread to incident edges (excluding itself)
                for ee in incident[u0]:
                    if ee != e:
                        lam[ee] += float(neighbor_coupling) * float(alpha)
                for ee in incident[v0]:
                    if ee != e:
                        lam[ee] += float(neighbor_coupling) * float(alpha)

    return bins, _empty_states(N), _empty_meta_list(t_bins)

SIMULATORS["hawkes_edges"] = simulate_hawkes_edges
