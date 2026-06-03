from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Literal, Optional, Sequence, Tuple, cast

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, diags

from interactiondynamics.core.events import EventBatch
from interactiondynamics.data.interfaces import DataSpec, EdgeTargetBatch, EventStreamDataset

GraphKind = Literal["grid", "small_world", "tree"]
DynamicsKind = Literal["wave", "wave_pulse", "sis", "sirs"]


@dataclass
class PhysicalDatasetConfig:
    root_name: str = "physical"
    graph_kind: GraphKind = "grid"
    dynamics_kind: DynamicsKind = "wave"
    num_bins: int = 120
    split_fracs: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    device: Optional[torch.device] = None
    seed: int = 0

    grid_m: int = 24
    grid_n: int = 24

    small_world_n: Optional[int] = None
    small_world_k: int = 8
    small_world_beta: float = 0.12

    tree_levels: Optional[int] = None
    tree_branching: int = 3

    wave_dt: float = 0.25
    wave_c: float = 3.0
    wave_gamma: float = 0.02
    wave_mass: float = 0.005
    wave_faucet_period: float = 6.0
    wave_faucet_amp: float = 1.0
    wave_faucet_width: float = 0.15
    wave_edge_threshold: float = 0.05

    sis_beta: float = 0.08
    sis_mu: float = 0.05
    sis_init_infected: float = 0.02

    sirs_beta: float = 0.08
    sirs_mu: float = 0.05
    sirs_rho: float = 0.03
    sirs_init_infected: float = 0.02


def _symmetrize(adj: csr_matrix) -> csr_matrix:
    out = (adj + adj.T).tocsr()
    out.setdiag(0)
    out.eliminate_zeros()
    return out


def _csr_from_edges(u: np.ndarray, v: np.ndarray, w: np.ndarray, n_nodes: int) -> csr_matrix:
    return csr_matrix((w.astype(np.float32, copy=False), (u, v)), shape=(n_nodes, n_nodes))


def _grid_coords(m: int, n: int) -> np.ndarray:
    ii, jj = np.indices((m, n))
    return np.stack([jj.ravel(), ii.ravel()], axis=1).astype(np.float32)


def _build_grid(m: int, n: int) -> tuple[csr_matrix, Dict]:
    ii, jj = np.indices((m, n))
    src = (ii * n + jj).astype(np.int64)

    right = jj + 1 < n
    down = ii + 1 < m

    u = np.concatenate([src[right].ravel(), src[down].ravel()])
    v = np.concatenate([(src[right] + 1).ravel(), (src[down] + n).ravel()])
    w = np.ones(u.size, dtype=np.float32)

    adj = _symmetrize(_csr_from_edges(u, v, w, m * n))
    return adj, {"kind": "grid", "shape": (m, n), "coords2d": _grid_coords(m, n)}


def _build_small_world(n_nodes: int, k: int, beta: float, seed: int) -> tuple[csr_matrix, Dict]:
    if not (0 < k < n_nodes and k % 2 == 0):
        raise ValueError("small_world requires even k with 0 < k < n")

    rng = np.random.default_rng(seed)
    nodes = np.arange(n_nodes, dtype=np.int64)
    u_parts: list[np.ndarray] = []
    v_parts: list[np.ndarray] = []
    for stride in range(1, k // 2 + 1):
        u_parts.append(nodes)
        v_parts.append((nodes + stride) % n_nodes)
    u = np.concatenate(u_parts)
    v = np.concatenate(v_parts)
    m_edges = int(u.size)

    def undirected_pair(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    pairs = {undirected_pair(int(u[idx]), int(v[idx])) for idx in range(m_edges)}
    adjacency_rows = [set() for _ in range(n_nodes)]
    for src, dst in zip(u, v):
        adjacency_rows[int(src)].add(int(dst))

    for idx in range(m_edges):
        if rng.random() >= beta:
            continue
        src = int(u[idx])
        old_dst = int(v[idx])
        pairs.discard(undirected_pair(src, old_dst))
        adjacency_rows[src].discard(old_dst)

        forbid = set(adjacency_rows[src])
        forbid.add(src)
        candidate = int(rng.integers(0, n_nodes))
        tries = 0
        while candidate in forbid or undirected_pair(src, candidate) in pairs:
            candidate = int(rng.integers(0, n_nodes))
            tries += 1
            if tries > 10 * n_nodes:
                for fallback in range(n_nodes):
                    if fallback not in forbid and undirected_pair(src, fallback) not in pairs:
                        candidate = fallback
                        break
                else:
                    candidate = old_dst
                    break

        v[idx] = candidate
        adjacency_rows[src].add(candidate)
        pairs.add(undirected_pair(src, candidate))

    adj = _symmetrize(_csr_from_edges(u, v, np.ones(m_edges, dtype=np.float32), n_nodes))
    theta = 2.0 * np.pi * np.arange(n_nodes, dtype=np.float32) / max(1, n_nodes)
    coords2d = np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)
    return adj, {"kind": "small_world", "coords2d": coords2d}


def _build_tree(levels: int, branching: int) -> tuple[csr_matrix, Dict]:
    if levels < 1 or branching < 2:
        raise ValueError("tree requires levels >= 1 and branching >= 2")
    n_nodes = (branching**levels - 1) // (branching - 1)

    u_list: list[int] = []
    v_list: list[int] = []
    for parent in range((branching**(levels - 1) - 1) // (branching - 1)):
        first_child = parent * branching + 1
        for offset in range(branching):
            child = first_child + offset
            if child < n_nodes:
                u_list.extend([parent, child])
                v_list.extend([child, parent])

    u = np.asarray(u_list, dtype=np.int64)
    v = np.asarray(v_list, dtype=np.int64)
    adj = _csr_from_edges(u, v, np.ones(u.size, dtype=np.float32), n_nodes).tocsr()

    coords2d = np.zeros((n_nodes, 2), dtype=np.float32)
    idx = 0
    for level in range(levels):
        n_level = branching**level
        angles = 2.0 * np.pi * np.linspace(0.0, 1.0, n_level, endpoint=False, dtype=np.float32)
        for angle in angles:
            if idx >= n_nodes:
                break
            coords2d[idx] = np.array([level * np.cos(angle), level * np.sin(angle)], dtype=np.float32)
            idx += 1
    return adj, {"kind": "tree", "levels": levels, "branching": branching, "coords2d": coords2d}


def _graph_laplacian(adj: csr_matrix) -> csr_matrix:
    deg = np.asarray(adj.sum(axis=1)).ravel().astype(np.float64)
    return cast(csr_matrix, (diags(deg, format="csr") - adj).tocsr())


def _degree(adj: csr_matrix) -> np.ndarray:
    return np.asarray(adj.sum(axis=1)).ravel().astype(np.float64)


def _normalized_laplacian(adj: csr_matrix) -> csr_matrix:
    n_nodes = int(adj.shape[0])
    deg = _degree(adj)
    with np.errstate(divide="ignore"):
        inv_sqrt_deg = np.where(deg > 0.0, 1.0 / np.sqrt(deg), 0.0)
    dmh = diags(inv_sqrt_deg, 0, format="csr")
    ident = diags(np.ones(n_nodes, dtype=np.float64), 0, format="csr")
    return cast(csr_matrix, (ident - (dmh @ adj @ dmh)).tocsr())


def _estimate_lmax_power(laplacian: csr_matrix, seed: int, iters: int = 40) -> float:
    n_nodes = int(laplacian.shape[0])
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n_nodes)
    x /= np.linalg.norm(x) + 1e-12
    lam = 0.0
    for _ in range(max(1, iters)):
        y = np.asarray(laplacian @ x).ravel()
        lam = float(np.dot(x, y))
        y_norm = np.linalg.norm(y)
        if y_norm < 1e-12:
            break
        x = y / y_norm
    return max(lam, 1e-12)


def _undirected_edge_list(adj: csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    indptr = adj.indptr
    indices = adj.indices
    n_nodes = int(adj.shape[0])
    uu: list[int] = []
    vv: list[int] = []
    for src in range(n_nodes):
        nbrs = indices[indptr[src]:indptr[src + 1]]
        keep = nbrs > src
        if np.any(keep):
            dst = nbrs[keep]
            uu.extend([src] * int(dst.size))
            vv.extend(dst.tolist())
    return np.asarray(uu, dtype=np.int64), np.asarray(vv, dtype=np.int64)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _directed_edge_values_from_state(
    state: NDArray[np.float64],
    src: np.ndarray,
    dst: np.ndarray,
) -> NDArray[np.float64]:
    return np.abs(state[src] - state[dst]).astype(np.float64, copy=False)


def _tree_levels_for_target(n_target: int, branching: int) -> int:
    levels = 1
    total = 1
    while total < n_target:
        levels += 1
        total = (branching**levels - 1) // (branching - 1)
    return levels


def _build_graph_from_cfg(cfg: PhysicalDatasetConfig) -> tuple[csr_matrix, Dict]:
    if cfg.graph_kind == "grid":
        return _build_grid(int(cfg.grid_m), int(cfg.grid_n))

    if cfg.graph_kind == "small_world":
        n_nodes = cfg.small_world_n if cfg.small_world_n is not None else cfg.grid_m * cfg.grid_n
        return _build_small_world(
            int(n_nodes),
            int(cfg.small_world_k),
            float(cfg.small_world_beta),
            int(cfg.seed),
        )

    if cfg.graph_kind == "tree":
        levels = cfg.tree_levels
        if levels is None:
            levels = _tree_levels_for_target(cfg.grid_m * cfg.grid_n, cfg.tree_branching)
        return _build_tree(int(levels), int(cfg.tree_branching))

    raise ValueError(f"Unsupported graph kind: {cfg.graph_kind}")


def _default_faucet_nodes(num_nodes: int, graph_meta: Dict, graph_kind: GraphKind) -> Sequence[int]:
    if graph_kind == "grid" and "shape" in graph_meta:
        m, n = graph_meta["shape"]
        center = (m // 2) * n + (n // 2)
        return (int(center),)
    return (int(num_nodes // 2),)


def _gaussian_pulse(t: np.ndarray, period: float, width: float, phase: float = 0.0) -> np.ndarray:
    wrapped = ((t - phase + 0.5 * period) % period) - 0.5 * period
    return np.exp(-(wrapped**2) / (2.0 * width**2))


def _simulate_wave_pulse(
    adj: csr_matrix,
    t_bins: int,
    *,
    dt: float,
    c: float,
    gamma: float,
    faucet_nodes: Sequence[int],
    faucet_period: float,
    faucet_amp: float,
    faucet_width: float,
    edge_threshold: float,
) -> tuple[list[csr_matrix], NDArray[np.float64], list[Dict]]:
    adj = adj.tocsr().astype(np.float64)
    laplacian = _graph_laplacian(adj)
    n_nodes = int(adj.shape[0])
    row, col = adj.tocoo().row.astype(np.int64), adj.tocoo().col.astype(np.int64)

    h = np.zeros(n_nodes, dtype=np.float64)
    v = np.zeros(n_nodes, dtype=np.float64)
    tgrid = np.arange(t_bins, dtype=np.float64) * float(dt)
    src_profile = float(faucet_amp) * _gaussian_pulse(
        tgrid,
        period=float(faucet_period),
        width=float(faucet_width),
    )

    faucet = np.fromiter(faucet_nodes, dtype=np.int64)
    faucet = faucet[(faucet >= 0) & (faucet < n_nodes)]
    if faucet.size == 0:
        raise ValueError("No valid faucet nodes for wave_pulse simulation.")

    b = np.zeros(n_nodes, dtype=np.float64)
    b[faucet] = 1.0 / max(1, faucet.size)

    damp_fac = (1.0 - 0.5 * float(gamma) * float(dt)) / (1.0 + 0.5 * float(gamma) * float(dt))
    acc_fac = float(dt) / (1.0 + 0.5 * float(gamma) * float(dt))
    a0 = -(float(c) * float(c)) * np.asarray(laplacian @ h).ravel() + src_profile[0] * b
    v_half = v - 0.5 * float(dt) * a0

    bins: list[csr_matrix] = []
    states: list[np.ndarray] = []
    meta: list[Dict] = []
    for step in range(t_bins):
        a = -(float(c) * float(c)) * np.asarray(laplacian @ h).ravel() + src_profile[step] * b
        v_half = damp_fac * v_half + acc_fac * a
        h = h + float(dt) * v_half
        states.append(h.copy())

        vals = np.abs(h[row] - h[col])
        if edge_threshold > 0.0:
            mask = vals > float(edge_threshold)
            rr = row[mask]
            cc = col[mask]
            data = vals[mask].astype(np.float32, copy=False)
        else:
            rr = row
            cc = col
            data = vals.astype(np.float32, copy=False)
        edge_values = csr_matrix((data, (rr, cc)), shape=(n_nodes, n_nodes))
        if edge_values.nnz == 0:
            bins.append(csr_matrix((n_nodes, n_nodes), dtype=np.uint8))
        else:
            edge_values.data[:] = 1.0
            bins.append(edge_values.astype(np.uint8, copy=False))
        meta.append({})
    return bins, np.stack(states, axis=0), meta


def _edges_from_flux_field(
    und_u: np.ndarray,
    und_v: np.ndarray,
    x: np.ndarray,
    rng: np.random.Generator,
    *,
    edge_scale: float,
    edge_thresh: float,
    edge_sparsity: float,
    min_edges: int,
) -> csr_matrix:
    n_nodes = int(x.size)
    num_edges = int(und_u.size)
    if num_edges == 0:
        return csr_matrix((n_nodes, n_nodes), dtype=np.uint8)

    diff = x[und_u] - x[und_v]
    mag = np.abs(diff)
    p = _sigmoid(float(edge_scale) * (mag - float(edge_thresh))) * float(edge_sparsity)
    active = rng.random(num_edges) < p
    min_edges = max(0, min(int(min_edges), num_edges))
    if not np.any(active) and min_edges > 0:
        idx = np.argpartition(p, -min_edges)[-min_edges:]
        active = np.zeros(num_edges, dtype=bool)
        active[idx] = True

    if not np.any(active):
        return csr_matrix((n_nodes, n_nodes), dtype=np.uint8)

    a_u = und_u[active].astype(np.int32)
    a_v = und_v[active].astype(np.int32)
    diff_active = diff[active]
    src = np.where(diff_active >= 0.0, a_u, a_v)
    dst = np.where(diff_active >= 0.0, a_v, a_u)
    return _edges_to_bin(src, dst, n_nodes)


def _simulate_wave_signed(
    adj: csr_matrix,
    t_bins: int,
    *,
    dt: float,
    c: float,
    gamma: float,
    mass: float,
    center_idx: int,
    amp: float,
    period: float,
    seed: int,
    edge_threshold: float,
) -> tuple[list[csr_matrix], NDArray[np.float64], list[Dict]]:
    adj = adj.tocsr().astype(np.float64)
    laplacian = _normalized_laplacian(adj)
    laplacian /= _estimate_lmax_power(laplacian, seed=seed)
    und_u, und_v = _undirected_edge_list(adj)

    rng = np.random.default_rng(seed)
    n_nodes = int(adj.shape[0])
    x = np.zeros(n_nodes, dtype=np.float64)
    v = np.zeros(n_nodes, dtype=np.float64)

    angular_freq = 2.0 * np.pi * float(dt) / max(float(period), 1e-6)

    bins: list[csr_matrix] = []
    states: list[np.ndarray] = []
    meta: list[Dict] = []
    for step in range(t_bins):
        source = np.zeros(n_nodes, dtype=np.float64)
        source[int(center_idx)] = float(amp) * np.sin(angular_freq * step)
        accel = -(float(c) * float(c)) * np.asarray(laplacian @ x).ravel()
        accel -= float(mass) * x
        accel -= float(gamma) * v
        accel += source
        v = v + float(dt) * accel
        x = x + float(dt) * v

        bins.append(
            _edges_from_flux_field(
                und_u,
                und_v,
                x,
                rng,
                edge_scale=6.0,
                edge_thresh=float(edge_threshold),
                edge_sparsity=1e-2,
                min_edges=max(1, int(0.02 * n_nodes)),
            )
        )
        states.append(x.copy())
        meta.append({"source": int(center_idx)})
    return bins, np.stack(states, axis=0), meta


def _simulate_sis(
    adj: csr_matrix,
    t_bins: int,
    *,
    beta: float,
    mu: float,
    init_infected: float,
    seed: int,
) -> tuple[list[csr_matrix], NDArray[np.float64], list[Dict]]:
    rng = np.random.default_rng(seed)
    n_nodes = int(adj.shape[0])
    indptr, indices = adj.indptr, adj.indices
    infected = np.asarray(rng.random(n_nodes) < float(init_infected), dtype=bool)

    bins: list[csr_matrix] = []
    states: list[np.ndarray] = []
    for _ in range(t_bins):
        recovered = infected & np.asarray(rng.random(n_nodes) < float(mu), dtype=bool)
        infected[recovered] = False

        new_u: list[int] = []
        new_v: list[int] = []
        susceptible = np.where(~infected)[0]
        for v in susceptible:
            nbrs = indices[indptr[v]:indptr[v + 1]]
            if nbrs.size == 0:
                continue
            infected_nbrs = nbrs[infected[nbrs]]
            k_inf = infected_nbrs.size
            if k_inf == 0:
                continue
            p = 1.0 - (1.0 - float(beta)) ** k_inf
            if rng.random() < p:
                infected[v] = True
                new_u.append(int(rng.choice(infected_nbrs)))
                new_v.append(int(v))

        bins.append(_edges_to_bin(np.asarray(new_u, np.int32), np.asarray(new_v, np.int32), n_nodes))
        states.append(infected.astype(np.float64))
    return bins, np.stack(states, axis=0), [{} for _ in range(t_bins)]


def _simulate_sirs(
    adj: csr_matrix,
    t_bins: int,
    *,
    beta: float,
    mu: float,
    rho: float,
    init_infected: float,
    seed: int,
) -> tuple[list[csr_matrix], NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    assert adj.shape is not None
    n_nodes = adj.shape[0]
    indptr, indices = adj.indptr, adj.indices

    state = np.zeros(n_nodes, dtype=np.int8)  # 0=S, 1=I, 2=R
    infected0 = rng.random(n_nodes) < float(init_infected)
    state[infected0] = 1

    bins: list[csr_matrix] = []
    states: list[np.ndarray] = []
    for _ in range(t_bins):
        infected = state == 1
        recovered = state == 2

        recover_mask = infected & (rng.random(n_nodes) < float(mu))
        state[recover_mask] = 2

        susceptible_again = recovered & (rng.random(n_nodes) < float(rho))
        state[susceptible_again] = 0

        new_u: list[int] = []
        new_v: list[int] = []
        susceptible = np.where(state == 0)[0]
        for v in susceptible:
            nbrs = indices[indptr[v]:indptr[v + 1]]
            if nbrs.size == 0:
                continue
            infected_nbrs = nbrs[state[nbrs] == 1]
            k_inf = infected_nbrs.size
            if k_inf == 0:
                continue
            p = 1.0 - (1.0 - float(beta)) ** k_inf
            if rng.random() < p:
                state[v] = 1
                u = int(rng.choice(infected_nbrs))
                new_u.append(u)
                new_v.append(int(v))

        bins.append(_edges_to_bin(np.asarray(new_u, np.int32), np.asarray(new_v, np.int32), n_nodes))
        states.append(state.astype(np.float64).copy())

    return bins, np.stack(states, axis=0)


def _edges_to_bin(u: np.ndarray, v: np.ndarray, n_nodes: int) -> csr_matrix:
    if u.size == 0:
        return csr_matrix((n_nodes, n_nodes), dtype=np.uint8)
    data = np.ones(u.size, dtype=np.uint8)
    out = csr_matrix((data, (u.astype(np.int32), v.astype(np.int32))), shape=(n_nodes, n_nodes), dtype=np.uint8)
    out.data[:] = 1
    return out


class PhysicalDynamicsDataset(EventStreamDataset):
    def __init__(self, cfg: PhysicalDatasetConfig):
        self.cfg = cfg
        self._adj, self._graph_meta = _build_graph_from_cfg(cfg)
        assert self._adj.shape is not None
        self._num_nodes = int(self._adj.shape[0])
        self._event_dim = 0

        bins, states, sim_meta = self._simulate()
        self._states = states
        self._sim_meta = sim_meta
        self._bins_all, self._node_targets_all, self._edge_targets_all = self._materialize_bins_and_targets(
            bins,
            states,
        )
        self._split_bins = self._compute_splits(len(self._bins_all))

    def _simulate(self) -> tuple[list[csr_matrix], NDArray[np.float64], list[Dict]]:
        if self.cfg.dynamics_kind == "wave":
            faucet_nodes = _default_faucet_nodes(self._num_nodes, self._graph_meta, self.cfg.graph_kind)
            center_idx = int(faucet_nodes[0])
            bins, states, meta = _simulate_wave_signed(
                self._adj,
                int(self.cfg.num_bins),
                dt=float(self.cfg.wave_dt),
                c=float(self.cfg.wave_c),
                gamma=float(self.cfg.wave_gamma),
                mass=float(self.cfg.wave_mass),
                center_idx=center_idx,
                amp=float(self.cfg.wave_faucet_amp),
                period=float(self.cfg.wave_faucet_period),
                seed=int(self.cfg.seed),
                edge_threshold=float(self.cfg.wave_edge_threshold),
            )
            return bins, states, meta

        if self.cfg.dynamics_kind == "wave_pulse":
            bins, states, meta = _simulate_wave_pulse(
                self._adj,
                int(self.cfg.num_bins),
                dt=float(self.cfg.wave_dt),
                c=float(self.cfg.wave_c),
                gamma=float(self.cfg.wave_gamma),
                faucet_nodes=_default_faucet_nodes(self._num_nodes, self._graph_meta, self.cfg.graph_kind),
                faucet_period=float(self.cfg.wave_faucet_period),
                faucet_amp=float(self.cfg.wave_faucet_amp),
                faucet_width=float(self.cfg.wave_faucet_width),
                edge_threshold=float(self.cfg.wave_edge_threshold),
            )
            return bins, states, meta

        if self.cfg.dynamics_kind == "sis":
            bins, states, meta = _simulate_sis(
                self._adj,
                int(self.cfg.num_bins),
                beta=float(self.cfg.sis_beta),
                mu=float(self.cfg.sis_mu),
                init_infected=float(self.cfg.sis_init_infected),
                seed=int(self.cfg.seed),
            )
            return bins, states, meta

        if self.cfg.dynamics_kind == "sirs":
            bins, states = _simulate_sirs(
                self._adj,
                int(self.cfg.num_bins),
                beta=float(self.cfg.sirs_beta),
                mu=float(self.cfg.sirs_mu),
                rho=float(self.cfg.sirs_rho),
                init_infected=float(self.cfg.sirs_init_infected),
                seed=int(self.cfg.seed),
            )
            return bins, states, [{} for _ in range(int(self.cfg.num_bins))]

        raise ValueError(f"Unsupported dynamics kind: {self.cfg.dynamics_kind}")

    def _materialize_bins_and_targets(
        self,
        bins: Sequence[csr_matrix],
        states: NDArray[np.float64],
    ) -> tuple[list[EventBatch], list[torch.Tensor], list[EdgeTargetBatch]]:
        out: list[EventBatch] = []
        node_targets: list[torch.Tensor] = []
        edge_targets: list[EdgeTargetBatch] = []
        adj_coo = self._adj.tocoo()
        edge_src = adj_coo.row.astype(np.int64)
        edge_dst = adj_coo.col.astype(np.int64)
        for t, mat in enumerate(bins):
            coo = mat.tocoo()
            if coo.nnz == 0:
                continue
            eb = EventBatch(
                src=cast(torch.LongTensor, torch.from_numpy(coo.row.astype(np.int64))),
                dst=cast(torch.LongTensor, torch.from_numpy(coo.col.astype(np.int64))),
                t=cast(torch.LongTensor, torch.full((coo.nnz,), t, dtype=torch.long)),
                features=None,
            )
            if self.cfg.device is not None:
                eb = eb.to(self.cfg.device)
            out.append(eb)
            target = torch.from_numpy(states[t].astype(np.float32, copy=False))
            if self.cfg.device is not None:
                target = target.to(self.cfg.device)
            node_targets.append(target)

            edge_events = EventBatch(
                src=cast(torch.LongTensor, torch.from_numpy(edge_src.copy())),
                dst=cast(torch.LongTensor, torch.from_numpy(edge_dst.copy())),
                t=cast(torch.LongTensor, torch.full((edge_src.size,), t, dtype=torch.long)),
                features=None,
            )
            edge_target_values = torch.from_numpy(
                _directed_edge_values_from_state(states[t], edge_src, edge_dst).astype(np.float32, copy=False)
            )
            if self.cfg.device is not None:
                edge_events = edge_events.to(self.cfg.device)
                edge_target_values = edge_target_values.to(self.cfg.device)
            edge_targets.append(EdgeTargetBatch(events=edge_events, targets=edge_target_values))
        return out, node_targets, edge_targets

    def _compute_splits(self, num_nonempty_bins: int) -> Dict[str, tuple[int, int]]:
        f_tr, f_va, f_te = self.cfg.split_fracs
        assert abs((f_tr + f_va + f_te) - 1.0) < 1e-6, "split_fracs must sum to 1.0"
        if num_nonempty_bins == 0:
            return {"train": (0, -1), "val": (0, -1), "test": (0, -1)}
        tr_end = int(num_nonempty_bins * f_tr)
        va_end = tr_end + int(num_nonempty_bins * f_va)
        return {
            "train": (0, max(0, tr_end - 1)),
            "val": (tr_end, max(tr_end, va_end - 1)),
            "test": (va_end, num_nonempty_bins - 1),
        }

    def spec(self) -> DataSpec:
        return DataSpec(
            name=f"{self.cfg.root_name}_{self.cfg.graph_kind}_{self.cfg.dynamics_kind}",
            num_nodes=self._num_nodes,
            event_dim=self._event_dim,
            num_events=sum(batch.num_events for batch in self._bins_all),
            num_bins=len(self._bins_all),
            extra={
                "graph_kind": self.cfg.graph_kind,
                "dynamics_kind": self.cfg.dynamics_kind,
                "graph_meta": self._graph_meta,
                "sim_seed": self.cfg.seed,
            },
        )

    def bins(self, split: str = "train") -> Iterable[EventBatch]:
        assert split in self._split_bins, f"unknown split={split}"
        start, end = self._split_bins[split]
        return _PhysicalBinnedStream(self._bins_all, start, end)

    def node_targets(self, split: str = "train") -> Optional[Iterable[torch.Tensor]]:
        assert split in self._split_bins, f"unknown split={split}"
        start, end = self._split_bins[split]
        return _PhysicalTargetStream(self._node_targets_all, start, end)

    def edge_targets(self, split: str = "train") -> Optional[Iterable[EdgeTargetBatch]]:
        assert split in self._split_bins, f"unknown split={split}"
        start, end = self._split_bins[split]
        return _PhysicalEdgeTargetStream(self._edge_targets_all, start, end)

    def states(self) -> NDArray[np.float64]:
        return self._states

    def graph_meta(self) -> Dict:
        return dict(self._graph_meta)


@dataclass
class _PhysicalBinnedStream(Iterable[EventBatch]):
    bins_all: Sequence[EventBatch]
    start: int
    end: int

    def __iter__(self) -> Iterator[EventBatch]:
        if self.end < self.start:
            return
        for idx in range(self.start, self.end + 1):
            yield self.bins_all[idx]


@dataclass
class _PhysicalTargetStream(Iterable[torch.Tensor]):
    targets_all: Sequence[torch.Tensor]
    start: int
    end: int

    def __iter__(self) -> Iterator[torch.Tensor]:
        if self.end < self.start:
            return
        for idx in range(self.start, self.end + 1):
            yield self.targets_all[idx]


@dataclass
class _PhysicalEdgeTargetStream(Iterable[EdgeTargetBatch]):
    targets_all: Sequence[EdgeTargetBatch]
    start: int
    end: int

    def __iter__(self) -> Iterator[EdgeTargetBatch]:
        if self.end < self.start:
            return
        for idx in range(self.start, self.end + 1):
            yield self.targets_all[idx]
