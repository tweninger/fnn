# Graphs are Measurements: experiment harness
#
# Mechanism:
#   build_graph(kind=...) -> substrate adjacency A0 (CSR)
#   run_simulator(kind=...) -> per-bin sparse activations -> events (u,v,t)
#
# Measurement graphs:
#   from the same (u,v,t), construct static graphs using:
#       - window sizes
#       - weights (count, exp decay)
#       - thresholds (top-k, weight>=tau, density)
#
# Downstream task (default):
#   link prediction: does a static measurement graph built at time T
#   predict which edges appear in [T, T+horizon)?
#
# Measurements:
#   performance variance across constructions
#   structure instability across time (within a construction)
#   mechanism recovery vs substrate A0
#
# Outputs:
#   - results CSV
#   - figures (bars + scatter + heatmap)
# ------------------------------------------------------------

from __future__ import annotations

import os
import math
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple, Any

import re
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix

# you already have these in THIS file per your dump:
from interactionfields.graphs import build_graph, edges_from_adj
from interactionfields.simulate import run_simulator

# reuse existing helper if you want it (fast, robust)
from interactionfields.eval import csr_to_set


def csr_bins_to_uvt(csr_bins: List[csr_matrix], dt: float = 1.0, t0: float = 0.0) -> np.ndarray:
    """Convert list[csr_matrix] (one per bin, with 0/1 entries) to an (M,3) array [u,v,t]."""
    rows: List[Tuple[int, int, float]] = []
    for i, A in enumerate(csr_bins):
        if A is None:
            continue
        coo = A.tocoo()
        t = t0 + i * dt
        for u, v in zip(coo.row, coo.col):
            rows.append((int(u), int(v), float(t)))
    if not rows:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(rows, dtype=float)


def simulate_event_stream(
    graph_kind: str = "torus_surface",
    graph_kwargs: Optional[dict] = None,
    dyn_kind: str = "waves",         # "waves" | "faucet"
    dyn_kwargs: Optional[dict] = None,
    t_bins: int = 600,
    dt: float = 1.0,
) -> Tuple[csr_matrix, Dict[str, Any], np.ndarray]:
    """
    Returns:
      A0: substrate adjacency (CSR) [mechanism]
      meta: graph metadata
      events_uvt: array (M,3) [u,v,t]
    """
    graph_kwargs = graph_kwargs or {}
    dyn_kwargs = dyn_kwargs or {}

    # NOTE: build_graph(**kwargs) expects named params including kind=...
    A0, meta = build_graph(graph_kind, **graph_kwargs)

    # ---- fill faucet center if needed
    if dyn_kind == "faucet" and dyn_kwargs.get("center_idx", None) is None:
        assert A0.shape is not None, "Substrate adjacency must have shape"
        N = int(A0.shape[0])
        m = meta.get("m", None)
        n = meta.get("n", None)
        if isinstance(m, int) and isinstance(n, int) and m * n == N:
            cy, cx = m // 2, n // 2
            center_idx = int(cy * n + cx)
        else:
            center_idx = int(N // 2)
        dyn_kwargs = dict(dyn_kwargs)
        dyn_kwargs["center_idx"] = center_idx

    bins, _H, _meta_list = run_simulator(dyn_kind, adj=A0, t_bins=t_bins, **dyn_kwargs)

    events_uvt = csr_bins_to_uvt(bins, dt=dt, t0=0.0)
    return A0, meta, events_uvt


# ============================================================
# 1) Measurement graph construction choices
# ============================================================

@dataclass(frozen=True)
class GraphConstructSpec:
    window: int                      # in bins, not seconds (since dt is your bin size)
    weight_mode: str                 # "count" | "exp_decay"
    decay_tau: Optional[float]       # in bins, used when exp_decay
    threshold_mode: str              # "topk" | "tau" | "density"
    topk: Optional[int] = None
    tau: Optional[float] = None
    density: Optional[float] = None
    directed: bool = True

    def key(self) -> str:
        if self.threshold_mode == "topk":
            thr = f"topk={self.topk}"
        elif self.threshold_mode == "tau":
            thr = f"tau={self.tau:g}"
        else:
            thr = f"dens={self.density:g}"
        if self.weight_mode == "exp_decay":
            return f"W={self.window}|{self.weight_mode}(tau={self.decay_tau:g})|{thr}"
        return f"W={self.window}|{self.weight_mode}|{thr}"


def _events_in_window(events_uvt: np.ndarray, t_end: float, window_bins: int, dt: float) -> np.ndarray:
    t0 = t_end - window_bins * dt
    m = (events_uvt[:, 2] >= t0) & (events_uvt[:, 2] < t_end)
    return events_uvt[m]


def _weighted_edges(window_events: np.ndarray, t_end: float, spec: GraphConstructSpec, dt: float) -> Dict[Tuple[int, int], float]:
    uv = window_events[:, :2].astype(np.int64, copy=False)
    ts = window_events[:, 2].astype(float, copy=False)
    w: Dict[Tuple[int, int], float] = {}

    if spec.weight_mode == "count":
        for u, v in uv:
            k = (int(u), int(v))
            w[k] = w.get(k, 0.0) + 1.0
        return w

    if spec.weight_mode == "exp_decay":
        if not spec.decay_tau or spec.decay_tau <= 0:
            raise ValueError("spec.decay_tau must be set (>0) for exp_decay")
        tau_sec = float(spec.decay_tau) * dt
        for (u, v), t in zip(uv, ts):
            k = (int(u), int(v))
            w[k] = w.get(k, 0.0) + math.exp(-(float(t_end) - float(t)) / tau_sec)
        return w

    raise ValueError(f"Unknown weight_mode: {spec.weight_mode}")


def _threshold(edge_w: Dict[Tuple[int, int], float], spec: GraphConstructSpec) -> Dict[Tuple[int, int], float]:
    items = list(edge_w.items())
    if not items:
        return {}

    items.sort(key=lambda x: x[1], reverse=True)

    if spec.threshold_mode == "topk":
        if spec.topk is None:
            raise ValueError("topk must be set for threshold_mode='topk'")
        return dict(items[: max(0, int(spec.topk))])

    if spec.threshold_mode == "tau":
        if spec.tau is None:
            raise ValueError("tau must be set for threshold_mode='tau'")
        tau = float(spec.tau)
        return {e: w for e, w in items if w >= tau}

    if spec.threshold_mode == "density":
        if spec.density is None:
            raise ValueError("density must be set for threshold_mode='density'")
        dens = float(spec.density)
        dens = min(max(dens, 0.0), 1.0)
        k = int(round(dens * len(items)))
        return dict(items[: max(0, k)])

    raise ValueError(f"Unknown threshold_mode: {spec.threshold_mode}")


def construct_measurement_graph(
    events_uvt: np.ndarray,
    t_end: float,
    spec: GraphConstructSpec,
    *,
    num_nodes: int,
    dt: float,
) -> nx.Graph:
    window_events = _events_in_window(events_uvt, t_end, spec.window, dt)
    ew = _weighted_edges(window_events, t_end, spec, dt)
    ew = _threshold(ew, spec)

    G = nx.DiGraph() if spec.directed else nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for (u, v), w in ew.items():
        if u == v:
            continue
        G.add_edge(u, v, weight=float(w))
    return G


# ============================================================
# 2) Metrics: instability + mechanism recovery
# ============================================================

def edge_set(G: nx.Graph) -> set[Tuple[int, int]]:
    return set((int(u), int(v)) for u, v in G.edges())


def jaccard_edges(E1: set[Tuple[int, int]], E2: set[Tuple[int, int]]) -> float:
    if not E1 and not E2:
        return 1.0
    if not E1 or not E2:
        return 0.0
    return len(E1 & E2) / len(E1 | E2)


def degree_rank_spearman(G1: nx.Graph, G2: nx.Graph) -> float:
    nodes = np.array(sorted(set(G1.nodes()) | set(G2.nodes())), dtype=np.int64)
    d1 = np.array([G1.degree(int(n)) for n in nodes], dtype=float) # pyright: ignore[reportCallIssue]
    d2 = np.array([G2.degree(int(n)) for n in nodes], dtype=float) # pyright: ignore[reportCallIssue]

    def rankdata(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, x.size + 1, dtype=float)
        sx = x[order]
        i = 0
        while i < sx.size:
            j = i
            while j + 1 < sx.size and sx[j + 1] == sx[i]:
                j += 1
            if j > i:
                avg = 0.5 * (i + j) + 1.0
                ranks[order[i:j + 1]] = avg
            i = j + 1
        return ranks

    r1, r2 = rankdata(d1), rankdata(d2)
    r1 -= r1.mean()
    r2 -= r2.mean()
    den = (np.sqrt((r1**2).sum()) * np.sqrt((r2**2).sum())) + 1e-12
    return float((r1 * r2).sum() / den)


def greedy_partition_labels(G: nx.Graph) -> Tuple[np.ndarray, np.ndarray]:
    from networkx.algorithms.community import greedy_modularity_communities
    H = G.to_undirected()
    nodes = np.array(sorted(H.nodes()), dtype=np.int64)
    if nodes.size == 0 or H.number_of_edges() == 0:
        return nodes, np.zeros(nodes.size, dtype=np.int64)
    comms = list(greedy_modularity_communities(H))
    node_to_idx = {int(n): i for i, n in enumerate(nodes)}
    lab = np.full(nodes.size, -1, dtype=np.int64)
    for cid, comm in enumerate(comms):
        for n in comm:
            lab[node_to_idx[int(n)]] = cid
    # assign leftovers as singletons
    next_id = int(lab.max() + 1) if np.any(lab >= 0) else 0
    for i in range(nodes.size):
        if lab[i] < 0:
            lab[i] = next_id
            next_id += 1
    return nodes, lab


def nmi_communities(G1: nx.Graph, G2: nx.Graph) -> float:
    from sklearn.metrics import normalized_mutual_info_score
    nodes = np.array(sorted(set(G1.nodes()) | set(G2.nodes())), dtype=np.int64)
    H1 = G1.subgraph(nodes.tolist())
    H2 = G2.subgraph(nodes.tolist())
    _, a = greedy_partition_labels(H1)
    _, b = greedy_partition_labels(H2)
    return float(normalized_mutual_info_score(a, b))


# ============================================================
# 3) Downstream task: link prediction (weight lookup)
# ============================================================

def _unique_edges_in_horizon(events_uvt: np.ndarray, t0: float, t1: float) -> np.ndarray:
    m = (events_uvt[:, 2] >= t0) & (events_uvt[:, 2] < t1)
    if not np.any(m):
        return np.zeros((0, 2), dtype=np.int64)
    uv = events_uvt[m, :2].astype(np.int64, copy=False)
    uv = np.unique(uv, axis=0)
    return uv


def _sample_negatives(num_nodes: int, forbidden: set[Tuple[int, int]], n: int, rng: np.random.Generator) -> np.ndarray:
    out: List[Tuple[int, int]] = []
    tries = 0
    max_tries = max(10000, 50 * n)
    while len(out) < n and tries < max_tries:
        u = int(rng.integers(0, num_nodes))
        v = int(rng.integers(0, num_nodes))
        if u == v:
            tries += 1
            continue
        if (u, v) in forbidden:
            tries += 1
            continue
        forbidden.add((u, v))
        out.append((u, v))
        tries += 1
    return np.asarray(out, dtype=np.int64)


def linkpred_metrics_weight_lookup(G: nx.Graph, pos_uv: np.ndarray, neg_uv: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score, average_precision_score

    def score(u: int, v: int) -> float:
        if G.has_edge(u, v):
            return float(G[u][v].get("weight", 1.0))
        return 0.0

    s_pos = np.array([score(int(u), int(v)) for u, v in pos_uv], dtype=float)
    s_neg = np.array([score(int(u), int(v)) for u, v in neg_uv], dtype=float)
    y = np.concatenate([np.ones(len(s_pos)), np.zeros(len(s_neg))]).astype(int)
    s = np.concatenate([s_pos, s_neg])

    if s.size == 0 or np.allclose(s, s[0]):
        return {"auc": 0.5, "ap": float(np.mean(y)) if y.size else np.nan}

    return {"auc": float(roc_auc_score(y, s)), "ap": float(average_precision_score(y, s))}


def eval_link_prediction(
    events_uvt: np.ndarray,
    G: nx.Graph,
    *,
    t_end: float,
    horizon_bins: int,
    dt: float,
    num_nodes: int,
    neg_ratio: float,
    rng: np.random.Generator,
) -> Dict[str, float]:
    pos = _unique_edges_in_horizon(events_uvt, t_end, t_end + horizon_bins * dt)
    if len(pos) == 0:
        return {"auc": np.nan, "ap": np.nan, "n_pos": 0, "n_neg": 0}
    forbidden = set(map(tuple, pos.tolist()))
    n_neg = int(round(len(pos) * float(neg_ratio)))
    neg = _sample_negatives(num_nodes, forbidden, n_neg, rng)
    m = linkpred_metrics_weight_lookup(G, pos, neg)
    m["n_pos"] = int(len(pos))
    m["n_neg"] = int(len(neg))
    return m


# ============================================================
# 4) Experiment runner
# ============================================================

@dataclass
class Row:
    t_end: float
    spec: str

    # downstream
    auc: float
    ap: float
    n_pos: int
    n_neg: int

    # instability (within spec; vs previous time)
    jacc_prev: float
    degspe_prev: float
    nmi_prev: float

    # mechanism recovery (vs substrate A0)
    jacc_sub: float
    degspe_sub: float
    nmi_sub: float


def _nx_from_substrate(A0: csr_matrix, directed: bool = True) -> nx.Graph:
    G0 = nx.DiGraph() if directed else nx.Graph()
    assert A0.shape is not None, "Substrate adjacency must have shape"
    N = A0.shape[0]
    G0.add_nodes_from(range(N))
    coo = A0.tocoo()
    for u, v in zip(coo.row, coo.col):
        if u == v:
            continue
        G0.add_edge(int(u), int(v), weight=1.0)
    return G0


def run_experiment(
    *,
    A0: csr_matrix,
    events_uvt: np.ndarray,
    dt: float,
    specs: List[GraphConstructSpec],
    eval_times: Iterable[int],          # in bin indices (integers)
    horizon_bins: int,
    neg_ratio: float,
    seed: int,
    pbar
) -> pd.DataFrame:

    rng = np.random.default_rng(seed)
    assert A0.shape is not None, "Substrate adjacency must have shape"
    N = int(A0.shape[0])
    G_sub = _nx_from_substrate(A0, directed=True)

    # precompute substrate edge set for fast jaccard
    E_sub = csr_to_set(A0)  # set[(u,v)]
    # for communities/NMI against substrate, we use nx graph:
    # (NMI is more expensive but fine for moderate N)

    prev_by_spec: Dict[str, nx.Graph] = {}
    rows: List[Row] = []

    for tbin in eval_times:
        t_end = float(tbin) * dt

        for spec in specs:
            key = spec.key()
            Gm = construct_measurement_graph(events_uvt, t_end, spec, num_nodes=N, dt=dt)
            pbar.update(1)
            # downstream
            dm = eval_link_prediction(
                events_uvt,
                Gm,
                t_end=t_end,
                horizon_bins=horizon_bins,
                dt=dt,
                num_nodes=N,
                neg_ratio=neg_ratio,
                rng=rng,
            )

            # instability vs prev time for same spec
            if key in prev_by_spec:
                Gprev = prev_by_spec[key]
                j_prev = jaccard_edges(edge_set(Gprev), edge_set(Gm))
                d_prev = degree_rank_spearman(Gprev, Gm)
                #n_prev = nmi_communities(Gprev, Gm)
            else:
                j_prev = np.nan
                d_prev = np.nan
                n_prev = np.nan

            prev_by_spec[key] = Gm

            # recovery vs substrate
            E_m = edge_set(Gm)
            # csr_to_set(A0) is directed set; align directed
            j_sub = jaccard_edges(E_m, E_sub)
            d_sub = degree_rank_spearman(G_sub, Gm)
            #n_sub = nmi_communities(G_sub, Gm)

            rows.append(Row(
                t_end=t_end,
                spec=key,
                auc=float(dm["auc"]),
                ap=float(dm["ap"]),
                n_pos=int(dm["n_pos"]),
                n_neg=int(dm["n_neg"]),
                jacc_prev=float(j_prev),
                degspe_prev=float(d_prev),
                nmi_prev=float(0.0),
                jacc_sub=float(j_sub),
                degspe_sub=float(d_sub),
                nmi_sub=float(0.0),
            ))

    df = pd.DataFrame([asdict(r) for r in rows])
    return df


# ============================================================
# 5) Plotting
# ============================================================

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def plot_all(df: pd.DataFrame, outdir: str, metric: str = "auc") -> None:
    """
    Creates:
      - performance bars (mean ± std across time)
      - performance vs recovery scatter
      - cross-spec heatmap of mean performance (window x threshold style) if parsable
    """
    import matplotlib.pyplot as plt

    _ensure_dir(outdir)

    # ---- bars: mean±std performance per spec
    g = df.groupby("spec")[metric].agg(["mean", "std"]).reset_index().sort_values("mean", ascending=False)
    plt.figure(figsize=(10, max(4, 0.32 * len(g))))
    y = np.arange(len(g))
    plt.barh(y, g["mean"].to_numpy(), xerr=g["std"].to_numpy())
    plt.yticks(y, g["spec"].tolist())
    plt.gca().invert_yaxis()
    plt.xlabel(f"{metric} (mean ± std over eval times)")
    plt.title("Downstream performance varies across measurement graphs")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"bars_{metric}.png"), dpi=250)
    plt.close()

    # ---- scatter: performance vs mechanism recovery
    # use mean over time per spec
    s = df.groupby("spec").agg(
        perf_mean=(metric, "mean"),
        jacc_sub_mean=("jacc_sub", "mean"),
        degspe_sub_mean=("degspe_sub", "mean"),
    ).reset_index()

    plt.figure()
    plt.scatter(s["jacc_sub_mean"].to_numpy(), s["perf_mean"].to_numpy())
    plt.xlabel("Mean edge Jaccard to substrate A0")
    plt.ylabel(f"Mean {metric}")
    plt.title("Performance ≠ mechanism recovery (graphs are measurements)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"scatter_perf_vs_jacc_sub_{metric}.png"), dpi=250)
    plt.close()

    plt.figure()
    plt.scatter(s["degspe_sub_mean"].to_numpy(), s["perf_mean"].to_numpy())
    plt.xlabel("Mean degree-rank Spearman to substrate A0")
    plt.ylabel(f"Mean {metric}")
    plt.title("Performance vs substrate degree agreement")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"scatter_perf_vs_degspe_sub_{metric}.png"), dpi=250)
    plt.close()

    # ---- instability over time (within spec): boxplot of jacc_prev
    # (drop NaNs: first time step per spec)
    d2 = df.dropna(subset=["jacc_prev"])
    if len(d2) > 0:
        specs = sorted(d2["spec"].unique().tolist())
        data = [np.asarray(d2.loc[d2["spec"] == sp, "jacc_prev"]) for sp in specs]
        plt.figure(figsize=(12, max(4, 0.25 * len(specs))))
        plt.boxplot(data, vert=False, tick_labels=specs, showfliers=False)
        plt.xlabel("Edge Jaccard vs previous time (within construction)")
        plt.title("Structural instability over time depends on graph construction choices")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "box_jacc_prev.png"), dpi=250)
        plt.close()




def make_specs(
    windows,
    weight_modes,
    thresholds,
    directed=True,
):
    """
    windows: list[int] (bins)
    weight_modes: list[("count", None)] or [("exp_decay", decay_tau_in_bins)]
    thresholds: list of dicts like {"mode":"topk","topk":2000} or {"mode":"tau","tau":3.0} or {"mode":"density","density":0.02}
    """
    specs = []
    for W in windows:
        for wm, decay_tau in weight_modes:
            for thr in thresholds:
                mode = thr["mode"]
                if mode == "topk":
                    specs.append(GraphConstructSpec(
                        window=W,
                        weight_mode=wm,
                        decay_tau=decay_tau,
                        threshold_mode="topk",
                        topk=int(thr["topk"]),
                        directed=directed,
                    ))
                elif mode == "tau":
                    specs.append(GraphConstructSpec(
                        window=W,
                        weight_mode=wm,
                        decay_tau=decay_tau,
                        threshold_mode="tau",
                        tau=float(thr["tau"]),
                        directed=directed,
                    ))
                elif mode == "density":
                    specs.append(GraphConstructSpec(
                        window=W,
                        weight_mode=wm,
                        decay_tau=decay_tau,
                        threshold_mode="density",
                        density=float(thr["density"]),
                        directed=directed,
                    ))
                else:
                    raise ValueError(f"Unknown threshold mode: {mode}")
    return specs


def main():
    # ------------------------------------------------------------
    # 1) Mechanism grid (what *creates* events)
    # ------------------------------------------------------------
    SEEDS = list(range(10))

    graph_options = [
        ("torus_surface", {"m": 30, "n": 30, "directed": True, "matrix_format": "csr"}),
        # ("grid", {"N": 900, "m": 30, "n": 30, "periodic_x": True, "periodic_y": True, "directed": True, "matrix_format": "csr"}),
        # ("grid_with_gate", {"m": 30, "n": 30, ...}),
    ]

    dyn_options = [
        ("waves",  {"c2": 0.4, "m2": 0.2, "gamma": 0.3, "noise": 0.02, "scale": 1.0, "grad_threshold": 0.0, "dt": 1.0}),
        ("faucet", {"center_idx": None, "faucet_period": 40, "speed_hops_per_step": 0.35, "sigma_hops": 1.5, "p_max": 0.6,
                   "return_states": False, "seed": 0}),  # center_idx filled after meta known
    ]

    # simulation length
    t_bins = 600
    dt = 1.0

    # ------------------------------------------------------------
    # 2) Measurement grid (how we turn events into graphs)
    # ------------------------------------------------------------
    windows = [5, 10, 20, 30, 60, 120, 240, 480]  # bins

    # weight modes: (mode, decay_tau_in_bins)
    # tie decay tau to window for a clean comparison
    weight_modes = [
        ("count", None),
        ("exp_decay", None),  # we'll fill per-window below (W/3) to keep it simple
    ]

    thresholds = [
        {"mode": "density", "density": 0.01},
        {"mode": "density", "density": 0.05},
        # {"mode": "topk", "topk": 3000},
        # {"mode": "tau", "tau": 2.0},
    ]

    # evaluation schedule (fixed)
    burn_in = 50
    eval_every = 10
    horizon_bins = 10
    neg_ratio = 1.0

    out_root = "out/graphs_are_measurements"
    _ensure_dir(out_root)

    all_dfs = []

    total = (
            len(SEEDS)
            * len(graph_options)
            * len(dyn_options)
            * len(windows)
            * len(weight_modes)
            * len(thresholds)
            * len(range(burn_in, t_bins - horizon_bins, eval_every))
    )

    pbar = tqdm(total=total, desc="Graphs-are-measurements")

    # ------------------------------------------------------------
    # Grid search
    # ------------------------------------------------------------
    for seed in SEEDS:
        for graph_kind, graph_kwargs in graph_options:
            for dyn_kind, dyn_kwargs0 in dyn_options:

                # make dyn kwargs copy and seed it if it has seed
                dyn_kwargs = dict(dyn_kwargs0)
                if "seed" in dyn_kwargs:
                    dyn_kwargs["seed"] = seed

                # simulate mechanism -> events
                A0, meta, events_uvt = simulate_event_stream(
                    graph_kind=graph_kind,
                    graph_kwargs=dict(graph_kwargs),
                    dyn_kind=dyn_kind,
                    dyn_kwargs=dyn_kwargs,
                    t_bins=t_bins,
                    dt=dt,
                )

                # faucet center_idx convenience
                if dyn_kind == "faucet" and dyn_kwargs.get("center_idx", None) is None:
                    assert A0.shape is not None, "Substrate adjacency must have shape"
                    N = int(A0.shape[0])
                    m = meta.get("m", None)
                    n = meta.get("n", None)
                    if isinstance(m, int) and isinstance(n, int) and m * n == N:
                        cy, cx = m // 2, n // 2
                        center_idx = int(cy * n + cx)
                    else:
                        center_idx = int(N // 2)

                    dyn_kwargs2 = dict(dyn_kwargs)
                    dyn_kwargs2["center_idx"] = center_idx

                    A0, meta, events_uvt = simulate_event_stream(
                        graph_kind=graph_kind,
                        graph_kwargs=dict(graph_kwargs),
                        dyn_kind=dyn_kind,
                        dyn_kwargs=dyn_kwargs2,
                        t_bins=t_bins,
                        dt=dt,
                    )

                # build specs (fill exp_decay tau per window)
                wm_expanded = []
                for wm, tau in weight_modes:
                    if wm == "exp_decay":
                        # placeholder; we’ll create per-window inside make_specs by passing None and replacing later
                        wm_expanded.append((wm, None))
                    else:
                        wm_expanded.append((wm, tau))

                specs = []
                for W in windows:
                    wm_for_W = []
                    for wm, tau in wm_expanded:
                        if wm == "exp_decay":
                            wm_for_W.append((wm, max(1.0, W / 3.0)))
                        else:
                            wm_for_W.append((wm, tau))
                    specs.extend(make_specs([W], wm_for_W, thresholds, directed=True))

                # eval times
                eval_times = list(range(burn_in, t_bins - horizon_bins, eval_every))

                df = run_experiment(
                    A0=A0,
                    events_uvt=events_uvt,
                    dt=dt,
                    specs=specs,
                    eval_times=eval_times,
                    horizon_bins=horizon_bins,
                    neg_ratio=neg_ratio,
                    seed=seed,
                    pbar=pbar
                )

                # annotate run metadata
                df["seed"] = seed
                df["graph_kind"] = graph_kind
                df["dyn_kind"] = dyn_kind
                df["t_bins"] = t_bins
                df["dt"] = dt

                run_name = f"{graph_kind}__{dyn_kind}__seed{seed}"
                run_dir = os.path.join(out_root, run_name)
                _ensure_dir(run_dir)

                df.to_csv(os.path.join(run_dir, "results.csv"), index=False)
                plot_all(df, run_dir, metric="auc")
                plot_all(df, run_dir, metric="ap")

                print("\nDONE:", run_name)
                print(df.groupby("spec")[["auc", "ap", "jacc_sub", "jacc_prev"]]
                        .mean()
                        .sort_values("auc", ascending=False)
                        .head(5))

                all_dfs.append(df)

    # combined output
    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
        df_all.to_csv(os.path.join(out_root, "all_results.csv"), index=False)
        print("\nWrote:", os.path.join(out_root, "all_results.csv"))





IN_CSV  = r"out/graphs_are_measurements/all_results.csv"
OUT_CSV = r"out/graphs_are_measurements/fig1_summary.csv"

# If you want to focus the intro figure, filter to one density.
# Set to None to keep all densities and average them (not recommended for intro clarity).
FILTER_DENS = 0.05  # e.g., 0.05; or None

# If you want one graph kind only (likely yes)
FILTER_GRAPH_KIND = "torus_surface"  # or None

# ------------ parsing helpers ------------

_spec_re = re.compile(
    r"""^W=(?P<W>\d+)\|
        (?P<wm>count|exp_decay(?:\(tau=(?P<tau>[-+]?(\d+(\.\d+)?)|(\.\d+))\))?)\|
        dens=(?P<dens>[-+]?(\d+(\.\d+)?)|(\.\d+))$
    """,
    re.VERBOSE,
)

def parse_spec(s: str):
    m = _spec_re.match(str(s).strip())
    if not m:
        raise ValueError(f"Could not parse spec: {s}")
    W = int(m.group("W"))
    wm_raw = m.group("wm")
    tau = m.group("tau")
    dens = float(m.group("dens"))
    if wm_raw.startswith("count"):
        wm = "count"
        tau_f = np.nan
    else:
        wm = "exp_decay"
        tau_f = float(tau)
    return W, wm, tau_f, dens

def to_pgf():
    df = pd.read_csv(IN_CSV)

    # Expected columns (based on your pipeline):
    # eval_time, spec, auc, ap, ..., jacc_sub, jacc_prev, deg_rank_sub, deg_rank_prev, seed, graph_kind, dyn_kind, t_bins, dt
    # We'll use: spec, auc, jacc_prev, seed, dyn_kind, graph_kind
    needed = {"spec", "auc", "jacc_prev", "seed", "dyn_kind"}
    missing = sorted(list(needed - set(df.columns)))
    if missing:
        raise RuntimeError(f"Missing columns in CSV: {missing}\nHave: {sorted(df.columns)}")

    # Parse spec fields
    parsed = df["spec"].apply(parse_spec)
    df["W"] = parsed.apply(lambda x: x[0])
    df["weight_mode"] = parsed.apply(lambda x: x[1])
    df["tau"] = parsed.apply(lambda x: x[2])
    df["dens"] = parsed.apply(lambda x: x[3])

    if FILTER_GRAPH_KIND is not None and "graph_kind" in df.columns:
        df = df[df["graph_kind"] == FILTER_GRAPH_KIND].copy()

    if FILTER_DENS is not None:
        df = df[np.isclose(df["dens"], FILTER_DENS)].copy()

    # ---- Important aggregation choice ----
    # We do NOT want eval_times to overweight a single seed.
    # So: (seed, dyn_kind, W, weight_mode) -> mean over eval_time
    g0 = ["seed", "dyn_kind", "W", "weight_mode"]
    per_seed = (
        df.groupby(g0, as_index=False)
          .agg(
              auc_mean_seed=("auc", "mean"),
              jprev_mean_seed=("jacc_prev", "mean"),
          )
    )

    # Then aggregate across seeds: mean + std
    g1 = ["dyn_kind", "W", "weight_mode"]
    summary = (
        per_seed.groupby(g1, as_index=False)
                .agg(
                    auc_mean=("auc_mean_seed", "mean"),
                    auc_std=("auc_mean_seed", "std"),
                    jprev_mean=("jprev_mean_seed", "mean"),
                    jprev_std=("jprev_mean_seed", "std"),
                    n_seeds=("seed", "nunique"),
                )
    )

    # Replace NaN std (e.g., if n_seeds=1) with 0
    summary["auc_std"] = summary["auc_std"].fillna(0.0)
    summary["jprev_std"] = summary["jprev_std"].fillna(0.0)

    # Sort nicely
    dyn_order = {"waves": 0, "faucet": 1}
    wm_order = {"count": 0, "exp_decay": 1}
    summary["_dyn"] = summary["dyn_kind"].map(lambda x: dyn_order.get(x, 999))
    summary["_wm"] = summary["weight_mode"].map(lambda x: wm_order.get(x, 999))
    summary = summary.sort_values(["_dyn", "W", "_wm"]).drop(columns=["_dyn", "_wm"])

    # Write for PGFPlots
    summary.to_csv(OUT_CSV, index=False)
    print("Wrote:", OUT_CSV)
    print(summary.head(12).to_string(index=False))

    OUT_DIR = r"out/graphs_are_measurements"

    def write_slice(dyn_kind, weight_mode, outname):
        sl = summary[(summary["dyn_kind"] == dyn_kind) & (summary["weight_mode"] == weight_mode)].copy()
        # keep only numeric columns PGFPlots needs
        sl = sl[["W", "auc_mean", "auc_std", "jprev_mean", "jprev_std", "n_seeds"]]
        outpath = os.path.join(OUT_DIR, outname)
        sl.to_csv(outpath, index=False)
        print("Wrote:", outpath, "rows=", len(sl))

    write_slice("waves", "count", "fig1_waves_count.csv")
    write_slice("waves", "exp_decay", "fig1_waves_exp.csv")
    write_slice("faucet", "count", "fig1_faucet_count.csv")
    write_slice("faucet", "exp_decay", "fig1_faucet_exp.csv")


if __name__ == "__main__":
    #main()
    to_pgf()
