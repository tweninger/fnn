# baselines_static.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Callable

import numpy as np
import scipy.sparse as sp

# Reuse your existing evaluation utilities
from .eval import (
    make_labels_for_risk_sets,
    aggregate_edge_metrics_over_horizon,
    node_series_from_csr_bins,
    prepare_series,
    ar1_baseline_from_train,
    rollout_ar1,
    eval_rollout,
)

# ------------------------------------------------------------
# Config + helpers
# ------------------------------------------------------------

@dataclass
class StaticGraphSpec:
    """
    Measurement choices (this is the point of the experiment):
      - window_bins: how much TRAIN history to aggregate
      - weight_mode: 'binary' | 'count' | 'exp_decay'
      - decay_tau_bins: only used for 'exp_decay' (in bins)
      - undirected: symmetrize or keep directed
      - include_self_loops: usually False
    """
    window_bins: Optional[int] = None
    weight_mode: str = "count"          # 'binary' | 'count' | 'exp_decay'
    decay_tau_bins: float = 50.0
    undirected: bool = True
    include_self_loops: bool = False


def _slice_bins(
    y_bins: Sequence[sp.csr_matrix],
    t0: int,
    t1: int
) -> List[sp.csr_matrix]:
    return list(y_bins[t0:t1])


def _symmetrize_csr(A) -> sp.csr_matrix:
    A = sp.csr_matrix(A)
    return (A + A.transpose()).tocsr()


def build_static_graph_from_train(
    y_train_bins: Sequence[sp.csr_matrix],
    num_nodes: int,
    spec: StaticGraphSpec,
) -> sp.csr_matrix:
    """
    Build a static adjacency/weight matrix from TRAIN bins only.

    This is intentionally where "graph construction choices" live.
    """
    if spec.window_bins is not None:
        y_train_bins = y_train_bins[-spec.window_bins :]

    if spec.weight_mode not in ("binary", "count", "exp_decay"):
        raise ValueError(f"Unknown weight_mode: {spec.weight_mode}")

    # Accumulate in COO chunks to keep it simple.
    # If you need more speed later, we can do a SparseMemory-like global scaling trick.
    rows_all, cols_all, data_all = [], [], []

    if spec.weight_mode in ("binary", "count"):
        # Sum across bins; binary uses 1's, count uses csr.data if present else 1's
        for A_t in y_train_bins:
            A_t = A_t.tocsr()
            coo = A_t.tocoo(copy=False)
            rows_all.append(coo.row)
            cols_all.append(coo.col)
            if spec.weight_mode == "binary":
                data_all.append(np.ones_like(coo.row, dtype=np.float32))
            else:
                # If A_t is 0/1, coo.data is all 1's. If weighted, preserves weights.
                data_all.append(coo.data.astype(np.float32, copy=False))

        if len(rows_all) == 0:
            return sp.csr_matrix((num_nodes, num_nodes), dtype=np.float32)

        rows = np.concatenate(rows_all)
        cols = np.concatenate(cols_all)
        data = np.concatenate(data_all)

        A = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float32).tocsr()
        A.sum_duplicates()

        if spec.weight_mode == "binary":
            A.data[:] = 1.0

    else:
        # Exponential decay over bins: A <- exp(-1/tau)*A + events
        tau = float(spec.decay_tau_bins)
        decay = np.exp(-1.0 / max(tau, 1e-9))

        A = sp.csr_matrix((num_nodes, num_nodes), dtype=np.float32)
        for A_t in y_train_bins:
            A = (A * decay).tocsr()
            coo = A_t.tocoo(copy=False)
            inc = sp.coo_matrix(
                (np.ones_like(coo.row, dtype=np.float32), (coo.row, coo.col)),
                shape=(num_nodes, num_nodes),
                dtype=np.float32,
            ).tocsr()
            A = (A + inc).tocsr()
            A.sum_duplicates()

    if not spec.include_self_loops:
        A.setdiag(0.0)
        A.eliminate_zeros()

    if spec.undirected:
        A = _symmetrize_csr(A)
        if spec.weight_mode == "binary":
            A.data[:] = 1.0
        else:
            A.sum_duplicates()

    # Ensure we return a scipy.sparse.csr_matrix (convert from csr_array if needed)
    return sp.csr_matrix(A)


def _unpack_risk_t(risk_t) -> Tuple[np.ndarray, np.ndarray]:
    """
    Your risk_sets are usually:
      - tuple/list of (u_arr, v_arr) OR
      - torch 2xK LongTensor OR
      - numpy 2xK
    We'll normalize to two numpy int64 arrays.
    """
    if isinstance(risk_t, (tuple, list)) and len(risk_t) == 2:
        u, v = risk_t
        u = np.asarray(u, dtype=np.int64).ravel()
        v = np.asarray(v, dtype=np.int64).ravel()
        return u, v

    # torch Tensor or numpy 2xK
    arr = np.asarray(risk_t)
    if arr.ndim == 2 and arr.shape[0] == 2:
        return arr[0].astype(np.int64, copy=False), arr[1].astype(np.int64, copy=False)

    raise TypeError(f"Unsupported risk set type/shape: {type(risk_t)} {getattr(risk_t,'shape',None)}")


# ------------------------------------------------------------
# Heuristics on a static graph
# ------------------------------------------------------------

def _common_neighbors_score_pairs(A: sp.csr_matrix, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    CN(u,v) = |N(u) ∩ N(v)| on an (optionally symmetrized) adjacency A.

    Implemented by CSR-row intersection. (Memory friendly; no A@A materialization.)
    """
    A = A.tocsr()
    indptr, indices = A.indptr, A.indices

    out = np.zeros(u.shape[0], dtype=np.float32)
    for i in range(u.shape[0]):
        ru0, ru1 = indptr[u[i]], indptr[u[i] + 1]
        rv0, rv1 = indptr[v[i]], indptr[v[i] + 1]
        nu = indices[ru0:ru1]
        nv = indices[rv0:rv1]
        # two-pointer intersection count
        a = b = 0
        c = 0
        while a < nu.size and b < nv.size:
            if nu[a] == nv[b]:
                c += 1
                a += 1
                b += 1
            elif nu[a] < nv[b]:
                a += 1
            else:
                b += 1
        out[i] = c
    return out


def _adamic_adar_score_pairs(A: sp.csr_matrix, u: np.ndarray, v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    AA(u,v) = sum_{w in N(u)∩N(v)} 1 / log(deg(w)+eps)
    """
    A = A.tocsr()
    indptr, indices = A.indptr, A.indices
    deg = np.diff(indptr).astype(np.float32)
    invlog = 1.0 / np.log(deg + eps + 1.0)  # +1 keeps log defined at deg=0

    out = np.zeros(u.shape[0], dtype=np.float32)
    for i in range(u.shape[0]):
        ru0, ru1 = indptr[u[i]], indptr[u[i] + 1]
        rv0, rv1 = indptr[v[i]], indptr[v[i] + 1]
        nu = indices[ru0:ru1]
        nv = indices[rv0:rv1]
        a = b = 0
        s = 0.0
        while a < nu.size and b < nv.size:
            if nu[a] == nv[b]:
                w = nu[a]
                s += float(invlog[w])
                a += 1
                b += 1
            elif nu[a] < nv[b]:
                a += 1
            else:
                b += 1
        out[i] = s
    return out


def _katz_truncated_score_pairs(
    A_bin: sp.csr_matrix,
    u: np.ndarray,
    v: np.ndarray,
    beta: float = 0.05,
    max_hops: int = 3,
) -> np.ndarray:
    """
    Truncated Katz:
      score = sum_{l=1..L} beta^l * (#paths length l from u to v)
    For stability and interpretability, we use A_bin (0/1).
    """
    if max_hops < 1:
        return np.zeros(u.shape[0], dtype=np.float32)

    A_bin = A_bin.tocsr().astype(np.float32)
    # Precompute powers up to L (sparse matmul)
    P = A_bin.copy()
    out = (beta ** 1) * _lookup_sparse(P, u, v)

    for l in range(2, max_hops + 1):
        P = (P @ A_bin).tocsr()
        P.sum_duplicates()
        out += (beta ** l) * _lookup_sparse(P, u, v)

    return out.astype(np.float32, copy=False)


def _lookup_sparse(M: sp.csr_matrix, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Lookup M[u[i], v[i]] for i=0..K-1 efficiently with CSR row slicing.
    """
    M = M.tocsr()
    indptr, indices, data = M.indptr, M.indices, M.data
    out = np.zeros(u.shape[0], dtype=np.float32)
    for i in range(u.shape[0]):
        r0, r1 = indptr[u[i]], indptr[u[i] + 1]
        cols = indices[r0:r1]
        # binary search in sorted indices
        j = np.searchsorted(cols, v[i])
        if j < cols.size and cols[j] == v[i]:
            out[i] = float(data[r0 + j])
    return out


# ------------------------------------------------------------
# Dyad EWMA baseline (temporal-only link score)
# ------------------------------------------------------------

@dataclass
class DyadEWMABaseline:
    """
    Temporal baseline on dyads (no graph heuristics):
      m_{uv} <- decay*m_{uv} + 1{event in bin}
    Scores are current m_{uv} for candidate pairs.
    """
    num_nodes: int
    decay: float = 0.98
    # Store only dyads that ever appear (sparse dict keyed by u*N+v)
    state: Dict[int, float] = field(default_factory=dict)

    def step(self, y_bin: sp.csr_matrix):
        # global decay (cheap)
        if self.state:
            for k in list(self.state.keys()):
                self.state[k] *= self.decay
                if self.state[k] < 1e-8:
                    del self.state[k]

        # add new positives
        coo = y_bin.tocoo(copy=False)
        N = self.num_nodes
        for uu, vv in zip(coo.row, coo.col):
            key = int(uu) * N + int(vv)
            self.state[key] = self.state.get(key, 0.0) + 1.0

    def score_pairs(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        N = self.num_nodes
        out = np.zeros(u.shape[0], dtype=np.float32)
        for i in range(u.shape[0]):
            key = int(u[i]) * N + int(v[i])
            out[i] = float(self.state.get(key, 0.0))
        return out


# ------------------------------------------------------------
# Logistic regression on static graph features
# ------------------------------------------------------------

def _sample_train_pairs_from_risk_sets(
    risk_sets: Sequence,
    label_sets: Sequence[np.ndarray],
    max_pos_per_bin: int = 5000,
    neg_ratio: float = 5.0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a training set by taking positives + sampled negatives from risk sets.

    Returns (u_all, v_all, y_all) arrays.
    """
    rng = np.random.default_rng(seed)
    U, V, Y = [], [], []
    for risk_t, y_t in zip(risk_sets, label_sets):
        u, v = _unpack_risk_t(risk_t)
        y_t = np.asarray(y_t, dtype=np.int64).ravel()
        pos_idx = np.flatnonzero(y_t == 1)
        neg_idx = np.flatnonzero(y_t == 0)

        if pos_idx.size == 0:
            continue

        # subsample positives if huge
        if pos_idx.size > max_pos_per_bin:
            pos_idx = rng.choice(pos_idx, size=max_pos_per_bin, replace=False)

        n_neg = int(np.ceil(neg_ratio * pos_idx.size))
        if neg_idx.size > 0:
            neg_take = rng.choice(neg_idx, size=min(n_neg, neg_idx.size), replace=False)
        else:
            neg_take = np.array([], dtype=np.int64)

        idx = np.concatenate([pos_idx, neg_take])
        rng.shuffle(idx)

        U.append(u[idx])
        V.append(v[idx])
        Y.append(y_t[idx])

    if not U:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    return np.concatenate(U), np.concatenate(V), np.concatenate(Y)


def _static_lr_features(
    A: sp.csr_matrix,
    A_bin: sp.csr_matrix,
    u: np.ndarray,
    v: np.ndarray,
    use_katz: bool = True,
    katz_beta: float = 0.05,
    katz_hops: int = 3,
) -> np.ndarray:
    """
    Features for LR:
      [CN, AA, Katz, deg(u), deg(v), log(1+deg(u)deg(v)), edge_exists]
    """
    A = A.tocsr()
    deg = np.diff(A.indptr).astype(np.float32)

    cn = _common_neighbors_score_pairs(A_bin, u, v)
    aa = _adamic_adar_score_pairs(A_bin, u, v)
    if use_katz:
        kz = _katz_truncated_score_pairs(A_bin, u, v, beta=katz_beta, max_hops=katz_hops)
    else:
        kz = np.zeros_like(cn)

    du = deg[u]
    dv = deg[v]
    prod = np.log1p(du * dv)

    edge_exists = _lookup_sparse(A_bin, u, v)
    edge_exists = (edge_exists > 0).astype(np.float32)

    X = np.stack([cn, aa, kz, du, dv, prod, edge_exists], axis=1).astype(np.float32)
    return X


# ------------------------------------------------------------
# Public experiment runners
# ------------------------------------------------------------

def run_link_prediction_static_baselines(
    y_bins_csr: Sequence[sp.csr_matrix],
    risk_sets: Sequence,  # aligned 1:1 with y_bins_csr
    split: Tuple[int, int],  # (T_train, T_val) with test = [T_val..T)
    num_nodes: int,
    horizon: int = 1,
    k: int = 200,
    k_frac: Optional[float] = None,
    static_spec: Optional[StaticGraphSpec] = None,
    katz_beta: float = 0.05,
    katz_hops: int = 3,
    dyad_decay: float = 0.98,
    lr_neg_ratio: float = 5.0,
    lr_max_pos_per_bin: int = 5000,
    seed: int = 0,
) -> List[dict]:
    """
    Returns a list of metric dict rows; you can print them with your existing print helpers.

    NOTE:
      - Uses TRAIN-only static graph for heuristics + LR.
      - Uses the SAME risk_sets for all methods (fair candidate set).
      - Evaluates over TEST bins only.
    """
    T_train, T_val = split
    T = len(y_bins_csr)
    if static_spec is None:
        static_spec = StaticGraphSpec()

    # --- Build labels aligned to risk sets (reuses your utility)
    y_labels = make_labels_for_risk_sets(y_bins_csr, risk_sets)

    # --- Build static graph from TRAIN only
    y_train_bins = y_bins_csr[:T_train]
    A = build_static_graph_from_train(y_train_bins, num_nodes, static_spec)
    A_bin = A.copy()
    A_bin.data[:] = 1.0
    A_bin.sum_duplicates()
    A_bin.eliminate_zeros()

    # --- Define TEST window
    test_bins = list(range(T_val, T))

    def _eval_scores_over_horizon(
        method: str,
        score_list: List[np.ndarray],
        label_list: List[np.ndarray],
        is_prob: bool,
    ) -> dict:
        m = aggregate_edge_metrics_over_horizon(
            score_list=score_list,
            label_list=label_list,
            horizon=horizon,
            is_prob=is_prob,
            k=k,
            k_frac=(k_frac if k_frac is not None else 0.05),
        )
        row = {"split": "test", "method": method, "horizon": horizon}
        row.update(m)
        return row

    # --- Heuristics: compute per-bin scores on the risk set
    def score_bin_with(fn_pairs: Callable[[np.ndarray, np.ndarray], np.ndarray], t: int) -> np.ndarray:
        u, v = _unpack_risk_t(risk_sets[t])
        return fn_pairs(u, v)

    rows = []

    # Common Neighbors
    cn_scores = [score_bin_with(lambda u, v: _common_neighbors_score_pairs(A_bin, u, v), t) for t in test_bins]
    cn_labels = [y_labels[t] for t in test_bins]
    rows.append(_eval_scores_over_horizon("static_CN", cn_scores, cn_labels, is_prob=False))

    # Adamic–Adar
    aa_scores = [score_bin_with(lambda u, v: _adamic_adar_score_pairs(A_bin, u, v), t) for t in test_bins]
    aa_labels = [y_labels[t] for t in test_bins]
    rows.append(_eval_scores_over_horizon("static_AA", aa_scores, aa_labels, is_prob=False))

    # Katz (truncated)
    kz_scores = [score_bin_with(lambda u, v: _katz_truncated_score_pairs(A_bin, u, v, beta=katz_beta, max_hops=katz_hops), t)
                 for t in test_bins]
    kz_labels = [y_labels[t] for t in test_bins]
    rows.append(_eval_scores_over_horizon(f"static_Katz_L{katz_hops}", kz_scores, kz_labels, is_prob=False))

    # --- Static graph + Logistic Regression (probabilistic)
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
    except Exception as e:
        LogisticRegression = None

    if LogisticRegression is not None:
        # Train dataset from TRAIN bins using the same candidate risk sets
        train_bins = list(range(0, T_train))
        u_tr, v_tr, y_tr = _sample_train_pairs_from_risk_sets(
            risk_sets=[risk_sets[t] for t in train_bins],
            label_sets=[y_labels[t] for t in train_bins],
            max_pos_per_bin=lr_max_pos_per_bin,
            neg_ratio=lr_neg_ratio,
            seed=seed,
        )

        if u_tr.size > 0:
            X_tr = _static_lr_features(
                A=A, A_bin=A_bin, u=u_tr, v=v_tr,
                use_katz=True, katz_beta=katz_beta, katz_hops=katz_hops,
            )

            clf = Pipeline([
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("lr", LogisticRegression(
                    max_iter=200,
                    class_weight="balanced",
                    solver="lbfgs",
                )),
            ])
            clf.fit(X_tr, y_tr)

            lr_scores = []
            lr_labels = []
            for t in test_bins:
                u, v = _unpack_risk_t(risk_sets[t])
                X_t = _static_lr_features(A=A, A_bin=A_bin, u=u, v=v, use_katz=True,
                                         katz_beta=katz_beta, katz_hops=katz_hops)
                p = clf.predict_proba(X_t)[:, 1].astype(np.float32, copy=False)
                lr_scores.append(p)
                lr_labels.append(y_labels[t])

            rows.append(_eval_scores_over_horizon("static_LR(CN,AA,Katz,...)", lr_scores, lr_labels, is_prob=True))
        else:
            rows.append({"split": "test", "method": "static_LR(CN,AA,Katz,...)", "note": "no positive training samples"})
    else:
        rows.append({"split": "test", "method": "static_LR(CN,AA,Katz,...)", "note": "sklearn not available"})

    # --- Dyad EWMA baseline (temporal-only link score)
    ew = DyadEWMABaseline(num_nodes=num_nodes, decay=dyad_decay)

    # warm up on TRAIN bins
    for t in range(T_train):
        ew.step(y_bins_csr[t])

    # evaluate on TEST bins: for each bin, score BEFORE stepping with y_t
    ew_scores = []
    ew_labels = []
    for t in test_bins:
        u, v = _unpack_risk_t(risk_sets[t])
        ew_scores.append(ew.score_pairs(u, v))
        ew_labels.append(y_labels[t])
        ew.step(y_bins_csr[t])

    rows.append(_eval_scores_over_horizon(f"dyad_EWMA(decay={dyad_decay})", ew_scores, ew_labels, is_prob=False))

    return rows


def run_node_forecasting_baselines(
    y_bins_csr: Sequence[sp.csr_matrix],
    split: Tuple[int, int],   # (T_train, T_val) with test = [T_val..T)
    num_nodes: int,
    ema_alpha: float = 0.9,
) -> List[dict]:
    """
    Node forecasting baselines:
      - AR(1) mean reverting baseline you already have in eval.py
    Returns rows with NMSE + nodewise correlations.

    (IFT node forecasts should be evaluated separately via your existing IF node rollout utilities;
     this just provides the baselines so you can compare in the same table.)
    """
    T_train, T_val = split
    T = len(y_bins_csr)

    y_train = y_bins_csr[:T_train]
    y_test = y_bins_csr[T_val:]

    # Build EMA node series (true)
    # prepare_series returns: u_train_in, x_train_true, u_hold_in, x_hold_true, x_train_counts, var_ref
    _, X_tr_true, _, X_te_true, _, _ = prepare_series(y_train, y_test, num_nodes, ema_alpha)

    # Fit AR(1) on TRAIN
    # ar1_baseline_from_train returns (alpha_hat, mu)
    alpha_hat, mu = ar1_baseline_from_train(X_tr_true, num_nodes)
    X_ar = rollout_ar1(mu, alpha_hat, horizon=X_te_true.shape[0], N=num_nodes)

    m = eval_rollout(X_true=X_te_true, X_pred=X_ar)
    return [{
        "split": "test",
        "method": f"node_AR1(alpha={alpha_hat:.3f})",
        **m
    }]


# ------------------------------------------------------------
# Example usage (minimal)
# ------------------------------------------------------------

def example_run_all(
    y_bins_csr: Sequence[sp.csr_matrix],
    risk_sets: Sequence,
    num_nodes: int,
    T_train: int,
    T_val: int,
):
    """
    Example orchestration:
      - link prediction baselines
      - node forecasting baseline
    """
    split = (T_train, T_val)

    link_rows = run_link_prediction_static_baselines(
        y_bins_csr=y_bins_csr,
        risk_sets=risk_sets,
        split=split,
        num_nodes=num_nodes,
        horizon=1,
        k=200,
        static_spec=StaticGraphSpec(
            window_bins=None,
            weight_mode="count",
            decay_tau_bins=50.0,
            undirected=True,
            include_self_loops=False,
        ),
        katz_beta=0.05,
        katz_hops=3,
        dyad_decay=0.98,
        lr_neg_ratio=5.0,
        lr_max_pos_per_bin=2000,
        seed=0,
    )

    node_rows = run_node_forecasting_baselines(
        y_bins_csr=y_bins_csr,
        split=split,
        num_nodes=num_nodes,
        ema_alpha=0.9,
    )

    return link_rows, node_rows
