from __future__ import annotations

# ==========================
# Imports & type aliases
# ==========================
from contextlib import contextmanager
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, average_precision_score
import math
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression

from interactionfields.core import (
    forward_rollout_if,
    IFParameters,
    IFConfig,
    csr_bins_to_uv_tensors,
    reduce_H_to_nodes,
    build_L_eff_from_train,  # keep this in core for reuse
)
from interactionfields.plot_results import grid_edges, plot_rollout

Pair = Tuple[int, int]
Tensor = torch.Tensor

# ==========================
# Small general helpers
# ==========================
def best_grid_factors(num_nodes: int) -> Tuple[int, int]:
    """Pick (H, W) with H*W == num_nodes and H as large as possible (≈ square)."""
    h = int(math.sqrt(num_nodes))
    while h > 1 and num_nodes % h != 0:
        h -= 1
    w = num_nodes // h
    return h, w


def _to_tensor(x: Any, device: torch.device, dtype: Optional[torch.dtype] = None) -> Tensor:
    """Robustly convert x to a contiguous torch.Tensor on device."""
    if x is None:
        raise ValueError("Cannot convert None to tensor.")
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    if dtype is None:
        dtype = torch.float32 if t.dtype.is_floating_point else t.dtype
    return t.to(device=device, dtype=dtype).contiguous()


def _to_numpy_1d(x: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Return a 1-D CPU NumPy array from torch/np/list/tuple."""
    try:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except Exception:
        pass
    arr = np.asarray(x)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


# ==========================
# Per-node affine calibration (moved to eval)
# ==========================
def fit_affine_calibration_per_node(
    H_train: np.ndarray,
    X_train_true: np.ndarray,
    *,
    ridge: float = 1e-6,
    max_lag: int = 2,
    use_lag: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Fit per-node affine calibration with optional lag:

        X_true[:, i] ≈ alpha[i] + beta[i] * H_train[:, i]

    Parameters
    ----------
    H_train : (T, N) predicted node series (from rollout).
    X_train_true : (T, N) ground-truth node series.
    ridge : float, ridge penalty on beta for stability.
    max_lag : int, search lag in [-max_lag, +max_lag].
    use_lag : bool, enable lag search via correlation.

    Returns
    -------
    dict with:
        alpha : (N,) intercept per node
        beta  : (N,) slope per node
        lag   : (N,) best lag per node (positive means H leads X)
    """
    T, N = H_train.shape
    if X_train_true.shape != (T, N):
        raise ValueError(f"Shape mismatch: H_train {H_train.shape}, X_train_true {X_train_true.shape}")

    alpha = np.zeros(N, dtype=float)
    beta = np.zeros(N, dtype=float)
    lags = np.zeros(N, dtype=int)

    def _best_lag_xcorr(a: np.ndarray, b: np.ndarray, max_l: int) -> int:
        """
        Find lag in [-max_l, max_l] that maximizes Pearson correlation
        between a[t] and b[t+lag]. Returns the best lag (int).
        """
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        T = min(len(a), len(b))

        if T < 3:  # not enough data
            return 0

        # demean for stability
        a = a - np.nanmean(a)
        b = b - np.nanmean(b)

        best_lag, best_corr = 0, -np.inf
        for L in range(-max_l, max_l + 1):
            if L > 0 and (T - L) > 2:
                x, y = a[:-L], b[L:]
            elif L < 0 and (T + L) > 2:
                x, y = a[-L:], b[:L]
            elif L == 0:
                x, y = a, b
            else:
                continue

            # safe correlation
            if np.std(x) > 1e-8 and np.std(y) > 1e-8:
                corr = float(np.corrcoef(x, y)[0, 1])
            else:
                corr = np.nan

            if np.isfinite(corr) and corr > best_corr:
                best_lag, best_corr = L, corr

        return best_lag

    for i in range(N):
        h = H_train[:, i]
        y = X_train_true[:, i]

        L = _best_lag_xcorr(h, y, max_lag) if (use_lag and T > 2) else 0
        lags[i] = L

        if L > 0:
            h_al, y_al = h[L:], y[L:]
        elif L < 0:
            h_al, y_al = h[:L], y[:L]
        else:
            h_al, y_al = h, y

        if h_al.size < 3:
            alpha[i], beta[i] = y.mean(), 0.0
            continue

        Xd = np.vstack([h_al, np.ones_like(h_al)]).T
        R = np.diag([ridge, 0.0])
        A = Xd.T @ Xd + R
        b = Xd.T @ y_al
        try:
            theta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            theta = np.linalg.lstsq(A, b, rcond=None)[0]
        beta[i], alpha[i] = float(theta[0]), float(theta[1])

    return {"alpha": alpha, "beta": beta, "lag": lags}


def apply_affine_calibration(
    H: np.ndarray,
    calib: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, slice]:
    """
    Apply per-node affine calibration with per-node lags.

    Parameters
    ----------
    H : (T, N) raw predicted node series.
    calib : dict with keys "alpha", "beta", "lag".

    Returns
    -------
    X_hat : (T_eff, N) calibrated, lag-aligned predictions.
    sl    : slice applied to time axis to align all nodes.
    """
    T, N = H.shape
    alpha = calib["alpha"]
    beta = calib["beta"]
    lags = calib["lag"]

    Lmin = max(0, int(lags.max(initial=0)))
    Rmin = max(0, int(-lags.min(initial=0)))
    sl = slice(Lmin, None if Rmin == 0 else -Rmin)

    Hs = H[sl]
    X_hat = beta[None, :] * Hs + alpha[None, :]
    return X_hat, sl


@torch.no_grad()
def calibrate_if_free_affine_on_train_tail(
    theta,
    cfg,
    y_train_csr,
    num_nodes: int,
    dt_if: float,
    ema_alpha: float,
    *,
    mode: str = "zero",   # "zero" → if-free, "self" → free-self
    T_cal: int | None = None,
    max_lag: int = 6,
) -> dict:
    """
    Warm up on TRAIN (driven) → run a short free/self segment on TRAIN tail →
    map λ→nodes (EMA) → fit per-node affine (with lags) to TRAIN EMA truth.
    Returns calib dict with keys {'alpha','beta','lag'}.
    """
    device = cfg.device if isinstance(cfg.device, torch.device) else torch.device(cfg.device)

    # 1) teacher-forced warm-up (so free starts from a realistic state)
    y_tr_uv = csr_bins_to_uv_tensors(y_train_csr, device=device)
    H_tr, _, _, _, _, mem_tr = forward_rollout_if(
        y_tr_uv, theta, cfg, free_start=None, free_mode="driven", enable_progress=False
    )
    h_last = H_tr[-1].detach()
    mem_last = mem_tr

    # 2) short free/self segment on TRAIN tail
    if T_cal is None:
        T_cal = min(50, len(y_train_csr))
    empty_uv = [torch.empty(2, 0, dtype=torch.long, device=device) for _ in range(T_cal)]
    free_mode = "self" if mode == "self" else "zero"

    _, _, _, rs_cal, lams_cal, _ = forward_rollout_if(
        empty_uv, theta, cfg,
        free_start=0, free_mode=free_mode,
        h0=h_last, mem_init_sparse=mem_last,
        enable_progress=False,
    )

    # 3) λ → node EMA on the same TRAIN tail bins; fit affine
    x_cal = lambdas_to_node_series_if(lams_cal, rs_cal, num_nodes, dt_if, ema_alpha=ema_alpha)
    x_true_tail = node_series_from_csr_bins(y_train_csr[-T_cal:], num_nodes, mode="recv_ema", ema_alpha=ema_alpha)

    calib = fit_affine_calibration_per_node(
        x_cal, x_true_tail, ridge=1e-6, max_lag=max_lag, use_lag=True
    )
    return calib



# ==========================
# Node time-series prep
# ==========================
def node_series_from_csr_bins(
    csr_bins: Sequence[csr_matrix],
    num_nodes: int,
    *,
    mode: str = "recv_count",
    ema_alpha: float = 0.2,
    dtype: np.dtype = np.dtype(np.float32),
) -> np.ndarray:
    """
    Build X[t, i] from directed CSR per-bin adjacency A_t (u->v = 1).

    Modes
    -----
      - 'recv_count': per-bin incoming counts (column-sum of A_t)
      - 'src_count' : per-bin outgoing counts (row-sum of A_t)
      - 'recv_ema'  : EMA over time of 'recv_count' with smoothing alpha
    """
    T = len(csr_bins)
    X = np.zeros((T, num_nodes), dtype=dtype)

    if mode in ("recv_count", "recv_ema"):
        for t, A in enumerate(csr_bins):
            if A.nnz:
                X[t] = np.asarray(A.sum(axis=0)).ravel().astype(dtype, copy=False)
    elif mode == "src_count":
        for t, A in enumerate(csr_bins):
            if A.nnz:
                X[t] = np.asarray(A.sum(axis=1)).ravel().astype(dtype, copy=False)
    else:
        raise ValueError("mode must be one of {'recv_count','recv_ema','src_count'}")

    if mode == "recv_ema":
        Y = np.zeros_like(X)
        a = float(ema_alpha)
        y = np.zeros(num_nodes, dtype=dtype)
        for t in range(T):
            y = (1.0 - a) * X[t] + a * y
            Y[t] = y
        X = Y
    return X


def prepare_series(
    y_train_csr: Sequence[csr_matrix],
    y_holdout_csr: Sequence[csr_matrix],
    num_nodes: int,
    ema_alpha: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Prepare input/target node series for train/holdout splits.

    Returns
    -------
    u_train_in   : (T_tr, N) recv_count on TRAIN
    x_train_true : (T_tr, N) recv_ema  on TRAIN
    u_hold_in    : (T_ho, N) recv_count on HOLDOUT
    x_hold_true  : (T_ho, N) recv_ema  on HOLDOUT
    x_train_cnts : (T_tr, N) alias of recv_count on TRAIN
    var_ref      : float     variance reference for NMSE
    """
    u_train_in = node_series_from_csr_bins(y_train_csr, num_nodes, mode="recv_count", ema_alpha=ema_alpha)
    x_train_true = node_series_from_csr_bins(y_train_csr, num_nodes, mode="recv_ema", ema_alpha=ema_alpha)
    u_hold_in = node_series_from_csr_bins(y_holdout_csr, num_nodes, mode="recv_count", ema_alpha=ema_alpha)
    x_hold_true = node_series_from_csr_bins(y_holdout_csr, num_nodes, mode="recv_ema", ema_alpha=ema_alpha)

    x_train_counts = u_train_in.copy()
    var_ref = float(np.var(x_train_true)) + 1e-8
    return u_train_in, x_train_true, u_hold_in, x_hold_true, x_train_counts, var_ref


def lambdas_to_node_series_if(
    lambda_list: List[Tensor],
    risk_sets: List[List[Pair]],
    N: int,
    dt: float,
    *,
    ema_alpha: Optional[float] = None,
) -> np.ndarray:
    """
    Aggregate per-edge hazards λ over receiver nodes to get node series.

    For each t:
        expected recv count at node v ≈ sum_{(u,v) in risk_set} λ_{uv}^t * dt
    """
    T = len(lambda_list)
    U = np.zeros((T, N), dtype=np.float32)
    for t, (lam_t, R_t) in enumerate(zip(lambda_list, risk_sets)):
        if len(R_t) == 0:
            x = np.zeros(N, dtype=np.float32)
        else:
            pairs = torch.as_tensor(R_t, device=lam_t.device, dtype=torch.long).T  # (2, M)
            v = pairs[1]
            exp_counts = (lam_t * dt).to(torch.float32)
            node_counts = torch.zeros(N, device=lam_t.device, dtype=torch.float32)
            node_counts.index_add_(0, v, exp_counts)
            x = node_counts.detach().cpu().numpy()
        if ema_alpha is not None:
            prev = U[t - 1] if t > 0 else 0.0
            U[t] = ema_alpha * x + (1.0 - ema_alpha) * prev
        else:
            U[t] = x
    return U


# ==========================
# Metrics (nodes)
# ==========================
def _nodewise_corr(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    *,
    eps: float = 1e-8,
    min_T: int = 3,
) -> Tuple[float|None, float|None]:
    """Median/mean Pearson across nodes with low-variance masking."""
    T, N = X_true.shape
    if T < min_T:
        return None, None
    Xt = X_true - X_true.mean(axis=0, keepdims=True)
    Xp = X_pred - X_pred.mean(axis=0, keepdims=True)
    st = Xt.std(axis=0, ddof=1)
    sp = Xp.std(axis=0, ddof=1)
    mask = (st > eps) & (sp > eps)
    if not np.any(mask):
        return None, None
    cov = (Xt[:, mask] * Xp[:, mask]).sum(axis=0) / max(T - 1, 1)
    corr = np.clip(cov / (st[mask] * sp[mask] + eps), -1.0, 1.0)
    return float(np.median(corr)), float(np.mean(corr))


def eval_rollout(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    *,
    var_ref: Optional[float] = None,
    eps: float = 1e-8,
    min_T_for_corr: int = 3,
) -> Dict[str, Optional[float]]:
    """
    Compare predicted vs. true node series with NMSE and nodewise correlations.
    """
    X_true = np.asarray(X_true, dtype=float)
    X_pred = np.asarray(X_pred, dtype=float)
    if X_true.shape != X_pred.shape or X_true.ndim != 2:
        raise ValueError(f"Shape mismatch: {X_true.shape} vs {X_pred.shape}")
    T, N = X_true.shape

    denom = float(X_true.var()) if var_ref is None else float(var_ref)
    denom = max(denom, eps)
    nmse = float(((X_true - X_pred) ** 2).mean()) / denom
    corr_med, corr_mean = _nodewise_corr(X_true, X_pred, eps=eps, min_T=min_T_for_corr)
    return dict(nmse=nmse, node_corr_median=corr_med, node_corr_mean=corr_mean)


# ==========================
# Simple temporal baseline
# ==========================
def ar1_baseline_from_train(X_tr_true: np.ndarray, N: int) -> Tuple[float, np.ndarray]:
    """
    Fit mean-reverting AR(1) on TRAIN EMA (global α shared across nodes).
    Returns (alpha_hat, mu_vector) where mu is the last TRAIN level.
    """
    mu = X_tr_true[-1].copy()
    if len(X_tr_true) >= 2:
        Xt, Xt1 = (X_tr_true[:-1] - mu).reshape(-1, N), (X_tr_true[1:] - mu).reshape(-1, N)
        num = float((Xt * Xt1).sum())
        den = float((Xt * Xt).sum()) + 1e-12
        alpha_hat = max(0.0, min(1.0, num / den))
    else:
        alpha_hat = 1.0
    return alpha_hat, mu


def rollout_ar1(mu: np.ndarray, alpha_hat: float, horizon: int, N: int) -> np.ndarray:
    """Rollout AR(1) baseline for h steps."""
    X = np.zeros((horizon, N), dtype=np.float32)
    x = mu.copy()
    for t in range(horizon):
        x = mu + alpha_hat * (x - mu)
        X[t] = x
    return X


# ==========================
# Laplacian/eigen rollouts (node-level diffusion surrogates)
# ==========================
@torch.no_grad()
def prep_eigendecomp(L_eff: Tensor) -> Tuple[Tensor, Tensor]:
    """Return (U, lam) with U orthonormal and lam eigenvalues of symmetric L_eff."""
    Ld = L_eff.to_dense() if L_eff.is_sparse else L_eff
    Ld = 0.5 * (Ld + Ld.T).float()
    lam, U = torch.linalg.eigh(Ld)
    return U, lam


def modal_decay(lam: Tensor, dt: float) -> Tensor:
    """Per-mode multiplier exp(-dt * lam) (lam ≥ 0 for Laplacian-like ops)."""
    lam = torch.clamp(lam, min=0.0)
    return torch.exp(-dt * lam)



# --- small helper: temporarily override attributes on cfg ---
@contextmanager
def _override(obj, **updates):
    old = {k: getattr(obj, k, None) for k in updates}
    try:
        for k, v in updates.items():
            setattr(obj, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


def _empty_uv_bins(T: int, device: torch.device) -> List[torch.Tensor]:
    """Create T empty (2,0) Long tensors on device (i.e., ‘no events’ bins)."""
    return [torch.empty(2, 0, dtype=torch.long, device=device) for _ in range(T)]


@torch.no_grad()
def eval_node_preds_from_Hseq(
    H_seq: Union[List[torch.Tensor], torch.Tensor],
    *,
    theta: IFParameters,
    cfg: IFConfig,
    num_nodes: int | None = None,
    mu: torch.Tensor | None = None,
    sig: torch.Tensor | None = None,
    reduce_mode: str = "sym",
    drop_first_if_vector: bool = True,
    center: bool = False,
    node_gain: torch.Tensor | None = None,
    node_bias: torch.Tensor | None = None,
    use_tanh: bool = False,
) -> torch.Tensor:
    """
    Reduce latent H[t] (N,d) -> node series X[t,N], apply the same readout used in training,
    then invert the training z-normalization to get back to EMA/count space.

    Steps (mirrors train_if):
      1) X_nodes = reduce_H_to_nodes(...)
      2) x_hat   = (optional) per-node gain/bias head
      3) z_pred  = (x_hat - z_shift) / softplus(z_scale)
      4) counts  = z_pred * readout_sig + readout_mu

    Returns: Tensor of shape [T, N] on cfg.device (float32).
    """
    device = cfg.device
    # Accept either a list of per-step tensors or a stacked tensor [T, ...].
    if isinstance(H_seq, torch.Tensor):
        H_seq = list(torch.unbind(H_seq, dim=0))
    N = num_nodes if num_nodes is not None else int(theta.h0.shape[0])

    # 1) latent -> nodes
    X_nodes = reduce_H_to_nodes(
        H_seq, N, device,
        drop_first_if_vector=drop_first_if_vector,
        reduce_mode=reduce_mode,
        center=center,
    )  # [T, N], float32 on device

    # 2) (optional) linear head
    x_hat = X_nodes
    gn = node_gain if node_gain is not None else getattr(theta, "node_gain", None)
    bs = node_bias if node_bias is not None else getattr(theta, "node_bias", None)
    if gn is not None:
        g = gn.view(1, -1) if getattr(cfg, "node_readout_per_node", False) else gn
        x_hat = x_hat * g
    if bs is not None:
        x_hat = x_hat + bs.view(1, -1)

    # 3) training z-standardizer (use learned params if present)
    z_scale_val = getattr(theta, "z_scale", None)
    z_shift_val = getattr(theta, "z_shift", None)
    if isinstance(z_scale_val, torch.Tensor) and isinstance(z_shift_val, torch.Tensor):
        den = 1e-6 + F.softplus(z_scale_val).view(1, -1)
        z_pred = (x_hat - z_shift_val.view(1, -1)) / den
    else:
        # Back-compat fallback if older checkpoints have no z_* params
        scale = x_hat.abs().mean(dim=0, keepdim=True).clamp_min(1.0)
        z_pred = x_hat / scale

    if use_tanh:
        z_pred = torch.tanh(z_pred)

    # 4) invert to counts using TRAIN buffers (mu,sig)
    # Ensure mu/sig are tensors on the correct device and shaped (1, N).
    def _to_row_tensor(x, default, name: str):
        if x is None:
            val = getattr(theta, name, None)
        else:
            val = x
        if val is None:
            return torch.tensor(default, dtype=torch.float32, device=device).view(1, -1)
        if isinstance(val, torch.Tensor):
            return val.to(device=device, dtype=torch.float32).view(1, -1)
        return torch.as_tensor(val, dtype=torch.float32, device=device).view(1, -1)

    mu_t = _to_row_tensor(mu, np.zeros((N,), dtype=np.float32), "readout_mu")
    sig_t = _to_row_tensor(sig, np.ones((N,), dtype=np.float32), "readout_sig")
    counts = z_pred * sig_t + mu_t
    return counts.contiguous()


@torch.no_grad()
def rollout_linear_driven(
    L_eff: Tensor,
    U_true: Union[np.ndarray, Tensor],
    *,
    x0: Optional[Union[np.ndarray, Tensor]] = None,
    dt: float = 1.0,
) -> Tensor:
    """
    Linear driven surrogate in the eigenbasis:
        z_{t+1} = exp(-dt * Λ) ⊙ z_t
        x_{t+1} = U z_{t+1} + u_t

    Args
    ----
    L_eff : (N,N) torch.Tensor (on target device)
    U_true: (T,N) array/tensor of node-wise input per step
    x0    : optional (N,) initial node state; if None, zeros
    dt    : time step for the modal decay

    Returns
    -------
    X : (T, N) torch.float32 on L_eff.device
    """
    device = L_eff.device
    U, lam = prep_eigendecomp(L_eff)       # (N,N), (N,)
    a = modal_decay(lam, float(dt))        # (N,)

    U_true_t = _to_tensor(U_true, device, dtype=torch.float32)
    T, N = U_true_t.shape
    x = torch.zeros(N, device=device, dtype=torch.float32) if x0 is None else _to_tensor(x0, device, dtype=torch.float32)
    x = x.view(N)

    Ut = U.T
    out = []
    for t in range(T):
        z = Ut @ x          # modal coords
        z = a * z           # decay
        x = U @ z + U_true_t[t]
        out.append(x.unsqueeze(0))
    return torch.cat(out, dim=0)


@torch.no_grad()
def rollout_if_nodes_driven(
    theta: IFParameters,
    cfg: IFConfig,
    y_bins_csr: List[csr_matrix],
    *,
    reduce_mode: str = "sym",
    x0: Optional[Union[np.ndarray, Tensor]] = None,   # optional latent init h0
    dt: Optional[float] = None,                       # optional dt override
    mem0: Optional[Tensor] = None,                    # optional sparse COO memory seed
) -> Tuple[np.ndarray, List[List[Tuple[int,int]]], List[Tensor], Tensor]:
    """
    Driven IF rollout (teacher-forced edges) with optional latent h0 and dt override.
    Returns the same outputs as before; node readout matches training.

    x0: (N,) or (N,d) in latent space; if 1D and d>1, it is broadcast across d.
    dt: if provided, used just for this rollout via a shallow cfg copy.
    """
    device = cfg.device if isinstance(cfg.device, torch.device) else torch.device(cfg.device)

    # Optional dt override (no mutation of the original cfg)
    cfg_use = replace(cfg, dt=float(dt)) if dt is not None else cfg

    # CSR -> (2, M_t) Long on device
    y_uv = csr_bins_to_uv_tensors(y_bins_csr, device=device)

    # Optional latent init
    h0: Optional[Tensor] = None
    if x0 is not None:
        N, d = theta.h0.shape
        x0_t = x0 if isinstance(x0, torch.Tensor) else torch.as_tensor(x0)
        x0_t = x0_t.to(device=device, dtype=torch.float32).contiguous()
        if x0_t.dim() == 1:
            if x0_t.numel() != N:
                raise ValueError(f"x0 has {x0_t.numel()} elements but expected N={N}")
            h0 = x0_t.view(N, 1).expand(N, d).contiguous()
        elif x0_t.dim() == 2:
            if x0_t.shape != (N, d):
                raise ValueError(f"x0 shape {tuple(x0_t.shape)} must be (N,d)=({N},{d})")
            h0 = x0_t
        else:
            raise ValueError("x0 must be 1D (N,) or 2D (N,d)")

    # Driven rollout (teacher-forced edges)
    H, J_list, K_list, risk_sets, lambda_list, M_final = forward_rollout_if(
        y_uv, theta, cfg_use,
        free_start=None, free_mode="driven",
        h0=h0, mem_init_sparse=mem0,
        enable_progress=False,
    )

    # Same readout/standardization as training
    X_nodes_t = eval_node_preds_from_Hseq(
        H, theta=theta, cfg=cfg_use,
        num_nodes=theta.h0.shape[0],
        mu=theta.readout_mu, sig=theta.readout_sig,
        reduce_mode=reduce_mode,
        node_gain=getattr(theta, "node_gain", None),
        node_bias=getattr(theta, "node_bias", None),
        use_tanh=False,
    )
    return X_nodes_t.cpu().numpy(), risk_sets, lambda_list, M_final


@torch.no_grad()
def rollout_if_nodes_free(
    theta,
    cfg,
    *,
    T: int,
    h0: Optional[torch.Tensor],
    mem0_sparse: Optional[torch.Tensor],
    reduce_mode: str = "sym",
) -> Tuple[np.ndarray, List[List[Tuple[int,int]]], List[torch.Tensor], torch.Tensor]:
    """
    Free diffusion with no new edges (memory only decays).
    Start from (h0, mem0_sparse) you got after training rollout.
    """
    device = cfg.device if isinstance(cfg.device, torch.device) else torch.device(cfg.device)
    y_empty = _empty_uv_bins(T, device)
    H, J_list, K_list, risk_sets, lambda_list, M_final = forward_rollout_if(
        y_empty, theta, cfg,
        free_start=0, free_mode="zero",
        h0=h0, mem_init_sparse=mem0_sparse,
        enable_progress=False,
    )
    X_nodes_t = eval_node_preds_from_Hseq(
        H, theta=theta, cfg=cfg,
        num_nodes=theta.h0.shape[0], mu=theta.readout_mu, sig=theta.readout_sig,
        reduce_mode=reduce_mode,
        node_gain=getattr(theta, "node_gain", None),
        node_bias=getattr(theta, "node_bias", None),
        use_tanh=False,
    )
    return X_nodes_t.cpu().numpy(), risk_sets, lambda_list, M_final

@torch.no_grad()
def rollout_if_nodes_self(
    theta,
    cfg,
    *,
    T: int,
    h0: Optional[torch.Tensor],
    mem0_sparse: torch.Tensor,
    hazard_tau: Optional[Union[float, List[float], np.ndarray]] = None,  # scalar or per-timestep τ̂
    reduce_mode: str = "sym",
) -> Tuple[np.ndarray, List[List[Tuple[int, int]]], List[torch.Tensor], torch.Tensor]:
    """
    Closed-loop ('self') rollout: sample edges from IF hazards and evolve.
    If hazard_tau is provided:
      - scalar: use the same τ for all steps: p = 1 - exp(-τ * λ * dt)
      - sequence/ndarray of length T: use τ[t] at step t
    Returns:
      X_nodes_t:  [T, N] node scalars
      risk_sets:  List per-step risk sets
      lambda_list: List per-step hazards (torch tensors as returned by forward_rollout_if)
      M_final:    final sparse memory tensor
    """
    device = cfg.device if isinstance(cfg.device, torch.device) else torch.device(cfg.device)

    # Fast path: scalar τ (original behavior). Explicitly handle None,
    # scalar (exclude complex), and per-timestep sequence to narrow types.
    if hazard_tau is None:
        y_empty = _empty_uv_bins(T, device)
        updates: Dict[str, float] = {}
        with _override(cfg, **updates):
            H, J_list, K_list, risk_sets, lambda_list, M_final = forward_rollout_if(
                y_empty, theta, cfg,
                free_start=0, free_mode="self",
                h0=h0, mem_init_sparse=mem0_sparse,
                enable_progress=False,
            )
    elif np.isscalar(hazard_tau) and not isinstance(hazard_tau, complex):
        # numpy scalar or Python number (but not complex): safe to cast to float
        y_empty = _empty_uv_bins(T, device)
        updates = {"hazard_tau": float(hazard_tau)}
        with _override(cfg, **updates):
            H, J_list, K_list, risk_sets, lambda_list, M_final = forward_rollout_if(
                y_empty, theta, cfg,
                free_start=0, free_mode="self",
                h0=h0, mem_init_sparse=mem0_sparse,
                enable_progress=False,
            )
        X_nodes_t = eval_node_preds_from_Hseq(
            H, theta=theta, cfg=cfg,
            num_nodes=theta.h0.shape[0], mu=theta.readout_mu, sig=theta.readout_sig,
            reduce_mode=reduce_mode,
            node_gain=getattr(theta, "node_gain", None),
            node_bias=getattr(theta, "node_bias", None),
            use_tanh=False,
        )
        return X_nodes_t.cpu().numpy(), risk_sets, lambda_list, M_final

    # Per-timestep τ̂: iterate step-by-step so each step uses its own tau
    tau_arr = np.asarray(hazard_tau, dtype=np.float32).ravel()
    if tau_arr.shape[0] != T:
        raise ValueError(f"hazard_tau length {tau_arr.shape[0]} != T {T}")

    H_last = h0
    M_last = mem0_sparse

    H_collect: List[torch.Tensor] = []
    risk_sets_all: List[List[Tuple[int, int]]] = []
    lambda_all: List[torch.Tensor] = []

    # step-wise empty input of length 1
    def _empty_step_bins(device_):
        return _empty_uv_bins(1, device_)

    for t in range(T):
        y_step = _empty_step_bins(device)
        with _override(cfg, hazard_tau=float(tau_arr[t])):
            H_step, J_list, K_list, risk_sets, lambda_list, M_final = forward_rollout_if(
                y_step, theta, cfg,
                free_start=0, free_mode="self",
                h0=H_last, mem_init_sparse=M_last,
                enable_progress=False,
            )
        # H_step is a sequence over length-1 rollout; take last state
        H_last = H_step[-1].detach()
        M_last = M_final

        # collect for readout & metrics
        H_collect.append(H_last)
        risk_sets_all.append(risk_sets[0])
        lambda_all.append(lambda_list[0])

    # stack hidden states over time and produce node preds
    H_seq = torch.stack(H_collect, dim=0)  # [T, ...hidden...]
    X_nodes_t = eval_node_preds_from_Hseq(
        H_seq, theta=theta, cfg=cfg,
        num_nodes=theta.h0.shape[0], mu=theta.readout_mu, sig=theta.readout_sig,
        reduce_mode=reduce_mode,
        node_gain=getattr(theta, "node_gain", None),
        node_bias=getattr(theta, "node_bias", None),
        use_tanh=False,
    )
    return X_nodes_t.cpu().numpy(), risk_sets_all, lambda_all, M_last




@torch.no_grad()
def rollout_node_head(node_head: torch.nn.Module, u_in: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """Simulate the NodeHeadLinear block forward in time given input drive u_in."""
    param = next(node_head.parameters(), None)
    if param is None:
        buf = next(node_head.buffers(), None)
        device = buf.device if buf is not None else torch.device("cpu")
    else:
        device = param.device
    u = torch.tensor(u_in, dtype=torch.float32, device=device)
    x_t = torch.tensor(x0, dtype=torch.float32, device=device)
    T, _ = u.shape
    out = []
    for t in range(T):
        out.append(x_t.unsqueeze(0))
        x_t = node_head(x_t, u[t]).squeeze(0)
    return torch.cat(out, dim=0).cpu().numpy()


# ==========================
# FREE-SELF helpers (IF forward model)
# ==========================
def tune_sample_cap(
    theta: IFParameters,
    cfg: IFConfig,
    h_last: Tensor,
    mem_last: Tensor,
    target_mean: float,
    N: int,
    dt_if: float,
    *,
    caps: Optional[List[float]] = None,
    T_cal: Optional[int] = None,
) -> float:
    """
    Grid-search a sample_cap so that the *self*-generated expected recv rate
    (mean per node per bin) matches target_mean.
    """
    device = cfg.device if isinstance(cfg.device, torch.device) else torch.device(cfg.device)
    caps = caps or [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
    T_cal = T_cal or 50

    def _empty_uv_bins(t_len: int) -> List[Tensor]:
        return [torch.empty(2, 0, dtype=torch.long, device=device) for _ in range(t_len)]

    best_cap, best_err = caps[0], float("inf")
    for cap in caps:
        with torch.no_grad():
            y_empty = _empty_uv_bins(T_cal)
            _, _, _, rs_tmp, lams_tmp, _ = forward_rollout_if(
                y_empty,
                theta,
                cfg,
                free_start=0,
                free_mode="self",
                sample_mode="bernoulli",
                sample_cap=cap,
                h0=h_last,
                mem_init_sparse=mem_last,
                enable_progress=False,
            )
        u_counts = lambdas_to_node_series_if(lams_tmp, rs_tmp, N, dt_if, ema_alpha=None)
        err = abs(float(u_counts.mean()) - target_mean)
        if err < best_err:
            best_err, best_cap = err, cap
    return best_cap


def estimate_self_rate(
    theta: IFParameters,
    cfg: IFConfig,
    h_last: Tensor,
    mem_last: Tensor,
    num_nodes: int,
    dt_if: float,
    *,
    T_rate: int = 100,
    sample_cap: float = 0.5,
) -> float:
    device = cfg.device if isinstance(cfg.device, torch.device) else torch.device(cfg.device)
    y_empty = [torch.empty(2, 0, dtype=torch.long, device=device) for _ in range(T_rate)]
    with torch.no_grad():
        _, _, _, rs_rate, lams_rate, _ = forward_rollout_if(
            y_empty,
            theta,
            cfg,
            free_start=0,
            free_mode="self",
            sample_mode="bernoulli",
            sample_cap=sample_cap,
            h0=h_last,
            mem_init_sparse=mem_last,
            enable_progress=False,
        )
    u_counts = lambdas_to_node_series_if(lams_rate, rs_rate, num_nodes, dt_if, ema_alpha=None)
    return float(u_counts.mean())


class temporary_b0_shift:
    """Context manager: temporarily add a shift to theta.b0 (no grad) and restore on exit."""
    def __init__(self, theta: IFParameters, shift: float):
        self.theta = theta
        self.shift = float(shift)
        self._orig: Optional[Tensor] = None

    def __enter__(self):
        self._orig = self.theta.b0.detach().clone()
        with torch.no_grad():
            self.theta.b0.copy_(self._orig + self.shift)

    def __exit__(self, exc_type, exc, tb):
        assert self._orig is not None
        with torch.no_grad():
            self.theta.b0.copy_(self._orig)


def fit_scalar_gamma(raw_np: np.ndarray, true_np: np.ndarray, clip: Tuple[float, float] = (0.0, 5.0)) -> float:
    """Least-squares scalar gain γ so that γ·raw ≈ true."""
    num = float((raw_np * true_np).sum())
    den = float((raw_np * raw_np).sum()) + 1e-12
    return float(np.clip(num / den if den > 0 else 0.0, *clip))


def fit_per_node_gamma(raw_np: np.ndarray, true_np: np.ndarray, clip: Tuple[float, float] = (0.0, 5.0)) -> np.ndarray:
    """Per-node gain γ_i so that γ_i·raw_i ≈ true_i."""
    num = (raw_np * true_np).sum(axis=0)
    den = (raw_np * raw_np).sum(axis=0) + 1e-12
    g = np.where(den > 0, num / den, 0.0)
    return np.clip(g, clip[0], clip[1])


# ==========================
# Edge metrics & conversions
# ==========================
def csr_to_set(csr: csr_matrix) -> set[Pair]:
    """Convert CSR to a set of (row, col) pairs with nonzero entries."""
    coo = csr.tocoo()
    return set(zip(coo.row.tolist(), coo.col.tolist()))


def make_labels_for_risk_sets(y_bins_csr: Sequence[csr_matrix], risk_sets: Sequence[List[Pair]]) -> List[np.ndarray]:
    """
    Build 0/1 label arrays y[t] aligned with risk_sets[t].
    """
    labels: List[np.ndarray] = []
    for t, pairs in enumerate(risk_sets):
        pos = csr_to_set(y_bins_csr[t])
        if not pairs:
            labels.append(np.zeros((0,), dtype=np.uint8))
            continue
        y = np.fromiter((1 if (u, v) in pos else 0 for (u, v) in pairs), dtype=np.uint8, count=len(pairs))
        labels.append(y)
    return labels

def probs_from_lambda_list(
    lambda_list: Sequence[np.ndarray | Tensor],
    *,
    dt: float = 1.0,
    tau: Union[float, List[float], np.ndarray] = 1.0,
    clip: Tuple[float, float] = (1e-6, 1.0 - 1e-6),
    out_dtype: np.dtype = np.dtype(np.float32)
) -> List[np.ndarray]:
    """
    Convert hazards λ to probabilities p = 1 - exp(-τ * λ * dt) per time step.

    Args:
        lambda_list: list of hazard arrays (E,) or (n_edges,) at each t
        dt: time step size
        tau: scalar (global temperature) or array/list of length len(lambda_list)
        clip: clamp probs into [clip[0], clip[1]]
        out_dtype: dtype for output arrays

    Returns:
        List[np.ndarray]: probabilities aligned with lambda_list
    """
    ps: List[np.ndarray] = []
    dt = float(dt)

    # normalize tau → np.ndarray if sequence
    if isinstance(tau, (int, float, np.floating)):
        tau_arr = None
        tau_scalar = float(tau)
    else:
        tau_arr = np.asarray(tau, dtype=np.float64).ravel()
        if tau_arr.shape[0] != len(lambda_list):
            raise ValueError(
                f"tau has length {tau_arr.shape[0]}, but lambda_list has length {len(lambda_list)}"
            )
        tau_scalar = None

    for t, lam_t in enumerate(lambda_list):
        lam = _to_numpy_1d(lam_t, dtype=np.dtype(np.float32))
        lam = np.clip(lam, 0.0, None)

        scale = tau_scalar if tau_arr is None else float(tau_arr[t])
        p = 1.0 - np.exp(-lam * scale * dt)
        p = np.clip(p, *clip).astype(out_dtype, copy=False)
        ps.append(p)

    return ps



def _unpack_risk_set(
    risk_t: Union[Tuple[Iterable[int], Iterable[int]], List[Pair], np.ndarray, Tensor]
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize a risk set into parallel integer arrays u, v (shape (K,))."""
    # (u, v)
    if isinstance(risk_t, tuple) and len(risk_t) == 2:
        u, v = risk_t
        return _to_numpy_1d(u, dtype=np.dtype(np.float64)), _to_numpy_1d(v, dtype=np.dtype(np.float32))

    # ndarray/Tensor (K,2)
    if isinstance(risk_t, (np.ndarray, torch.Tensor)):
        arr = np.asarray(risk_t) if isinstance(risk_t, np.ndarray) else risk_t.detach().cpu().numpy()
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr[:, 0].astype(np.int64, copy=False), arr[:, 1].astype(np.int64, copy=False)

    # list of (u, v)
    if isinstance(risk_t, list):
        if len(risk_t) == 0:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
        u, v = zip(*risk_t)
        return _to_numpy_1d(u, dtype=np.dtype(np.float32)), _to_numpy_1d(v, dtype=np.dtype(np.float32))

    return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)


def align_tau_to_length(
    tau: Union[float, List[float], np.ndarray],
    T: int,
    *,
    mode: str = "pad_last",   # "pad_last" | "repeat" | "interp"
    default_scalar: float = 1.0,
) -> Union[float, np.ndarray]:
    """
    Align a scalar or 1D sequence of tau-hats to a target length T.
    - If tau is scalar: return it unchanged.
    - If tau is 1D:
        * same length -> return as-is
        * longer      -> truncate
        * shorter     -> extend per `mode`:
            - "pad_last": pad with last value (recommended, no leakage)
            - "repeat":   tile cyclically (ok if τ is seasonal)
            - "interp":   linear resample to length T (smooth)
    """
    if isinstance(tau, (int, float, np.floating)):
        return float(tau)

    a = np.asarray(tau, dtype=np.float64).ravel()
    if a.size == 0:
        return float(default_scalar)
    if a.size == T:
        return a
    if a.size > T:
        return a[:T]

    # a.size < T → need to extend
    if mode == "pad_last":
        pad_val = a[-1]
        return np.pad(a, (0, T - a.size), mode="edge", constant_values=(a[0], pad_val))
    elif mode == "repeat":
        reps = int(np.ceil(T / a.size))
        out = np.tile(a, reps)[:T]
        return out
    elif mode == "interp":
        x_old = np.linspace(0.0, 1.0, a.size)
        x_new = np.linspace(0.0, 1.0, T)
        return np.interp(x_new, x_old, a)
    else:
        raise ValueError(f"Unknown tau align mode: {mode}")


def fit_hazard_temperature_per_bin(lambda_list_train, risk_sets_train, y_train, dt_hazard=1.0, window=3):
    """
    Fit τ per timestep bucket by smoothing over a small window on TRAIN.
    Returns a list tau_t of length T_train (or callable that maps t->tau).
    """
    T = min(len(lambda_list_train), len(risk_sets_train), len(y_train))
    taus = []
    for t in range(T):
        t0 = max(0, t - window)
        t1 = min(T, t + window + 1)
        lam_win = lambda_list_train[t0:t1]
        rs_win  = risk_sets_train[t0:t1]
        y_win   = y_train[t0:t1]
        tau = _fit_hazard_temperature(lam_win, rs_win, y_win, dt_hazard=dt_hazard)
        taus.append(float(tau))
    return taus


def _fit_hazard_temperature(
    lambda_list_train: List[Union[np.ndarray, Tensor]],
    risk_sets_train: List[List[Pair]],
    y_train_bins: List[csr_matrix],
    *,
    dt_hazard: float = 1.0,
    clip: Tuple[float, float] = (1e-6, 1.0 - 1e-6),
) -> float:
    """
    Choose τ so expected positives over TRAIN match true prevalence:
        sum_t sum_e [1 - exp(-τ * λ_t(e) * dt)] ≈ sum_t sum_e 1{edge fired}
    """
    # True prevalence over train risk edges
    true_cnt, tot_cnt = 0.0, 0.0
    for t, risk_t in enumerate(risk_sets_train):
        A = y_train_bins[t]
        u, v = _unpack_risk_set(risk_t)
        K = u.size
        if K == 0:
            continue
        hits = np.asarray(A[u, v]).ravel().sum()
        true_cnt += float(hits)
        tot_cnt += float(K)
    true_rate = true_cnt / max(tot_cnt, 1e-12)

    # Expected prevalence at τ
    def expected_rate(tau: float) -> float:
        exp_cnt, total = 0.0, 0.0
        for lam_t in lambda_list_train:
            lam = _to_numpy_1d(lam_t, dtype=np.dtype(np.float64))
            if lam.size == 0:
                continue
            p = 1.0 - np.exp(-np.clip(lam, 0.0, None) * tau * dt_hazard)
            p = np.clip(p, *clip)
            exp_cnt += float(p.sum())
            total += float(p.size)
        return exp_cnt / max(total, 1e-12)

    # Bracket and bisect
    lo, hi = 0.0, 1.0
    er_lo = expected_rate(lo)
    er_hi = expected_rate(hi)
    max_hi = 1e6
    while er_hi < true_rate and hi < max_hi:
        hi *= 2.0
        er_hi = expected_rate(hi)
    if er_hi < true_rate:
        return hi
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        er = expected_rate(mid)
        if er < true_rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def bins_from_probs(prob_list, risk_sets, N: int) -> List[csr_matrix]:
    """
    Build CSR matrices with values = probability for each (u,v) in risk_sets[t].

    Accepts:
      - prob_list[t]: array-like of shape (P_t,) or (P_t,1) or any flat-like
      - risk_sets[t]: (u,v) pairs in any supported format (tuple of arrays, list of pairs, ndarray/torch (K,2))

    Handles the common case where probs are for UNDIRECTED edges (length E) but
    risk_sets contain BIDIRECTIONAL pairs (length 2E) by repeating probs.
    """
    out = []
    for p_t, risk_t in zip(prob_list, risk_sets):
        # unpack risk set to parallel arrays u, v
        u, v = _unpack_risk_set(risk_t)  # returns np.int64 arrays of len K (may be 0)
        K = u.size
        if K == 0:
            out.append(csr_matrix((N, N), dtype=np.float32))
            continue

        # probs -> flat float array on CPU
        p = _to_numpy_1d(p_t, dtype=np.dtype(np.float64))  # shape (P,)
        # Allow column vectors etc.
        p = p.reshape(-1)

        P = p.size
        if P == K:
            p_use = p
        elif (P * 2) == K:
            # broadcast undirected probs to symmetric (u,v) and (v,u)
            # assumes risk_t was constructed by stacking [E, E[:,::-1]]
            p_use = np.tile(p, 2)
        elif K % P == 0 and (K // P) <= 4:
            # general small replication (defensive)
            p_use = np.tile(p, K // P)
        else:
            raise ValueError(
                f"probs and risk set are misaligned: probs len={P}, pairs K={K}. "
                f"Expected K == P or K == 2*P (bidirectional)."
            )

        # clip to [0,1] and cast
        p_use = np.clip(p_use, 0.0, 1.0).astype(np.float32, copy=False)
        assert p_use.size == K, f"Internal size mismatch after broadcast: {p_use.size} vs {K}"

        out.append(csr_matrix((p_use, (u, v)), shape=(N, N), dtype=np.float32))

    return out



def aggregate_edge_metrics_over_horizon(
    score_list: List[np.ndarray],
    label_list: List[np.ndarray],
    horizon: int,
    *,
    is_prob: bool = False,
    k=None, k_frac=0.05
) -> Dict[str, float | str | None]:
    """
    Concatenate scores/labels over steps [0..horizon-1] and compute metrics.
    If is_prob=True, 'scores' are probabilities and we report brier/logloss/pred_rate.
    """


    S = np.concatenate([np.asarray(score_list[t]) for t in range(min(horizon, len(score_list)))], axis=0)
    Y = np.concatenate([np.asarray(label_list[t]) for t in range(min(horizon, len(label_list)))], axis=0).astype(np.uint8)
    E = len(S)

    # --- AUC/PR with degeneracy guard ---
    pos = int(Y.sum()); neg = E - pos
    if pos == 0 or neg == 0:
        roc_auc = 'NA (no pos/neg)'
        pr_auc  = 'NA (no pos/neg)'
    else:
        roc_auc = float(roc_auc_score(Y, S))
        pr_auc  = float(average_precision_score(Y, S))

    # --- Fixed top-k across ALL methods ---
    if k is None:
        k = max(1, int(round(k_frac * E)))
    idx = np.argsort(S)[::-1][:k]
    pred = np.zeros(E, dtype=np.uint8); pred[idx] = 1

    TP = int((pred & Y).sum())
    FP = int((pred & (1 - Y)).sum())
    FN = int(((1 - pred) & Y).sum())

    prec = TP / max(TP + FP, 1)
    rec  = TP / max(TP + FN, 1)
    f1   = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    pred_rate = k / E
    true_rate = float(Y.mean())

    # --- Prob losses only for prob rows ---
    if is_prob:
        P = np.clip(S, 1e-7, 1 - 1e-7)
        try:
            brier = float(brier_score_loss(Y, P))
        except Exception:
            brier = None
        try:
            logloss = float(log_loss(Y, P, labels=[0, 1]))
        except Exception:
            logloss = None
    else:
        brier = None
        logloss = None

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": brier,
        "logloss": logloss,
        "p_at_k": prec,
        "r_at_k": rec,
        "f1_at_k": f1,
        "topk_rate": pred_rate,
        "k": k,
        "true_pos_rate": true_rate,
    }


# ==========================
# Pretty-print helpers
# ==========================
def print_epoch_history(seed: int, hist: Dict[str, List[float]], fallback_metrics: Dict[str, float]) -> None:
    """Print one CSV line per epoch, or a summary if no history present."""
    print("\n=== training history ===")
    if hist and hist.get("loss"):
        num_epochs_logged = len(hist["loss"])
        header = ["seed", "epoch", "lr", "loss", "nll", "action", "m2", "c2", "b0"]
        print(",".join(header))
        for ep in range(num_epochs_logged):
            row = [
                seed,
                ep + 1,
                hist.get("lr", [None] * num_epochs_logged)[ep],
                hist.get("loss", [None] * num_epochs_logged)[ep],
                hist.get("nll", [None] * num_epochs_logged)[ep],
                hist.get("action", [None] * num_epochs_logged)[ep],
                hist.get("m2", [None] * num_epochs_logged)[ep],
                hist.get("c2", [None] * num_epochs_logged)[ep],
                hist.get("b0", [None] * num_epochs_logged)[ep],
            ]
            print(",".join("" if v is None else str(v) for v in row))
    else:
        print("no per-epoch history found; summary:")
        for k in ["final_loss", "final_nll", "final_action", "final_m2", "final_c2", "final_b0", "epochs"]:
            print(f"  {k:>12}: {fallback_metrics.get(k)}")


def print_eval_rows(rows: List[List[object]], header: List[str], title: str) -> None:
    """Print a CSV-style table to the console."""
    print(f"\n{title}")
    print(",".join(header))
    for r in rows:
        print(",".join("" if x is None else str(x) for x in r))


@torch.no_grad()
def calibrate_gamma_for_free_mode(
    theta, cfg, y_train_csr, num_nodes, dt_if, ema_alpha, mode: str = "zero", T_cal: int | None = None
) -> float:
    """Fit scalar γ to map IF free/self node series → EMA space on a short TRAIN tail."""
    dev = cfg.device if isinstance(cfg.device, torch.device) else torch.device(cfg.device)

    # 1) teacher-forced pass to get (h_last, mem_last)
    y_tr_uv = csr_bins_to_uv_tensors(y_train_csr, device=dev)
    H_tr, _, _, _, _, mem_tr = forward_rollout_if(
        y_tr_uv, theta, cfg, free_start=None, free_mode="driven", enable_progress=False
    )
    h_last = H_tr[-1].detach(); mem_last = mem_tr

    # 2) short free/self run on TRAIN tail
    if T_cal is None:
        T_cal = min(50, len(y_train_csr))
    empty_uv = [torch.empty(2, 0, dtype=torch.long, device=dev) for _ in range(T_cal)]
    free_mode = "self" if mode == "self" else "zero"
    _, _, _, rs_cal, lams_cal, _ = forward_rollout_if(
        empty_uv, theta, cfg, free_start=0, free_mode=free_mode,
        h0=h_last, mem_init_sparse=mem_last, enable_progress=False
    )

    # 3) map λ→nodes (EMA space) and fit γ
    x_cal = lambdas_to_node_series_if(lams_cal, rs_cal, num_nodes, dt_if, ema_alpha=ema_alpha)
    x_true_tail = node_series_from_csr_bins(y_train_csr[-T_cal:], num_nodes, mode="recv_ema", ema_alpha=ema_alpha)
    return fit_scalar_gamma(x_cal, x_true_tail)






def _align_len(*seqs):
    """Truncate all sequences to common min length."""
    L = min(len(s) for s in seqs)
    return [s[:L] for s in seqs]

def _align_pred_truth(Xp: np.ndarray, Xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Align 2D [T, N] predictor and truth along time."""
    T = min(len(Xp), len(Xy))
    return Xp[:T], Xy[:T]

def _align_with_slice(x_pred_cal: np.ndarray, sl: slice, x_truth_full: np.ndarray):
    """Slice truth by 'sl' (from apply_affine_calibration) then time-align."""
    x_truth_sl = x_truth_full[sl]
    T = min(x_pred_cal.shape[0], x_truth_sl.shape[0])
    return x_pred_cal[:T], x_truth_sl[:T]

def _print_header(title: str, headers: List[str]):
    print(title)
    print(",".join(headers))

def _print_node_row(seed, split, method, dt, gamma, h, m):
    print(",".join([
        str(seed), split, method,
        f"{dt}", "" if gamma is None else f"{gamma}",
        str(h),
        f"{m['nmse']}", f"{m['node_corr_median']}", f"{m['node_corr_mean']}"
    ]))

def _print_edge_row(seed, split, method, h, m):
    print(",".join([
        str(seed), split, method, str(h),
        f"{m['roc_auc']}", f"{m['pr_auc']}", f"{m['brier']}", f"{m['logloss']}",
        f"{m['p_at_k']}", f"{m['r_at_k']}", f"{m['f1_at_k']}",
        f"{m['topk_rate']}", f"{m['k']}", f"{m['true_pos_rate']}"
    ]))

# --------------------------------------------
# Linear surrogate calibration & rollouts
# --------------------------------------------
def _build_and_calibrate_linear(theta, cfg, y_train, num_nodes, device, dt_grid,
                                u_train_in, x_train_true):
    L_eff = build_L_eff_from_train(theta, y_train, num_nodes, cfg).to(dtype=torch.float32, device=device)

    def _calibrate_dt_gamma(L: torch.Tensor, U_tr_in: np.ndarray, X_tr_true: np.ndarray,
                            grid: Tuple[float, ...]) -> Tuple[float, float]:
        U_tr_in_t = torch.tensor(U_tr_in, dtype=torch.float32, device=device)
        ridge = 1e-12
        best: Optional[Tuple[float, float]] = None
        best_key: Optional[Tuple[float, float]] = None
        for dt in grid:
            X_unit_tr = rollout_linear_driven(L, U_tr_in_t, x0=None, dt=float(dt))
            Xu = X_unit_tr
            Xt = torch.tensor(X_tr_true, dtype=torch.float32, device=device)
            num = (Xu * Xt).sum().item()
            den = (Xu * Xu).sum().item() + ridge
            gamma = float(np.clip(num / den, 0.0, 5.0))
            m = eval_rollout(X_tr_true, (gamma * Xu.detach().cpu().numpy()), var_ref=None)
            nmse = m.get("nmse")
            node_corr = m.get("node_corr_median")
            if nmse is None or node_corr is None:
                continue
            key = (float(node_corr), -float(nmse))
            if best_key is None or key > best_key:
                best, best_key = (float(dt), float(gamma)), key
        assert best is not None
        return best[0], best[1]

    dt_star, gamma_star = _calibrate_dt_gamma(L_eff, u_train_in, x_train_true, dt_grid)

    # linear node baselines
    u_train_in_t = torch.tensor(u_train_in, dtype=torch.float32, device=device)
    x_unit_train_full = rollout_linear_driven(L_eff, u_train_in_t, x0=None, dt=dt_star).detach().cpu().numpy()
    calib_driven_affine = fit_affine_calibration_per_node(
        x_unit_train_full, x_train_true, ridge=1e-6, max_lag=0, use_lag=False
    )

    return L_eff, dt_star, gamma_star, calib_driven_affine

def _rollout_linear_nodes(L_eff, dt_star, gamma_star, calib_driven_affine,
                          u_hold_in, x_train_true_last, device):
    u_hold_in_t = torch.tensor(u_hold_in, dtype=torch.float32, device=device)
    x_unit_hold_full = rollout_linear_driven(L_eff, u_hold_in_t, x0=None, dt=dt_star)
    x_driven_gamma_full = (gamma_star * x_unit_hold_full).detach().cpu().numpy()
    x_driven_affine_full, _ = apply_affine_calibration(
        x_unit_hold_full.detach().cpu().numpy(), calib_driven_affine
    )

    x0_state = torch.tensor(x_train_true_last, dtype=torch.float32, device=device)
    zeros_hold_in = torch.zeros_like(u_hold_in_t)
    x_free_hold_full = rollout_linear_driven(L_eff, zeros_hold_in, x0=x0_state, dt=dt_star).detach().cpu().numpy()

    return x_driven_gamma_full, x_driven_affine_full, x_free_hold_full





# --------------------------------------------
# IF hazards calibration (τ + isotonic)
# --------------------------------------------
def _calibrate_if_hazards(theta, cfg, y_train, y_holdout):
    _, risk_sets_train, lambda_list_train, _ = rollout_if_nodes_driven(
        theta, cfg, y_train, reduce_mode="sym"
    )
    tau_hats = fit_hazard_temperature_per_bin(lambda_list_train, risk_sets_train, y_train, dt_hazard=1.0)
    cfg.hazard_tau = tau_hats

    train_probs_raw = probs_from_lambda_list(lambda_list_train, dt=1.0, tau=tau_hats)
    train_labels = make_labels_for_risk_sets(y_train, risk_sets_train)

    iso = None
    _apply_iso = lambda x: x
    try:
        from sklearn.isotonic import IsotonicRegression
        if np.unique(np.concatenate(train_labels)).size >= 2:
            iso = IsotonicRegression(out_of_bounds="clip").fit(
                np.concatenate([p.ravel() for p in train_probs_raw]),
                np.concatenate([y.ravel() for y in train_labels]).astype(float),
            )
            _apply_iso = lambda ps_list: [iso.transform(p.ravel()).reshape(p.shape) for p in ps_list]
    except Exception:
        iso = None

    _, risk_sets_hold, lambda_list_hold, _ = rollout_if_nodes_driven(
        theta, cfg, y_holdout, reduce_mode="sym"
    )
    tau_hats_hold = align_tau_to_length(tau_hats, len(lambda_list_hold))
    hold_probs_raw = probs_from_lambda_list(lambda_list_hold, dt=1.0, tau=tau_hats_hold)
    hold_probs_cal = _apply_iso(hold_probs_raw)
    risk_labels = make_labels_for_risk_sets(y_holdout, risk_sets_hold)
    return tau_hats, hold_probs_cal, risk_sets_hold, risk_labels, risk_sets_train


# --------------------------------------------
# Warm start latent/memory from TRAIN
# --------------------------------------------
def _warm_start_from_train(theta, cfg, y_train, device):
    y_train_uv = csr_bins_to_uv_tensors(y_train, device=device)
    H_tr, _, _, _, _, M_sparse_tr = forward_rollout_if(
        y_train_uv, theta, cfg, free_start=None, free_mode="driven", enable_progress=False
    )
    h_last = H_tr[-1].detach()
    return h_last, M_sparse_tr

# --------------------------------------------
# IF-free tail calibration (robust, lag=0)
# --------------------------------------------
def _safe_fit_affine_per_node(X_pred, X_true, *, ridge=1e-3, beta_clip=5.0):
    calib = fit_affine_calibration_per_node(
        X_pred, X_true, ridge=ridge, max_lag=0, use_lag=False
    )
    alpha = calib["alpha"].astype(np.float32).copy()
    beta  = calib["beta"].astype(np.float32).copy()
    beta  = np.clip(beta, 0, beta_clip)
    lag   = np.zeros_like(alpha, dtype=int)
    return {"alpha": alpha, "beta": beta, "lag": lag}

@torch.no_grad()
def _calibrate_if_free_tail(theta, cfg, y_train, x_train_true, num_nodes, device,
                            dt_grid: Tuple[float, ...], gamma_grid=(0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
                            tail_frac=0.15, beta_clip=5.0):
    # common length to avoid off-by-one
    T = min(len(y_train), len(x_train_true))
    if T < 10:
        # fallback: identity affine & gamma=1, keep cfg.dt
        N = num_nodes
        return float(cfg.dt), 1.0, {
            "alpha": np.zeros(N, dtype=np.float32),
            "beta":  np.ones(N,  dtype=np.float32),
            "lag":   np.zeros(N, dtype=int),
        }

    T_tail = max(5, min(T - 1, int(round(tail_frac * T))))
    t0 = T - T_tail

    # warm state at split
    y_train_uv = csr_bins_to_uv_tensors(y_train[:T], device=device)
    H_pref, _, _, _, _, M_pref = forward_rollout_if(
        y_train_uv[:t0], theta, cfg, free_start=None, free_mode="driven", enable_progress=False
    )
    h0, mem0 = H_pref[-1].detach(), M_pref
    X_tail_true = x_train_true[:T][t0:]  # [T_tail, N]

    best, best_key = None, None
    dt_saved = float(cfg.dt)
    try:
        for dt in dt_grid:
            cfg.dt = float(dt)
            X_free_tail, _, _, _ = rollout_if_nodes_free(
                theta, cfg, T=T_tail, h0=h0, mem0_sparse=mem0, reduce_mode="sym"
            )
            # defensive align
            L = min(len(X_free_tail), len(X_tail_true))
            if L <= 0:
                continue
            Xp, Y = X_free_tail[:L], X_tail_true[:L]

            # skip degenerate predictors
            if not np.isfinite(np.median(Xp.std(axis=0))) or (np.median(Xp.std(axis=0)) < 1e-8):
                continue

            for g in gamma_grid:
                Xg = float(g) * Xp
                calib = _safe_fit_affine_per_node(Xg, Y, ridge=1e-3, beta_clip=beta_clip)
                Xc, _ = apply_affine_calibration(Xg, calib)
                L2 = min(len(Xc), len(Y))
                if L2 <= 0 or not np.isfinite(Xc).all():
                    continue
                m = eval_rollout(Y[:L2], Xc[:L2], var_ref=None)
                nmse = m.get("nmse")
                node_corr = m.get("node_corr_median", 0.0)
                if nmse is None or node_corr is None:
                    continue
                # small penalty for large betas
                beta_med = float(np.median(np.abs(calib["beta"])))
                key = (-float(nmse) - 1e-3 * beta_med, float(node_corr))
                if best_key is None or key > best_key:
                    best, best_key = (float(dt), float(g), calib), key
    finally:
        cfg.dt = dt_saved

    if best is None:
        N = num_nodes
        return float(dt_saved), 1.0, {
            "alpha": np.zeros(N, dtype=np.float32),
            "beta":  np.ones(N,  dtype=np.float32),
            "lag":   np.zeros(N, dtype=int),
        }
    return best[0], best[1], best[2]

# --------------------------------------------
# Node metrics block (streamed printing)
# --------------------------------------------
def _eval_node_metrics(seed, dt_star, gamma_star, alpha_hat, num_nodes, var_ref,
                       x_true_h, x_driven_gamma_h, x_driven_affine_h, x_free_h,
                       x_free_if_h, x_true_if_free_h, x_self_if_h, x_true_if_self_h,
                       dt_free_star, gamma_free_star) -> List[Dict[str, Any]]:
    NODE_HEADERS = ["seed","split","method","dt","gamma","horizon",
                    "nmse","node_corr_median","node_corr_mean"]
    _print_header("=== NODE metrics (holdout, per horizon) ===", NODE_HEADERS)

    var_ref_hold = float(np.var(x_true_h)) + 1e-8

    H_global = min(
        len(x_true_h), len(x_driven_gamma_h), len(x_driven_affine_h), len(x_free_h),
        len(x_free_if_h), len(x_true_if_free_h),
        len(x_self_if_h), len(x_true_if_self_h)
    )
    rows: List[Dict[str, Any]] = []
    for h in range(1, H_global+1):
        # driven
        m = eval_rollout(x_true_h[:h], x_driven_gamma_h[:h], var_ref=var_ref_hold)
        _print_node_row(seed, "holdout", "driven", dt_star, gamma_star, h, m)
        rows.append({
            "seed": seed, "split": "holdout", "method": "driven", "dt": dt_star, "gamma": gamma_star,
            "horizon": h, "nmse": m["nmse"], "node_corr_median": m["node_corr_median"],
            "node_corr_mean": m["node_corr_mean"],
        })

        m = eval_rollout(x_true_h[:h], x_driven_affine_h[:h], var_ref=var_ref_hold)
        _print_node_row(seed, "holdout", "driven-affine", dt_star, None, h, m)
        rows.append({
            "seed": seed, "split": "holdout", "method": "driven-affine", "dt": dt_star, "gamma": None,
            "horizon": h, "nmse": m["nmse"], "node_corr_median": m["node_corr_median"],
            "node_corr_mean": m["node_corr_mean"],
        })

        # IF-free (calibrated dt/gamma/affine)
        m = eval_rollout(x_true_if_free_h[:h], x_free_if_h[:h], var_ref=var_ref_hold)
        _print_node_row(seed, "holdout", "if-free", dt_free_star, gamma_free_star, h, m)
        rows.append({
            "seed": seed, "split": "holdout", "method": "if-free", "dt": dt_free_star,
            "gamma": gamma_free_star, "horizon": h, "nmse": m["nmse"],
            "node_corr_median": m["node_corr_median"], "node_corr_mean": m["node_corr_mean"],
        })

        # IF-self (report dt_star for simplicity; gamma N/A)
        m = eval_rollout(x_true_if_self_h[:h], x_self_if_h[:h], var_ref=var_ref_hold)
        _print_node_row(seed, "holdout", "free-self", dt_star, None, h, m)
        rows.append({
            "seed": seed, "split": "holdout", "method": "free-self", "dt": dt_star, "gamma": None,
            "horizon": h, "nmse": m["nmse"], "node_corr_median": m["node_corr_median"],
            "node_corr_mean": m["node_corr_mean"],
        })

        # linear free
        m = eval_rollout(x_true_h[:h], x_free_h[:h], var_ref=var_ref_hold)
        _print_node_row(seed, "holdout", "free", dt_star, None, h, m)
        rows.append({
            "seed": seed, "split": "holdout", "method": "free", "dt": dt_star, "gamma": None,
            "horizon": h, "nmse": m["nmse"], "node_corr_median": m["node_corr_median"],
            "node_corr_mean": m["node_corr_mean"],
        })

        # AR(1)
        X_tmp = rollout_ar1(x_true_h[0].copy(), alpha_hat, h, num_nodes)
        m = eval_rollout(x_true_h[:h], X_tmp, var_ref=var_ref_hold)
        _print_node_row(seed, "holdout", "temporal", dt_star,
                        float(-np.log(max(alpha_hat, 1e-12)) / max(dt_star, 1e-12)),
                        h, m)
        rows.append({
            "seed": seed, "split": "holdout", "method": "temporal", "dt": dt_star,
            "gamma": float(-np.log(max(alpha_hat, 1e-12)) / max(dt_star, 1e-12)),
            "horizon": h, "nmse": m["nmse"], "node_corr_median": m["node_corr_median"],
            "node_corr_mean": m["node_corr_mean"],
        })

    return rows

# --------------------------------------------
# Edge metrics block (streamed printing)
# --------------------------------------------
def _scalar_edge_scores_from_nodes(X_series: np.ndarray, edges_undirected: np.ndarray) -> List[np.ndarray]:
    E = np.asarray(edges_undirected, dtype=np.int64)
    out = []
    for t in range(X_series.shape[0]):
        x = X_series[t]
        s = x[E[:,0]] + x[E[:,1]]
        s = (s - s.mean()) / (s.std() + 1e-9)
        p = 1.0 / (1.0 + np.exp(-s))
        out.append(p.astype(np.float64))   # shape (E,)
    return out

def _undirected_edge_labels_from_bins(y_bins: List[csr_matrix], edges_undirected: np.ndarray, limit: int) -> List[np.ndarray]:
    E = np.asarray(edges_undirected, dtype=np.int64)
    labels: List[np.ndarray] = []
    for csr in y_bins[:limit]:
        pos = csr_to_set(csr)
        y = np.fromiter(((1 if ((u, v) in pos or (v, u) in pos) else 0) for (u, v) in E),
                        dtype=np.uint8, count=E.shape[0])
        labels.append(y)
    return labels

def _eval_edge_metrics(seed, grid_E, y_holdout,
                       hold_probs_cal, risk_sets_hold, risk_labels,
                       x_driven_gamma_h, x_driven_affine_h, x_free_h,
                       static_scores: Optional[Dict[str, List[np.ndarray]]] = None) -> List[Dict[str, Any]]:
    EDGE_HEADERS = ["seed","split","method","horizon","roc_auc","pr_auc","brier","logloss",
                    "P@k","R@k","F1@k","pred_rate","true_rate"]
    _print_header("=== EDGE metrics (holdout, per horizon) ===", EDGE_HEADERS)

    # hazards (directed) lists are: hold_probs_cal & risk_sets_hold
    Te_haz = min(len(hold_probs_cal), len(risk_labels))

    # scalar (undirected) lists built from node rollouts
    scalar_scores_driven = _scalar_edge_scores_from_nodes(x_driven_gamma_h, grid_E)
    scalar_scores_driven_aff = _scalar_edge_scores_from_nodes(x_driven_affine_h, grid_E)
    scalar_scores_free = _scalar_edge_scores_from_nodes(x_free_h, grid_E)
    Te_scalar = min(len(scalar_scores_driven), len(y_holdout))

    scalar_labels = _undirected_edge_labels_from_bins(y_holdout, grid_E, Te_scalar)

    rows: List[Dict[str, Any]] = []

    def _agg_row(method, scores_list, labels_list, h, is_prob):
        H = min(h, len(scores_list), len(labels_list))
        m = aggregate_edge_metrics_over_horizon(scores_list, labels_list, H, is_prob=is_prob)
        _print_edge_row(seed, "holdout", method, h, m)
        rows.append({
            "seed": seed, "split": "holdout", "method": method, "horizon": h,
            "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"], "brier": m["brier"],
            "logloss": m["logloss"], "p_at_k": m["p_at_k"], "r_at_k": m["r_at_k"],
            "f1_at_k": m["f1_at_k"], "topk_rate": m["topk_rate"], "k": m["k"],
            "true_pos_rate": m["true_pos_rate"],
        })

    H_edge_global = max(Te_haz, Te_scalar)
    for h in range(1, H_edge_global+1):
        _agg_row("if-hazards-cal", hold_probs_cal, risk_labels, h, is_prob=True)
        if static_scores is not None:
            _agg_row("static-cn", static_scores["cn"], risk_labels, h, is_prob=False)
            _agg_row("static-aa", static_scores["aa"], risk_labels, h, is_prob=False)
            _agg_row("static-katz", static_scores["katz"], risk_labels, h, is_prob=False)
            _agg_row("static-lr", static_scores["static-lr"], risk_labels, h, is_prob=True)  # prob output

        _agg_row("scalar-driven", scalar_scores_driven, scalar_labels, h, is_prob=False)
        _agg_row("scalar-driven-affine", scalar_scores_driven_aff, scalar_labels, h, is_prob=False)
        _agg_row("scalar-free", scalar_scores_free, scalar_labels, h, is_prob=False)

    return rows

# --------------------------------------------
# Visualization wiring (with bidirectional fix)
# --------------------------------------------
def _viz_variants(num_nodes, hgt, wdt, y_holdout,
                  x_true_h, x_driven_gamma_h, x_driven_affine_h, x_free_h,
                  hold_probs_cal, risk_sets_hold, grid_E,
                  frame_kwargs):
    hold_probs_cal_aln, risk_sets_hold_aln = _align_len(hold_probs_cal, risk_sets_hold)
    event_bins_pred_soft = bins_from_probs(hold_probs_cal_aln, risk_sets_hold_aln, N=num_nodes)

    def _bidir_pairs(E: np.ndarray) -> np.ndarray:
        return np.vstack([E, E[:, ::-1]])  # (2E,2)

    Lviz = min(len(x_driven_gamma_h), len(x_driven_affine_h), len(x_free_h), len(event_bins_pred_soft))
    X_truth_nodes_viz = x_true_h[:Lviz]
    X_driven_nodes_gamma_viz = x_driven_gamma_h[:Lviz]
    X_driven_nodes_aff_viz   = x_driven_affine_h[:Lviz]
    X_free_nodes_viz         = x_free_h[:Lviz]
    event_bins_pred_viz      = event_bins_pred_soft[:Lviz]

    # Build scalar edge bins on bidirectional pairs -> duplicate scores to 2E
    bidir = _bidir_pairs(grid_E)       # (2E,2)
    scalar_pairs_list = [bidir] * Lviz # risk sets

    def _dup_scores_to_bidir(scores_undirected_list: List[np.ndarray]) -> List[np.ndarray]:
        out = []
        for s in scores_undirected_list[:Lviz]:
            out.append(np.concatenate([s, s], axis=0))   # (2E,)
        return out

    scalar_scores_driven = _scalar_edge_scores_from_nodes(x_driven_gamma_h, grid_E)[:Lviz]
    scalar_scores_driven_aff = _scalar_edge_scores_from_nodes(x_driven_affine_h, grid_E)[:Lviz]
    scalar_scores_free = _scalar_edge_scores_from_nodes(x_free_h, grid_E)[:Lviz]

    event_bins_scalar_gamma  = bins_from_probs(_dup_scores_to_bidir(scalar_scores_driven),     scalar_pairs_list, N=num_nodes)
    event_bins_scalar_affine = bins_from_probs(_dup_scores_to_bidir(scalar_scores_driven_aff), scalar_pairs_list, N=num_nodes)
    event_bins_scalar_free   = bins_from_probs(_dup_scores_to_bidir(scalar_scores_free),       scalar_pairs_list, N=num_nodes)

    variants = {
        "driven(y+hazards)":       {"X": X_driven_nodes_gamma_viz, "edges": event_bins_pred_viz},
        "driven(y scalar)":        {"X": X_driven_nodes_gamma_viz, "edges": event_bins_scalar_gamma},
        "driven(affine scalar)":   {"X": X_driven_nodes_aff_viz,   "edges": event_bins_scalar_affine},
        "free(scalar)":            {"X": X_free_nodes_viz,         "edges": event_bins_scalar_free},
    }

    plot_rollout(
        variants=variants,
        h=hgt, w=wdt, t=None,
        outdir="exports_nodes_edges",
        reference_nodes=X_truth_nodes_viz,
        reference_edges=y_holdout[:Lviz],
        pred_edge_bins=event_bins_pred_viz,
        cm_nodes="seismic",
        cm_edges_activity="Greys",
        cm_edges_residual="seismic",
        decay_k=10, decay_gamma=0.6, tau_decay=5.0,
        frame_kwargs=frame_kwargs,
        z_exaggeration=0.8,
        do_nodes_panels=True,
        do_edges_activity=True,
        do_edges_residual=True,
        do_edges_confusion=True,
        do_combined_activity=True,
        do_nodes_residual=True,
    )

# ==============
# Static Graph Baselines
# ==================================

@dataclass
class StaticGraphSpec:
    # how to turn TRAIN bins into a single static adjacency
    mode: str = "binary"        # "binary" | "count"
    undirected: bool = True     # treat graph as undirected for heuristics

def _static_adj_from_train(y_train: List[csr_matrix], N: int, spec: StaticGraphSpec) -> csr_matrix:
    """
    Build a single static adjacency from TRAIN bins.
    - binary: edge exists if it appeared at least once
    - count : edge weight = total count over TRAIN
    """
    # accumulate as float64 then cast down
    A = csr_matrix((N, N), dtype=np.float64)
    for Yt in y_train:
        if Yt is None or Yt.nnz == 0:
            continue
        A = A + Yt.astype(np.float64)

    if spec.undirected:
        A = A + A.T

    if spec.mode == "binary":
        A.data[:] = 1.0
        A.eliminate_zeros()
    elif spec.mode == "count":
        A.eliminate_zeros()
    else:
        raise ValueError("StaticGraphSpec.mode must be {'binary','count'}")

    return A.tocsr()

def _prep_neighbors(A: csr_matrix):
    """Cache neighbor lists for CN/AA."""
    nbrs = []
    assert A.shape is not None, "Adjacency matrix must have known shape"
    for i in range(A.shape[0]):
        nbrs.append(A.indices[A.indptr[i]:A.indptr[i+1]])
    deg = np.asarray(A.sum(axis=1)).ravel()
    return nbrs, deg

def _score_cn_aa_on_pairs(A: csr_matrix, pairs: List[Pair]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (CN, AA) scores for a list of (u,v) pairs on undirected adjacency A.
    CN(u,v) = |Γ(u) ∩ Γ(v)|
    AA(u,v) = sum_{w in Γ(u)∩Γ(v)} 1/log(deg(w)+1)
    """
    nbrs, deg = _prep_neighbors(A)
    invlog = 1.0 / np.log(deg + 1.0 + 1e-12)

    cn = np.zeros(len(pairs), dtype=np.float64)
    aa = np.zeros(len(pairs), dtype=np.float64)

    # intersection via two-pointer merge on sorted neighbor lists
    for k, (u, v) in enumerate(pairs):
        Nu = nbrs[u]; Nv = nbrs[v]
        i = j = 0
        c = 0
        s = 0.0
        while i < len(Nu) and j < len(Nv):
            a = Nu[i]; b = Nv[j]
            if a == b:
                c += 1
                s += invlog[a]
                i += 1; j += 1
            elif a < b:
                i += 1
            else:
                j += 1
        cn[k] = c
        aa[k] = s
    return cn, aa

def _score_katz_on_pairs(A: csr_matrix, pairs: List[Pair], beta: float = 0.01, max_iter: int = 3) -> np.ndarray:
    """
    Cheap Katz approximation: sum_{l=1..max_iter} beta^l * (#paths length l).
    We do this by repeated sparse multiplies on a vector basis is too expensive per pair,
    so we approximate Katz by using weighted CN at short lengths:
      l=1 => A[u,v]
      l=2 => (A^2)[u,v]
      l=3 => (A^3)[u,v]
    """
    A = A.astype(np.float64)
    # precompute A^2, A^3 (ok for small/medium N; if huge, we’ll replace later)
    A2 = (A @ A)
    A3 = (A2 @ A) if max_iter >= 3 else None

    out = np.zeros(len(pairs), dtype=np.float64)
    for k, (u, v) in enumerate(pairs):
        s = 0.0
        s += (beta ** 1) * (A[u, v])
        if max_iter >= 2:
            s += (beta ** 2) * (A2[u, v])
        if max_iter >= 3 and A3 is not None:
            s += (beta ** 3) * (A3[u, v])
        out[k] = float(s)
    return out

def _build_static_features(A_static: csr_matrix, pairs: List[Pair]) -> np.ndarray:
    """
    Feature vector per (u,v) using common static heuristics.
    """
    cn, aa = _score_cn_aa_on_pairs(A_static, pairs)
    katz = _score_katz_on_pairs(A_static, pairs, beta=0.01, max_iter=3)
    # log-scale helps LR
    X = np.stack([
        np.log1p(cn),
        np.log1p(aa),
        np.log1p(katz),
    ], axis=1).astype(np.float32)
    return X

def static_baselines_predict_on_risk_sets(
    *,
    y_train: List[csr_matrix],
    risk_sets_train: List[List[Pair]],
    y_holdout: List[csr_matrix],
    risk_sets_hold: List[List[Pair]],
    N: int,
    spec: StaticGraphSpec,
) -> Dict[str, List[np.ndarray]]:
    """
    Returns dict of method -> score_list aligned with risk_sets_hold (same lengths per t).
    Provides:
      - cn
      - aa
      - katz
      - static-lr  (trained on TRAIN risk sets)
    """
    A_static = _static_adj_from_train(y_train, N, spec)

    # --- heuristic scores on HOLDOUT risk sets ---
    cn_scores = []
    aa_scores = []
    katz_scores = []

    for pairs in risk_sets_hold:
        if not pairs:
            cn_scores.append(np.zeros((0,), dtype=np.float64))
            aa_scores.append(np.zeros((0,), dtype=np.float64))
            katz_scores.append(np.zeros((0,), dtype=np.float64))
            continue
        cn, aa = _score_cn_aa_on_pairs(A_static, pairs)
        kz = _score_katz_on_pairs(A_static, pairs, beta=0.01, max_iter=3)
        cn_scores.append(cn)
        aa_scores.append(aa)
        katz_scores.append(kz)

    out = {"cn": cn_scores, "aa": aa_scores, "katz": katz_scores}

    # --- static graph + logistic regression baseline ---
    # Train on TRAIN risk sets (same candidate construction as IF hazards)
    Xtr = []
    Ytr = []
    for t, pairs in enumerate(risk_sets_train):
        if not pairs:
            continue
        X = _build_static_features(A_static, pairs)
        y = make_labels_for_risk_sets(y_train, [pairs])[0].astype(np.int32)
        if y.size == 0:
            continue
        Xtr.append(X)
        Ytr.append(y)

    if len(Xtr) > 0 and np.unique(np.concatenate(Ytr)).size >= 2:
        Xtr = np.concatenate(Xtr, axis=0)
        Ytr = np.concatenate(Ytr, axis=0)

        lr = LogisticRegression(
            solver="lbfgs",
            max_iter=200,
            class_weight="balanced",
        ).fit(Xtr, Ytr)

        lr_scores = []
        for pairs in risk_sets_hold:
            if not pairs:
                lr_scores.append(np.zeros((0,), dtype=np.float64))
                continue
            Xh = _build_static_features(A_static, pairs)
            p = lr.predict_proba(Xh)[:, 1].astype(np.float64)
            lr_scores.append(p)
        out["static-lr"] = lr_scores
    else:
        # not enough signal to fit
        out["static-lr"] = [np.zeros((len(p),), dtype=np.float64) for p in risk_sets_hold]

    return out





# ============================================
# Main evaluation
# ============================================
def rollout_eval(
    theta: IFParameters,
    cfg: IFConfig,
    y_train: List[csr_matrix],
    y_holdout: List[csr_matrix],
    num_nodes: int,
    ema_alpha: float,
    dt_grid: Tuple[float, ...] = (0.5, 1.0, 2.0, 3.0),
    *,
    seed: int = 123,
    do_viz: bool = False,
    frame_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    device = cfg.device if isinstance(cfg.device, torch.device) else torch.device(cfg.device)
    hgt, wdt = best_grid_factors(num_nodes)

    # --- prep series ---
    u_train_in, x_train_true, u_hold_in, x_hold_true, _, var_ref = prepare_series(
        y_train, y_holdout, num_nodes, ema_alpha
    )
    alpha_hat, _ = ar1_baseline_from_train(x_train_true, num_nodes)

    # --- linear calibration & linear baselines ---
    L_eff, dt_star, gamma_star, calib_driven_affine = _build_and_calibrate_linear(
        theta, cfg, y_train, num_nodes, device, dt_grid, u_train_in, x_train_true
    )
    x_driven_gamma_full, x_driven_affine_full, x_free_hold_full = _rollout_linear_nodes(
        L_eff, dt_star, gamma_star, calib_driven_affine,
        u_hold_in, x_train_true[-1], device
    )

    # unify horizon for linear baselines vs truth
    Tn_base = min(len(x_hold_true), len(x_driven_gamma_full), len(x_driven_affine_full), len(x_free_hold_full))
    x_true_h             = x_hold_true[:Tn_base]
    x_driven_gamma_h     = x_driven_gamma_full[:Tn_base]
    x_driven_affine_h    = x_driven_affine_full[:Tn_base]
    x_free_h             = x_free_hold_full[:Tn_base]

    # --- IF hazards calibration & holdout probs ---
    tau_hats, hold_probs_cal, risk_sets_hold, risk_labels, risk_sets_train = _calibrate_if_hazards(
        theta, cfg, y_train, y_holdout
    )

    static_scores = static_baselines_predict_on_risk_sets(
        y_train=y_train,
        risk_sets_train=risk_sets_train,
        y_holdout=y_holdout,
        risk_sets_hold=risk_sets_hold,
        N=num_nodes,
        spec=StaticGraphSpec(mode="binary", undirected=True),
    )

    # --- warm start latent/memory from TRAIN ---
    h_last, mem0_sparse = _warm_start_from_train(theta, cfg, y_train, device)

    # --- IF-free tail calibration (robust) on TRAIN; apply on HOLDOUT ---
    dt_free_star, gamma_free_star, calib_free_star = _calibrate_if_free_tail(
        theta, cfg, y_train, x_train_true, num_nodes, device,
        dt_grid=dt_grid, gamma_grid=(0.25,0.5,0.75,1.0,1.5,2.0), tail_frac=0.15, beta_clip=5.0
    )
    print(f"[calib] IF-free tail grid → dt*={dt_free_star:.6g}, gamma*={gamma_free_star:.6g}")

    # HOLDOUT IF-free with calibrated dt/gamma/affine
    _dt_saved = float(cfg.dt)
    cfg.dt = float(dt_free_star)
    X_free_if_hold, _, _, _ = rollout_if_nodes_free(
        theta, cfg, T=len(y_holdout), h0=h_last, mem0_sparse=mem0_sparse, reduce_mode="sym"
    )
    cfg.dt = _dt_saved
    X_free_if_hold_scaled = float(gamma_free_star) * X_free_if_hold
    X_free_if_cal, sl_free = apply_affine_calibration(X_free_if_hold_scaled, calib_free_star)
    x_free_if_h, x_true_if_free_h = _align_with_slice(X_free_if_cal, sl_free, x_hold_true)

    # HOLDOUT IF-self (uses tau_hat sampling)
    tau_hats_hold = align_tau_to_length(tau_hats, len(y_holdout))
    X_self_if, _, _, _ = rollout_if_nodes_self(
        theta, cfg, T=len(y_holdout), h0=h_last, mem0_sparse=mem0_sparse,
        hazard_tau=tau_hats_hold, reduce_mode="sym"
    )
    x_self_if_h, x_true_if_self_h = _align_pred_truth(X_self_if, x_hold_true)

    def _series_diag(name, Xp, Y):
        # T x N → flatten over time for quick signals
        xs, ys = Xp.reshape(-1), Y.reshape(-1)
        corr = np.corrcoef(xs, ys)[0, 1]
        sXp, sY = Xp.std(), Y.std()
        print(f"[diag:{name}] corr={corr:.4f}  std_pred/std_true={sXp / sY:.3f}  std_pred={sXp:.3g}  std_true={sY:.3g}")

    _series_diag("if-free", x_free_if_h, x_true_if_free_h)
    _series_diag("free-self", x_self_if_h, x_true_if_self_h)
    _series_diag("driven", x_driven_gamma_h, x_true_h)

    def _nmse_diag(Xp, Y, name, var_ref):
        # flatten T x N
        xs, ys = Xp.reshape(-1), Y.reshape(-1)
        corr = np.corrcoef(xs, ys)[0, 1]
        mse = float(np.mean((xs - ys) ** 2))
        vy = float(np.var(ys))
        # what eval_rollout uses:
        nmse_eval = eval_rollout(Y, Xp, var_ref=var_ref)["nmse"]
        print(
            f"[nmse:{name}] corr={corr:.4f}  mse={mse:.4g}  var(Y_slice)={vy:.4g}  nmse_eval={nmse_eval:.4g}  mse/var(Y_slice)={mse / vy if vy > 0 else np.nan:.4g}")

    _nmse_diag(x_free_if_h, x_true_if_free_h, "if-free", var_ref)
    _nmse_diag(x_driven_gamma_h, x_true_h, "driven", var_ref)


    # --- NODE METRICS (streamed printing) ---
    node_rows = _eval_node_metrics(
        seed, dt_star, gamma_star, alpha_hat, num_nodes, var_ref,
        x_true_h, x_driven_gamma_h, x_driven_affine_h, x_free_h,
        x_free_if_h, x_true_if_free_h, x_self_if_h, x_true_if_self_h,
        dt_free_star, gamma_free_star
    )

    # --- EDGE METRICS (streamed printing) ---
    grid_E = np.array(grid_edges(hgt, wdt), dtype=np.int64)  # undirected (E,2)
    edge_rows = _eval_edge_metrics(
        seed, grid_E, y_holdout,
        hold_probs_cal, risk_sets_hold, risk_labels,
        x_driven_gamma_h, x_driven_affine_h, x_free_h,
        static_scores=static_scores
    )

    # --- VISUALIZATION ---
    if do_viz:
        _viz_variants(
            num_nodes, hgt, wdt, y_holdout,
            x_true_h, x_driven_gamma_h, x_driven_affine_h, x_free_h,
            hold_probs_cal, risk_sets_hold, grid_E,
            frame_kwargs
        )

    return {
        "node_metrics": node_rows,
        "edge_metrics": edge_rows,
        "dt_star": float(dt_star),
        "gamma_star": float(gamma_star),
        "dt_free_star": float(dt_free_star),
        "gamma_free_star": float(gamma_free_star),
    }


# ============================================
# Static graph measurements (from events)
# ============================================

def csr_bins_to_uvt(csr_bins: List[csr_matrix], dt: float = 1.0, t0: float = 0.0) -> np.ndarray:
    """Convert list[csr_matrix] (one per bin, with 0/1 entries) to (M,3) array [u,v,t]."""
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


@dataclass(frozen=True)
class GraphConstructSpec:
    window: int
    weight_mode: str
    decay_tau: Optional[float]
    threshold_mode: str
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


def _weighted_edges(
    window_events: np.ndarray, t_end: float, spec: GraphConstructSpec, dt: float
) -> Dict[Tuple[int, int], float]:
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
    deg1 = dict(nx.degree(G1))
    deg2 = dict(nx.degree(G2))
    d1 = np.array([deg1.get(int(n), 0.0) for n in nodes], dtype=float)
    d2 = np.array([deg2.get(int(n), 0.0) for n in nodes], dtype=float)

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


def make_specs(
    windows,
    weight_modes,
    thresholds,
    directed=True,
):
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
    eval_times: Iterable[int],
    horizon_bins: int,
    neg_ratio: float,
    seed: int,
    pbar,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    assert A0.shape is not None, "Substrate adjacency must have shape"
    N = int(A0.shape[0])
    G_sub = _nx_from_substrate(A0, directed=True)

    E_sub = csr_to_set(A0)
    prev_by_spec: Dict[str, nx.Graph] = {}
    rows: List[Dict[str, Any]] = []

    for tbin in eval_times:
        t_end = float(tbin) * dt

        for spec in specs:
            key = spec.key()
            Gm = construct_measurement_graph(events_uvt, t_end, spec, num_nodes=N, dt=dt)
            pbar.update(1)
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

            if key in prev_by_spec:
                Gprev = prev_by_spec[key]
                j_prev = jaccard_edges(edge_set(Gprev), edge_set(Gm))
                d_prev = degree_rank_spearman(Gprev, Gm)
            else:
                j_prev = np.nan
                d_prev = np.nan

            prev_by_spec[key] = Gm

            E_m = edge_set(Gm)
            j_sub = jaccard_edges(E_m, E_sub)
            d_sub = degree_rank_spearman(G_sub, Gm)

            rows.append({
                "t_end": t_end,
                "spec": key,
                "auc": float(dm["auc"]),
                "ap": float(dm["ap"]),
                "n_pos": int(dm["n_pos"]),
                "n_neg": int(dm["n_neg"]),
                "jacc_prev": float(j_prev),
                "degspe_prev": float(d_prev),
                "nmi_prev": float(0.0),
                "jacc_sub": float(j_sub),
                "degspe_sub": float(d_sub),
                "nmi_sub": float(0.0),
            })

    return pd.DataFrame(rows)
