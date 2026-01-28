# core.py
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union, cast, Sequence

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from scipy.sparse import csr_matrix, coo_matrix
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm.auto import trange
from contextlib import nullcontext
import numpy as np

# ---------------------------
# Typing aliases
# ---------------------------
PairIdx = Tuple[int, int]


# ---------------------------
# Config
# ---------------------------
@dataclass
class IFConfig:
    """
    Configuration container for Interaction-Field training.

    Attributes
    ----------
    mode : {"diffusion","attention"}
        Dynamics choice. This file supports "diffusion" end-to-end.
    show_progress : bool
        Progress bars via tqdm.
    dt : float
        Discrete time step length (in "bins").
    epochs : int
        Maximum training epochs.
    alpha : float
        Weight on the discrete action (regularizer) term.
    lambda_eff : float
        Local quartic potential strength (0 disables).
    k : int
        (Reserved) Top-k neighborhood for candidate sets (not used in diffusion mode rollout).
    tau_mem : float
        Exponential memory time constant for edge counts.
    s_neg : int
        (Reserved) Negatives per bin (not used in diffusion mode rollout).
    eps_kernel : float
        Stability epsilon for building the normalized kernel S.
    use_candidates_diffusion : bool
        (Reserved) If diffusion should also use candidate survival.
    link : {"exp","softplus"}
        Link function used for intensities.
    weight_decay : float
        L2 weight decay in Adam.
    early_stop_patience : int
        Patience on relative NLL improvements.
    early_stop_min_rel_improv : float
        Minimum relative improvement in NLL to reset patience.
    device : str | torch.device
        Target device.
    lr : float
        Base learning rate.
    lr_min : float
        Minimum LR for cosine schedules.
    lr_schedule : {"constant","cosine","onecycle","warmup_cosine"}
        LR schedule strategy.
    warmup_epochs : int
        Warmup epochs for "warmup_cosine".
    warmup_factor : float
        Start factor for warmup.
    max_lr : float
        Peak LR for onecycle.
    free_eval : {"driven","zero","self"}
        How to free-run during evaluation (training uses "driven").
    free_eval_start_frac : float
        Fraction where to split driven/free during eval.
    sample_mode : {"bernoulli","poisson"}
        Event sampling mode when free_eval="self".
    sample_cap : float
        Cap on lam*dt during event sampling.
    do_eval_each_epoch : bool
        (Reserved) Enable per-epoch eval.
    node_head : {"none","linear"}
        Optional node readout head to align latent to node-level targets.
    node_target_mode : {"in","out","sum"}
        Which node count stream to track as auxiliary target.
    node_ema_alpha : float
        EMA smoothing for node targets over time.
    node_loss_w : float
        (Deprecated—auto-weighted below) retained for compatibility.
    node_readout_bias : bool
        Include bias term in node head.
    node_readout_per_node : bool
        Per-node gain (diag) vs scalar gain.
    node_head_outputs_z : bool
        If True, the head predicts z directly; here we z-standardize after head.
    exposure_mask : Optional[Tensor]
        Optional (N,N) mask restricting risk sets (1=allowed).
    z_tanh_epochs : int
        Optional warmup with tanh on z.
    node_huber_delta : float
        Huber delta for node auxiliary loss.
    node_loss_target_frac : float
        Target fraction of total loss for node aux (auto-weight).
    node_w_min_floor : float
        Floor for node aux weight during warm-up.
    node_loss_warmup_epochs : int
        Epochs to warm up node aux weight.
    """
    mode: str = "diffusion"
    show_progress: bool = True
    dt: float = 0.1
    epochs: int = 50
    alpha: float = 1.0
    lambda_eff: float = 0.0
    k: int = 8
    tau_mem: float = 1.0
    s_neg: int = 32
    eps_kernel: float = 1e-3
    use_candidates_diffusion: bool = False
    link: str = "exp"
    weight_decay: float = 1e-4
    early_stop_patience: int = 100
    early_stop_min_rel_improv: float = 1e-3
    device: Union[str, torch.device] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # LR scheduling
    lr: float = 5e-3
    lr_min: float = 1e-4
    lr_schedule: str = "cosine"
    warmup_epochs: int = 0
    warmup_factor: float = 0.1
    max_lr: float = 3e-2

    # Free/eval
    free_eval: str = "driven"
    free_eval_start_frac: float = 0.7
    sample_mode: str = "poisson"
    sample_cap: float = 0.5
    do_eval_each_epoch: bool = False

    # Node readout/aux
    node_head: str = "linear"
    node_target_mode: str = "sum"
    node_ema_alpha: float = 0.2
    node_loss_w: float = 0.5
    node_readout_bias: bool = True
    node_readout_per_node: bool = True
    node_head_outputs_z: bool = True
    exposure_mask: Optional[Tensor] = None
    z_tanh_epochs: int = 0
    node_huber_delta: float = 1.0
    node_loss_target_frac: float = 0.4
    node_w_min_floor: float = 1e-1
    node_loss_warmup_epochs: int = 5


# ---------------------------
# Utilities
# ---------------------------
def inv_softplus(y: Tensor) -> Tensor:
    """
    Stable inverse of softplus.

    Parameters
    ----------
    y : Tensor
        Positive input.

    Returns
    -------
    Tensor
        x such that softplus(x) ≈ y.
    """
    return torch.where(y > 20, y, y + torch.log1p(-torch.exp(-y)))


# Compatibility helper for autocast: prefer `torch.amp.autocast` if available,
# fall back to `torch.cuda.amp.autocast`, otherwise no-op `nullcontext`.
def _autocast_ctx(device_type: str | None, enabled: bool):
    amp_mod = getattr(torch, "amp", None)
    if amp_mod is not None and getattr(amp_mod, "autocast", None) is not None:
        return amp_mod.autocast(device_type=device_type, enabled=enabled)
    cuda_mod = getattr(torch, "cuda", None)
    if cuda_mod is not None and getattr(cuda_mod, "amp", None) is not None and getattr(cuda_mod.amp, "autocast", None) is not None:
        return cuda_mod.amp.autocast(enabled=enabled)
    return nullcontext()


def _make_grad_scaler(device_type: str | None, enabled: bool):
    """Return a compatible GradScaler instance or a noop shim.

    Tries `torch.amp.GradScaler`, then `torch.cuda.amp.GradScaler`.
    Falls back to a no-op scaler with the same minimal interface.
    """
    amp_mod = getattr(torch, "amp", None)
    if amp_mod is not None and getattr(amp_mod, "GradScaler", None) is not None:
        try:
            return amp_mod.GradScaler(device_type, enabled=enabled)
        except TypeError:
            try:
                return amp_mod.GradScaler(enabled=enabled)
            except TypeError:
                try:
                    return amp_mod.GradScaler()
                except Exception:
                    pass

    cuda_amp = getattr(getattr(torch, "cuda", None), "amp", None)
    if cuda_amp is not None and getattr(cuda_amp, "GradScaler", None) is not None:
        try:
            return cuda_amp.GradScaler(enabled=enabled)
        except TypeError:
            try:
                return cuda_amp.GradScaler()
            except Exception:
                pass

    # No-op shim
    class _NoOpScaled:
        def __init__(self, loss):
            self.loss = loss

        def backward(self):
            # perform real backward
            self.loss.backward()

    class _NoOpScaler:
        def __init__(self, *args, **kwargs):
            pass

        def scale(self, loss):
            return _NoOpScaled(loss)

        def step(self, optimizer):
            try:
                optimizer.step()
            except Exception:
                pass

        def update(self):
            return None

        def get_scale(self):
            return 1.0

        def state_dict(self):
            return {}

        def load_state_dict(self, d):
            return None

    return _NoOpScaler()


def _resolve_tau(cfg: IFConfig) -> float:
    """Resolve the memory τ from config."""
    return float(getattr(cfg, "tau_mem", 3.0))


def csr_bins_to_uv_tensors(csr_bins: List[csr_matrix], *, device: Union[str, torch.device] = "cpu") -> List[Tensor]:
    """
    Convert CSR bins to (2,M) long tensors of (u,v) per bin on the target device.
    Empty bins → shape (2,0).

    Parameters
    ----------
    csr_bins : list of scipy.sparse.csr_matrix
    device : torch device or str

    Returns
    -------
    list[Tensor]
    """
    dev = torch.device(device)
    out: List[Tensor] = []
    for A in csr_bins:
        if not isinstance(A, csr_matrix):
            raise TypeError("All bins must be scipy.sparse.csr_matrix")
        if A.nnz == 0:
            out.append(torch.empty((2, 0), dtype=torch.long, device=dev))
        else:
            coo = coo_matrix(A)
            u = torch.from_numpy(coo.row.astype(np.int64, copy=False))
            v = torch.from_numpy(coo.col.astype(np.int64, copy=False))
            out.append(torch.stack((u, v), dim=0).to(device=dev))
    return out


def pairs_from_mask(mask: Tensor) -> Tensor:
    """
    Convert boolean/0-1 (N,N) mask to 2xM LongTensor of (u,v) on same device.

    Parameters
    ----------
    mask : Tensor
        (N,N) bool or 0/1 float.

    Returns
    -------
    Tensor
        (2, M) long tensor.
    """
    uv = torch.nonzero(mask, as_tuple=False).T
    return uv.to(dtype=torch.long)


def csr_nonzero_pairs(S_csr: Tensor) -> Tensor:
    """
    Extract (u,v) pairs from torch CSR matrix.

    Parameters
    ----------
    S_csr : Tensor
        torch.sparse_csr_tensor of shape (N,N)

    Returns
    -------
    Tensor
        (2, M) long tensor on same device.
    """
    rowptr = S_csr.crow_indices()
    cols = S_csr.col_indices()
    N = rowptr.numel() - 1
    rows = torch.arange(N, device=S_csr.device).repeat_interleave(rowptr[1:] - rowptr[:-1])
    return torch.stack([rows, cols], dim=0)


def csr_to_uv_tensor(M: csr_matrix, *, max_repeat: int = 20, device: Union[str, torch.device] = "cuda") -> torch.Tensor:
    """Expand weighted CSR into (2, M) long tensor of (u,v) events.
       Caps repeats per edge to keep things fast."""
    coo = coo_matrix(M)
    if coo.nnz == 0:
        return torch.empty((2,0), dtype=torch.long, device=device)
    u = coo.row
    v = coo.col
    w = np.asarray(coo.data, dtype=np.int64)
    # repeat caps
    r = np.minimum(w, max_repeat)
    if (r <= 0).all():
        return torch.empty((2,0), dtype=torch.long, device=device)
    idx = np.repeat(np.arange(len(u)), r)
    uv = np.stack([u[idx], v[idx]], axis=0)
    return torch.as_tensor(uv, dtype=torch.long, device=device)


def scatter_incident_counts(
    N: int,
    pairs: Union[List[PairIdx], Tensor],
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """
    Count node incidents in a bin.

    Parameters
    ----------
    N : int
        Number of nodes.
    pairs : list[(u,v)] | Tensor
        (M,2) or (2,M) int/long.
    device : torch.device, optional
    dtype : torch.dtype, optional

    Returns
    -------
    Tensor
        Shape (N,) on device/dtype.
    """
    dtype = dtype or torch.float32

    # Fast empty paths
    if isinstance(pairs, list) and len(pairs) == 0:
        return torch.zeros(N, device=device or torch.device("cpu"), dtype=dtype)
    if isinstance(pairs, Tensor) and pairs.numel() == 0:
        return torch.zeros(N, device=pairs.device, dtype=dtype)

    # Normalize -> (2,M) on device
    if isinstance(pairs, Tensor):
        uv = pairs.T if (pairs.dim() == 2 and pairs.size(1) == 2) else pairs
        uv = uv.to(dtype=torch.long)
        dev = uv.device
    else:
        dev = device or torch.device("cpu")
        uv = torch.as_tensor(pairs, dtype=torch.long, device=dev).T

    idx_flat = torch.cat([uv[0], uv[1]], dim=0)
    counts = torch.bincount(idx_flat, minlength=N)
    return counts.to(dtype=dtype)


def bin_source_from_events(
    N: int,
    events_b: Union[List[PairIdx], Tensor],
    dt: float,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    expand_to_d: Optional[int] = None,
) -> Tensor:
    """
    Build node source term J^b from bin edge events via incident counts.

    Parameters
    ----------
    N : int
    events_b : list[(u,v)] | Tensor
        (M,2) or (2,M) of ints/longs.
    dt : float
    device : torch.device, optional
    dtype : torch.dtype, optional
    expand_to_d : int, optional
        If set, returns (N,d) by broadcasting.

    Returns
    -------
    Tensor
        (N,) or (N,d)
    """
    prefer_dev = events_b.device if isinstance(events_b, Tensor) else device
    inc = scatter_incident_counts(N, events_b, device=prefer_dev, dtype=dtype)
    J = inc / max(float(dt), 1e-12)
    if expand_to_d is not None:
        J = J.unsqueeze(-1).expand(N, expand_to_d)
    return J


# ---------------------------
# SparseMemory and kernels
# ---------------------------
class SparseMemory:
    """
    Exponentially-decayed directed counts stored lazily via a global multiplier.
    """
    def __init__(self, N: int, decay_per_step: float, device: torch.device):
        self.N = N
        self.decay = float(decay_per_step)
        self.device = device
        self._idx = torch.empty((2, 0), dtype=torch.long, device=device)
        self._val = torch.empty((0,), dtype=torch.float32, device=device)
        self._g = 1.0

    @torch.no_grad()
    def step_decay(self) -> None:
        """Apply one step of global decay."""
        self._g *= (1.0 - self.decay)

    @torch.no_grad()
    def add_events(self, events_b: Union[List[PairIdx], Tensor]) -> None:
        """Accumulate new directed (u,v) events for the current bin."""
        if (isinstance(events_b, list) and not events_b) or (
            isinstance(events_b, Tensor) and events_b.numel() == 0
        ):
            return

        if isinstance(events_b, Tensor):
            ij = events_b if (events_b.dim() == 2 and events_b.size(0) == 2) else events_b.T
            ij = ij.to(device=self.device, dtype=torch.long, non_blocking=True)
        else:
            ij = torch.as_tensor(events_b, dtype=torch.long, device=self.device).T

        val = torch.full((ij.shape[1],), 1.0 / max(self._g, 1e-20), device=self.device)
        self._idx = torch.cat([self._idx, ij], dim=1)
        self._val = torch.cat([self._val, val], dim=0)

    @torch.no_grad()
    def as_sparse(self) -> Tensor:
        """Return current memory as sparse COO (N,N) with decay applied once."""
        M = torch.sparse_coo_tensor(self._idx, self._val, (self.N, self.N), device=self.device).coalesce()
        return torch.sparse_coo_tensor(M.indices(), self._g * M.values(), (self.N, self.N), device=self.device).coalesce()


@torch.no_grad()
def normalized_S_from_events_sparse(M: Tensor, eps: float = 1e-3) -> Tensor:
    """
    Build symmetric degree-normalized kernel S from directed counts M.

    Parameters
    ----------
    M : Tensor
        Sparse COO (N,N) of decayed directed counts.
    eps : float
        Diagonal jitter.

    Returns
    -------
    Tensor
        Sparse COO (N,N) S = D^{-1/2} C D^{-1/2}, C = 0.5(M+M^T)+eps I.
    """
    M = M.coalesce()
    N = M.size(0)

    Mt = torch.sparse_coo_tensor(M.indices().flip(0), M.values(), (N, N), device=M.device).coalesce()
    C = (0.5 * (M + Mt)).coalesce()

    diag = torch.arange(N, device=M.device)
    I_eps = torch.sparse_coo_tensor(torch.stack([diag, diag]), torch.full((N,), eps, device=M.device),
                                    (N, N), device=M.device)
    C = (C + I_eps).coalesce()

    deg = torch.sparse.sum(C, dim=1).to_dense()
    dmi = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))

    rows, cols = C.indices()
    vals = C.values() * dmi[rows] * dmi[cols]
    return torch.sparse_coo_tensor(C.indices(), vals, (N, N), device=M.device).coalesce()


# ---------------------------
# Dynamics
# ---------------------------
def _euler_step_if_with_Kh(h: Tensor, J: Tensor, m2: Tensor, c2: Tensor, Kh: Tensor, lambda_eff: float, dt: float) -> Tensor:
    """
    One Euler step with precomputed Kh.

    dh = -m^2 h - c^2 Kh - (λ/3!) h^3 + J

    Parameters
    ----------
    h : (N,d)
    J : (N,) | (N,1) | (N,d)
    m2, c2 : scalars (Tensor)
    Kh : (N,d) == (K @ h)
    lambda_eff : float
    dt : float

    Returns
    -------
    Tensor
        Next h, shape (N,d)
    """
    N, d = h.shape
    if J.dim() == 1:
        J = J[:, None].expand(N, d)
    elif J.shape[1] == 1 and d > 1:
        J = J.expand(N, d)

    g = 0.0
    if lambda_eff and lambda_eff != 0.0:
        g = (lambda_eff / 6.0) * (h ** 3)

    dh = -m2 * h - c2 * Kh - g + J
    return h + dt * dh


try:
    euler_step_if_with_Kh_fast = torch.compile(_euler_step_if_with_Kh, fullgraph=False)
except Exception:
    euler_step_if_with_Kh_fast = _euler_step_if_with_Kh


# ---------------------------
# Observation / intensities
# ---------------------------
def intensities_on_pairs(h: Tensor, W_obs: Tensor, b0: Tensor, link: str, pairs_uv: Tensor) -> Tensor:
    """
    Compute λ_{uv} for a set of (u,v) pairs.

    Supports W_obs shapes:
      - (d,d): full bilinear
      - (d,): elementwise dot weights
      - scalar: scales dot
      - other: fallback to dot

    Parameters
    ----------
    h : (N,d)
    W_obs : Tensor
    b0 : Tensor
    link : {"exp","softplus","sigmoid"}
    pairs_uv : (2, M) long

    Returns
    -------
    Tensor
        (M,)
    """
    u, v = pairs_uv[0], pairs_uv[1]
    hu = h.index_select(0, u)
    hv = h.index_select(0, v)
    d = h.size(1)

    W = W_obs
    if W.ndim == 2:
        if W.shape == (d, d):
            s = (hu @ W * hv).sum(-1)
        elif W.numel() == 1:
            s = (hu * hv).sum(-1) * W.reshape(())
        else:
            s = (hu * hv).sum(-1)
    elif W.ndim == 1:
        if W.numel() == d:
            s = (hu * hv * W).sum(-1)
        elif W.numel() == 1:
            s = (hu * hv).sum(-1) * W.reshape(())
        else:
            s = (hu * hv).sum(-1)
    else:
        s = (hu * hv).sum(-1) * (W.reshape(()) if W.numel() == 1 else 1.0)

    s = s + b0.reshape(())

    if link == "exp":
        lam = torch.exp(s)
    elif link == "softplus":
        lam = F.softplus(s)
    elif link == "sigmoid":
        lam = torch.sigmoid(s)
    else:
        lam = s
    return torch.clamp(lam, min=1e-12)


# NOTE: `scipy.sparse.csr_matrix/coo_matrix` and `torch` are imported at file top;
# `numpy` is imported above to keep usage consistent across functions.

def build_L_eff_from_train(
    theta: IFParameters,
    y_train_csr: List[csr_matrix],
    n: int,
    cfg: IFConfig,
    *,
    eps: float = 1e-3,
    to_device: bool = True,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Rebuild (from TRAIN bins only) a symmetric, degree-normalized kernel S using the
    same memory decay as the IF rollout, then form

        L_eff = m^2 I + c^2 (I - S)

    and return it as a **dense** float32 tensor (symmetrized) on cfg.device.

    Parameters
    ----------
    theta : IFParameters
        Provides m^2 and c^2.
    y_train_csr : list[csr_matrix]
        One directed (u->v) CSR per training bin.
    n : int
        Number of nodes.
    cfg : IFConfig
        Uses cfg.dt and cfg.tau_mem for decay, and cfg.device for output device.
    eps : float
        Small diagonal jitter added to C before normalization.
    to_device : bool
        If True, move result to cfg.device.
    dtype : torch.dtype
        Output dtype.

    Returns
    -------
    torch.Tensor
        Dense (n,n) symmetric L_eff suitable for eigendecomposition.
    """
    # ---- rebuild decayed directed counts M via M_{t+1} = decay_factor * M_t + A_t
    # where decay_factor = exp(-dt / tau) and mu = 1 - decay_factor (incremental decay)
    dt = float(cfg.dt)
    tau = float(getattr(cfg, "tau_mem", 1.0))
    decay_factor = float(np.exp(-dt / max(tau, 1e-12)))
    mu = 1.0 - decay_factor  # incremental decay per bin (kept for documentation)

    M = csr_matrix((n, n), dtype=np.float32)
    for A_t in y_train_csr:
        if not isinstance(A_t, csr_matrix):
            raise TypeError("All bins must be scipy.sparse.csr_matrix")
        if M.nnz:
            M = M.multiply(decay_factor) + A_t
        else:
            # First non-empty, just assign (empty stays empty)
            M = A_t.copy()

    # ---- symmetrize and add jitter: C = 0.5 (M + M^T) + eps I
    if M.nnz:
        C = (M + M.T).multiply(0.5)
    else:
        C = csr_matrix((n, n), dtype=np.float32)

    if eps > 0:
        diag_idx = np.arange(n, dtype=np.int64)
        C = C + csr_matrix((np.full(n, eps, dtype=np.float32), (diag_idx, diag_idx)), shape=(n, n))

    # ---- degree-normalize: S = D^{-1/2} C D^{-1/2}
    deg = np.asarray(C.sum(axis=1)).ravel().astype(np.float32)
    with np.errstate(divide="ignore"):
        d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 0.0))
    d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.0

    C_coo = coo_matrix(C)
    if C_coo.nnz:
        scaled = C_coo.data * d_inv_sqrt[C_coo.row] * d_inv_sqrt[C_coo.col]
        S = coo_matrix((scaled, (C_coo.row, C_coo.col)), shape=(n, n), dtype=np.float32).tocsr()
    else:
        S = csr_matrix((n, n), dtype=np.float32)

    # ---- to dense torch, form K = I - S, then L_eff = m^2 I + c^2 K
    dev = cfg.device if isinstance(cfg.device, torch.device) else torch.device(cfg.device)
    S_dense = torch.from_numpy(S.toarray()).to(dtype=dtype)
    if to_device:
        S_dense = S_dense.to(dev)

    I = torch.eye(n, dtype=dtype, device=S_dense.device)
    K = I - S_dense
    L = theta.m2.float() * I + theta.c2.float() * K
    # Symmetrize for numerical cleanliness
    L = 0.5 * (L + L.T)
    return L.contiguous()



def discrete_poisson_nll_risk(
    lambda_list: Sequence[Tensor],
    Y_bins: Sequence[Union[List[PairIdx], Tensor]],
    risk_sets: Sequence[List[PairIdx]],
    dt: float,
    *,
    N: Optional[int] = None,
    device: Optional[torch.device] = None,
    normalize: bool = True,
) -> Tensor:
    """
    Poisson NLL over risk sets:

        sum_b [ sum_{(u,v) in R_b} λ_{uv}^b dt  -  sum_{events in R_b} log λ_{uv}^b ].

    With normalize=True:
        ( dt * mean_{risk} λ ) - ( mean_{matched events} log λ )

    Parameters
    ----------
    lambda_list : list[(M_b,)]
    Y_bins : list[list[(u,v)] | Tensor]
        Observed events per bin.
    risk_sets : list[list[(u,v)]]
        Risk pairs per bin (same indexing as lambda_list).
    dt : float
    N : int, optional
        Used for hashing pairs. If None, inferred from risk sets.
    device : torch.device, optional
    normalize : bool
        Normalize by counts to reduce dataset-size dependence.

    Returns
    -------
    Tensor
        Scalar NLL.
    """
    if not lambda_list:
        return torch.tensor(0.0)

    if device is None:
        device = lambda_list[0].device

    def _to_uv2(pairs) -> Tensor:
        if isinstance(pairs, Tensor):
            if pairs.numel() == 0:
                return torch.empty(2, 0, dtype=torch.long, device=device)
            return pairs.T.to(device=device, dtype=torch.long) if pairs.size(1) == 2 else pairs.to(device=device, dtype=torch.long)
        if len(pairs) == 0:
            return torch.empty(2, 0, dtype=torch.long, device=device)
        return torch.as_tensor(pairs, dtype=torch.long, device=device).T

    if N is None:
        max_v = 0
        for R in risk_sets:
            if R:
                t = torch.as_tensor(R, dtype=torch.long)
                max_v = max(max_v, int(t[:, 1].max().item()))
        N = max_v + 1

    pos_sum = torch.zeros((), device=device)
    neg_sum = torch.zeros((), device=device)
    pos_cnt = 0
    neg_cnt = 0

    for b, lam_b in enumerate(lambda_list):
        if lam_b.numel() > 0:
            pos_sum = pos_sum + lam_b.sum() * dt
            pos_cnt += lam_b.numel()

        if not risk_sets[b]:
            continue

        R_uv = _to_uv2(risk_sets[b])
        Ru, Rv = R_uv[0], R_uv[1]
        risk_flat = Ru * N + Rv

        order = torch.argsort(risk_flat)
        risk_sorted = risk_flat[order]

        E_uv = _to_uv2(Y_bins[b])
        if E_uv.numel() == 0:
            continue

        Eu, Ev = E_uv[0], E_uv[1]
        ev_flat = Eu * N + Ev

        pos_in_sorted = torch.searchsorted(risk_sorted, ev_flat)
        in_bounds = pos_in_sorted < risk_sorted.numel()
        pos_in_sorted = torch.where(in_bounds, pos_in_sorted, risk_sorted.numel() - 1)
        matches = (risk_sorted[pos_in_sorted] == ev_flat) & in_bounds

        if matches.any():
            idx_in_risk = order[pos_in_sorted[matches]]
            ev_lam = lam_b.index_select(0, idx_in_risk)
            neg_sum = neg_sum + torch.log(ev_lam + 1e-12).sum()
            neg_cnt += ev_lam.numel()

    if normalize:
        pos_term = pos_sum / max(pos_cnt, 1)
        neg_term = neg_sum / max(neg_cnt, 1)
        return pos_term - neg_term

    return pos_sum - neg_sum


# ---------------------------
# Model parameters
# ---------------------------
class IFParameters(nn.Module):
    """
    Learnable parameters θ for diffusion-mode Interaction Fields.

    θ includes:
      - ρ_c2, ρ_m2: reparameterized via softplus for c^2, m^2 > 0
      - W_obs, b0 : observation bilinear and bias
      - h0        : initial latent field (N,d)
      - Optional node head / z-standardizer added by train_if.

    Methods
    -------
    c2, m2 : properties giving strictly-positive tensors.
    link(x) : apply selected link ("exp" or "softplus").
    parameter_count(trainable_only=True, by_group=False) : count parameters.
    """

    def __init__(self, N: int, d: int, config: IFConfig, attention: bool = False, eps: float = 1e-6):
        super().__init__()
        self.N, self.d = N, d
        self.config = config
        self.attention = attention

        self.register_buffer("eps", torch.tensor(eps))
        self.rho_c2 = nn.Parameter(inv_softplus(torch.as_tensor(0.1 - eps)))
        self.rho_m2 = nn.Parameter(inv_softplus(torch.as_tensor(0.5 - eps)))

        self.W_obs = nn.Parameter(torch.empty(d, d).uniform_(-0.1, 0.1))
        self.b0 = nn.Parameter(torch.tensor(0.0))

        self.h0 = nn.Parameter(torch.zeros(N, d))
        self.link_name = config.link

        # Buffers registered later by training code (readout calibrators)
        self.readout_mu: Optional[Tensor]
        self.readout_sig: Optional[Tensor]

        # Optional attention parameters (kept for API compatibility)
        if attention:
            self.beta = nn.Parameter(torch.tensor(1.0))
            self.W = nn.Parameter(torch.empty(d, d).uniform_(-0.1, 0.1))
            self.a = nn.Parameter(torch.empty(2 * d).uniform_(-0.1, 0.1))

        # These may be attached later by train_if (node_gain, node_bias, z_shift, z_scale)
        self.node_gain: Optional[Tensor] = None
        self.node_bias: Optional[Tensor] = None
        self.z_shift: Optional[Tensor] = None
        self.z_scale: Optional[Tensor] = None

    # ---- physical scalars ----
    @property
    def c2(self) -> Tensor:
        """Strictly-positive diffusion coefficient c^2."""
        s = F.softplus(self.rho_c2)
        return s + cast(Tensor, self.eps)

    @property
    def m2(self) -> Tensor:
        """Strictly-positive mass m^2."""
        s = F.softplus(self.rho_m2)
        return s + cast(Tensor, self.eps)

    # ---- observation link ----
    def link(self, x: Tensor) -> Tensor:
        if self.link_name == "exp":
            return torch.exp(x)
        if self.link_name == "softplus":
            return F.softplus(x)
        raise ValueError(f"Unknown link '{self.link_name}'")

    # ---- parameter counting ----
    def parameter_count(
        self,
        *,
        trainable_only: bool = True,
        by_group: bool = False,
    ) -> Union[int, Dict[str, int]]:
        """
        Count parameters.

        Parameters
        ----------
        trainable_only : bool
            If True, only count requires_grad params.
        by_group : bool
            If True, return a dict grouped by top-level names
            (e.g., 'core', 'node_head', 'z_standardizer').

        Returns
        -------
        int | dict[str,int]
        """
        def _numel(params: Iterable[nn.Parameter]) -> int:
            return sum(int(p.numel()) for p in params if (p.requires_grad or not trainable_only))

        if not by_group:
            return _numel(self.parameters())

        groups: Dict[str, List[nn.Parameter]] = {
            "core": [self.rho_c2, self.rho_m2, self.W_obs, self.b0, self.h0],
        }
        if self.attention:
            groups["attention"] = [self.beta, self.W, self.a]  # type: ignore[arg-type]

        if self.node_gain is not None or self.node_bias is not None:
            nh: List[nn.Parameter] = []
            if isinstance(self.node_gain, nn.Parameter):
                nh.append(self.node_gain)
            if isinstance(self.node_bias, nn.Parameter):
                nh.append(self.node_bias)
            if nh:
                groups["node_head"] = nh

        if isinstance(self.z_shift, nn.Parameter) or isinstance(self.z_scale, nn.Parameter):
            zs: List[nn.Parameter] = []
            if isinstance(self.z_shift, nn.Parameter):
                zs.append(self.z_shift)
            if isinstance(self.z_scale, nn.Parameter):
                zs.append(self.z_scale)
            if zs:
                groups["z_standardizer"] = zs

        return {k: _numel(v) for k, v in groups.items()}


def summarize_param_sizes(model):
    """
    Report number of parameters and memory usage (bytes) for each named parameter
    and per-module group.
    """
    rows = []
    total_params, total_bytes = 0, 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        b = n * p.element_size()  # element_size() in bytes (4 for float32, 2 for float16, etc.)
        rows.append((name, n, b, p.shape, str(p.dtype)))
        total_params += n
        total_bytes += b

    print(f"\nTrainable params: {total_params:,} "
          f"({total_bytes:,} bytes ≈ {total_bytes/1024/1024:.3f} MB)\n")

    # Detailed table
    print(f"{'name':30s} {'#params':>12s} {'bytes':>12s} {'shape':>20s} {'dtype':>10s}")
    for name, n, b, shape, dtype in rows:
        print(f"{name:30s} {n:12,d} {b:12,d} {str(tuple(shape)):>20s} {dtype:>10s}")

    # Group by top-level module prefix (everything before first '.')
    groups = {}
    for name, n, b, _, _ in rows:
        group = name.split('.')[0]
        groups.setdefault(group, {"params":0, "bytes":0})
        groups[group]["params"] += n
        groups[group]["bytes"] += b

    print("\nBy group:")
    for g, d in groups.items():
        print(f"  {g:15s}: {d['params']:12,d} params, {d['bytes']:12,d} bytes "
              f"(~{d['bytes']/max(d['params'],1):.2f} B/param)")

    print()
    return rows


# ---------------------------
# Forward rollout (diffusion)
# ---------------------------
@torch.inference_mode(False)
def forward_rollout_if(
    Y_bins: List[Tensor],
    theta: IFParameters,
    cfg: IFConfig,
    *,
    free_start: Optional[int] = None,
    free_mode: str = "driven",
    sample_mode: str = "bernoulli",
    sample_cap: float = 0.5,
    h0: Optional[Tensor] = None,
    mem_init_sparse: Optional[Tensor] = None,
    progress_desc: Optional[str] = None,
    progress_position: int = 1,
    enable_progress: bool = True,
) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[List[PairIdx]], List[Tensor], Tensor]:
    """
    Driven rollout through bins (diffusion mode).

    Parameters
    ----------
    Y_bins : list[Tensor]
        Each (2,M_t) long (u,v) edges for bin t on device.
    theta : IFParameters
    cfg : IFConfig
    free_start : int | None
        If set, switch to free dynamics at this bin index.
    free_mode : {"driven","zero","self"}
        Driven uses ground-truth events; zero clears; self samples from λ.
    sample_mode : {"bernoulli","poisson"}
    sample_cap : float
        Cap on lam*dt for sampling.
    h0 : Tensor | None
        Optional initial h (N,d).
    mem_init_sparse : Tensor | None
        Optional sparse COO memory to seed.
    progress_desc : str | None
        tqdm description.
    progress_position : int
        tqdm bar position.
    enable_progress : bool
        Toggle tqdm.

    Returns
    -------
    (H, J_list, K_list, risk_sets, lambda_list, M_final)
    """
    device = cfg.device if isinstance(cfg.device, torch.device) else torch.device(cfg.device)
    N, d = theta.h0.shape
    B = len(Y_bins)
    dt = cfg.dt

    H: List[Tensor] = [(h0 if h0 is not None else theta.h0).to(device, non_blocking=True)]
    J_list: List[Tensor] = []
    K_list: List[Tensor] = []
    risk_sets: List[List[PairIdx]] = []
    lambda_list: List[Tensor] = []

    # memory with exponential decay
    # `mu` is the incremental decay per step: mu = 1 - exp(-dt / tau).
    # The multiplicative factor applied to past counts is exp(-dt/tau) == (1.0 - mu).
    mu = 1.0 - math.exp(-dt / max(cfg.tau_mem, 1e-12))
    mem = SparseMemory(N, mu, torch.device(device))
    if mem_init_sparse is not None:
        # Just load the values into our lazy store
        M0 = mem_init_sparse.coalesce()
        mem._idx = M0.indices().clone()
        mem._val = M0.values().clone()
        mem._g = 1.0

    it: Iterable[int]
    if enable_progress and progress_desc:
        it = trange(B, desc=progress_desc, position=progress_position, leave=False, dynamic_ncols=True)
    else:
        it = range(B)

    torch.set_float32_matmul_precision("high")
    use_amp = (torch.device(device).type == "cuda")

    for b in it:
        mem.step_decay()

        # 1) Risk set from pre-events memory
        S_pre = normalized_S_from_events_sparse(mem.as_sparse(), eps=cfg.eps_kernel).coalesce()
        S_pre_csr = S_pre.to_sparse_csr()

        if cfg.exposure_mask is not None:
            R_pairs = pairs_from_mask((cfg.exposure_mask > 0).to(device))
        else:
            R_pairs = csr_nonzero_pairs(S_pre_csr)

        # 2) Events for this bin (driven vs free)
        in_free = (free_start is not None) and (b >= free_start)

        if (not in_free) or (free_mode == "driven"):
            events_uv = Y_bins[b]  # already (2,M) long on device
        elif free_mode == "zero":
            events_uv = torch.empty((2, 0), dtype=torch.long, device=device)
        elif free_mode == "self":
            lam_b = intensities_on_pairs(H[-1], theta.W_obs, theta.b0, cfg.link, R_pairs)
            # sample Bernoulli/Poisson
            if sample_mode == "bernoulli":
                tau = getattr(cfg, "hazard_tau", 1.0)  # set from fit_hazard_temperature on TRAIN
                p = 1.0 - torch.exp(-tau * lam_b * dt)  # calibrated Bernoulli
                keep = (torch.rand_like(p) < p.clamp_max(getattr(cfg, "p_cap", 1.0)))
                events_uv = R_pairs[:, keep]
            else:
                rate = torch.clamp(lam_b * dt, min=0.0, max=sample_cap)
                k = torch.poisson(rate).clamp_max(max(1, int(math.ceil(sample_cap))))
                if (k > 0).any():
                    idx = torch.repeat_interleave(torch.arange(R_pairs.size(1), device=device), k.long())
                    events_uv = R_pairs[:, idx]
                else:
                    events_uv = torch.empty((2, 0), dtype=torch.long, device=device)
        else:
            raise ValueError(f"free_mode must be one of 'driven','zero','self', got {free_mode}")

        # 3) Update memory and source J
        mem.add_events(events_uv)
        J_b = bin_source_from_events(N, events_uv, dt, device=device)  # (N,)
        J_list.append(J_b)

        # 4) Euler step using S after adding current events
        S_coo = normalized_S_from_events_sparse(mem.as_sparse(), eps=cfg.eps_kernel).coalesce()
        S_csr = S_coo.to_sparse_csr()
        K_b = S_csr  # for diffusion, K := I - S; we apply as Kh = h - S h below
        K_list.append(K_b)

        with _autocast_ctx(device.type, use_amp):
            Sh = S_csr @ H[-1]
            Kh = H[-1] - Sh
            h_next = euler_step_if_with_Kh_fast(H[-1], J_b, theta.m2, theta.c2, Kh, cfg.lambda_eff, dt)

        # 5) Store λ on the *pre-events* risk set (for NLL)
        lam_store = intensities_on_pairs(H[-1], theta.W_obs, theta.b0, cfg.link, R_pairs)
        lambda_list.append(lam_store)

        # 6) Bookkeeping
        H.append(h_next)
        risk_sets.append(list(map(tuple, R_pairs.T.detach().cpu().tolist())))

    return H, J_list, K_list, risk_sets, lambda_list, mem.as_sparse()


# ---------------------------
# Node reduction / readout
# ---------------------------
def reduce_H_to_nodes(
    H_seq,
    N: int,
    device,
    *,
    drop_first_if_vector: bool = True,
    reduce_mode: str = "sym",     # only used when x is (N,N): {"sym","out","in","skew","diag"}
    use_mean: bool = True,         # only used when x is (N,N)
    center: bool = True,
    channel_reduce: str = "mean",  # how to map (N,d) -> (N,). {"mean","sum","l2","first","max"}
) -> torch.Tensor:
    """
    Reduce a sequence of latent/state tensors to node scalars X[T, N].

    Accepts per-step tensors x_t of shapes:
      - (N, 1) or (N,)            → node vector
      - (N, d) with d>1           → node *channels* (reduced via `channel_reduce`)
      - (N, N)                    → node×node matrix (reduced via `reduce_mode`)

    Args
    ----
    reduce_mode: only affects (N,N) inputs.
    channel_reduce:
        - "mean": mean over channels dim
        - "sum" : sum over channels
        - "l2"  : L2 norm over channels
        - "first": take channel 0
        - "max" : max over channels
    """
    X = []
    for t, Ht in enumerate(H_seq):
        x = Ht

        # (N,1) or (N,) → flatten to (N,)
        if (x.dim() == 2 and x.shape == (N, 1)) or (x.dim() == 1 and x.shape[0] == N):
            if drop_first_if_vector and t == 0:
                continue
            x = x.view(N)

        # (N,d) with d>1 → reduce channels to (N,)
        elif x.dim() == 2 and x.shape[0] == N and x.shape[1] != N:
            d = x.shape[1]
            if drop_first_if_vector and t == 0 and d == 1:
                # handled above; keep for completeness
                continue
            if channel_reduce == "mean":
                x = x.mean(dim=1)
            elif channel_reduce == "sum":
                x = x.sum(dim=1)
            elif channel_reduce == "l2":
                x = torch.linalg.vector_norm(x, dim=1)
            elif channel_reduce == "first":
                x = x[:, 0]
            elif channel_reduce == "max":
                x = x.max(dim=1).values
            else:
                raise ValueError(f"Unknown channel_reduce='{channel_reduce}'")

        # (N,N) matrix → row/col reduce without collapsing node axis
        elif x.dim() == 2 and x.shape == (N, N):
            if use_mean:
                row = x.mean(dim=1)  # [N]
                col = x.mean(dim=0)  # [N]
            else:
                row = x.sum(dim=1)
            # NOTE: only compute col if needed
                col = x.sum(dim=0)

            if reduce_mode == "sym":
                x = 0.5 * (row + col)
            elif reduce_mode == "out":
                x = row
            elif reduce_mode == "in":
                x = col
            elif reduce_mode == "skew":
                x = 0.5 * (row - col)
            elif reduce_mode == "diag":
                x = x.diag()
            else:
                x = 0.5 * (row + col)

        else:
            # As a last resort, if last dim matches N, collapse it; else raise
            if x.shape[-1] == N:
                x = x.reshape(-1, N)
                if x.shape[0] != 1:
                    # e.g., (d,N) → choose a safe reduce over leading dim
                    if channel_reduce == "mean":
                        x = x.mean(dim=0)
                    elif channel_reduce == "sum":
                        x = x.sum(dim=0)
                    elif channel_reduce == "l2":
                        x = torch.linalg.vector_norm(x, dim=0)
                    elif channel_reduce == "first":
                        x = x[0]
                    elif channel_reduce == "max":
                        x = x.max(dim=0).values
                    else:
                        raise ValueError(f"Unknown channel_reduce='{channel_reduce}'")
                else:
                    x = x.view(N)
            else:
                raise RuntimeError(f"Unsupported tensor shape at t={t}: {tuple(x.shape)}; "
                                   f"expected (N,), (N,1), (N,d), or (N,N) with N={N}")

        if center:
            x = x - x.mean()

        X.append(x.to(dtype=torch.float32, device=device))

    return torch.stack(X, dim=0)  # [T', N]



# ---------------------------
# Optimizer / scheduler
# ---------------------------
def make_optimizer_and_scheduler(theta: IFParameters, cfg: IFConfig, steps_per_epoch: int = 1):
    """
    Build Adam optimizer and LR scheduler for the model.
    """
    opt = optim.Adam(theta.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    schedule = getattr(cfg, "lr_schedule", "constant").lower()
    scheduler = None
    step_when = "epoch"

    if schedule == "constant":
        scheduler = None

    elif schedule == "cosine":
        scheduler = lrs.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr_min)
        step_when = "epoch"

    elif schedule == "warmup_cosine":
        warm = max(0, int(getattr(cfg, "warmup_epochs", 0)))
        warmup = lrs.LinearLR(opt, start_factor=getattr(cfg, "warmup_factor", 0.1), end_factor=1.0, total_iters=max(1, warm))
        cosine = lrs.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs - warm), eta_min=cfg.lr_min)
        scheduler = lrs.SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warm])
        step_when = "epoch"

    elif schedule == "onecycle":
        total_steps = cfg.epochs * max(1, steps_per_epoch)
        scheduler = lrs.OneCycleLR(
            opt,
            max_lr=getattr(cfg, "max_lr", cfg.lr * 3),
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=max(1.0, getattr(cfg, "max_lr", cfg.lr * 3) / cfg.lr),
            final_div_factor=getattr(cfg, "max_lr", cfg.lr * 3) / max(cfg.lr_min, 1e-8),
        )
        step_when = "step"
    else:
        raise ValueError(f"Unknown lr_schedule={schedule}")

    return opt, scheduler, step_when


# ---------------------------
# Training
# ---------------------------
def train_if(
    y_bins_csr: List[csr_matrix],
    num_nodes: int,
    d: int,
    cfg: IFConfig,
    seed: int = 123,
) -> Tuple[IFParameters, Dict[str, float], Dict[str, List[float]]]:
    """
    Train Interaction-Field parameters (diffusion mode) by minimizing:

        loss = NLL + α * S_disc + w_node * node_aux + small_z_reg

    where:
        - NLL is Poisson NLL on risk sets vs observed events
        - S_disc is a discrete action penalty
        - node_aux aligns latent node readout to EMA node incident counts
        - small_z_reg keeps z zero-mean, unit-variance (gentle)

    Parameters
    ----------
    y_bins_csr : list[csr_matrix]
        One CSR of directed edges per time bin.
    num_nodes : int
        N.
    d : int
        Latent field dimension.
    cfg : IFConfig
    seed : int

    Returns
    -------
    (theta, metrics, hist)
    """
    torch.manual_seed(seed)
    random.seed(seed)

    device = cfg.device if isinstance(cfg.device, torch.device) else torch.device(cfg.device)
    attention = (cfg.mode == "attention")
    theta = IFParameters(N=num_nodes, d=d, config=cfg, attention=attention).to(device)

    # ----- node targets (EMA of in/out/sum counts) -----
    T = len(y_bins_csr)
    N = num_nodes
    in_counts = np.zeros((T, N), dtype=np.float32)
    out_counts = np.zeros((T, N), dtype=np.float32)

    for t, A in enumerate(y_bins_csr):
        rows, cols = A.nonzero()
        if rows.size:
            np.add.at(out_counts[t], rows, 1.0)
            np.add.at(in_counts[t], cols, 1.0)

    if cfg.node_target_mode == "in":
        raw_node = in_counts
    elif cfg.node_target_mode == "out":
        raw_node = out_counts
    else:
        raw_node = in_counts + out_counts

    alpha = float(cfg.node_ema_alpha)
    X_target = np.zeros_like(raw_node)
    for n in range(N):
        ema = 0.0
        for t in range(T):
            ema = alpha * raw_node[t, n] + (1.0 - alpha) * ema
            X_target[t, n] = ema

    X_target_t = torch.tensor(X_target, dtype=torch.float32, device=device)
    mu = X_target_t.mean(dim=0, keepdim=True)
    sig = X_target_t.std(dim=0, keepdim=True).clamp_min(1e-6)
    X_target_z = (X_target_t - mu) / sig

    theta.register_buffer("readout_mu", mu.squeeze(0))
    theta.register_buffer("readout_sig", sig.squeeze(0))

    # ----- normalize bins to (2,M) tensors -----
    y_bins = csr_bins_to_uv_tensors(y_bins_csr, device=device)

    # ----- history buffers -----
    hist: Dict[str, List[float]] = {"lr": [], "loss": [], "nll": [], "action": [], "m2": [], "c2": [], "b0": [], "node_aux": []}
    steps_per_epoch = 1

    # ----- optimizer/scheduler -----
    opt, sched, sched_mode = make_optimizer_and_scheduler(theta, cfg, steps_per_epoch)

    # ----- optional node head & z-standardizer -----
    head_params: List[nn.Parameter] = []
    if getattr(cfg, "node_head", "none") == "linear":
        if cfg.node_readout_per_node:
            theta.node_gain = nn.Parameter(torch.ones(num_nodes, device=device))
        else:
            theta.node_gain = nn.Parameter(torch.tensor(1.0, device=device))
        theta.node_bias = nn.Parameter(torch.zeros(num_nodes, device=device)) if cfg.node_readout_bias else None
        head_params = [theta.node_gain] + ([theta.node_bias] if theta.node_bias is not None else [])

    theta.z_shift = nn.Parameter(torch.zeros(num_nodes, device=device))  # subtract
    theta.z_scale = nn.Parameter(torch.ones(num_nodes, device=device))   # divide via softplus
    std_params = [theta.z_shift, theta.z_scale]

    # add param groups after base creation
    base_lr = opt.param_groups[0]["lr"]
    if head_params:
        opt.add_param_group({"params": head_params, "lr": base_lr * 2.0, "weight_decay": 0.0})
    opt.add_param_group({"params": std_params, "lr": base_lr * 5.0, "weight_decay": 0.0})

    # ----- scalars for auto-weighting -----
    ema_beta = 0.9
    nll_log_ema = torch.tensor(0.0, device=device)
    aux_log_ema = torch.tensor(0.0, device=device)
    target_frac = getattr(cfg, "node_loss_target_frac", 0.4)
    w_min, w_max = 5e-3, 5.0

    # AMP
    use_amp = (device.type == "cuda")
    scaler = _make_grad_scaler(device.type, enabled=use_amp)
    torch.set_float32_matmul_precision("high")

    best_nll = float("inf")
    no_improve = 0

    bar_epochs = trange(cfg.epochs, desc="epochs", disable=not cfg.show_progress, position=0, leave=True, dynamic_ncols=True)

    for epoch in bar_epochs:
        opt.zero_grad(set_to_none=True)

        # ---- rollout (driven) ----
        H, J_list, K_list, risk_sets, lambda_list, _ = forward_rollout_if(
            y_bins, theta, cfg,
            free_start=None,
            free_mode="driven",
            progress_desc=f"bins (ep {epoch + 1})",
            progress_position=1,
            enable_progress=False,
        )

        # ---- reduce to nodes; build head prediction ----
        X_nodes = reduce_H_to_nodes(H, num_nodes, device, drop_first_if_vector=True, reduce_mode="sym",
                                    channel_reduce="mean", center=False)
        x_hat = X_nodes
        if theta.node_gain is not None:
            gain = theta.node_gain.view(1, -1) if cfg.node_readout_per_node else theta.node_gain
            x_hat = x_hat * gain
        if theta.node_bias is not None:
            x_hat = x_hat + theta.node_bias.view(1, -1)

        # Bootstrap z standardizer on first epoch
        if epoch == 0 and not getattr(theta, "_z_bootstrap_done", False):
            with torch.no_grad():
                mu0 = x_hat.mean(dim=0)
                sd0 = x_hat.std(dim=0)
                small = sd0 < 1e-6
                proxy = x_hat.abs().mean(dim=0).clamp_min(1.0)
                sd0 = torch.where(small, proxy, sd0)
                theta.z_shift.copy_(mu0)
                theta.z_scale.copy_(inv_softplus(sd0))
                _ = 1e-6 + F.softplus(theta.z_scale)  # sanity
                theta._z_bootstrap_done = True  # type: ignore[attr-defined]

        den = 1e-6 + F.softplus(theta.z_scale).view(1, -1)
        z_pred = (x_hat - theta.z_shift.view(1, -1)) / den
        if epoch < getattr(cfg, "z_tanh_epochs", 0):
            z_pred = torch.tanh(z_pred)

        # ---- auxiliary node loss (Huber in z-space) ----
        def huber(x: Tensor, delta: float = 1.0) -> Tensor:
            ax = x.abs()
            return torch.where(ax <= delta, 0.5 * ax * ax, delta * (ax - 0.5 * delta))

        T_run = min(z_pred.shape[0], X_target_z.shape[0])
        z_err = z_pred[:T_run] - X_target_z[:T_run]
        node_aux = huber(z_err, delta=getattr(cfg, "node_huber_delta", 1.0)).mean()

        # ---- discrete action penalty (compact form) ----
        # kinetic
        S_kin = torch.zeros((), device=device)
        S_quad = torch.zeros((), device=device)
        S_quart = torch.zeros((), device=device)
        for b in range(len(K_list)):
            h_b, h_nb = H[b], H[b + 1]
            S_kin = S_kin + (1.0 / (2.0 * cfg.dt)) * ((h_nb - h_b) ** 2).sum()

            Kb = K_list[b]
            if Kb.layout == torch.sparse_csr:
                Sh = Kb @ h_b
                S_coo = Kb.to_sparse_coo().coalesce()
            else:
                S_coo = Kb.coalesce()
                Sh = S_coo @ h_b

            St = torch.sparse_coo_tensor(S_coo.indices().flip(0), S_coo.values(), S_coo.size(), device=device).coalesce()
            St_h = St @ h_b
            Ksym_h = h_b - 0.5 * (Sh + St_h)
            quad_vec = theta.m2 * h_b + theta.c2 * Ksym_h
            S_quad = S_quad + (cfg.dt / 2.0) * (h_b * quad_vec).sum()

            if cfg.lambda_eff and cfg.lambda_eff != 0.0:
                S_quart = S_quart + cfg.dt * (cfg.lambda_eff / 24.0) * (h_b ** 4).sum()

        scale = 1.0 / (max(1, len(K_list)) * N * d)
        S_disc = (S_kin + S_quad + S_quart) * scale

        # ---- NLL ----
        nll = discrete_poisson_nll_risk(lambda_list, y_bins, risk_sets, cfg.dt, N=num_nodes, device=device)

        # ---- auto weight node aux ----
        with torch.no_grad():
            eps = 1e-12
            nll_log_ema.mul_(ema_beta).add_((1 - ema_beta) * torch.log(nll + eps))
            aux_log_ema.mul_(ema_beta).add_((1 - ema_beta) * torch.log(node_aux + eps))
            log_w = torch.log(torch.tensor(target_frac, device=device)) + (nll_log_ema - aux_log_ema)
            node_w_auto = torch.exp(log_w).clamp_(w_min, w_max)

        warm = min(1.0, (epoch + 1) / max(1, cfg.node_loss_warmup_epochs))

        # small z-distribution regularizer
        Z = z_pred[:T_run]
        z_mean = Z.mean(dim=0, keepdim=True)
        z_var = (Z - z_mean).pow(2).mean(dim=0, keepdim=True)
        mean_pen = (z_mean.pow(2)).mean()
        var_pen = (z_var - 1.0).pow(2).mean()
        beta0, betaT, Ttot = 1e-2, 1e-4, max(1, cfg.epochs)
        decay = betaT + (beta0 - betaT) * max(0.0, 1.0 - epoch / max(1, min(Ttot, 10)))

        loss = nll + cfg.alpha * S_disc + (warm * node_w_auto) * node_aux + decay * (mean_pen + var_pen)

        # ---- backward ----
        did_step = True
        if use_amp:
            prev_scale = scaler.get_scale()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(list(theta.parameters()), max_norm=5.0)
            scaler.step(opt)
            scaler.update()
            did_step = scaler.get_scale() >= prev_scale
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(theta.parameters()), max_norm=5.0)
            opt.step()

        if did_step and (sched is not None) and (sched_mode == "epoch"):
            sched.step()

        # ---- logging ----
        curr_lr = opt.param_groups[0]["lr"]
        hist["lr"].append(curr_lr)
        hist["loss"].append(float(loss.detach().cpu()))
        hist["nll"].append(float(nll.detach().cpu()))
        hist["action"].append(float(S_disc.detach().cpu()))
        hist["m2"].append(float(theta.m2.detach().cpu()))
        hist["c2"].append(float(theta.c2.detach().cpu()))
        hist["b0"].append(float(theta.b0.detach().cpu()))
        hist["node_aux"].append(float(node_aux.detach().cpu()))

        if cfg.show_progress:
            bar_epochs.set_postfix_str(f"lr={curr_lr:.4g} loss={hist['loss'][-1]:.4f} NLL={hist['nll'][-1]:.4f} S={hist['action'][-1]:.4f}")

        # ---- early stopping on NLL ----
        curr_nll = hist["nll"][-1]
        if not math.isfinite(best_nll):
            best_nll, no_improve = curr_nll, 0
        else:
            rel_improv = (best_nll - curr_nll) / max(best_nll, 1e-12)
            if rel_improv > cfg.early_stop_min_rel_improv:
                best_nll, no_improve = curr_nll, 0
            else:
                no_improve += 1
            if no_improve >= cfg.early_stop_patience:
                print(
                    f"Early stopping at epoch {epoch + 1}: "
                    f"no >{100 * cfg.early_stop_min_rel_improv:.1f}% NLL improvement for "
                    f"{cfg.early_stop_patience} epochs."
                )
                break

    # ----- summary metrics -----
    metrics: Dict[str, float] = {
        "final_loss": hist["loss"][-1],
        "final_nll": hist["nll"][-1],
        "final_action": hist["action"][-1],
        "final_m2": float(theta.m2.detach().cpu()),
        "final_c2": float(theta.c2.detach().cpu()),
        "final_b0": float(theta.b0.detach().cpu()),
    }
    if attention and hasattr(theta, "beta"):
        metrics["beta"] = float(theta.beta.detach().cpu())

    return theta, metrics, hist
