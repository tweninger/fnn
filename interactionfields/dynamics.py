import numpy as np
import scipy.sparse as sp
from typing import Iterable, Tuple, Optional, Literal, List

def _graph_laplacian_from_csr(A):
    import scipy.sparse as sp
    A = A.tocsr().astype(np.float32)
    A.sum_duplicates()
    A.eliminate_zeros()
    deg = np.asarray(A.sum(axis=1)).ravel().astype(np.float32)
    D = sp.diags(deg, format="csr")
    return (D - A).tocsr()

def _gaussian_pulse(t: np.ndarray, period: float, width: float, phase: float = 0.0) -> np.ndarray:
    x = ((t - phase + 0.5 * period) % period) - 0.5 * period
    return np.exp(-(x**2) / (2.0 * width**2))

def simulate_dripping_wave(
    A_csr,
    *,
    T: int = 1000,
    dt: float = 0.01,
    c: float = 1.0,
    gamma: float = 0.02,
    faucet_nodes: Iterable[int] = (0,),
    faucet_period: float = 1.0,
    faucet_amp: float = 1.0,
    faucet_width: float = 0.05,
    source_kind: Literal["impulse", "gaussian"] = "gaussian",
    h0: Optional[np.ndarray] = None,
    v0: Optional[np.ndarray] = None,
    edge_activation: Literal["absdiff", "flux"] = "absdiff",
    edge_threshold: float = 0.0,  # drop small edge activations when forming each CSR
    keep_pattern: Literal["as_is","symmetrize"] = "as_is",  # optionally reflect to (v,u)
) -> Tuple[np.ndarray, List[sp.csr_matrix]]:
    """
    Damped wave on a graph: h_tt + gamma h_t + c^2 L h = s(t)

    Returns
    -------
    H : (T, N) float32
        Node displacements over time.
    E_list : list of length T
        Each item is a CSR matrix of shape (N, N) with the *adjacency pattern* of A.
        Entry (u, v) holds the edge-activation magnitude at time t for that directed edge.
        If keep_pattern="symmetrize", we mirror values to (v, u) as well.
    """

    A = A_csr.tocsr().astype(np.float32)
    A.sum_duplicates(); A.eliminate_zeros()
    N = A.shape[0]

    # Laplacian for dynamics
    L = _graph_laplacian_from_csr(A)

    # Pre-extract the (row, col) structure we’ll reuse for every time step
    Acoo = A.tocoo()
    r = Acoo.row.astype(np.int64)
    c_idx = Acoo.col.astype(np.int64)
    nnz = r.size

    # Initial conditions
    h = np.zeros(N, dtype=np.float32) if h0 is None else np.asarray(h0, dtype=np.float32).copy()
    v = np.zeros(N, dtype=np.float32) if v0 is None else np.asarray(v0, dtype=np.float32).copy()

    # Time grid & source
    tgrid = np.arange(T, dtype=np.float32) * dt
    if source_kind == "gaussian":
        src_profile = faucet_amp * _gaussian_pulse(tgrid, period=faucet_period, width=faucet_width)
    elif source_kind == "impulse":
        kperiod = max(1, int(round(faucet_period / dt)))
        src_profile = np.zeros(T, dtype=np.float32); src_profile[::kperiod] = faucet_amp
    else:
        raise ValueError("source_kind must be 'gaussian' or 'impulse'.")

    faucet_nodes = np.fromiter(faucet_nodes, dtype=np.int64)
    faucet_nodes = faucet_nodes[(faucet_nodes >= 0) & (faucet_nodes < N)]
    if faucet_nodes.size == 0:
        raise ValueError("No valid faucet_nodes given for this graph.")

    b = np.zeros(N, dtype=np.float32)
    b[faucet_nodes] = 1.0 / max(1, faucet_nodes.size)

    # Storage
    H = np.empty((T, N), dtype=np.float32)
    E_list: List[sp.csr_matrix] = []

    # Leapfrog-like integration constants
    damp_fac = (1.0 - 0.5 * gamma * dt) / (1.0 + 0.5 * gamma * dt)
    acc_fac  = dt / (1.0 + 0.5 * gamma * dt)

    # Start v at half-step: v_{-1/2} = v0 - 0.5*dt*a0
    a0 = - (c * c) * (L @ h) + src_profile[0] * b
    v_half = v - 0.5 * dt * a0

    for t in range(T):
        a = - (c * c) * (L @ h) + src_profile[t] * b
        v_half = damp_fac * v_half + acc_fac * a
        h = h + dt * v_half

        H[t, :] = h

        # Edge activation values on the fixed (r, c_idx) pattern
        if edge_activation == "absdiff":
            vals = np.abs(h[r] - h[c_idx])
        elif edge_activation == "flux":
            vals = np.abs((h[r] - h[c_idx]) / dt)
        else:
            raise ValueError("edge_activation must be 'absdiff' or 'flux'.")

        if edge_threshold > 0.0:
            mask = vals > edge_threshold
            rr = r[mask]; cc = c_idx[mask]; dd = vals[mask].astype(np.float32, copy=False)
        else:
            rr = r; cc = c_idx; dd = vals.astype(np.float32, copy=False)

        if keep_pattern == "symmetrize":
            # mirror into (v,u); if A already has both directions, this just duplicates which CSR will sum.
            rr = np.concatenate([rr, cc])
            cc = np.concatenate([cc, rr[:len(dd)]])  # use original rr slice
            dd = np.concatenate([dd, dd])

        # Build one CSR for this timestep with the same (sparse) pattern
        E_t = sp.csr_matrix((dd, (rr, cc)), shape=(N, N))
        E_t.sum_duplicates(); E_t.eliminate_zeros()
        E_list.append(E_t)

    return H, E_list
