from __future__ import annotations

# ── Headless backend (configure before pyplot) ─────────────────────────────────
import os

from interactionfields.graphs import edges_from_adj

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)

# ── Stdlib / typing ───────────────────────────────────────────────────────────
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any, Literal

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm, Normalize, ListedColormap
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from matplotlib.animation import FuncAnimation
from scipy.sparse import csr_matrix

# =============================================================================
#                               COLOR HELPERS
# =============================================================================

def _compress_midpoint(x: np.ndarray, power: float = 0.5) -> np.ndarray:
    """Symmetric compression of [-1,1] that steepens slope near 0."""
    return np.sign(x) * (np.abs(x) ** power)

def make_seismic_with_white_mid(power: float = 0.5, *, base: str = "seismic") -> ListedColormap:
    """
    Diverging colormap with a brighter mid-band by nonlinearly compressing values
    around 0 before mapping through the base colormap.
    """
    base_cmap = plt.get_cmap(base)
    vals = np.linspace(-1, 1, 256)
    colors = base_cmap((_compress_midpoint(vals, power=power) + 1) / 2)
    return ListedColormap(colors)

# Default node colormap: “seismic” with a whiter mid band
DEFAULT_NODE_CMAP: ListedColormap = make_seismic_with_white_mid(power=0.5)

def _squash_white(vals01: np.ndarray, cut: Tuple[float, float]) -> np.ndarray:
    """
    Narrow the neutral (white) region in a diverging map by compressing an
    interval [cut[0], cut[1]] toward its left endpoint. Returns values in [0,1].
    """
    a, b = cut
    y = vals01.copy()
    mask = (y >= a) & (y <= b)
    y[mask] = a + (y[mask] - a) * 0.25
    return np.clip(y, 0.0, 1.0)

# =============================================================================
#                         ARRAY SHAPE / NORMALIZATION
# =============================================================================

def _ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def _as_TMN(X: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Ensure X is shaped (T, m, n). Accepts (T,N) or (T,m,n)."""
    X = np.asarray(X)
    if X.ndim == 3:
        return X
    if X.ndim == 2:
        m, n = shape
        return X.reshape(X.shape[0], m, n)
    raise ValueError(f"X must be (T,N) or (T,m,n), got {X.shape}")

def _as_TN(X: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Ensure X is shaped (T, N). Accepts (T,N) or (T,m,n)."""
    X = np.asarray(X)
    if X.ndim == 2:
        return X
    if X.ndim == 3:
        return X.reshape(X.shape[0], -1)
    raise ValueError(f"X must be (T,N) or (T,m,n), got {X.shape}")

def _frame_2d(X: np.ndarray, t: int, shape: Tuple[int, int] | None) -> np.ndarray:
    """Return X[t] as (m,n). Accepts X as (T,N) or (T,m,n)."""
    xt = np.asarray(X)[t]
    if xt.ndim == 2:
        return xt
    if xt.ndim == 1:
        if shape is None:
            side = int(np.sqrt(xt.size))
            shape = (side, side)
        return xt.reshape(shape)
    raise ValueError(f"Bad frame shape at t={t}: {xt.shape}")

def _global_minmax(
    X: np.ndarray,
    shape: Tuple[int, int] | None = None,
    *,
    center_zero: bool = True,
    robust: bool = False,
    lo_hi: Tuple[float, float] | None = None
) -> Tuple[float, float, Normalize]:
    """
    Compute global vmin/vmax and a Normalize object.
      - If center_zero, returns a TwoSlopeNorm centered at 0 with symmetric range.
      - If robust, uses percentiles (1,99) for stability.
      - lo_hi overrides computed vmin/vmax if provided.
    """
    if lo_hi is not None:
        vmin, vmax = lo_hi
    else:
        Xall = _as_TMN(X, shape) if (shape is not None and np.asarray(X).ndim != 3) else np.asarray(X)
        if robust:
            vmin = float(np.nanpercentile(Xall, 1))
            vmax = float(np.nanpercentile(Xall, 99))
        else:
            vmin = float(np.nanmin(Xall))
            vmax = float(np.nanmax(Xall))
        if center_zero:
            vmax_abs = max(abs(vmin), abs(vmax), 1e-12)
            vmin, vmax = -vmax_abs, +vmax_abs

    if center_zero:
        norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=(vmax if vmax > 0 else 1.0))
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
    return vmin, vmax, norm

def _norm01_percentile(x: np.ndarray, lo_p=1, hi_p=99, eps=1e-12) -> np.ndarray:
    """Robust [0,1] scaling using percentiles; returns zeros if degenerate."""
    x = np.asarray(x)
    if not np.any(np.isfinite(x)):
        return np.zeros_like(x, dtype=float)
    lo, hi = np.percentile(x.astype(float), [lo_p, hi_p])
    if hi <= lo:
        lo, hi = np.nanmin(x), np.nanmax(x)
        if not np.isfinite(hi - lo) or (hi - lo) <= eps:
            return np.zeros_like(x, dtype=float)
    return np.clip((x - lo) / (hi - lo + eps), 0.0, 1.0)

# =============================================================================
#                         GRID GEOMETRY & EDGE HELPERS
# =============================================================================

def _idx(i: int, j: int, w: int) -> int:
    return i * w + j

def grid_node_xyz(h: int, w: int) -> np.ndarray:
    """Return base (x,y,z=0) coordinates for all nodes on an h×w grid."""
    jj, ii = np.meshgrid(np.arange(w), np.arange(h))
    return np.stack([jj.ravel(), ii.ravel(), np.zeros(h * w, dtype=float)], axis=1)  # (N,3)

def grid_edges(h: int, w: int) -> np.ndarray:
    """Return undirected 4-neighbor edge list as pairs of node indices (M,2)."""
    edges = []
    for i in range(h):
        for j in range(w):
            u = _idx(i, j, w)
            if j + 1 < w:
                edges.append((u, _idx(i, j + 1, w)))  # right
            if i + 1 < h:
                edges.append((u, _idx(i + 1, j, w)))  # down
    return np.asarray(edges, dtype=int)

def _coo_from_csr(M: csr_matrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(row, col, data) from CSR; keeps duplicates/weights if present."""
    C = M.tocoo(copy=False)
    return C.row, C.col, C.data

def build_edge_segments(xyz: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return (M,2,3) segments array from xyz and edge index pairs."""
    return np.stack([xyz[edges[:, 0]], xyz[edges[:, 1]]], axis=1)

def _compute_grid_normals(xyz: np.ndarray, h: int, w: int, *, periodic: bool = True) -> np.ndarray:
    """
    Per-node unit normals for an h×w parametric grid embedded in 3D.
    Assumes xyz is (N,3) in row-major order. Uses forward diffs with wrap.
    """
    xyz3 = xyz.reshape(h, w, 3)
    ip = np.roll(xyz3, -1, axis=0) if periodic else np.pad(xyz3[1:], ((0,1),(0,0),(0,0)), mode="edge")
    jp = np.roll(xyz3, -1, axis=1) if periodic else np.pad(xyz3[:,1:], ((0,0),(0,1),(0,0)), mode="edge")
    vi, vj = (ip - xyz3), (jp - xyz3)
    n = np.cross(vj, vi)   # right-hand (vj × vi)
    n /= (np.linalg.norm(n, axis=2, keepdims=True) + 1e-12)
    return n.reshape(h*w, 3)

def edge_activations_for_frame(
    event_bins: List[csr_matrix],
    t: int,
    M: int,
    edges: np.ndarray
) -> np.ndarray:
    """
    Map sparse activations to the fixed undirected edge list (sum u↔v if directed).
    """
    rows, cols, data = _coo_from_csr(event_bins[t])
    wmap: Dict[Tuple[int, int], float] = {}
    for r, c, v in zip(rows, cols, data):
        a, b = (r, c) if r <= c else (c, r)
        wmap[(a, b)] = wmap.get((a, b), 0.0) + float(v)

    w = np.zeros(M, dtype=float)
    for k, (u, v) in enumerate(edges):
        a, b = (u, v) if u <= v else (v, u)
        w[k] = wmap.get((a, b), 0.0)
    return w

def edge_strength_and_recency(
    event_bins: List[csr_matrix],
    t: int,
    edges: np.ndarray,
    *,
    k: int = 8,
    gamma: float = 0.6,
    tau_decay: float = 5.0,
    lookback: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each edge at time t, returns:
      strength = sum_{i=0..k-1} gamma^i * w_{t-i}(e)  (geometric window)
      recency  = exp(-lag / tau_decay), lag = steps since last activation in lookback
    """
    M = edges.shape[0]
    if lookback is None:
        lookback = max(k, int(3 * tau_decay))

    L = min(lookback, t + 1)
    if L <= 0:
        return np.zeros(M, float), np.zeros(M, float)

    T_bins = len(event_bins)  # <-- NEW

    def _edge_w(at: int) -> np.ndarray:
        # guard both negative and beyond-the-end indices
        if at < 0 or at >= T_bins:
            return np.zeros(M, float)
        return edge_activations_for_frame(event_bins, at, M, edges)

    history = np.stack([_edge_w(t - lag) for lag in range(L)], axis=0)  # (L,M)

    # Strength over last k frames with geometric weights
    use = min(k, L)
    if use > 0:
        ws = np.array([gamma ** i for i in range(use)], dtype=float)
        ws /= (ws.sum() + 1e-12)
        strength = (ws[:, None] * history[:use]).sum(axis=0)
    else:
        strength = np.zeros(M, float)

    # Recency from most-recent nonzero in lookback (0 means “just now”)
    mask = (history > 0)
    has_any = mask.any(axis=0)
    most_recent_lag = np.argmax(mask, axis=0)
    tau = float(max(tau_decay, 1e-12))
    recency = np.zeros(M, float)
    recency[has_any] = np.exp(-most_recent_lag[has_any] / tau)

    return strength, recency

# =============================================================================
#                                  2D HEATMAP
# =============================================================================

def plot_png(
    X: np.ndarray,
    t: int,
    *,
    shape: Tuple[int, int] | None = None,
    outfile: str = "figs/wave.png",
    cmap: ListedColormap | str = DEFAULT_NODE_CMAP
) -> str:
    """Save a 2D heatmap of X[t]. X is (T,N) or (T,m,n). Diverging scale centered at 0."""
    out = Path(outfile).with_suffix(".png")
    _ensure_dir(out.parent)
    x_t = _frame_2d(X, t, shape)
    vmax = float(np.abs(x_t).max()) or 1.0

    fig, ax = plt.subplots(figsize=(6, 5), dpi=600)
    im = ax.imshow(x_t, cmap=cmap, vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax)
    ax.set_title(f"t = {t}")
    fig.tight_layout()
    fig.savefig(out, dpi=600)
    plt.close(fig)
    print(f"Saved {out}")
    return str(out)

# =============================================================================
#                           SURFACE SNAPSHOT (3D)
# =============================================================================

def _upsample_lanczos_scalar(x: np.ndarray, *, h: int, w: int, shape: Tuple[int, int] | None = None) -> np.ndarray:
    """
    Upscale a 2D scalar field to (h,w) with Lanczos (Pillow).
    Accepts (m,n) or flat (N,) with optional `shape`.
    """
    if x.ndim == 1:
        if shape is None:
            side = int(np.sqrt(x.size))
            shape = (side, side)
        x2 = x.reshape(shape)
    elif x.ndim == 2:
        x2 = x
    else:
        raise ValueError(f"Expected 1D or 2D scalar field, got {x.shape}")

    try:
        from PIL import Image
    except Exception as e:
        raise RuntimeError("Install Pillow for Lanczos upsampling: pip install pillow") from e

    im = Image.fromarray(x2.astype(np.float32), mode="F")
    im_hi = im.resize((w, h), resample=Image.LANCZOS)
    return np.asarray(im_hi, dtype=np.float32)

def plot_surface_snapshot(
    X: np.ndarray,
    shape: Tuple[int, int],
    t: int,
    outpath: str,
    *,
    upsample_to: Tuple[int, int] = (180, 180),
    cmap: ListedColormap | str = DEFAULT_NODE_CMAP,
    center_zero: bool = True,
    global_norm: bool = True,
    elev: float = 35.0,
    azim: float = 45.0,
) -> str:
    """Render a single 3D surface snapshot: height = upscaled field value."""
    out = Path(outpath).with_suffix(".png")
    _ensure_dir(out.parent)

    # Select and upsample frame
    x_frame = _frame_2d(X, t, shape)
    h, w = upsample_to
    Z = _upsample_lanczos_scalar(x_frame, h=h, w=w, shape=shape)

    # Norm (global vs per-frame)
    _, _, norm = (_global_minmax(X, shape=shape, center_zero=center_zero)
                  if global_norm else _global_minmax(Z, center_zero=center_zero))

    # Axes mesh
    Y, Xg = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    # Plot
    fig = plt.figure(figsize=(9, 5.2), dpi=600)
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        Xg, Y, Z, rstride=1, cstride=1,
        cmap=cmap, norm=norm, linewidth=0, antialiased=True, shade=True
    )

    ax.view_init(elev=elev, azim=azim)
    vmax = float(np.max(np.abs(Z))) or 1.0
    ax.set_zlim(-2 * vmax, 2 * vmax)
    ax.set_axis_off()
    ax.set_title(f"t = {t}")

    # Sparse wireframe for depth cues
    ax.plot_wireframe(Xg, Y, Z, rstride=20, cstride=20, color="0.5", alpha=0.6, linewidth=0.5)

    # Shared colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.03, fraction=0.055, label="wave")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return str(out)

# =============================================================================
#                          ROW-MEAN HOVMÖLLER DIAGRAM
# =============================================================================

def _dominant_value(a: np.ndarray, bins: int = 256) -> float:
    """Histogram mode (bin center) for continuous data; 0.0 if empty."""
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0
    hist, edges = np.histogram(a, bins=bins)
    i = int(hist.argmax())
    return 0.5 * (edges[i] + edges[i + 1])

def compute_rowmean_array(X: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Return M(i,t) = mean over row i at time t as array (m, T)."""
    Xr = _as_TMN(X, shape)
    return Xr.mean(axis=2).T  # (m, T)

def plot_rowmean_hovmoller(
    X: np.ndarray,
    vmin: float,
    vmax: float,
    shape: Tuple[int, int],
    outpath: str = "figs/rowmean_hovmoller.png",
    *,
    htrain: int | None = None,
    cmap: str = "seismic",
    dpi: int = 600,
    scale_m: int = 180,
) -> str:
    """
    Hovmöller diagram (time × row index). Uses TwoSlopeNorm centered at an
    early-time mode for stability.
    """
    out = Path(outpath).with_suffix(".png")
    _ensure_dir(out.parent)

    M = compute_rowmean_array(X, shape)  # (m, T)
    m, T = M.shape

    # Optional vertical upscaling for smoothness
    if scale_m and scale_m != m:
        try:
            from PIL import Image
            im = Image.fromarray(M.astype(np.float32), mode="F")
            M = np.asarray(im.resize((T, scale_m), resample=Image.BICUBIC), dtype=np.float32)
        except Exception:
            pass  # gracefully skip if Pillow isn't installed

    # Mode over first few steps to anchor the center
    T0 = min(10, T)
    mode_val = _dominant_value(M[:, :T0])
    eps = 1e-6
    vmin_adj = min(vmin, mode_val - eps)
    vmax_adj = max(vmax, mode_val + eps)
    norm = TwoSlopeNorm(vmin=vmin_adj, vcenter=mode_val, vmax=vmax_adj)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
    im = ax.imshow(
        M, origin="lower", aspect="auto", cmap=cmap, norm=norm,
        extent=(0, T - 1, 0, M.shape[0] - 1), interpolation="nearest",
    )
    im.set_rasterized(True)
    fig.colorbar(im, ax=ax, pad=0.01, label="Row-mean value")
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Row index $i$")
    ax.set_title("Row-mean Hovmöller (time × row)")
    if htrain is not None:
        ax.axvline(htrain, color="k", lw=2, ls="--", alpha=0.85)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return str(out)

# =============================================================================
#                            3D NODE/EDGE RENDERING
# =============================================================================

def _wall_polys(
    *,
    h: int, w: int, zmin: float, zmax: float,
    axis: str, index: int, gate_start: int, gate_end: int,
    thickness: float = 0.15, x_offset: float = 0.0, y_offset: float = 0.0,
) -> list[list[tuple[float, float, float]]]:
    """Construct vertical/horizontal wall slabs (split around z=0)."""
    polys: list[list[tuple[float, float, float]]] = []
    t = max(thickness, 1e-3)
    z0, z1 = float(zmin), float(zmax)

    def add_rect(x0, x1, y0, y1):
        x0 += x_offset; x1 += x_offset
        y0 += y_offset; y1 += y_offset
        # top/bottom faces
        polys.append([(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)])
        polys.append([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)])
        # sides
        polys.append([(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)])
        polys.append([(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)])
        polys.append([(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)])
        polys.append([(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)])

    if axis == "vertical":
        x0, x1 = index - t/2, index + t/2
        if gate_start > 0:
            add_rect(x0, x1, 0, max(0, min(h, gate_start)))
        if gate_end < h:
            add_rect(x0, x1, max(0, min(h, gate_end)), h)
    elif axis == "horizontal":
        y0, y1 = index - t/2, index + t/2
        if gate_start > 0:
            add_rect(0, max(0, min(w, gate_start)), y0, y1)
        if gate_end < w:
            add_rect(max(0, min(w, gate_end)), w, y0, y1)
    else:
        raise ValueError("axis must be 'vertical' or 'horizontal'")

    return polys

def _add_poly3d(ax, polys, *, color, alpha, sort_hint=None, lw=0.5, zorder=None):
    coll = Poly3DCollection(polys, facecolors=color, edgecolors=color, linewidths=lw, alpha=alpha)
    if sort_hint is not None and hasattr(coll, "set_sort_zpos"):
        try:
            coll.set_sort_zpos(float(sort_hint))
        except Exception:
            pass
    if zorder is not None:
        try:
            coll.set_zorder(zorder)
        except Exception:
            pass
    ax.add_collection3d(coll)
    return coll

def _merge_kwargs(base: dict | None, override: dict | None) -> dict:
    """Copy-merge dictionaries (override wins)."""
    out = dict(base or {})
    if override:
        out.update(override)
    return out

# ---- small local helper for edge visuals ---------------------------------
def _compute_edge_style(
    *,
    mode: EdgeMode,
    event_bins: List[csr_matrix] | None,
    t: int,
    E_base: np.ndarray,
    N_nodes: int,
    decay_k: int,
    decay_gamma: float,
    tau_decay: float,
    lookback: int | None,
    edge_ref_bins: List[csr_matrix] | None,
    edge_pred_bins: List[csr_matrix] | None,
    lw_min: float, lw_max: float,
    cmap_edges_activity: str,
    cmap_edges_residual: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (E_draw, linewidths, rgba). Each can be empty if nothing to draw.
    """
    if (mode == "none") or (event_bins is None):
        return (np.zeros((0, 2), dtype=int),
                np.zeros((0,), dtype=float),
                np.zeros((0, 4), dtype=float))

    E = np.asarray(E_base, int)

    if mode == "activity":
        strength, recency = edge_strength_and_recency(
            event_bins, t+1, E,
            k=decay_k, gamma=decay_gamma, tau_decay=tau_decay, lookback=lookback
        )
        s01 = _norm01_percentile(strength, 1, 99)
        r01 = np.clip(recency, 0.3, 1.0)
        lws = lw_min + (lw_max - lw_min) * s01
        rgba = plt.get_cmap(cmap_edges_activity)(r01)
        rgba[:, 3] = 1.0
        return E, lws, rgba

    if mode == "residual":
        if edge_ref_bins is None:
            return (np.zeros((0, 2), dtype=int),
                    np.zeros((0,), dtype=float),
                    np.zeros((0, 4), dtype=float))
        s_pred, _ = edge_strength_and_recency(
            event_bins, t+1, E, k=decay_k, gamma=decay_gamma, tau_decay=tau_decay, lookback=lookback
        )
        s_true, _ = edge_strength_and_recency(
            edge_ref_bins, t, E, k=decay_k, gamma=decay_gamma, tau_decay=tau_decay, lookback=lookback
        )
        delta = s_pred - s_true
        amp = np.abs(delta)
        if amp.size == 0 or not np.any(np.isfinite(amp)):
            return (np.zeros((0, 2), dtype=int),
                    np.zeros((0,), dtype=float),
                    np.zeros((0, 4), dtype=float))

        # sparsify to top decile by amplitude
        q = np.percentile(amp, 90.0)
        keep = amp >= q
        if not keep.any():
            return (np.zeros((0, 2), dtype=int),
                    np.zeros((0,), dtype=float),
                    np.zeros((0, 4), dtype=float))

        E_k = E[keep]
        a01 = _norm01_percentile(amp[keep], 1, 99)
        lws = lw_min + (lw_max - lw_min) * a01

        vmax = np.percentile(np.abs(delta[keep]), 99) if delta[keep].size else 1.0
        vmax = vmax if np.isfinite(vmax) and vmax > 1e-12 else 1.0
        c01 = 0.5 * (np.clip(delta[keep] / vmax, -1.0, 1.0) + 1.0)
        rgba = plt.get_cmap(cmap_edges_residual)(c01)
        rgba[:, 3] = 1.0
        return E_k, lws, rgba

    if mode == "confusion":
        if edge_ref_bins is None:
            return (np.zeros((0, 2), dtype=int),
                    np.zeros((0,), dtype=float),
                    np.zeros((0, 4), dtype=float))

        # allow off-grid edges by union over a window
        E_conf = E
        if edge_pred_bins is not None:
            E_pred_dyn = _edges_from_bins_window(edge_pred_bins, t, lookback, N_nodes)
            E_true_dyn = _edges_from_bins_window(edge_ref_bins, t, lookback, N_nodes)
            if E_pred_dyn.size or E_true_dyn.size:
                E_conf = _unique_rows(np.vstack([E, E_pred_dyn, E_true_dyn]))

        s_pred, _ = edge_strength_and_recency(
            edge_pred_bins if edge_pred_bins is not None else edge_ref_bins,
            t+1, E_conf, k=decay_k, gamma=decay_gamma, tau_decay=tau_decay, lookback=lookback
        )
        s_true, _ = edge_strength_and_recency(
            edge_ref_bins, t, E_conf, k=decay_k, gamma=decay_gamma, tau_decay=tau_decay, lookback=lookback
        )

        eps = 1e-12
        true_pos = s_true > eps
        k_true = int(true_pos.sum())
        order = np.argsort(s_pred)[::-1]
        pred_pos = np.zeros_like(s_pred, bool)
        if k_true > 0: pred_pos[order[:k_true]] = True


        tp = pred_pos & true_pos
        fp = pred_pos & (~true_pos)
        fn = (~pred_pos) & true_pos
        print("TP/FP/FN:", int(tp.sum()), int(fp.sum()), int(fn.sum()))

        keep = tp | fp | fn
        if not np.any(keep):
            return (np.zeros((0, 2), dtype=int),
                    np.zeros((0,), dtype=float),
                    np.zeros((0, 4), dtype=float))

        E_k = E_conf[keep]
        amp = np.zeros_like(s_pred)
        amp[tp] = np.minimum(s_pred[tp], s_true[tp])
        amp[fp] = s_pred[fp]
        amp[fn] = s_true[fn]
        a01 = _norm01_percentile(amp[keep], 1, 99)
        lws = lw_min + (lw_max - lw_min) * a01

        # green / red / blue categories
        cat = np.full_like(s_pred, -1, dtype=int)
        cat[tp] = 0; cat[fp] = 1; cat[fn] = 2
        cat_k = cat[keep]
        base = {
            0: np.array([0.20, 0.90, 0.20]),  # TP
            1: np.array([0.90, 0.10, 0.10]),  # FP
            2: np.array([0.10, 0.30, 0.95])   # FN
        }
        alphas = 0.75 + 0.25 * a01
        rgba = np.ones((cat_k.size, 4), float)
        for i, c in enumerate(cat_k):
            rgba[i, :3] = base.get(int(c), (0.5, 0.5, 0.5))
            rgba[i, 3] = float(alphas[i])
        return E_k, lws, rgba

    raise ValueError(f"Unknown edge mode: {mode}")

EdgeMode = Literal["none", "activity", "residual", "confusion"]


def _force_projection_depth(artist, bias: float):
    """
    Wrap artist.do_3d_projection so it still updates itself, but returns depth+bias.
    Works across Matplotlib versions where some artists take (renderer) and others don't.
    Positive bias -> drawn later (on top).
    """
    if not hasattr(artist, "do_3d_projection"):
        return
    orig = artist.do_3d_projection  # bound method (may be 0-arg or 1-arg)

    def _proj(renderer=None):
        # Call the original with the right signature
        try:
            d = orig()  # try no-arg form first
        except TypeError:
            d = orig(renderer)  # fall back to (renderer) form
        # Bias the sort depth
        try:
            d = 0.0 if d is None else float(d)
        except Exception:
            d = 0.0
        return d + float(bias)

    artist.do_3d_projection = _proj

def _coerce_xyz_base(
    *,
    xyz_base: Optional[np.ndarray],
    N: int,
    h: int,
    w: int,
) -> np.ndarray:
    """
    Return (N,3) base coordinates.
    - If xyz_base is provided:
        * accepts (N,3) or (N,2) and pads z=0.
    - Else uses grid_node_xyz(h,w) and requires N==h*w.
    """
    if xyz_base is None:
        if N != h * w:
            raise ValueError(
                f"No xyz_base provided, so I assumed a grid. "
                f"But N={N} != h*w={h*w}. Pass xyz_base for non-grid graphs."
            )
        return grid_node_xyz(h, w).astype(float)

    X = np.asarray(xyz_base, float)
    if X.ndim != 2 or X.shape[0] != N:
        raise ValueError(f"xyz_base must have shape (N,2) or (N,3). Got {X.shape}, N={N}")

    if X.shape[1] == 3:
        return X
    if X.shape[1] == 2:
        Z = np.zeros((N, 1), dtype=float)
        return np.concatenate([X, Z], axis=1)

    raise ValueError(f"xyz_base must have 2 or 3 columns, got {X.shape[1]}")

def _coerce_edges_base(
    *,
    edges_fixed: Optional[np.ndarray],
    N: int,
    h: int,
    w: int,
) -> np.ndarray:
    """
    Return (M,2) int edges.
    - If edges_fixed is provided, uses it.
    - Else uses grid_edges(h,w) and requires N==h*w.
    """
    if edges_fixed is None:
        if N != h * w:
            raise ValueError(
                f"No edges_fixed provided, so I assumed a grid. "
                f"But N={N} != h*w={h*w}. Pass edges_fixed for non-grid graphs."
            )
        return grid_edges(h, w)

    E = np.asarray(edges_fixed, int)
    if E.ndim != 2 or E.shape[1] != 2:
        raise ValueError(f"edges_fixed must be shape (M,2). Got {E.shape}")
    # basic bounds check
    if E.size and (E.min() < 0 or E.max() >= N):
        raise ValueError(f"edges_fixed has node ids out of bounds for N={N}")
    return E


def draw_frame_3d(
    H: np.ndarray,
    event_bins: List[csr_matrix] | None,
    h: int, w: int, t: int,
    outpath: str,
    *,
    # shared node scale
    vmin: Optional[float],
    vmax: Optional[float],
    # visibility toggles
    show_nodes: bool = True,
    show_edges: bool = True,
    # node styling
    node_cmap: str = "seismic",
    node_white_cut: Tuple[float, float] = (0.45, 0.55),
    node_size: float = 10.0,
    node_alpha: Optional[float] = None,
    # edge styling
    edge_mode: EdgeMode = "activity",
    edge_ref_bins: List[csr_matrix] | None = None,
    edge_pred_bins: List[csr_matrix] | None = None,
    lw_min: float = 0.5,
    lw_max: float = 2.0,
    cmap_edges_activity: str = "Greys",
    cmap_edges_residual: str = "seismic",
    # strength/recency decay
    decay_k: int = 30,
    decay_gamma: float = 0.6,
    tau_decay: float = 5.0,
    lookback: int | None = None,
    # geometry / displacement
    xyz_base: Optional[np.ndarray] = None,      # (N,3)
    edges_fixed: Optional[np.ndarray] = None,   # (M,2)
    displace_mode: str = "z",                   # "z" | "normal"
    normals_periodic: bool = True,
    z_aspect_mode: str = "auto",                # "auto" | "grid"
    min_z_aspect_frac: float = 0.25,
    z_field_mode: str = "data",                 # "data" | "base" | "flat"
    H_base_for_z: Optional[np.ndarray] = None,  # (T,N) if z_field_mode="base"
    z_exaggeration: float = 1.0,
    elev: float = 35,
    azim: float = -60,
    # wall overlay (optional)
    wall: Optional[dict] = None,
) -> str:
    """
    Unified renderer: nodes, edges, or both.

    Make nodes transparent by show_nodes=False (or node_alpha=0).
    Make edges transparent with show_edges=False, or edge_mode="none".

    edge_mode:
      - "none": no edges
      - "activity": width ~ decayed strength; color ~ recency (Greys)
      - "residual": needs edge_ref_bins; colors diverging on strength delta
      - "confusion": needs edge_ref_bins; TP/FP/FN coloring (green/red/blue)
    """

    if t is None:
        # choose output file (gif by default)
        out_anim = Path(outpath)
        if out_anim.suffix.lower() not in {".gif", ".mp4"}:
            out_anim = out_anim.with_suffix(".gif")

        anim = animate_3d(
            H=H,
            event_bins=event_bins,
            h=h, w=w,
            # node scale (match static)
            vmin=vmin if vmin is not None else float(np.nanmin(_as_TN(H, (h, w))) - 1e-6),
            vmax=vmax if vmax is not None else float(np.nanmax(_as_TN(H, (h, w))) + 1e-6),
            # frame range (all)
            t_start=0, t_end=None,
            # visuals mirrored from draw_frame_3d
            show_nodes=show_nodes,
            show_edges=show_edges,
            node_cmap=node_cmap,
            node_white_cut=node_white_cut,
            node_size=node_size,
            edge_mode=edge_mode,
            edge_ref_bins=edge_ref_bins,
            edge_pred_bins=edge_pred_bins,
            lw_min=lw_min, lw_max=lw_max,
            cmap_edges_activity=cmap_edges_activity,
            cmap_edges_residual=cmap_edges_residual,
            decay_k=decay_k, decay_gamma=decay_gamma, tau_decay=tau_decay, lookback=lookback,
            xyz_base=xyz_base, edges_fixed=edges_fixed,
            displace_mode=displace_mode, normals_periodic=normals_periodic,
            z_aspect_mode=z_aspect_mode, min_z_aspect_frac=min_z_aspect_frac,
            z_field_mode=z_field_mode, H_base_for_z=H_base_for_z,
            z_exaggeration=z_exaggeration, elev=elev, azim=azim,
            wall=wall,
        )

        # Save using Pillow (GIF) or ffmpeg (MP4) if available
        _ensure_dir(out_anim.parent)
        try:
            if out_anim.suffix.lower() == ".gif":
                anim.save(str(out_anim), writer="pillow", fps=max(1, int(1000/60)))
            else:  # .mp4
                try:
                    anim.save(str(out_anim), writer="ffmpeg", fps=max(1, int(1000/60)))
                except Exception:
                    # fallback to gif if ffmpeg unavailable
                    out_anim = out_anim.with_suffix(".gif")
                    anim.save(str(out_anim), writer="pillow", fps=max(1, int(1000/60)))
        finally:
            plt.close(anim._fig)  # cleanup

        print(f"Saved {out_anim}")
        return str(out_anim)

    out = Path(outpath)
    _ensure_dir(out.parent)

    H = _as_TN(H, (h, w))
    T, N = H.shape
    assert N == h * w and 0 <= t < T, "Bad H shape or t out of range."

    # ---- choose the z field (geometry) ---------------------------------------
    if z_field_mode == "data":
        zfield_t = H[t]
    elif z_field_mode == "base":
        if H_base_for_z is None:
            raise ValueError("H_base_for_z required when z_field_mode='base'")
        Hb = _as_TN(np.asarray(H_base_for_z), (h, w))
        assert Hb.shape[1] == N and t < Hb.shape[0], "Bad H_base_for_z shape"
        zfield_t = Hb[t]
    else:  # "flat"
        zfield_t = np.zeros(N, dtype=float)

    # ---- base coordinates and displacement -----------------------------------
    xyz0 = _coerce_xyz_base(xyz_base=xyz_base, N=N, h=h, w=w)
    xyz = xyz0.copy()

    zdisp = z_exaggeration * zfield_t
    if displace_mode == "normal" and xyz_base is not None:
        nrm = _compute_grid_normals(xyz0, h, w, periodic=normals_periodic)
        xyz += nrm * zdisp[:, None]
    else:
        xyz[:, 2] += zdisp

    # ---- edges & centering ---------------------------------------------------
    E_base = _coerce_edges_base(edges_fixed=edges_fixed, N=N, h=h, w=w)
    cx = 0.5 * (xyz[:, 0].min() + xyz[:, 0].max())
    cy = 0.5 * (xyz[:, 1].min() + xyz[:, 1].max())
    xyz[:, 0] -= cx
    xyz[:, 1] -= cy

    # ---- colormaps / node normalization -------------------------------------
    cmap_nodes = plt.get_cmap(node_cmap)
    if vmin is None or vmax is None:
        vmin = float(H.min() - 1e-6)
        vmax = float(H.max() + 1e-6)
        if vmin == vmax:
            vmin, vmax = vmin - 1e-3, vmax + 1e-3
    norm_nodes = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    node_scalar01 = norm_nodes(H[t])
    node_scalar01 = _squash_white(node_scalar01, node_white_cut)
    node_rgba = cmap_nodes(node_scalar01)

    # ---- figure --------------------------------------------------------------
    fig = plt.figure(figsize=(8, 7), dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    # ---- edges (optional) ----------------------------------------------------
    if show_edges:
        E_draw, lws, edge_rgba = _compute_edge_style(
            mode=edge_mode, event_bins=event_bins, t=t, E_base=E_base, N_nodes=N,
            decay_k=decay_k, decay_gamma=decay_gamma, tau_decay=tau_decay, lookback=lookback,
            edge_ref_bins=edge_ref_bins, edge_pred_bins=edge_pred_bins,
            lw_min=lw_min, lw_max=lw_max,
            cmap_edges_activity=cmap_edges_activity, cmap_edges_residual=cmap_edges_residual,
        )
        if E_draw.size:
            segs = build_edge_segments(xyz, E_draw)
            lc = Line3DCollection(segs, linewidths=lws)
            lc.set_color(edge_rgba)
            ax.add_collection3d(lc)
            _force_projection_depth(lc, +1e9)

    # ---- nodes (optional) ----------------------------------------------------
    if show_nodes:
        sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                        s=node_size, c=node_rgba, depthshade=True)
        _force_projection_depth(sc, +1e9)  # push behind walls

    # ---- limits & aspect -----------------------------------------------------
    ax.auto_scale_xyz(xyz[:, 0], xyz[:, 1], xyz[:, 2], had_data=False)
    xmin, xmax = float(xyz[:, 0].min()), float(xyz[:, 0].max())
    ymin, ymax = float(xyz[:, 1].min()), float(xyz[:, 1].max())
    zmin, zmax = float(xyz[:, 2].min()), float(xyz[:, 2].max())

    padx = 0.03 * max(1.0, xmax - xmin);
    ax.set_xlim(xmin - padx, xmax + padx)
    pady = 0.03 * max(1.0, ymax - ymin);
    ax.set_ylim(ymin - pady, ymax + pady)
    padz = 0.08 * max(1.0, zmax - zmin);
    ax.set_zlim(zmin - padz, zmax + padz)

    x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
    z_span = ax.get_zlim()[1] - ax.get_zlim()[0]
    mean_xy = 0.5 * (x_span + y_span)

    is_grid_geom = (N == h * w) and (xyz_base is None)  # true grid default
    is_grid_like = (N == h * w) and (xyz0.shape == (N, 3))  # allows torus_surface which is still grid-indexed

    if displace_mode == "normal" and is_grid_like:
        nrm = _compute_grid_normals(xyz0, h, w, periodic=normals_periodic)
        xyz += nrm * zdisp[:, None]
        z_box = z_exaggeration * mean_xy
    else:
        xyz[:, 2] += zdisp
        z_floor = min_z_aspect_frac * mean_xy
        z_box = z_exaggeration * max(z_span, z_floor)

    ax.set_box_aspect((x_span, y_span, z_box))
    try:
        ax.set_proj_type('persp')
    except Exception:
        pass


    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(f"Grid wave — frame t={t}")

    # ---- optional wall overlay ----------------------------------------------
    if wall:
        axis = wall.get("axis", "vertical")
        index = int(wall["index"])
        gate_start = int(wall.get("gate_start", 0))
        gate_end = int(wall.get("gate_end", h if axis == "vertical" else w))
        color = wall.get("color", "#ff9900")
        alpha = float(wall.get("alpha", 0.28))
        thickness = float(wall.get("thickness", 0.18))

        xoff = -cx if (xyz_base is None) else 0.0
        yoff = -cy if (xyz_base is None) else 0.0

        zlo, zhi = ax.get_zlim()
        data_max = float(np.abs(xyz[:, 2]).max() if xyz.size else 1.0)
        z_back = min(zlo, -data_max)
        z_front = max(zhi, +data_max)
        z_mid = 0.0

        polys_bot = _wall_polys(
            h=h, w=w, zmin=z_back, zmax=z_mid,
            axis=axis, index=index, gate_start=gate_start, gate_end=gate_end,
            thickness=thickness, x_offset=xoff, y_offset=yoff,
        )
        coll_bot = _add_poly3d(ax, polys_bot, color=color, alpha=alpha, sort_hint=z_back - 1e6, lw=0.0, zorder=0)
        _force_projection_depth(coll_bot, +1e9)

        polys_top = _wall_polys(
            h=h, w=w, zmin=z_mid, zmax=z_front,
            axis=axis, index=index, gate_start=gate_start, gate_end=gate_end,
            thickness=thickness, x_offset=xoff, y_offset=yoff,
        )
        coll_top = _add_poly3d(ax, polys_top, color=color, alpha=alpha, sort_hint=z_front + 1e6, lw=0.0, zorder=10)
        _force_projection_depth(coll_top, -1e9)

    #fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return str(out)

# =============================================================================
#                                   ANIMATION
# =============================================================================

def animate_3d(
    H: np.ndarray,
    event_bins: List[csr_matrix] | None,
    h: int, w: int,
    vmin: float,
    vmax: float,
    *,
    # frame range / speed
    t_start: int = 0,
    t_end: Optional[int] = None,
    interval_ms: int = 60,
    # visibility toggles
    show_nodes: bool = True,
    show_edges: bool = True,
    # node styling
    node_cmap: str = "seismic",
    node_white_cut: Tuple[float, float] = (0.45, 0.55),
    node_size: float = 20.0,
    # edge styling / modes
    edge_mode: EdgeMode = "activity",
    edge_ref_bins: List[csr_matrix] | None = None,
    edge_pred_bins: List[csr_matrix] | None = None,
    lw_min: float = 0.5,
    lw_max: float = 2.0,
    cmap_edges_activity: str = "Greys",
    cmap_edges_residual: str = "seismic",
    alpha_edges: float = 1.0,      # constant alpha; color encodes recency or divergence
    # strength/recency decay
    decay_k: int = 30,
    decay_gamma: float = 0.6,
    tau_decay: float = 5.0,
    lookback: int | None = None,
    # geometry / displacement
    xyz_base: Optional[np.ndarray] = None,      # (N,3)
    edges_fixed: Optional[np.ndarray] = None,   # (M,2)
    displace_mode: str = "z",                   # "z" | "normal"
    normals_periodic: bool = True,
    z_aspect_mode: str = "auto",                # "auto" | "grid"
    min_z_aspect_frac: float = 0.25,
    z_field_mode: str = "data",                 # "data" | "base" | "flat"
    H_base_for_z: Optional[np.ndarray] = None,  # (T,N) if z_field_mode="base"
    z_exaggeration: float = 1.0,
    elev: float = 35,
    azim: float = -60,
    # wall overlay (optional)
    wall: Optional[dict] = None,
) -> FuncAnimation:
    """Animate with visuals that match draw_frame_3d exactly."""
    # Normalize H
    H = _as_TN(H, (h, w))
    T, N = H.shape
    if t_end is None:
        t_end = T
    t_start = max(0, int(t_start))
    t_end = min(T, int(t_end))

    # ── choose the z-field provider ───────────────────────────────────────────
    if z_field_mode == "data":
        def zfield_at(tt: int) -> np.ndarray: return H[tt]
    elif z_field_mode == "base":
        if H_base_for_z is None:
            raise ValueError("H_base_for_z required when z_field_mode='base'")
        Hb = _as_TN(np.asarray(H_base_for_z), (h, w))
        assert Hb.shape[1] == N, "Bad H_base_for_z shape"
        def zfield_at(tt: int) -> np.ndarray: return Hb[min(tt, Hb.shape[0]-1)]
    else:  # "flat"
        def zfield_at(tt: int) -> np.ndarray: return np.zeros(N, float)

    # ── layout (xyz and edges) ────────────────────────────────────────────────
    xyz0 = grid_node_xyz(h, w) if xyz_base is None else np.asarray(xyz_base, float)
    assert xyz0.shape == (N, 3), f"xyz_base must be (N,3), got {xyz0.shape}"
    E_base = grid_edges(h, w) if edges_fixed is None else np.asarray(edges_fixed, int)

    def frame_xyz(tt: int) -> np.ndarray:
        out = xyz0.copy()
        zdisp = z_exaggeration * zfield_at(tt)
        if displace_mode == "normal" and xyz_base is not None:
            nrm = _compute_grid_normals(xyz0, h, w, periodic=normals_periodic)
            out += nrm * zdisp[:, None]
        else:
            out[:, 2] += zdisp
        cx = 0.5 * (out[:, 0].min() + out[:, 0].max())
        cy = 0.5 * (out[:, 1].min() + out[:, 1].max())
        out[:, 0] -= cx
        out[:, 1] -= cy
        return out

    # ── node colormap / normalization ─────────────────────────────────────────
    cmap_nodes = plt.get_cmap(node_cmap)
    norm_nodes = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    # ── figure & initial artists ──────────────────────────────────────────────
    fig = plt.figure(figsize=(8, 7), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    xyz = frame_xyz(t_start)

    # Nodes
    if show_nodes:
        node_scalar01 = _squash_white(norm_nodes(H[t_start]), node_white_cut)
        node_rgba = cmap_nodes(node_scalar01)
        sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=node_size, c=node_rgba, depthshade=False)
        sc.set_zorder(2)
        if hasattr(sc, "set_sort_zpos"):
            sc.set_sort_zpos(-1e9)
    else:
        sc = None

    # Edges
    if show_edges:
        segs = build_edge_segments(xyz, E_base)
        lc = Line3DCollection(segs, linewidths=np.full(E_base.shape[0], lw_min))
        init_rgba = np.zeros((E_base.shape[0], 4), float)
        init_rgba[:, :3] = 0.5  # placeholder grey; will be set in first update
        init_rgba[:, 3] = alpha_edges
        lc.set_color(init_rgba)
        lc.set_zorder(10)
        if hasattr(lc, "set_sort_zpos"):
            lc.set_sort_zpos(+1e9)
        ax.add_collection3d(lc)
    else:
        lc = None

    # Limits & aspect (match draw_frame_3d)
    ax.auto_scale_xyz(xyz[:, 0], xyz[:, 1], xyz[:, 2], had_data=False)
    xmin, xmax = float(xyz[:, 0].min()), float(xyz[:, 0].max())
    ymin, ymax = float(xyz[:, 1].min()), float(xyz[:, 1].max())
    zmin, zmax = float(xyz[:, 2].min()), float(xyz[:, 2].max())
    padx = 0.03 * max(1.0, xmax - xmin); ax.set_xlim(xmin - padx, xmax + padx)
    pady = 0.03 * max(1.0, ymax - ymin); ax.set_ylim(ymin - pady, ymax + pady)
    padz = 0.08 * max(1.0, zmax - zmin); ax.set_zlim(zmin - padz, zmax + padz)

    x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
    z_span = ax.get_zlim()[1] - ax.get_zlim()[0]
    mean_xy = 0.5 * (x_span + y_span)

    if z_aspect_mode == "grid" or (xyz_base is None and displace_mode != "normal"):
        z_box = z_exaggeration * mean_xy
    else:
        z_floor = min_z_aspect_frac * mean_xy
        z_box = z_exaggeration * max(z_span, z_floor)

    ax.set_box_aspect((x_span, y_span, z_box))
    try: ax.set_proj_type('persp')
    except Exception: pass
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    title = ax.set_title(f"Grid wave — frame t={t_start}")

    # Prebind edge colormaps for modes
    cmap_edges_act = plt.get_cmap(cmap_edges_activity)
    cmap_edges_res = plt.get_cmap(cmap_edges_residual)

    # ── wall overlay (static slabs spanning z-range; same as draw_frame_3d) ───
    # (We draw once; if your wall needs to move with z-limits dynamically,
    #  you can recompute inside update().)
    if wall:
        axis = wall.get("axis", "vertical")
        index = int(wall["index"])
        gate_start = int(wall.get("gate_start", 0))
        gate_end = int(wall.get("gate_end", h if axis == "vertical" else w))
        color = wall.get("color", "#ff9900")
        alpha = float(wall.get("alpha", 0.28))
        thickness = float(wall.get("thickness", 0.18))
        if xyz_base is None:
            # grid_node_xyz uses x∈[0,w-1], y∈[0,h-1]
            cx0 = 0.5 * (w - 1)
            cy0 = 0.5 * (h - 1)
            xoff, yoff = -cx0, -cy0
        else:
            # when a custom xyz_base is supplied, it's rendered "as is";
            # frame_xyz centers with that base's cx,cy, so we don't offset walls.
            xoff, yoff = 0.0, 0.0

        zlo, zhi = ax.get_zlim()
        data_max = float(np.abs(xyz[:, 2]).max() if xyz.size else 1.0)
        z_back = min(zlo, -data_max)
        z_front = max(zhi, +data_max)
        z_mid = 0.0
        polys_bot = _wall_polys(
            h=h, w=w, zmin=z_back, zmax=z_mid,
            axis=axis, index=index, gate_start=gate_start, gate_end=gate_end,
            thickness=thickness, x_offset=xoff, y_offset=yoff,
        )
        coll_bot = _add_poly3d(ax, polys_bot, color=color, alpha=alpha, sort_hint=z_back - 1e6, lw=0.0, zorder=0)
        _force_projection_depth(coll_bot, +1e9)

        polys_top = _wall_polys(
            h=h, w=w, zmin=z_mid, zmax=z_front,
            axis=axis, index=index, gate_start=gate_start, gate_end=gate_end,
            thickness=thickness, x_offset=xoff, y_offset=yoff,
        )
        coll_top = _add_poly3d(ax, polys_top, color=color, alpha=alpha, sort_hint=z_front + 1e6, lw=0.0, zorder=10)
        _force_projection_depth(coll_top, -1e9)

    # ── per-frame update (mirror draw_frame_3d logic) ─────────────────────────
    def update(tt: int):
        print(tt)
        xyz = frame_xyz(tt)
        if sc is not None:
            sc._offsets3d = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
            node_scalar01 = _squash_white(norm_nodes(H[tt]), node_white_cut)
            sc.set_facecolors(cmap_nodes(node_scalar01))

        if lc is not None:
            # choose edge visuals per mode (uses same helper as draw_frame_3d)
            E_draw, lws, edge_rgba = _compute_edge_style(
                mode=edge_mode,
                event_bins=event_bins,
                t=tt,
                E_base=E_base,
                N_nodes=N,
                decay_k=decay_k,
                decay_gamma=decay_gamma,
                tau_decay=tau_decay,
                lookback=lookback,
                edge_ref_bins=edge_ref_bins,
                edge_pred_bins=edge_pred_bins,
                lw_min=lw_min,
                lw_max=lw_max,
                cmap_edges_activity=cmap_edges_activity,
                cmap_edges_residual=cmap_edges_residual,
            )
            if E_draw.size:
                segs = build_edge_segments(xyz, E_draw)
                lc.set_segments(segs)
                lc.set_linewidths(lws)
                # colors already computed by _compute_edge_style; just set alpha
                edge_rgba = edge_rgba.copy()
                edge_rgba[:, 3] = alpha_edges
                lc.set_color(edge_rgba)
            else:
                # nothing to draw this frame
                lc.set_segments(np.zeros((0, 2, 3)))
                lc.set_linewidths([])
                lc.set_color([])

        title.set_text(f"Grid wave — frame t={tt}")
        return (tuple(x for x in (sc, lc, title) if x is not None))

    return FuncAnimation(fig, update, frames=range(t_start, t_end), interval=interval_ms, blit=False)

# =============================================================================
#                           EDGES-ONLY RENDERING MODES
# =============================================================================

def _unique_rows(arr: np.ndarray) -> np.ndarray:
    """Return unique rows of a (M,2) integer array."""
    if arr.size == 0:
        return arr
    arr = np.ascontiguousarray(arr)
    order = np.lexsort((arr[:,1], arr[:,0]))
    arr_sorted = arr[order]
    keep = np.ones(arr_sorted.shape[0], dtype=bool)
    keep[1:] = np.any(arr_sorted[1:] != arr_sorted[:-1], axis=1)
    return arr_sorted[keep]

def _edges_from_bins_window(bins_list: List[csr_matrix], t: int, lookback: Optional[int], N: int) -> np.ndarray:
    """Collect undirected edges (u<v) that appear in [t-lookback, t] window."""
    if not bins_list:
        return np.zeros((0,2), dtype=np.int64)
    t0 = max(0, int(t - (lookback if lookback is not None else 0)))
    t1 = min(len(bins_list), int(t) + 1)
    pairs = []
    for k in range(t0, t1):
        B = bins_list[k].tocoo()
        if B.nnz == 0:
            continue
        u, v = B.row.astype(np.int64), B.col.astype(np.int64)
        uu, vv = np.minimum(u, v), np.maximum(u, v)
        mask = (uu != vv) & (uu >= 0) & (vv < N)
        if mask.any():
            pairs.append(np.stack([uu[mask], vv[mask]], axis=1))
    if not pairs:
        return np.zeros((0,2), dtype=np.int64)
    return _unique_rows(np.concatenate(pairs, axis=0))


def plot_rollout(
    *,
    variants: Dict[str, Dict[str, object]],
    # variants[name] must include:
    #   "X": np.ndarray with shape (T,N) or (T,h,w)
    #   "edges": List[csr_matrix]  (event bins per time)
    h: int,
    w: int,
    t: int|None,
    outdir: str = "exports_nodes_edges",
    # Nodes scale reference (for consistent diverging colors)
    reference_nodes: Optional[np.ndarray] = None,   # e.g., X_truth_nodes_viz
    include_variants_in_node_scale: bool = True,    # include all variants when picking vmin/vmax
    # Edge reference / predictions (optional, enables residual & confusion panels)
    reference_edges: Optional[List[csr_matrix]] = None,  # e.g., y_holdout_viz
    pred_edge_bins: Optional[List[csr_matrix]] = None,   # e.g., event_bins_pred_soft
    # Which panels to render
    do_nodes_panels: bool = True,          # nodes-only per variant
    do_edges_activity: bool = True,        # edges-only activity per variant
    do_edges_residual: bool = True,        # edges-only residual per variant (requires reference_edges)
    do_edges_confusion: bool = True,       # edges-only confusion per variant (requires reference_edges; uses top-quantile thresholds)
    do_combined_activity: bool = True,     # nodes+edges activity per variant
    do_nodes_residual: bool = True,        # nodes-only residual vs reference_nodes (geometry from reference)
    # Colormaps / styling
    cm_nodes: str = "seismic",
    cm_edges_activity: str = "Greys",
    cm_edges_residual: str = "seismic",
    node_size: float = 20.0,
    lw_min: float = 1.9,
    lw_max: float = 2.4,
    # Edge-strength / recency decay knobs
    decay_k: int = 10,
    decay_gamma: float = 0.6,
    tau_decay: float = 5.0,
    lookback: Optional[int] = None,
    # Geometry / camera
    frame_kwargs: Optional[Dict] = None,  # can include: xyz_base, edges_fixed, displace_mode, normals_periodic, z_aspect_mode, min_z_aspect_frac, elev, azim, wall
    z_exaggeration: float = 0.8,
) -> None:
    """
    Generate a small panel suite per variant (nodes-only, edges-only modes, combined),
    using the unified draw_frame_3d. This replaces export_nodes_suite, export_edges_suite,
    and export_variants in a single, explicit call.

    Parameters
    ----------
    variants : dict
      Mapping name -> {"X": np.ndarray, "edges": List[csr_matrix], (optional) "frame_kwargs": dict}.
    h, w : int
      Grid height/width. N must equal h*w after reshaping.
    t : int
      Frame index to render.
    outdir : str
      Base output directory. Subfolders are created for nodes/edges panels.
    reference_nodes : np.ndarray, optional
      If provided, used to set shared node vmin/vmax (recommended for comparability).
    include_variants_in_node_scale : bool
      If True, node vmin/vmax consider all variant X as well as reference_nodes.
    reference_edges : List[csr_matrix], optional
      If provided, enables residual/confusion edge panels.
    pred_edge_bins : List[csr_matrix], optional
      Optional predictions for confusion panels (if None, uses reference_edges for both).
    """
    # --- Shared scales (so all panels compare apples-to-apples) ---------------
    def _robust_minmax(arrs, lo=1, hi=99):
        Xcat = np.concatenate([np.asarray(a) for a in arrs], axis=0)
        vmin = float(np.nanpercentile(Xcat, lo))
        vmax = float(np.nanpercentile(Xcat, hi))
        if vmin == vmax:
            vmin, vmax = float(np.nanmin(Xcat)), float(np.nanmax(Xcat) + 1e-6)
        return vmin, vmax

    # --- paths
    base = Path(outdir)
    nodes_dir = base / "nodes_suite"
    edges_dir = base / "edges_suite"
    _ensure_dir(base)
    _ensure_dir(nodes_dir)
    _ensure_dir(edges_dir)

    # --- common node color scale (diverging) for comparability ---------------
    node_scale_sources = []
    if reference_nodes is not None:
        node_scale_sources.append(reference_nodes)
    if include_variants_in_node_scale:
        node_scale_sources.extend([v["X"] for v in variants.values() if "X" in v])

    if not node_scale_sources:
        # fallback to any variant (shouldn’t happen in practice)
        node_scale_sources = [v["X"] for v in variants.values() if "X" in v]

    node_vmin, node_vmax = _robust_minmax(node_scale_sources, 1, 99)

    # --- pass-through geometry/camera knobs ----------------------------------
    frame_kwargs = dict(frame_kwargs or {})
    geo = dict(
        xyz_base=frame_kwargs.get("xyz_base"),
        edges_fixed=frame_kwargs.get("edges_fixed"),
        displace_mode=frame_kwargs.get("displace_mode", "z"),
        normals_periodic=frame_kwargs.get("normals_periodic", True),
        z_aspect_mode=frame_kwargs.get("z_aspect_mode", "auto"),
        min_z_aspect_frac=frame_kwargs.get("min_z_aspect_frac", 0.25),
        z_exaggeration=z_exaggeration,
        elev=frame_kwargs.get("elev", 35),
        azim=frame_kwargs.get("azim", -60),
        wall=frame_kwargs.get("wall"),
    )

    edge_decay = dict(decay_k=decay_k, decay_gamma=decay_gamma, tau_decay=tau_decay, lookback=lookback)

    # --- render per variant ---------------------------------------------------
    for name, pack in variants.items():
        X = pack["X"]
        bins = pack["edges"]

        # Nodes-only (replaces export_nodes_suite)
        if do_nodes_panels:
            out = nodes_dir / f"{name}_nodes_t{t}.png"
            draw_frame_3d(
                H=X, event_bins=bins, h=h, w=w, t=t, outpath=str(out),
                vmin=node_vmin, vmax=node_vmax,
                show_nodes=True, show_edges=False,
                node_cmap=cm_nodes, node_size=node_size, node_alpha=1.0,
                **geo
            )

        # Edges-only: activity (replaces export_edges_suite activity)
        if do_edges_activity:
            out = edges_dir / f"{name}_edges_activity_t{t}.png"
            draw_frame_3d(
                H=X, event_bins=bins, h=h, w=w, t=t, outpath=str(out),
                vmin=None, vmax=None,  # nodes hidden -> node scale irrelevant
                show_nodes=False, show_edges=True,
                edge_mode="activity",
                cmap_edges_activity=cm_edges_activity,
                lw_min=lw_min, lw_max=lw_max,
                z_field_mode="flat",
                **edge_decay, **geo
            )

        # Edges-only: residuals (requires reference_edges)
        if do_edges_residual and (reference_edges is not None):
            out = edges_dir / f"{name}_edges_residual_t{t}.png"
            draw_frame_3d(
                H=X, event_bins=bins, h=h, w=w, t=t, outpath=str(out),
                vmin=None, vmax=None,
                show_nodes=False, show_edges=True,
                edge_mode="residual",
                edge_ref_bins=reference_edges,
                cmap_edges_residual=cm_edges_residual,
                lw_min=lw_min, lw_max=lw_max,
                z_field_mode="flat",
                **edge_decay, **geo
            )

        # Edges-only: confusion (requires reference_edges)
        if do_edges_confusion and (reference_edges is not None):
            out = edges_dir / f"{name}_edges_confusion_t{t}.png"
            draw_frame_3d(
                H=X, event_bins=bins, h=h, w=w, t=t, outpath=str(out),
                vmin=None, vmax=None,
                show_nodes=False, show_edges=True,
                edge_mode="confusion",
                edge_ref_bins=reference_edges,
                edge_pred_bins=pred_edge_bins,
                lw_min=lw_min, lw_max=lw_max,
                z_field_mode="flat",
                **edge_decay, **geo
            )

        # Combined nodes+edges (replaces export_variants base panels)
        if do_combined_activity:
            out = base / f"{name}_frame_t{t}.png"
            draw_frame_3d(
                H=X, event_bins=bins, h=h, w=w, t=t, outpath=str(out),
                vmin=node_vmin, vmax=node_vmax,
                show_nodes=True, show_edges=True,
                node_cmap=cm_nodes, node_size=node_size,
                edge_mode="activity",
                cmap_edges_activity=cm_edges_activity,
                **edge_decay, **geo
            )

        # Nodes-only residual (geometry from reference_nodes)
        if do_nodes_residual and (reference_nodes is not None):
            # Color by residual (X - reference), but use base geometry from reference field
            # for a stable surface; scales remain the activity scale for comparability.
            R = X  # pass X; z geometry comes from reference via z_field_mode="base"
            out = base / f"res_{name}_frame_t{t}.png"
            draw_frame_3d(
                H=R, event_bins=bins, h=h, w=w, t=t, outpath=str(out),
                vmin=node_vmin, vmax=node_vmax,
                show_nodes=True, show_edges=False,
                node_cmap="seismic", node_size=node_size,
                z_field_mode="base", H_base_for_z=reference_nodes,
                **geo
            )

def make_plot_kwargs(kind: str,
                     h: int, w: int,
                     meta: dict,
                     adj: csr_matrix,
                     *, z_exaggeration: float = 1.0):
    """
    Return (frame_kwargs, anim_kwargs) tailored for:
      - grid-family ("grid", "gate"): grid aspect
      - torus_surface: 3D surface with normals + wrapped periodic assumption
      - embedded graphs (ring, small_world, rgg_*, sbm, tree, sphere_*): coords2d/coords3d scatter
    """
    def clean(d):
        return {k: v for k, v in (d or {}).items() if v is not None}

    frame = {"z_exaggeration": z_exaggeration}
    anim  = {"z_exaggeration": z_exaggeration}

    # ---- 1) torus_surface (your existing special case) ----
    xyz = meta.get("coords3d")
    if kind == "torus_surface" and xyz is not None:
        efix = edges_from_adj(adj)
        torus_common = {
            "xyz_base": xyz,
            "edges_fixed": efix,
            "displace_mode": "normal",
            "normals_periodic": True,
            "z_aspect_mode": "auto",
            "min_z_aspect_frac": 0.20,
        }
        frame.update(torus_common)
        anim.update(torus_common)
        return clean(frame), clean(anim)

    # ---- 2) gate (your existing special case) ----
    if kind == "gate":
        g = meta.get("gate", {}) or {}
        axis = g.get("axis", 'vertical')
        if axis not in ("vertical", "horizontal"):
            axis = "vertical"

        default_index = (w // 2) if axis == "vertical" else (h // 2)
        index = int(g.get("wall_index", meta.get("wall_index", default_index)))

        if "span" in g and g["span"] is not None:
            try:
                gate_start, gate_end = map(int, g["span"])
            except Exception:
                gate_start = (h // 3) if axis == "vertical" else (w // 3)
                gate_end = (2 * h // 3) if axis == "vertical" else (2 * w // 3)
        else:
            span_len = h if axis == "vertical" else w
            gate_start = int(meta.get("gate_start", span_len // 3))
            gate_end = int(meta.get("gate_end", (2 * span_len) // 3))

        span_len = h if axis == "vertical" else w
        gate_start = max(0, min(span_len, gate_start))
        gate_end = max(gate_start, min(span_len, gate_end))

        frame.update({
            "z_aspect_mode": "grid",
            "min_z_aspect_frac": 0.18,
            "wall": {
                "axis": axis,
                "index": index,
                "gate_start": gate_start,
                "gate_end": gate_end,
                "color": "#aaaaaa",
                "alpha": 0.50,
                "thickness": 0.12,
            }
        })
        anim.update({
            "z_aspect_mode": "grid",
            "min_z_aspect_frac": 0.18,
        })
        return clean(frame), clean(anim)

    # ---- 3) NEW: generic embedded graphs (2D or 3D) ----
    # Prefer coords2d; fall back to coords (random geometric currently uses "coords").
    xy = meta.get("coords2d", None)
    if xy is None:
        X = meta.get("coords", None)
        if isinstance(X, (list, tuple)):
            X = None
        if X is not None and getattr(X, "ndim", None) == 2 and X.shape[1] >= 2:
            xy = X[:, :2]

    xyz = meta.get("coords3d", None)

    if xyz is not None:
        # Generic 3D embedding, but *not* necessarily periodic normals.
        efix = edges_from_adj(adj)
        common3d = {
            "xyz_base": xyz,
            "edges_fixed": efix,
            "displace_mode": meta.get("displace_mode", "z"),   # "z" is safer generically than "normal"
            "normals_periodic": bool(meta.get("normals_periodic", False)),
            "z_aspect_mode": "auto",
            "min_z_aspect_frac": 0.20,
        }
        frame.update(common3d)
        anim.update(common3d)
        return clean(frame), clean(anim)

    if xy is not None:
        # 2D embedding scatter; edges from adjacency.
        efix = edges_from_adj(adj)
        common2d = {
            "xy_base": xy,
            "edges_fixed": efix,
            "z_aspect_mode": "embed",   # tell renderer “don’t treat as grid”
            "min_z_aspect_frac": 0.20,
        }
        frame.update(common2d)
        anim.update(common2d)
        return clean(frame), clean(anim)

    # ---- 4) fallback: original grid default ----
    frame.update({"z_aspect_mode": "grid", "min_z_aspect_frac": 0.20})
    anim.update({"z_aspect_mode": "grid", "min_z_aspect_frac": 0.20})
    return clean(frame), clean(anim)
