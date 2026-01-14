import numpy as np
import scipy.sparse as sp
from typing import Callable, Dict, Tuple, Optional, Literal, Sequence
from abc import ABC, abstractmethod

# ---------- helpers ----------
from scipy.sparse import csr_matrix
from mpl_toolkits.mplot3d import Axes3D

def _idx(i: np.ndarray, j: np.ndarray, n: int) -> np.ndarray:
    return i.astype(np.int64) * n + j.astype(np.int64)

def _assemble_sparse(rows: np.ndarray, cols: np.ndarray, data: np.ndarray, N: int) -> sp.csr_matrix:
    A = sp.csr_matrix((data.astype(np.float32, copy=False), (rows, cols)), shape=(N, N))
    return A

def _csr_from_edges(u, v, w, N) -> sp.csr_matrix:
    A = sp.csr_matrix((w.astype(np.float32), (u, v)), shape=(N, N))
    return A

def _symmetrize(A: sp.csr_matrix) -> sp.csr_matrix:
    B = A + A.T
    B.setdiag(0); B.eliminate_zeros()
    return B

def _pairwise_dists(X: np.ndarray, periodic_box: Optional[float]=None):
    # X: [N,d]; if periodic_box set (e.g., 1.0), use min-image distance on a torus box
    N, d = X.shape
    # chunked to keep memory modest for large N (simple version here)
    D = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        dif = X - X[i]
        if periodic_box is not None:
            dif = dif - np.round(dif / periodic_box) * periodic_box
        D[i] = np.sqrt((dif * dif).sum(axis=1))
    return D

def _knn_edges(X: np.ndarray, k: int, periodic_box: Optional[float]=None):
    D = _pairwise_dists(X, periodic_box)
    idx = np.argsort(D, axis=1)[:, 1:k+1]  # skip self
    rows = np.repeat(np.arange(X.shape[0]), k)
    cols = idx.ravel()
    return rows, cols

def _radius_edges(X: np.ndarray, r: float, periodic_box: Optional[float]=None):
    D = _pairwise_dists(X, periodic_box)
    u, v = np.where((D <= r) & (D > 0))
    return u.astype(np.int64), v.astype(np.int64)

def edges_from_adj(adj: csr_matrix) -> np.ndarray:
    """Undirected unique edge list from an adjacency CSR (row-major node ids)."""
    rows, cols = adj.nonzero()
    a = np.minimum(rows, cols)
    b = np.maximum(rows, cols)
    keep = a < b
    return np.stack([a[keep], b[keep]], axis=1).astype(int)

def _circle_coords2d(n: int, radius: float = 1.0, phase: float = 0.0) -> np.ndarray:
    t = phase + 2 * np.pi * np.arange(n) / n
    return np.stack([radius * np.cos(t), radius * np.sin(t)], axis=1).astype(np.float32)

def _finalize_meta(A: sp.csr_matrix, meta: Dict, kind: str) -> Dict:
    """
    Normalize meta across all graph builders so downstream plotting/eval can be uniform.

    Guarantees:
      - meta["N"] : int
      - meta["kind"] : str
      - meta["coords2d"] if any coords are available (best-effort)
      - meta["coords3d"] if available (unchanged)
      - meta["edges_fixed"] : (E,2) ndarray unique undirected edges (best-effort)
      - meta["shape"] kept if provided by grid-like builders
    """
    meta = dict(meta or {})
    assert A.shape is not None, "Adjacency must have shape"
    N = int(A.shape[0])
    meta["N"] = N
    meta["kind"] = meta.get("kind", kind)

    # ---- normalize coords keys ----
    # If builder provides "coords" (generic), treat as coords2d if dim>=2, coords3d if dim==3.
    if "coords2d" not in meta and "coords" in meta:
        X = np.asarray(meta["coords"])
        if X.ndim == 2 and X.shape[1] >= 2:
            meta["coords2d"] = X[:, :2].astype(np.float32, copy=False)
        if "coords3d" not in meta and X.ndim == 2 and X.shape[1] == 3:
            meta["coords3d"] = X.astype(np.float32, copy=False)

    # If we have coords3d but no coords2d, make a reasonable 2D projection
    if "coords2d" not in meta and "coords3d" in meta:
        X3 = np.asarray(meta["coords3d"])
        if X3.ndim == 2 and X3.shape[1] >= 2:
            meta["coords2d"] = X3[:, :2].astype(np.float32, copy=False)

    # If we have grid shape but no coords2d (shouldn't happen, but be safe)
    if "coords2d" not in meta and "shape" in meta:
        m, n = meta["shape"]
        I, J = np.indices((m, n))
        meta["coords2d"] = np.stack([J.ravel(), I.ravel()], axis=1).astype(np.float32)

    # ---- precompute fixed edges for plotting (unique undirected) ----
    try:
        meta["edges_fixed"] = edges_from_adj(A.tocsr())
    except Exception:
        pass

    return meta


# ---------- builder abstraction (runtime enforcement) ----------
class GraphBuilder(ABC):
    """Abstract graph-builder interface.

    Implementations must return a non-None (A, meta) tuple where A is a
    scipy sparse adjacency matrix and meta is a dict (may be empty).
    """

    @abstractmethod
    def build(self, **kwargs) -> Tuple[sp.csr_matrix, Dict]:
        raise NotImplementedError()


class FunctionBuilder(GraphBuilder):
    """Wrap a plain function so it conforms to `GraphBuilder` and validates
    the return value at runtime.
    """

    def __init__(self, fn: Callable[..., object]):
        self._fn = fn

    def build(self, **kwargs) -> Tuple[sp.csr_matrix, Dict]:
        res = self._fn(**kwargs)
        if res is None:
            raise RuntimeError(f"graph builder {getattr(self._fn, '__name__', repr(self._fn))} returned None")
        if not isinstance(res, (tuple, list)) or len(res) < 2:
            raise RuntimeError(f"graph builder {getattr(self._fn, '__name__', repr(self._fn))} returned invalid result: {type(res)}")
        A, meta = res[0], res[1]
        if A is None:
            raise RuntimeError(f"graph builder {getattr(self._fn, '__name__', repr(self._fn))} returned adjacency None")
        return A, meta




# ---------- canonical visual graphs ----------

def build_ring(n: int, *,
               directed: bool = False,
               matrix_format: str = "csr") -> Tuple[sp.csr_matrix, Dict]:
    nodes = np.arange(n, dtype=np.int64)
    u = nodes
    v = (nodes + 1) % n
    w = np.ones_like(u, dtype=np.float32)

    A = _csr_from_edges(u, v, w, n)
    if not directed:
        A = _symmetrize(A)

    coords2d = _circle_coords2d(n, radius=1.0)
    meta = {"coords2d": coords2d, "kind": "ring", "directed": directed}
    return A, meta


def build_directed_ring(n: int, *,
                        p_back: float = 0.0,
                        matrix_format: str = "csr",
                        seed: int = 0) -> Tuple[sp.csr_matrix, Dict]:
    # directed i -> i+1, optional sparse back edges
    rng = np.random.default_rng(seed)
    nodes = np.arange(n, dtype=np.int64)

    u = nodes
    v = (nodes + 1) % n

    if p_back > 0:
        mask = rng.random(n) < p_back
        ub = (nodes + 1)[mask] % n
        vb = nodes[mask]
        u = np.concatenate([u, ub])
        v = np.concatenate([v, vb])

    w = np.ones_like(u, dtype=np.float32)
    A = _csr_from_edges(u, v, w, n)  # keep directed

    coords2d = _circle_coords2d(n, radius=1.0)
    meta = {"coords2d": coords2d, "kind": "directed_ring", "directed": True, "p_back": p_back}
    return A, meta


def build_ring_chords(n: int, *,
                      n_chords: int = 4,
                      chord_span: Optional[int] = None,
                      directed: bool = False,
                      matrix_format: str = "csr",
                      seed: int = 0) -> Tuple[sp.csr_matrix, Dict]:
    rng = np.random.default_rng(seed)

    # base ring edges
    nodes = np.arange(n, dtype=np.int64)
    u = nodes
    v = (nodes + 1) % n

    # chord edges
    if chord_span is None:
        chord_span = max(2, n // 3)

    cu, cv = [], []
    for _ in range(n_chords):
        a = int(rng.integers(0, n))
        # pick b "far enough" from a on the ring
        delta = int(rng.integers(chord_span, n - chord_span))
        b = (a + delta) % n
        if a != b:
            cu.append(a); cv.append(b)

    if len(cu) > 0:
        u = np.concatenate([u, np.asarray(cu, dtype=np.int64)])
        v = np.concatenate([v, np.asarray(cv, dtype=np.int64)])

    w = np.ones_like(u, dtype=np.float32)
    A = _csr_from_edges(u, v, w, n)
    if not directed:
        A = _symmetrize(A)

    coords2d = _circle_coords2d(n, radius=1.0)
    meta = {
        "coords2d": coords2d,
        "kind": "ring_chords",
        "directed": directed,
        "n_chords": n_chords,
        "chord_span": chord_span,
    }
    return A, meta


def build_wheel(n: int, *,
                hub: int = 0,
                directed: bool = False,
                matrix_format: str = "csr") -> Tuple[sp.csr_matrix, Dict]:
    if n < 4:
        raise ValueError("wheel needs n>=4")
    hub = int(hub)

    nodes = np.arange(n, dtype=np.int64)
    rim = nodes[nodes != hub]

    # rim cycle
    u1 = rim
    v1 = np.roll(rim, -1)

    # spokes
    u2 = np.full_like(rim, hub)
    v2 = rim

    u = np.concatenate([u1, u2])
    v = np.concatenate([v1, v2])
    w = np.ones_like(u, dtype=np.float32)

    A = _csr_from_edges(u, v, w, n)
    if not directed:
        A = _symmetrize(A)

    coords2d = np.zeros((n, 2), dtype=np.float32)
    coords2d[hub] = np.array([0.0, 0.0], dtype=np.float32)
    coords2d[rim] = _circle_coords2d(len(rim), radius=1.0)
    meta = {"coords2d": coords2d, "kind": "wheel", "hub": hub, "directed": directed}
    return A, meta

def build_sbm(*,
              sizes: Sequence[int] = (40, 40, 40),
              p_in: float = 0.18,
              p_out: float = 0.02,
              P: Optional[np.ndarray] = None,
              directed: bool = False,
              matrix_format: str = "csr",
              seed: int = 0) -> Tuple[sp.csr_matrix, Dict]:
    rng = np.random.default_rng(seed)

    sizes = [int(s) for s in sizes]
    k = len(sizes)
    n = int(sum(sizes))

    if P is None:
        P = np.full((k, k), float(p_out), dtype=np.float32)
        np.fill_diagonal(P, float(p_in))
    else:
        P = np.asarray(P, dtype=np.float32)
        if P.shape != (k, k):
            raise ValueError(f"P must be shape {(k, k)}")

    # block node lists
    blocks = []
    block_ids = np.empty(n, dtype=np.int64)
    off = 0
    for bi, sz in enumerate(sizes):
        nodes = np.arange(off, off + sz, dtype=np.int64)
        blocks.append(nodes)
        block_ids[nodes] = bi
        off += sz

    u_list, v_list = [], []

    for a in range(k):
        Na = blocks[a]
        for b in range(a, k):
            Nb = blocks[b]
            pab = float(P[a, b])
            if pab <= 0:
                continue

            if a == b:
                m = len(Na)
                ii, jj = np.triu_indices(m, k=1)
                mask = rng.random(len(ii)) < pab
                u = Na[ii[mask]]
                v = Na[jj[mask]]
            else:
                U = np.repeat(Na, len(Nb))
                V = np.tile(Nb, len(Na))
                mask = rng.random(len(U)) < pab
                u = U[mask]
                v = V[mask]

            u_list.append(u); v_list.append(v)

    u = np.concatenate(u_list) if len(u_list) else np.array([], dtype=np.int64)
    v = np.concatenate(v_list) if len(v_list) else np.array([], dtype=np.int64)
    w = np.ones_like(u, dtype=np.float32)

    A = _csr_from_edges(u, v, w, n)
    if not directed:
        A = _symmetrize(A)

    # coords: communities on big circle, nodes on small circles
    centers = _circle_coords2d(k, radius=2.5)
    coords2d = np.zeros((n, 2), dtype=np.float32)
    for bi, nodes in enumerate(blocks):
        local = _circle_coords2d(len(nodes), radius=0.7)
        coords2d[nodes] = local + centers[bi]

    meta = {
        "coords2d": coords2d,
        "kind": "sbm",
        "directed": directed,
        "sizes": sizes,
        "P": P,
        "block_ids": block_ids,
        "blocks": blocks,
    }
    return A, meta


# ---------- 3. Random graphs with geometry ----------
def build_random_geometric(
    N: int,
    *,
    dim: int = 2,
    r: Optional[float] = None,   # connect if dist <= r
    k: Optional[int] = None,     # or use kNN
    periodic_box: Optional[float] = None,  # e.g., 1.0 for torus box
    directed: bool = False,
    seed: int = 0,
):
    assert (r is not None) ^ (k is not None), "Specify either r or k (exclusively)."
    rng = np.random.default_rng(seed)
    X = rng.random((N, dim)).astype(np.float32)

    if r is not None:
        u, v = _radius_edges(X, r, periodic_box)
    else:
        assert k is not None, "Specify k for kNN graph"
        u, v = _knn_edges(X, k, periodic_box)

    w = np.ones_like(u, dtype=np.float32)
    A = _csr_from_edges(u, v, w, N)
    if not directed:
        A = _symmetrize(A)
    return A, {"coords": X, "coords2d": X[:, :2].copy(), "metric": "periodic" if periodic_box else "euclidean", "dim": dim}

def build_small_world_watts_strogatz(n: int, k: int, beta: float, *,
                                     directed: bool = False,
                                     matrix_format: str = "csr",
                                     seed: int = 0):
    import numpy as np
    import scipy.sparse as sp

    assert 0 < k < n and k % 2 == 0, "k must be a positive even integer less than n"
    rng = np.random.default_rng(seed)
    nodes = np.arange(n)

    # 1) ring lattice: build one direction only (i -> i+s), s=1..k/2
    u_list, v_list = [], []
    for s in range(1, k // 2 + 1):
        u = nodes
        v = (nodes + s) % n
        u_list.append(u); v_list.append(v)
    u = np.concatenate(u_list); v = np.concatenate(v_list)
    m = len(u)  # = n * (k/2)

    # track undirected pairs to avoid duplicates after symmetrization
    def upair(a, b): return (a, b) if a < b else (b, a)
    pairs = {upair(int(u[i]), int(v[i])) for i in range(m)}

    # adjacency (directed) for fast neighbor checks from src
    # use LIL for efficient structural updates during rewiring
    Adir = sp.lil_matrix((n, n), dtype=np.int8)
    Adir[u, v] = 1

    # 2) rewire each directed edge with prob beta, avoiding duplicate undirected pairs
    for i in range(m):
        if rng.random() >= beta:
            continue
        src, old_tgt = int(u[i]), int(v[i])

        # remove old pair from the set (we're rewiring this edge)
        pairs.discard(upair(src, old_tgt))
        Adir[src, old_tgt] = 0  # mark removal in pattern

        # forbid: self, current out-neighbors of src, and any cand forming existing undirected pair
        forbid = set(int(x) for x in Adir.rows[src])
        forbid.add(src)

        # sample until we find an allowed candidate that doesn't create an existing undirected pair
        cand = rng.integers(0, n)
        tries = 0
        while (cand in forbid) or (upair(src, int(cand)) in pairs):
            cand = rng.integers(0, n)
            tries += 1
            if tries > 10 * n:
                # fall back to a deterministic scan (rare)
                for cand2 in range(n):
                    if cand2 not in forbid and upair(src, cand2) not in pairs:
                        cand = cand2
                        break
                else:
                    # if truly stuck, restore old edge and continue
                    cand = old_tgt
                    break

        # commit
        v[i] = int(cand)
        Adir[src, v[i]] = 1
        pairs.add(upair(src, v[i]))

    # 3) build final adjacency
    A = sp.coo_matrix((np.ones(m, np.float32), (u, v)), shape=(n, n))
    if not directed:
        A = (A + A.T)
        A.setdiag(0); A.eliminate_zeros()
        A = A.tocsr()
    else:
        A = A.tocsr()

    # coords for plotting
    theta = 2 * np.pi * np.arange(n) / n
    coords2d = np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)
    return (A if matrix_format == "csr" else A.tocoo()), {"coords2d": coords2d, "kind": "small_world"}



# ---------- 4. Structured “physics-y” graphs ----------
# Cylinders/strips: you already get these via your grid:
#   cylinder_x = build_grid(N, periodic_x=True, periodic_y=False)
#   cylinder_y = build_grid(N, periodic_x=False, periodic_y=True)

def build_sphere_discretization(
    n_points: int,
    *,
    connect: Literal["radius","knn"] = "radius",
    r: Optional[float] = None,
    k: Optional[int] = None,
    directed: bool = False,
    matrix_format: str = "csr",
    seed: int = 0,
):
    """
    Fibonacci sphere for points; edges via radius or kNN on chordal distance.
    """
    rng = np.random.default_rng(seed)
    # Fibonacci sphere
    i = np.arange(n_points)
    phi = (1 + np.sqrt(5)) / 2
    z = 1 - 2*(i + 0.5)/n_points
    theta = 2*np.pi * (i / phi % 1.0)
    r_xy = np.sqrt(1 - z*z)
    X = np.stack([r_xy*np.cos(theta), r_xy*np.sin(theta), z], axis=1).astype(np.float32)

    if connect == "radius":
        assert r is not None, "Specify r for radius graph"
        u, v = _radius_edges(X, r, periodic_box=None)
    else:
        assert k is not None, "Specify k for kNN graph"
        u, v = _knn_edges(X, k, periodic_box=None)

    w = np.ones_like(u, dtype=np.float32)
    A = _csr_from_edges(u, v, w, n_points)
    if not directed:
        A = _symmetrize(A)
    return A, {"coords3d": X, "coords2d": X[:, :2].copy(), "kind": "sphere"}

def build_tree_lattice(
    levels: int,
    *,
    branching: int = 3,
    directed: bool = False,     # if True: parent -> child
):
    """
    Perfect b-ary tree with 'levels' (root at level 0). N = (b^levels - 1)/(b-1).
    """
    b = branching
    assert levels >= 1 and b >= 2
    N = (b**levels - 1) // (b - 1)
    # parent index p -> children indices (p*b + 1 .. p*b + b) within a heap-like indexing
    u_list, v_list = [], []
    last_parent = (b**(levels-1) - 1) // (b - 1) - 1  # inclusive index of deepest parent
    last_parent = max(last_parent, 0)
    for p in range(0, (b**(levels-1) - 1) // (b - 1)):  # nodes before last level
        first_child = p*b + 1
        for t in range(b):
            c = first_child + t
            if c < N:
                if directed:
                    u_list.append(p); v_list.append(c)
                else:
                    u_list.append(p); v_list.append(c)
                    u_list.append(c); v_list.append(p)
    u = np.array(u_list, dtype=np.int64); v = np.array(v_list, dtype=np.int64)
    w = np.ones_like(u, dtype=np.float32)
    A = _csr_from_edges(u, v, w, N)

    # radial coords for plotting (simple polar fan per level)
    coords2d = np.zeros((N, 2), dtype=np.float32)
    idx = 0
    for L in range(levels):
        nL = b**L
        r = L
        ang = 2*np.pi*np.linspace(0, 1, nL, endpoint=False)
        for j in range(nL):
            if idx >= N: break
            coords2d[idx] = np.array([r*np.cos(ang[j]), r*np.sin(ang[j])], np.float32)
            idx += 1
    return A, {"coords2d": coords2d, "branching": b, "levels": levels, "kind": "tree"}

# ---------- 5. Heterogeneity / Multi-layer / Defects ----------
def reweight_grid_edges_by_coord(
    A: sp.csr_matrix,
    shape: Tuple[int,int],
    weight_fn: Callable[[int,int,int,int], float],  # (iu,ju,iv,jv) -> weight
    *,
    symmetric: bool = True
):
    """
    Given a grid adjacency and its (m,n) shape, assign per-edge weights via weight_fn.
    """
    m, n = shape
    assert A.shape is not None, "Adjacency must have shape"
    assert A.shape[0] == m*n and A.shape[1] == m*n, "Adjacency shape does not match provided grid shape"
    C = A.tocoo()
    iu, ju = np.divmod(C.row, n)
    iv, jv = np.divmod(C.col, n)
    w = np.array([weight_fn(int(a),int(b),int(c),int(d)) for a,b,c,d in zip(iu,ju,iv,jv)], dtype=np.float32)
    B = _csr_from_edges(C.row, C.col, w, A.shape[0])
    if symmetric:
        B = _symmetrize(B)
    return B

def build_multilayer(
    layers: Sequence[Tuple[sp.csr_matrix, Dict]],
    *,
    gamma: float = 0.1,   # interlayer coupling between corresponding nodes
    z_gap: float = 1.0,   # for plotting coords
    matrix_format: str = "csr",
):
    """
    Block-diagonal of layer adjacencies with interlayer identity couplings.
    Each layer must have same N.
    """
    if not layers:
        raise ValueError("layers must be non-empty")
    first_shape = layers[0][0].shape
    if first_shape is None:
        raise ValueError("Layers must have nonzero size")
    N = int(first_shape[0])
    for A, _meta in layers:
        shape = A.shape
        if shape is None:
            raise ValueError("Layers must have nonzero size")
        if shape[0] != N:
            raise ValueError("All layers must have same N")
    Lk = len(layers)
    # block diagonal
    A_block = sp.block_diag([L[0] for L in layers], format="coo")
    # interlayer edges between corresponding nodes (layer l -> l+1)
    rows, cols = [], []
    for l in range(Lk - 1):
        base = l*N
        nxt  = (l+1)*N
        rows.append(np.arange(N) + base)
        cols.append(np.arange(N) + nxt)
    if rows:
        rows = np.concatenate(rows); cols = np.concatenate(cols)
        data = np.full(rows.size, gamma, np.float32)
        inter = sp.coo_matrix((data, (rows, cols)), shape=A_block.shape)
        A_block = (A_block + inter + inter.T).tocsr()
    else:
        A_block = A_block.tocsr()
    # coords: stack with z offsets if available
    coords = None
    if all(("coords2d" in L[1]) for L in layers):
        coords = []
        for l, (_, meta) in enumerate(layers):
            xy = meta["coords2d"].astype(np.float32)
            z = np.full((N,1), l*z_gap, np.float32)
            coords.append(np.concatenate([xy, z], axis=1))
        coords = np.concatenate(coords, axis=0)
        meta_out = {"coords3d": coords, "layers": Lk, "interlayer_gamma": gamma}
    elif all(("coords3d" in L[1]) for L in layers):
        coords = []
        for l, (_, meta) in enumerate(layers):
            xyz = meta["coords3d"].astype(np.float32).copy()
            xyz[:,2] += l*z_gap
            coords.append(xyz)
        coords = np.concatenate(coords, axis=0)
        meta_out = {"coords3d": coords, "layers": Lk, "interlayer_gamma": gamma}
    else:
        meta_out = {"layers": Lk, "interlayer_gamma": gamma}
    return A_block if matrix_format=="csr" else A_block.tocoo(), meta_out

def remove_defects(
    A: sp.csr_matrix,
    *,
    remove_nodes: Optional[Sequence[int]] = None,
    remove_edges: Optional[Sequence[Tuple[int,int]]] = None,
) -> sp.csr_matrix:
    """
    Delete nodes/edges from any base graph (useful for pinning/removing).
    """
    B = A.tocoo().copy()
    if remove_edges:
        rm = set((int(u), int(v)) for (u,v) in remove_edges)
        keep = np.array([(int(uu),int(vv)) not in rm for uu,vv in zip(B.row, B.col)])
        B = sp.coo_matrix((B.data[keep], (B.row[keep], B.col[keep])), shape=A.shape)

    assert A.shape is not None, "Adjacency must have shape"
    if remove_nodes:
        rem = np.array(sorted(set(int(x) for x in remove_nodes)), dtype=np.int64)
        keep_nodes = np.ones(A.shape[0], dtype=bool); keep_nodes[rem] = False
        # filter edges
        keep_edge = keep_nodes[B.row] & keep_nodes[B.col]
        row = B.row[keep_edge]; col = B.col[keep_edge]; dat = B.data[keep_edge]
        # reindex nodes
        old_to_new = -np.ones(A.shape[0], dtype=np.int64)
        old_to_new[np.where(keep_nodes)[0]] = np.arange(keep_nodes.sum(), dtype=np.int64)
        row = old_to_new[row]; col = old_to_new[col]
        B = sp.coo_matrix((dat, (row, col)), shape=(keep_nodes.sum(), keep_nodes.sum()))
    return sp.csr_matrix(B)

# ---------- base grid (periodicity toggles -> line, cylinder, torus) ----------
def build_grid(
    N: int | None = None,
    *,
    m: int | None = None,
    n: int | None = None,
    w_vert: float = 1.0,
    w_horiz: float = 1.0,
    periodic_x: bool = False,
    periodic_y: bool = False,
    directed: bool = False,
):
    """
    Build an m*n 4-neighbor grid with optional wrap-around (cylinder/torus).
    Provide either:
      - N (if square → m=n=√N), or
      - m and n explicitly (rectangles allowed).
    Returns (A, meta) with meta['shape']=(m,n) and meta['coords2d'].
    """
    if N is not None:
        r = int(np.sqrt(N))
        if r * r != N:
            raise ValueError(f"N={N} not square; pass m and n explicitly.")
        m = n = r
    else:
        if m is None or n is None:
            raise ValueError("Provide either N (square) or both m and n.")
        N = int(m) * int(n)

    rows_parts, cols_parts, w_parts = [], [], []

    # Horizontal (j -> j+1)
    if n >= 2:
        i = np.repeat(np.arange(m), n - 1)
        j = np.tile(np.arange(n - 1), m)
        u = _idx(i, j, n); v = _idx(i, j + 1, n)
        rows_parts += [u]; cols_parts += [v]; w_parts += [np.full(u.size, w_horiz, np.float32)]
        if not directed:
            rows_parts += [v]; cols_parts += [u]; w_parts += [np.full(u.size, w_horiz, np.float32)]

    # Horizontal wrap (j=n-1 -> j=0)
    if periodic_x:
        i = np.arange(m)
        u = _idx(i, np.full_like(i, n - 1), n); v = _idx(i, np.zeros_like(i), n)
        rows_parts += [u]; cols_parts += [v]; w_parts += [np.full(u.size, w_horiz, np.float32)]
        if not directed:
            rows_parts += [v]; cols_parts += [u]; w_parts += [np.full(u.size, w_horiz, np.float32)]

    # Vertical (i -> i+1)
    if m >= 2:
        i = np.repeat(np.arange(m - 1), n)
        j = np.tile(np.arange(n), m - 1)
        u = _idx(i, j, n); v = _idx(i + 1, j, n)
        rows_parts += [u]; cols_parts += [v]; w_parts += [np.full(u.size, w_vert, np.float32)]
        if not directed:
            rows_parts += [v]; cols_parts += [u]; w_parts += [np.full(u.size, w_vert, np.float32)]

    # Vertical wrap (i=m-1 -> i=0)
    if periodic_y:
        j = np.arange(n)
        u = _idx(np.full_like(j, m - 1), j, n); v = _idx(np.zeros_like(j), j, n)
        rows_parts += [u]; cols_parts += [v]; w_parts += [np.full(u.size, w_vert, np.float32)]
        if not directed:
            rows_parts += [v]; cols_parts += [u]; w_parts += [np.full(u.size, w_vert, np.float32)]

    if not rows_parts:
        A = sp.csr_matrix((N, N), dtype=np.float32)
        return A, {"shape": (m, n)}

    rows = np.concatenate(rows_parts).astype(np.int64, copy=False)
    cols = np.concatenate(cols_parts).astype(np.int64, copy=False)
    data = np.concatenate(w_parts, dtype=np.float32)
    A = _assemble_sparse(rows, cols, data, N)
    # 2D integer grid coordinates for plotting
    I, J = np.indices((m, n))
    xy = np.stack([J.ravel(), I.ravel()], axis=1).astype(np.float32)  # x=j, y=i
    return A, {"shape": (m, n), "coords2d": xy}

# ---------- torus surface (true 3D embedding of C_m × C_n) ----------
def build_torus_surface(
    N: int | None = None,
    *,
    m: int | None = None,
    n: int | None = None,
    w_vert: float = 1.0,
    w_horiz: float = 1.0,
    R: float = 3.0,
    r0: float = 1.0,
    directed: bool = False,
    matrix_format: str = "csr",
):
    """
    Torus topology = periodic grid in both axes, with 3D torus embedding.
    Accepts N (square) or explicit (m,n) for rectangles.
    """
    if N is not None:
        r = int(np.sqrt(N))
        if r * r != N:
            raise ValueError(f"N={N} not square; pass m and n.")
        m = n = r
    else:
        if m is None or n is None:
            raise ValueError("Provide either N (square) or both m and n.")
        N = int(m) * int(n)

    # 1) topology from periodic grid
    A, _ = build_grid(
        None, m=m, n=n,
        w_vert=w_vert, w_horiz=w_horiz,
        periodic_x=True, periodic_y=True,
        directed=directed
    )

    # 2) 3D parametric embedding (rectangles OK)
    # use meshgrid to ensure shapes are (m,n)
    u = 2.0 * np.pi * (np.arange(m) / m)
    v = 2.0 * np.pi * (np.arange(n) / n)
    U, V = np.meshgrid(u, v, indexing="ij")

    X = (R + r0 * np.cos(V)) * np.cos(U)
    Y = (R + r0 * np.cos(V)) * np.sin(U)
    Z = r0 * np.sin(V)

    coords3d = np.stack([X, Y, Z], axis=-1).reshape(N, 3).astype(np.float32)
    return A, {"shape": (m, n), "coords3d": coords3d}


# ---------- grid with a gate (a wall w/ one opening) ----------
def build_grid_with_gate(
    N: int | None = None,
    *,
    m: int | None = None,
    n: int | None = None,
    gate_axis: str = "vertical",   # "vertical" blocks horizontal edges across a column; gate spans rows
    wall_index: int | None = None, # column (vertical) or row (horizontal) where the wall sits (between wall_index-1 and wall_index)
    gate_pos: int | None = None,   # single gate position (alias for single-row/col gate)
    gate_start: int | None = None, # start of gate span along the perpendicular axis (inclusive)
    gate_end: int | None = None,   # end of gate span along the perpendicular axis (inclusive)
    w_vert: float = 1.0,
    w_horiz: float = 1.0,
    periodic_x: bool = False,
    periodic_y: bool = False,
    directed: bool = False,
    matrix_format: str = "csr",
):
    """
    Build an m×n grid with a solid wall that blocks adjacency across it,
    except for a single gate segment kept open.

    Parameters
    ----------
    N : int, optional
        Total number of nodes (must be a perfect square → m=n=√N).
    m, n : int, optional
        Grid dimensions. Provide instead of N for rectangles.
    gate_axis : {"vertical","horizontal"}
        Orientation of the wall.
        - "vertical": wall splits columns (blocks horizontal edges); gate spans rows.
        - "horizontal": wall splits rows (blocks vertical edges); gate spans columns.
    wall_index : int
        The *right* side index of the seam (between wall_index-1 and wall_index).
        If the corresponding axis is periodic, `wall_index=0` means the wrap seam
        between the last index and 0.
    gate_start, gate_end : int, optional
        Inclusive span along the perpendicular axis to keep open.
        For vertical wall → rows in [gate_start, gate_end] are open.
        For horizontal wall → cols in [gate_start, gate_end] are open.

    Returns
    -------
    A_gate : scipy.sparse.csr_matrix
        Adjacency with the wall removed except along the gate span.
    meta : dict
        Includes 'shape', 'coords2d' (from base grid, if present), and 'gate' info.
    """
    # ---- resolve shape ----
    if N is not None:
        r = int(np.sqrt(N))
        if r * r != N:
            raise ValueError(f"N={N} not square; pass m and n instead.")
        m = n = r
    else:
        if m is None or n is None:
            raise ValueError("Provide either N (square) or both m and n.")
        N = int(m) * int(n)

    # ---- base grid ----
    A, meta = build_grid(
        None, m=m, n=n,
        w_vert=w_vert,
        w_horiz=w_horiz,
        periodic_x=periodic_x,
        periodic_y=periodic_y,
        directed=directed,
    )

    # ---- defaults for wall & gate ----
    if wall_index is None:
        wall_index = (n // 3) if gate_axis == "vertical" else (m // 3)

    # allow `gate_pos` as a convenient alias for a single-row/col gate
    if gate_pos is not None:
        gate_start = gate_end = int(gate_pos)

    # prefer span if provided; otherwise fall back to a default single gate span
    if gate_start is None or gate_end is None:
        if gate_start is None and gate_end is None:
            gate_start = gate_end = (m // 9) * 4 if gate_axis == "vertical" else (n // 9) * 4
            if gate_axis == "vertical":
                gate_end += (m // 9)
            else:
                gate_end += (n // 9)
        elif gate_start is None:
            gate_start = gate_end
        else:
            gate_end = gate_start

    assert gate_start is not None and gate_end is not None, "gate_start and gate_end must be specified"
    if gate_axis == "vertical":
        lo, hi = max(0, min(gate_start, gate_end)), min(m - 1, max(gate_start, gate_end))
    elif gate_axis == "horizontal":
        lo, hi = max(0, min(gate_start, gate_end)), min(n - 1, max(gate_start, gate_end))
    else:
        raise ValueError("gate_axis must be 'vertical' or 'horizontal'")

    # ---- find and drop edges crossing the wall outside the gate span ----
    Acoo = A.tocoo(copy=True)
    u, v = Acoo.row, Acoo.col
    iu, ju = np.divmod(u, n)
    iv, jv = np.divmod(v, n)

    if gate_axis == "vertical":
        # columns seam: between col (left) and col (right)
        left  = (wall_index - 1) % n if periodic_x else (wall_index - 1)
        right = (wall_index    ) % n if periodic_x else (wall_index    )

        crosses = (iu == iv) & (
            ((ju == left)  & (jv == right)) |
            ((ju == right) & (jv == left))
        )
        # allowed rows: gate span [lo..hi]
        allowed = (iu >= lo) & (iu <= hi)

    else:  # "horizontal"
        # rows seam: between row (top) and row (bottom)
        top    = (wall_index - 1) % m if periodic_y else (wall_index - 1)
        bottom = (wall_index    ) % m if periodic_y else (wall_index    )

        crosses = (ju == jv) & (
            ((iu == top)    & (iv == bottom)) |
            ((iu == bottom) & (iv == top))
        )
        # allowed columns: gate span [lo..hi]
        allowed = (ju >= lo) & (ju <= hi)

    drop = crosses & (~allowed)
    keep = ~drop

    A_gate = sp.coo_matrix((Acoo.data[keep], (u[keep], v[keep])), shape=A.shape)
    if matrix_format == "csr":
        A_gate = A_gate.tocsr()

    meta = {
        **meta,
        "shape": (m, n),
        "gate": {
            "axis": gate_axis,
            "wall_index": int(wall_index),
            "span": (int(lo), int(hi)),
            "periodic_x": bool(periodic_x),
            "periodic_y": bool(periodic_y),
        },
    }
    return A_gate, meta



# ---------- convenience wrappers that reuse existing builders ----------

def build_multilayer_from(
    base_kind: str,
    *,
    layers: int = 2,
    gamma: float = 0.1,
    z_gap: float = 1.0,
    layer_kwargs: list[dict] | None = None,
    **common_kw
):
    """
    Build L identical (or varied) layers via _BUILDERS[base_kind] and couple them.
    layer_kwargs: optional list of per-layer kwargs dicts (length==layers).
    common_kw: kwargs applied to each layer builder.
    """
    if layer_kwargs is None:
        layer_kwargs = [{} for _ in range(layers)]
    assert len(layer_kwargs) == layers, "layer_kwargs must match 'layers' length"

    built = []
    for lk in layer_kwargs:
        A, meta = _BUILDERS[base_kind].build(**{**common_kw, **lk})
        built.append((A, meta))

    return build_multilayer(built, gamma=gamma, z_gap=z_gap)

def build_defects_from(
    base_kind: str,
    *,
    remove_nodes: list[int] | None = None,
    remove_edges: list[tuple[int, int]] | None = None,
    **base_kw
):
    """
    Build a base graph via _BUILDERS[base_kind], then remove nodes/edges.
    """
    A, meta = _BUILDERS[base_kind].build(**base_kw)
    B = remove_defects(A, remove_nodes=remove_nodes, remove_edges=remove_edges)
    return B, meta

def build_weighted_grid(
    N: int | None = None,
    *,
    m: int | None = None,
    n: int | None = None,
    weight_fn,                  # (iu, ju, iv, jv) -> float
    symmetric: bool = True,
    **grid_kw
):
    """
    Build a grid via build_grid, then assign heterogeneous per-edge weights.
    Accepts either N (square) or explicit (m, n).
    """
    # resolve shape
    if N is not None:
        r = int(np.sqrt(N))
        if r * r != N:
            raise ValueError(f"N={N} not square; pass m and n instead.")
        m = n = r
    else:
        if m is None or n is None:
            raise ValueError("Provide either N (square) or both m and n.")
        N = int(m) * int(n)

    # base grid
    A, meta = build_grid(None, m=m, n=n, **grid_kw)
    shape = meta["shape"]
    B = reweight_grid_edges_by_coord(A, shape, weight_fn, symmetric=symmetric)
    return B, meta

# ---------- your expanded registry ----------

_BUILDERS: Dict[str, GraphBuilder] = {
    # Base lattices / periodic variants
    "grid":            FunctionBuilder(lambda **kw: build_grid(**kw)),
    "cylinder_x":      FunctionBuilder(lambda **kw: build_grid(periodic_x=True,  periodic_y=False, **kw)),
    "cylinder_y":      FunctionBuilder(lambda **kw: build_grid(periodic_x=False, periodic_y=True,  **kw)),
    "torus_grid":      FunctionBuilder(lambda **kw: build_grid(periodic_x=True,  periodic_y=True,  **kw)),

    # Thin wrappers (reuse grid connectivity; add embedding or edits)
    "torus_surface":   FunctionBuilder(lambda **kw: build_torus_surface(**kw)),
    "gate":            FunctionBuilder(lambda **kw: build_grid_with_gate(**kw)),
    "weighted_grid":   FunctionBuilder(lambda weight_fn=None, **kw: build_weighted_grid(weight_fn=weight_fn, **kw)),

    # 3. Random graphs with geometry
    #    Use either radius or kNN; provide both names for clarity.
    "rgg_radius":      FunctionBuilder(lambda dim=2, r=0.1, **kw: build_random_geometric(dim=dim, r=r, k=None, **kw)),
    "rgg_knn":         FunctionBuilder(lambda dim=2, k=8,   **kw: build_random_geometric(dim=dim, r=None, k=k, **kw)),

    # Small-world (Watts–Strogatz)
    "small_world":     FunctionBuilder(lambda n, k, beta, **kw: build_small_world_watts_strogatz(n=n, k=k, beta=beta, **kw)),

    # 4. Structured “physics-y” graphs
    "sphere_radius":   FunctionBuilder(lambda n_points, r, **kw: build_sphere_discretization(n_points, connect="radius", r=r, **kw)),
    "sphere_knn":      FunctionBuilder(lambda n_points, k, **kw: build_sphere_discretization(n_points, connect="knn",    k=k, **kw)),
    "tree":            FunctionBuilder(lambda levels, **kw: build_tree_lattice(levels=levels, **kw)),

    # 5. Heterogeneity / Multi-layer / Defects (compositional)
    "multilayer_from": FunctionBuilder(lambda base_kind, **kw: build_multilayer_from(base_kind, **kw)),
    "defects_from":    FunctionBuilder(lambda base_kind, **kw: build_defects_from(base_kind, **kw)),

    # 6. Canonical non-grid but highly visualizable
    "ring": FunctionBuilder(lambda n, **kw: build_ring(n=n, **kw)),
    "ring_chords": FunctionBuilder(lambda n, **kw: build_ring_chords(n=n, **kw)),
    "wheel": FunctionBuilder(lambda n, **kw: build_wheel(n=n, **kw)),
    "sbm": FunctionBuilder(lambda **kw: build_sbm(**kw)),
    "directed_ring": FunctionBuilder(lambda n, **kw: build_directed_ring(n=n, **kw)),
}


def build_graph(kind: str, /, **kwargs) -> Tuple[sp.csr_matrix, Dict]:
    """
    Construct a graph of a given kind by dispatching to the appropriate builder.

    Parameters
    ----------
    kind : str
        Key identifying the graph type (must exist in `_BUILDERS`).
    **kwargs
        Arguments passed through to the underlying builder.

    Returns
    -------
    A : scipy.sparse matrix
        Graph adjacency matrix.
    meta : dict
        Metadata (e.g., shape, coordinates, parameters).
    """
    if kind not in _BUILDERS:
        raise ValueError(f"Unknown kind='{kind}'. Available: {sorted(_BUILDERS)}")
    builder = _BUILDERS[kind]
    A, meta = builder.build(**kwargs)
    assert A is not None, "builder produced None adjacency"

    meta = _finalize_meta(A, meta, kind)
    return A, meta

# ---------- minimal plotting helpers ----------
def plot_graph_2d(
    A: sp.csr_matrix,
    xy: np.ndarray,
    *,
    ax=None,
    node_size: float = 2,
    lw: float = 0.5,
):
    """
    Plot a 2D graph embedding with nodes and undirected edges.

    Parameters
    ----------
    A : sp.csr_matrix
        Adjacency matrix (undirected or directed; edges drawn once for u < v).
    xy : np.ndarray
        Node coordinates, shape (N, 2).
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw into; if None, creates a new figure/axes.
    node_size : float
        Marker size for nodes.
    lw : float
        Line width for edges.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    ax.scatter(xy[:, 0], xy[:, 1], s=node_size)
    A_coo = A.tocoo()
    for u, v in zip(A_coo.row, A_coo.col):
        if u < v:  # draw each undirected edge once
            x = [xy[u, 0], xy[v, 0]]
            y = [xy[u, 1], xy[v, 1]]
            ax.plot(x, y, linewidth=lw)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return ax
