import numpy as np
import scipy.sparse as sp
import pytest

# 👇 Adjust these imports to your actual module names
from interactionfields.graphs import build_graph
from interactionfields.simulate import (
    SIMULATORS,
    run_simulator,
    simulate_faucet_on_graph,
    simulate_waves_on_graph,
)

def _pick_center(meta, default=0):
    if "shape" in meta:
        m, n = meta["shape"]
        return (m // 2) * n + (n // 2)
    return default

def _make_small_grid(nside=6):
    # Use your builder for a small undirected grid
    A, meta = build_graph("grid", m=nside, n=nside)
    return A.tocsr(), meta

def _bfs_hops_csr(adj: sp.csr_matrix, src: int) -> np.ndarray:
    # Minimal BFS used locally for an orientation check
    from collections import deque
    N = adj.shape[0]
    dist = np.full(N, np.iinfo(np.int32).max, dtype=np.int32)
    dist[src] = 0
    q = deque([src])
    indptr, indices = adj.indptr, adj.indices
    while q:
        u = q.popleft()
        for v in indices[indptr[u]:indptr[u+1]]:
            if dist[v] == np.iinfo(np.int32).max:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

# ---------------- Registry & dispatcher ----------------

def test_simulator_registry_has_entries():
    assert "faucet" in SIMULATORS
    assert "waves" in SIMULATORS
    assert callable(SIMULATORS["faucet"])
    assert callable(SIMULATORS["waves"])

def test_run_simulator_unknown_raises():
    with pytest.raises(ValueError):
        run_simulator("nope", adj=sp.csr_matrix((4,4)), t_bins=1)

# ---------------- Faucet simulator ----------------

def test_faucet_bins_only_shapes_and_types():
    A, meta = _make_small_grid(6)
    center = (meta["shape"][0] // 2) * meta["shape"][1] + (meta["shape"][1] // 2)
    T = 50
    bins = run_simulator(
        "faucet", adj=A, t_bins=T, center_idx=center, faucet_period=10, return_states=False, seed=123
    )
    assert isinstance(bins, list) and len(bins) == T
    for B in bins:
        assert isinstance(B, sp.csr_matrix)
        assert B.shape == A.shape
        assert B.dtype == np.uint8

def test_faucet_with_states_history_shape():
    A, meta = _make_small_grid(5)
    center = (meta["shape"][0] // 2) * meta["shape"][1] + (meta["shape"][1] // 2)
    T = 40
    bins, H = simulate_faucet_on_graph(
        adj=A, t_bins=T, center_idx=center, faucet_period=8, return_states=True, seed=0
    )
    assert isinstance(bins, list) and len(bins) == T
    assert isinstance(H, np.ndarray) and H.shape == (T, A.shape[0])

def test_faucet_edges_oriented_outward_when_present():
    A, meta = _make_small_grid(6)
    n = A.shape[0]
    center = (meta["shape"][0] // 2) * meta["shape"][1] + (meta["shape"][1] // 2)
    dist = _bfs_hops_csr(A, center)
    T = 60
    bins = simulate_faucet_on_graph(
        adj=A, t_bins=T, center_idx=center, faucet_period=6, seed=1, return_states=False
    )
    # All nonzero entries (u,v) in any bin should satisfy dist[u] < dist[v]
    for B in bins:
        B = B.tocoo()
        for u, v in zip(B.row, B.col):
            assert dist[u] < dist[v]

def test_faucet_seeding_is_deterministic():
    A, meta = _make_small_grid(5)
    center = (meta["shape"][0] // 2) * meta["shape"][1] + (meta["shape"][1] // 2)
    bins1 = simulate_faucet_on_graph(adj=A, t_bins=30, center_idx=center, seed=7)
    bins2 = simulate_faucet_on_graph(adj=A, t_bins=30, center_idx=center, seed=7)
    # Compare nonzero patterns per bin
    for B1, B2 in zip(bins1, bins2):
        assert (B1 != B2).nnz == 0

def test_faucet_stability_warning(capsys):
    A, meta = _make_small_grid(6)
    center = (meta["shape"][0] // 2) * meta["shape"][1] + (meta["shape"][1] // 2)
    # Choose c*dt > 1.5 to trigger warning (only when return_states=True)
    simulate_faucet_on_graph(adj=A, t_bins=2, center_idx=center, c=2.0, dt=1.0,
                             return_states=True, seed=0)
    captured = capsys.readouterr().out
    assert "c*dt" in captured and "warn" in captured.lower()

# ---------------- Waves simulator ----------------

def test_waves_bins_sparse_and_uint8():
    A, _ = _make_small_grid(6)
    T = 40
    bins = run_simulator("waves", adj=A, t_bins=T, seed=123, dt=0.25)
    assert len(bins) == T
    # At least one bin should have some activity, and dtype/shape correct
    nnzs = [b.nnz for b in bins]
    assert any(nz > 0 for nz in nnzs)
    for B in bins:
        assert isinstance(B, sp.csr_matrix)
        assert B.dtype == np.uint8
        assert B.shape == A.shape

def test_waves_seeding_deterministic():
    A, _ = _make_small_grid(6)
    T = 30
    bins1 = simulate_waves_on_graph(adj=A, t_bins=T, seed=5)
    bins2 = simulate_waves_on_graph(adj=A, t_bins=T, seed=5)
    for B1, B2 in zip(bins1, bins2):
        assert (B1 != B2).nnz == 0

# ---------------- Tiny edge cases ----------------

def test_empty_graph_yields_empty_bins():
    A = sp.csr_matrix((4, 4), dtype=np.float32)
    bins = simulate_waves_on_graph(adj=A, t_bins=5, seed=0)
    for B in bins:
        assert B.nnz == 0
        assert B.shape == (4, 4)

def test_single_node_graph():
    A = sp.csr_matrix((1, 1), dtype=np.float32)
    bins = simulate_waves_on_graph(adj=A, t_bins=3, seed=0)
    assert len(bins) == 3
    assert all(B.nnz == 0 for B in bins)


# --- PARAMS ----------------------------------------------------------------------
# Keep sizes small for speed; choose settings that produce some activity.
GRAPH_CASES = [
    ("torus_surface", dict(m=8, n=10), True),                       # outward orientation meaningful
    ("cylinder_x",    dict(m=8, n=12), True),
    ("rgg_radius",    dict(N=300, dim=2, r=0.16, seed=3), True),
    ("sphere_knn",    dict(n_points=512, k=8, seed=1), True),
    ("tree",          dict(levels=5, branching=3), True),
    # multilayer (two small 6x6 grids)
    ("multilayer_from", dict(
        base_kind="grid",
        layers=2,
        gamma=0.05,
        z_gap=1.0,
        layer_kwargs=[{"m":6, "n":6}, {"m":6, "n":6}],
    ), False),  # orientation still works, but we'll skip it to avoid layer offset confusion
    # defects: 12x12 grid with a 3x3 hole; center selection ambiguous after reindex → skip orientation
    ("defects_from", dict(
        base_kind="grid",
        m=12, n=12,
        remove_nodes=[(i*12 + j) for i in range(5,8) for j in range(5,8)]
    ), False),
]

# --- TESTS -----------------------------------------------------------------------

@pytest.mark.parametrize("kind, kwargs, check_orient", GRAPH_CASES)
def test_faucet_runs_and_shapes(kind, kwargs, check_orient):
    A, meta = build_graph(kind, **kwargs)
    A = A.tocsr()
    n = A.shape[0]
    # center: use grid-center when available; else 0 (ok for geometric/sphere/tree)
    center = _pick_center(meta, default=0)
    T = 40

    bins = run_simulator("faucet", adj=A, t_bins=T,
                         center_idx=center, faucet_period=max(6, T//8), seed=7)
    assert isinstance(bins, list) and len(bins) == T
    for B in bins:
        assert isinstance(B, sp.csr_matrix)
        assert B.dtype == np.uint8
        assert B.shape == (n, n)

    if check_orient:
        # All nonzero (u,v) must satisfy dist[u] < dist[v] (outward)
        dist = _bfs_hops_csr(A, center)
        for B in bins:
            C = B.tocoo()
            if C.nnz == 0:
                continue
            assert np.all(dist[C.row] < dist[C.col])

def test_faucet_seed_determinism_all_graphs():
    for kind, kwargs, _ in GRAPH_CASES:
        A, meta = build_graph(kind, **kwargs)
        center = _pick_center(meta, default=0)
        T = 25
        bins1 = simulate_faucet_on_graph(adj=A, t_bins=T, center_idx=center, seed=11)
        bins2 = simulate_faucet_on_graph(adj=A, t_bins=T, center_idx=center, seed=11)
        for B1, B2 in zip(bins1, bins2):
            assert (B1 != B2).nnz == 0

@pytest.mark.parametrize("kind, kwargs, _", GRAPH_CASES)
def test_waves_runs_and_has_activity(kind, kwargs, _):
    A, _meta = build_graph(kind, **kwargs)
    T = 50
    bins = run_simulator("waves", adj=A, t_bins=T, seed=5)
    assert isinstance(bins, list) and len(bins) == T
    # shapes/dtypes and at least one active bin
    any_nz = False
    for B in bins:
        assert isinstance(B, sp.csr_matrix)
        assert B.dtype == np.uint8
        assert B.shape == A.shape
        any_nz = any_nz or (B.nnz > 0)
    assert any_nz, f"waves produced no activations for {kind}"

def test_waves_seed_determinism_all_graphs():
    for kind, kwargs, _ in GRAPH_CASES:
        A, _ = build_graph(kind, **kwargs)
        T = 30
        bins1 = simulate_waves_on_graph(adj=A, t_bins=T, seed=123)
        bins2 = simulate_waves_on_graph(adj=A, t_bins=T, seed=123)
        for B1, B2 in zip(bins1, bins2):
            assert (B1 != B2).nnz == 0

def test_multilayer_shape_and_determinism():
    # explicit test to ensure multilayer behaves well
    A, meta = build_graph("multilayer_from",
                          base_kind="grid",
                          layers=2,
                          gamma=0.05,
                          z_gap=1.0,
                          layer_kwargs=[{"m":6,"n":6}, {"m":6,"n":6}])
    T = 20
    bins1 = run_simulator("waves", adj=A, t_bins=T, seed=9)
    bins2 = run_simulator("waves", adj=A, t_bins=T, seed=9)
    assert A.shape == (72, 72)  # 2 * 6*6
    for B1, B2 in zip(bins1, bins2):
        assert (B1 != B2).nnz == 0
