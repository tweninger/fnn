# test_builders.py
import math
import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")  # headless rendering for CI
import matplotlib.pyplot as plt
import pytest

# Adjust this import to your actual module (e.g., from graphs import ...)
from interactionfields.graphs import (
    build_graph,
    _BUILDERS,
    plot_graph_2d,
)

def deg(A: sp.csr_matrix) -> np.ndarray:
    return np.asarray(A.sum(axis=1)).ravel()

def is_same_adj(A: sp.csr_matrix, B: sp.csr_matrix) -> bool:
    A = A.tocsr(); B = B.tocsr()
    if A.shape != B.shape:
        return False
    return (A - B).nnz == 0

def test_grid_degrees_small_square():
    # 4x4 grid (no wrap)
    N = 4 * 4
    A, meta = build_graph("grid", N=N)
    d = deg(A)

    # corners=4 with degree 2, edges (non-corner)=8 with degree 3, interior=4 with degree 4
    assert (d == 2).sum() == 4
    assert (d == 3).sum() == 8
    assert (d == 4).sum() == 4

    # coords2d present for plotting
    assert "coords2d" in meta and meta["coords2d"].shape == (N, 2)

def test_torus_topology_equals_periodic_grid():
    # torus adjacency should match fully periodic grid
    m = n = 8
    N = m * n
    A_grid, _ = build_graph("torus_grid", N=N)
    A_torus, meta = build_graph("torus_surface", N=N, R=3.0, r0=1.0)

    assert is_same_adj(A_grid, A_torus)
    # torus adds coords3d for plotting
    assert "coords3d" in meta and meta["coords3d"].shape == (N, 3)

def test_gate_blocks_all_but_one():
    # 6x6 grid, vertical wall at column 3 → wall between col 2|3, gate at row 2
    m = n = 6
    N = m * n
    wall_index = 3
    gate_row = 2
    A, meta = build_graph("gate", N=N, gate_axis="vertical", wall_index=wall_index, gate_pos=gate_row)

    def idx(i, j): return i * n + j

    # Check all rows except gate row have NO crossing edge between (i,2) and (i,3)
    for i in range(m):
        u = idx(i, wall_index - 1)
        v = idx(i, wall_index)
        if i == gate_row:
            assert A[u, v] != 0 and A[v, u] != 0
        else:
            assert A[u, v] == 0 and A[v, u] == 0

def test_random_geometric_knn_min_degree():
    # kNN (undirected symmetrized), degree should be at least k for all nodes
    N = 200
    k = 6
    A, meta = build_graph("rgg_knn", N=N, dim=2, k=k, seed=7)
    d = deg(A)
    assert d.min() >= k
    assert "coords" in meta and meta["coords"].shape == (N, 2)

def test_random_geometric_radius_nonempty():
    # radius graph should create some edges but not be fully disconnected
    N = 200
    A, _ = build_graph("rgg_radius", N=N, dim=2, r=0.15, seed=3)
    assert A.nnz > 0
    # should not be complete
    assert A.nnz < N * (N - 1)

def test_small_world_watts_strogatz_degree_stats():
    n, k, beta = 100, 10, 0.2
    A, meta = build_graph("small_world", n=n, k=k, beta=beta, seed=1)
    A = A.tocsr()
    d = np.asarray(A.sum(axis=1)).ravel()

    # unique undirected edges = A.nnz // 2
    assert A.nnz % 2 == 0
    assert (A.nnz // 2) == (n * k // 2)

    # degrees vary, but mean ≈ k
    assert np.isclose(d.mean(), k, atol=0.25)

    # sanity bounds
    assert d.min() >= 1
    assert d.max() <= min(n - 1, k + 5)

    assert "coords2d" in meta and meta["coords2d"].shape == (n, 2)



def test_sphere_points_on_unit_sphere():
    npts = 1024
    A, meta = build_graph("sphere_knn", n_points=npts, k=8, seed=5)
    X = meta["coords3d"]
    norms = np.linalg.norm(X, axis=1)
    assert np.allclose(norms.mean(), 1.0, atol=5e-3)
    assert X.shape == (npts, 3)
    assert A.nnz > 0

def test_tree_lattice_size_and_structure():
    levels = 4
    b = 3
    A, meta = build_graph("tree", levels=levels, branching=b, directed=False)
    N_expected = (b**levels - 1) // (b - 1)
    assert A.shape == (N_expected, N_expected)
    assert "coords2d" in meta and meta["coords2d"].shape == (N_expected, 2)
    # root should have degree = branching
    d = deg(A)
    assert d[0] == b

def test_multilayer_from_grid_block_structure():
    # two 3x3 layers (N=9 each) + interlayer couplings
    N = 3 * 3
    A, meta = build_graph(
        "multilayer_from",
        base_kind="grid",
        layers=2,
        gamma=0.5,
        z_gap=1.0,
        layer_kwargs=[{"N": N}, {"N": N}],
    )
    assert A.shape == (2 * N, 2 * N)
    # Interlayer block (top-right and bottom-left) should be nonzero
    A = A.tocsr()
    top_right = A[:N, N:]
    bottom_left = A[N:, :N]
    assert top_right.nnz > 0 and bottom_left.nnz > 0
    # coords3d should be stacked (if available)
    assert "coords3d" in meta and meta["coords3d"].shape[0] == 2 * N

def test_defects_from_removes_nodes():
    N = 8 * 8
    # remove a 2x2 hole in middle
    def idx(i, j): return i * 8 + j
    hole = [idx(3,3), idx(3,4), idx(4,3), idx(4,4)]
    A_def, meta = build_graph("defects_from", base_kind="grid", N=N, remove_nodes=hole)
    assert A_def.shape == (N - len(hole), N - len(hole))

def test_weighted_grid_edge_weights():
    # 5x5 grid; set horizontal edges=2.0, vertical edges=1.0 via coord-based rule
    N = 5 * 5
    def wfn(iu, ju, iv, jv):
        return 2.0 if (iu == iv) else 1.0  # same row → horizontal
    A_w, meta = build_graph("weighted_grid", N=N, weight_fn=wfn, symmetric=False)
    m, n = meta["shape"]
    C = A_w.tocoo()
    iu, ju = np.divmod(C.row, n)
    iv, jv = np.divmod(C.col, n)
    horiz = (iu == iv)
    if C.data.size:
        assert np.allclose(C.data[horiz], 2.0)
        assert np.allclose(C.data[~horiz], 1.0)

def test_plot_graph_2d_runs():
    N = 6 * 6
    A, meta = build_graph("grid", N=N)
    fig, ax = plt.subplots()
    ax = plot_graph_2d(A, meta["coords2d"], ax=ax, node_size=5, lw=0.5)
    assert ax is not None
    plt.close(fig)


def test_rectangular_torus_surface_matches_torus_grid():
    m, n = 10, 12
    A_g, _ = build_graph("torus_grid", m=m, n=n)
    A_s, meta = build_graph("torus_surface", m=m, n=n, R=3.0, r0=1.0)
    assert is_same_adj(A_g, A_s)
    assert meta["coords3d"].shape == (m*n, 3)

def test_cylinder_periodicity_only_one_axis():
    m, n = 6, 8
    A_cx, _ = build_graph("cylinder_x", m=m, n=n)
    A_cy, _ = build_graph("cylinder_y", m=m, n=n)
    # check wrap along x (columns): (i,n-1) connects to (i,0)
    def idx(i,j): return i*n + j
    for i in range(m):
        assert A_cx[idx(i, n-1), idx(i, 0)] != 0
        assert A_cy[idx(i, n-1), idx(i, 0)] == 0
    # check wrap along y (rows): (m-1,j) connects to (0,j)
    for j in range(n):
        assert A_cy[idx(m-1, j), idx(0, j)] != 0
        assert A_cx[idx(m-1, j), idx(0, j)] == 0

# ========== GATE + PERIODIC ==========

@pytest.mark.parametrize("axis", ["vertical", "horizontal"])
def test_gate_with_periodicity_blocks_all_but_one(axis):
    m = n = 8
    wall_index = 4
    gate_pos = 3
    # allow periodicity; gate logic should still hold across the chosen seam
    A, _ = build_graph("gate", m=m, n=n, gate_axis=axis,
                       wall_index=wall_index, gate_pos=gate_pos,
                       periodic_x=True, periodic_y=True)

    def idx(i,j): return i*n + j
    if axis == "vertical":
        # seam between col 3|4
        for i in range(m):
            u, v = idx(i, wall_index-1), idx(i, wall_index)
            if i == gate_pos:
                assert A[u, v] != 0 and A[v, u] != 0
            else:
                assert A[u, v] == 0 and A[v, u] == 0
    else:
        # seam between row 3|4
        for j in range(n):
            u, v = idx(wall_index-1, j), idx(wall_index, j)
            if j == gate_pos:
                assert A[u, v] != 0 and A[v, u] != 0
            else:
                assert A[u, v] == 0 and A[v, u] == 0

# ========== WEIGHTING & SYMMETRIZATION ==========

def test_weighted_grid_no_double_count_when_symmetric_false():
    m = n = 5
    def wfn(iu, ju, iv, jv):
        return 2.0 if iu == iv else 1.0
    A, meta = build_graph("weighted_grid", m=m, n=n, weight_fn=wfn, symmetric=False)
    C = A.tocoo()
    iu, ju = np.divmod(C.row, n)
    iv, jv = np.divmod(C.col, n)
    horiz = (iu == iv)
    if C.data.size:
        assert np.allclose(C.data[horiz], 2.0)
        assert np.allclose(C.data[~horiz], 1.0)

def test_weighted_grid_double_count_when_symmetric_sum():
    m = n = 5
    def wfn(iu, ju, iv, jv):
        return 2.0 if iu == iv else 1.0
    # assuming your API supports symmetric="sum" or True
    A, meta = build_graph("weighted_grid", m=m, n=n, weight_fn=wfn, symmetric=True)
    C = A.tocoo()
    iu, ju = np.divmod(C.row, n)
    iv, jv = np.divmod(C.col, n)
    horiz = (iu == iv)
    if C.data.size:
        assert np.allclose(C.data[horiz], 4.0)  # 2+2
        assert np.allclose(C.data[~horiz], 2.0) # 1+1

# ========== WATTS–STROGATZ EXTREMES & EDGE COUNT ==========

def test_small_world_beta_zero_is_ring_lattice():
    n, k, beta = 60, 8, 0.0
    A, _ = build_graph("small_world", n=n, k=k, beta=beta, seed=42)
    d = deg(A)
    # all degrees equal k; edge count exactly n*k/2
    assert np.all(d == k)
    assert (A.nnz // 2) == (n * k // 2)

def test_small_world_beta_one_edge_count_and_mean_degree():
    n, k, beta = 200, 12, 1.0
    A, _ = build_graph("small_world", n=n, k=k, beta=beta, seed=7)
    d = deg(A)
    assert (A.nnz // 2) == (n * k // 2)
    assert np.isclose(d.mean(), k, atol=0.25)
    # loose sanity bounds
    assert d.min() >= 1
    assert d.max() <= min(n-1, k+8)

# ========== MULTILAYER & ERRORS ==========

def test_multilayer_mismatched_sizes_raises():
    # first layer 3x3 (N=9), second 4x4 (N=16) -> should assert/fail
    with pytest.raises(ValueError):
        build_graph("multilayer_from",
                    base_kind="grid",
                    layers=2,
                    layer_kwargs=[{"m":3, "n":3}, {"m":4, "n":4}],
                    gamma=0.1, z_gap=1.0)

def test_dispatcher_unknown_kind_raises():
    with pytest.raises(ValueError):
        build_graph("i_do_not_exist", N=4)

# ========== DIRECTED OPTION & SYMMETRY ==========

def test_grid_directed_is_asymmetric():
    m = n = 4
    A, _ = build_graph("grid", m=m, n=n, directed=True)
    # Should contain only one direction per local move
    asym = (A != A.T)
    assert asym.nnz > 0  # not symmetric

def test_grid_undirected_is_symmetric():
    m = n = 4
    A, _ = build_graph("grid", m=m, n=n, directed=False)
    assert (A != A.T).nnz == 0
