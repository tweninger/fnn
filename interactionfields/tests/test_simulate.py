import numpy as np
import scipy.sparse as sp
import pytest

from interactionfields.graphs import build_graph
from interactionfields.simulate import (
    SIMULATORS,
    run_simulator,
    simulate_dripping_wave_on_graph,
)

def _make_small_grid(nside=6):
    A, meta = build_graph("grid", m=nside, n=nside)
    return A.tocsr(), meta

# ---------------- Registry & dispatcher ----------------

def test_simulator_registry_has_entries():
    assert "dripping_wave" in SIMULATORS
    assert callable(SIMULATORS["dripping_wave"])

def test_run_simulator_unknown_raises():
    with pytest.raises(ValueError):
        run_simulator("nope", adj=sp.csr_matrix((4, 4)), t_bins=1)

# ---------------- Dripping-wave simulator ----------------

def test_dripping_wave_bins_and_states_shapes():
    A, _ = _make_small_grid(6)
    T = 20
    bins, H, meta_list = simulate_dripping_wave_on_graph(
        adj=A,
        t_bins=T,
        faucet_nodes=[0],
        return_states=True,
        edge_threshold=0.0,
    )
    assert isinstance(bins, list) and len(bins) == T
    assert isinstance(meta_list, list) and len(meta_list) == T
    assert isinstance(H, np.ndarray) and H.shape == (T, A.shape[0])
    for B in bins:
        assert isinstance(B, sp.csr_matrix)
        assert B.dtype == np.uint8
        assert B.shape == A.shape

def test_dripping_wave_empty_graph_yields_empty_bins():
    A = sp.csr_matrix((4, 4), dtype=np.float32)
    bins, _H, _meta = simulate_dripping_wave_on_graph(
        adj=A,
        t_bins=5,
        faucet_nodes=[0],
        return_states=False,
    )
    for B in bins:
        assert B.nnz == 0
        assert B.shape == (4, 4)
