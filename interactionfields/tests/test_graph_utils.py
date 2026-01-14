import numpy as np
import scipy.sparse as sp
import pytest

from interactionfields.graphs import (
    build_grid,
    build_multilayer,
    edges_from_adj,
    remove_defects,
)


def test_edges_from_adj_unique_undirected():
    # Build a symmetric adjacency with duplicate directed edges.
    rows = np.array([0, 1, 1, 2])
    cols = np.array([1, 0, 2, 1])
    data = np.ones(rows.size, dtype=np.float32)
    A = sp.csr_matrix((data, (rows, cols)), shape=(3, 3))

    edges = edges_from_adj(A)
    edge_set = {tuple(pair) for pair in edges.tolist()}
    assert edge_set == {(0, 1), (1, 2)}


def test_remove_defects_removes_nodes():
    A, _ = build_grid(m=3, n=3)
    B = remove_defects(A, remove_nodes=[0, 1])
    assert B.shape == (7, 7)
    # No row/col should reference removed nodes after reindexing.
    assert B.nnz > 0


def test_remove_defects_removes_edges():
    A, _ = build_grid(m=2, n=2)
    B = remove_defects(A, remove_edges=[(0, 1), (1, 0)])
    assert B[0, 1] == 0
    assert B[1, 0] == 0


def test_build_multilayer_empty_layers_raises():
    with pytest.raises(ValueError):
        build_multilayer([])
