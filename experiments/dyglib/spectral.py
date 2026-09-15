"""Fixed, train-only low-frequency basis for event-clock graph propagation."""
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import eigsh


def train_basis(num_nodes, keys, rank):
    keys = keys.detach().cpu().numpy()
    src, dst = keys // num_nodes, keys % num_nodes
    keep = src != dst
    adjacency = sparse.coo_matrix((np.ones(keep.sum()), (src[keep], dst[keep])),
                                  shape=(num_nodes, num_nodes)).tocsr()
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.data[:] = 1
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    active = np.flatnonzero(degree > 0)
    if not len(active):
        raise ValueError("Spectral mode requires non-self training edges")
    norm = sparse.diags(1 / np.sqrt(degree[active]))
    lap = sparse.eye(len(active)) - norm @ adjacency[active][:, active] @ norm
    count = min(rank, len(active))
    if len(active) <= 256:
        values, vectors = np.linalg.eigh(lap.toarray())
        values, vectors = values[:count], vectors[:, :count]
    else:
        if count >= len(active):
            raise ValueError("Use spectral rank smaller than the active graph size")
        values, vectors = eigsh(lap, k=count, which="SM", tol=1e-7,
                                v0=np.random.default_rng(0).normal(size=len(active)))
        order = np.argsort(values)
        values, vectors = values[order], vectors[:, order]
    basis = np.zeros((num_nodes, count), dtype=np.float32)
    basis[active] = vectors
    return torch.from_numpy(basis), torch.tensor(np.maximum(values, 0), dtype=torch.float32)
