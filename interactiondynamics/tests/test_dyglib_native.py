import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.dyglib.fnn import FNN
from interactiondynamics.core.events import EventBatch
from interactiondynamics.core.interfaces import ModelState


def backbone():
    sampler = SimpleNamespace(nodes_neighbor_ids=[np.array([], dtype=int), np.array([2, 3]),
                                                  np.array([1]), np.array([1])])
    return FNN(np.zeros((4, 8)), np.ones((4, 1)), sampler)


def embed(model, time, positive=False):
    return model.compute_src_dst_node_temporal_embeddings(np.array([1]), np.array([2]),
                                                         np.array([time]), edges_are_positive=positive)


def test_vectorized_field_composition_matches_steps_and_gradients():
    torch.manual_seed(4)
    model = backbone()
    src = torch.ones(40, dtype=torch.long)
    dst = torch.randint(1, 4, (40,))
    times = torch.arange(40) // 3
    model.memory_bank.h = torch.randn(4, 1)
    model.memory_bank.v = torch.randn(4, 1)
    state = ModelState(node=model.memory_bank.h.clone(), node_prev=model.memory_bank.v.clone())
    for t in times.unique():
        mask = times == t
        state, _ = model.field.step(state, EventBatch(src=src[mask], dst=dst[mask],
                                   features=torch.ones(int(mask.sum()), 1),
                                   is_external=torch.zeros(int(mask.sum()), dtype=torch.bool)))
    parameters = [p for p in model.field.parameters() if p.requires_grad]
    reference_grad = torch.autograd.grad(state.node.sum() + state.node_prev.sum(), parameters, allow_unused=True)
    model.advance(src, dst, times)
    torch.testing.assert_close(model.memory_bank.h, state.node, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(model.memory_bank.v, state.node_prev, atol=2e-5, rtol=2e-5)
    actual_grad = torch.autograd.grad(model.memory_bank.h.sum() + model.memory_bank.v.sum(), parameters, allow_unused=True)
    for expected, actual in zip(reference_grad, actual_grad):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


def test_native_fnn_causality_gradients_and_memory_restore():
    torch.manual_seed(0)
    model = backbone()
    before = embed(model, 1)
    positive = embed(model, 1, True)
    for a, b in zip(before, positive):
        torch.testing.assert_close(a, b)
    assert model.memory_bank.h.count_nonzero() == 0
    embed(model, 1)  # Same-time observations are still not visible.
    assert model.memory_bank.h.count_nonzero() == 0
    out = embed(model, 2)
    sum(x.square().sum() for x in out).backward()
    assert model.field.sparse_topology_logits.grad is not None
    assert model.field.sparse_topology_logits.grad.abs().sum() > 0
    model.memory_bank.detach_memory_bank()
    embed(model, 2, True)
    saved = model.memory_bank.backup_memory_bank()
    expected = embed(model, 3)
    model.memory_bank.__init_memory_bank__()
    model.memory_bank.reload_memory_bank(saved)
    actual = embed(model, 3)
    for a, b in zip(expected, actual):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("name", ["FNN", "GraphMixer"])
def test_upstream_native_training_and_checkpoint_roundtrip(name, tmp_path):
    """Offline integration check once the pinned checkout has been installed."""
    from experiments.dyglib.setup import install
    import pandas as pd
    root = Path(__file__).resolve().parents[2]
    source = root / "derived/dyglib"
    if not source.exists():
        pytest.skip("Install experiments.dyglib.setup to run upstream integration tests")
    target = tmp_path / "dyglib"
    install(target, str(source))
    directory = target / "processed_data/college_msg"
    directory.mkdir(parents=True)
    rng = np.random.RandomState(7)
    count, nodes = 400, 20
    pd.DataFrame(dict(u=rng.randint(1, nodes + 1, count), i=rng.randint(1, nodes + 1, count),
                      ts=np.arange(count), idx=np.arange(1, count + 1), label=np.ones(count))).to_csv(
                          directory / "ml_college_msg.csv", index=False)
    np.save(directory / "ml_college_msg.npy", np.ones((count + 1, 1), dtype=np.float32))
    np.save(directory / "ml_college_msg_node.npy", np.zeros((nodes + 1, 1), dtype=np.float32))
    env = dict(os.environ, PYTHONPATH=str(root), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    run = subprocess.run([sys.executable, "train_link_prediction.py", "--dataset_name", "college_msg",
                          "--model_name", name, "--num_epochs", "1", "--num_runs", "1",
                          "--batch_size", "50", "--num_neighbors", "5", "--gpu", "-1"],
                         cwd=target, env=env, capture_output=True, text=True, timeout=90)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-6000:]
    assert list((target / "saved_results").rglob("*.json"))
