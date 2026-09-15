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


def backbone(state_dim=1, coupling=0.0):
    sampler = SimpleNamespace(nodes_neighbor_ids=[np.array([], dtype=int), np.array([2, 3]),
                                                  np.array([1]), np.array([1])])
    return FNN(np.zeros((4, 8)), np.ones((4, 1)), sampler, fnn_state_dim=state_dim,
               fnn_coupling=coupling)


def embed(model, time, positive=False):
    return model.compute_src_dst_node_temporal_embeddings(np.array([1]), np.array([2]),
                                                         np.array([time]), edges_are_positive=positive)


@pytest.mark.parametrize("state_dim", [1, 4, 16])
def test_vectorized_field_composition_matches_steps_and_gradients(state_dim):
    torch.manual_seed(4)
    model = backbone(state_dim)
    src = torch.ones(40, dtype=torch.long)
    dst = torch.randint(1, 4, (40,))
    times = torch.arange(40) // 3
    model.memory_bank.h = torch.randn(4, state_dim)
    model.memory_bank.v = torch.randn(4, state_dim)
    state = ModelState(node=model.memory_bank.h.clone(), node_prev=model.memory_bank.v.clone())
    for t in times.unique():
        mask = times == t
        state, _ = model.field.step(state, EventBatch(src=src[mask], dst=dst[mask],
                                   features=(model.drive_vector.expand(int(mask.sum()), -1) if state_dim > 1
                                             else torch.ones(int(mask.sum()), 1)),
                                   is_external=torch.zeros(int(mask.sum()), dtype=torch.bool)))
    parameters = [p for p in model.field.parameters() if p.requires_grad]
    if state_dim > 1:
        parameters.append(model.drive_vector)
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


@pytest.mark.parametrize("state_dim,coupling", [(1, 0), (8, 0), (4, 0.1)])
def test_native_fnn_causality_gradients_and_memory_restore(state_dim, coupling):
    torch.manual_seed(0)
    model = backbone(state_dim, coupling)
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


@pytest.mark.parametrize("name,state_dim,coupling", [("FNN", 1, 0), ("FNN", 8, 0),
                                                     ("FNN", 4, 0.1), ("GraphMixer", 1, 0.1)])
def test_upstream_native_training_and_checkpoint_roundtrip(name, state_dim, coupling, tmp_path):
    """Offline integration check once the pinned checkout has been installed."""
    from experiments.dyglib.setup import install
    import pandas as pd
    root = Path(__file__).resolve().parents[2]
    source = root / "derived/dyglib"
    if not source.exists():
        pytest.skip("Install experiments.dyglib.setup to run upstream integration tests")
    target = tmp_path / "dyglib"
    install(target, str(source))
    from experiments.dyglib.setup import update
    before = (target / "train_link_prediction.py").read_text()
    update(target)
    assert (target / "train_link_prediction.py").read_text() == before
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
                          "--batch_size", "50", "--num_neighbors", "5", "--gpu", "-1",
                          "--fnn_state_dim", str(state_dim), "--fnn_coupling", str(coupling)],
                         cwd=target, env=env, capture_output=True, text=True, timeout=90)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-6000:]
    assert list((target / "saved_results").rglob("*.json"))
    if state_dim > 1:
        assert list((target / "saved_results").rglob(f"*_dim{state_dim}.json"))


@pytest.mark.parametrize("state_dim", [1, 4])
def test_zero_coupling_matches_fast_path(state_dim):
    model = backbone(state_dim)
    model.memory_bank.h = torch.randn(4, state_dim)
    model.memory_bank.v = torch.randn(4, state_dim)
    saved = model.memory_bank.backup_memory_bank()
    src, dst, times = torch.tensor([1, 2, 1]), torch.tensor([2, 1, 3]), torch.tensor([2, 1, 2])
    model.advance(src, dst, times)
    expected = model.memory_bank.backup_memory_bank()
    model.memory_bank.reload_memory_bank(saved)
    model._advance_coupled(src, dst, times, 0.0)
    torch.testing.assert_close(model.memory_bank.h, expected[0])
    torch.testing.assert_close(model.memory_bank.v, expected[1])


def test_coupling_propagates_without_edge_event_and_has_gradients():
    model = backbone(1, 0.2)
    model.memory_bank.h[2] = 1
    # Event at isolated node zero: coupling must independently move 2 -> 1.
    model.advance(torch.tensor([0]), torch.tensor([0]), torch.tensor([1]))
    assert model.memory_bank.h[1].item() > 0
    model.memory_bank.h[1].sum().backward()
    assert model.kappa_raw.grad.abs() > 0
    assert model.field.sparse_topology_logits.grad.abs().sum() > 0


def test_coupling_matches_dense_reference_and_gradients():
    model = backbone(4, 0.2)
    h, v = torch.randn(4, 4), torch.randn(4, 4)
    model.memory_bank.h, model.memory_bank.v = h.clone(), v.clone()
    keys = model.field.sparse_candidate_keys
    src, dst = keys // 4, keys % 4
    weights = model.field.sparse_topology_logits.sigmoid()
    adjacency = weights.new_zeros(4, 4).index_put((dst, src), weights)
    adjacency = adjacency / (adjacency.sum(1, keepdim=True) + 1e-8)
    p = model.field.physical_parameters()
    event_gate = model.field._topology_logits_for(torch.tensor([0]), torch.tensor([0])).sigmoid()
    for _ in range(3):
        force = torch.zeros_like(h).index_add(0, torch.tensor([0]),
                    (p["input_force_scale"] * event_gate * model.drive_vector)[None])
        v = (1-p["gamma"]*p["dt"])*v + p["dt"]*(force + torch.nn.functional.softplus(model.kappa_raw)*
                    (adjacency @ h - adjacency.sum(1)[:, None]*h) - p["omega"].square()*h)
        h = h + p["dt"]*v
    parameters = [model.kappa_raw, model.field.sparse_topology_logits, model.drive_vector,
                  model.field.gamma_raw, model.field.omega_raw]
    expected_grad = torch.autograd.grad(h.sum()+v.sum(), parameters)
    model.advance(torch.zeros(3, dtype=torch.long), torch.zeros(3, dtype=torch.long), torch.arange(3))
    torch.testing.assert_close(model.memory_bank.h, h)
    torch.testing.assert_close(model.memory_bank.v, v)
    actual_grad = torch.autograd.grad(model.memory_bank.h.sum()+model.memory_bank.v.sum(), parameters)
    for actual, expected in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_invalid_coupling():
    for value in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="nonnegative"):
            backbone(coupling=value)


def test_empty_support_coupling_is_finite_and_matches_uncoupled():
    model = backbone(4, 0.1)
    empty = torch.empty(0, dtype=torch.long)
    model.field.set_sparse_topology_candidates(empty, empty)
    model.memory_bank.h.fill_(1)
    p = model.field.physical_parameters()
    expected_v = -p["dt"] * p["omega"].square()
    # Unknown event goes to node zero; other nodes evolve independently.
    model.advance(torch.tensor([0]), torch.tensor([0]), torch.tensor([1]))
    assert torch.isfinite(model.memory_bank.h).all()
    torch.testing.assert_close(model.memory_bank.v[1], expected_v)


def test_channel_initialization_and_validation():
    scalar = backbone()
    assert not hasattr(scalar, "drive_vector")  # Legacy checkpoint keys unchanged.
    assert scalar.projection.in_features == 2
    model = backbone(8)
    params = model.field.physical_parameters()
    assert params["gamma"].unique().numel() == 8
    assert params["omega"].unique().numel() == 8
    assert model.projection.in_features == 16
    torch.testing.assert_close(model.drive_vector.norm(), torch.tensor(1.))
    for value in (0, -1, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            backbone(value)


def test_bridge_update_preserves_existing_data_and_preflights(tmp_path):
    from experiments.dyglib.setup import COMMIT, update
    (tmp_path / "FNN_PATCH.txt").write_text(COMMIT)
    (tmp_path / "utils").mkdir()
    config = tmp_path / "utils/load_configs.py"
    config.write_text("    parser.add_argument('--batch_size', type=int)\n")
    train = tmp_path / "train_link_prediction.py"
    original = "dst_node_std_time_shift=dst_node_std_time_shift, device=args.device)\nf'{args.model_name}_seed{args.seed}'"
    train.write_text(original)
    evaluation = tmp_path / "evaluate_link_prediction.py"
    evaluation.write_text("unrecognized checkout")
    with pytest.raises(SystemExit, match="anchors"):
        update(tmp_path)
    assert "fnn_state_dim" not in config.read_text()
    assert train.read_text() == original
    evaluation.write_text(original)
    data = tmp_path / "keep-result.json"
    data.write_text("preserve me")
    update(tmp_path)
    assert data.read_text() == "preserve me"
    assert "fnn_state_dim" in train.read_text()
