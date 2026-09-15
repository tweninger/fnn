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


def backbone(state_dim=1, rank=0, sparse=False):
    sampler = SimpleNamespace(nodes_neighbor_ids=[np.array([], dtype=int), np.array([2, 3]),
                                                  np.array([1]), np.array([1])])
    return FNN(np.zeros((4, 8)), np.ones((4, 1)), sampler, fnn_state_dim=state_dim, fnn_spectral_rank=rank, fnn_sparse_propagation=sparse)


def test_sparse_inputs_dense_reference_and_gradients():
    model = backbone(4, sparse=True)
    dst = torch.tensor([0, 1, 2, 1])
    group = torch.arange(4)
    amplitude = torch.tensor([1., 2., 3., 4.], requires_grad=True)
    keys = model.field.sparse_candidate_keys
    adjacency = torch.zeros(4, 4).index_put((keys // 4, keys % 4), model.field.sparse_topology_logits.sigmoid())
    degree = adjacency.sum(1)
    alpha = model.spread_raw.sigmoid() * (degree > 0)
    transition = torch.diag(1-alpha) + alpha[:, None] * adjacency / degree[:, None].clamp_min(1e-12)
    expected = amplitude[:, None] * transition[dst]
    targets, groups, values = model._spread_inputs(dst, group, amplitude)
    actual = torch.zeros(16).index_add(0, groups*4 + targets, values).reshape(4, 4)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.sum(1), amplitude)
    assert actual[2, 3] == 0  # 2 -> 1 -> 3 must not recurse.
    params = (amplitude, model.spread_raw, model.field.sparse_topology_logits)
    a = torch.autograd.grad(actual.square().sum(), params, retain_graph=True)
    b = torch.autograd.grad(expected.square().sum(), params)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y)


def test_sparse_zero_spread_matches_original():
    model, original = backbone(4, sparse=True), backbone(4)
    original.load_state_dict(model.state_dict(), strict=False)
    model.spread_raw.data.fill_(-100)
    src, dst, times = torch.tensor([1, 2, 1]), torch.tensor([2, 1, 0]), torch.tensor([0, 1, 1])
    model.advance(src, dst, times)
    original.advance(src, dst, times)
    torch.testing.assert_close(model.memory_bank.h, original.memory_bank.h)
    torch.testing.assert_close(model.memory_bank.v, original.memory_bank.v)
    with pytest.raises(ValueError, match="not both"):
        backbone(4, rank=2, sparse=True)


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


@pytest.mark.parametrize("state_dim,rank,sparse", [(1, 0, False), (8, 0, False), (4, 3, False), (4, 0, True)])
def test_native_fnn_causality_gradients_and_memory_restore(state_dim, rank, sparse):
    torch.manual_seed(0)
    model = backbone(state_dim, rank, sparse)
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


@pytest.mark.parametrize("name,state_dim,rank,sparse", [("FNN", 1, 0, False), ("FNN", 8, 0, False), ("FNN", 4, 4, False), ("FNN", 4, 0, True), ("GraphMixer", 1, 0, False)])
def test_upstream_native_training_and_checkpoint_roundtrip(name, state_dim, rank, sparse, tmp_path):
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
                          "--fnn_state_dim", str(state_dim), "--fnn_spectral_rank", str(rank)] + (["--fnn_sparse_propagation"] if sparse else []),
                         cwd=target, env=env, capture_output=True, text=True, timeout=90)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-6000:]
    assert list((target / "saved_results").rglob("*.json"))
    if state_dim > 1:
        assert list((target / "saved_results").rglob(f"*_dim{state_dim}.json"))
    if name == "FNN" and state_dim > 1:
        for strategy in ("random", "historical", "inductive"):
            evaluation = subprocess.run(["bash", str(root / "scripts/7_dyglib.sh"), "eval",
                "--dataset_name", "college_msg", "--model_name", name, "--num_runs", "1",
                "--batch_size", "50", "--gpu", "-1", "--fnn_state_dim", str(state_dim),
                "--fnn_spectral_rank", str(rank),
                "--negative_sample_strategy", strategy] + (["--fnn_sparse_propagation"] if sparse else []),
                env=dict(env, DYGLIB_DIR=str(target), PYTHON=sys.executable),
                capture_output=True, text=True, timeout=90)
            assert evaluation.returncode == 0, evaluation.stdout[-2000:] + evaluation.stderr[-6000:]
            suffix = "_sparseprop" if sparse else (f"_spectral{rank}" if rank else "")
            assert list((target / "saved_results").rglob(f"{strategy}_negative_sampling_FNN_seed0{suffix}_dim{state_dim}.json"))


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


def test_event_spectral_matches_dense_wave_and_gradients():
    torch.manual_seed(7)
    model = backbone(4, rank=3)
    lap = torch.zeros(4, 4)
    lap[1:, 1:] = torch.eye(3)
    lap[1, 2] = lap[2, 1] = lap[1, 3] = lap[3, 1] = -1 / 2**0.5
    torch.testing.assert_close(model.basis @ torch.diag(model.eigenvalues) @ model.basis.T,
                               lap, atol=1e-6, rtol=1e-6)
    h, v = torch.zeros(4, 4), torch.zeros(4, 4)
    src, dst = torch.tensor([1, 2, 1, 1]), torch.tensor([2, 1, 3, 2])
    times = torch.tensor([0, 3, 3, 100])
    p = model.field.physical_parameters()
    kappa = torch.nn.functional.softplus(model.kappa_raw)
    for t in times.unique():
        mask = times == t
        amplitude = p["input_force_scale"] * model.field._topology_logits_for(src[mask], dst[mask]).sigmoid()
        incoming = torch.zeros_like(h).index_add(0, dst[mask], amplitude[:, None]*model.drive_vector)
        v = (1-p["gamma"]*p["dt"])*v + p["dt"]*(incoming-p["omega"].square()*h-kappa*(lap@h))
        h = h+p["dt"]*v
    parameters = [model.kappa_raw, model.field.gamma_raw, model.field.omega_raw,
                  model.field.sparse_topology_logits, model.drive_vector]
    expected_grad = torch.autograd.grad(h.square().sum()+v.square().sum(), parameters)
    model.advance(src, dst, times)
    ids = torch.arange(4)
    actual, _ = model._node_states(ids, ids)
    torch.testing.assert_close(actual, torch.cat((h, v), -1), atol=1e-6, rtol=1e-5)
    actual_grad = torch.autograd.grad(actual.square().sum(), parameters)
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-4)


def test_spectral_zero_strength_and_event_gap_invariance():
    model = backbone(4, rank=2)
    with torch.no_grad():
        model.kappa_raw.fill_(-80)
    local = backbone(4)
    local.load_state_dict(model.state_dict(), strict=False)
    src, dst = torch.tensor([1, 0, 1]), torch.tensor([2, 0, 3])
    model.advance(src, dst, torch.tensor([0, 1, 2]))
    local.advance(src, dst, torch.tensor([0, 100, 100000]))
    ids = torch.arange(4)
    actual, _ = model._node_states(ids, ids)
    expected, _ = local._node_states(ids, ids)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert actual[0].abs().sum() > 0  # Isolated nodes retain their local drive.
    assert not model.basis.requires_grad


def test_spectral_propagation_without_incoming_event():
    model = backbone(1, rank=3)
    # Only node 2 receives impulses, but node 1 gains state through the graph.
    model.advance(torch.tensor([1, 1]), torch.tensor([2, 2]), torch.tensor([0, 1]))
    actual, _ = model._node_states(torch.tensor([1]), torch.tensor([1]))
    assert actual[0, 0] > 0


def test_sparse_spectral_basis_ring():
    from experiments.dyglib.spectral import train_basis
    ids = torch.arange(300)
    basis, values = train_basis(300, ids*300+(ids+1)%300, 4)
    torch.testing.assert_close(basis.T@basis, torch.eye(4), atol=1e-5, rtol=1e-5)
    applied = basis-.5*(basis.roll(1, 0)+basis.roll(-1, 0))
    torch.testing.assert_close(applied, basis*values, atol=1e-6, rtol=1e-5)
