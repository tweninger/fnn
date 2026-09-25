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


def backbone(state_dim=1, rank=0, sparse=False, clock="event", time_cap=10.0,
             ablation="none", fixed_gate_value=0.5, gamma_init=0.15,
             omega_init=0.8, input_scale_init=1.0, order=2):
    sampler = SimpleNamespace(nodes_neighbor_ids=[np.array([], dtype=int), np.array([2, 3]),
                                                  np.array([1]), np.array([1])],
                              nodes_neighbor_times=[np.array([], dtype=float), np.array([0., 10.]),
                                                    np.array([0.]), np.array([10.])])
    return FNN(np.zeros((4, 8)), np.ones((4, 1)), sampler, fnn_state_dim=state_dim,
               fnn_spectral_rank=rank, fnn_propagate=int(sparse), fnn_clock=clock,
               fnn_time_cap=time_cap, fnn_ablation=ablation,
               fnn_fixed_gate_value=fixed_gate_value, fnn_gamma_init=gamma_init,
               fnn_omega_init=omega_init, fnn_input_scale_init=input_scale_init,
               fnn_order=order)


def test_configurable_physical_initialization_uses_channel_scales():
    model = backbone(4, gamma_init=0.075, omega_init=0.4, input_scale_init=0.5)
    physical = model.field.physical_parameters()
    scales = torch.logspace(-0.3, 0.3, 4)
    torch.testing.assert_close(physical["gamma"], 0.075 * scales)
    torch.testing.assert_close(physical["omega"], 0.4 * scales)
    torch.testing.assert_close(physical["input_force_scale"], torch.tensor(0.5))
    with pytest.raises(ValueError, match="must be positive"):
        backbone(gamma_init=0.0)


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


@pytest.mark.parametrize("hops", [0, 1, 2, 4])
def test_multihop_matches_dense_conservative_diffusion(hops):
    model = backbone(4, sparse=hops)
    dst, group, amplitude = torch.tensor([0, 2, 2]), torch.tensor([0, 1, 1]), torch.tensor([2., 1., 3.])
    if hops == 0:
        assert not hasattr(model, "spread_raw")
        return
    keys = model.field.sparse_candidate_keys
    adjacency = torch.zeros(4, 4).index_put((keys // 4, keys % 4), model.field.sparse_topology_logits.sigmoid())
    degree = adjacency.sum(1)
    alpha = model.spread_raw.sigmoid() * (degree > 0)
    transition = alpha[:, None] * adjacency / degree[:, None].clamp_min(1e-12)
    frontier = torch.zeros(8).index_add(0, group*4+dst, amplitude).reshape(2, 4)
    expected = torch.zeros_like(frontier)
    for _ in range(hops):
        expected = expected + frontier * (1-alpha)
        frontier = frontier @ transition
    expected = expected + frontier
    nodes, groups, values = model._spread_inputs(dst, group, amplitude)
    actual = torch.zeros(8).index_add(0, groups*4+nodes, values).reshape(2, 4)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.sum(1), torch.tensor([2., 4.]))
    params = (model.spread_raw, model.field.sparse_topology_logits)
    a = torch.autograd.grad(actual.square().sum(), params, retain_graph=True)
    b = torch.autograd.grad(expected.square().sum(), params)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y)


def embed(model, time, positive=False):
    return model.compute_src_dst_node_temporal_embeddings(np.array([1]), np.array([2]),
                                                         np.array([time]), edges_are_positive=positive)


@pytest.mark.parametrize("state_dim,order", [(1, 1), (4, 1), (1, 2), (4, 2), (16, 2)])
def test_vectorized_field_composition_matches_steps_and_gradients(state_dim, order):
    torch.manual_seed(4)
    model = backbone(state_dim, order=order)
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


@pytest.mark.parametrize("name,state_dim,rank,sparse", [("FNN", 1, 0, False), ("FNN", 8, 0, False), ("FNN", 4, 4, False), ("FNN", 4, 0, 2), ("GraphMixer", 1, 0, False)])
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
    runner = (target / "train_link_prediction.py").read_text()
    evaluator = (target / "evaluate_models_utils.py").read_text()
    config = (target / "utils/load_configs.py").read_text()
    assert "compute_link_prediction_batch(" in runner and "compute_link_prediction_batch(" in evaluator
    assert "not args.fnn_legacy_weight_decay" in runner
    assert "fnn_zero_velocity_readout" in (target / "evaluate_link_prediction.py").read_text()
    assert "event_exact_unit" in config and "normalized_substep_unit" in config
    assert "fixed_dynamics" in config
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
                          "--fnn_state_dim", str(state_dim), "--fnn_spectral_rank", str(rank), "--fnn_propagate", str(int(sparse))],
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
                "--negative_sample_strategy", strategy, "--fnn_propagate", str(int(sparse))],
                env=dict(env, DYGLIB_DIR=str(target), PYTHON=sys.executable),
                capture_output=True, text=True, timeout=90)
            assert evaluation.returncode == 0, evaluation.stdout[-2000:] + evaluation.stderr[-6000:]
            suffix = f"_propagate{sparse}" if sparse else (f"_spectral{rank}" if rank else "")
            assert list((target / "saved_results").rglob(f"{strategy}_negative_sampling_FNN_seed0_lr0.0001_wd0.0_bs50{suffix}_dim{state_dim}.json"))


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


def test_dyglib_weight_decay_penalizes_positive_physical_values():
    model = backbone(4)
    transformed = model.transformed_decay_parameters()
    transformed_ids = {id(parameter) for parameter in transformed}
    assert id(model.field.gamma_raw) in transformed_ids
    assert id(model.field.omega_raw) in transformed_ids
    assert id(model.field.input_force_scale_raw) in transformed_ids
    with torch.no_grad():
        model.field.gamma_raw.zero_()
        model.field.omega_raw.zero_()
        model.field.input_force_scale_raw.zero_()

    penalty = model.transformed_weight_decay(0.01)
    penalty.backward()

    # A positive gradient at raw=0 makes an optimizer step decrease the raw,
    # so softplus(raw) moves below ln(2) toward the true minimum at zero.
    assert model.field.gamma_raw.grad.min() > 0
    assert model.field.omega_raw.grad.min() > 0
    assert model.field.input_force_scale_raw.grad > 0


def test_dyglib_fixed_topology_ablation_is_binary_and_frozen():
    model = backbone(ablation="fixed_topology")
    logits = model.field._topology_logits_for(
        torch.tensor([1, 0]), torch.tensor([2, 3])
    )
    torch.testing.assert_close(logits.sigmoid(), torch.tensor([1.0, 0.0]), atol=1e-12, rtol=0)
    assert not model.field.sparse_topology_logits.requires_grad
    assert model.field.gamma_raw.requires_grad


def test_dyglib_fixed_gates_are_constant_for_seen_and_unseen_pairs():
    for value in (0.5, 0.25):
        model = backbone(ablation="fixed_gates", fixed_gate_value=value)
        logits = model.field._topology_logits_for(
            torch.tensor([1, 0]), torch.tensor([2, 3])
        )
        torch.testing.assert_close(logits.sigmoid(), torch.full((2,), value))
        assert not model.field.sparse_topology_logits.requires_grad
        assert model.field.gamma_raw.requires_grad
    with pytest.raises(ValueError, match="strictly between"):
        backbone(ablation="fixed_gates", fixed_gate_value=1.0)


def test_dyglib_fixed_physical_ablation_freezes_coefficients_only():
    model = backbone(ablation="fixed_physical")
    params = model.field.physical_parameters()
    torch.testing.assert_close(params["gamma"], torch.tensor(0.15))
    torch.testing.assert_close(params["omega"], torch.tensor(0.8))
    torch.testing.assert_close(params["input_force_scale"], torch.tensor(1.0))
    assert not model.field.gamma_raw.requires_grad
    assert not model.field.omega_raw.requires_grad
    assert not model.field.input_force_scale_raw.requires_grad
    assert model.field.sparse_topology_logits.requires_grad


def test_dyglib_fixed_gates_physical_ablation_freezes_both():
    model = backbone(ablation="fixed_gates_physical", fixed_gate_value=0.5)
    logits = model.field._topology_logits_for(
        torch.tensor([1, 0]), torch.tensor([2, 3])
    )
    torch.testing.assert_close(logits.sigmoid(), torch.full((2,), 0.5))
    assert not model.field.sparse_topology_logits.requires_grad
    assert not model.field.gamma_raw.requires_grad
    assert not model.field.omega_raw.requires_grad
    assert not model.field.input_force_scale_raw.requires_grad


def test_bridge_update_preflights_missing_runner_anchor(tmp_path):
    from experiments.dyglib.setup import COMMIT, update
    (tmp_path / "FNN_PATCH.txt").write_text(COMMIT)
    (tmp_path / "utils").mkdir()
    config = tmp_path / "utils/load_configs.py"
    config.write_text("    parser.add_argument('--batch_size', type=int)\n")
    train = tmp_path / "train_link_prediction.py"
    train.write_text("dst_node_std_time_shift=dst_node_std_time_shift, device=args.device)\n")
    (tmp_path / "evaluate_link_prediction.py").write_text("unrecognized checkout")
    with pytest.raises(SystemExit, match="anchors"):
        update(tmp_path)
    assert "fnn_state_dim" not in config.read_text()
    assert "fnn_state_dim" not in train.read_text()


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


def test_normalized_clock_uses_elapsed_time_caps_gaps_and_restores_time():
    near = backbone(4, clock="normalized", time_cap=2.0)
    far = backbone(4, clock="normalized", time_cap=2.0)
    far.load_state_dict(near.state_dict())
    torch.testing.assert_close(near.time_scale, torch.tensor(10., dtype=torch.float64))
    src, dst = torch.tensor([1, 1]), torch.tensor([2, 2])
    near.advance(src, dst, torch.tensor([0., 10.], dtype=torch.float64))
    far.advance(src, dst, torch.tensor([0., 1000.], dtype=torch.float64))
    assert not torch.equal(near.memory_bank.h, far.memory_bank.h)

    capped = backbone(4, clock="normalized", time_cap=2.0)
    capped.load_state_dict(near.state_dict())
    capped.memory_bank.__init_memory_bank__()
    capped.advance(src, dst, torch.tensor([0., 20.], dtype=torch.float64))
    torch.testing.assert_close(far.memory_bank.h, capped.memory_bank.h)
    torch.testing.assert_close(far.memory_bank.v, capped.memory_bank.v)
    assert torch.isfinite(far.memory_bank.h).all()

    loss = near.memory_bank.h.square().sum() + near.memory_bank.v.square().sum()
    loss.backward()
    assert near.field.gamma_raw.grad is not None
    saved = near.memory_bank.backup_memory_bank()
    before = near.memory_bank.last_update_time.clone()
    near.memory_bank.__init_memory_bank__()
    near.memory_bank.reload_memory_bank(saved)
    torch.testing.assert_close(near.memory_bank.last_update_time, before)


def test_normalized_clock_evolves_during_quiet_query_gaps():
    model = backbone(clock="normalized")
    embed(model, 0, positive=True)
    embed(model, 10)
    after_event = model.memory_bank.h.clone()
    embed(model, 20)
    assert not torch.equal(model.memory_bank.h, after_event)
    assert model.memory_bank.last_update_time == 20


@pytest.mark.parametrize("clock", [
    "normalized_exact", "event_exact", "event_exact_unit",
    "normalized_substep", "normalized_substep_unit",
])
def test_exact_clock_scores_timestamp_groups_causally(clock):
    exact = backbone(clock=clock)
    batch_min = backbone(clock="normalized")
    batch_min.load_state_dict(exact.state_dict(), strict=False)
    src = np.array([1, 1])
    dst = np.array([2, 2])
    neg_dst = np.array([3, 3])
    times = np.array([0., 10.])

    exact_embeddings = exact.compute_link_prediction_batch(src, dst, src, neg_dst, times)
    batch_min.compute_src_dst_node_temporal_embeddings(src, neg_dst, times,
                                                       edges_are_positive=False)
    batch_min_embeddings = batch_min.compute_src_dst_node_temporal_embeddings(
        src, dst, times, edges_are_positive=True)
    # Batch-min scoring uses the same pre-event state at both timestamps.
    torch.testing.assert_close(batch_min_embeddings[1][0], batch_min_embeddings[1][1])
    # Exact scoring exposes the time-0 event to the time-10 query, but not to time 0.
    assert not torch.equal(exact_embeddings[1][0], exact_embeddings[1][1])
    assert exact.memory_bank.node_raw_messages[2].tolist() == [10.]
    loss = sum(value.square().sum() for value in exact_embeddings)
    loss.backward()
    assert exact.field.gamma_raw.grad is not None


@pytest.mark.parametrize("clock", [
    "normalized_exact", "event_exact", "event_exact_unit",
    "normalized_substep", "normalized_substep_unit",
])
def test_exact_clock_never_consumes_same_time_events(clock):
    model = backbone(clock=clock)
    first = model.compute_link_prediction_batch(np.array([1]), np.array([2]),
                                                np.array([1]), np.array([3]), np.array([0.]))
    second = model.compute_link_prediction_batch(np.array([1]), np.array([2]),
                                                 np.array([1]), np.array([3]), np.array([0.]))
    for before, after in zip(first, second):
        torch.testing.assert_close(before, after)
    assert model.memory_bank.h.count_nonzero() == 0
    model.compute_link_prediction_batch(np.array([1]), np.array([2]),
                                        np.array([1]), np.array([3]), np.array([10.]))
    assert model.memory_bank.h.count_nonzero() > 0


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
