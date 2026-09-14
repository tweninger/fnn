import gzip
import math
import pytest
import torch
from torch import nn

from interactiondynamics.aggregators.settransformers import SetTransformerAggregator
from interactiondynamics.aggregators.hopfield import HopfieldAggregator
from interactiondynamics.core.config import ModelConfig
from interactiondynamics.core.events import EventBatch
from interactiondynamics.core.interfaces import ModelState
from interactiondynamics.models.dyglib_adapter import DyGLibAdapter, EdgeBankAdapter
from interactiondynamics.eval.evaluate import evaluate_stream_sliced, evaluate_physical_force_rollout
from interactiondynamics.training.types import TrainConfig
from interactiondynamics.updates.hnn import HNNUpdate
from interactiondynamics.models.fnn import FieldNeuralNetwork


def batch(t):
    return EventBatch(src=torch.tensor([0]), dst=torch.tensor([1]),
                      t=torch.tensor([float(t)]), features=torch.ones(1, 1))


def test_set_attention_padding_invariance():
    torch.manual_seed(12)
    a = SetTransformerAggregator(8, num_heads=2, max_events_per_node=2)
    b = SetTransformerAggregator(8, num_heads=2, max_events_per_node=8)
    b.load_state_dict(a.state_dict())
    x = torch.randn(1, 8, requires_grad=True)
    left, right = a(None, x, batch(0), 3), b(None, x, batch(0), 3)
    torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)
    right.square().sum().backward()
    assert x.grad.abs().sum() > 0
    assert torch.equal(right[2], torch.zeros(8))


def test_hopfield_refines_query_and_preserves_recency_across_roles():
    torch.manual_seed(12)
    a = HopfieldAggregator(4, 4, hidden_dim=4, num_heads=1, max_events_per_node=2)
    events = EventBatch(src=torch.tensor([0, 1, 0, 1]), dst=torch.tensor([1, 0, 1, 0]))
    packed, _ = a._build_padded_sets(torch.arange(4.).view(4, 1).expand(-1, 4), events, 2)
    torch.testing.assert_close(packed[0, :, 0], torch.tensor([2., 3.]))
    x = torch.randn(4, 4)
    state = ModelState(node=torch.randn(2, 4))
    first = a(state, x, events, 2)
    a.steps = 3
    assert not torch.allclose(first, a(state, x, events, 2))


def test_graphmixer_width_and_feature_padding():
    model = DyGLibAdapter(3, 1, ModelConfig(temporal_model="graphmixer", node_dim=8,
                                          time_emb_dim=4, temporal_num_neighbors=2))
    state = model.init_state(1, 3, torch.device("cpu"))
    state, _ = model.step(state, batch(0))
    model._prepare(state)
    assert model.backbone.num_channels == 172
    assert model.backbone.edge_raw_features.shape == (2, 172)
    assert model.backbone.edge_raw_features[1, 0] == 1
    assert model.backbone.edge_raw_features[:, 1:].count_nonzero() == 0


def test_jodie_train_clock_stats_and_checkpoint():
    cfg = ModelConfig(temporal_model="jodie", node_dim=8, time_emb_dim=4)
    model = DyGLibAdapter(3, 1, cfg)
    model.configure_training_history([batch(2), batch(6)])
    assert model.src_node_mean_time_shift == 3
    assert model.src_node_std_time_shift == 1
    restored = DyGLibAdapter(3, 1, cfg)
    restored.load_state_dict(model.state_dict())
    state = restored.init_state(1, 3, torch.device("cpu"))
    restored._prepare(state)
    assert restored.backbone.src_node_mean_time_shift == 3
    model.configure_training_history([batch(0), batch(0)])
    assert model.src_node_std_time_shift == 1


def test_history_replay_scores_first_heldout_without_observing_it():
    class RecordingBank(EdgeBankAdapter):
        def score(self, state, events):
            seen = state.aux["time"]
            time = int(events.t.min())
            # The evaluator also scores a deliberately frozen persistence
            # control. Neither path may contain the queried event.
            assert seen.max() < time
            assert seen.tolist() in ([0., 1.], list(range(time)))
            self.histories.append((time, seen.tolist()))
            assert time in (2, 3)
            self.calls += 1
            return super().score(state, events)
    model = RecordingBank(3, 1)
    model.calls = 0
    model.histories = []
    evaluate_stream_sliced(model, [batch(2), batch(3)], None, None, TrainConfig(num_nodes=3, num_neg=1),
                           history_bins=[batch(0), batch(1)])
    assert model.calls > 0
    assert (3, [0., 1., 2.]) in model.histories


def test_rollout_replays_past_before_heldout_observations():
    class RecordingFNN(FieldNeuralNetwork):
        def step(self, state, events, drive=None):
            self.seen.append(int(events.t[0]))
            return super().step(state, events, drive)
    model = RecordingFNN(num_nodes=3, force_dim=1, state_dim=1,
                         gamma_init=0.1, omega_init=0.5, dt=0.1)
    model.seen = []
    before = {k: v.clone() for k, v in model.state_dict().items()}
    evaluate_physical_force_rollout(model, [batch(2), batch(3), batch(4)],
                                   TrainConfig(num_nodes=3, num_neg=1), horizon=1,
                                   history_bins=[batch(0), batch(1)])
    assert model.seen[:5] == [0, 1, 2, 3, 4]
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_no_history_keeps_independent_episode_initialization():
    class RecordingBank(EdgeBankAdapter):
        def score(self, state, events):
            assert state.aux["time"].tolist() == [2.]
            assert events.t.min() == 3
            return super().score(state, events)
    evaluate_stream_sliced(RecordingBank(3, 1), [batch(2), batch(3)], None, None,
                           TrainConfig(num_nodes=3, num_neg=1))


def test_hnn_uses_both_derivatives_at_old_state():
    class Product(nn.Module):
        def forward(self, x):
            return (x[:, 0] * x[:, 1]).unsqueeze(-1)
    model = HNNUpdate(2, 1, dt=0.1)
    model.H = Product()
    state, _ = model(ModelState(node=torch.tensor([[1., 2.]])), torch.zeros(1, 1))
    torch.testing.assert_close(state.node, torch.tensor([[1.1, 1.8]]))


@pytest.mark.parametrize("kind", ["edgebank", "jodie"])
def test_social_runner_uses_history_and_train_only_calibration(kind, tmp_path):
    from interactiondynamics.data.social import SocialConfig, SocialEventDataset
    from interactiondynamics.data.download_benchmarks import SOURCES
    from interactiondynamics.models.model_factory import build_model
    from interactiondynamics.training.runner import run_one_experiment
    from interactiondynamics.training.types import SweepRun
    from interactiondynamics.training.task_metrics import TaskMetricSpec
    folder = tmp_path / "college_msg"
    folder.mkdir()
    with gzip.open(folder / SOURCES["college_msg"][0], "wt") as handle:
        for t in range(10):
            handle.write(f"{t % 3} {(t + 1) % 3} {20 * t}\n")
    ds = SocialEventDataset(SocialConfig(root=str(tmp_path), bin_size=20, split_by="time"))
    models = []
    def factory(spec, cfg):
        model = build_model(spec, cfg)
        models.append(model)
        return model
    result = run_one_experiment(
        ds, ds.spec(), TrainConfig(num_nodes=3, num_neg=1, eval_every=1),
        SweepRun(kind, ModelConfig(temporal_model=kind, node_dim=8, time_emb_dim=4,
                                  temporal_num_neighbors=2, dropout=0.)),
        factory, epochs=1, rollout_horizon=2,
        objective_metric=TaskMetricSpec("val.event_auroc", "max"),
    )
    assert result.final_snapshot["evaluation_protocol"] == "continuous_history_replay_v1"
    # Validation contains just one bin: it must still be scored using train history.
    assert math.isfinite(result.final_snapshot["val"]["loss"])
    if kind == "jodie":
        reference = DyGLibAdapter(3, 1, ModelConfig(temporal_model=kind, node_dim=8, time_emb_dim=4))
        reference.configure_training_history(ds.bins("train"))
        torch.testing.assert_close(models[0].src_node_mean_time_shift, reference.src_node_mean_time_shift)
