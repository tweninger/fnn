import pytest
import torch

from interactiondynamics.core.config import ModelConfig
from interactiondynamics.core.events import EventBatch
from interactiondynamics.models.dyglib_adapter import DyGLibAdapter, EdgeBankAdapter


def events(time, destinations=(1, 2)):
    return EventBatch(src=torch.tensor([0, 1]), dst=torch.tensor(destinations),
                      t=torch.tensor([time, time]), features=torch.ones(2, 1))


@pytest.mark.parametrize("kind", ["graphmixer", "tgn", "dygformer", "jodie"])
def test_temporal_adapter_gradients_and_isolated_history(kind):
    torch.manual_seed(2)
    cfg = ModelConfig(temporal_model=kind, node_dim=8, time_emb_dim=4,
                      temporal_num_neighbors=2, temporal_history_length=4, dropout=0.)
    model = DyGLibAdapter(4, 1, cfg)
    state = model.init_state(1, 4, torch.device("cpu"))
    for time in range(3):
        state, _ = model.step(state, events(time))
        snapshot = state.clone(detach=True)
        query = events(time + 1, (2, 3))
        scores = model.score(state, query)
        torch.testing.assert_close(scores, model.score(state, query))
        # Target features are not part of the query representation.
        query.features.fill_(1000)
        torch.testing.assert_close(scores, model.score(state, query))
        loss = scores.square().sum() + model.predict_event_features(state, query).square().sum()
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.backbone.parameters())
        if kind in {"tgn", "jodie"} and time > 0:
            assert any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.backbone.memory_updater.memory_updater.parameters())
        state.detach_()
        model.zero_grad()
        model.eval()
        before = model.score(snapshot, query).detach()
        other = model.init_state(1, 4, torch.device("cpu"))
        other, _ = model.step(other, events(99))
        model.score(other, events(100))
        torch.testing.assert_close(before, model.score(snapshot, query))
        model.train()


def test_edgebank_has_no_parameters_and_scores_observed_pairs_only():
    model = EdgeBankAdapter(4, 1)
    assert list(model.parameters()) == []
    state = model.init_state(1, 4, torch.device("cpu"))
    old = state.clone()
    state, _ = model.step(state, events(0))
    torch.testing.assert_close(model.score(state, events(1)), torch.ones(2))
    torch.testing.assert_close(model.score(old, events(1)), torch.zeros(2))
    torch.testing.assert_close(model.score(state, events(1, (2, 3))), torch.zeros(2))


@pytest.mark.parametrize("kind", ["edgebank", "graphmixer", "tgn", "dygformer", "jodie"])
def test_temporal_model_full_experiment(kind, tmp_path):
    from interactiondynamics.data.synthetic import SyntheticDataset, SyntheticDatasetConfig
    from interactiondynamics.models.model_factory import build_model
    from interactiondynamics.training.runner import run_one_experiment
    from interactiondynamics.training.types import SweepRun, TrainConfig
    from interactiondynamics.training.task_metrics import TaskMetricSpec
    ds = SyntheticDataset(SyntheticDatasetConfig(task="wave", num_nodes=4,
                          num_bins=5, num_episodes=5, events_per_bin=9))
    cfg = ModelConfig(temporal_model=kind, node_dim=8, time_emb_dim=4,
                      temporal_num_neighbors=2, temporal_history_length=4,
                      predict_event_features=kind != "edgebank", dropout=0.)
    result = run_one_experiment(
        ds, ds.spec(), TrainConfig(num_nodes=4, num_neg=2, eval_every=1),
        SweepRun(kind, cfg), build_model, epochs=2, rollout_horizon=2,
        objective_metric=TaskMetricSpec("val.event_auroc", "max"),
        save_jsonl_path=str(tmp_path / f"{kind}.jsonl"),
    )
    assert result.best_epoch > 0
    assert result.epochs == (1 if kind == "edgebank" else 2)
    assert "event_auroc" in result.final_snapshot["test"]
