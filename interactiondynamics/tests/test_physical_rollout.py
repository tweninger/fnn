from types import SimpleNamespace

import torch

from interactiondynamics.core.events import EventBatch
from interactiondynamics.eval.evaluate import evaluate_physical_force_rollout
from interactiondynamics.models.fnn import FieldNeuralNetwork


def test_physical_rollout_treats_episode_free_stream_as_continuous() -> None:
    model = FieldNeuralNetwork(
        num_nodes=3,
        force_dim=1,
        state_dim=1,
        gamma_init=0.1,
        omega_init=0.5,
        dt=0.1,
    )
    bins = [
        EventBatch(
            src=torch.tensor([step % 3]),
            dst=torch.tensor([(step + 1) % 3]),
            features=torch.ones((1, 1)),
            t=torch.tensor([step]),
        )
        for step in range(4)
    ]
    cfg = SimpleNamespace(device="cpu", num_nodes=3, num_neg=1)

    metrics = evaluate_physical_force_rollout(model, bins, cfg, horizon=2)

    assert metrics["rollout_horizon"] == 2.0
    assert metrics["rollout_steps"] == 2.0
    assert "rollout_force_mse" in metrics


def test_state_score_head_is_an_opt_in_topology_residual() -> None:
    common = dict(
        num_nodes=3,
        force_dim=1,
        state_dim=1,
        gamma_init=0.1,
        omega_init=0.5,
        dt=0.1,
    )
    static_model = FieldNeuralNetwork(**common)
    dynamic_model = FieldNeuralNetwork(**common, state_score=True)
    events = EventBatch(src=torch.tensor([0]), dst=torch.tensor([1]))
    state = static_model.init_state(batch_size=1, num_nodes=3, device=torch.device("cpu"))

    assert torch.equal(static_model.score(state, events), static_model._topology_logits_for(events.src, events.dst))
    assert dynamic_model.state_score_head is not None
    final_layer = dynamic_model.state_score_head[-1]
    assert isinstance(final_layer, torch.nn.Linear)
    final_layer.bias.data.fill_(2.0)
    dynamic_state = dynamic_model.init_state(batch_size=1, num_nodes=3, device=torch.device("cpu"))
    assert torch.allclose(
        dynamic_model.score(dynamic_state, events),
        dynamic_model._topology_logits_for(events.src, events.dst) + 2.0,
    )


def test_fnn_dt_omega_bound_is_differentiable_and_strict() -> None:
    model = FieldNeuralNetwork(
        num_nodes=3,
        force_dim=1,
        state_dim=1,
        gamma_init=0.1,
        omega_init=0.7,
        dt=0.1,
        learn_dt=True,
        learn_omega=True,
        max_dt_omega=1.5,
    )
    assert model.dt_raw.requires_grad
    with torch.no_grad():
        model.dt_raw.fill_(30.0)
    params = model.physical_parameters()
    assert float((params["dt"] * params["omega"]).item()) < 1.5
