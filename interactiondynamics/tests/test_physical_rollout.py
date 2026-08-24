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
