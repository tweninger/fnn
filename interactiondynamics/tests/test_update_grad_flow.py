from __future__ import annotations

import pytest
import torch

from interactiondynamics.core.config import ModelConfig
from interactiondynamics.data.synthetic import SyntheticDataset, SyntheticDatasetConfig
from interactiondynamics.models.tgn_model import build_tgn_model
from interactiondynamics.training.targets import edge_regression_loss
from interactiondynamics.training.types import TrainConfig


def _update_grad_norms(update_name: str) -> list[float]:
    device = torch.device("cpu")
    ds = SyntheticDataset(
        SyntheticDatasetConfig(
            task="temporal_memory",
            num_nodes=16,
            num_bins=8,
            events_per_bin=16,
            seed=0,
            device=device,
        )
    )
    spec = ds.spec()
    model_cfg = ModelConfig(
        node_dim=64,
        msg_dim=64,
        event_dim=spec.event_dim,
        scorer="mlp",
        scorer_hidden=128,
        aggregator="sum",
        update=update_name,
        dropout=0.0,
        scorer_dropout=0.0,
        encoder_hidden=128,
    )
    model = build_tgn_model(spec, model_cfg).to(device)
    model.train()

    cfg = TrainConfig(num_nodes=spec.num_nodes, num_neg=4, device=device)

    bins = iter(ds.bins("train"))
    edge_targets = iter(ds.edge_targets("train") or [])
    prev = next(bins).to(device)
    _ = next(edge_targets)
    curr_edge_target = next(edge_targets)

    state = model.init_state(batch_size=1, num_nodes=cfg.num_nodes, device=device)
    state, _ = model.step(state, prev)
    preds = model.score(state, curr_edge_target.events.to(device))
    targets = curr_edge_target.targets.to(device)
    loss = edge_regression_loss(preds, targets, cfg)

    model.zero_grad(set_to_none=True)
    loss.backward()

    grad_norms: list[float] = []
    for param in model.update.parameters():
        if param.grad is not None:
            grad_norms.append(float(param.grad.norm().item()))
    return grad_norms


@pytest.mark.parametrize("update_name", ["lnn", "hnn"])
def test_update_laws_receive_gradient_from_scoring_loss(update_name: str):
    grad_norms = _update_grad_norms(update_name)

    assert grad_norms, f"{update_name} update parameters received no gradients"
    assert max(grad_norms) > 0.0, f"{update_name} update gradients were all zero"
