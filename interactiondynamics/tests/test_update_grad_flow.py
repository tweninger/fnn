from __future__ import annotations

from typing import cast

import pytest
import torch

from interactiondynamics.core.config import ModelConfig, UpdateType
from interactiondynamics.core.interfaces import ModelState
from interactiondynamics.data.synthetic import SyntheticDataset, SyntheticDatasetConfig
from interactiondynamics.models.tgn_model import build_tgn_model
from interactiondynamics.training.targets import edge_regression_loss
from interactiondynamics.training.types import TrainConfig
from interactiondynamics.updates.ift_update import IFTDiffusionUpdate


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
        update=cast(UpdateType, update_name),
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


def test_build_tgn_model_passes_ift_kappa_cap_and_max() -> None:
    spec = SyntheticDataset(
        SyntheticDatasetConfig(
            task="temporal_memory",
            num_nodes=8,
            num_bins=4,
            events_per_bin=8,
            seed=0,
            device=torch.device("cpu"),
        )
    ).spec()

    model = build_tgn_model(
        spec,
        ModelConfig(
            node_dim=32,
            msg_dim=32,
            event_dim=spec.event_dim,
            aggregator="ift",
            update="ift_update",
            ift_kappa_cap=True,
            ift_kappa_max=2.5,
        ),
    )

    assert isinstance(model.update, IFTDiffusionUpdate)
    assert model.update.kappa_cap is True
    assert model.update.kappa_max == 2.5


def test_first_order_ift_force_mask_suppresses_learned_forcing() -> None:
    update = IFTDiffusionUpdate(
        node_dim=1,
        msg_dim=1,
        dt=1.0,
        gamma=0.0,
        kappa=0.0,
        learn_kappa=False,
        inj_clip=None,
        forcing_mode="generic_mlp",
    )
    with torch.no_grad():
        update.force_encoder.msg_proj.weight.zero_()
        update.force_encoder.msg_proj.bias.fill_(2.0)

    messages = torch.zeros((2, 1))
    driven_state = ModelState(node=torch.zeros((2, 1)), aux={})
    free_state = ModelState(node=torch.zeros((2, 1)), aux={"ift_force_mask": torch.tensor(0.0)})

    driven_next, _ = update(driven_state, messages)
    free_next, _ = update(free_state, messages)

    assert driven_next is not None and free_next is not None
    assert torch.allclose(driven_next.node, torch.full((2, 1), 2.0))
    assert torch.allclose(free_next.node, torch.zeros((2, 1)))
