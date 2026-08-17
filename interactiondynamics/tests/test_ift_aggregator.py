from __future__ import annotations

from typing import cast

import torch

from interactiondynamics.aggregators.ift import IFTLaplacianAggregator
from interactiondynamics.core.config import ModelConfig
from interactiondynamics.core.events import EventBatch
from interactiondynamics.core.interfaces import ModelState
from interactiondynamics.data.interfaces import DataSpec
from interactiondynamics.models.model_factory import build_model
from interactiondynamics.scorers.event_scorer import IFTSecondOrderLinearHVForceScorer
from interactiondynamics.training.targets import edge_regression_readout
from interactiondynamics.training.types import TrainConfig
from interactiondynamics.updates.ift_update import IFTSecondOrderUpdate


def _events(src: list[int], dst: list[int], t: int) -> EventBatch:
    n = len(src)
    return EventBatch(
        src=cast(torch.LongTensor, torch.tensor(src, dtype=torch.long)),
        dst=cast(torch.LongTensor, torch.tensor(dst, dtype=torch.long)),
        t=cast(torch.LongTensor, torch.full((n,), t, dtype=torch.long)),
    )


def test_ift_current_bin_mode_rebuilds_operator_from_only_current_events() -> None:
    aggregator = IFTLaplacianAggregator(
        add_to_dst=True,
        add_to_src=False,
        make_undirected=False,
        laplacian_mode="current_bin",
        ema_beta=0.9,
    )
    state = ModelState(node=torch.zeros(3, 4), aux={})
    msg = torch.randn(1, 4)

    _ = aggregator(state, msg, _events([0], [1], t=0), num_nodes=3)
    _ = aggregator(state, msg, _events([1], [2], t=1), num_nodes=3)

    assert state.aux is not None
    assert "A_ema" not in state.aux
    expected = aggregator._laplacian_from_adjacency(  # pylint: disable=protected-access
        aggregator._adjacency_from_events(  # pylint: disable=protected-access
            torch.tensor([1], dtype=torch.long),
            torch.tensor([2], dtype=torch.long),
            num_nodes=3,
        ),
        num_nodes=3,
    )
    assert torch.allclose(state.aux["L"].to_dense(), expected.to_dense())
    assert state.aux["ift_laplacian_mode"] == "current_bin"


def test_ift_ema_mode_blends_previous_and_current_adjacency() -> None:
    aggregator = IFTLaplacianAggregator(
        add_to_dst=True,
        add_to_src=False,
        make_undirected=False,
        laplacian_mode="ema",
        ema_beta=0.5,
    )
    state = ModelState(node=torch.zeros(3, 4), aux={})
    msg = torch.randn(1, 4)

    _ = aggregator(state, msg, _events([0], [1], t=0), num_nodes=3)
    _ = aggregator(state, msg, _events([1], [2], t=1), num_nodes=3)

    assert state.aux is not None
    assert "A_ema" in state.aux
    A_ema = state.aux["A_ema"].to_dense()
    expected = torch.tensor(
        [
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.0],
        ]
    )
    assert torch.allclose(A_ema, expected)
    assert state.aux["ift_laplacian_mode"] == "ema"


def test_build_tgn_model_passes_ift_laplacian_mode_and_beta() -> None:
    spec = DataSpec(name="tiny", num_nodes=4, event_dim=0)
    model = build_model(
        spec,
        ModelConfig(
            aggregator="ift",
            update="ift_update",
            ift_laplacian_mode="ema",
            ift_laplacian_beta=0.8,
        ),
    )

    assert isinstance(model.aggregator, IFTLaplacianAggregator)
    assert model.aggregator.laplacian_mode == "ema"
    assert model.aggregator.ema_beta == 0.8


def test_build_tgn_model_passes_ift_diagnostic_knobs() -> None:
    spec = DataSpec(name="tiny", num_nodes=4, event_dim=2)
    model = build_model(
        spec,
        ModelConfig(
            aggregator="ift",
            update="ift_update",
            ift_laplacian_mode="fixed_ring",
            ift_message_reduce="sum",
            ift_disable_laplacian=False,
            ift_randomize_laplacian=True,
            ift_zero_messages=True,
            ift_zero_injection=True,
            ift_inj_clip=None,
            ift_direct_drive=True,
        ),
    )

    assert isinstance(model.aggregator, IFTLaplacianAggregator)
    assert model.aggregator.laplacian_mode == "fixed_ring"
    assert model.aggregator.message_reduce == "sum"
    assert model.aggregator.randomize_laplacian is True
    assert model.aggregator.zero_messages is True
    assert model.aggregator.direct_drive is True
    assert model.update.zero_injection is True
    assert model.update.inj_clip is None
    assert model.update.direct_drive is True


def test_build_tgn_model_passes_structured_forcing_and_second_order() -> None:
    spec = DataSpec(name="tiny", num_nodes=4, event_dim=2)
    model = build_model(
        spec,
        ModelConfig(
            aggregator="ift",
            update="ift_update",
            ift_forcing_mode="gated_direct_scalar",
            ift_drive_feature_idx=0,
            ift_force_scale_init=0.5,
            ift_force_learn_scale=False,
            ift_force_target_dim=None,
            ift_force_reduce="sum",
            ift_update_order="second",
            ift_second_order_alpha=0.9,
            ift_second_order_dt=0.2,
            ift_second_order_gamma=0.1,
            ift_second_order_kappa=0.7,
            ift_second_order_learn_params=False,
        ),
    )

    assert isinstance(model.update, IFTSecondOrderUpdate)
    assert model.aggregator.force_reduce == "sum"
    assert model.update.forcing_mode == "gated_direct_scalar"
    assert model.update.force_encoder.drive_feature_idx == 0
    assert model.update.force_encoder.force_target_dim is None


def test_ift_direct_drive_summary_is_stashed_in_aux() -> None:
    aggregator = IFTLaplacianAggregator(
        add_to_dst=True,
        add_to_src=False,
        make_undirected=False,
        laplacian_mode="current_bin",
        direct_drive=True,
    )
    state = ModelState(node=torch.zeros(3, 4), aux={})
    events = EventBatch(
        src=cast(torch.LongTensor, torch.tensor([0, 1, 2], dtype=torch.long)),
        dst=cast(torch.LongTensor, torch.tensor([1, 2, 2], dtype=torch.long)),
        features=torch.tensor(
            [
                [1.5, 0.0],
                [2.0, 1.0],
                [3.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        t=cast(torch.LongTensor, torch.zeros(3, dtype=torch.long)),
    )

    _ = aggregator(state, torch.randn(3, 4), events, num_nodes=3)

    assert state.aux is not None
    assert "ift_direct_drive" in state.aux
    expected = torch.tensor([[0.0], [0.0], [5.0]])
    assert torch.allclose(state.aux["ift_direct_drive"], expected)


def test_ift_force_feature_summary_is_stashed_in_aux() -> None:
    aggregator = IFTLaplacianAggregator(
        add_to_dst=True,
        add_to_src=False,
        make_undirected=False,
        laplacian_mode="current_bin",
        force_reduce="sum",
    )
    state = ModelState(node=torch.zeros(3, 4), aux={})
    events = EventBatch(
        src=cast(torch.LongTensor, torch.tensor([0, 1, 2], dtype=torch.long)),
        dst=cast(torch.LongTensor, torch.tensor([1, 2, 2], dtype=torch.long)),
        features=torch.tensor(
            [
                [1.5, 0.0],
                [2.0, 1.0],
                [3.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        t=cast(torch.LongTensor, torch.zeros(3, dtype=torch.long)),
    )

    _ = aggregator(state, torch.randn(3, 4), events, num_nodes=3)

    assert state.aux is not None
    assert "ift_force_features" in state.aux
    expected = torch.tensor(
        [
            [0.0, 0.0],
            [1.5, 0.0],
            [5.0, 2.0],
        ]
    )
    assert torch.allclose(state.aux["ift_force_features"], expected)


def test_model_step_preserves_ema_history_for_tgn_gru() -> None:
    spec = DataSpec(name="tiny", num_nodes=3, event_dim=0)
    model = build_model(
        spec,
        ModelConfig(
            node_dim=8,
            msg_dim=8,
            event_dim=0,
            aggregator="ift",
            update="tgn_gru",
            ift_laplacian_mode="ema",
            ift_laplacian_beta=0.5,
        ),
    )

    state = model.init_state(batch_size=1, num_nodes=spec.num_nodes, device=torch.device("cpu"))
    state, _ = model.step(state, _events([0], [1], t=0))
    assert state is not None
    assert state.aux is not None
    assert "A_ema" in state.aux

    state, _ = model.step(state, _events([1], [2], t=1))

    assert state is not None
    assert state.aux is not None
    assert state.aux["ift_laplacian_mode"] == "ema"
    A_ema = state.aux["A_ema"].to_dense()
    expected = torch.tensor(
        [
            [0.0, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.0],
        ]
    )
    assert torch.allclose(A_ema, expected)


def test_model_step_preserves_ema_history_for_ift_update() -> None:
    spec = DataSpec(name="tiny", num_nodes=3, event_dim=0)
    model = build_model(
        spec,
        ModelConfig(
            node_dim=8,
            msg_dim=8,
            event_dim=0,
            aggregator="ift",
            update="ift_update",
            ift_laplacian_mode="ema",
            ift_laplacian_beta=0.5,
            ift_kappa=1.0,
            ift_learn_kappa=False,
        ),
    )

    state = model.init_state(batch_size=1, num_nodes=spec.num_nodes, device=torch.device("cpu"))
    state, aux = model.step(state, _events([0], [1], t=0))
    assert state is not None
    assert state.aux is not None
    assert "A_ema" in state.aux
    assert "kappa" in aux

    state, aux = model.step(state, _events([1], [2], t=1))

    assert state is not None
    assert state.aux is not None
    assert state.aux["ift_laplacian_mode"] == "ema"
    A_ema = state.aux["A_ema"].to_dense()
    expected = torch.tensor(
        [
            [0.0, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.0],
        ]
    )
    assert torch.allclose(A_ema, expected)
    assert torch.allclose(aux["kappa"], torch.tensor(1.0))


def test_second_order_ift_keeps_velocity_in_node_prev() -> None:
    spec = DataSpec(name="tiny", num_nodes=3, event_dim=1)
    model = build_model(
        spec,
        ModelConfig(
            node_dim=8,
            msg_dim=8,
            event_dim=1,
            aggregator="ift",
            update="ift_update",
            ift_update_order="second",
            ift_forcing_mode="direct_scalar",
            ift_drive_feature_idx=0,
            ift_second_order_learn_params=False,
        ),
    )
    events = EventBatch(
        src=cast(torch.LongTensor, torch.tensor([0, 1, 2], dtype=torch.long)),
        dst=cast(torch.LongTensor, torch.tensor([0, 1, 2], dtype=torch.long)),
        features=torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float32),
        t=cast(torch.LongTensor, torch.zeros(3, dtype=torch.long)),
    )

    state = model.init_state(batch_size=1, num_nodes=spec.num_nodes, device=torch.device("cpu"))
    state, aux = model.step(state, events)

    assert state is not None
    assert state.node is not None
    assert state.node_prev is not None
    assert state.node.shape == (spec.num_nodes, 8)
    assert state.node_prev.shape == (spec.num_nodes, 8)
    assert "alpha" in aux
    assert "v_norm" in aux


def test_second_order_ift_initializes_velocity_from_finite_difference_history() -> None:
    update = IFTSecondOrderUpdate(
        node_dim=4,
        msg_dim=4,
        event_dim=1,
        learn_params=False,
        zero_injection=True,
        velocity_init_mode="finite_difference",
        velocity_supervision=True,
    )
    state = update.init_state(batch_size=1, num_nodes=3, device=torch.device("cpu"))
    assert state is not None
    assert state.aux is not None
    state.aux["ift_state_observed_target"] = torch.tensor([1.0, 2.0, 3.0])
    state.aux["ift_prev_observed_target"] = torch.tensor([0.5, 1.5, 2.5])

    next_state, aux = update(state, torch.zeros(3, 4))

    assert next_state is not None
    assert next_state.node_prev is not None
    assert float(next_state.node_prev.abs().sum().item()) > 0.0
    assert "decoded_velocity" in aux


def test_edge_regression_readout_supports_delta_prediction_modes() -> None:
    preds = torch.tensor([0.2, -0.1], dtype=torch.float32)
    curr = torch.tensor([1.3, 0.5], dtype=torch.float32)
    prev = torch.tensor([1.0, 0.7], dtype=torch.float32)

    delta_cfg = TrainConfig(num_nodes=2, prediction_mode="delta")
    delta_view = edge_regression_readout(preds, curr, prev, delta_cfg)
    assert torch.allclose(delta_view.raw_preds, prev + preds)
    assert torch.allclose(delta_view.loss_targets, curr - prev)

    spd_cfg = TrainConfig(num_nodes=2, prediction_mode="state_plus_delta")
    spd_view = edge_regression_readout(preds, curr, prev, spd_cfg)
    assert torch.allclose(spd_view.raw_preds, prev + preds)
    assert torch.allclose(spd_view.loss_preds, prev + preds)
    assert torch.allclose(spd_view.loss_targets, curr)


def test_build_tgn_model_uses_linear_h_v_force_scorer_for_second_order_ift() -> None:
    spec = DataSpec(
        name="osc",
        num_nodes=3,
        event_dim=1,
        extra={"generator_params": {"a": 1.92, "b": -0.96, "c": 0.08}},
    )
    model = build_model(
        spec,
        ModelConfig(
            aggregator="ift",
            update="ift_update",
            ift_update_order="second",
            ift2_readout_mode="linear_h_v_force",
            ift2_oracle_init=True,
        ),
    )
    assert isinstance(model.scorer, IFTSecondOrderLinearHVForceScorer)


def test_linear_h_v_force_scorer_can_match_oracle_delta() -> None:
    scorer = IFTSecondOrderLinearHVForceScorer(oracle_coeffs=(1.92, -0.96, 0.08))
    state = ModelState(
        node=torch.zeros(3, 4),
        aux={
            "ift_readout_position_scalar": torch.tensor([1.0, 0.5, -0.25]),
            "ift_readout_velocity_scalar": torch.tensor([0.2, -0.1, 0.4]),
            "ift_readout_force_scalar": torch.tensor([0.3, -0.2, 0.1]),
        },
    )
    events = EventBatch(
        src=cast(torch.LongTensor, torch.tensor([0, 1, 2], dtype=torch.long)),
        dst=cast(torch.LongTensor, torch.tensor([0, 1, 2], dtype=torch.long)),
        t=cast(torch.LongTensor, torch.zeros(3, dtype=torch.long)),
    )
    delta = scorer(state, events)
    assert state.aux is not None
    pos = cast(torch.Tensor, state.aux["ift_readout_position_scalar"])
    vel = cast(torch.Tensor, state.aux["ift_readout_velocity_scalar"])
    force = cast(torch.Tensor, state.aux["ift_readout_force_scalar"])
    expected = (
        (1.92 - 0.96 - 1.0) * pos
        - (-0.96) * vel
        + 0.08 * force
    )
    assert torch.allclose(delta, expected)


def test_linear_h_v_force_scorer_supports_near_ar1_init() -> None:
    scorer = IFTSecondOrderLinearHVForceScorer(
        init_mode="near_ar1",
        near_ar1_coeffs=(0.25, 0.0, 0.08, 0.0),
    )
    coeffs = scorer.coefficient_dict()
    assert abs(coeffs["w_y"] - 0.25) < 1e-6
    assert abs(coeffs["w_v"] - 0.0) < 1e-6
    assert abs(coeffs["w_drive"] - 0.08) < 1e-6
    assert abs(coeffs["bias"] - 0.0) < 1e-6


def test_build_tgn_model_allows_trainable_oracle_initialized_linear_readout() -> None:
    spec = DataSpec(
        name="osc",
        num_nodes=3,
        event_dim=1,
        extra={"generator_params": {"a": 1.92, "b": -0.96, "c": 0.08}},
    )
    model = build_model(
        spec,
        ModelConfig(
            aggregator="ift",
            update="ift_update",
            ift_update_order="second",
            ift2_readout_mode="linear_h_v_force",
            ift2_oracle_init=True,
            ift2_readout_trainable=True,
            ift2_readout_init_mode="oracle",
        ),
    )
    assert isinstance(model.scorer, IFTSecondOrderLinearHVForceScorer)
    params = list(model.scorer.parameters())
    assert params
    assert all(param.requires_grad for param in params)
