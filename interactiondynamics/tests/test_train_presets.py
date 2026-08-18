from __future__ import annotations

import argparse

import pytest
import torch

from interactiondynamics.data.synthetic import SYNTHETIC_TASKS
from interactiondynamics.training.presets import (
    DIFFUSION_COMPARISON_PANEL,
    FIELD_COMPARISON_PANEL,
    PHYSICAL_EVENT_COMPARISON_PANEL,
    build_suite,
    load_dataset,
)


def _synthetic_args(task_name: str) -> argparse.Namespace:
    return argparse.Namespace(
        dataset="synthetic",
        synthetic_task=task_name,
        synthetic_num_nodes=20,
        synthetic_events_per_bin=30,
        num_bins=18,
        seed=3,
        ift_variants=None,
        ift_orders=None,
        ift_history_steps=None,
    )


@pytest.mark.parametrize("task_name", list(SYNTHETIC_TASKS.keys()))
def test_quick_synthetic_suite_uses_task_specific_shortlist(task_name: str):
    args = _synthetic_args(task_name)
    suite = build_suite(
        "quick",
        torch.device("cpu"),
        dataset_override="synthetic",
        args=args,
    )

    assert suite.dataset == "synthetic"
    assert suite.dataset_kwargs["task"] == task_name

    task_pairs = {(run.model_cfg.aggregator, run.model_cfg.update) for run in suite.runs}
    expected_pairs = (
        set(PHYSICAL_EVENT_COMPARISON_PANEL)
        if task_name in {"diffusion", "wave", "coupled_oscillator"}
        else set(SYNTHETIC_TASKS[task_name].recommended_pairs)
    )
    assert task_pairs == expected_pairs
    if SYNTHETIC_TASKS[task_name].requires_node_scorer:
        assert all(run.model_cfg.use_node_scorer for run in suite.runs)

    ds = load_dataset(suite.dataset, suite.dataset_kwargs)
    assert ds.spec().num_nodes == 20


def test_physical_tasks_use_the_shared_event_prediction_panel() -> None:
    args = _synthetic_args("wave")
    suite = build_suite(
        "quick",
        torch.device("cpu"),
        dataset_override="synthetic",
        args=args,
    )

    assert [run.name for run in suite.runs] == [
        "fnn",
        "sum/tgn_gru",
        "deepsets/tgn_gru",
        "settransformer/tgn_gru",
        "hopfield/hopfield_update",
        "settransformer/lnn",
        "settransformer/hnn",
    ]
    pairs = {(run.model_cfg.aggregator, run.model_cfg.update) for run in suite.runs}
    assert pairs == set(PHYSICAL_EVENT_COMPARISON_PANEL)
    assert all(run.model_cfg.predict_event_features for run in suite.runs)
    assert suite.runs[0].model_cfg.fnn


def test_wave_uses_the_shared_event_panel() -> None:
    task_name = "wave"
    args = _synthetic_args(task_name)
    suite = build_suite(
        "quick",
        torch.device("cpu"),
        dataset_override="synthetic",
        args=args,
    )

    pairs = {(run.model_cfg.aggregator, run.model_cfg.update) for run in suite.runs}
    assert pairs == set(PHYSICAL_EVENT_COMPARISON_PANEL)


def test_grid_wave_suite_can_add_a_closed_loop_self_rollout() -> None:
    args = _synthetic_args("wave_grid")
    args.ift_variants = ["auto"]
    args.ift_orders = [2]
    args.ift_history_steps = [1]
    args.ift_self_rollout = True
    suite = build_suite(
        "quick",
        torch.device("cpu"),
        dataset_override="synthetic",
        args=args,
    )

    self_run = next(run for run in suite.runs if run.name == "ift2_self_hist_vel_k1")
    assert self_run.prediction_mode == "delta"
    assert self_run.model_cfg.ift_rollout_self_generated
    assert self_run.model_cfg.ift_history_vel_steps == 1


def test_quick_synthetic_suite_allows_variant_selection_on_non_ift_task() -> None:
    args = _synthetic_args("deepsets_sum")
    args.ift_variants = ["direct"]
    args.ift_orders = [1]
    suite = build_suite(
        "quick",
        torch.device("cpu"),
        dataset_override="synthetic",
        args=args,
    )

    assert [run.name for run in suite.runs] == [
        "ift1_direct",
        "agg=settransformer|update=tgn_gru|do=0.0|sdo=0.0|time=False",
        "agg=sum|update=tgn_gru|do=0.0|sdo=0.0|time=False",
        "agg=deepsets|update=tgn_gru|do=0.0|sdo=0.0|time=False",
    ]
    assert suite.runs[0].model_cfg.ift_drive_feature_idx == 0


def test_quick_synthetic_suite_expands_empty_ift_variant_selection_on_zero_event_task() -> None:
    args = _synthetic_args("node_count_threshold")
    args.ift_variants = []
    suite = build_suite(
        "quick",
        torch.device("cpu"),
        dataset_override="synthetic",
        args=args,
    )

    assert [run.name for run in suite.runs] == [
        "ift1_generic",
        "ift2_generic",
        "agg=settransformer|update=tgn_gru|do=0.0|sdo=0.0|time=False",
        "agg=sum|update=tgn_gru|do=0.0|sdo=0.0|time=False",
        "agg=deepsets|update=tgn_gru|do=0.0|sdo=0.0|time=False",
    ]


def test_quick_synthetic_suite_rejects_unsupported_variants_for_zero_event_task() -> None:
    args = _synthetic_args("edge_threshold_classification")
    args.ift_variants = ["linear"]

    with pytest.raises(ValueError, match="Supported variants: generic"):
        build_suite(
            "quick",
            torch.device("cpu"),
            dataset_override="synthetic",
            args=args,
        )
