from __future__ import annotations

import argparse

import pytest
import torch

from interactiondynamics.data.synthetic import SYNTHETIC_TASKS
from interactiondynamics.training.presets import (
    DIFFUSION_COMPARISON_PANEL,
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
        {("ift", "ift_update"), *DIFFUSION_COMPARISON_PANEL}
        if task_name == "diffusion"
        else set(SYNTHETIC_TASKS[task_name].recommended_pairs)
    )
    assert task_pairs == expected_pairs
    if SYNTHETIC_TASKS[task_name].requires_node_scorer:
        assert all(run.model_cfg.use_node_scorer for run in suite.runs)

    ds = load_dataset(suite.dataset, suite.dataset_kwargs)
    assert ds.spec().num_nodes == 20


def test_diffusion_task_uses_the_fixed_first_order_comparison_panel() -> None:
    args = _synthetic_args("diffusion")
    suite = build_suite(
        "quick",
        torch.device("cpu"),
        dataset_override="synthetic",
        args=args,
    )

    assert [run.name for run in suite.runs[:3]] == [
        "ift1_generic",
        "ift1_linear",
        "ift1_direct",
    ]
    pairs = {(run.model_cfg.aggregator, run.model_cfg.update) for run in suite.runs}
    assert pairs == {("ift", "ift_update"), *DIFFUSION_COMPARISON_PANEL}
    assert len(suite.runs) == 3 + len(DIFFUSION_COMPARISON_PANEL)


def test_quick_synthetic_suite_can_replace_default_ift_run_with_selected_variants() -> None:
    args = _synthetic_args("wave")
    args.ift_variants = ["linear", "auto"]
    args.ift_history_steps = [3]
    suite = build_suite(
        "quick",
        torch.device("cpu"),
        dataset_override="synthetic",
        args=args,
    )

    run_names = [run.name for run in suite.runs]
    assert run_names == [
        "ift1_linear",
        "ift2_linear",
        "ift2_auto",
        "ift2_hist_vel_k3",
        "agg=sum|update=tgn_gru|do=0.0|sdo=0.0|time=False",
        "agg=sum|update=lnn|do=0.0|sdo=0.0|time=False",
        "agg=sum|update=hnn|do=0.0|sdo=0.0|time=False",
    ]
    assert suite.runs[3].prediction_mode == "delta"
    assert suite.runs[3].lr == 1e-2
    assert suite.runs[3].model_cfg.ift_history_vel_steps == 3


def test_quick_synthetic_suite_expands_empty_ift_variant_selection_to_task_defaults() -> None:
    args = _synthetic_args("diffusion")
    args.ift_variants = []
    suite = build_suite(
        "quick",
        torch.device("cpu"),
        dataset_override="synthetic",
        args=args,
    )

    assert [run.name for run in suite.runs[:10]] == [
        "ift1_generic",
        "ift1_linear",
        "ift1_direct",
        "ift2_generic",
        "ift2_linear",
        "ift2_direct",
        "ift2_auto",
        "ift2_hist_vel_k1",
        "ift2_hist_vel_k2",
        "ift2_hist_vel_k3",
    ]


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


def test_ring_wave_suite_can_add_drive_free_and_self_rollouts() -> None:
    args = _synthetic_args("wave")
    args.ift_variants = ["auto"]
    args.ift_orders = [2]
    args.ift_history_steps = [1]
    args.ift_free_rollout = True
    args.ift_self_rollout = True
    suite = build_suite(
        "quick",
        torch.device("cpu"),
        dataset_override="synthetic",
        args=args,
    )

    free_run = next(run for run in suite.runs if run.name == "ift2_free_hist_vel_k1")
    self_run = next(run for run in suite.runs if run.name == "ift2_self_hist_vel_k1")
    assert free_run.model_cfg.ift_rollout_free_drive
    assert not free_run.model_cfg.ift_rollout_self_generated
    assert self_run.model_cfg.ift_rollout_self_generated


def test_diffusion_suite_can_add_drive_free_and_self_rollouts() -> None:
    args = _synthetic_args("diffusion")
    args.ift_variants = ["auto"]
    args.ift_orders = [2]
    args.ift_history_steps = [1]
    args.ift_free_rollout = True
    args.ift_self_rollout = True
    suite = build_suite(
        "quick",
        torch.device("cpu"),
        dataset_override="synthetic",
        args=args,
    )

    run_names = {run.name for run in suite.runs}
    assert "ift2_free_hist_vel_k1" in run_names
    assert "ift2_self_hist_vel_k1" in run_names
    free_run = next(run for run in suite.runs if run.name == "ift2_free_hist_vel_k1")
    self_run = next(run for run in suite.runs if run.name == "ift2_self_hist_vel_k1")
    assert free_run.model_cfg.ift_rollout_free_drive
    assert self_run.model_cfg.ift_rollout_self_generated


def test_quick_synthetic_suite_rejects_auto_without_second_order() -> None:
    args = _synthetic_args("diffusion")
    args.ift_variants = ["auto"]
    args.ift_orders = [1]

    with pytest.raises(ValueError, match="require including second-order IFT"):
        build_suite(
            "quick",
            torch.device("cpu"),
            dataset_override="synthetic",
            args=args,
        )


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
