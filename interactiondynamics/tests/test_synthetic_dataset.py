from __future__ import annotations

import pytest
import torch

from interactiondynamics.data.synthetic import (
    IFT_PAIR,
    SYNTHETIC_TASKS,
    SyntheticDataset,
    SyntheticDatasetConfig,
    group_synthetic_tasks_by,
    list_synthetic_tasks,
    synthetic_task_axes,
    synthetic_task_tags,
)


@pytest.mark.parametrize("task_name", list(SYNTHETIC_TASKS.keys()))
def test_synthetic_dataset_materializes_expected_supervision(task_name: str):
    cfg = SyntheticDatasetConfig(
        task=task_name,
        num_nodes=16,
        num_bins=20,
        events_per_bin=24,
        seed=7,
    )
    ds = SyntheticDataset(cfg)
    spec = ds.spec()

    assert spec.num_nodes == cfg.num_nodes
    assert spec.event_dim == SYNTHETIC_TASKS[task_name].event_dim
    assert spec.extra is not None
    assert spec.extra["synthetic_task"] == task_name
    assert "primary_metric" in spec.extra
    assert "supported_metrics" in spec.extra
    assert "task_axes" in spec.extra
    assert "task_tags" in spec.extra
    assert SYNTHETIC_TASKS[task_name].primary_metric_path == spec.extra["primary_metric"]["path"]
    assert spec.extra["task_axes"]["graph_type"] == SYNTHETIC_TASKS[task_name].graph_type
    assert spec.extra["task_axes"]["dynamics_type"] == SYNTHETIC_TASKS[task_name].dynamics_type
    assert spec.extra["task_axes"]["supervision_level"] == SYNTHETIC_TASKS[task_name].supervision_level
    assert spec.extra["task_axes"]["supervision_type"] == SYNTHETIC_TASKS[task_name].supervision_type
    assert spec.extra["task_tags"] == list(SYNTHETIC_TASKS[task_name].tags())

    train_bins = list(ds.bins("train"))
    train_edge_targets = list(ds.edge_targets("train") or [])
    train_node_targets = list(ds.node_targets("train") or [])

    assert train_bins
    metric_family = SYNTHETIC_TASKS[task_name].metric_family
    if metric_family in {"edge_regression", "edge_classification"}:
        assert train_edge_targets
        assert not train_node_targets
        assert len(train_bins) == len(train_edge_targets)
    elif metric_family in {"node_regression", "node_classification"}:
        assert train_node_targets
        assert not train_edge_targets
        assert len(train_bins) == len(train_node_targets)
    else:
        assert not train_edge_targets
        assert not train_node_targets
        assert len(train_bins) > 0

    first_batch = train_bins[0]
    expected_events = cfg.events_per_bin
    if task_name in {
        "temporal_memory",
        "node_temporal_state",
        "node_temporal_regression",
        "edge_temporal_state",
        "edge_ranking_temporal",
        "next_dst_temporal_ranking",
        "conservative_oscillator",
    }:
        expected_events = cfg.num_nodes
    if task_name == "ift_diffusion":
        expected_events = 3 * cfg.num_nodes
    if task_name == "associative_retrieval":
        expected_events = cfg.num_nodes * (max(3, cfg.events_per_bin // cfg.num_nodes) + 1)
    assert first_batch.num_events == expected_events

    if spec.event_dim == 0:
        assert first_batch.features is None
    else:
        assert first_batch.features is not None
        assert first_batch.features.shape[1] == spec.event_dim

    if train_edge_targets:
        first_target = train_edge_targets[0]
        assert first_target.targets.shape == (cfg.num_nodes,)
        assert first_target.events.num_events == cfg.num_nodes
        assert torch.equal(first_target.events.src, first_target.events.dst)
    if train_node_targets:
        first_target = train_node_targets[0]
        assert first_target.shape == (cfg.num_nodes,)
        assert first_target.dtype == torch.float32


def test_synthetic_task_families_cover_full_node_edge_matrix_and_ranking():
    families = {task.metric_family for task in SYNTHETIC_TASKS.values()}

    assert "edge_regression" in families
    assert "edge_classification" in families
    assert "node_regression" in families
    assert "node_classification" in families
    assert "edge_ranking" in families


def test_edge_rollout_tasks_advertise_persistent_rollout_r2():
    for task in SYNTHETIC_TASKS.values():
        if task.metric_family != "edge_regression":
            continue
        if task.primary_metric_path != "rollout_val.rollout_edge_r2":
            continue
        assert "persistent_edge_r2" in task.supported_metrics
        assert "rollout_persistent_edge_r2" in task.supported_metrics


def test_synthetic_task_metric_metadata_no_longer_uses_skill_wording():
    for task in SYNTHETIC_TASKS.values():
        assert all("skill" not in metric for metric in task.supported_metrics)
        assert all("skill" not in path for path in task.summary_metric_paths)


def test_every_synthetic_task_shortlist_includes_ift_update():
    for task in SYNTHETIC_TASKS.values():
        assert IFT_PAIR in task.recommended_pairs


def test_synthetic_task_axes_capture_graph_dynamics_and_supervision():
    ift_axes = synthetic_task_axes("ift_diffusion")
    assert ift_axes["graph_type"] == "ring"
    assert ift_axes["dynamics_type"] == "diffusion"
    assert ift_axes["supervision_level"] == "edge"
    assert ift_axes["supervision_type"] == "regression"
    assert ift_axes["temporal_mode"] == "rollout"

    tags = synthetic_task_tags("ift_diffusion")
    assert "graph:ring" in tags
    assert "dynamics:diffusion" in tags
    assert "target:edge_regression" in tags


def test_synthetic_task_query_helpers_group_related_benchmarks():
    additive_tasks = set(list_synthetic_tasks(dynamics_type="additive"))
    assert {"deepsets_sum", "node_sum_regression"} <= additive_tasks

    stateful_edge_tasks = set(
        list_synthetic_tasks(
            graph_type="ring",
            supervision_level="edge",
            supervision_type="ranking",
            temporal_mode="stateful",
        )
    )
    assert {"edge_ranking_temporal", "next_dst_temporal_ranking"} <= stateful_edge_tasks

    grouped = group_synthetic_tasks_by("graph_type")
    assert "ift_diffusion" in grouped["ring"]
    assert "temporal_memory" in grouped["self_loop"]
    assert "deepsets_sum" in grouped["random_pair"]
