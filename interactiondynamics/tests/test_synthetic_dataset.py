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
from interactiondynamics.core.events import EventBatch
from interactiondynamics.eval.evaluate import _remove_wave_drive_events, _self_generate_grid_wave_events


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
    if task_name == "associative_retrieval":
        expected_events = cfg.num_nodes * (max(3, cfg.events_per_bin // cfg.num_nodes) + 1)
    if task_name in {"diffusion", "wave", "coupled_oscillator"}:
        # Physical episodes begin at rest, so their first observed bin contains
        # only the externally observed raindrop. Endogenous force events begin
        # after that impulse has propagated.
        assert first_batch.num_events == 1
        assert first_batch.is_external is not None
        assert bool(first_batch.is_external.all())
    elif task_name in {"wave_grid", "wave_torus", "wave_doorway", "wave_swisscheese"}:
        assert first_batch.num_events > 0
    else:
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


def test_every_legacy_synthetic_task_has_a_recommended_baseline():
    for task in SYNTHETIC_TASKS.values():
        assert task.recommended_pairs
    assert IFT_PAIR not in SYNTHETIC_TASKS["wave"].recommended_pairs


def test_synthetic_task_axes_capture_graph_dynamics_and_supervision():
    ift_axes = synthetic_task_axes("diffusion")
    assert ift_axes["graph_type"] == "ring"
    assert ift_axes["dynamics_type"] == "first_order_force_field"
    assert ift_axes["supervision_level"] == "edge"
    assert ift_axes["supervision_type"] == "ranking"
    assert ift_axes["temporal_mode"] == "episodic"

    tags = synthetic_task_tags("diffusion")
    assert "graph:ring" in tags
    assert "dynamics:first_order_force_field" in tags
    assert "target:edge_ranking" in tags

    wave_axes = synthetic_task_axes("wave")
    assert wave_axes["graph_type"] == "ring"
    assert wave_axes["dynamics_type"] == "second_order_force_field"
    assert wave_axes["temporal_mode"] == "episodic"

    wave_tags = synthetic_task_tags("wave")
    assert "graph:ring" in wave_tags
    assert "dynamics:second_order_force_field" in wave_tags
    assert "events:directed_physical_force_events" in wave_tags


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
    assert "diffusion" in grouped["ring"]
    assert "wave" in grouped["ring"]
    assert "temporal_memory" in grouped["self_loop"]
    assert "deepsets_sum" in grouped["random_pair"]


def test_grid_wave_topologies_materialize_distinct_structured_event_sets():
    event_counts: dict[str, int] = {}
    for task_name in ("wave_grid", "wave_torus", "wave_doorway", "wave_swisscheese"):
        dataset = SyntheticDataset(SyntheticDatasetConfig(task=task_name, num_nodes=64, num_bins=16, seed=3))
        # The first bin is deliberately quiet except for its raindrop. Inspect
        # the next bin to compare visible endogenous interactions by topology.
        first_batch = list(dataset.bins("train"))[1]
        targets = next(iter(dataset.edge_targets("train") or []))
        assert first_batch.features is not None
        assert first_batch.features.shape[1] == 2
        assert torch.isfinite(targets.targets).all()
        event_counts[task_name] = first_batch.num_events

    assert event_counts["wave_torus"] > event_counts["wave_grid"]
    assert event_counts["wave_doorway"] < event_counts["wave_grid"]
    assert event_counts["wave_swisscheese"] < event_counts["wave_grid"]


def test_diffusion_topology_selector_materializes_grid_derived_domains():
    graph_types = {
        "grid": "grid",
        "torus": "torus_grid",
        "doorway": "grid_doorway",
        "swisscheese": "grid_swisscheese",
    }
    topology_edge_counts: dict[str, int] = {}
    for topology, graph_type in graph_types.items():
        dataset = SyntheticDataset(
            SyntheticDatasetConfig(
                task="diffusion",
                diffusion_topology=topology,
                num_nodes=64,
                num_bins=16,
                seed=3,
            )
        )
        # The first bin is deliberately quiet except for its raindrop. Inspect
        # the next bin to compare visible endogenous interactions by topology.
        first_batch = list(dataset.bins("train"))[1]
        assert dataset.spec().extra is not None
        assert dataset.spec().extra["task_axes"]["graph_type"] == graph_type
        assert first_batch.features is not None
        assert first_batch.features.shape[1] == 4
        assert first_batch.is_external is not None
        assert torch.isfinite(first_batch.features).all()
        # The observation budget may cap all topologies at the same number of
        # emitted events. Check the hidden simulator support instead.
        topology_edge_counts[topology] = int(dataset._field_topology_edges(topology)[1].size)

    assert topology_edge_counts["torus"] > topology_edge_counts["grid"]
    assert topology_edge_counts["doorway"] < topology_edge_counts["grid"]
    assert topology_edge_counts["swisscheese"] < topology_edge_counts["grid"]


def test_physical_event_threshold_filters_only_observed_endogenous_events():
    common = dict(
        task="wave",
        num_nodes=16,
        num_bins=12,
        num_episodes=3,
        events_per_bin=32,
        raindrop_interval=4,
        seed=11,
    )
    unthresholded = SyntheticDataset(SyntheticDatasetConfig(**common, event_threshold=0.0))
    thresholded = SyntheticDataset(SyntheticDatasetConfig(**common, event_threshold=1.0))

    assert sum(batch.num_events for batch in thresholded._bins_all) <= sum(
        batch.num_events for batch in unthresholded._bins_all
    )
    for batch in thresholded._bins_all:
        assert batch.is_external is not None
        internal = ~batch.is_external
        if bool(internal.any()):
            assert batch.features is not None
            assert bool((torch.linalg.vector_norm(batch.features[internal], dim=1) > 1.0).all())


def test_physical_dynamics_support_every_field_topology():
    for dynamic in ("diffusion", "wave", "coupled_oscillator"):
        for topology in ("ring", "grid", "torus", "doorway", "swisscheese"):
            dataset = SyntheticDataset(
                SyntheticDatasetConfig(
                    task=dynamic,
                    field_topology=topology,
                    num_nodes=64,
                    num_bins=16,
                    seed=5,
                )
            )
            first_batch = next(iter(dataset.bins("train")))
            assert first_batch.features is not None
            assert first_batch.features.shape[1] == 4
            assert first_batch.is_external is not None
            assert torch.isfinite(first_batch.features).all()
            assert dataset.spec().extra is not None
            axes = dataset.spec().extra["task_axes"]
            expected_dynamic = {
                "diffusion": "first_order_force_field",
                "wave": "second_order_force_field",
                "coupled_oscillator": "second_order_coupled_oscillator",
            }[dynamic]
            assert axes["dynamics_type"] == expected_dynamic
            assert axes["generator_params"]["topology"] == topology


def test_self_generated_grid_wave_events_retain_topology_and_remove_external_drive():
    template = EventBatch(
        src=torch.tensor([0, 1, 0, 1]),
        dst=torch.tensor([1, 0, 0, 1]),
        features=torch.tensor([[0.2, 0.0], [-0.3, 0.0], [0.7, 1.0], [-0.5, 1.0]]),
    )

    generated = _self_generate_grid_wave_events(template, predicted_field=torch.tensor([1.5, -2.0]))

    assert torch.equal(generated.src, template.src)
    assert torch.equal(generated.dst, template.dst)
    assert torch.equal(generated.features[:, 0], torch.tensor([1.5, -2.0, 0.0, 0.0]))
    assert torch.equal(generated.features[:, 1], torch.tensor([0.0, 0.0, 0.0, 0.0]))


def test_drive_free_wave_events_keep_neighbor_signals_and_remove_only_drives():
    template = EventBatch(
        src=torch.tensor([0, 1, 0, 1]),
        dst=torch.tensor([1, 0, 0, 1]),
        features=torch.tensor([[0.2, 0.0], [-0.3, 0.0], [0.7, 1.0], [-0.5, 1.0]]),
    )

    free = _remove_wave_drive_events(template)

    assert torch.equal(free.src, template.src)
    assert torch.equal(free.dst, template.dst)
    assert torch.equal(free.features[:, 0], torch.tensor([0.2, -0.3, 0.0, 0.0]))
    assert torch.equal(free.features[:, 1], torch.tensor([0.0, 0.0, 0.0, 0.0]))


def test_physical_raindrops_repeat_within_each_episode():
    dataset = SyntheticDataset(
        SyntheticDatasetConfig(
            task="wave",
            num_nodes=16,
            num_bins=12,
            num_episodes=3,
            events_per_bin=32,
            raindrop_interval=4,
            seed=11,
        )
    )

    drops = []
    for batch in dataset._bins_all:
        assert batch.is_external is not None
        external = batch.is_external
        if bool(external.any()):
            assert int(external.sum()) == 1
            drops.append((int(batch.episode[external][0]), int(batch.t[external][0]) % 12))

    assert drops == [
        (episode, local_t)
        for episode in range(3)
        for local_t in (0, 4, 8)
    ]
