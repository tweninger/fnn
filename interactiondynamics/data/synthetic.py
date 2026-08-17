from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence, cast

import numpy as np
import torch

from interactiondynamics.core.events import EventBatch
from interactiondynamics.data.interfaces import DataSpec, EdgeTargetBatch, EventStreamDataset


@dataclass(frozen=True)
class SyntheticTaskSpec:
    name: str
    description: str
    focus: str
    event_dim: int
    recommended_pairs: tuple[tuple[str, str], ...]
    metric_family: str
    generator_family: str
    graph_type: str
    dynamics_type: str
    event_structure: str
    temporal_mode: str
    supported_metrics: tuple[str, ...]
    primary_metric_path: str
    primary_metric_goal: str
    summary_metric_paths: tuple[str, ...] = ()
    requires_node_scorer: bool = False
    feature_schema: tuple[str, ...] = ()
    generator_params: Optional[Dict[str, Any]] = None

    @property
    def supervision_level(self) -> str:
        return self.metric_family.split("_", maxsplit=1)[0]

    @property
    def supervision_type(self) -> str:
        return self.metric_family.split("_", maxsplit=1)[1]

    def axes(self) -> dict[str, Any]:
        return {
            "generator_family": self.generator_family,
            "graph_type": self.graph_type,
            "dynamics_type": self.dynamics_type,
            "event_structure": self.event_structure,
            "temporal_mode": self.temporal_mode,
            "supervision_level": self.supervision_level,
            "supervision_type": self.supervision_type,
            "feature_schema": list(self.feature_schema),
            "generator_params": dict(self.generator_params) if self.generator_params is not None else None,
        }

    def tags(self) -> tuple[str, ...]:
        tags = (
            f"family:{self.generator_family}",
            f"graph:{self.graph_type}",
            f"dynamics:{self.dynamics_type}",
            f"events:{self.event_structure}",
            f"temporal:{self.temporal_mode}",
            f"target:{self.supervision_level}_{self.supervision_type}",
        )
        if self.feature_schema:
            tags += (f"features:{'+'.join(self.feature_schema)}",)
        else:
            tags += ("features:none",)
        return tags


IFT_PAIR = ("ift", "ift_update")
FIELD_TOPOLOGY_CHOICES = ("ring", "grid", "torus", "doorway", "swisscheese")
# Backward-compatible import name for callers that used the original
# diffusion-only selector before field dynamics and topology were separated.
DIFFUSION_TOPOLOGY_CHOICES = FIELD_TOPOLOGY_CHOICES


def _second_order_field_task(
    name: str,
    *,
    description: str,
    focus: str,
    dynamics_type: str,
    generator_family: str,
) -> SyntheticTaskSpec:
    return SyntheticTaskSpec(
        name=name,
        description=description,
        focus=focus,
        event_dim=2,
        recommended_pairs=(IFT_PAIR, ("sum", "hnn"), ("sum", "lnn"), ("sum", "tgn_gru")),
        metric_family="edge_regression",
        generator_family=generator_family,
        graph_type="ring",
        dynamics_type=dynamics_type,
        event_structure="topology_neighbor_and_sparse_drive_events",
        temporal_mode="rollout",
        feature_schema=("signal", "is_drive"),
        supported_metrics=("edge_mse", "edge_r2", "persistent_edge_r2", "rollout_edge_r2", "rollout_edge_nrmse", "rollout_persistent_edge_r2"),
        primary_metric_path="rollout_val.rollout_edge_r2",
        primary_metric_goal="max",
        summary_metric_paths=("val.edge_r2", "rollout_val.rollout_edge_r2", "rollout_val.rollout_persistent_edge_r2", "rollout_test.rollout_edge_r2", "rollout_test.rollout_persistent_edge_r2"),
    )


def _physical_force_task(
    name: str,
    *,
    description: str,
    focus: str,
    dynamics_type: str,
    generator_family: str,
) -> SyntheticTaskSpec:
    """Metadata for an episodic, event-only physical field benchmark."""
    return SyntheticTaskSpec(
        name=name,
        description=description,
        focus=focus,
        event_dim=4,
        recommended_pairs=(("sum", "tgn_gru"),),
        metric_family="edge_ranking",
        generator_family=generator_family,
        graph_type="ring",
        dynamics_type=dynamics_type,
        event_structure="directed_physical_force_events",
        temporal_mode="episodic",
        feature_schema=("force_x", "force_y", "force_z", "force_w"),
        supported_metrics=(
            "mrr", "hits@1", "filtered_mrr", "filtered_hits@1", "filtered_hits@10",
            "force_mse", "topology_auc", "topology_f1",
        ),
        primary_metric_path="val.force_mse",
        primary_metric_goal="min",
        summary_metric_paths=(
            "val.force_mse", "test.force_mse", "test.active_force_mse",
            "test.filtered_mrr", "test.filtered_hits@1",
            "rollout_test.rollout_force_mse", "rollout_test.rollout_persistent_force_mse",
        ),
    )


def _grid_wave_task(
    name: str,
    *,
    description: str,
    focus: str,
    graph_type: str,
    topology: str,
) -> SyntheticTaskSpec:
    """Build metadata for a second-order wave benchmark on a square lattice."""
    return SyntheticTaskSpec(
        name=name,
        description=description,
        focus=focus,
        event_dim=2,
        recommended_pairs=(
            IFT_PAIR,
            ("sum", "hnn"),
            ("sum", "lnn"),
            ("sum", "tgn_gru"),
        ),
        metric_family="edge_regression",
        generator_family="grid_wave",
        graph_type=graph_type,
        dynamics_type="wave",
        event_structure="topology_neighbor_and_sparse_drive_events",
        temporal_mode="rollout",
        feature_schema=("signal", "is_drive"),
        generator_params={
            "a": 1.88,
            "b": -0.93,
            "lap": 0.075,
            "drive": 0.17,
            "topology": topology,
        },
        supported_metrics=(
            "edge_mse",
            "edge_r2",
            "persistent_edge_r2",
            "rollout_edge_r2",
            "rollout_edge_nrmse",
            "rollout_persistent_edge_r2",
        ),
        primary_metric_path="rollout_val.rollout_edge_r2",
        primary_metric_goal="max",
        summary_metric_paths=(
            "val.edge_r2",
            "rollout_val.rollout_edge_r2",
            "rollout_val.rollout_persistent_edge_r2",
            "rollout_test.rollout_edge_r2",
            "rollout_test.rollout_persistent_edge_r2",
        ),
    )


SYNTHETIC_TASKS: Dict[str, SyntheticTaskSpec] = {
    "deepsets_sum": SyntheticTaskSpec(
        name="deepsets_sum",
        description="Predict the next-step per-node sum of incident event values.",
        focus="Permutation-invariant additive set aggregation.",
        event_dim=1,
        recommended_pairs=(
            ("sum", "tgn_gru"),
            ("deepsets", "tgn_gru"),
            ("settransformer", "tgn_gru"),
            IFT_PAIR,
        ),
        metric_family="edge_regression",
        generator_family="incident_sum",
        graph_type="random_pair",
        dynamics_type="additive",
        event_structure="incident_events",
        temporal_mode="stateless",
        feature_schema=("value",),
        supported_metrics=("edge_mse", "edge_r2", "edge_nrmse", "edge_corr"),
        primary_metric_path="val.edge_r2",
        primary_metric_goal="max",
        summary_metric_paths=(
            "val.edge_r2",
            "val.edge_nrmse",
            "test.edge_r2",
            "test.edge_nrmse",
        ),
    ),
    "settransformer_max": SyntheticTaskSpec(
        name="settransformer_max",
        description="Predict the value attached to the highest-key incident event.",
        focus="Content-based selection over unordered event sets.",
        event_dim=2,
        recommended_pairs=(
            ("deepsets", "tgn_gru"),
            ("settransformer", "tgn_gru"),
            ("hopfield", "tgn_gru"),
            IFT_PAIR,
        ),
        metric_family="edge_regression",
        generator_family="incident_argmax",
        graph_type="random_pair",
        dynamics_type="keyed_selection",
        event_structure="incident_events",
        temporal_mode="stateless",
        feature_schema=("value", "key"),
        supported_metrics=("edge_mse", "edge_r2", "edge_corr"),
        primary_metric_path="val.edge_r2",
        primary_metric_goal="max",
        summary_metric_paths=(
            "val.edge_r2",
            "val.edge_corr",
            "test.edge_r2",
            "test.edge_corr",
        ),
    ),
    "temporal_memory": SyntheticTaskSpec(
        name="temporal_memory",
        description="Predict a damped latent trajectory driven by per-node self events.",
        focus="Temporal state propagation and memory in the update law.",
        event_dim=1,
        recommended_pairs=(
            ("sum", "tgn_gru"),
            ("sum", "lnn"),
            ("sum", "hnn"),
            IFT_PAIR,
        ),
        metric_family="edge_regression",
        generator_family="driven_second_order_state",
        graph_type="self_loop",
        dynamics_type="temporal_memory",
        event_structure="self_events",
        temporal_mode="rollout",
        feature_schema=("drive",),
        supported_metrics=(
            "edge_mse",
            "edge_r2",
            "persistent_edge_r2",
            "rollout_edge_r2",
            "rollout_edge_nrmse",
            "rollout_persistent_edge_r2",
        ),
        primary_metric_path="rollout_val.rollout_edge_r2",
        primary_metric_goal="max",
        summary_metric_paths=(
            "val.edge_r2",
            "val.persistent_edge_r2",
            "rollout_val.rollout_edge_r2",
            "rollout_test.rollout_edge_r2",
        ),
    ),
    "node_count_threshold": SyntheticTaskSpec(
        name="node_count_threshold",
        description="Predict whether the next-step incident event count crosses a node-level threshold.",
        focus="Set aggregation for node-level binary classification.",
        event_dim=0,
        recommended_pairs=(
            ("sum", "tgn_gru"),
            ("deepsets", "tgn_gru"),
            ("settransformer", "tgn_gru"),
            IFT_PAIR,
        ),
        metric_family="node_classification",
        generator_family="incident_count_threshold",
        graph_type="random_pair",
        dynamics_type="threshold",
        event_structure="incident_events",
        temporal_mode="stateless",
        supported_metrics=("node_bce", "node_acc", "node_f1", "node_auroc", "node_auprc"),
        primary_metric_path="val.node_auroc",
        primary_metric_goal="max",
        summary_metric_paths=(
            "val.node_auroc",
            "val.node_f1",
            "test.node_auroc",
            "test.node_f1",
        ),
        requires_node_scorer=True,
    ),
    "node_keyed_trigger": SyntheticTaskSpec(
        name="node_keyed_trigger",
        description="Predict whether the highest-key incident event carries a positive trigger value.",
        focus="Content-based event selection for node-level binary classification.",
        event_dim=2,
        recommended_pairs=(
            ("deepsets", "tgn_gru"),
            ("settransformer", "tgn_gru"),
            ("hopfield", "tgn_gru"),
            IFT_PAIR,
        ),
        metric_family="node_classification",
        generator_family="incident_keyed_trigger",
        graph_type="random_pair",
        dynamics_type="keyed_trigger",
        event_structure="incident_events",
        temporal_mode="stateless",
        feature_schema=("value", "key"),
        supported_metrics=("node_bce", "node_acc", "node_f1", "node_auroc", "node_auprc"),
        primary_metric_path="val.node_auroc",
        primary_metric_goal="max",
        summary_metric_paths=(
            "val.node_auroc",
            "val.node_auprc",
            "test.node_auroc",
            "test.node_f1",
        ),
        requires_node_scorer=True,
    ),
    "node_temporal_state": SyntheticTaskSpec(
        name="node_temporal_state",
        description="Predict whether the next-step latent node state is positive under driven dynamics.",
        focus="Temporal memory for node-level binary classification.",
        event_dim=1,
        recommended_pairs=(
            ("sum", "tgn_gru"),
            ("sum", "lnn"),
            ("sum", "hnn"),
            IFT_PAIR,
        ),
        metric_family="node_classification",
        generator_family="driven_second_order_state_sign",
        graph_type="self_loop",
        dynamics_type="temporal_memory",
        event_structure="self_events",
        temporal_mode="stateful",
        feature_schema=("drive",),
        supported_metrics=("node_bce", "node_acc", "node_f1", "node_auroc", "node_auprc"),
        primary_metric_path="val.node_auroc",
        primary_metric_goal="max",
        summary_metric_paths=(
            "val.node_auroc",
            "val.node_f1",
            "test.node_auroc",
            "test.node_f1",
        ),
        requires_node_scorer=True,
    ),
    "node_sum_regression": SyntheticTaskSpec(
        name="node_sum_regression",
        description="Predict the next-step per-node sum of incident event values with the node scorer.",
        focus="Permutation-invariant additive set aggregation for node regression.",
        event_dim=1,
        recommended_pairs=(
            ("sum", "tgn_gru"),
            ("deepsets", "tgn_gru"),
            ("settransformer", "tgn_gru"),
            IFT_PAIR,
        ),
        metric_family="node_regression",
        generator_family="incident_sum",
        graph_type="random_pair",
        dynamics_type="additive",
        event_structure="incident_events",
        temporal_mode="stateless",
        feature_schema=("value",),
        supported_metrics=("node_mse", "node_r2", "node_nrmse", "node_corr"),
        primary_metric_path="val.node_r2",
        primary_metric_goal="max",
        summary_metric_paths=(
            "val.node_r2",
            "val.node_nrmse",
            "test.node_r2",
            "test.node_nrmse",
        ),
        requires_node_scorer=True,
    ),
    "node_keyed_value": SyntheticTaskSpec(
        name="node_keyed_value",
        description="Predict the value attached to the highest-key incident event with the node scorer.",
        focus="Content-based selection over unordered event sets for node regression.",
        event_dim=2,
        recommended_pairs=(
            ("deepsets", "tgn_gru"),
            ("settransformer", "tgn_gru"),
            ("hopfield", "tgn_gru"),
            IFT_PAIR,
        ),
        metric_family="node_regression",
        generator_family="incident_argmax",
        graph_type="random_pair",
        dynamics_type="keyed_selection",
        event_structure="incident_events",
        temporal_mode="stateless",
        feature_schema=("value", "key"),
        supported_metrics=("node_mse", "node_r2", "node_corr"),
        primary_metric_path="val.node_r2",
        primary_metric_goal="max",
        summary_metric_paths=(
            "val.node_r2",
            "val.node_corr",
            "test.node_r2",
            "test.node_corr",
        ),
        requires_node_scorer=True,
    ),
    "node_temporal_regression": SyntheticTaskSpec(
        name="node_temporal_regression",
        description="Predict the next-step latent node state under driven dynamics with the node scorer.",
        focus="Temporal memory for node-level regression.",
        event_dim=1,
        recommended_pairs=(
            ("sum", "tgn_gru"),
            ("sum", "lnn"),
            ("sum", "hnn"),
            IFT_PAIR,
        ),
        metric_family="node_regression",
        generator_family="driven_second_order_state",
        graph_type="self_loop",
        dynamics_type="temporal_memory",
        event_structure="self_events",
        temporal_mode="rollout",
        feature_schema=("drive",),
        supported_metrics=(
            "node_mse",
            "node_r2",
            "persistent_node_r2",
            "rollout_node_r2",
            "rollout_node_nrmse",
        ),
        primary_metric_path="rollout_val.rollout_node_r2",
        primary_metric_goal="max",
        summary_metric_paths=(
            "val.node_r2",
            "val.persistent_node_r2",
            "rollout_val.rollout_node_r2",
            "rollout_test.rollout_node_r2",
        ),
        requires_node_scorer=True,
    ),
    "edge_count_threshold": SyntheticTaskSpec(
        name="edge_count_threshold",
        description="Predict whether the next-step incident event count crosses a node-level threshold with the edge head.",
        focus="Set aggregation for binary edge-style supervision.",
        event_dim=0,
        recommended_pairs=(
            ("sum", "tgn_gru"),
            ("deepsets", "tgn_gru"),
            ("settransformer", "tgn_gru"),
            IFT_PAIR,
        ),
        metric_family="edge_classification",
        generator_family="incident_count_threshold",
        graph_type="random_pair",
        dynamics_type="threshold",
        event_structure="incident_events",
        temporal_mode="stateless",
        supported_metrics=("edge_bce", "edge_acc", "edge_f1", "edge_auroc", "edge_auprc"),
        primary_metric_path="val.edge_auroc",
        primary_metric_goal="max",
        summary_metric_paths=(
            "val.edge_auroc",
            "val.edge_f1",
            "test.edge_auroc",
            "test.edge_f1",
        ),
    ),
    "edge_threshold_classification": SyntheticTaskSpec(
        name="edge_threshold_classification",
        description="Predict whether the next-step incident event count crosses a node-level threshold with the edge head.",
        focus="Set aggregation for binary edge-style supervision.",
        event_dim=0,
        recommended_pairs=(
            ("sum", "tgn_gru"),
            ("deepsets", "tgn_gru"),
            ("settransformer", "tgn_gru"),
            IFT_PAIR,
        ),
        metric_family="edge_classification",
        generator_family="incident_count_threshold",
        graph_type="random_pair",
        dynamics_type="threshold",
        event_structure="incident_events",
        temporal_mode="stateless",
        supported_metrics=("edge_bce", "edge_acc", "edge_f1", "edge_auroc", "edge_auprc"),
        primary_metric_path="val.edge_auroc",
        primary_metric_goal="max",
        summary_metric_paths=(
            "val.edge_auroc",
            "val.edge_f1",
            "test.edge_auroc",
            "test.edge_f1",
        ),
    ),
    "edge_keyed_trigger": SyntheticTaskSpec(
        name="edge_keyed_trigger",
        description="Predict whether the highest-key incident event carries a positive trigger value with the edge head.",
        focus="Content-based event selection for binary edge-style supervision.",
        event_dim=2,
        recommended_pairs=(
            ("deepsets", "tgn_gru"),
            ("settransformer", "tgn_gru"),
            ("hopfield", "tgn_gru"),
            IFT_PAIR,
        ),
        metric_family="edge_classification",
        generator_family="incident_keyed_trigger",
        graph_type="random_pair",
        dynamics_type="keyed_trigger",
        event_structure="incident_events",
        temporal_mode="stateless",
        feature_schema=("value", "key"),
        supported_metrics=("edge_bce", "edge_acc", "edge_f1", "edge_auroc", "edge_auprc"),
        primary_metric_path="val.edge_auroc",
        primary_metric_goal="max",
        summary_metric_paths=(
            "val.edge_auroc",
            "val.edge_auprc",
            "test.edge_auroc",
            "test.edge_f1",
        ),
    ),
    "edge_keyed_trigger_classification": SyntheticTaskSpec(
        name="edge_keyed_trigger_classification",
        description="Predict whether the highest-key incident event carries a positive trigger value with the edge head.",
        focus="Content-based event selection for binary edge-style supervision.",
        event_dim=2,
        recommended_pairs=(
            ("deepsets", "tgn_gru"),
            ("settransformer", "tgn_gru"),
            ("hopfield", "tgn_gru"),
            IFT_PAIR,
        ),
        metric_family="edge_classification",
        generator_family="incident_keyed_trigger",
        graph_type="random_pair",
        dynamics_type="keyed_trigger",
        event_structure="incident_events",
        temporal_mode="stateless",
        feature_schema=("value", "key"),
        supported_metrics=("edge_bce", "edge_acc", "edge_f1", "edge_auroc", "edge_auprc"),
        primary_metric_path="val.edge_auroc",
        primary_metric_goal="max",
        summary_metric_paths=(
            "val.edge_auroc",
            "val.edge_auprc",
            "test.edge_auroc",
            "test.edge_f1",
        ),
    ),
    "edge_temporal_state": SyntheticTaskSpec(
        name="edge_temporal_state",
        description="Predict whether the next-step latent state is positive under driven dynamics with the edge head.",
        focus="Temporal memory for binary edge-style supervision.",
        event_dim=1,
        recommended_pairs=(
            ("sum", "tgn_gru"),
            ("sum", "lnn"),
            ("sum", "hnn"),
            IFT_PAIR,
        ),
        metric_family="edge_classification",
        generator_family="driven_second_order_state_sign",
        graph_type="self_loop",
        dynamics_type="temporal_memory",
        event_structure="self_events",
        temporal_mode="stateful",
        feature_schema=("drive",),
        supported_metrics=("edge_bce", "edge_acc", "edge_f1", "edge_auroc", "edge_auprc"),
        primary_metric_path="val.edge_auroc",
        primary_metric_goal="max",
        summary_metric_paths=(
            "val.edge_auroc",
            "val.edge_f1",
            "test.edge_auroc",
            "test.edge_f1",
        ),
    ),
    "associative_retrieval": SyntheticTaskSpec(
        name="associative_retrieval",
        description="Retrieve the value whose key best matches a query event in the same node-local set.",
        focus="Associative content retrieval over unordered event sets.",
        event_dim=6,
        recommended_pairs=(
            ("hopfield", "tgn_gru"),
            ("settransformer", "tgn_gru"),
            ("deepsets", "tgn_gru"),
            IFT_PAIR,
        ),
        metric_family="edge_regression",
        generator_family="query_key_value_set",
        graph_type="self_loop",
        dynamics_type="associative_retrieval",
        event_structure="query_candidate_self_events",
        temporal_mode="stateless",
        feature_schema=("value", "key_x", "key_y", "query_x", "query_y", "is_query"),
        supported_metrics=("edge_mse", "edge_r2", "edge_corr"),
        primary_metric_path="val.edge_r2",
        primary_metric_goal="max",
        summary_metric_paths=(
            "val.edge_r2",
            "val.edge_corr",
            "test.edge_r2",
            "test.edge_corr",
        ),
    ),
    "conservative_oscillator": SyntheticTaskSpec(
        name="conservative_oscillator",
        description="Predict a lightly driven second-order oscillator with long rollout memory.",
        focus="Conservative temporal dynamics for update-law comparisons.",
        event_dim=1,
        recommended_pairs=(
            ("sum", "hnn"),
            ("sum", "lnn"),
            ("sum", "tgn_gru"),
            IFT_PAIR,
        ),
        metric_family="edge_regression",
        generator_family="driven_conservative_oscillator",
        graph_type="self_loop",
        dynamics_type="oscillator",
        event_structure="self_events",
        temporal_mode="rollout",
        feature_schema=("drive",),
        generator_params={"a": 1.92, "b": -0.96, "c": 0.08},
        supported_metrics=(
            "edge_mse",
            "edge_r2",
            "persistent_edge_r2",
            "rollout_edge_r2",
            "rollout_edge_nrmse",
            "rollout_persistent_edge_r2",
        ),
        primary_metric_path="rollout_val.rollout_edge_r2",
        primary_metric_goal="max",
        summary_metric_paths=(
            "val.edge_r2",
            "val.persistent_edge_r2",
            "rollout_val.rollout_edge_r2",
            "rollout_test.rollout_edge_r2",
        ),
    ),
    "diffusion": _physical_force_task(
        "diffusion",
        description="Event-only first-order diffusion from one observed raindrop impulse over hidden topology.",
        focus="Recover a persistent interaction operator and first-order dissipative dynamics from force events alone.",
        dynamics_type="first_order_force_field",
        generator_family="hidden_force_diffusion",
    ),
    "coupled_oscillator": _physical_force_task(
        "coupled_oscillator",
        description="Event-only coupled oscillator response from one observed raindrop impulse over hidden topology.",
        focus="Recover persistent topology and second-order local restoring dynamics from force events alone.",
        dynamics_type="second_order_coupled_oscillator",
        generator_family="hidden_force_coupled_oscillator",
    ),
    "wave": _physical_force_task(
        "wave",
        description="Event-only damped wave response from one observed raindrop impulse over hidden topology.",
        focus="Recover a persistent interaction operator and damped wave dynamics from force events alone.",
        dynamics_type="second_order_force_field",
        generator_family="hidden_force_wave",
    ),
    "wave_grid": _grid_wave_task(
        "wave_grid",
        description="Predict a driven second-order wave on a rectangular grid with reflecting outer boundaries.",
        focus="Second-order propagation over a bounded two-dimensional lattice.",
        graph_type="grid",
        topology="bounded_grid",
    ),
    "wave_torus": _grid_wave_task(
        "wave_torus",
        description="Predict a driven second-order wave on a periodic two-dimensional torus grid.",
        focus="Topology-aware propagation across periodic seams.",
        graph_type="torus_grid",
        topology="periodic_grid",
    ),
    "wave_doorway": _grid_wave_task(
        "wave_doorway",
        description="Predict a driven wave on a grid divided by a wall with a narrow doorway.",
        focus="Propagation constrained by a topology-changing barrier and aperture.",
        graph_type="grid_doorway",
        topology="doorway_barrier",
    ),
    "wave_swisscheese": _grid_wave_task(
        "wave_swisscheese",
        description="Predict a driven wave on a grid containing multiple circular node holes.",
        focus="Propagation around disconnected obstacles in a perforated lattice.",
        graph_type="grid_swisscheese",
        topology="swisscheese",
    ),
    "edge_ranking_sum_shift": SyntheticTaskSpec(
        name="edge_ranking_sum_shift",
        description="Predict the next-step destination shift induced by the signed sum of per-node event values.",
        focus="Permutation-invariant additive set aggregation for destination ranking.",
        event_dim=1,
        recommended_pairs=(
            ("sum", "tgn_gru"),
            ("deepsets", "tgn_gru"),
            ("settransformer", "tgn_gru"),
            IFT_PAIR,
        ),
        metric_family="edge_ranking",
        generator_family="ring_sum_shift",
        graph_type="ring",
        dynamics_type="destination_shift",
        event_structure="source_destination_ring_events",
        temporal_mode="stateless",
        feature_schema=("value",),
        supported_metrics=("mrr", "hits@1", "hits@10", "pairwise_auc_tie_half"),
        primary_metric_path="val.mrr",
        primary_metric_goal="max",
        summary_metric_paths=(
            "val.mrr",
            "val.hits@1",
            "test.mrr",
            "test.hits@10",
        ),
    ),
    "next_dst_ranking": SyntheticTaskSpec(
        name="next_dst_ranking",
        description="Predict the next-step destination shift induced by the signed sum of per-node event values.",
        focus="Permutation-invariant additive set aggregation for destination ranking.",
        event_dim=1,
        recommended_pairs=(
            ("sum", "tgn_gru"),
            ("deepsets", "tgn_gru"),
            ("settransformer", "tgn_gru"),
            IFT_PAIR,
        ),
        metric_family="edge_ranking",
        generator_family="ring_sum_shift",
        graph_type="ring",
        dynamics_type="destination_shift",
        event_structure="source_destination_ring_events",
        temporal_mode="stateless",
        feature_schema=("value",),
        supported_metrics=("mrr", "hits@1", "hits@10", "pairwise_auc_tie_half"),
        primary_metric_path="val.mrr",
        primary_metric_goal="max",
        summary_metric_paths=(
            "val.mrr",
            "val.hits@1",
            "test.mrr",
            "test.hits@10",
        ),
    ),
    "edge_ranking_keyed_shift": SyntheticTaskSpec(
        name="edge_ranking_keyed_shift",
        description="Predict the next-step destination chosen by the highest-key marked event.",
        focus="Content-based event selection for destination ranking.",
        event_dim=2,
        recommended_pairs=(
            ("deepsets", "tgn_gru"),
            ("settransformer", "tgn_gru"),
            ("hopfield", "tgn_gru"),
            IFT_PAIR,
        ),
        metric_family="edge_ranking",
        generator_family="ring_keyed_shift",
        graph_type="ring",
        dynamics_type="destination_shift",
        event_structure="source_destination_ring_events",
        temporal_mode="stateless",
        feature_schema=("shift", "key"),
        supported_metrics=("mrr", "hits@1", "hits@10", "pairwise_auc_tie_half"),
        primary_metric_path="val.mrr",
        primary_metric_goal="max",
        summary_metric_paths=(
            "val.mrr",
            "val.hits@1",
            "test.mrr",
            "test.hits@10",
        ),
    ),
    "edge_retrieval": SyntheticTaskSpec(
        name="edge_retrieval",
        description="Retrieve the correct next destination from a keyed event set.",
        focus="Content-based associative retrieval for destination ranking.",
        event_dim=2,
        recommended_pairs=(
            ("deepsets", "tgn_gru"),
            ("settransformer", "tgn_gru"),
            ("hopfield", "tgn_gru"),
            IFT_PAIR,
        ),
        metric_family="edge_ranking",
        generator_family="ring_keyed_shift",
        graph_type="ring",
        dynamics_type="destination_retrieval",
        event_structure="source_destination_ring_events",
        temporal_mode="stateless",
        feature_schema=("shift", "key"),
        supported_metrics=("mrr", "hits@1", "hits@10", "pairwise_auc_tie_half"),
        primary_metric_path="val.mrr",
        primary_metric_goal="max",
        summary_metric_paths=(
            "val.mrr",
            "val.hits@1",
            "test.mrr",
            "test.hits@10",
        ),
    ),
    "edge_ranking_temporal": SyntheticTaskSpec(
        name="edge_ranking_temporal",
        description="Predict the next-step destination bucket induced by a driven latent node state.",
        focus="Temporal memory for destination ranking.",
        event_dim=1,
        recommended_pairs=(
            ("sum", "tgn_gru"),
            ("sum", "lnn"),
            ("sum", "hnn"),
            IFT_PAIR,
        ),
        metric_family="edge_ranking",
        generator_family="temporal_destination_routing",
        graph_type="ring",
        dynamics_type="temporal_routing",
        event_structure="source_destination_ring_events",
        temporal_mode="stateful",
        feature_schema=("drive",),
        supported_metrics=("mrr", "hits@1", "hits@10", "pairwise_auc_tie_half"),
        primary_metric_path="val.mrr",
        primary_metric_goal="max",
        summary_metric_paths=(
            "val.mrr",
            "val.hits@1",
            "test.mrr",
            "test.hits@10",
        ),
    ),
    "next_dst_temporal_ranking": SyntheticTaskSpec(
        name="next_dst_temporal_ranking",
        description="Predict the next-step destination bucket induced by a driven latent node state.",
        focus="Temporal memory for destination ranking.",
        event_dim=1,
        recommended_pairs=(
            ("sum", "tgn_gru"),
            ("sum", "lnn"),
            ("sum", "hnn"),
            IFT_PAIR,
        ),
        metric_family="edge_ranking",
        generator_family="temporal_destination_routing",
        graph_type="ring",
        dynamics_type="temporal_routing",
        event_structure="source_destination_ring_events",
        temporal_mode="stateful",
        feature_schema=("drive",),
        supported_metrics=("mrr", "hits@1", "hits@10", "pairwise_auc_tie_half"),
        primary_metric_path="val.mrr",
        primary_metric_goal="max",
        summary_metric_paths=(
            "val.mrr",
            "val.hits@1",
            "test.mrr",
            "test.hits@10",
        ),
    ),
}


SYNTHETIC_TASK_AXES = (
    "generator_family",
    "graph_type",
    "dynamics_type",
    "event_structure",
    "temporal_mode",
    "supervision_level",
    "supervision_type",
)


def synthetic_task_axes(task_name: str) -> dict[str, Any]:
    return SYNTHETIC_TASKS[task_name].axes()


def synthetic_task_tags(task_name: str) -> tuple[str, ...]:
    return SYNTHETIC_TASKS[task_name].tags()


def list_synthetic_tasks(
    *,
    generator_family: Optional[str] = None,
    graph_type: Optional[str] = None,
    dynamics_type: Optional[str] = None,
    event_structure: Optional[str] = None,
    temporal_mode: Optional[str] = None,
    supervision_level: Optional[str] = None,
    supervision_type: Optional[str] = None,
) -> tuple[str, ...]:
    filters = {
        "generator_family": generator_family,
        "graph_type": graph_type,
        "dynamics_type": dynamics_type,
        "event_structure": event_structure,
        "temporal_mode": temporal_mode,
        "supervision_level": supervision_level,
        "supervision_type": supervision_type,
    }
    task_names: list[str] = []
    for task_name, task in SYNTHETIC_TASKS.items():
        axes = task.axes()
        if any(value is not None and axes[key] != value for key, value in filters.items()):
            continue
        task_names.append(task_name)
    return tuple(task_names)


def group_synthetic_tasks_by(axis_name: str) -> dict[str, tuple[str, ...]]:
    if axis_name not in SYNTHETIC_TASK_AXES:
        available = ", ".join(SYNTHETIC_TASK_AXES)
        raise ValueError(f"Unsupported synthetic task axis {axis_name!r}. Available: {available}")
    grouped: dict[str, list[str]] = {}
    for task_name, task in SYNTHETIC_TASKS.items():
        axis_value = str(task.axes()[axis_name])
        grouped.setdefault(axis_value, []).append(task_name)
    return {key: tuple(values) for key, values in sorted(grouped.items())}


@dataclass
class SyntheticDatasetConfig:
    task: str = "deepsets_sum"
    name: Optional[str] = None
    num_nodes: int = 64
    num_bins: int = 120
    events_per_bin: int = 256
    num_episodes: int = 10
    raindrop_interval: Optional[int] = None
    event_threshold: float = 0.0
    diffusion_topology: Optional[str] = None
    field_topology: Optional[str] = None
    split_fracs: tuple[float, float, float] = (0.6, 0.2, 0.2)
    seed: int = 0
    device: Optional[torch.device] = None


class SyntheticDataset(EventStreamDataset):
    def __init__(self, cfg: SyntheticDatasetConfig):
        if cfg.task not in SYNTHETIC_TASKS:
            raise ValueError(
                f"Unknown synthetic task: {cfg.task}. "
                f"Available: {', '.join(sorted(SYNTHETIC_TASKS))}"
            )
        if cfg.diffusion_topology is not None and cfg.field_topology is not None:
            raise ValueError("Specify only field_topology; diffusion_topology is a legacy diffusion-only alias.")
        selected_topology = cfg.field_topology or cfg.diffusion_topology
        if cfg.raindrop_interval is not None:
            if cfg.task not in {"diffusion", "wave", "coupled_oscillator"}:
                raise ValueError("raindrop_interval is supported only for physical field-dynamics synthetic tasks.")
            if int(cfg.raindrop_interval) < 1:
                raise ValueError("raindrop_interval must be at least one local episode step.")
        if cfg.event_threshold < 0.0 or cfg.event_threshold > 1.0:
            raise ValueError("event_threshold must lie in [0, 1], in nominal raindrop-force units.")
        if selected_topology is not None:
            if cfg.task not in {"diffusion", "wave", "coupled_oscillator"}:
                raise ValueError("Topology selection is supported only for field-dynamics synthetic tasks.")
            if selected_topology not in FIELD_TOPOLOGY_CHOICES:
                allowed = ", ".join(FIELD_TOPOLOGY_CHOICES)
                raise ValueError(f"Unknown field topology {selected_topology!r}. Allowed: {allowed}.")
        self.cfg = cfg
        self._task = SYNTHETIC_TASKS[cfg.task]
        self._rng = np.random.default_rng(int(cfg.seed))
        self._hidden_truth: Optional[dict[str, Any]] = None
        self._bins_all, self._node_targets_all, self._edge_targets_all = self._materialize()
        self._split_bins = (
            self._compute_episode_splits()
            if cfg.task in {"diffusion", "wave", "coupled_oscillator"}
            else self._compute_splits(len(self._bins_all))
        )

    def _materialize(
        self,
    ) -> tuple[list[EventBatch], Optional[list[torch.Tensor]], Optional[list[EdgeTargetBatch]]]:
        if self.cfg.task == "deepsets_sum":
            bins, edge_targets = self._materialize_shifted_edge_targets(self._build_deepsets_sum_step)
            return bins, None, edge_targets
        if self.cfg.task == "settransformer_max":
            bins, edge_targets = self._materialize_shifted_edge_targets(self._build_settransformer_max_step)
            return bins, None, edge_targets
        if self.cfg.task == "temporal_memory":
            bins, edge_targets = self._materialize_temporal_memory()
            return bins, None, edge_targets
        if self.cfg.task == "node_count_threshold":
            bins, node_targets = self._materialize_shifted_node_targets(self._build_node_count_threshold_step)
            return bins, node_targets, None
        if self.cfg.task == "node_keyed_trigger":
            bins, node_targets = self._materialize_shifted_node_targets(self._build_node_keyed_trigger_step)
            return bins, node_targets, None
        if self.cfg.task == "node_temporal_state":
            bins, node_targets = self._materialize_temporal_node_classification()
            return bins, node_targets, None
        if self.cfg.task == "node_sum_regression":
            bins, node_targets = self._materialize_shifted_node_targets(self._build_deepsets_sum_step)
            return bins, node_targets, None
        if self.cfg.task == "node_keyed_value":
            bins, node_targets = self._materialize_shifted_node_targets(self._build_settransformer_max_step)
            return bins, node_targets, None
        if self.cfg.task == "node_temporal_regression":
            bins, node_targets = self._materialize_temporal_node_regression()
            return bins, node_targets, None
        if self.cfg.task in {"edge_count_threshold", "edge_threshold_classification"}:
            bins, edge_targets = self._materialize_shifted_edge_targets(self._build_node_count_threshold_step)
            return bins, None, edge_targets
        if self.cfg.task in {"edge_keyed_trigger", "edge_keyed_trigger_classification"}:
            bins, edge_targets = self._materialize_shifted_edge_targets(self._build_node_keyed_trigger_step)
            return bins, None, edge_targets
        if self.cfg.task == "edge_temporal_state":
            bins, edge_targets = self._materialize_temporal_edge_classification()
            return bins, None, edge_targets
        if self.cfg.task == "associative_retrieval":
            bins, edge_targets = self._materialize_shifted_edge_targets(self._build_associative_retrieval_step)
            return bins, None, edge_targets
        if self.cfg.task == "conservative_oscillator":
            bins, edge_targets = self._materialize_conservative_oscillator()
            return bins, None, edge_targets
        if self.cfg.task in {"diffusion", "wave", "coupled_oscillator"}:
            return self._materialize_physical_force_events(
                self._selected_field_topology(), dynamics=self.cfg.task
            ), None, None
        if self.cfg.task in {"wave_grid", "wave_torus", "wave_doorway", "wave_swisscheese"}:
            bins, edge_targets = self._materialize_grid_wave(self.cfg.task)
            return bins, None, edge_targets
        if self.cfg.task in {"edge_ranking_sum_shift", "next_dst_ranking"}:
            return self._materialize_shifted_ranking_stream(self._build_edge_ranking_sum_shift_step), None, None
        if self.cfg.task in {"edge_ranking_keyed_shift", "edge_retrieval"}:
            return self._materialize_shifted_ranking_stream(self._build_edge_ranking_keyed_shift_step), None, None
        if self.cfg.task in {"edge_ranking_temporal", "next_dst_temporal_ranking"}:
            return self._materialize_temporal_ranking_stream(), None, None
        raise AssertionError(f"Unhandled synthetic task: {self.cfg.task}")

    def _materialize_shifted_edge_targets(self, step_builder) -> tuple[list[EventBatch], list[EdgeTargetBatch]]:
        bins: list[EventBatch] = []
        edge_targets: list[EdgeTargetBatch] = []
        pending_target = np.zeros((self.cfg.num_nodes,), dtype=np.float32)
        for t in range(int(self.cfg.num_bins)):
            events, next_target = step_builder(t)
            bins.append(events)
            edge_targets.append(self._make_edge_target_batch(pending_target, t))
            pending_target = next_target.astype(np.float32, copy=False)
        return bins, edge_targets

    def _materialize_shifted_node_targets(self, step_builder) -> tuple[list[EventBatch], list[torch.Tensor]]:
        bins: list[EventBatch] = []
        node_targets: list[torch.Tensor] = []
        pending_target = np.zeros((self.cfg.num_nodes,), dtype=np.float32)
        for t in range(int(self.cfg.num_bins)):
            events, next_target = step_builder(t)
            bins.append(events)
            node_targets.append(self._make_node_target_tensor(pending_target))
            pending_target = next_target.astype(np.float32, copy=False)
        return bins, node_targets

    def _materialize_shifted_ranking_stream(self, step_builder) -> list[EventBatch]:
        bins: list[EventBatch] = []
        pending_dst = np.arange(int(self.cfg.num_nodes), dtype=np.int64)
        for t in range(int(self.cfg.num_bins)):
            events, next_pending_dst = step_builder(t, pending_dst)
            bins.append(events)
            pending_dst = next_pending_dst.astype(np.int64, copy=False)
        return bins

    def _build_deepsets_sum_step(self, t: int) -> tuple[EventBatch, np.ndarray]:
        num_nodes = int(self.cfg.num_nodes)
        events_per_bin = int(self.cfg.events_per_bin)
        src = self._rng.integers(0, num_nodes, size=events_per_bin, dtype=np.int64)
        dst = self._rng.integers(0, num_nodes, size=events_per_bin, dtype=np.int64)
        values = self._rng.normal(loc=0.0, scale=1.0, size=events_per_bin).astype(np.float32)
        features = values.reshape(-1, 1)
        target = np.zeros((num_nodes,), dtype=np.float32)
        np.add.at(target, src, values)
        np.add.at(target, dst, values)
        scale = math.sqrt(max(1.0, (2.0 * events_per_bin) / max(1, num_nodes)))
        target /= scale
        return self._make_event_batch(src, dst, features, t), target

    def _build_settransformer_max_step(self, t: int) -> tuple[EventBatch, np.ndarray]:
        num_nodes = int(self.cfg.num_nodes)
        events_per_bin = int(self.cfg.events_per_bin)
        src = self._rng.integers(0, num_nodes, size=events_per_bin, dtype=np.int64)
        dst = self._rng.integers(0, num_nodes, size=events_per_bin, dtype=np.int64)
        values = self._rng.uniform(-1.0, 1.0, size=events_per_bin).astype(np.float32)
        keys = self._rng.normal(loc=0.0, scale=1.0, size=events_per_bin).astype(np.float32)
        features = np.stack([values, keys], axis=1)

        target = np.zeros((num_nodes,), dtype=np.float32)
        best_key = np.full((num_nodes,), -np.inf, dtype=np.float32)
        for idx in range(events_per_bin):
            value = float(values[idx])
            key = float(keys[idx])
            src_idx = int(src[idx])
            dst_idx = int(dst[idx])
            if key > best_key[src_idx]:
                best_key[src_idx] = key
                target[src_idx] = value
            if key > best_key[dst_idx]:
                best_key[dst_idx] = key
                target[dst_idx] = value
        return self._make_event_batch(src, dst, features, t), target

    def _build_node_count_threshold_step(self, t: int) -> tuple[EventBatch, np.ndarray]:
        num_nodes = int(self.cfg.num_nodes)
        events_per_bin = int(self.cfg.events_per_bin)
        src = self._rng.integers(0, num_nodes, size=events_per_bin, dtype=np.int64)
        dst = self._rng.integers(0, num_nodes, size=events_per_bin, dtype=np.int64)
        counts = np.zeros((num_nodes,), dtype=np.int32)
        np.add.at(counts, src, 1)
        np.add.at(counts, dst, 1)
        lam = (2.0 * events_per_bin) / max(1, num_nodes)
        threshold = max(1, int(round(lam)))
        labels = (counts >= threshold).astype(np.float32)
        return self._make_event_batch(src, dst, None, t), labels

    def _build_node_keyed_trigger_step(self, t: int) -> tuple[EventBatch, np.ndarray]:
        num_nodes = int(self.cfg.num_nodes)
        events_per_bin = int(self.cfg.events_per_bin)
        src = self._rng.integers(0, num_nodes, size=events_per_bin, dtype=np.int64)
        dst = self._rng.integers(0, num_nodes, size=events_per_bin, dtype=np.int64)
        values = self._rng.uniform(-1.0, 1.0, size=events_per_bin).astype(np.float32)
        keys = self._rng.normal(loc=0.0, scale=1.0, size=events_per_bin).astype(np.float32)
        features = np.stack([values, keys], axis=1)

        labels = np.zeros((num_nodes,), dtype=np.float32)
        best_key = np.full((num_nodes,), -np.inf, dtype=np.float32)
        best_value = np.zeros((num_nodes,), dtype=np.float32)
        for idx in range(events_per_bin):
            value = float(values[idx])
            key = float(keys[idx])
            src_idx = int(src[idx])
            dst_idx = int(dst[idx])
            if key > best_key[src_idx]:
                best_key[src_idx] = key
                best_value[src_idx] = value
            if key > best_key[dst_idx]:
                best_key[dst_idx] = key
                best_value[dst_idx] = value
        labels = (best_value > 0.0).astype(np.float32)
        return self._make_event_batch(src, dst, features, t), labels

    def _build_associative_retrieval_step(self, t: int) -> tuple[EventBatch, np.ndarray]:
        num_nodes = int(self.cfg.num_nodes)
        candidates_per_node = max(3, int(self.cfg.events_per_bin) // max(1, num_nodes))
        total_events = num_nodes * (candidates_per_node + 1)

        src = np.repeat(np.arange(num_nodes, dtype=np.int64), candidates_per_node + 1)
        dst = src.copy()
        features = np.zeros((total_events, 6), dtype=np.float32)
        target = np.zeros((num_nodes,), dtype=np.float32)

        row = 0
        query_angles = self._rng.uniform(0.0, 2.0 * np.pi, size=num_nodes)
        query_vecs = np.stack([np.cos(query_angles), np.sin(query_angles)], axis=1).astype(np.float32)
        for node_idx in range(num_nodes):
            query = query_vecs[node_idx]
            best_score = -float("inf")
            best_value = 0.0
            for _ in range(candidates_per_node):
                key_angle = float(self._rng.uniform(0.0, 2.0 * np.pi))
                key = np.array([math.cos(key_angle), math.sin(key_angle)], dtype=np.float32)
                key += self._rng.normal(loc=0.0, scale=0.08, size=2).astype(np.float32)
                value = float(self._rng.uniform(-1.0, 1.0))
                score = float(np.dot(key, query))
                features[row] = np.array([value, key[0], key[1], 0.0, 0.0, 0.0], dtype=np.float32)
                if score > best_score:
                    best_score = score
                    best_value = value
                row += 1
            features[row] = np.array([0.0, 0.0, 0.0, query[0], query[1], 1.0], dtype=np.float32)
            target[node_idx] = best_value
            row += 1

        return self._make_event_batch(src, dst, features, t), target

    def _build_edge_ranking_sum_shift_step(
        self,
        t: int,
        pending_dst: np.ndarray,
    ) -> tuple[EventBatch, np.ndarray]:
        num_nodes = int(self.cfg.num_nodes)
        events_per_bin = int(self.cfg.events_per_bin)
        src = self._balanced_node_ids(events_per_bin)
        dst = pending_dst[src]
        values = self._rng.normal(loc=0.0, scale=1.0, size=events_per_bin).astype(np.float32)
        features = values.reshape(-1, 1)
        node_signal = np.zeros((num_nodes,), dtype=np.float32)
        np.add.at(node_signal, src, values)
        node_idx = np.arange(num_nodes, dtype=np.int64)
        positive_dst = (node_idx + 1) % num_nodes
        negative_dst = (node_idx - 1) % num_nodes
        next_pending_dst = np.where(node_signal >= 0.0, positive_dst, negative_dst).astype(np.int64)
        return self._make_event_batch(src, dst, features, t), next_pending_dst

    def _build_edge_ranking_keyed_shift_step(
        self,
        t: int,
        pending_dst: np.ndarray,
    ) -> tuple[EventBatch, np.ndarray]:
        num_nodes = int(self.cfg.num_nodes)
        events_per_bin = int(self.cfg.events_per_bin)
        src = self._balanced_node_ids(events_per_bin)
        dst = pending_dst[src]
        shift_choices = self._rng.choice(np.array([-2, -1, 1, 2], dtype=np.int64), size=events_per_bin)
        keys = self._rng.normal(loc=0.0, scale=1.0, size=events_per_bin).astype(np.float32)
        features = np.stack([shift_choices.astype(np.float32) / 2.0, keys], axis=1)

        best_key = np.full((num_nodes,), -np.inf, dtype=np.float32)
        best_shift = np.ones((num_nodes,), dtype=np.int64)
        for idx in range(events_per_bin):
            src_idx = int(src[idx])
            key = float(keys[idx])
            if key > best_key[src_idx]:
                best_key[src_idx] = key
                best_shift[src_idx] = int(shift_choices[idx])

        node_idx = np.arange(num_nodes, dtype=np.int64)
        next_pending_dst = (node_idx + best_shift) % num_nodes
        return self._make_event_batch(src, dst, features, t), next_pending_dst

    def _materialize_temporal_memory(self) -> tuple[list[EventBatch], list[EdgeTargetBatch]]:
        num_nodes = int(self.cfg.num_nodes)
        node_idx = np.arange(num_nodes, dtype=np.int64)
        phases = self._rng.uniform(0.0, 2.0 * np.pi, size=num_nodes)
        freqs = self._rng.uniform(0.06, 0.18, size=num_nodes)
        amps = self._rng.uniform(0.6, 1.1, size=num_nodes)
        x_prev = np.zeros((num_nodes,), dtype=np.float32)
        x_curr = self._rng.normal(loc=0.0, scale=0.1, size=num_nodes).astype(np.float32)

        bins: list[EventBatch] = []
        edge_targets: list[EdgeTargetBatch] = []
        for t in range(int(self.cfg.num_bins)):
            drive = amps * np.sin(freqs * t + phases)
            drive += 0.35 * np.cos((0.5 * freqs * t) + (1.7 * phases))
            drive += self._rng.normal(loc=0.0, scale=0.02, size=num_nodes)
            drive = drive.astype(np.float32)

            bins.append(self._make_event_batch(node_idx, node_idx, drive.reshape(-1, 1), t))
            edge_targets.append(self._make_edge_target_batch(x_curr, t))

            x_next = (1.35 * x_curr) - (0.55 * x_prev) + (0.30 * drive)
            x_next = np.clip(x_next, -3.0, 3.0).astype(np.float32)
            x_prev, x_curr = x_curr, x_next

        return bins, edge_targets

    def _materialize_conservative_oscillator(self) -> tuple[list[EventBatch], list[EdgeTargetBatch]]:
        num_nodes = int(self.cfg.num_nodes)
        node_idx = np.arange(num_nodes, dtype=np.int64)
        phases = self._rng.uniform(0.0, 2.0 * np.pi, size=num_nodes)
        freqs = self._rng.uniform(0.04, 0.12, size=num_nodes)
        amps = self._rng.uniform(0.35, 0.75, size=num_nodes)
        x_prev = self._rng.normal(loc=0.0, scale=0.05, size=num_nodes).astype(np.float32)
        x_curr = self._rng.normal(loc=0.0, scale=0.10, size=num_nodes).astype(np.float32)

        bins: list[EventBatch] = []
        edge_targets: list[EdgeTargetBatch] = []
        for t in range(int(self.cfg.num_bins)):
            drive = amps * np.sin((freqs * t) + phases)
            drive += 0.18 * np.cos((0.45 * freqs * t) + (0.7 * phases))
            drive += self._rng.normal(loc=0.0, scale=0.01, size=num_nodes)
            drive = drive.astype(np.float32)

            bins.append(self._make_event_batch(node_idx, node_idx, drive.reshape(-1, 1), t))
            edge_targets.append(self._make_edge_target_batch(x_curr, t))

            x_next = (1.92 * x_curr) - (0.96 * x_prev) + (0.08 * drive)
            x_next = np.clip(x_next, -4.0, 4.0).astype(np.float32)
            x_prev, x_curr = x_curr, x_next

        return bins, edge_targets

    def _materialize_diffusion(self) -> tuple[list[EventBatch], list[EdgeTargetBatch]]:
        topology = self._selected_field_topology()
        if topology == "ring":
            return self._materialize_ring_diffusion()
        return self._materialize_grid_diffusion(topology)

    def _selected_field_topology(self) -> str:
        return self.cfg.field_topology or self.cfg.diffusion_topology or "ring"

    @staticmethod
    def _topology_task_name(topology: str) -> str:
        return {
            "grid": "wave_grid",
            "torus": "wave_torus",
            "doorway": "wave_doorway",
            "swisscheese": "wave_swisscheese",
        }[topology]

    def _materialize_ring_diffusion(self) -> tuple[list[EventBatch], list[EdgeTargetBatch]]:
        num_nodes = int(self.cfg.num_nodes)
        node_idx = np.arange(num_nodes, dtype=np.int64)
        ring_dst_fwd = (node_idx + 1) % num_nodes
        ring_dst_bwd = (node_idx - 1) % num_nodes
        phases = self._rng.uniform(0.0, 2.0 * np.pi, size=num_nodes)
        freqs = self._rng.uniform(0.05, 0.16, size=num_nodes)
        amps = self._rng.uniform(0.30, 0.70, size=num_nodes)
        x_curr = self._rng.normal(loc=0.0, scale=0.10, size=num_nodes).astype(np.float32)

        bins: list[EventBatch] = []
        edge_targets: list[EdgeTargetBatch] = []
        for t in range(int(self.cfg.num_bins)):
            drive = amps * np.sin((freqs * t) + phases)
            drive += 0.22 * np.cos((0.35 * freqs * t) + (1.1 * phases))
            drive += self._rng.normal(loc=0.0, scale=0.015, size=num_nodes)
            drive = drive.astype(np.float32)

            src = np.concatenate([node_idx, node_idx, node_idx])
            dst = np.concatenate([ring_dst_fwd, ring_dst_bwd, node_idx])
            signal = np.concatenate([x_curr, x_curr, drive]).astype(np.float32)
            is_drive = np.concatenate(
                [
                    np.zeros((num_nodes,), dtype=np.float32),
                    np.zeros((num_nodes,), dtype=np.float32),
                    np.ones((num_nodes,), dtype=np.float32),
                ]
            )
            features = np.stack([signal, is_drive], axis=1)
            bins.append(self._make_event_batch(src, dst, features, t))
            edge_targets.append(self._make_edge_target_batch(x_curr, t))

            x_next = (
                0.58 * x_curr
                + 0.18 * np.roll(x_curr, 1)
                + 0.18 * np.roll(x_curr, -1)
                + 0.25 * drive
            )
            x_curr = np.clip(x_next, -3.0, 3.0).astype(np.float32)

        return bins, edge_targets

    def _materialize_grid_diffusion(
        self,
        topology: str,
    ) -> tuple[list[EventBatch], list[EdgeTargetBatch]]:
        """Materialize first-order diffusion on a grid-derived topology."""
        topology_task = self._topology_task_name(topology)
        active, edge_src, edge_dst, degree = self._grid_wave_topology(topology_task)
        num_nodes = int(self.cfg.num_nodes)
        active_nodes = np.flatnonzero(active).astype(np.int64)
        phases = self._rng.uniform(0.0, 2.0 * np.pi, size=active_nodes.size)
        freqs = self._rng.uniform(0.05, 0.16, size=active_nodes.size)
        amps = self._rng.uniform(0.30, 0.70, size=active_nodes.size)
        x_curr = np.zeros((num_nodes,), dtype=np.float32)
        x_curr[active] = self._rng.normal(loc=0.0, scale=0.10, size=active_nodes.size)

        bins: list[EventBatch] = []
        edge_targets: list[EdgeTargetBatch] = []
        for t in range(int(self.cfg.num_bins)):
            drive = np.zeros((num_nodes,), dtype=np.float32)
            drive[active_nodes] = (
                amps * np.sin(freqs * t + phases)
                + 0.22 * np.cos((0.35 * freqs * t) + (1.1 * phases))
                + self._rng.normal(loc=0.0, scale=0.015, size=active_nodes.size)
            ).astype(np.float32)

            src = np.concatenate([edge_src, active_nodes])
            dst = np.concatenate([edge_dst, active_nodes])
            signal = np.concatenate([x_curr[edge_src], drive[active_nodes]]).astype(np.float32)
            is_drive = np.concatenate([
                np.zeros(edge_src.size, dtype=np.float32),
                np.ones(active_nodes.size, dtype=np.float32),
            ])
            bins.append(self._make_event_batch(src, dst, np.stack([signal, is_drive], axis=1), t))
            edge_targets.append(self._make_edge_target_batch(x_curr, t))

            neighbor_sum = np.zeros((num_nodes,), dtype=np.float32)
            np.add.at(neighbor_sum, edge_dst, x_curr[edge_src])
            neighbor_mean = np.divide(
                neighbor_sum,
                degree,
                out=np.zeros_like(neighbor_sum),
                where=degree > 0.0,
            )
            x_next = 0.58 * x_curr + 0.36 * neighbor_mean + 0.25 * drive
            x_next[~active] = 0.0
            x_curr = np.clip(x_next, -3.0, 3.0).astype(np.float32)

        return bins, edge_targets

    def _grid_wave_topology(self, task_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return active nodes, directed edges, and degrees for a square wave lattice."""
        num_nodes = int(self.cfg.num_nodes)
        height = math.isqrt(num_nodes)
        while height > 1 and num_nodes % height != 0:
            height -= 1
        width = num_nodes // height
        if height < 4 or width < 4:
            raise ValueError(
                f"{task_name} requires num_nodes to factor into a lattice of at least 4x4; "
                f"got {self.cfg.num_nodes}."
            )
        active = np.ones((height, width), dtype=bool)
        if task_name == "wave_swisscheese":
            radius = max(1.15, 0.115 * min(height, width))
            rows, cols = np.ogrid[:height, :width]
            for row_fraction, col_fraction in ((0.28, 0.30), (0.72, 0.32), (0.50, 0.72)):
                center_row = row_fraction * (height - 1)
                center_col = col_fraction * (width - 1)
                active[(rows - center_row) ** 2 + (cols - center_col) ** 2 <= radius**2] = False

        periodic = task_name == "wave_torus"
        doorway = task_name == "wave_doorway"
        wall_col = width // 2
        doorway_rows = set(range(max(0, height // 2 - 1), min(height, height // 2 + 2)))
        undirected_edges: list[tuple[int, int]] = []
        for row in range(height):
            for col in range(width):
                if not active[row, col]:
                    continue
                for row_step, col_step in ((0, 1), (1, 0)):
                    next_row = row + row_step
                    next_col = col + col_step
                    if periodic:
                        next_row %= height
                        next_col %= width
                    elif next_row >= height or next_col >= width:
                        continue
                    if not active[next_row, next_col]:
                        continue
                    crosses_doorway_wall = (
                        doorway
                        and row_step == 0
                        and col == wall_col - 1
                        and next_col == wall_col
                        and row not in doorway_rows
                    )
                    if not crosses_doorway_wall:
                        undirected_edges.append((row * width + col, next_row * width + next_col))

        if not undirected_edges:
            raise ValueError(f"{task_name} produced no active lattice edges.")
        edge_pairs = np.asarray(undirected_edges, dtype=np.int64)
        edge_src = np.concatenate([edge_pairs[:, 0], edge_pairs[:, 1]])
        edge_dst = np.concatenate([edge_pairs[:, 1], edge_pairs[:, 0]])
        degree = np.bincount(edge_src, minlength=num_nodes).astype(np.float32)
        return active.reshape(-1), edge_src, edge_dst, degree

    def _materialize_grid_wave(self, task_name: str) -> tuple[list[EventBatch], list[EdgeTargetBatch]]:
        """Materialize a sparse-driven second-order wave over a grid topology."""
        active, edge_src, edge_dst, degree = self._grid_wave_topology(task_name)
        num_nodes = int(self.cfg.num_nodes)
        active_nodes = np.flatnonzero(active).astype(np.int64)
        target_locations = ((0.16, 0.16), (0.78, 0.22), (0.24, 0.78))
        source_nodes: list[int] = []
        height = math.isqrt(num_nodes)
        while height > 1 and num_nodes % height != 0:
            height -= 1
        width = num_nodes // height
        node_rows, node_cols = np.divmod(active_nodes, width)
        for row_fraction, col_fraction in target_locations:
            distance_sq = (
                (node_rows - row_fraction * (height - 1)) ** 2
                + (node_cols - col_fraction * (width - 1)) ** 2
            )
            for candidate in active_nodes[np.argsort(distance_sq)]:
                if int(candidate) not in source_nodes:
                    source_nodes.append(int(candidate))
                    break
        sources = np.asarray(source_nodes, dtype=np.int64)
        phases = self._rng.uniform(0.0, 2.0 * np.pi, size=sources.size)
        frequencies = self._rng.uniform(0.045, 0.105, size=sources.size)
        amplitudes = self._rng.uniform(0.75, 1.10, size=sources.size)
        x_prev = np.zeros((num_nodes,), dtype=np.float32)
        x_curr = np.zeros((num_nodes,), dtype=np.float32)
        x_curr[active] = self._rng.normal(loc=0.0, scale=0.04, size=int(active.sum()))

        bins: list[EventBatch] = []
        edge_targets: list[EdgeTargetBatch] = []
        for t in range(int(self.cfg.num_bins)):
            drive = np.zeros((num_nodes,), dtype=np.float32)
            drive[sources] = (
                amplitudes * np.sin(frequencies * t + phases)
                + 0.22 * np.cos(0.40 * frequencies * t + 0.9 * phases)
            ).astype(np.float32)
            src = np.concatenate([edge_src, active_nodes])
            dst = np.concatenate([edge_dst, active_nodes])
            signal = np.concatenate([x_curr[edge_src], drive[active_nodes]]).astype(np.float32)
            is_drive = np.concatenate([
                np.zeros(edge_src.size, dtype=np.float32),
                np.ones(active_nodes.size, dtype=np.float32),
            ])
            bins.append(self._make_event_batch(src, dst, np.stack([signal, is_drive], axis=1), t))
            edge_targets.append(self._make_edge_target_batch(x_curr, t))

            neighbor_sum = np.zeros((num_nodes,), dtype=np.float32)
            np.add.at(neighbor_sum, edge_src, x_curr[edge_dst])
            laplacian = neighbor_sum - degree * x_curr
            x_next = 1.88 * x_curr - 0.93 * x_prev + 0.075 * laplacian + 0.17 * drive
            x_next[~active] = 0.0
            x_prev, x_curr = x_curr, np.clip(x_next, -4.0, 4.0).astype(np.float32)

        return bins, edge_targets

    def _materialize_wave_field(self, topology: str) -> tuple[list[EventBatch], list[EdgeTargetBatch]]:
        """Materialize the wave dynamic independently of the chosen topology."""
        if topology == "ring":
            return self._materialize_wave()
        return self._materialize_grid_wave(self._topology_task_name(topology))

    def _materialize_physical_force_events(self, topology: str, *, dynamics: str) -> list[EventBatch]:
        """Hidden physical simulator whose only public output is force events.

        ``force[i -> j]`` is the actual vector force exerted on target ``j`` by
        source ``i``. At the beginning of each episode, one self-event
        ``j -> j`` is a raindrop impulse applied at ``j``. The fixed support
        that generates pair forces, hidden node fields, and physical parameters
        stay outside the model pathway.
        """
        active, edge_src, edge_dst, _degree = self._field_topology_edges(topology)
        num_nodes = int(self.cfg.num_nodes)
        force_dim = int(self._task.event_dim)
        episodes = max(3, int(self.cfg.num_episodes))
        steps = int(self.cfg.num_bins)
        if dynamics == "diffusion":
            dt, gamma, omega, force_scale = 0.10, 0.18, 0.0, 0.80
        elif dynamics == "wave":
            dt, gamma, omega, force_scale = 0.10, 0.15, 0.80, 0.80
        elif dynamics == "coupled_oscillator":
            dt, gamma, omega, force_scale = 0.10, 0.10, 1.15, 0.65
        else:
            raise ValueError(f"Unknown physical dynamic {dynamics!r}.")
        adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        adjacency[edge_src, edge_dst] = 1.0
        active_nodes = np.flatnonzero(active)
        # A single physical rain direction, shared by every drop.  The first
        # two channels are interpreted as a planar field, so this is downward.
        drop_direction = np.zeros((force_dim,), dtype=np.float32)
        drop_direction[min(1, force_dim - 1)] = -1.0
        bins: list[EventBatch] = []

        for episode in range(episodes):
            h = np.zeros((num_nodes, force_dim), dtype=np.float32)
            v = np.zeros_like(h)
            # A quiet field: all energy is introduced by observed raindrop
            # impulses rather than hidden randomized initial conditions.
            for local_t in range(steps):
                # This is the physical interaction measurement.  It is not a
                # noisy encoding of a hidden state and no drive marker is sent.
                full_force = force_scale * (h[edge_src] - h[edge_dst])
                # Each episode starts from a quiet field.  Optional later drops
                # are observed exogenous interventions into that same evolving
                # field, rather than episode resets.
                interval = self.cfg.raindrop_interval
                has_drop = local_t == 0 or (
                    interval is not None and local_t % int(interval) == 0
                )
                force_magnitude = np.linalg.norm(full_force, axis=1)
                active_pairs = np.flatnonzero(force_magnitude > float(self.cfg.event_threshold))
                pair_budget = max(0, int(self.cfg.events_per_bin) - int(has_drop))
                if pair_budget == 0:
                    chosen = np.empty((0,), dtype=np.int64)
                elif active_pairs.size > pair_budget:
                    # A measurement budget should retain the strongest active
                    # interactions, not randomly discard the visible wavefront.
                    chosen = active_pairs[np.argsort(force_magnitude[active_pairs])[-pair_budget:]]
                else:
                    chosen = active_pairs
                bin_src, bin_dst = edge_src[chosen], edge_dst[chosen]
                force = full_force[chosen]
                if has_drop:
                    drop_node = int(self._rng.choice(active_nodes))
                    amplitude = float(self._rng.uniform(0.9, 1.3))
                    drop_force = (amplitude * drop_direction).reshape(1, force_dim)
                    bin_src = np.concatenate([bin_src, np.array([drop_node], dtype=np.int64)])
                    bin_dst = np.concatenate([bin_dst, np.array([drop_node], dtype=np.int64)])
                    force = np.concatenate([force, drop_force], axis=0)
                    is_external = np.zeros((force.shape[0],), dtype=bool)
                    is_external[-1] = True
                else:
                    drop_node = None
                    drop_force = None
                    is_external = np.zeros((force.shape[0],), dtype=bool)
                global_t = episode * steps + local_t
                bins.append(self._make_event_batch(
                    bin_src,
                    bin_dst,
                    force,
                    global_t,
                    episode=episode,
                    is_external=is_external,
                ))
                incoming = np.zeros_like(h)
                np.add.at(incoming, edge_dst, full_force)
                if drop_node is not None and drop_force is not None:
                    incoming[drop_node] += drop_force[0]
                if dynamics == "diffusion":
                    h = (1.0 - gamma * dt) * h + dt * incoming
                    v.fill(0.0)
                else:
                    v = (1.0 - gamma * dt) * v + dt * (incoming - (omega ** 2) * h)
                    h = h + dt * v
                h[~active] = 0.0
                v[~active] = 0.0

        self._hidden_truth = {
            "adjacency": torch.from_numpy(adjacency),
            "params": {
                "gamma": gamma,
                "dt": dt,
                "force_scale": force_scale,
                **({"omega": omega} if dynamics != "diffusion" else {}),
            },
            "raindrop_interval": self.cfg.raindrop_interval,
            "event_threshold": float(self.cfg.event_threshold),
        }
        return bins

    def _field_topology_edges(self, topology: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if topology != "ring":
            return self._grid_wave_topology(self._topology_task_name(topology))
        num_nodes = int(self.cfg.num_nodes)
        nodes = np.arange(num_nodes, dtype=np.int64)
        edge_src = np.concatenate([nodes, nodes])
        edge_dst = np.concatenate([(nodes + 1) % num_nodes, (nodes - 1) % num_nodes])
        return np.ones((num_nodes,), dtype=bool), edge_src, edge_dst, np.full((num_nodes,), 2.0, dtype=np.float32)

    def _materialize_coupled_oscillator_field(self, topology: str) -> tuple[list[EventBatch], list[EdgeTargetBatch]]:
        """Driven damped oscillators with local restoring force and graph coupling."""
        active, edge_src, edge_dst, degree = self._field_topology_edges(topology)
        num_nodes = int(self.cfg.num_nodes)
        active_nodes = np.flatnonzero(active).astype(np.int64)
        source_count = min(3, active_nodes.size)
        sources = active_nodes[np.linspace(0, active_nodes.size - 1, source_count, dtype=np.int64)]
        phases = self._rng.uniform(0.0, 2.0 * np.pi, size=source_count)
        frequencies = self._rng.uniform(0.045, 0.095, size=source_count)
        amplitudes = self._rng.uniform(0.70, 1.00, size=source_count)
        x_prev = np.zeros((num_nodes,), dtype=np.float32)
        x_curr = np.zeros((num_nodes,), dtype=np.float32)
        x_curr[active] = self._rng.normal(loc=0.0, scale=0.05, size=active_nodes.size)

        bins: list[EventBatch] = []
        edge_targets: list[EdgeTargetBatch] = []
        for t in range(int(self.cfg.num_bins)):
            drive = np.zeros((num_nodes,), dtype=np.float32)
            drive[sources] = (amplitudes * np.sin(frequencies * t + phases)).astype(np.float32)
            src = np.concatenate([edge_src, active_nodes])
            dst = np.concatenate([edge_dst, active_nodes])
            signal = np.concatenate([x_curr[edge_src], drive[active_nodes]]).astype(np.float32)
            is_drive = np.concatenate([np.zeros(edge_src.size, dtype=np.float32), np.ones(active_nodes.size, dtype=np.float32)])
            bins.append(self._make_event_batch(src, dst, np.stack([signal, is_drive], axis=1), t))
            edge_targets.append(self._make_edge_target_batch(x_curr, t))

            neighbor_sum = np.zeros((num_nodes,), dtype=np.float32)
            np.add.at(neighbor_sum, edge_src, x_curr[edge_dst])
            laplacian = neighbor_sum - degree * x_curr
            # 1.66 and -0.84 encode damping plus a local harmonic restoring force;
            # the Laplacian is the independent topology-dependent coupling term.
            x_next = 1.66 * x_curr - 0.84 * x_prev + 0.055 * laplacian + 0.12 * drive
            x_next[~active] = 0.0
            x_prev, x_curr = x_curr, np.clip(x_next, -4.0, 4.0).astype(np.float32)
        return bins, edge_targets

    def _materialize_wave(self) -> tuple[list[EventBatch], list[EdgeTargetBatch]]:
        num_nodes = int(self.cfg.num_nodes)
        node_idx = np.arange(num_nodes, dtype=np.int64)
        ring_dst_fwd = (node_idx + 1) % num_nodes
        ring_dst_bwd = (node_idx - 1) % num_nodes
        phases = self._rng.uniform(0.0, 2.0 * np.pi, size=num_nodes)
        freqs = self._rng.uniform(0.04, 0.12, size=num_nodes)
        amps = self._rng.uniform(0.30, 0.65, size=num_nodes)
        x_prev = self._rng.normal(loc=0.0, scale=0.05, size=num_nodes).astype(np.float32)
        x_curr = self._rng.normal(loc=0.0, scale=0.10, size=num_nodes).astype(np.float32)

        bins: list[EventBatch] = []
        edge_targets: list[EdgeTargetBatch] = []
        for t in range(int(self.cfg.num_bins)):
            drive = amps * np.sin((freqs * t) + phases)
            drive += 0.20 * np.cos((0.40 * freqs * t) + (0.9 * phases))
            drive += self._rng.normal(loc=0.0, scale=0.01, size=num_nodes)
            drive = drive.astype(np.float32)

            src = np.concatenate([node_idx, node_idx, node_idx])
            dst = np.concatenate([ring_dst_fwd, ring_dst_bwd, node_idx])
            signal = np.concatenate([x_curr, x_curr, drive]).astype(np.float32)
            is_drive = np.concatenate(
                [
                    np.zeros((num_nodes,), dtype=np.float32),
                    np.zeros((num_nodes,), dtype=np.float32),
                    np.ones((num_nodes,), dtype=np.float32),
                ]
            )
            features = np.stack([signal, is_drive], axis=1)
            bins.append(self._make_event_batch(src, dst, features, t))
            edge_targets.append(self._make_edge_target_batch(x_curr, t))

            lap = np.roll(x_curr, 1) - (2.0 * x_curr) + np.roll(x_curr, -1)
            x_next = (1.86 * x_curr) - (0.92 * x_prev) + (0.10 * lap) + (0.08 * drive)
            x_next = np.clip(x_next, -4.0, 4.0).astype(np.float32)
            x_prev, x_curr = x_curr, x_next

        return bins, edge_targets

    def _materialize_temporal_node_classification(self) -> tuple[list[EventBatch], list[torch.Tensor]]:
        num_nodes = int(self.cfg.num_nodes)
        node_idx = np.arange(num_nodes, dtype=np.int64)
        phases = self._rng.uniform(0.0, 2.0 * np.pi, size=num_nodes)
        freqs = self._rng.uniform(0.06, 0.18, size=num_nodes)
        amps = self._rng.uniform(0.6, 1.1, size=num_nodes)
        x_prev = np.zeros((num_nodes,), dtype=np.float32)
        x_curr = self._rng.normal(loc=0.0, scale=0.1, size=num_nodes).astype(np.float32)
        pending_target = (x_curr > 0.0).astype(np.float32)

        bins: list[EventBatch] = []
        node_targets: list[torch.Tensor] = []
        for t in range(int(self.cfg.num_bins)):
            drive = amps * np.sin(freqs * t + phases)
            drive += 0.35 * np.cos((0.5 * freqs * t) + (1.7 * phases))
            drive += self._rng.normal(loc=0.0, scale=0.02, size=num_nodes)
            drive = drive.astype(np.float32)

            bins.append(self._make_event_batch(node_idx, node_idx, drive.reshape(-1, 1), t))
            node_targets.append(self._make_node_target_tensor(pending_target))

            x_next = (1.35 * x_curr) - (0.55 * x_prev) + (0.30 * drive)
            x_next = np.clip(x_next, -3.0, 3.0).astype(np.float32)
            pending_target = (x_next > 0.0).astype(np.float32)
            x_prev, x_curr = x_curr, x_next

        return bins, node_targets

    def _materialize_temporal_node_regression(self) -> tuple[list[EventBatch], list[torch.Tensor]]:
        num_nodes = int(self.cfg.num_nodes)
        node_idx = np.arange(num_nodes, dtype=np.int64)
        phases = self._rng.uniform(0.0, 2.0 * np.pi, size=num_nodes)
        freqs = self._rng.uniform(0.06, 0.18, size=num_nodes)
        amps = self._rng.uniform(0.6, 1.1, size=num_nodes)
        x_prev = np.zeros((num_nodes,), dtype=np.float32)
        x_curr = self._rng.normal(loc=0.0, scale=0.1, size=num_nodes).astype(np.float32)
        pending_target = x_curr.astype(np.float32, copy=True)

        bins: list[EventBatch] = []
        node_targets: list[torch.Tensor] = []
        for t in range(int(self.cfg.num_bins)):
            drive = amps * np.sin(freqs * t + phases)
            drive += 0.35 * np.cos((0.5 * freqs * t) + (1.7 * phases))
            drive += self._rng.normal(loc=0.0, scale=0.02, size=num_nodes)
            drive = drive.astype(np.float32)

            bins.append(self._make_event_batch(node_idx, node_idx, drive.reshape(-1, 1), t))
            node_targets.append(self._make_node_target_tensor(pending_target))

            x_next = (1.35 * x_curr) - (0.55 * x_prev) + (0.30 * drive)
            x_next = np.clip(x_next, -3.0, 3.0).astype(np.float32)
            pending_target = x_next.astype(np.float32, copy=True)
            x_prev, x_curr = x_curr, x_next

        return bins, node_targets

    def _materialize_temporal_edge_classification(self) -> tuple[list[EventBatch], list[EdgeTargetBatch]]:
        num_nodes = int(self.cfg.num_nodes)
        node_idx = np.arange(num_nodes, dtype=np.int64)
        phases = self._rng.uniform(0.0, 2.0 * np.pi, size=num_nodes)
        freqs = self._rng.uniform(0.06, 0.18, size=num_nodes)
        amps = self._rng.uniform(0.6, 1.1, size=num_nodes)
        x_prev = np.zeros((num_nodes,), dtype=np.float32)
        x_curr = self._rng.normal(loc=0.0, scale=0.1, size=num_nodes).astype(np.float32)
        pending_target = (x_curr > 0.0).astype(np.float32)

        bins: list[EventBatch] = []
        edge_targets: list[EdgeTargetBatch] = []
        for t in range(int(self.cfg.num_bins)):
            drive = amps * np.sin(freqs * t + phases)
            drive += 0.35 * np.cos((0.5 * freqs * t) + (1.7 * phases))
            drive += self._rng.normal(loc=0.0, scale=0.02, size=num_nodes)
            drive = drive.astype(np.float32)

            bins.append(self._make_event_batch(node_idx, node_idx, drive.reshape(-1, 1), t))
            edge_targets.append(self._make_edge_target_batch(pending_target, t))

            x_next = (1.35 * x_curr) - (0.55 * x_prev) + (0.30 * drive)
            x_next = np.clip(x_next, -3.0, 3.0).astype(np.float32)
            pending_target = (x_next > 0.0).astype(np.float32)
            x_prev, x_curr = x_curr, x_next

        return bins, edge_targets

    def _materialize_temporal_ranking_stream(self) -> list[EventBatch]:
        num_nodes = int(self.cfg.num_nodes)
        node_idx = np.arange(num_nodes, dtype=np.int64)
        phases = self._rng.uniform(0.0, 2.0 * np.pi, size=num_nodes)
        freqs = self._rng.uniform(0.06, 0.18, size=num_nodes)
        amps = self._rng.uniform(0.6, 1.1, size=num_nodes)
        x_prev = np.zeros((num_nodes,), dtype=np.float32)
        x_curr = self._rng.normal(loc=0.0, scale=0.1, size=num_nodes).astype(np.float32)
        pending_dst = self._temporal_destinations_from_state(x_curr)

        bins: list[EventBatch] = []
        for t in range(int(self.cfg.num_bins)):
            drive = amps * np.sin(freqs * t + phases)
            drive += 0.35 * np.cos((0.5 * freqs * t) + (1.7 * phases))
            drive += self._rng.normal(loc=0.0, scale=0.02, size=num_nodes)
            drive = drive.astype(np.float32)

            bins.append(self._make_event_batch(node_idx, pending_dst, drive.reshape(-1, 1), t))

            x_next = (1.35 * x_curr) - (0.55 * x_prev) + (0.30 * drive)
            x_next = np.clip(x_next, -3.0, 3.0).astype(np.float32)
            pending_dst = self._temporal_destinations_from_state(x_next)
            x_prev, x_curr = x_curr, x_next

        return bins

    def _balanced_node_ids(self, total_events: int) -> np.ndarray:
        num_nodes = int(self.cfg.num_nodes)
        src = np.arange(total_events, dtype=np.int64) % max(1, num_nodes)
        self._rng.shuffle(src)
        return src

    def _temporal_destinations_from_state(self, x: np.ndarray) -> np.ndarray:
        node_idx = np.arange(int(self.cfg.num_nodes), dtype=np.int64)
        shifts = np.where(
            x < -0.5,
            -2,
            np.where(x < 0.0, -1, np.where(x < 0.5, 1, 2)),
        ).astype(np.int64)
        return (node_idx + shifts) % int(self.cfg.num_nodes)

    def _make_event_batch(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        features: Optional[np.ndarray],
        t: int,
        episode: Optional[int] = None,
        is_external: Optional[np.ndarray] = None,
    ) -> EventBatch:
        batch = EventBatch(
            src=cast(torch.LongTensor, torch.from_numpy(src.astype(np.int64, copy=False))),
            dst=cast(torch.LongTensor, torch.from_numpy(dst.astype(np.int64, copy=False))),
            features=None if features is None else torch.from_numpy(features.astype(np.float32, copy=False)),
            # Empty physical bins still carry sequence bookkeeping.  The
            # scalar metadata is never sent to a model, but lets trainers
            # preserve independent-episode boundaries through quiet periods.
            t=cast(torch.LongTensor, torch.full((max(1, src.size),), int(t), dtype=torch.long)),
            episode=(
                None
                if episode is None
                else cast(torch.LongTensor, torch.full((max(1, src.size),), int(episode), dtype=torch.long))
            ),
            is_external=(
                None
                if is_external is None
                else cast(torch.BoolTensor, torch.from_numpy(is_external.astype(bool, copy=False)))
            ),
        )
        if self.cfg.device is not None:
            batch = batch.to(self.cfg.device)
        return batch

    def _make_edge_target_batch(self, targets: np.ndarray, t: int) -> EdgeTargetBatch:
        num_nodes = int(self.cfg.num_nodes)
        src = np.arange(num_nodes, dtype=np.int64)
        dst = np.arange(num_nodes, dtype=np.int64)
        events = self._make_event_batch(src, dst, None, t)
        target_tensor = torch.from_numpy(targets.astype(np.float32, copy=False))
        if self.cfg.device is not None:
            target_tensor = target_tensor.to(self.cfg.device)
        return EdgeTargetBatch(events=events, targets=target_tensor)

    def _make_node_target_tensor(self, targets: np.ndarray) -> torch.Tensor:
        target_tensor = torch.from_numpy(targets.astype(np.float32, copy=False))
        if self.cfg.device is not None:
            target_tensor = target_tensor.to(self.cfg.device)
        return target_tensor

    def _compute_splits(self, num_bins: int) -> Dict[str, tuple[int, int]]:
        train_frac, val_frac, test_frac = self.cfg.split_fracs
        if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-6:
            raise ValueError("split_fracs must sum to 1.0")
        if num_bins == 0:
            return {"train": (0, -1), "val": (0, -1), "test": (0, -1)}
        train_end = int(num_bins * train_frac)
        val_end = train_end + int(num_bins * val_frac)
        return {
            "train": (0, max(0, train_end - 1)),
            "val": (train_end, max(train_end, val_end - 1)),
            "test": (val_end, num_bins - 1),
        }

    def _compute_episode_splits(self) -> Dict[str, tuple[int, int]]:
        """Keep independently simulated trajectories intact across splits."""
        episodes = max(3, int(self.cfg.num_episodes))
        steps = int(self.cfg.num_bins)
        train_eps = max(1, int(episodes * self.cfg.split_fracs[0]))
        val_eps = max(1, int(episodes * self.cfg.split_fracs[1]))
        if train_eps + val_eps >= episodes:
            val_eps = 1
            train_eps = episodes - 2
        train_end = train_eps * steps - 1
        val_end = (train_eps + val_eps) * steps - 1
        return {
            "train": (0, train_end),
            "val": (train_end + 1, val_end),
            "test": (val_end + 1, episodes * steps - 1),
        }

    def spec(self) -> DataSpec:
        dataset_name = self.cfg.name or f"synthetic_{self.cfg.task}"
        task_axes = self._task.axes()
        task_tags = list(self._task.tags())
        topology = self._selected_field_topology()
        field_dynamic = self.cfg.task in {"diffusion", "wave", "coupled_oscillator"}
        if field_dynamic:
            generator_params = {} if task_axes["generator_params"] is None else dict(task_axes["generator_params"])
            generator_params["topology"] = topology
            generator_params["event_threshold"] = float(self.cfg.event_threshold)
            task_axes["generator_params"] = generator_params
        if field_dynamic and topology != "ring":
            graph_types = {
                "grid": "grid",
                "torus": "torus_grid",
                "doorway": "grid_doorway",
                "swisscheese": "grid_swisscheese",
            }
            task_axes["graph_type"] = graph_types[topology]
            task_axes["generator_family"] = f"grid_{self._task.dynamics_type}"
            task_axes["event_structure"] = "topology_neighbor_and_drive_events"
            generator_params = dict(task_axes["generator_params"])
            generator_params["topology"] = topology
            task_axes["generator_params"] = generator_params
            task_tags = [
                tag
                for tag in task_tags
                if not tag.startswith(("family:", "graph:", "events:"))
            ]
            task_tags.append(f"family:grid_{self._task.dynamics_type}")
            task_tags.append(f"graph:{graph_types[topology]}")
            task_tags.append("events:topology_neighbor_and_drive_events")
        return DataSpec(
            name=dataset_name,
            num_nodes=int(self.cfg.num_nodes),
            event_dim=int(self._task.event_dim),
            num_events=sum(batch.num_events for batch in self._bins_all),
            num_bins=len(self._bins_all),
            extra={
                "synthetic_task": self._task.name,
                "focus": self._task.focus,
                "description": self._task.description,
                "task_axes": task_axes,
                "task_tags": task_tags,
                "recommended_pairs": [f"{agg}/{upd}" for agg, upd in self._task.recommended_pairs],
                "metric_family": self._task.metric_family,
                "generator_params": task_axes["generator_params"],
                "supported_metrics": list(self._task.supported_metrics),
                "primary_metric": {
                    "path": self._task.primary_metric_path,
                    "goal": self._task.primary_metric_goal,
                },
                "summary_metrics": list(self._task.summary_metric_paths),
                "requires_node_scorer": self._task.requires_node_scorer,
                "num_episodes": int(self.cfg.num_episodes) if self.cfg.task in {"diffusion", "wave", "coupled_oscillator"} else None,
            },
        )

    def hidden_truth(self) -> Optional[dict[str, Any]]:
        """Synthetic evaluation truth; never consumed by a model or trainer."""
        return self._hidden_truth

    def bins(self, split: str = "train") -> Iterable[EventBatch]:
        start, end = self._split_bins[split]
        return _SyntheticEventStream(self._bins_all, start, end)

    def node_targets(self, split: str = "train") -> Optional[Iterable[torch.Tensor]]:
        if self._node_targets_all is None:
            return None
        start, end = self._split_bins[split]
        return _SyntheticNodeTargetStream(self._node_targets_all, start, end)

    def edge_targets(self, split: str = "train") -> Optional[Iterable[EdgeTargetBatch]]:
        if self._edge_targets_all is None:
            return None
        start, end = self._split_bins[split]
        return _SyntheticEdgeTargetStream(self._edge_targets_all, start, end)


@dataclass
class _SyntheticEventStream(Iterable[EventBatch]):
    batches: Sequence[EventBatch]
    start: int
    end: int

    def __iter__(self) -> Iterator[EventBatch]:
        if self.end < self.start:
            return
        for idx in range(self.start, self.end + 1):
            yield self.batches[idx]


@dataclass
class _SyntheticNodeTargetStream(Iterable[torch.Tensor]):
    batches: Sequence[torch.Tensor]
    start: int
    end: int

    def __iter__(self) -> Iterator[torch.Tensor]:
        if self.end < self.start:
            return
        for idx in range(self.start, self.end + 1):
            yield self.batches[idx]


@dataclass
class _SyntheticEdgeTargetStream(Iterable[EdgeTargetBatch]):
    batches: Sequence[EdgeTargetBatch]
    start: int
    end: int

    def __iter__(self) -> Iterator[EdgeTargetBatch]:
        if self.end < self.start:
            return
        for idx in range(self.start, self.end + 1):
            yield self.batches[idx]
