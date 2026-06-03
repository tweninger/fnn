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
    "ift_diffusion": SyntheticTaskSpec(
        name="ift_diffusion",
        description="Predict graph diffusion over a ring with per-node drives carried through edge events.",
        focus="Topology-aware diffusion and smoothing over repeated interaction structure.",
        event_dim=2,
        recommended_pairs=(
            IFT_PAIR,
            ("sum", "lnn"),
            ("sum", "tgn_gru"),
        ),
        metric_family="edge_regression",
        generator_family="ring_diffusion",
        graph_type="ring",
        dynamics_type="diffusion",
        event_structure="ring_neighbor_and_self_events",
        temporal_mode="rollout",
        feature_schema=("signal", "is_drive"),
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
        self.cfg = cfg
        self._task = SYNTHETIC_TASKS[cfg.task]
        self._rng = np.random.default_rng(int(cfg.seed))
        self._bins_all, self._node_targets_all, self._edge_targets_all = self._materialize()
        self._split_bins = self._compute_splits(len(self._bins_all))

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
        if self.cfg.task == "ift_diffusion":
            bins, edge_targets = self._materialize_ift_diffusion()
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

    def _materialize_ift_diffusion(self) -> tuple[list[EventBatch], list[EdgeTargetBatch]]:
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
    ) -> EventBatch:
        batch = EventBatch(
            src=cast(torch.LongTensor, torch.from_numpy(src.astype(np.int64, copy=False))),
            dst=cast(torch.LongTensor, torch.from_numpy(dst.astype(np.int64, copy=False))),
            features=None if features is None else torch.from_numpy(features.astype(np.float32, copy=False)),
            t=cast(torch.LongTensor, torch.full((src.size,), int(t), dtype=torch.long)),
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

    def spec(self) -> DataSpec:
        dataset_name = self.cfg.name or f"synthetic_{self.cfg.task}"
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
                "task_axes": self._task.axes(),
                "task_tags": list(self._task.tags()),
                "recommended_pairs": [f"{agg}/{upd}" for agg, upd in self._task.recommended_pairs],
                "metric_family": self._task.metric_family,
                "supported_metrics": list(self._task.supported_metrics),
                "primary_metric": {
                    "path": self._task.primary_metric_path,
                    "goal": self._task.primary_metric_goal,
                },
                "summary_metrics": list(self._task.summary_metric_paths),
                "requires_node_scorer": self._task.requires_node_scorer,
            },
        )

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
