from __future__ import annotations

import math

import torch

from interactiondynamics.eval.node_metrics import edge_prediction_metrics, node_prediction_metrics
from interactiondynamics.training.task_metrics import (
    is_better_metric,
    parse_task_metric_spec,
    snapshot_metric_value,
)


def test_node_prediction_metrics_include_auc_and_f1():
    logits = torch.tensor([-2.0, -0.2, 0.3, 2.5], dtype=torch.float32)
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)

    metrics = node_prediction_metrics(logits, labels)

    assert math.isfinite(metrics["node_f1"])
    assert math.isfinite(metrics["node_auroc"])
    assert math.isfinite(metrics["node_auprc"])
    assert 0.0 <= metrics["node_auroc"] <= 1.0


def test_edge_prediction_metrics_include_auc_and_f1():
    logits = torch.tensor([-2.0, -0.2, 0.3, 2.5], dtype=torch.float32)
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)

    metrics = edge_prediction_metrics(logits, labels)

    assert math.isfinite(metrics["edge_f1"])
    assert math.isfinite(metrics["edge_auroc"])
    assert math.isfinite(metrics["edge_auprc"])
    assert 0.0 <= metrics["edge_auroc"] <= 1.0


def test_task_metric_helpers_handle_nested_snapshot_paths():
    spec = parse_task_metric_spec({"path": "rollout_val.rollout_edge_r2", "goal": "max"})
    assert spec is not None
    snapshot = {"rollout_val": {"rollout_edge_r2": 0.87}}
    assert snapshot_metric_value(snapshot, spec.path) == 0.87
    assert is_better_metric(0.87, 0.70, spec.goal)
