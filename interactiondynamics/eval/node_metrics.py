from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from interactiondynamics.core.events import EventBatch
from interactiondynamics.eval.prediction_metrics import binary_metrics_from_logits


def node_labels_from_events(
    events: EventBatch,
    num_nodes: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    labels = torch.zeros((num_nodes,), device=device, dtype=torch.float32)
    labels[events.src.to(device=device, dtype=torch.long)] = 1.0
    labels[events.dst.to(device=device, dtype=torch.long)] = 1.0
    return labels


@torch.no_grad()
def regression_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    *,
    prefix: str,
) -> Dict[str, float]:
    mse = F.mse_loss(preds, targets).item()
    mae = F.l1_loss(preds, targets).item()
    rmse = mse ** 0.5
    pred_mean = preds.mean().item()
    pred_std = preds.std(unbiased=False).item() if preds.numel() > 0 else 0.0
    target_mean = targets.mean().item()
    target_std = targets.std(unbiased=False).item() if targets.numel() > 0 else 0.0
    nrmse = rmse / max(target_std, 1e-12)
    zmse = mse / max(target_std * target_std, 1e-12)
    sse = ((preds - targets) ** 2).sum().item()
    sst = ((targets - targets.mean()) ** 2).sum().item()
    r2 = 1.0 - (sse / sst) if sst > 1e-12 else 0.0
    if preds.numel() > 1:
        vx = preds - preds.mean()
        vy = targets - targets.mean()
        denom = (vx.norm() * vy.norm()).item()
        corr = float((vx * vy).sum().item() / denom) if denom > 0 else 0.0
    else:
        corr = 0.0
    return {
        f"{prefix}_mse": mse,
        f"{prefix}_mae": mae,
        f"{prefix}_rmse": rmse,
        f"{prefix}_nrmse": nrmse,
        f"{prefix}_zmse": zmse,
        f"{prefix}_r2": r2,
        f"{prefix}_corr": corr,
        f"{prefix}_pred_mean": pred_mean,
        f"{prefix}_pred_std": pred_std,
        f"{prefix}_target_mean": target_mean,
        f"{prefix}_target_std": target_std,
    }


@torch.no_grad()
def node_prediction_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, float]:
    binary = binary_metrics_from_logits(logits, labels)

    return {
        "node_bce": F.binary_cross_entropy_with_logits(logits, labels).item(),
        "node_acc": binary["accuracy"],
        "node_precision": binary["precision"],
        "node_recall": binary["recall"],
        "node_specificity": binary["specificity"],
        "node_f1": binary["f1"],
        "node_auroc": binary["auroc"],
        "node_auprc": binary["auprc"],
        "node_balanced_acc": binary["balanced_acc"],
        "node_fpr": binary["false_positive_rate"],
        "node_pred_rate": binary["pred_positive_rate"],
        "node_pos_rate": binary["true_positive_rate"],
    }


@torch.no_grad()
def edge_prediction_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, float]:
    binary = binary_metrics_from_logits(logits, labels)

    return {
        "edge_bce": F.binary_cross_entropy_with_logits(logits, labels).item(),
        "edge_acc": binary["accuracy"],
        "edge_precision": binary["precision"],
        "edge_recall": binary["recall"],
        "edge_specificity": binary["specificity"],
        "edge_f1": binary["f1"],
        "edge_auroc": binary["auroc"],
        "edge_auprc": binary["auprc"],
        "edge_balanced_acc": binary["balanced_acc"],
        "edge_fpr": binary["false_positive_rate"],
        "edge_pred_rate": binary["pred_positive_rate"],
        "edge_pos_rate": binary["true_positive_rate"],
    }


@torch.no_grad()
def node_regression_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    return regression_metrics(preds, targets, prefix="node")


@torch.no_grad()
def edge_regression_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    return regression_metrics(preds, targets, prefix="edge")
