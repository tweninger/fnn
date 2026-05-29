from __future__ import annotations

from typing import Dict

import torch


def _safe_div(num: float, den: float) -> float:
    return num / den if den > 0.0 else 0.0


@torch.no_grad()
def binary_metrics_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    threshold: float = 0.0,
) -> Dict[str, float]:
    """
    Compute thresholded binary prediction metrics from logits.

    The default threshold is 0.0, matching the usual BCE-with-logits convention:
    logits >= 0 correspond to predicted probabilities >= 0.5.
    """
    logits = logits.view(-1)
    labels = labels.to(device=logits.device, dtype=torch.float32).view(-1)
    pred = (logits >= float(threshold)).to(dtype=torch.float32)

    tp = float(((pred == 1.0) & (labels == 1.0)).sum().item())
    fp = float(((pred == 1.0) & (labels == 0.0)).sum().item())
    fn = float(((pred == 0.0) & (labels == 1.0)).sum().item())
    tn = float(((pred == 0.0) & (labels == 0.0)).sum().item())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, tp + fp + fn + tn)
    balanced_acc = 0.5 * (recall + specificity)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_acc": balanced_acc,
        "false_positive_rate": _safe_div(fp, fp + tn),
        "pred_positive_rate": pred.mean().item(),
        "true_positive_rate": labels.mean().item(),
    }
