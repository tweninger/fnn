from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, Optional

import torch

from interactiondynamics.data.interfaces import EdgeTargetBatch


def edge_regression_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    cfg: Any,
) -> torch.Tensor:
    if getattr(cfg, "edge_target_scale", "raw") == "zscore":
        target_mean = float(getattr(cfg, "edge_target_mean", 0.0))
        target_std = max(float(getattr(cfg, "edge_target_std", 1.0)), 1e-12)
        preds = (preds - target_mean) / target_std
        targets = (targets - target_mean) / target_std
    return torch.nn.functional.mse_loss(preds, targets)


def transform_edge_targets(
    curr_targets: torch.Tensor,
    prev_targets: Optional[torch.Tensor],
    cfg: Any,
) -> torch.Tensor:
    if getattr(cfg, "edge_target_mode", "raw") == "residual":
        if prev_targets is None:
            raise ValueError("Residual edge-target mode requires previous edge targets.")
        return curr_targets - prev_targets.to(device=curr_targets.device, dtype=curr_targets.dtype)
    return curr_targets


def reconstruct_raw_edge_predictions(
    preds: torch.Tensor,
    prev_targets: Optional[torch.Tensor],
    cfg: Any,
) -> torch.Tensor:
    if getattr(cfg, "edge_target_mode", "raw") == "residual":
        if prev_targets is None:
            raise ValueError("Residual edge-target mode requires previous edge targets.")
        return preds + prev_targets.to(device=preds.device, dtype=preds.dtype)
    return preds


def transform_node_targets(
    curr_targets: torch.Tensor,
    prev_targets: Optional[torch.Tensor],
    cfg: Any,
) -> torch.Tensor:
    if getattr(cfg, "node_target_mode", "raw") == "residual":
        if prev_targets is None:
            raise ValueError("Residual node-target mode requires previous node targets.")
        return curr_targets - prev_targets.to(device=curr_targets.device, dtype=curr_targets.dtype)
    return curr_targets


def reconstruct_raw_node_predictions(
    preds: torch.Tensor,
    prev_targets: Optional[torch.Tensor],
    cfg: Any,
) -> torch.Tensor:
    if getattr(cfg, "node_target_mode", "raw") == "residual":
        if prev_targets is None:
            raise ValueError("Residual node-target mode requires previous node targets.")
        return preds + prev_targets.to(device=preds.device, dtype=preds.dtype)
    return preds


def _summarize_value_stream(
    batches: Iterable[Any],
    *,
    extract_values: Callable[[Any], torch.Tensor],
    mode: str,
) -> Optional[Dict[str, float]]:
    count = 0
    total = 0.0
    total_sq = 0.0
    min_value = float("inf")
    max_value = float("-inf")
    prev_targets: Optional[torch.Tensor] = None

    for batch in batches:
        curr_targets = extract_values(batch).detach().to(device="cpu", dtype=torch.float64).view(-1)
        if mode == "residual":
            if prev_targets is None:
                prev_targets = curr_targets
                continue
            values = curr_targets - prev_targets
            prev_targets = curr_targets
        else:
            values = curr_targets
        if values.numel() == 0:
            continue
        count += int(values.numel())
        total += float(values.sum().item())
        total_sq += float((values * values).sum().item())
        min_value = min(min_value, float(values.min().item()))
        max_value = max(max_value, float(values.max().item()))

    if count == 0:
        return None

    mean = total / count
    var = max(0.0, (total_sq / count) - (mean * mean))
    return {
        "count": float(count),
        "mean": mean,
        "std": math.sqrt(var),
        "min": min_value,
        "max": max_value,
    }


def summarize_targets(
    targets: Optional[Iterable[torch.Tensor]],
    *,
    mode: str = "raw",
) -> Optional[Dict[str, float]]:
    if targets is None:
        return None
    return _summarize_value_stream(targets, extract_values=lambda batch: batch, mode=mode)


def summarize_edge_targets(
    edge_targets: Optional[Iterable[EdgeTargetBatch]],
    *,
    mode: str = "raw",
) -> Optional[Dict[str, float]]:
    if edge_targets is None:
        return None
    return _summarize_value_stream(
        edge_targets,
        extract_values=lambda batch: batch.targets,
        mode=mode,
    )


def global_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total_sq = 0.0
    found = False
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        total_sq += float((grad * grad).sum().item())
        found = True
    if not found:
        return 0.0
    return math.sqrt(total_sq)
