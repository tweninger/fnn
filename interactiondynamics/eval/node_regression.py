# eval/node_regression.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F

from core.events import EventBatch


# -----------------------------
# helpers
# -----------------------------

def _apply_mask(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mask is None:
        return pred, target

    mask = mask.bool()
    return pred[mask], target[mask]


def _safe_mean(total: float, n: int) -> float:
    return total / n if n > 0 else 0.0


# -----------------------------
# per-step metrics / loss
# -----------------------------

@torch.no_grad()
def node_regression_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> Dict[str, float]:
    pred, target = _apply_mask(pred, target, mask)

    if pred.numel() == 0:
        return {"mse": 0.0, "mae": 0.0, "rmse": 0.0}

    diff = pred - target
    mse = diff.pow(2).mean()
    mae = diff.abs().mean()
    rmse = torch.sqrt(mse)

    return {
        "mse": float(mse.item()),
        "mae": float(mae.item()),
        "rmse": float(rmse.item()),
    }


def node_regression_loss_and_metrics(
    model,
    state,
    next_events: EventBatch,
    loss_name: str = "huber",   # "mse" | "mae" | "huber"
) -> Tuple[torch.Tensor, Dict[str, float]]:
    assert next_events.node_targets is not None, \
        "next_events.node_targets is required for node-level regression"

    assert state is not None and state.node is not None, \
        "node regression requires a non-empty latent node state"

    before = state.node.detach().clone()
    detach_for_pred = not torch.is_grad_enabled()
    state_eval = state.clone(detach=detach_for_pred)

    pred = model.predict_nodes(state_eval)  # [N, d_y]
    target = next_events.node_targets.to(pred.device, pred.dtype)
    mask = next_events.node_mask
    if mask is not None:
        mask = mask.to(pred.device)

    # safety: prediction should not mutate the real state
    assert torch.equal(before, state.node.detach()), \
        "predict_nodes() mutated state.node"

    pred_used, target_used = _apply_mask(pred, target, mask)

    if pred_used.numel() == 0:
        raise RuntimeError("All nodes were masked out in node_regression_loss_and_metrics")

    if loss_name == "mse":
        loss = F.mse_loss(pred_used, target_used)
    elif loss_name == "mae":
        loss = F.l1_loss(pred_used, target_used)
    elif loss_name == "huber":
        loss = F.smooth_l1_loss(pred_used, target_used)
    else:
        raise ValueError(f"unknown loss_name={loss_name}")

    metrics = node_regression_metrics(
        pred=pred.detach(),
        target=target.detach(),
        mask=mask,
    )
    return loss, metrics


# -----------------------------
# sliced eval
# -----------------------------

@dataclass(frozen=True)
class NodeEvalSlices:
    early_steps: int = 10


@torch.no_grad()
def evaluate_node_stream_sliced(
    model,
    bins: Iterable[EventBatch],
    cfg,
    slices: Optional[NodeEvalSlices] = None,
) -> Dict[str, float]:
    """
    Evaluate one-step-ahead node prediction over a split.

    For consecutive bins (prev, curr):
      1) update state with prev
      2) predict node targets for curr
      3) accumulate overall metrics
      4) also split into early vs late scored steps

    Returns a dict like:
      {
        "loss", "mae", "rmse", "mse", "steps",
        "early_loss", "early_mae", "early_rmse", "early_mse", "early_steps",
        "late_loss", "late_mae", "late_rmse", "late_mse", "late_steps",
      }
    """
    model.eval()
    device = torch.device(cfg.device)

    if slices is None:
        slices = NodeEvalSlices(early_steps=10)

    state = model.init_state(batch_size=1, num_nodes=cfg.num_nodes, device=device)

    # overall
    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_mse = 0.0
    n_steps = 0

    # early
    early_loss = 0.0
    early_mae = 0.0
    early_rmse = 0.0
    early_mse = 0.0
    n_early = 0

    # late
    late_loss = 0.0
    late_mae = 0.0
    late_rmse = 0.0
    late_mse = 0.0
    n_late = 0

    prev: Optional[EventBatch] = None
    for curr in bins:
        curr = curr.to(device)

        if prev is None:
            prev = curr
            continue

        state, _aux = model.step(state, prev)

        loss, metrics = node_regression_loss_and_metrics(
            model=model,
            state=state,
            next_events=curr,
            loss_name=cfg.loss_name,
        )

        # overall
        total_loss += float(loss.item())
        total_mae += float(metrics["mae"])
        total_rmse += float(metrics["rmse"])
        total_mse += float(metrics["mse"])
        n_steps += 1

        # sliced
        if n_steps <= slices.early_steps:
            early_loss += float(loss.item())
            early_mae += float(metrics["mae"])
            early_rmse += float(metrics["rmse"])
            early_mse += float(metrics["mse"])
            n_early += 1
        else:
            late_loss += float(loss.item())
            late_mae += float(metrics["mae"])
            late_rmse += float(metrics["rmse"])
            late_mse += float(metrics["mse"])
            n_late += 1

        prev = curr

    return {
        "loss": _safe_mean(total_loss, n_steps),
        "mae": _safe_mean(total_mae, n_steps),
        "rmse": _safe_mean(total_rmse, n_steps),
        "mse": _safe_mean(total_mse, n_steps),
        "steps": float(n_steps),

        "early_loss": _safe_mean(early_loss, n_early),
        "early_mae": _safe_mean(early_mae, n_early),
        "early_rmse": _safe_mean(early_rmse, n_early),
        "early_mse": _safe_mean(early_mse, n_early),
        "early_steps": float(n_early),

        "late_loss": _safe_mean(late_loss, n_late),
        "late_mae": _safe_mean(late_mae, n_late),
        "late_rmse": _safe_mean(late_rmse, n_late),
        "late_mse": _safe_mean(late_mse, n_late),
        "late_steps": float(n_late),
    }