# eval/node_regression.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr

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


def _iter_context_target_bins(
    context_bins: Iterable[EventBatch],
    target_bins: Optional[Iterable[EventBatch]] = None,
) -> Iterator[tuple[EventBatch, EventBatch]]:
    if target_bins is None:
        for batch in context_bins:
            yield batch, batch
        return
    for ctx, tgt in zip(context_bins, target_bins):
        yield ctx, tgt


@torch.no_grad()
def evaluate_node_stream_sliced(
    model,
    bins: Iterable[EventBatch],
    cfg,
    slices: Optional[NodeEvalSlices] = None,
    target_bins: Optional[Iterable[EventBatch]] = None,
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

    y_true_chunks = []
    y_pred_chunks = []
    mask_chunks = []
    saw_mask = False

    prev_obs: Optional[EventBatch] = None
    for curr_obs, curr_target in _iter_context_target_bins(bins, target_bins):
        curr_obs = curr_obs.to(device)
        curr_target = curr_target.to(device)

        if prev_obs is None:
            prev_obs = curr_obs
            continue

        state, _aux = model.step(state, prev_obs)

        loss, metrics = node_regression_loss_and_metrics(
            model=model,
            state=state,
            next_events=curr_target,
            loss_name=cfg.loss_name,
        )
        pred = model.predict_nodes(state.clone(detach=True))
        true = curr_target.node_targets
        if true is not None:
            y_true_chunks.append(true.detach().cpu().numpy())
            y_pred_chunks.append(pred.detach().cpu().numpy())
            if curr_target.node_mask is not None:
                saw_mask = True
                mask_chunks.append(curr_target.node_mask.detach().cpu().numpy().astype(bool))

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

        prev_obs = curr_obs

    out = {
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

    if y_true_chunks:
        y_true_np = np.stack(y_true_chunks, axis=0)
        y_pred_np = np.stack(y_pred_chunks, axis=0)
        node_mask_np = None
        if saw_mask:
            node_mask_np = np.stack(
                [
                    m if m is not None else np.ones((cfg.num_nodes,), dtype=bool)
                    for m in mask_chunks
                ],
                axis=0,
            )
        out.update(compute_prediction_analysis(y_true_np, y_pred_np, node_mask_np))

    return out

@torch.no_grad()
def collect_node_predictions_over_time(model, bins, cfg, target_bins=None):
    """
    Collect one-step-ahead node predictions across a stream.

    Assumes:
      - each curr EventBatch has curr.node_targets of shape [N, d_y]
      - model.predict_nodes(state) returns [N, d_y]
      - curr.t contains the bin timestamp for events in that bin

    Returns
    -------
    times : np.ndarray [T_scored]
    y_true : np.ndarray [T_scored, N, d_y]
    y_pred : np.ndarray [T_scored, N, d_y]
    node_mask : np.ndarray [T_scored, N] or None
        Returned only if masks are present in the stream.
    """
    model.eval()
    device = torch.device(cfg.device)

    state = model.init_state(batch_size=1, num_nodes=cfg.num_nodes, device=device)

    times = []
    y_true = []
    y_pred = []
    masks = []

    prev_obs = None
    saw_mask = False

    for curr_obs, curr_target in _iter_context_target_bins(bins, target_bins):
        curr_obs = curr_obs.to(device)
        curr_target = curr_target.to(device)

        if prev_obs is None:
            prev_obs = curr_obs
            continue

        # update memory with previous bin
        state, _aux = model.step(state, prev_obs)

        # optional safety: predict on a detached clone so prediction cannot
        # accidentally mutate the live recurrent state
        state_eval = state.clone(detach=True) if state is not None else None

        pred = model.predict_nodes(state_eval)  # [N, d_y]
        true = curr_target.node_targets

        if true is None:
            raise ValueError("curr_target.node_targets is None; dataset must provide node targets")

        true = true.to(pred.device, pred.dtype)

        # record one scalar time for this bin
        if curr_target.t is not None:
            t_min = int(curr_target.t.min().item())
            t_max = int(curr_target.t.max().item())
            if t_min != t_max:
                raise ValueError(f"Expected one timestamp per bin, got [{t_min}, {t_max}]")
            t_val = t_min
        else:
            t_val = len(times)

        times.append(t_val)
        y_true.append(true.detach().cpu().numpy())
        y_pred.append(pred.detach().cpu().numpy())

        if curr_target.node_mask is not None:
            saw_mask = True
            masks.append(curr_target.node_mask.detach().cpu().numpy().astype(bool))
        else:
            masks.append(None)

        prev_obs = curr_obs

    if len(times) == 0:
        raise ValueError("No scored steps were collected. Need at least two bins in the stream.")

    times_np = np.asarray(times)
    y_true_np = np.stack(y_true, axis=0)   # [T, N, d_y]
    y_pred_np = np.stack(y_pred, axis=0)   # [T, N, d_y]

    if saw_mask:
        node_mask_np = np.stack(
            [m if m is not None else np.ones((cfg.num_nodes,), dtype=bool) for m in masks],
            axis=0,
        )  # [T, N]

        # debugging
    # times = np.asarray(times)
    # y_true = np.asarray(y_true)
    # y_pred = np.asarray(y_pred)

    # print("times shape:", times.shape)
    # print("y_true shape:", y_true.shape)
    # print("y_pred shape:", y_pred.shape)
    # print("first 5 times:", times[:5])
    # print("first 5 pred node0:", y_pred[:5, 0, :])
    # print("first 5 true node0:", y_true[:5, 0, :])
        return times_np, y_true_np, y_pred_np, node_mask_np

    return times_np, y_true_np, y_pred_np

def compute_prediction_analysis(y_true, y_pred, node_mask=None):
    """
    y_true, y_pred: [T, N, D]
    node_mask: [T, N] or None
    """
    eps = 1e-12

    if node_mask is None:
        yt = y_true.reshape(-1, y_true.shape[-1])
        yp = y_pred.reshape(-1, y_pred.shape[-1])
        mask = np.ones((y_true.shape[0], y_true.shape[1]), dtype=bool)
    else:
        mask = node_mask.astype(bool)
        yt = y_true[mask]   # [M, D]
        yp = y_pred[mask]   # [M, D]

    diff = yp - yt
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))

    yt_mean = np.mean(yt, axis=0, keepdims=True)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt_mean) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > eps else float("nan")

    # per-node trajectory metrics over time
    T, N, D = y_true.shape
    node_pearsons = []
    node_spearmans = []
    node_r2s = []

    for n in range(N):
        valid_t = mask[:, n]
        if valid_t.sum() < 2:
            continue

        for d in range(D):
            yt_nd = y_true[valid_t, n, d]
            yp_nd = y_pred[valid_t, n, d]

            if yt_nd.size < 2:
                continue

            # Pearson
            if np.std(yt_nd) > eps and np.std(yp_nd) > eps:
                r = np.corrcoef(yt_nd, yp_nd)[0, 1]
                if np.isfinite(r):
                    node_pearsons.append(float(r))

                rho = spearmanr(yt_nd, yp_nd).statistic
                if np.isfinite(rho):
                    node_spearmans.append(float(rho))

            # per-node R²
            ss_res_nd = float(np.sum((yt_nd - yp_nd) ** 2))
            ss_tot_nd = float(np.sum((yt_nd - np.mean(yt_nd)) ** 2))
            if ss_tot_nd > eps:
                node_r2s.append(float(1.0 - ss_res_nd / ss_tot_nd))

    out = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mean_node_pearson": float(np.mean(node_pearsons)) if node_pearsons else float("nan"),
        "mean_node_spearman": float(np.mean(node_spearmans)) if node_spearmans else float("nan"),
        "mean_node_r2": float(np.mean(node_r2s)) if node_r2s else float("nan"),
        "median_node_pearson": float(np.median(node_pearsons)) if node_pearsons else float("nan"),
        "median_node_spearman": float(np.median(node_spearmans)) if node_spearmans else float("nan"),
        "median_node_r2": float(np.median(node_r2s)) if node_r2s else float("nan"),
    }

    # vector metrics for D > 1
    if yt.shape[1] > 1:
        yt_norm = np.linalg.norm(yt, axis=1)
        yp_norm = np.linalg.norm(yp, axis=1)
        valid = (yt_norm > eps) & (yp_norm > eps)

        if np.any(valid):
            cos = np.sum(yt[valid] * yp[valid], axis=1) / (yt_norm[valid] * yp_norm[valid])
            cos = np.clip(cos, -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos))

            out["mean_cosine"] = float(np.mean(cos))
            out["mean_angle_deg"] = float(np.mean(angle_deg))
            out["magnitude_rmse"] = float(np.sqrt(np.mean((yp_norm - yt_norm) ** 2)))
        else:
            out["mean_cosine"] = float("nan")
            out["mean_angle_deg"] = float("nan")
            out["magnitude_rmse"] = float("nan")

    return out