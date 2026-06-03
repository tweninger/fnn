# eval/evaluate.py

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Dict, Iterable, Optional
import torch

from interactiondynamics.core.events import EventBatch
from interactiondynamics.data.interfaces import EdgeTargetBatch
from interactiondynamics.eval.node_metrics import (
    edge_regression_metrics,
    node_labels_from_events,
    node_prediction_metrics,
    node_regression_metrics,
)
from interactiondynamics.eval.ranking_metrics import ranking_loss_and_metrics


@dataclass
class EvalSlices:
    early_steps: int = 10   # first N scored steps
    # late = everything after early_steps


def _acc_init() -> dict:
    return {"loss_sum": 0.0, "metric_sums": {}, "metric_counts": {}, "n": 0.0}


def _acc_update(acc: dict, loss: float, metrics: Dict[str, float]) -> None:
    acc["loss_sum"] += float(loss)
    for key, value in metrics.items():
        value_f = float(value)
        if math.isnan(value_f):
            continue
        acc["metric_sums"][key] = acc["metric_sums"].get(key, 0.0) + value_f
        acc["metric_counts"][key] = acc["metric_counts"].get(key, 0.0) + 1.0
    acc["n"] += 1.0


def _acc_finalize(acc: dict) -> Dict[str, float]:
    if acc["n"] <= 0:
        return {"loss": 0.0, "steps": 0}
    n = acc["n"]
    out = {"loss": acc["loss_sum"] / n, "steps": int(n)}
    for key, value in acc["metric_sums"].items():
        out[key] = value / acc["metric_counts"][key]
    return out


def _add_skill_vs_persistent(
    out: Dict[str, float],
    *,
    model_prefix: str,
    persistent_prefix: str,
) -> None:
    model_key = f"{model_prefix}edge_mse"
    persistent_key = f"{persistent_prefix}edge_mse"
    if model_key not in out or persistent_key not in out:
        return
    persistent_mse = float(out[persistent_key])
    model_mse = float(out[model_key])
    if persistent_mse <= 0.0 or math.isnan(persistent_mse) or math.isnan(model_mse):
        out[f"{model_prefix}edge_skill_vs_persistent"] = float("nan")
        return
    out[f"{model_prefix}edge_skill_vs_persistent"] = 1.0 - (model_mse / persistent_mse)


@torch.no_grad()
def evaluate_stream_sliced(
    model,
    bins: Iterable[EventBatch],
    node_targets: Optional[Iterable[torch.Tensor]],
    edge_targets: Optional[Iterable[EdgeTargetBatch]],
    cfg,
    *,
    slices: EvalSlices = EvalSlices(),
) -> Dict[str, float]:
    """
    Evaluate ranking metrics over a binned stream, returning:
      - overall loss/mrr
      - early (first N scored steps)
      - late (remaining scored steps)

    No warmup. State is initialized fresh.
    """
    model.eval()
    device = torch.device(cfg.device)

    state = model.init_state(batch_size=1, num_nodes=cfg.num_nodes, device=device)

    overall = _acc_init()
    early = _acc_init()
    late = _acc_init()
    persistent_overall = _acc_init()
    persistent_early = _acc_init()
    persistent_late = _acc_init()

    prev: Optional[EventBatch] = None
    scored_step = 0  # counts only steps where we actually compute loss (prev exists)
    persistent_state = None
    target_iter = iter(node_targets) if node_targets is not None else None
    edge_target_iter = iter(edge_targets) if edge_targets is not None else None

    for events in bins:
        events = events.to(device)
        curr_node_target = None if target_iter is None else next(target_iter).to(device)
        curr_edge_target = None if edge_target_iter is None else next(edge_target_iter)

        if prev is None:
            prev = events
            continue

        # predict current from state(after consuming prev)
        state, _ = model.step(state, prev)

        if curr_edge_target is not None:
            edge_events = curr_edge_target.events.to(device)
            edge_target_values = curr_edge_target.targets.to(device)
            edge_preds = model.score(state, edge_events)
            loss_t = torch.nn.functional.mse_loss(edge_preds, edge_target_values)
            metrics = edge_regression_metrics(edge_preds.detach(), edge_target_values)
        else:
            loss_t, metrics = ranking_loss_and_metrics(
                model=model,
                state=state,
                next_events=events,
                num_nodes=cfg.num_nodes,
                num_neg=cfg.num_neg,
            )
        total_loss_t = loss_t
        if getattr(model, "node_scorer", None) is not None:
            node_logits = model.score_nodes(state)
            if curr_node_target is not None:
                node_target = curr_node_target.to(node_logits.device, dtype=node_logits.dtype)
                node_loss_t = torch.nn.functional.mse_loss(node_logits, node_target)
                metrics.update(node_regression_metrics(node_logits.detach(), node_target))
            else:
                node_labels = node_labels_from_events(events, cfg.num_nodes, device=node_logits.device)
                node_loss_t = torch.nn.functional.binary_cross_entropy_with_logits(node_logits, node_labels)
                metrics.update(node_prediction_metrics(node_logits.detach(), node_labels))
            metrics["node_loss"] = float(node_loss_t.detach().item())
            total_loss_t = total_loss_t + (cfg.node_loss_weight * node_loss_t)

        if persistent_state is None:
            persistent_state = state.clone(detach=True) if state is not None else None

        if curr_edge_target is not None:
            persistent_edge_events = curr_edge_target.events.to(device)
            persistent_edge_targets = curr_edge_target.targets.to(device)
            persistent_edge_preds = model.score(persistent_state, persistent_edge_events)
            persistent_loss_t = torch.nn.functional.mse_loss(
                persistent_edge_preds,
                persistent_edge_targets,
            )
            persistent_metrics = edge_regression_metrics(
                persistent_edge_preds.detach(),
                persistent_edge_targets,
            )
        else:
            persistent_loss_t, persistent_metrics = ranking_loss_and_metrics(
                model=model,
                state=persistent_state,
                next_events=events,
                num_nodes=cfg.num_nodes,
                num_neg=cfg.num_neg,
            )
        persistent_total_loss_t = persistent_loss_t
        if getattr(model, "node_scorer", None) is not None:
            persistent_node_logits = model.score_nodes(persistent_state)
            if curr_node_target is not None:
                persistent_node_target = curr_node_target.to(
                    persistent_node_logits.device, dtype=persistent_node_logits.dtype
                )
                persistent_node_loss_t = torch.nn.functional.mse_loss(
                    persistent_node_logits, persistent_node_target
                )
                persistent_metrics.update(
                    node_regression_metrics(persistent_node_logits.detach(), persistent_node_target)
                )
            else:
                persistent_node_labels = node_labels_from_events(
                    events, cfg.num_nodes, device=persistent_node_logits.device
                )
                persistent_node_loss_t = torch.nn.functional.binary_cross_entropy_with_logits(
                    persistent_node_logits, persistent_node_labels
                )
                persistent_metrics.update(
                    node_prediction_metrics(persistent_node_logits.detach(), persistent_node_labels)
                )
            persistent_metrics["node_loss"] = float(persistent_node_loss_t.detach().item())
            persistent_total_loss_t = persistent_total_loss_t + (cfg.node_loss_weight * persistent_node_loss_t)

        if state is not None:
            state.detach_()

        loss_val = float(total_loss_t.item())
        persistent_loss_val = float(persistent_total_loss_t.item())
        _acc_update(overall, loss_val, metrics)
        _acc_update(persistent_overall, persistent_loss_val, persistent_metrics)
        if scored_step < slices.early_steps:
            _acc_update(early, loss_val, metrics)
            _acc_update(persistent_early, persistent_loss_val, persistent_metrics)
        else:
            _acc_update(late, loss_val, metrics)
            _acc_update(persistent_late, persistent_loss_val, persistent_metrics)

        scored_step += 1
        prev = events

    out: Dict[str, float] = {}
    o = _acc_finalize(overall)
    e = _acc_finalize(early)
    l = _acc_finalize(late)
    po = _acc_finalize(persistent_overall)
    pe = _acc_finalize(persistent_early)
    pl = _acc_finalize(persistent_late)

    for key, value in o.items():
        out[key] = float(value)

    for prefix, block in (("early", e), ("late", l)):
        for key, value in block.items():
            out[f"{prefix}_{key}"] = float(value)

    for prefix, block in (
        ("persistent", po),
        ("persistent_early", pe),
        ("persistent_late", pl),
    ):
        for key, value in block.items():
            out[f"{prefix}_{key}"] = float(value)

    _add_skill_vs_persistent(out, model_prefix="", persistent_prefix="persistent_")
    _add_skill_vs_persistent(out, model_prefix="early_", persistent_prefix="persistent_early_")
    _add_skill_vs_persistent(out, model_prefix="late_", persistent_prefix="persistent_late_")

    return out
