# eval/evaluate.py

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Dict, Iterable, Optional
import torch

from interactiondynamics.core.events import EventBatch
from interactiondynamics.data.interfaces import EdgeTargetBatch
from interactiondynamics.eval.node_metrics import (
    edge_prediction_metrics,
    edge_regression_metrics,
    node_labels_from_events,
    node_prediction_metrics,
    node_regression_metrics,
)
from interactiondynamics.eval.ranking_metrics import ranking_loss_and_metrics
from interactiondynamics.training.targets import (
    edge_regression_loss as _edge_regression_loss,
    reconstruct_raw_edge_predictions as _reconstruct_raw_edge_predictions,
    reconstruct_raw_node_predictions as _reconstruct_raw_node_predictions,
    transform_edge_targets as _transform_edge_targets,
    transform_node_targets as _transform_node_targets,
)


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


@torch.no_grad()
def evaluate_k_step_rollout(
    model,
    bins: Iterable[EventBatch],
    node_targets: Optional[Iterable[torch.Tensor]],
    edge_targets: Optional[Iterable[EdgeTargetBatch]],
    cfg,
    *,
    horizon: int = 5,
) -> Dict[str, float]:
    model.eval()
    device = torch.device(cfg.device)
    horizon = max(1, int(horizon))

    events_seq = [events.to(device) for events in bins]
    node_target_seq = None if node_targets is None else [target.to(device) for target in node_targets]
    edge_target_seq = None if edge_targets is None else [
        EdgeTargetBatch(events=batch.events.to(device), targets=batch.targets.to(device))
        for batch in edge_targets
    ]

    num_steps = len(events_seq)
    if num_steps <= horizon:
        return {"rollout_horizon": horizon, "rollout_steps": 0}

    state = model.init_state(batch_size=1, num_nodes=cfg.num_nodes, device=device)
    state_after_prev: list[Optional[object]] = [None] * num_steps
    prev_node_target_seq: list[Optional[torch.Tensor]] = [None] * num_steps
    prev_edge_target_seq: list[Optional[torch.Tensor]] = [None] * num_steps

    prev_node_target: Optional[torch.Tensor] = None
    prev_edge_target: Optional[torch.Tensor] = None
    for idx in range(num_steps):
        if idx == 0:
            prev_node_target = None if node_target_seq is None else node_target_seq[idx].detach()
            prev_edge_target = None if edge_target_seq is None else edge_target_seq[idx].targets.detach()
            continue
        state, _ = model.step(state, events_seq[idx - 1])
        state_after_prev[idx] = None if state is None else state.clone(detach=True)
        prev_node_target_seq[idx] = prev_node_target
        prev_edge_target_seq[idx] = prev_edge_target
        if state is not None:
            state.detach_()
        if node_target_seq is not None:
            prev_node_target = node_target_seq[idx].detach()
        if edge_target_seq is not None:
            prev_edge_target = edge_target_seq[idx].targets.detach()

    edge_acc = _acc_init()
    edge_persistent_acc = _acc_init()
    node_acc = _acc_init()
    node_persistent_acc = _acc_init()
    rollout_steps = 0

    for start_idx in range(1, num_steps - horizon + 1):
        end_idx = start_idx + horizon - 1
        rollout_state = state_after_prev[start_idx]
        if rollout_state is None:
            continue
        rollout_prev_node = prev_node_target_seq[start_idx]
        rollout_prev_edge = prev_edge_target_seq[start_idx]
        persistent_prev_node = rollout_prev_node
        persistent_prev_edge = rollout_prev_edge
        curr_state = rollout_state

        final_edge_pred_raw: Optional[torch.Tensor] = None
        final_edge_true_raw: Optional[torch.Tensor] = None
        final_node_pred_raw: Optional[torch.Tensor] = None
        final_node_true_raw: Optional[torch.Tensor] = None

        for idx in range(start_idx, end_idx + 1):
            if edge_target_seq is not None:
                edge_batch = edge_target_seq[idx]
                edge_pred = model.score(curr_state, edge_batch.events)
                final_edge_pred_raw = _reconstruct_raw_edge_predictions(edge_pred, rollout_prev_edge, cfg).detach()
                final_edge_true_raw = edge_batch.targets
                rollout_prev_edge = final_edge_pred_raw

            if node_target_seq is not None and getattr(model, "node_scorer", None) is not None:
                node_pred = model.score_nodes(curr_state)
                final_node_pred_raw = _reconstruct_raw_node_predictions(node_pred, rollout_prev_node, cfg).detach()
                final_node_true_raw = node_target_seq[idx]
                rollout_prev_node = final_node_pred_raw

            if idx < end_idx:
                curr_state, _ = model.step(curr_state, events_seq[idx])
                if curr_state is not None:
                    curr_state.detach_()

        if final_edge_pred_raw is not None and final_edge_true_raw is not None and persistent_prev_edge is not None:
            edge_metrics = edge_regression_metrics(final_edge_pred_raw, final_edge_true_raw)
            edge_loss = torch.nn.functional.mse_loss(final_edge_pred_raw, final_edge_true_raw)
            _acc_update(edge_acc, float(edge_loss.item()), edge_metrics)

            persistent_edge_pred_raw = persistent_prev_edge.to(
                device=final_edge_true_raw.device,
                dtype=final_edge_true_raw.dtype,
            )
            persistent_edge_metrics = edge_regression_metrics(persistent_edge_pred_raw, final_edge_true_raw)
            persistent_edge_loss = torch.nn.functional.mse_loss(persistent_edge_pred_raw, final_edge_true_raw)
            _acc_update(edge_persistent_acc, float(persistent_edge_loss.item()), persistent_edge_metrics)

        if final_node_pred_raw is not None and final_node_true_raw is not None and persistent_prev_node is not None:
            node_metrics = node_regression_metrics(final_node_pred_raw, final_node_true_raw)
            node_loss = torch.nn.functional.mse_loss(final_node_pred_raw, final_node_true_raw)
            _acc_update(node_acc, float(node_loss.item()), node_metrics)

            persistent_node_pred_raw = persistent_prev_node.to(
                device=final_node_true_raw.device,
                dtype=final_node_true_raw.dtype,
            )
            persistent_node_metrics = node_regression_metrics(persistent_node_pred_raw, final_node_true_raw)
            persistent_node_loss = torch.nn.functional.mse_loss(persistent_node_pred_raw, final_node_true_raw)
            _acc_update(node_persistent_acc, float(persistent_node_loss.item()), persistent_node_metrics)

        rollout_steps += 1

    out: Dict[str, float] = {
        "rollout_horizon": float(horizon),
        "rollout_steps": float(rollout_steps),
    }
    for prefix, acc in (("rollout", edge_acc), ("rollout_persistent", edge_persistent_acc)):
        block = _acc_finalize(acc)
        for key, value in block.items():
            if key in {"loss", "steps"}:
                continue
            out[f"{prefix}_{key}"] = float(value)
    for prefix, acc in (("rollout", node_acc), ("rollout_persistent", node_persistent_acc)):
        block = _acc_finalize(acc)
        for key, value in block.items():
            if key in {"loss", "steps"}:
                continue
            out[f"{prefix}_{key}"] = float(value)
    return out


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
    scored_step = 0
    persistent_state = None
    target_iter = iter(node_targets) if node_targets is not None else None
    edge_target_iter = iter(edge_targets) if edge_targets is not None else None
    prev_node_target: Optional[torch.Tensor] = None
    prev_edge_target: Optional[torch.Tensor] = None

    for events in bins:
        events = events.to(device)
        curr_node_target = None if target_iter is None else next(target_iter).to(device)
        curr_edge_target = None if edge_target_iter is None else next(edge_target_iter)

        if prev is None:
            prev = events
            if curr_node_target is not None:
                prev_node_target = curr_node_target.detach().to(device)
            if curr_edge_target is not None:
                prev_edge_target = curr_edge_target.targets.detach().to(device)
            continue

        state, _ = model.step(state, prev)

        node_target_type = getattr(cfg, "node_target_type", "regression")
        edge_target_type = getattr(cfg, "edge_target_type", "regression")
        node_primary = (
            curr_edge_target is None
            and curr_node_target is not None
            and getattr(model, "node_scorer", None) is not None
        )

        raw_edge_target_values: Optional[torch.Tensor] = None
        if curr_edge_target is not None:
            edge_events = curr_edge_target.events.to(device)
            raw_edge_target_values = curr_edge_target.targets.to(device)
            edge_preds = model.score(state, edge_events)
            if edge_target_type == "classification":
                edge_labels = raw_edge_target_values.to(edge_preds.device, dtype=edge_preds.dtype)
                loss_t = torch.nn.functional.binary_cross_entropy_with_logits(edge_preds, edge_labels)
                metrics = edge_prediction_metrics(edge_preds.detach(), edge_labels)
                metrics["edge_loss"] = float(loss_t.detach().item())
            else:
                edge_target_values = _transform_edge_targets(raw_edge_target_values, prev_edge_target, cfg)
                loss_t = _edge_regression_loss(edge_preds, edge_target_values, cfg)
                raw_edge_preds = _reconstruct_raw_edge_predictions(edge_preds.detach(), prev_edge_target, cfg)
                metrics = edge_regression_metrics(raw_edge_preds, raw_edge_target_values)
                if getattr(cfg, "edge_target_mode", "raw") != "raw":
                    resid_metrics = edge_regression_metrics(edge_preds.detach(), edge_target_values)
                    metrics.update({f"edge_resid_{k.removeprefix('edge_')}": v for k, v in resid_metrics.items()})
        elif node_primary:
            node_logits = model.score_nodes(state)
            assert curr_node_target is not None
            if node_target_type == "classification":
                node_labels = curr_node_target.to(node_logits.device, dtype=node_logits.dtype)
                loss_t = torch.nn.functional.binary_cross_entropy_with_logits(node_logits, node_labels)
                metrics = node_prediction_metrics(node_logits.detach(), node_labels)
            else:
                raw_node_target = curr_node_target.to(node_logits.device, dtype=node_logits.dtype)
                node_target = _transform_node_targets(raw_node_target, prev_node_target, cfg)
                loss_t = torch.nn.functional.mse_loss(node_logits, node_target)
                raw_node_preds = _reconstruct_raw_node_predictions(node_logits.detach(), prev_node_target, cfg)
                metrics = node_regression_metrics(raw_node_preds, raw_node_target)
                if getattr(cfg, "node_target_mode", "raw") != "raw":
                    resid_node_metrics = node_regression_metrics(node_logits.detach(), node_target)
                    metrics.update({f"node_resid_{k.removeprefix('node_')}": v for k, v in resid_node_metrics.items()})
            metrics["node_loss"] = float(loss_t.detach().item())
        else:
            loss_t, metrics = ranking_loss_and_metrics(
                model=model,
                state=state,
                next_events=events,
                num_nodes=cfg.num_nodes,
                num_neg=cfg.num_neg,
            )

        total_loss_t = loss_t
        if curr_edge_target is not None and getattr(model, "node_scorer", None) is not None:
            node_logits = model.score_nodes(state)
            if curr_node_target is not None:
                if node_target_type == "classification":
                    node_labels = curr_node_target.to(node_logits.device, dtype=node_logits.dtype)
                    node_loss_t = torch.nn.functional.binary_cross_entropy_with_logits(node_logits, node_labels)
                    metrics.update(node_prediction_metrics(node_logits.detach(), node_labels))
                else:
                    raw_node_target = curr_node_target.to(node_logits.device, dtype=node_logits.dtype)
                    node_target = _transform_node_targets(raw_node_target, prev_node_target, cfg)
                    node_loss_t = torch.nn.functional.mse_loss(node_logits, node_target)
                    raw_node_preds = _reconstruct_raw_node_predictions(node_logits.detach(), prev_node_target, cfg)
                    metrics.update(node_regression_metrics(raw_node_preds, raw_node_target))
                    if getattr(cfg, "node_target_mode", "raw") != "raw":
                        resid_node_metrics = node_regression_metrics(node_logits.detach(), node_target)
                        metrics.update({f"node_resid_{k.removeprefix('node_')}": v for k, v in resid_node_metrics.items()})
            else:
                node_labels = node_labels_from_events(events, cfg.num_nodes, device=node_logits.device)
                node_loss_t = torch.nn.functional.binary_cross_entropy_with_logits(node_logits, node_labels)
                metrics.update(node_prediction_metrics(node_logits.detach(), node_labels))
            metrics["node_loss"] = float(node_loss_t.detach().item())
            total_loss_t = total_loss_t + (cfg.node_loss_weight * node_loss_t)

        if curr_edge_target is not None and edge_target_type == "classification":
            assert raw_edge_target_values is not None
            if persistent_state is None:
                persistent_state = state.clone(detach=True) if state is not None else None
            persistent_edge_logits = model.score(persistent_state, edge_events)
            persistent_edge_labels = raw_edge_target_values.to(
                device=persistent_edge_logits.device,
                dtype=persistent_edge_logits.dtype,
            )
            persistent_loss_t = torch.nn.functional.binary_cross_entropy_with_logits(
                persistent_edge_logits,
                persistent_edge_labels,
            )
            persistent_metrics = edge_prediction_metrics(
                persistent_edge_logits.detach(),
                persistent_edge_labels,
            )
            persistent_metrics["edge_loss"] = float(persistent_loss_t.detach().item())
        elif curr_edge_target is not None:
            persistent_raw_edge_targets = curr_edge_target.targets.to(device)
            if prev_edge_target is None:
                raise ValueError("Persistence baseline for edge regression requires previous edge targets.")
            persistent_raw_edge_preds = prev_edge_target.to(
                device=persistent_raw_edge_targets.device,
                dtype=persistent_raw_edge_targets.dtype,
            )
            persistent_edge_targets = _transform_edge_targets(persistent_raw_edge_targets, prev_edge_target, cfg)
            persistent_edge_preds = _transform_edge_targets(persistent_raw_edge_preds, prev_edge_target, cfg)
            persistent_loss_t = _edge_regression_loss(
                persistent_edge_preds,
                persistent_edge_targets,
                cfg,
            )
            persistent_metrics = edge_regression_metrics(
                persistent_raw_edge_preds,
                persistent_raw_edge_targets,
            )
            if getattr(cfg, "edge_target_mode", "raw") != "raw":
                persistent_resid_metrics = edge_regression_metrics(
                    persistent_edge_preds,
                    persistent_edge_targets,
                )
                persistent_metrics.update(
                    {f"edge_resid_{k.removeprefix('edge_')}": v for k, v in persistent_resid_metrics.items()}
                )
        elif node_primary and node_target_type == "classification":
            assert curr_node_target is not None
            if persistent_state is None:
                persistent_state = state.clone(detach=True) if state is not None else None
            persistent_node_logits = model.score_nodes(persistent_state)
            persistent_node_labels = curr_node_target.to(
                device=persistent_node_logits.device,
                dtype=persistent_node_logits.dtype,
            )
            persistent_loss_t = torch.nn.functional.binary_cross_entropy_with_logits(
                persistent_node_logits,
                persistent_node_labels,
            )
            persistent_metrics = node_prediction_metrics(
                persistent_node_logits.detach(),
                persistent_node_labels,
            )
            persistent_metrics["node_loss"] = float(persistent_loss_t.detach().item())
        elif node_primary and curr_node_target is not None:
            persistent_raw_node_target = curr_node_target.to(device=device, dtype=torch.float32)
            if prev_node_target is None:
                raise ValueError("Persistence baseline for node regression requires previous node targets.")
            persistent_raw_node_preds = prev_node_target.to(
                device=persistent_raw_node_target.device,
                dtype=persistent_raw_node_target.dtype,
            )
            persistent_node_target = _transform_node_targets(
                persistent_raw_node_target, prev_node_target, cfg
            )
            persistent_node_preds = _transform_node_targets(
                persistent_raw_node_preds, prev_node_target, cfg
            )
            persistent_loss_t = torch.nn.functional.mse_loss(
                persistent_node_preds, persistent_node_target
            )
            persistent_metrics = node_regression_metrics(
                persistent_raw_node_preds,
                persistent_raw_node_target,
            )
            if getattr(cfg, "node_target_mode", "raw") != "raw":
                persistent_resid_node_metrics = node_regression_metrics(
                    persistent_node_preds, persistent_node_target
                )
                persistent_metrics.update(
                    {f"node_resid_{k.removeprefix('node_')}": v for k, v in persistent_resid_node_metrics.items()}
                )
        else:
            if persistent_state is None:
                persistent_state = state.clone(detach=True) if state is not None else None
            persistent_loss_t, persistent_metrics = ranking_loss_and_metrics(
                model=model,
                state=persistent_state,
                next_events=events,
                num_nodes=cfg.num_nodes,
                num_neg=cfg.num_neg,
            )

        persistent_total_loss_t = persistent_loss_t
        if curr_edge_target is not None and getattr(model, "node_scorer", None) is not None:
            if curr_node_target is not None:
                persistent_raw_node_target = curr_node_target.to(device=device, dtype=torch.float32)
                if prev_node_target is None:
                    raise ValueError("Persistence baseline for node regression requires previous node targets.")
                if node_target_type == "classification":
                    if persistent_state is None:
                        persistent_state = state.clone(detach=True) if state is not None else None
                    persistent_node_logits = model.score_nodes(persistent_state)
                    persistent_node_labels = curr_node_target.to(
                        device=persistent_node_logits.device,
                        dtype=persistent_node_logits.dtype,
                    )
                    persistent_node_loss_t = torch.nn.functional.binary_cross_entropy_with_logits(
                        persistent_node_logits, persistent_node_labels
                    )
                    persistent_metrics.update(
                        node_prediction_metrics(persistent_node_logits.detach(), persistent_node_labels)
                    )
                else:
                    persistent_raw_node_preds = prev_node_target.to(
                        device=persistent_raw_node_target.device,
                        dtype=persistent_raw_node_target.dtype,
                    )
                    persistent_node_target = _transform_node_targets(
                        persistent_raw_node_target, prev_node_target, cfg
                    )
                    persistent_node_preds = _transform_node_targets(
                        persistent_raw_node_preds, prev_node_target, cfg
                    )
                    persistent_node_loss_t = torch.nn.functional.mse_loss(
                        persistent_node_preds, persistent_node_target
                    )
                    persistent_metrics.update(
                        node_regression_metrics(persistent_raw_node_preds, persistent_raw_node_target)
                    )
                    if getattr(cfg, "node_target_mode", "raw") != "raw":
                        persistent_resid_node_metrics = node_regression_metrics(
                            persistent_node_preds, persistent_node_target
                        )
                        persistent_metrics.update(
                            {f"node_resid_{k.removeprefix('node_')}": v for k, v in persistent_resid_node_metrics.items()}
                        )
            else:
                if persistent_state is None:
                    persistent_state = state.clone(detach=True) if state is not None else None
                persistent_node_logits = model.score_nodes(persistent_state)
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
        if curr_node_target is not None:
            prev_node_target = curr_node_target.detach().to(device)
        if curr_edge_target is not None and raw_edge_target_values is not None:
            prev_edge_target = raw_edge_target_values.detach()

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

    return out
