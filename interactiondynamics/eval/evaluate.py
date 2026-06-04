# eval/evaluate.py

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Dict, Iterable, Optional, cast
import torch

from interactiondynamics.core.events import EventBatch
from interactiondynamics.core.interfaces import ModelState
from interactiondynamics.data.interfaces import EdgeTargetBatch
from interactiondynamics.eval.node_metrics import (
    edge_prediction_metrics,
    edge_regression_metrics,
    node_labels_from_events,
    node_prediction_metrics,
    node_regression_metrics,
    regression_metrics,
)
from interactiondynamics.eval.ranking_metrics import ranking_loss_and_metrics
from interactiondynamics.training.targets import (
    edge_regression_loss as _edge_regression_loss,
    edge_regression_readout as _edge_regression_readout,
    node_regression_readout as _node_regression_readout,
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


def _stash_observed_history(
    state,
    *,
    current_target: Optional[torch.Tensor],
    prev_target: Optional[torch.Tensor],
    history_targets: Optional[list[torch.Tensor]] = None,
) -> None:
    if state is None:
        return
    aux = {} if state.aux is None else dict(state.aux)
    if current_target is None:
        aux.pop("ift_state_observed_target", None)
    else:
        aux["ift_state_observed_target"] = current_target.detach()
    if prev_target is None:
        aux.pop("ift_prev_observed_target", None)
    else:
        aux["ift_prev_observed_target"] = prev_target.detach()
    if history_targets:
        aux["ift_readout_history_scalar"] = torch.stack([target.detach() for target in history_targets], dim=0)
    else:
        aux.pop("ift_readout_history_scalar", None)
    state.aux = aux


def _extract_rollout_drive(
    events: EventBatch,
    *,
    num_nodes: int,
    drive_feature_idx: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if events.features is None or events.features.numel() == 0:
        return torch.zeros((num_nodes,), device=device, dtype=dtype)

    feats = events.features.to(device=device, dtype=dtype)
    src = events.src.to(device=device, dtype=torch.long)
    dst = events.dst.to(device=device, dtype=torch.long)
    drive = torch.zeros((num_nodes,), device=device, dtype=dtype)
    values = feats[:, drive_feature_idx]
    if feats.size(1) > drive_feature_idx + 1:
        mask = feats[:, drive_feature_idx + 1]
        if torch.all((mask == 0) | (mask == 1)):
            values = values * mask
    drive.index_add_(0, dst, values)
    self_mask = src == dst
    if torch.any(self_mask):
        drive.zero_()
        drive.index_add_(0, dst[self_mask], values[self_mask])
    return drive


def _print_rollout_trace(
    *,
    coeffs: tuple[float, float, float],
    events_seq: list[EventBatch],
    start_idx: int,
    end_idx: int,
    num_nodes: int,
    drive_feature_idx: int,
    y_prev0: torch.Tensor,
    y_t0: torch.Tensor,
    trace_rows: list[dict[str, torch.Tensor | int]],
) -> None:
    a, b, c = coeffs
    ar2_y_prev = y_prev0.detach().clone()
    ar2_y_t = y_t0.detach().clone()
    max_y_prev = 0.0
    max_y_t = 0.0
    max_v = 0.0
    max_drive = 0.0
    max_delta = 0.0
    max_y_next = 0.0
    n = min(5, y_t0.numel())

    print("  rollout trace: ar2_baseline vs ift2_ar2_oracle_init")
    print(f"    start_idx={start_idx} end_idx={end_idx} horizon={end_idx - start_idx + 1}")
    if trace_rows:
        first_row = trace_rows[0]
        first_ift_y_prev = cast(torch.Tensor, first_row["ift_y_prev"])
        first_ift_y_t = cast(torch.Tensor, first_row["ift_y_t"])
        first_ift_drive = cast(torch.Tensor, first_row["ift_drive_t"])
        first_drive = _extract_rollout_drive(
            events_seq[int(cast(int, first_row["step"])) - 1],
            num_nodes=num_nodes,
            drive_feature_idx=drive_feature_idx,
            device=ar2_y_t.device,
            dtype=ar2_y_t.dtype,
        )
        first_ar2_v = ar2_y_t - ar2_y_prev
        first_ift_v = first_ift_y_t - first_ift_y_prev
        print(
            "    initial_check"
            f" | max_abs(ift_y_prev-ar2_y_prev)={(first_ift_y_prev - ar2_y_prev).abs().max().item():.6f}"
            f" max_abs(ift_y_t-ar2_y_t)={(first_ift_y_t - ar2_y_t).abs().max().item():.6f}"
            f" max_abs(ift_v_t-ar2_v_t)={(first_ift_v - first_ar2_v).abs().max().item():.6f}"
            f" max_abs(ift_drive_t-ar2_drive_t)={(first_ift_drive - first_drive).abs().max().item():.6f}"
        )
    for row in trace_rows:
        idx = int(cast(int, row["step"]))
        drive_t = _extract_rollout_drive(
            events_seq[idx - 1],
            num_nodes=num_nodes,
            drive_feature_idx=drive_feature_idx,
            device=ar2_y_t.device,
            dtype=ar2_y_t.dtype,
        )
        ar2_v_t = ar2_y_t - ar2_y_prev
        ar2_delta = (a + b - 1.0) * ar2_y_t - b * ar2_v_t + c * drive_t
        ar2_y_next = ar2_y_t + ar2_delta

        ift_y_prev = cast(torch.Tensor, row["ift_y_prev"])
        ift_y_t = cast(torch.Tensor, row["ift_y_t"])
        ift_v_t = ift_y_t - ift_y_prev
        ift_drive_t = cast(torch.Tensor, row["ift_drive_t"])
        ift_delta = cast(torch.Tensor, row["ift_delta"])
        ift_y_next = cast(torch.Tensor, row["ift_y_next"])

        abs_y_prev = (ar2_y_prev - ift_y_prev).abs()
        abs_y_t = (ar2_y_t - ift_y_t).abs()
        abs_v = (ar2_v_t - ift_v_t).abs()
        abs_drive = (drive_t - ift_drive_t).abs()
        abs_delta = (ar2_delta - ift_delta).abs()
        abs_y_next = (ar2_y_next - ift_y_next).abs()

        max_y_prev = max(max_y_prev, float(abs_y_prev.max().item()))
        max_y_t = max(max_y_t, float(abs_y_t.max().item()))
        max_v = max(max_v, float(abs_v.max().item()))
        max_drive = max(max_drive, float(abs_drive.max().item()))
        max_delta = max(max_delta, float(abs_delta.max().item()))
        max_y_next = max(max_y_next, float(abs_y_next.max().item()))

        print(f"    step={idx} sample[:{n}]")
        print(f"      drive_t={drive_t[:n].detach().cpu().tolist()}")
        print(f"      ar2_y_prev={ar2_y_prev[:n].detach().cpu().tolist()}")
        print(f"      ar2_y_t={ar2_y_t[:n].detach().cpu().tolist()}")
        print(f"      ar2_v_t={ar2_v_t[:n].detach().cpu().tolist()}")
        print(f"      ar2_delta={ar2_delta[:n].detach().cpu().tolist()}")
        print(f"      ar2_y_next={ar2_y_next[:n].detach().cpu().tolist()}")
        print(f"      ift_y_prev={ift_y_prev[:n].detach().cpu().tolist()}")
        print(f"      ift_y_t={ift_y_t[:n].detach().cpu().tolist()}")
        print(f"      ift_v_t={ift_v_t[:n].detach().cpu().tolist()}")
        print(f"      ift_delta={ift_delta[:n].detach().cpu().tolist()}")
        print(f"      ift_y_next={ift_y_next[:n].detach().cpu().tolist()}")
        print(f"      abs_diff_y_prev={abs_y_prev[:n].detach().cpu().tolist()}")
        print(f"      abs_diff_y_t={abs_y_t[:n].detach().cpu().tolist()}")
        print(f"      abs_diff_v_t={abs_v[:n].detach().cpu().tolist()}")
        print(f"      abs_diff_drive={abs_drive[:n].detach().cpu().tolist()}")
        print(f"      abs_diff_delta={abs_delta[:n].detach().cpu().tolist()}")
        print(f"      abs_diff_y_next={abs_y_next[:n].detach().cpu().tolist()}")

        ar2_y_prev = ar2_y_t
        ar2_y_t = ar2_y_next

    print(
        "    aggregate"
        f" | max_abs_y_prev_diff={max_y_prev:.6f}"
        f" max_abs_y_t_diff={max_y_t:.6f}"
        f" max_abs_v_diff={max_v:.6f}"
        f" max_abs_drive_diff={max_drive:.6f}"
        f" max_abs_delta_diff={max_delta:.6f}"
        f" max_abs_y_next_diff={max_y_next:.6f}"
    )


def _set_autonomous_rollout_readout(
    state: ModelState,
    *,
    y_prev: torch.Tensor,
    y_t: torch.Tensor,
) -> None:
    aux = {} if state.aux is None else dict(state.aux)
    velocity = y_t - y_prev
    aux["ift_position_scalar"] = y_t.detach()
    aux["ift_velocity_scalar"] = velocity.detach()
    aux["ift_readout_position_scalar"] = y_t.detach()
    aux["ift_readout_velocity_scalar"] = velocity.detach()
    state.aux = aux


def _set_rollout_history(
    state: ModelState,
    *,
    history_values: list[Optional[torch.Tensor]],
) -> None:
    aux = {} if state.aux is None else dict(state.aux)
    filtered = [value.detach() for value in history_values if value is not None]
    if filtered:
        aux["ift_readout_history_scalar"] = torch.stack(filtered, dim=0)
    else:
        aux.pop("ift_readout_history_scalar", None)
    state.aux = aux


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

    autonomous_rollout = bool(getattr(cfg, "ift_rollout_autonomous", False))

    state = model.init_state(batch_size=1, num_nodes=cfg.num_nodes, device=device)
    state_after_prev: list[Optional[ModelState]] = [None] * num_steps
    prev_node_target_seq: list[Optional[torch.Tensor]] = [None] * num_steps
    prev_edge_target_seq: list[Optional[torch.Tensor]] = [None] * num_steps
    prev_prev_prev_node_target_seq: list[Optional[torch.Tensor]] = [None] * num_steps
    prev_prev_prev_edge_target_seq: list[Optional[torch.Tensor]] = [None] * num_steps

    prev_node_target: Optional[torch.Tensor] = None
    prev_edge_target: Optional[torch.Tensor] = None
    prev_prev_prev_node_target: Optional[torch.Tensor] = None
    prev_prev_prev_edge_target: Optional[torch.Tensor] = None
    prev_prev_node_target: Optional[torch.Tensor] = None
    prev_prev_edge_target: Optional[torch.Tensor] = None
    for idx in range(num_steps):
        if idx == 0:
            prev_node_target = None if node_target_seq is None else node_target_seq[idx].detach()
            prev_edge_target = None if edge_target_seq is None else edge_target_seq[idx].targets.detach()
            continue
        observed_target = prev_edge_target if prev_edge_target is not None else prev_node_target
        observed_prev_target = prev_prev_edge_target if prev_prev_edge_target is not None else prev_prev_node_target
        _stash_observed_history(
            state,
            current_target=observed_target,
            prev_target=observed_prev_target,
            history_targets=[
                target
                for target in (
                    prev_prev_prev_edge_target if prev_edge_target is not None else prev_prev_prev_node_target,
                    prev_prev_edge_target if prev_edge_target is not None else prev_prev_node_target,
                    prev_edge_target if prev_edge_target is not None else prev_node_target,
                )
                if target is not None
            ],
        )
        state, _ = model.step(state, events_seq[idx - 1])
        state_after_prev[idx] = None if state is None else state.clone(detach=True)
        prev_node_target_seq[idx] = prev_node_target
        prev_edge_target_seq[idx] = prev_edge_target
        prev_prev_prev_node_target_seq[idx] = prev_prev_prev_node_target
        prev_prev_prev_edge_target_seq[idx] = prev_prev_prev_edge_target
        if state is not None:
            state.detach_()
        if node_target_seq is not None:
            prev_prev_prev_node_target = prev_prev_node_target
            prev_prev_node_target = prev_node_target
            prev_node_target = node_target_seq[idx].detach()
        if edge_target_seq is not None:
            prev_prev_prev_edge_target = prev_prev_edge_target
            prev_prev_edge_target = prev_edge_target
            prev_edge_target = edge_target_seq[idx].targets.detach()

    edge_acc = _acc_init()
    edge_persistent_acc = _acc_init()
    node_acc = _acc_init()
    node_persistent_acc = _acc_init()
    rollout_steps = 0

    if autonomous_rollout:
        start_range = range(1, num_steps - horizon)
    else:
        start_range = range(1, num_steps - horizon + 1)

    for start_idx in start_range:
        end_idx = start_idx + horizon if autonomous_rollout else start_idx + horizon - 1
        state_idx = start_idx + 1 if autonomous_rollout else start_idx
        rollout_state = state_after_prev[state_idx]
        if rollout_state is None:
            continue
        if autonomous_rollout:
            rollout_prev_node = None if node_target_seq is None else node_target_seq[start_idx]
            rollout_prev_edge = None if edge_target_seq is None else edge_target_seq[start_idx].targets
            rollout_prev_prev_prev_node = None if node_target_seq is None else node_target_seq[start_idx - 2] if start_idx - 2 >= 0 else None
            rollout_prev_prev_prev_edge = None if edge_target_seq is None else edge_target_seq[start_idx - 2].targets if start_idx - 2 >= 0 else None
            rollout_prev_prev_node = None if node_target_seq is None else node_target_seq[start_idx - 1]
            rollout_prev_prev_edge = None if edge_target_seq is None else edge_target_seq[start_idx - 1].targets
        else:
            rollout_prev_node = prev_node_target_seq[start_idx]
            rollout_prev_edge = prev_edge_target_seq[start_idx]
            rollout_prev_prev_prev_node = prev_prev_prev_node_target_seq[start_idx]
            rollout_prev_prev_prev_edge = prev_prev_prev_edge_target_seq[start_idx]
            rollout_prev_prev_node = prev_node_target_seq[start_idx - 1] if start_idx - 1 >= 0 else None
            rollout_prev_prev_edge = prev_edge_target_seq[start_idx - 1] if start_idx - 1 >= 0 else None
        persistent_prev_node = rollout_prev_node
        persistent_prev_edge = rollout_prev_edge
        curr_state = rollout_state.clone(detach=True)
        init_rollout_prev_prev_edge = rollout_prev_prev_edge
        init_rollout_prev_edge = rollout_prev_edge

        final_edge_pred_raw: Optional[torch.Tensor] = None
        final_edge_true_raw: Optional[torch.Tensor] = None
        final_node_pred_raw: Optional[torch.Tensor] = None
        final_node_true_raw: Optional[torch.Tensor] = None
        trace_rows: list[dict[str, torch.Tensor | int]] = []
        trace_enabled = (
            bool(getattr(cfg, "ift_rollout_trace_print", False))
            and not bool(getattr(cfg, "ift_rollout_trace_done", False))
            and edge_target_seq is not None
            and rollout_prev_edge is not None
        )

        score_start_idx = start_idx + 1 if autonomous_rollout else start_idx
        for idx in range(score_start_idx, end_idx + 1):
            prior_rollout_prev_edge = rollout_prev_edge
            prior_rollout_prev_node = rollout_prev_node
            if edge_target_seq is not None:
                if (
                    autonomous_rollout
                    and curr_state is not None
                    and rollout_prev_edge is not None
                    and rollout_prev_prev_edge is not None
                ):
                    _set_autonomous_rollout_readout(
                        curr_state,
                        y_prev=rollout_prev_prev_edge,
                        y_t=rollout_prev_edge,
                    )
                if curr_state is not None:
                    _set_rollout_history(
                        curr_state,
                        history_values=[
                            rollout_prev_prev_prev_edge,
                            rollout_prev_prev_edge,
                            rollout_prev_edge,
                        ],
                    )
                edge_batch = edge_target_seq[idx]
                edge_pred = model.score(curr_state, edge_batch.events)
                edge_view = _edge_regression_readout(edge_pred, edge_batch.targets, rollout_prev_edge, cfg)
                final_edge_pred_raw = edge_view.raw_preds.detach()
                final_edge_true_raw = edge_batch.targets
                if (
                    trace_enabled
                    and prior_rollout_prev_edge is not None
                    and curr_state is not None
                    and curr_state.aux is not None
                ):
                    force_t = curr_state.aux.get("ift_readout_force_scalar")
                    if torch.is_tensor(force_t) and edge_view.delta_preds is not None:
                        trace_rows.append(
                            {
                                "step": idx,
                                "ift_y_prev": rollout_prev_prev_edge.detach().clone()
                                if rollout_prev_prev_edge is not None
                                else prior_rollout_prev_edge.detach().clone(),
                                "ift_y_t": prior_rollout_prev_edge.detach().clone(),
                                "ift_drive_t": force_t.detach().clone(),
                                "ift_delta": edge_view.delta_preds.detach().clone(),
                                "ift_y_next": final_edge_pred_raw.detach().clone(),
                            }
                        )
                rollout_prev_edge = final_edge_pred_raw

            if node_target_seq is not None and getattr(model, "node_scorer", None) is not None:
                node_pred = model.score_nodes(curr_state)
                node_view = _node_regression_readout(node_pred, node_target_seq[idx], rollout_prev_node, cfg)
                final_node_pred_raw = node_view.raw_preds.detach()
                final_node_true_raw = node_target_seq[idx]
                rollout_prev_node = final_node_pred_raw

            if idx < end_idx:
                _stash_observed_history(
                    curr_state,
                    current_target=rollout_prev_edge if rollout_prev_edge is not None else rollout_prev_node,
                    prev_target=rollout_prev_prev_edge if rollout_prev_prev_edge is not None else rollout_prev_prev_node,
                    history_targets=[
                        target
                        for target in (
                            rollout_prev_prev_prev_edge if rollout_prev_edge is not None else rollout_prev_prev_prev_node,
                            rollout_prev_prev_edge if rollout_prev_edge is not None else rollout_prev_prev_node,
                            rollout_prev_edge if rollout_prev_edge is not None else rollout_prev_node,
                        )
                        if target is not None
                    ],
                )
                curr_state, _ = model.step(curr_state, events_seq[idx])
                if curr_state is not None:
                    curr_state.detach_()
                rollout_prev_prev_prev_edge = rollout_prev_prev_edge
                rollout_prev_prev_edge = prior_rollout_prev_edge
                rollout_prev_prev_prev_node = rollout_prev_prev_node
                rollout_prev_prev_node = prior_rollout_prev_node

        if trace_enabled and trace_rows:
            params = dict(getattr(cfg, "ift_generator_params", {}) or {})
            drive_feature_idx = getattr(cfg, "ift_drive_feature_idx", None)
            if {"a", "b", "c"} <= set(params) and drive_feature_idx is not None:
                assert init_rollout_prev_edge is not None
                trace_y_prev0 = (
                    init_rollout_prev_prev_edge
                    if init_rollout_prev_prev_edge is not None
                    else init_rollout_prev_edge
                )
                _print_rollout_trace(
                    coeffs=(float(params["a"]), float(params["b"]), float(params["c"])),
                    events_seq=events_seq,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    num_nodes=cfg.num_nodes,
                    drive_feature_idx=int(drive_feature_idx),
                    y_prev0=trace_y_prev0,
                    y_t0=init_rollout_prev_edge,
                    trace_rows=trace_rows,
                )
                setattr(cfg, "ift_rollout_trace_done", True)

        if final_edge_pred_raw is not None and final_edge_true_raw is not None and persistent_prev_edge is not None:
            edge_metrics = edge_regression_metrics(final_edge_pred_raw, final_edge_true_raw)
            edge_delta_metrics = regression_metrics(
                final_edge_pred_raw - persistent_prev_edge,
                final_edge_true_raw - persistent_prev_edge,
                prefix="edge_delta",
            )
            edge_metrics.update(edge_delta_metrics)
            edge_loss = torch.nn.functional.mse_loss(final_edge_pred_raw, final_edge_true_raw)
            _acc_update(edge_acc, float(edge_loss.item()), edge_metrics)

            persistent_edge_pred_raw = persistent_prev_edge.to(
                device=final_edge_true_raw.device,
                dtype=final_edge_true_raw.dtype,
            )
            persistent_edge_metrics = edge_regression_metrics(persistent_edge_pred_raw, final_edge_true_raw)
            persistent_edge_delta_metrics = regression_metrics(
                persistent_edge_pred_raw - persistent_prev_edge,
                final_edge_true_raw - persistent_prev_edge,
                prefix="edge_delta",
            )
            persistent_edge_metrics.update(persistent_edge_delta_metrics)
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
    prev_prev_prev_node_target: Optional[torch.Tensor] = None
    prev_edge_target: Optional[torch.Tensor] = None
    prev_prev_prev_edge_target: Optional[torch.Tensor] = None
    prev_prev_node_target: Optional[torch.Tensor] = None
    prev_prev_edge_target: Optional[torch.Tensor] = None
    sanity_done = bool(getattr(cfg, "ift_batch_sanity_done", False))

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

        observed_target = prev_edge_target if prev_edge_target is not None else prev_node_target
        observed_prev_target = prev_prev_edge_target if prev_prev_edge_target is not None else prev_prev_node_target
        _stash_observed_history(
            state,
            current_target=observed_target,
            prev_target=observed_prev_target,
            history_targets=[
                target
                for target in (
                    prev_prev_prev_edge_target if prev_edge_target is not None else prev_prev_prev_node_target,
                    prev_prev_edge_target if prev_edge_target is not None else prev_prev_node_target,
                    prev_edge_target if prev_edge_target is not None else prev_node_target,
                )
                if target is not None
            ],
        )
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
                edge_view = _edge_regression_readout(edge_preds, raw_edge_target_values, prev_edge_target, cfg)
                loss_t = _edge_regression_loss(edge_view.loss_preds, edge_view.loss_targets, cfg)
                raw_edge_preds = edge_view.raw_preds.detach()
                metrics = edge_regression_metrics(raw_edge_preds, raw_edge_target_values)
                if edge_view.delta_preds is not None and edge_view.delta_targets is not None:
                    metrics.update(
                        regression_metrics(
                            edge_view.delta_preds.detach(),
                            edge_view.delta_targets.detach(),
                            prefix="edge_delta",
                        )
                    )
                if getattr(cfg, "edge_target_mode", "raw") != "raw" and str(getattr(cfg, "prediction_mode", "state")) == "state":
                    resid_metrics = edge_regression_metrics(edge_preds.detach(), edge_view.loss_targets)
                    metrics.update({f"edge_resid_{k.removeprefix('edge_')}": v for k, v in resid_metrics.items()})
                if (
                    not sanity_done
                    and bool(getattr(cfg, "ift_batch_sanity_print", False))
                    and state is not None
                    and state.aux is not None
                    and prev_edge_target is not None
                    and prev_prev_edge_target is not None
                ):
                    pos = state.aux.get("ift_readout_position_scalar")
                    stored_v = state.aux.get("ift_readout_velocity_scalar")
                    drive = state.aux.get("ift_readout_force_scalar")
                    if torch.is_tensor(pos) and torch.is_tensor(stored_v) and torch.is_tensor(drive):
                        true_v = prev_edge_target - prev_prev_edge_target
                        true_delta = raw_edge_target_values - prev_edge_target
                        pred_delta = edge_view.delta_preds.detach() if edge_view.delta_preds is not None else torch.zeros_like(true_delta)
                        params = dict(getattr(cfg, "ift_generator_params", {}) or {})
                        if {"a", "b", "c"} <= set(params):
                            ar2_delta = (
                                (float(params["a"]) + float(params["b"]) - 1.0) * prev_edge_target
                                - float(params["b"]) * true_v
                                + float(params["c"]) * drive.to(device=true_v.device, dtype=true_v.dtype)
                            )
                        else:
                            ar2_delta = torch.zeros_like(true_delta)
                        n = min(5, true_delta.numel())
                        def _corr(x: torch.Tensor, y: torch.Tensor) -> float:
                            vx = x - x.mean()
                            vy = y - y.mean()
                            denom = float(vx.norm().item() * vy.norm().item())
                            return 0.0 if denom <= 0.0 else float((vx * vy).sum().item() / denom)
                        print("  sanity batch")
                        print(f"    y_prev={prev_edge_target[:n].detach().cpu().tolist()}")
                        print(f"    y_t={raw_edge_target_values[:n].detach().cpu().tolist()}")
                        print(f"    true_v={true_v[:n].detach().cpu().tolist()}")
                        print(f"    stored_v={stored_v[:n].detach().cpu().tolist()}")
                        print(f"    drive={drive[:n].detach().cpu().tolist()}")
                        print(f"    true_delta={true_delta[:n].detach().cpu().tolist()}")
                        print(f"    ar2_delta={ar2_delta[:n].detach().cpu().tolist()}")
                        print(f"    pred_delta={pred_delta[:n].detach().cpu().tolist()}")
                        print(
                            "    stats"
                            f" | corr(stored_v,true_v)={_corr(stored_v, true_v):.3f}"
                            f" mse(stored_v,true_v)={torch.nn.functional.mse_loss(stored_v, true_v).item():.4f}"
                            f" | corr(pred_delta,true_delta)={_corr(pred_delta, true_delta):.3f}"
                            f" mse(pred_delta,true_delta)={torch.nn.functional.mse_loss(pred_delta, true_delta).item():.4f}"
                        )
                        sanity_done = True
                        setattr(cfg, "ift_batch_sanity_done", True)
        elif node_primary:
            node_logits = model.score_nodes(state)
            assert curr_node_target is not None
            if node_target_type == "classification":
                node_labels = curr_node_target.to(node_logits.device, dtype=node_logits.dtype)
                loss_t = torch.nn.functional.binary_cross_entropy_with_logits(node_logits, node_labels)
                metrics = node_prediction_metrics(node_logits.detach(), node_labels)
            else:
                raw_node_target = curr_node_target.to(node_logits.device, dtype=node_logits.dtype)
                node_view = _node_regression_readout(node_logits, raw_node_target, prev_node_target, cfg)
                loss_t = torch.nn.functional.mse_loss(node_view.loss_preds, node_view.loss_targets)
                raw_node_preds = node_view.raw_preds.detach()
                metrics = node_regression_metrics(raw_node_preds, raw_node_target)
                if getattr(cfg, "node_target_mode", "raw") != "raw" and str(getattr(cfg, "prediction_mode", "state")) == "state":
                    resid_node_metrics = node_regression_metrics(node_logits.detach(), node_view.loss_targets)
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
                    node_view = _node_regression_readout(node_logits, raw_node_target, prev_node_target, cfg)
                    node_loss_t = torch.nn.functional.mse_loss(node_view.loss_preds, node_view.loss_targets)
                    raw_node_preds = node_view.raw_preds.detach()
                    metrics.update(node_regression_metrics(raw_node_preds, raw_node_target))
                    if getattr(cfg, "node_target_mode", "raw") != "raw" and str(getattr(cfg, "prediction_mode", "state")) == "state":
                        resid_node_metrics = node_regression_metrics(node_logits.detach(), node_view.loss_targets)
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
            if str(getattr(cfg, "prediction_mode", "state")) == "state":
                persistent_edge_targets = _transform_edge_targets(persistent_raw_edge_targets, prev_edge_target, cfg)
                persistent_edge_preds = _transform_edge_targets(persistent_raw_edge_preds, prev_edge_target, cfg)
                persistent_loss_t = _edge_regression_loss(
                    persistent_edge_preds,
                    persistent_edge_targets,
                    cfg,
                )
            else:
                persistent_delta_preds = torch.zeros_like(persistent_raw_edge_targets)
                persistent_view = _edge_regression_readout(
                    persistent_delta_preds,
                    persistent_raw_edge_targets,
                    prev_edge_target,
                    cfg,
                )
                persistent_loss_t = _edge_regression_loss(
                    persistent_view.loss_preds,
                    persistent_view.loss_targets,
                    cfg,
                )
            persistent_metrics = edge_regression_metrics(
                persistent_raw_edge_preds,
                persistent_raw_edge_targets,
            )
            persistent_metrics.update(
                regression_metrics(
                    persistent_raw_edge_preds - prev_edge_target,
                    persistent_raw_edge_targets - prev_edge_target,
                    prefix="edge_delta",
                )
            )
            if getattr(cfg, "edge_target_mode", "raw") != "raw" and str(getattr(cfg, "prediction_mode", "state")) == "state":
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
            if str(getattr(cfg, "prediction_mode", "state")) == "state":
                persistent_node_target = _transform_node_targets(
                    persistent_raw_node_target, prev_node_target, cfg
                )
                persistent_node_preds = _transform_node_targets(
                    persistent_raw_node_preds, prev_node_target, cfg
                )
                persistent_loss_t = torch.nn.functional.mse_loss(
                    persistent_node_preds, persistent_node_target
                )
            else:
                persistent_delta_preds = torch.zeros_like(persistent_raw_node_target)
                persistent_view = _node_regression_readout(
                    persistent_delta_preds,
                    persistent_raw_node_target,
                    prev_node_target,
                    cfg,
                )
                persistent_loss_t = torch.nn.functional.mse_loss(
                    persistent_view.loss_preds,
                    persistent_view.loss_targets,
                )
            persistent_metrics = node_regression_metrics(
                persistent_raw_node_preds,
                persistent_raw_node_target,
            )
            if getattr(cfg, "node_target_mode", "raw") != "raw" and str(getattr(cfg, "prediction_mode", "state")) == "state":
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
                    if str(getattr(cfg, "prediction_mode", "state")) == "state":
                        persistent_node_target = _transform_node_targets(
                            persistent_raw_node_target, prev_node_target, cfg
                        )
                        persistent_node_preds = _transform_node_targets(
                            persistent_raw_node_preds, prev_node_target, cfg
                        )
                        persistent_node_loss_t = torch.nn.functional.mse_loss(
                            persistent_node_preds, persistent_node_target
                        )
                    else:
                        persistent_delta_preds = torch.zeros_like(persistent_raw_node_target)
                        persistent_view = _node_regression_readout(
                            persistent_delta_preds,
                            persistent_raw_node_target,
                            prev_node_target,
                            cfg,
                        )
                        persistent_node_loss_t = torch.nn.functional.mse_loss(
                            persistent_view.loss_preds,
                            persistent_view.loss_targets,
                        )
                    persistent_metrics.update(
                        node_regression_metrics(persistent_raw_node_preds, persistent_raw_node_target)
                    )
                    if getattr(cfg, "node_target_mode", "raw") != "raw" and str(getattr(cfg, "prediction_mode", "state")) == "state":
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
            prev_prev_prev_node_target = prev_prev_node_target
            prev_prev_node_target = prev_node_target
            prev_node_target = curr_node_target.detach().to(device)
        if curr_edge_target is not None and raw_edge_target_values is not None:
            prev_prev_prev_edge_target = prev_prev_edge_target
            prev_prev_edge_target = prev_edge_target
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
