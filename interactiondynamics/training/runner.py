from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from typing import Any, Callable, Iterable, Optional

import numpy as np
import torch

from interactiondynamics.core.config import ModelConfig
from interactiondynamics.core.events import EventBatch
from interactiondynamics.data.interfaces import EdgeTargetBatch
from interactiondynamics.eval.evaluate import EvalSlices, evaluate_k_step_rollout, evaluate_stream_sliced
from interactiondynamics.eval.node_metrics import (
    edge_prediction_metrics,
    edge_regression_metrics,
    node_labels_from_events,
    node_prediction_metrics,
    node_regression_metrics,
)
from interactiondynamics.eval.ranking_metrics import ranking_loss_and_metrics
from interactiondynamics.training.reporting import (
    format_edge_classification_bundle,
    format_edge_metric_bundle,
    format_edge_target_summary,
    format_node_metric,
    format_node_metric_bundle,
    format_primary_metric,
    format_ranking_metric_bundle,
    format_rollout_metric_bundle,
    infer_primary_metric,
    node_metric_name,
    primary_metric_name,
)
from interactiondynamics.training.task_metrics import (
    TaskMetricSpec,
    is_better_metric,
    snapshot_metric_value,
)
from interactiondynamics.training.targets import (
    edge_regression_loss,
    global_grad_norm,
    reconstruct_raw_edge_predictions,
    reconstruct_raw_node_predictions,
    summarize_edge_targets,
    summarize_targets,
    transform_edge_targets,
    transform_node_targets,
)
from interactiondynamics.training.types import RunResult, SweepRun, TrainConfig


def short_run_label(run: SweepRun) -> str:
    agg = run.model_cfg.aggregator
    if agg == "settransformer":
        agg = "settf"
    upd = run.model_cfg.update
    return f"{agg}/{upd}"


def short_run_label_from_name(name: str) -> str:
    parts = dict(piece.split("=", 1) for piece in name.split("|") if "=" in piece)
    agg = parts.get("agg", "?")
    if agg == "settransformer":
        agg = "settf"
    upd = parts.get("update", "?")
    return f"{agg}/{upd}"


def apply_model_overrides(model_cfg: ModelConfig, args: argparse.Namespace) -> ModelConfig:
    cfg = ModelConfig(**asdict(model_cfg))
    cfg.use_node_scorer = bool(args.use_node_scorer) or bool(cfg.use_node_scorer)
    cfg.node_scorer_hidden = int(args.node_scorer_hidden)
    return cfg


def train_one_epoch(
    model,
    bins: Iterable[EventBatch],
    node_targets: Optional[Iterable[torch.Tensor]],
    edge_targets: Optional[Iterable[EdgeTargetBatch]],
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
) -> dict[str, float]:
    model.train()
    device = torch.device(cfg.device)
    state = model.init_state(batch_size=1, num_nodes=cfg.num_nodes, device=device)

    total_loss = 0.0
    total_primary = 0.0
    n_steps = 0
    kappa_sum = 0.0
    kappa_n = 0
    grad_norm_sum = 0.0
    state_std_sum = 0.0
    state_abs_sum = 0.0
    state_delta_sum = 0.0
    state_stat_n = 0
    state_delta_n = 0

    prev: Optional[EventBatch] = None
    target_iter = iter(node_targets) if node_targets is not None else None
    edge_target_iter = iter(edge_targets) if edge_targets is not None else None
    prev_node_target: Optional[torch.Tensor] = None
    prev_edge_target: Optional[torch.Tensor] = None
    primary_name: Optional[str] = None
    for curr in bins:
        curr = curr.to(device)
        curr_node_target = None if target_iter is None else next(target_iter).to(device)
        curr_edge_target = None if edge_target_iter is None else next(edge_target_iter)
        if prev is None:
            prev = curr
            if curr_node_target is not None:
                prev_node_target = curr_node_target.detach().to(device)
            if curr_edge_target is not None:
                prev_edge_target = curr_edge_target.targets.detach().to(device)
            continue

        state_before = None if state is None or state.node is None else state.node.detach().clone()
        state, aux = model.step(state, prev)

        if aux is not None and "kappa" in aux:
            kappa = aux["kappa"]
            if torch.is_tensor(kappa):
                kappa_sum += float(kappa.detach().item())
                kappa_n += 1

        h = state.node
        if h is not None:
            state_std_sum += float(h.std(unbiased=False).item())
            state_abs_sum += float(h.abs().mean().item())
            state_stat_n += 1
            if state_before is not None and state_before.shape == h.shape:
                state_delta_sum += float((h.detach() - state_before).pow(2).mean().sqrt().item())
                state_delta_n += 1
        if cfg.debug and h is not None:
            print(
                "DEBUG node variance:",
                float(h.std(dim=0).mean().item()),
                "max|h|:",
                float(h.abs().max().item()),
            )

        if prev.t is not None and getattr(state, "aux", None) is not None and "L_bin_t_min" in state.aux:
            assert state.aux["L_bin_t_min"] == int(prev.t.min().item()), (
                "step() did not use prev bin for operator"
            )

        if state.node is not None and (not torch.isfinite(state.node).all()):
            raise RuntimeError("Non-finite state.node after model.step()")

        optimizer.zero_grad(set_to_none=True)

        assert prev.t is not None and int(prev.t.min().item()) == int(prev.t.max().item())
        assert curr.t is not None and int(curr.t.min().item()) == int(curr.t.max().item())
        assert int(prev.t.max().item()) < int(curr.t.min().item())

        if getattr(state, "aux", None) is not None:
            if "L_bin_t_min" in state.aux and "L_bin_t_max" in state.aux:
                prev_time = int(prev.t.min().item())
                assert state.aux["L_bin_t_min"] == prev_time and state.aux["L_bin_t_max"] == prev_time, (
                    f"L bin mismatch: L=({state.aux['L_bin_t_min']},{state.aux['L_bin_t_max']}) prev.t={prev_time}"
                )

        node_target_type = getattr(cfg, "node_target_type", "regression")
        edge_target_type = getattr(cfg, "edge_target_type", "regression")
        node_primary = curr_edge_target is None and curr_node_target is not None and getattr(model, "node_scorer", None) is not None

        if curr_edge_target is not None:
            edge_events = curr_edge_target.events.to(device)
            edge_preds = model.score(state, edge_events)
            raw_edge_target_values = curr_edge_target.targets.to(device)
            if edge_target_type == "classification":
                edge_labels = raw_edge_target_values.to(edge_preds.device, dtype=edge_preds.dtype)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(edge_preds, edge_labels)
                metrics = edge_prediction_metrics(edge_preds.detach(), edge_labels)
                metrics["edge_loss"] = float(loss.detach().item())
            else:
                edge_target_values = transform_edge_targets(raw_edge_target_values, prev_edge_target, cfg)
                loss = edge_regression_loss(edge_preds, edge_target_values, cfg)
                raw_edge_preds = reconstruct_raw_edge_predictions(edge_preds.detach(), prev_edge_target, cfg)
                metrics = edge_regression_metrics(raw_edge_preds, raw_edge_target_values)
                if cfg.edge_target_mode != "raw":
                    resid_metrics = edge_regression_metrics(edge_preds.detach(), edge_target_values)
                    metrics.update({f"edge_resid_{k.removeprefix('edge_')}": v for k, v in resid_metrics.items()})
            total_step_loss = loss
        elif node_primary:
            node_logits = model.score_nodes(state)
            assert curr_node_target is not None
            if node_target_type == "classification":
                node_labels = curr_node_target.to(node_logits.device, dtype=node_logits.dtype)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(node_logits, node_labels)
                metrics = node_prediction_metrics(node_logits.detach(), node_labels)
            else:
                raw_node_target = curr_node_target.to(node_logits.device, dtype=node_logits.dtype)
                node_target = transform_node_targets(raw_node_target, prev_node_target, cfg)
                loss = torch.nn.functional.mse_loss(node_logits, node_target)
                raw_node_preds = reconstruct_raw_node_predictions(node_logits.detach(), prev_node_target, cfg)
                metrics = node_regression_metrics(raw_node_preds, raw_node_target)
                if cfg.node_target_mode != "raw":
                    resid_node_metrics = node_regression_metrics(node_logits.detach(), node_target)
                    metrics.update({f"node_resid_{k.removeprefix('node_')}": v for k, v in resid_node_metrics.items()})
            metrics["node_loss"] = float(loss.detach().item())
            total_step_loss = loss
        else:
            loss, metrics = ranking_loss_and_metrics(
                model=model,
                state=state,
                next_events=curr,
                num_nodes=cfg.num_nodes,
                num_neg=cfg.num_neg,
            )
            total_step_loss = loss
        if curr_edge_target is not None and getattr(model, "node_scorer", None) is not None:
            node_logits = model.score_nodes(state)
            if curr_node_target is not None:
                if node_target_type == "classification":
                    node_labels = curr_node_target.to(node_logits.device, dtype=node_logits.dtype)
                    node_loss = torch.nn.functional.binary_cross_entropy_with_logits(node_logits, node_labels)
                    metrics.update(node_prediction_metrics(node_logits.detach(), node_labels))
                else:
                    raw_node_target = curr_node_target.to(node_logits.device, dtype=node_logits.dtype)
                    node_target = transform_node_targets(raw_node_target, prev_node_target, cfg)
                    node_loss = torch.nn.functional.mse_loss(node_logits, node_target)
                    raw_node_preds = reconstruct_raw_node_predictions(node_logits.detach(), prev_node_target, cfg)
                    metrics.update(node_regression_metrics(raw_node_preds, raw_node_target))
                    if cfg.node_target_mode != "raw":
                        resid_node_metrics = node_regression_metrics(node_logits.detach(), node_target)
                        metrics.update({f"node_resid_{k.removeprefix('node_')}": v for k, v in resid_node_metrics.items()})
            else:
                node_labels = node_labels_from_events(curr, cfg.num_nodes, device=node_logits.device)
                node_loss = torch.nn.functional.binary_cross_entropy_with_logits(node_logits, node_labels)
                metrics.update(node_prediction_metrics(node_logits.detach(), node_labels))
            metrics["node_loss"] = float(node_loss.detach().item())
            total_step_loss = total_step_loss + (cfg.node_loss_weight * node_loss)

        total_step_loss.backward()
        if cfg.grad_clip and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        grad_norm_sum += global_grad_norm(model.parameters())
        optimizer.step()

        if cfg.tbptt_steps and ((n_steps + 1) % cfg.tbptt_steps == 0) and state is not None:
            state.detach_()

        total_loss += float(total_step_loss.item())
        primary_name = primary_metric_name(metrics)
        total_primary += float(metrics[primary_name])
        n_steps += 1

        if cfg.log_every and (n_steps % cfg.log_every) == 0:
            if "mrr" in metrics:
                print(
                    f"[step {n_steps:6d}] "
                    f"loss={loss.item():.4f} "
                    f"mrr={metrics['mrr']:.4f} "
                    f"hits@1={metrics.get('hits@1', 0):.4f} "
                    f"hits@10={metrics.get('hits@10', 0):.4f}"
                )
            elif "edge_auroc" in metrics:
                print(
                    f"[step {n_steps:6d}] "
                    f"loss={loss.item():.4f} "
                    f"{format_edge_classification_bundle(metrics)}"
                )
            elif "node_auroc" in metrics:
                print(
                    f"[step {n_steps:6d}] "
                    f"loss={loss.item():.4f} "
                    f"{format_node_metric(metrics)}"
                )
            else:
                print(
                    f"[step {n_steps:6d}] "
                    f"loss={loss.item():.4f} "
                    f"{format_primary_metric(metrics)}"
                )

        prev = curr
        if curr_node_target is not None:
            prev_node_target = curr_node_target.detach().to(device)
        if curr_edge_target is not None:
            prev_edge_target = raw_edge_target_values.detach()

    if n_steps == 0:
        return {"loss": 0.0}

    if primary_name is None:
        primary_name = "mrr"
    out = {"loss": total_loss / n_steps, primary_name: total_primary / n_steps}
    if kappa_n > 0:
        out["kappa_mean"] = kappa_sum / kappa_n
    if n_steps > 0:
        out["grad_norm_mean"] = grad_norm_sum / n_steps
    if state_stat_n > 0:
        out["state_node_std_mean"] = state_std_sum / state_stat_n
        out["state_node_abs_mean"] = state_abs_sum / state_stat_n
    if state_delta_n > 0:
        out["state_delta_rms_mean"] = state_delta_sum / state_delta_n
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_one_experiment(
    ds,
    spec,
    base_train_cfg: TrainConfig,
    run: SweepRun,
    build_model_fn: Callable[[Any, ModelConfig], torch.nn.Module],
    epochs: int = 5,
    objective_metric: Optional[TaskMetricSpec] = None,
    eval_slices: Optional[EvalSlices] = None,
    save_jsonl_path: Optional[str] = None,
    rollout_horizon: int = 5,
) -> RunResult:
    device = torch.device(base_train_cfg.device)
    set_seed(run.seed)

    train_cfg = TrainConfig(**asdict(base_train_cfg))
    if run.lr is not None:
        train_cfg.lr = run.lr
    if run.weight_decay is not None:
        train_cfg.weight_decay = run.weight_decay
    if run.num_neg is not None:
        train_cfg.num_neg = run.num_neg
    if run.tbptt_steps is not None:
        train_cfg.tbptt_steps = run.tbptt_steps

    model = build_model_fn(spec, run.model_cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )

    if eval_slices is None:
        eval_slices = EvalSlices(early_steps=10)

    best_val_loss = float("inf")
    best_val_mrr = float("nan")
    best_epoch = -1
    best_snapshot: dict[str, Any] = {}
    best_objective_value = float("nan")
    t0 = time.time()
    train_node_targets = ds.node_targets("train") if hasattr(ds, "node_targets") else None
    val_node_targets = ds.node_targets("val") if hasattr(ds, "node_targets") else None
    test_node_targets = ds.node_targets("test") if hasattr(ds, "node_targets") else None
    train_node_stats = summarize_targets(train_node_targets, mode="raw")
    val_node_stats = summarize_targets(val_node_targets, mode="raw")
    test_node_stats = summarize_targets(test_node_targets, mode="raw")
    train_node_mode_stats = summarize_targets(train_node_targets, mode=train_cfg.node_target_mode)
    val_node_mode_stats = summarize_targets(val_node_targets, mode=train_cfg.node_target_mode)
    test_node_mode_stats = summarize_targets(test_node_targets, mode=train_cfg.node_target_mode)
    train_edge_targets = ds.edge_targets("train") if hasattr(ds, "edge_targets") else None
    val_edge_targets = ds.edge_targets("val") if hasattr(ds, "edge_targets") else None
    test_edge_targets = ds.edge_targets("test") if hasattr(ds, "edge_targets") else None
    train_edge_stats = summarize_edge_targets(train_edge_targets, mode="raw")
    val_edge_stats = summarize_edge_targets(val_edge_targets, mode="raw")
    test_edge_stats = summarize_edge_targets(test_edge_targets, mode="raw")
    train_edge_mode_stats = summarize_edge_targets(train_edge_targets, mode=train_cfg.edge_target_mode)
    val_edge_mode_stats = summarize_edge_targets(val_edge_targets, mode=train_cfg.edge_target_mode)
    test_edge_mode_stats = summarize_edge_targets(test_edge_targets, mode=train_cfg.edge_target_mode)
    if train_edge_mode_stats is not None:
        train_cfg.edge_target_mean = float(train_edge_mode_stats["mean"])
        train_cfg.edge_target_std = max(float(train_edge_mode_stats["std"]), 1e-12)

    if train_edge_stats is not None:
        print(
            "  edge targets(raw)"
            f" | {format_edge_target_summary('train', train_edge_stats)}"
            f" | {format_edge_target_summary('val', val_edge_stats)}"
            f" | {format_edge_target_summary('test', test_edge_stats)}"
        )
    if train_node_stats is not None:
        print(
            "  node targets(raw)"
            f" | {format_edge_target_summary('train', train_node_stats)}"
            f" | {format_edge_target_summary('val', val_node_stats)}"
            f" | {format_edge_target_summary('test', test_node_stats)}"
        )
    if train_cfg.edge_target_mode != "raw" and train_edge_mode_stats is not None:
        print(
            f"  edge targets({train_cfg.edge_target_mode})"
            f" | scale={train_cfg.edge_target_scale}"
            f" | {format_edge_target_summary('train', train_edge_mode_stats)}"
            f" | {format_edge_target_summary('val', val_edge_mode_stats)}"
            f" | {format_edge_target_summary('test', test_edge_mode_stats)}"
        )
    elif train_edge_mode_stats is not None:
        print(f"  edge supervision | mode=raw | scale={train_cfg.edge_target_scale}")
    if train_cfg.node_target_mode != "raw" and train_node_mode_stats is not None:
        print(
            f"  node targets({train_cfg.node_target_mode})"
            f" | {format_edge_target_summary('train', train_node_mode_stats)}"
            f" | {format_edge_target_summary('val', val_node_mode_stats)}"
            f" | {format_edge_target_summary('test', test_node_mode_stats)}"
        )

    baseline_val = evaluate_stream_sliced(
        model,
        ds.bins("val"),
        val_node_targets,
        val_edge_targets,
        train_cfg,
        slices=eval_slices,
    )
    baseline_test = evaluate_stream_sliced(
        model,
        ds.bins("test"),
        test_node_targets,
        test_edge_targets,
        train_cfg,
        slices=eval_slices,
    )
    run_primary_name = infer_primary_metric(baseline_val, baseline_test)
    baseline_str = (
        f"  baseline (no-update)"
        f" | {format_primary_metric(baseline_val, prefix='persistent', name=run_primary_name)}"
        f" | {format_primary_metric(baseline_test, prefix='persistent', name=run_primary_name)}"
    )
    if run_primary_name == "edge_mse":
        baseline_val_edge = format_edge_metric_bundle(baseline_val, prefix="persistent")
        baseline_test_edge = format_edge_metric_bundle(baseline_test, prefix="persistent")
        baseline_str = (
            f"  baseline (no-update)"
            f" | val {baseline_val_edge}"
            f" | test {baseline_test_edge}"
        )
    elif run_primary_name == "edge_auroc":
        baseline_val_edge = format_edge_classification_bundle(baseline_val, prefix="persistent")
        baseline_test_edge = format_edge_classification_bundle(baseline_test, prefix="persistent")
        baseline_str = (
            f"  baseline (no-update)"
            f" | val {baseline_val_edge}"
            f" | test {baseline_test_edge}"
        )
    baseline_node_val = format_node_metric(
        {
            key.replace("persistent_", "", 1): value
            for key, value in baseline_val.items()
            if key.startswith("persistent_node_")
        }
    )
    baseline_node_test = format_node_metric(
        {
            key.replace("persistent_", "", 1): value
            for key, value in baseline_test.items()
            if key.startswith("persistent_node_")
        }
    )
    if baseline_node_val or baseline_node_test:
        baseline_str += f" | val {baseline_node_val} | test {baseline_node_test}"
    print(baseline_str)

    snapshot: dict[str, Any] = {
        "epoch": 0,
        "train_step": {},
        "train_eval": {},
        "val": baseline_val,
        "test": baseline_test,
        "rollout_val": {},
        "rollout_test": {},
    }
    for epoch in range(1, epochs + 1):
        train_stats_step = train_one_epoch(
            model,
            ds.bins("train"),
            train_node_targets,
            train_edge_targets,
            optimizer,
            train_cfg,
        )
        train_eval = evaluate_stream_sliced(
            model,
            ds.bins("train"),
            train_node_targets,
            train_edge_targets,
            train_cfg,
            slices=eval_slices,
        )
        val_stats = evaluate_stream_sliced(
            model,
            ds.bins("val"),
            val_node_targets,
            val_edge_targets,
            train_cfg,
            slices=eval_slices,
        )
        test_stats = evaluate_stream_sliced(
            model,
            ds.bins("test"),
            test_node_targets,
            test_edge_targets,
            train_cfg,
            slices=eval_slices,
        )
        rollout_val_stats: dict[str, float] = {}
        rollout_test_stats: dict[str, float] = {}
        if (
            train_edge_targets is not None and train_cfg.edge_target_type == "regression"
        ) or (
            train_node_targets is not None and train_cfg.node_target_type == "regression"
        ):
            rollout_val_stats = evaluate_k_step_rollout(
                model,
                ds.bins("val"),
                val_node_targets,
                val_edge_targets,
                train_cfg,
                horizon=rollout_horizon,
            )
            rollout_test_stats = evaluate_k_step_rollout(
                model,
                ds.bins("test"),
                test_node_targets,
                test_edge_targets,
                train_cfg,
                horizon=rollout_horizon,
            )

        snapshot = {
            "epoch": epoch,
            "train_step": train_stats_step,
            "train_eval": train_eval,
            "val": val_stats,
            "test": test_stats,
            "rollout_val": rollout_val_stats,
            "rollout_test": rollout_test_stats,
        }

        km = train_stats_step.get("kappa_mean", None)
        kappa_str = f" | kappa={km:.4f}" if km is not None else ""
        node_val_str = format_node_metric(val_stats)
        node_test_str = format_node_metric(test_stats)
        run_primary_name = infer_primary_metric(train_eval, val_stats, test_stats)
        val_primary = format_primary_metric(val_stats, name=run_primary_name)
        test_primary = format_primary_metric(test_stats, name=run_primary_name)
        train_primary = format_primary_metric(train_eval, name=run_primary_name)
        metric_str = (
            f" | train {train_primary}"
            f" | val {val_primary}"
            f" | test {test_primary}"
        )
        if run_primary_name == "edge_mse":
            metric_str = (
                f" | val {format_edge_metric_bundle(val_stats)}"
                f" | test {format_edge_metric_bundle(test_stats)}"
            )
        elif run_primary_name == "edge_auroc":
            metric_str = (
                f" | val {format_edge_classification_bundle(val_stats)}"
                f" | test {format_edge_classification_bundle(test_stats)}"
            )
        elif run_primary_name == "node_mse":
            metric_str = (
                f" | val {format_node_metric_bundle(val_stats)}"
                f" | test {format_node_metric_bundle(test_stats)}"
            )
        elif run_primary_name in {"node_auroc", "node_f1", "node_acc"}:
            metric_str = (
                f" | val {node_val_str or 'n/a'}"
                f" | test {node_test_str or 'n/a'}"
            )
        elif run_primary_name == "mrr":
            metric_str = (
                f" | val {format_ranking_metric_bundle(val_stats)}"
                f" | test {format_ranking_metric_bundle(test_stats)}"
            )
        losses_str = (
            f"           losses  | train={train_stats_step['loss']:.4f}"
            f" | val={val_stats['loss']:.4f}"
            f" | test={test_stats['loss']:.4f}"
            f"{kappa_str}"
        )
        print(f"  ep {epoch:03d}")
        print(losses_str)
        print(f"           metrics{metric_str}")
        if (node_val_str or node_test_str) and run_primary_name not in {"node_mse", "node_auroc", "node_f1", "node_acc"}:
            print(
                "           nodes"
                f"   | val={node_val_str or 'n/a'}"
                f" | test={node_test_str or 'n/a'}"
            )
        if rollout_val_stats:
            rollout_edge_str = format_rollout_metric_bundle(rollout_val_stats, stem="edge")
            rollout_test_edge_str = format_rollout_metric_bundle(rollout_test_stats, stem="edge")
            rollout_node_str = format_rollout_metric_bundle(rollout_val_stats, stem="node")
            rollout_test_node_str = format_rollout_metric_bundle(rollout_test_stats, stem="node")
            rollout_parts = []
            if rollout_edge_str:
                rollout_parts.append(f"val edge {rollout_edge_str}")
            if rollout_test_edge_str:
                rollout_parts.append(f"test edge {rollout_test_edge_str}")
            if rollout_node_str:
                rollout_parts.append(f"val node {rollout_node_str}")
            if rollout_test_node_str:
                rollout_parts.append(f"test node {rollout_test_node_str}")
            if rollout_parts:
                print(
                    f"           rollout@{rollout_horizon}"
                    f" | {' | '.join(rollout_parts)}"
                )

        candidate_objective = (
            snapshot_metric_value(snapshot, objective_metric.path)
            if objective_metric is not None
            else float(val_stats["loss"])
        )
        if objective_metric is not None:
            better_snapshot = is_better_metric(
                candidate_objective,
                best_objective_value,
                objective_metric.goal,
            )
            if best_epoch < 0 and candidate_objective != candidate_objective:
                better_snapshot = True
        else:
            better_snapshot = float(val_stats["loss"]) < best_val_loss

        if better_snapshot:
            best_val_loss = float(val_stats["loss"])
            best_val_mrr = float(val_stats["mrr"]) if "mrr" in val_stats else float("nan")
            best_epoch = epoch
            best_snapshot = snapshot
            best_objective_value = candidate_objective

        if save_jsonl_path is not None:
            row = {
                "run": run.name,
                "seed": run.seed,
                "model_cfg": asdict(run.model_cfg),
                "train_cfg_overrides": {
                    key: value
                    for key, value in {
                        "lr": run.lr,
                        "weight_decay": run.weight_decay,
                        "num_neg": run.num_neg,
                        "tbptt_steps": run.tbptt_steps,
                    }.items()
                    if value is not None
                },
                **snapshot,
            }
            with open(save_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

    wall = time.time() - t0
    final_snapshot = snapshot
    return RunResult(
        name=run.name,
        seed=run.seed,
        epochs=epochs,
        best_val_loss=best_val_loss,
        best_val_mrr=best_val_mrr,
        best_epoch=best_epoch,
        best_snapshot=best_snapshot,
        final_snapshot=final_snapshot,
        wall_sec=wall,
        best_objective_path=None if objective_metric is None else objective_metric.path,
        best_objective_goal=None if objective_metric is None else objective_metric.goal,
        best_objective_value=best_objective_value,
    )
