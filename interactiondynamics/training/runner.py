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
from interactiondynamics.core.events import EventBatch, pack_independent_episode_bins
from interactiondynamics.data.interfaces import EdgeTargetBatch
from interactiondynamics.eval.evaluate import (
    EvalSlices,
    _counterfactual_field_events,
    _field_counterfactual_targets,
    _remove_wave_drive_events,
    _self_generate_grid_wave_events,
    evaluate_physical_force_rollout,
    evaluate_k_step_rollout,
    evaluate_stream_sliced,
)
from interactiondynamics.eval.node_metrics import (
    edge_prediction_metrics,
    edge_regression_metrics,
    node_labels_from_events,
    node_prediction_metrics,
    node_regression_metrics,
    regression_metrics,
)
from interactiondynamics.eval.ranking_metrics import ranking_loss_and_metrics
from interactiondynamics.training.reporting import (
    format_edge_classification_bundle,
    format_edge_metric_bundle,
    format_edge_target_summary,
    format_node_metric,
    format_node_metric_bundle,
    format_physical_rollout_bundle,
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
    edge_regression_readout,
    global_grad_norm,
    node_regression_readout,
    reconstruct_raw_edge_predictions,
    reconstruct_raw_node_predictions,
    summarize_edge_targets,
    summarize_targets,
    transform_edge_targets,
    transform_node_targets,
)
from interactiondynamics.training.types import RunResult, SweepRun, TrainConfig
from interactiondynamics.updates.ift_update import IFTSecondOrderUpdate


def _linear_hvf_readout_snapshot(model) -> dict[str, float]:
    scorer = getattr(model, "scorer", None)
    if scorer is None or not hasattr(scorer, "coefficient_dict"):
        return {}
    coeffs = scorer.coefficient_dict()
    oracle = scorer.oracle_coefficient_dict() if hasattr(scorer, "oracle_coefficient_dict") else None
    out = dict(coeffs)
    if oracle is not None:
        out.update(oracle)
        out["abs_diff_w_y"] = abs(coeffs["w_y"] - oracle["w_y_oracle"])
        out["abs_diff_w_v"] = abs(coeffs["w_v"] - oracle["w_v_oracle"])
        out["abs_diff_w_drive"] = abs(coeffs["w_drive"] - oracle["w_drive_oracle"])
        out["abs_diff_bias"] = abs(coeffs["bias"] - oracle["bias_oracle"])
    return out


def _parameter_recovery_snapshot(model, hidden_truth: Optional[dict[str, Any]], device: torch.device) -> dict[str, float]:
    """Return cheap synthetic-parameter recovery metrics without stream evaluation."""
    recovery_fn = getattr(model, "recovery_metrics", None)
    if hidden_truth is None or not callable(recovery_fn):
        return {}
    return recovery_fn(hidden_truth["adjacency"].to(device), hidden_truth["params"])


def short_run_label(run: SweepRun) -> str:
    if "agg=" not in run.name or "update=" not in run.name:
        return run.name
    agg = run.model_cfg.aggregator
    if agg == "settransformer":
        agg = "settf"
    upd = run.model_cfg.update
    return f"{agg}/{upd}"


def short_run_label_from_name(name: str) -> str:
    parts = dict(piece.split("=", 1) for piece in name.split("|") if "=" in piece)
    if "agg" not in parts or "update" not in parts:
        return name
    agg = parts.get("agg", "?")
    if agg == "settransformer":
        agg = "settf"
    upd = parts.get("update", "?")
    return f"{agg}/{upd}"


def apply_model_overrides(model_cfg: ModelConfig, args: argparse.Namespace) -> ModelConfig:
    cfg = ModelConfig(**asdict(model_cfg))
    cfg.use_node_scorer = bool(args.use_node_scorer) or bool(cfg.use_node_scorer)
    cfg.node_scorer_hidden = int(args.node_scorer_hidden)
    if getattr(args, "fnn_force_decoder", None) is not None:
        cfg.fnn_force_decoder = str(args.fnn_force_decoder)
    cfg.fnn_learn_physical_params = bool(
        getattr(args, "fnn_learn_physical_params", False)
    ) or bool(cfg.fnn_learn_physical_params)
    # Deliberately expose only the compact physical-event experiment surface;
    # the legacy IFT knobs remain preset-owned.
    for arg_name, config_name in (
        ("event_feature_loss_weight", "event_feature_loss_weight"),
        ("event_feature_magnitude_weight", "event_feature_magnitude_weight"),
        ("fnn_dt", "fnn_dt"),
        ("fnn_gamma_init", "fnn_gamma_init"),
        ("fnn_omega_init", "fnn_omega_init"),
        ("fnn_force_scale_init", "fnn_force_scale_init"),
        ("fnn_topology_init", "fnn_topology_init"),
        ("lnn_dt", "lnn_dt"),
        ("lnn_hidden", "lnn_hidden"),
        ("lnn_layers", "lnn_layers"),
        ("lnn_damping", "lnn_damping"),
        ("hnn_dt", "hnn_dt"),
        ("hnn_hidden", "hnn_hidden"),
        ("hnn_layers", "hnn_layers"),
        ("hnn_damping", "hnn_damping"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(cfg, config_name, value)
    return cfg


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


def _stash_rollout_history(
    state,
    *,
    current_target: torch.Tensor,
    prev_target: Optional[torch.Tensor],
    history_targets: list[torch.Tensor],
) -> None:
    """Store differentiable autoregressive readout history for rollout training."""
    if state is None:
        return
    aux = {} if state.aux is None else dict(state.aux)
    aux["ift_state_observed_target"] = current_target
    if prev_target is None:
        aux.pop("ift_prev_observed_target", None)
    else:
        aux["ift_prev_observed_target"] = prev_target
    aux["ift_readout_history_scalar"] = torch.stack(history_targets, dim=0)
    state.aux = aux


def _supports_rollout_training(model, cfg: TrainConfig, edge_targets) -> bool:
    return bool(
        int(getattr(cfg, "rollout_train_steps", 1)) > 1
        and edge_targets is not None
        and getattr(cfg, "edge_target_type", "regression") == "regression"
        and isinstance(getattr(model, "update", None), IFTSecondOrderUpdate)
        and getattr(model.update, "readout_mode", "default") == "linear_h_v_force"
    )


def _supports_physical_rollout_training(model, cfg: TrainConfig, edge_targets) -> bool:
    """Physical event rollouts have force events, not revealed edge targets."""
    return bool(
        int(getattr(cfg, "rollout_train_steps", 1)) > 1
        and
        edge_targets is None
        and callable(getattr(model, "predict_event_features", None))
    )


def _train_one_epoch_physical_rollout(
    model,
    bins: Iterable[EventBatch],
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
) -> dict[str, float]:
    """Differentiable truncated force rollout with oracle pair queries.

    Each chunk begins from a no-gradient, teacher-forced history.  It then
    predicts the next ``K`` force bins, consumes its generated force events,
    and accumulates the ordinary ranking-plus-force objective at every step.
    Source/destination query schedules remain observed evaluation structure;
    this is not an event-count/source generator.
    """
    model.train()
    device = torch.device(cfg.device)
    events_seq = [batch.to(device) for batch in bins]
    horizon = int(cfg.rollout_train_steps)
    if len(events_seq) < 2:
        return {"loss": 0.0, "steps": 0}

    episode_ranges: list[tuple[int, int]] = []
    begin = 0
    while begin < len(events_seq):
        episode = None if events_seq[begin].episode is None else int(events_seq[begin].episode[0].item())
        end = begin + 1
        while end < len(events_seq):
            candidate_episode = None if events_seq[end].episode is None else int(events_seq[end].episode[0].item())
            if candidate_episode != episode:
                break
            end += 1
        episode_ranges.append((begin, end))
        begin = end

    loss_sum = 0.0
    metric_sums: dict[str, float] = {}
    metric_counts: dict[str, int] = {}
    updates = 0
    prediction_steps = 0
    for episode_begin, episode_end in episode_ranges:
        # Chunks are independent after their observed burn-in.  This is a
        # truncated rollout objective, not backpropagation through an entire
        # episode, and keeps memory proportional to the requested horizon.
        for start_idx in range(episode_begin, episode_end - 1, horizon):
            end_idx = min(start_idx + horizon, episode_end - 1)
            optimizer.zero_grad(set_to_none=True)
            state = model.init_state(batch_size=1, num_nodes=cfg.num_nodes, device=device)
            with torch.no_grad():
                for observed_idx in range(episode_begin, start_idx + 1):
                    state, _ = model.step(state, events_seq[observed_idx])
                if state is not None:
                    state.detach_()

            step_losses: list[torch.Tensor] = []
            for target_idx in range(start_idx + 1, end_idx + 1):
                target = events_seq[target_idx]
                if target.features is None:
                    continue
                loss, metrics = ranking_loss_and_metrics(
                    model=model,
                    state=state,
                    next_events=target,
                    num_nodes=cfg.num_nodes,
                    num_neg=cfg.num_neg,
                )
                step_losses.append(loss)
                for key, value in metrics.items():
                    metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
                    metric_counts[key] = metric_counts.get(key, 0) + 1
                prediction_steps += 1

                query = EventBatch(
                    src=target.src,
                    dst=target.dst,
                    t=target.t,
                    episode=target.episode,
                    is_external=target.is_external,
                )
                predicted_force = model.predict_event_features(state, query)
                # Raindrops are observed interventions: carry their true force
                # into the recurrent state while keeping only endogenous force
                # responses in the loss above.
                generated_features = predicted_force.clone()
                if target.is_external is not None and bool(target.is_external.any()):
                    external = target.is_external.to(dtype=torch.bool)
                    generated_features[external] = target.features[external]
                generated = EventBatch(
                    src=target.src,
                    dst=target.dst,
                    features=generated_features,
                    t=target.t,
                    episode=target.episode,
                    is_external=target.is_external,
                )
                state, _ = model.step(state, generated)

            if not step_losses:
                continue
            chunk_loss = torch.stack(step_losses).mean()
            # A sufficiently high observation threshold can make an entire
            # chunk externally driven or quiet. There is then no endogenous
            # supervision and consequently no graph-connected loss to
            # differentiate; simply advance to the next chunk.
            if not chunk_loss.requires_grad:
                continue
            chunk_loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            loss_sum += float(chunk_loss.detach().item())
            updates += 1

    out = {"loss": loss_sum / max(updates, 1), "steps": prediction_steps}
    for key, value in metric_sums.items():
        out[key] = value / metric_counts[key]
    return out


def _train_one_epoch_rollout(
    model,
    bins: Iterable[EventBatch],
    edge_targets: Iterable[EdgeTargetBatch],
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
) -> dict[str, float]:
    """Truncated differentiable autoregressive training for second-order IFT.

    Each optimizer update unrolls ``rollout_train_steps`` bins. Readout history
    uses the model's previous raw prediction after the first step. When the
    self-rollout flag is active, those same predictions regenerate lattice
    event signals and future drive events are removed.
    """
    model.train()
    device = torch.device(cfg.device)
    events_seq = [batch.to(device) for batch in bins]
    target_seq = [
        EdgeTargetBatch(events=batch.events.to(device), targets=batch.targets.to(device))
        for batch in edge_targets
    ]
    if len(events_seq) != len(target_seq):
        raise ValueError("Rollout training requires one edge-target batch per event bin.")
    if len(events_seq) < 2:
        return {"loss": 0.0, "steps": 0}

    horizon = int(cfg.rollout_train_steps)
    self_generated = bool(getattr(cfg, "ift_rollout_self_generated", False))
    free_drive = bool(getattr(cfg, "ift_rollout_free_drive", False))
    free_train_percent = float(getattr(cfg, "synthetic_free_train_percent", 0.0))
    free_train_cutoff = int(getattr(cfg, "synthetic_free_train_cutoff", 1))
    field_dynamics = getattr(cfg, "synthetic_field_dynamics", None)
    field_topology = str(getattr(cfg, "synthetic_field_topology", "ring"))
    state = model.init_state(batch_size=1, num_nodes=cfg.num_nodes, device=device)
    loss_sum = 0.0
    metric_sums: dict[str, float] = {}
    metric_counts: dict[str, int] = {}
    updates = 0
    prediction_steps = 0

    # At chunk i, event bin i - 1 advances the state and target i is scored.
    chunk_starts = list(range(1, len(events_seq), horizon))
    # Second-order counterfactuals need y_{t-1}; the first chunk has only one
    # observed target and is therefore always driven.
    eligible_free_indices = list(range(1, len(chunk_starts)))
    free_chunk_count = round(len(eligible_free_indices) * free_train_percent / 100.0)
    # Even spacing makes the requested fraction exact per epoch while avoiding
    # a front-loaded curriculum artefact.
    free_chunk_indices = set(
        [eligible_free_indices[index] for index in np.linspace(0, len(eligible_free_indices) - 1, num=free_chunk_count, dtype=int)]
    ) if free_chunk_count else set()
    for chunk_number, chunk_start in enumerate(chunk_starts):
        chunk_end = min(chunk_start + horizon, len(events_seq))
        if state is not None:
            state.detach_()
        optimizer.zero_grad(set_to_none=True)

        previous = target_seq[chunk_start - 1].targets
        two_back = target_seq[chunk_start - 2].targets if chunk_start >= 2 else None
        three_back = target_seq[chunk_start - 3].targets if chunk_start >= 3 else None
        free_chunk = chunk_number in free_chunk_indices
        counterfactual_targets = None
        if free_chunk:
            if field_dynamics not in {"diffusion", "wave", "coupled_oscillator"}:
                raise ValueError("Free-response rollout training requires a supported synthetic field dynamic.")
            counterfactual_targets = _field_counterfactual_targets(
                events_seq,
                start_idx=chunk_start - 1,
                horizon=chunk_end - chunk_start,
                cutoff=free_train_cutoff,
                initial_field=previous,
                previous_field=two_back,
                dynamics=field_dynamics,
                topology=field_topology,
            )
        step_losses: list[torch.Tensor] = []

        for idx in range(chunk_start, chunk_end):
            step_events = events_seq[idx - 1]
            relative_step = idx - chunk_start + 1
            if free_chunk and relative_step > free_train_cutoff:
                step_events = _counterfactual_field_events(
                    step_events, predicted_field=previous, drive_enabled=False
                )
            elif self_generated:
                step_events = _self_generate_grid_wave_events(step_events, predicted_field=previous)
            elif free_drive:
                step_events = _remove_wave_drive_events(step_events)

            history = [target for target in (three_back, two_back, previous) if target is not None]
            _stash_rollout_history(
                state,
                current_target=previous,
                prev_target=two_back,
                history_targets=history,
            )
            state, _ = model.step(state, step_events)
            edge_target = target_seq[idx]
            if free_chunk and relative_step > free_train_cutoff:
                assert counterfactual_targets is not None
                edge_target = EdgeTargetBatch(events=edge_target.events, targets=counterfactual_targets[relative_step])
            edge_preds = model.score(state, edge_target.events)
            edge_view = edge_regression_readout(edge_preds, edge_target.targets, previous, cfg)
            loss = edge_regression_loss(edge_view.loss_preds, edge_view.loss_targets, cfg)
            step_losses.append(loss)

            metrics = edge_regression_metrics(edge_view.raw_preds.detach(), edge_target.targets.detach())
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
                metric_counts[key] = metric_counts.get(key, 0) + 1
            prediction_steps += 1

            three_back, two_back, previous = two_back, previous, edge_view.raw_preds

        if not step_losses:
            continue
        chunk_loss = torch.stack(step_losses).mean()
        chunk_loss.backward()
        if cfg.grad_clip and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        if state is not None:
            state.detach_()
        loss_sum += float(chunk_loss.detach().item())
        updates += 1

    out = {"loss": loss_sum / max(updates, 1), "steps": prediction_steps}
    for key, value in metric_sums.items():
        out[key] = value / metric_counts[key]
    return out


def train_one_epoch(
    model,
    bins: Iterable[EventBatch],
    node_targets: Optional[Iterable[torch.Tensor]],
    edge_targets: Optional[Iterable[EdgeTargetBatch]],
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
) -> dict[str, float]:
    if _supports_rollout_training(model, cfg, edge_targets):
        assert edge_targets is not None
        return _train_one_epoch_rollout(model, bins, edge_targets, optimizer, cfg)
    if _supports_physical_rollout_training(model, cfg, edge_targets):
        return _train_one_epoch_physical_rollout(model, bins, optimizer, cfg)
    # Physical episodes are independent trajectories. Pack equal local-time
    # bins into a disjoint node space so the GPU sees useful tensor widths
    # instead of ten tiny sequential launches. Non-physical and legacy paths
    # retain their original stream representation.
    bins = list(bins)
    if edge_targets is None and node_targets is None and bins and bins[0].features is not None:
        bins = pack_independent_episode_bins(bins, num_nodes=cfg.num_nodes)
    model.train()
    device = torch.device(cfg.device)
    batch_size = 1
    if bins and bins[0].batch is not None:
        batch_size = int(bins[0].batch.max().item()) + 1
    state = model.init_state(batch_size=batch_size, num_nodes=cfg.num_nodes, device=device)

    total_loss = 0.0
    total_primary = 0.0
    total_primary_count = 0
    n_steps = 0
    kappa_sum = 0.0
    kappa_n = 0
    grad_norm_sum = 0.0
    state_std_sum = 0.0
    state_abs_sum = 0.0
    state_delta_sum = 0.0
    state_stat_n = 0
    state_delta_n = 0
    aux_sums: dict[str, float] = {}
    aux_counts: dict[str, int] = {}
    tracked_metric_keys = {
        "delta_pred_norm",
        "true_delta_norm",
        "pred_delta_corr",
        "velocity_loss",
        "decoded_v_r2_against_finite_difference",
        "used_velocity_mse",
        "used_velocity_r2",
        "internal_velocity_loss",
        "internal_velocity_mse",
        "internal_velocity_r2",
    }

    prev: Optional[EventBatch] = None
    target_iter = iter(node_targets) if node_targets is not None else None
    edge_target_iter = iter(edge_targets) if edge_targets is not None else None
    prev_node_target: Optional[torch.Tensor] = None
    prev_prev_prev_node_target: Optional[torch.Tensor] = None
    prev_prev_node_target: Optional[torch.Tensor] = None
    prev_edge_target: Optional[torch.Tensor] = None
    prev_prev_prev_edge_target: Optional[torch.Tensor] = None
    prev_prev_edge_target: Optional[torch.Tensor] = None
    primary_name: Optional[str] = None
    for curr in bins:
        curr = curr.to(device)
        curr_node_target = None if target_iter is None else next(target_iter).to(device)
        curr_edge_target = None if edge_target_iter is None else next(edge_target_iter)
        episode_changed = (
            curr.batch is None
            and
            prev is not None
            and prev.episode is not None
            and curr.episode is not None
            and int(prev.episode[0].item()) != int(curr.episode[0].item())
        )
        if prev is None or episode_changed:
            if episode_changed:
                # Episodes are independent physical trajectories.  Their ID
                # is sequence bookkeeping only; it is never passed to model.step.
                state = model.init_state(batch_size=1, num_nodes=cfg.num_nodes, device=device)
                prev_node_target = None
                prev_prev_node_target = None
                prev_prev_prev_node_target = None
                prev_edge_target = None
                prev_prev_edge_target = None
                prev_prev_prev_edge_target = None
            prev = curr
            if curr_node_target is not None:
                prev_node_target = curr_node_target.detach().to(device)
            if curr_edge_target is not None:
                prev_edge_target = curr_edge_target.targets.detach().to(device)
            continue

        state_before = None if state is None or state.node is None else state.node.detach().clone()
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
        state, aux = model.step(state, prev)

        if aux is not None and "kappa" in aux:
            kappa = aux["kappa"]
            if torch.is_tensor(kappa):
                kappa_sum += float(kappa.detach().item())
                kappa_n += 1
        if aux is not None:
            for key, value in aux.items():
                scalar: float | None = None
                if torch.is_tensor(value):
                    if value.numel() == 1 and torch.isfinite(value).all():
                        scalar = float(value.detach().item())
                elif isinstance(value, (int, float, bool)):
                    scalar = float(value)
                if scalar is None:
                    continue
                aux_sums[key] = aux_sums.get(key, 0.0) + scalar
                aux_counts[key] = aux_counts.get(key, 0) + 1

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

        if curr.batch is None:
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
                edge_view = edge_regression_readout(edge_preds, raw_edge_target_values, prev_edge_target, cfg)
                loss = edge_regression_loss(edge_view.loss_preds, edge_view.loss_targets, cfg)
                raw_edge_preds = edge_view.raw_preds.detach()
                metrics = edge_regression_metrics(raw_edge_preds, raw_edge_target_values)
                if edge_view.delta_preds is not None and edge_view.delta_targets is not None:
                    delta_metrics = regression_metrics(
                        edge_view.delta_preds.detach(),
                        edge_view.delta_targets.detach(),
                        prefix="edge_delta",
                    )
                    metrics.update(delta_metrics)
                    metrics["delta_pred_norm"] = float(
                        edge_view.delta_preds.detach().pow(2).mean().sqrt().item()
                    )
                    metrics["true_delta_norm"] = float(
                        edge_view.delta_targets.detach().pow(2).mean().sqrt().item()
                    )
                    metrics["pred_delta_corr"] = float(delta_metrics.get("edge_delta_corr", 0.0))
                if cfg.edge_target_mode != "raw" and str(getattr(cfg, "prediction_mode", "state")) == "state":
                    resid_metrics = edge_regression_metrics(edge_preds.detach(), edge_view.loss_targets)
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
                node_view = node_regression_readout(node_logits, raw_node_target, prev_node_target, cfg)
                loss = torch.nn.functional.mse_loss(node_view.loss_preds, node_view.loss_targets)
                raw_node_preds = node_view.raw_preds.detach()
                metrics = node_regression_metrics(raw_node_preds, raw_node_target)
                if cfg.node_target_mode != "raw" and str(getattr(cfg, "prediction_mode", "state")) == "state":
                    resid_node_metrics = node_regression_metrics(node_logits.detach(), node_view.loss_targets)
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
                    node_view = node_regression_readout(node_logits, raw_node_target, prev_node_target, cfg)
                    node_loss = torch.nn.functional.mse_loss(node_view.loss_preds, node_view.loss_targets)
                    raw_node_preds = node_view.raw_preds.detach()
                    metrics.update(node_regression_metrics(raw_node_preds, raw_node_target))
                    if cfg.node_target_mode != "raw" and str(getattr(cfg, "prediction_mode", "state")) == "state":
                        resid_node_metrics = node_regression_metrics(node_logits.detach(), node_view.loss_targets)
                        metrics.update({f"node_resid_{k.removeprefix('node_')}": v for k, v in resid_node_metrics.items()})
            else:
                node_labels = node_labels_from_events(curr, cfg.num_nodes, device=node_logits.device)
                node_loss = torch.nn.functional.binary_cross_entropy_with_logits(node_logits, node_labels)
                metrics.update(node_prediction_metrics(node_logits.detach(), node_labels))
            metrics["node_loss"] = float(node_loss.detach().item())
            total_step_loss = total_step_loss + (cfg.node_loss_weight * node_loss)

        if (
            curr_edge_target is not None
            and prev_edge_target is not None
            and aux is not None
            and bool(getattr(model.update, "velocity_supervision", False))
        ):
            decoded_velocity = aux.get("decoded_velocity")
            if torch.is_tensor(decoded_velocity):
                velocity_target = (raw_edge_target_values - prev_edge_target).to(
                    device=decoded_velocity.device,
                    dtype=decoded_velocity.dtype,
                )
                velocity_loss = torch.nn.functional.mse_loss(decoded_velocity, velocity_target)
                total_step_loss = total_step_loss + (
                    float(getattr(model.update, "velocity_loss_weight", 0.0)) * velocity_loss
                )
                velocity_metrics = regression_metrics(
                    decoded_velocity.detach(),
                    velocity_target.detach(),
                    prefix="decoded_v",
                )
                metrics["velocity_loss"] = float(velocity_loss.detach().item())
                metrics["decoded_v_r2_against_finite_difference"] = float(
                    velocity_metrics.get("decoded_v_r2", 0.0)
                )
        if (
            curr_edge_target is not None
            and prev_edge_target is not None
            and aux is not None
        ):
            stored_velocity = None
            if state is not None and state.aux is not None:
                stored_velocity = state.aux.get("ift_readout_velocity_scalar")
            if not torch.is_tensor(stored_velocity):
                stored_velocity = aux.get("stored_velocity_scalar")
            true_velocity = aux.get("true_velocity_scalar")
            if torch.is_tensor(stored_velocity) and torch.is_tensor(true_velocity):
                used_velocity_target = true_velocity.to(
                    device=stored_velocity.device,
                    dtype=stored_velocity.dtype,
                )
                used_velocity_mse = torch.nn.functional.mse_loss(
                    stored_velocity,
                    used_velocity_target,
                )
                used_velocity_metrics = regression_metrics(
                    stored_velocity.detach(),
                    used_velocity_target.detach(),
                    prefix="used_velocity",
                )
                metrics["used_velocity_mse"] = float(used_velocity_mse.detach().item())
                metrics["used_velocity_r2"] = float(
                    used_velocity_metrics.get("used_velocity_r2", 0.0)
                )
        if (
            curr_edge_target is not None
            and prev_edge_target is not None
            and aux is not None
            and float(getattr(model.update, "internal_velocity_loss_weight", 0.0)) > 0.0
        ):
            stored_velocity = aux.get("internal_velocity_scalar")
            if not torch.is_tensor(stored_velocity):
                stored_velocity = aux.get("stored_velocity_scalar")
            true_velocity = aux.get("true_velocity_scalar")
            if torch.is_tensor(stored_velocity) and torch.is_tensor(true_velocity):
                internal_velocity_loss = torch.nn.functional.mse_loss(
                    stored_velocity,
                    true_velocity.to(device=stored_velocity.device, dtype=stored_velocity.dtype),
                )
                total_step_loss = total_step_loss + (
                    float(getattr(model.update, "internal_velocity_loss_weight", 0.0))
                    * internal_velocity_loss
                )
                internal_velocity_metrics = regression_metrics(
                    stored_velocity.detach(),
                    true_velocity.detach().to(device=stored_velocity.device, dtype=stored_velocity.dtype),
                    prefix="internal_velocity",
                )
                metrics["internal_velocity_loss"] = float(internal_velocity_loss.detach().item())
                metrics["internal_velocity_mse"] = float(internal_velocity_loss.detach().item())
                metrics["internal_velocity_r2"] = float(
                    internal_velocity_metrics.get("internal_velocity_r2", 0.0)
                )

        for key in tracked_metric_keys:
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                aux_sums[key] = aux_sums.get(key, 0.0) + float(value)
                aux_counts[key] = aux_counts.get(key, 0) + 1

        total_step_loss.backward()
        if cfg.grad_clip and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        grad_norm_sum += global_grad_norm(model.parameters())
        optimizer.step()

        if cfg.tbptt_steps and ((n_steps + 1) % cfg.tbptt_steps == 0) and state is not None:
            state.detach_()

        total_loss += float(total_step_loss.item())
        # Thresholded physical streams legitimately contain quiet bins with
        # only an observed external raindrop (or no events). Their ranking
        # loss is a valid zero connected to the model state, but they have no
        # positive interaction and therefore no MRR/AUC to aggregate.
        if metrics:
            step_primary_name = primary_metric_name(metrics)
            if step_primary_name in metrics:
                primary_name = step_primary_name
                total_primary += float(metrics[step_primary_name])
                total_primary_count += 1
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
            prev_prev_prev_node_target = prev_prev_node_target
            prev_prev_node_target = prev_node_target
            prev_node_target = curr_node_target.detach().to(device)
        if curr_edge_target is not None:
            prev_prev_prev_edge_target = prev_prev_edge_target
            prev_prev_edge_target = prev_edge_target
            prev_edge_target = raw_edge_target_values.detach()

    if n_steps == 0:
        return {"loss": 0.0}

    if primary_name is None:
        primary_name = "mrr"
    out = {
        "loss": total_loss / n_steps,
        primary_name: total_primary / max(total_primary_count, 1),
    }
    if kappa_n > 0:
        out["kappa_mean"] = kappa_sum / kappa_n
    if n_steps > 0:
        out["grad_norm_mean"] = grad_norm_sum / n_steps
    if state_stat_n > 0:
        out["state_node_std_mean"] = state_std_sum / state_stat_n
        out["state_node_abs_mean"] = state_abs_sum / state_stat_n
    if state_delta_n > 0:
        out["state_delta_rms_mean"] = state_delta_sum / state_delta_n
    for key, total in aux_sums.items():
        count = aux_counts.get(key, 0)
        if count > 0:
            out[f"{key}_mean"] = total / count
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
    for key, value in vars(base_train_cfg).items():
        if not hasattr(train_cfg, key):
            setattr(train_cfg, key, value)
    setattr(train_cfg, "ift_drive_feature_idx", getattr(run.model_cfg, "ift_drive_feature_idx", None))
    setattr(train_cfg, "ift_rollout_trace_print", run.name == "ift2_ar2_oracle_init")
    setattr(train_cfg, "ift_rollout_trace_done", False)
    setattr(
        train_cfg,
        "ift_rollout_autonomous",
        bool(
            getattr(run.model_cfg, "update", None) == "ift_update"
            and getattr(run.model_cfg, "ift_update_order", "first") == "second"
            and getattr(run.model_cfg, "ift2_readout_mode", "default") == "linear_h_v_force"
            and getattr(run.model_cfg, "ift_velocity_init_mode", "finite_difference") == "finite_difference"
        ),
    )
    setattr(
        train_cfg,
        "ift_rollout_self_generated",
        bool(getattr(run.model_cfg, "ift_rollout_self_generated", False)),
    )
    setattr(
        train_cfg,
        "ift_rollout_free_drive",
        bool(getattr(run.model_cfg, "ift_rollout_free_drive", False)),
    )
    if run.lr is not None:
        train_cfg.lr = run.lr
    if run.weight_decay is not None:
        train_cfg.weight_decay = run.weight_decay
    if run.num_neg is not None:
        train_cfg.num_neg = run.num_neg
    if run.tbptt_steps is not None:
        train_cfg.tbptt_steps = run.tbptt_steps
    if run.prediction_mode is not None:
        train_cfg.prediction_mode = run.prediction_mode

    model = build_model_fn(spec, run.model_cfg).to(device)
    # Calibrate force supervision exactly once from *training* target forces.
    # External impulses start episodes and are not next internal interactions,
    # so exclude them from the target distribution.  The calibration is used
    # only by the loss, never passed to a model as an observation.
    configure_force_objective = getattr(model, "configure_event_feature_objective", None)
    if callable(configure_force_objective) and bool(getattr(run.model_cfg, "predict_event_features", False) or getattr(run.model_cfg, "fnn", False)):
        train_forces = []
        for batch in ds.bins("train"):
            if batch.features is None or batch.num_events == 0:
                continue
            keep = torch.ones(batch.num_events, dtype=torch.bool)
            if batch.is_external is not None:
                keep &= ~batch.is_external.detach().cpu()
            if bool(keep.any()):
                train_forces.append(batch.features.detach().cpu()[keep])
        if train_forces:
            forces = torch.cat(train_forces, dim=0).float()
            raw_target_std = forces.std(dim=0, unbiased=False)
            # Some physical tasks intentionally excite only a subspace (for
            # example a vertical raindrop).  A zero-variance channel cannot
            # define its own z-score; give it the typical *nonzero* channel
            # scale so decoder noise is penalized without exploding the loss.
            nonzero_scales = raw_target_std[raw_target_std > 1e-8]
            fallback_scale = (
                nonzero_scales.median()
                if nonzero_scales.numel() > 0
                else torch.tensor(1.0, dtype=raw_target_std.dtype)
            )
            target_std = torch.where(
                raw_target_std > 1e-8,
                raw_target_std,
                fallback_scale.expand_as(raw_target_std),
            ).clamp_min(1e-8)
            magnitudes = forces.norm(dim=-1)
            active_threshold = float(torch.quantile(magnitudes, 0.75).item())
            magnitude_q90 = float(torch.quantile(magnitudes, 0.90).item())
            configure_force_objective(
                target_std=target_std,
                active_threshold=active_threshold,
                magnitude_q90=magnitude_q90,
                magnitude_weight=float(getattr(run.model_cfg, "event_feature_magnitude_weight", 2.0)),
            )
            print(
                "  force supervision"
                f" | normalized by train std={target_std.tolist()}"
                f" | active>=q75={active_threshold:.4g}"
                f" | magnitude q90={magnitude_q90:.4g}"
                f" | large-force weight={getattr(model, 'event_feature_magnitude_weight', 2.0):.3g}"
            )
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

    snapshot: dict[str, Any] = {
        "epoch": 0,
        "train_step": {},
        "train_eval": {},
        "val": {},
        "test": {},
        "rollout_val": {},
        "rollout_test": {},
        "readout": _linear_hvf_readout_snapshot(model),
    }
    task_axes = {} if spec.extra is None else dict(spec.extra.get("task_axes", {}))
    generator_params = task_axes.get("generator_params") or {}
    field_dynamics = task_axes.get("dynamics_type")
    field_topology = str(generator_params.get("topology", "ring"))
    setattr(train_cfg, "synthetic_field_dynamics", field_dynamics)
    setattr(train_cfg, "synthetic_field_topology", field_topology)
    hidden_truth_fn = getattr(ds, "hidden_truth", None)
    hidden_truth = hidden_truth_fn() if callable(hidden_truth_fn) else None
    baseline_printed = False
    for epoch in range(1, epochs + 1):
        timing_sec: dict[str, float] = {}
        timing_enabled = bool(getattr(train_cfg, "debug_timing", False))

        def timed(label: str, fn):
            if timing_enabled and device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            value = fn()
            if timing_enabled and device.type == "cuda":
                torch.cuda.synchronize(device)
            if timing_enabled:
                timing_sec[label] = time.perf_counter() - started
            return value

        epoch_started = time.perf_counter()
        train_stats_step = timed(
            "train",
            lambda: train_one_epoch(
                model,
                ds.bins("train"),
                train_node_targets,
                train_edge_targets,
                optimizer,
                train_cfg,
            ),
        )
        parameter_trace = _parameter_recovery_snapshot(model, hidden_truth, device)
        eval_every = train_cfg.eval_every
        eval_due = epoch == epochs or (
            eval_every is not None and epoch % eval_every == 0
        )
        if not eval_due:
            print(f"  ep {epoch:03d} | train loss={train_stats_step['loss']:.4f}")
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
                    "epoch": epoch,
                    "train_step": train_stats_step,
                    "parameter_trace": parameter_trace,
                }
                with open(save_jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
            continue

        if not baseline_printed:
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
                baseline_str = (
                    "  baseline (no-update)"
                    f" | val {format_edge_metric_bundle(baseline_val, prefix='persistent')}"
                    f" | test {format_edge_metric_bundle(baseline_test, prefix='persistent')}"
                )
            elif run_primary_name == "edge_auroc":
                baseline_str = (
                    "  baseline (no-update)"
                    f" | val {format_edge_classification_bundle(baseline_val, prefix='persistent')}"
                    f" | test {format_edge_classification_bundle(baseline_test, prefix='persistent')}"
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
            baseline_printed = True
        train_eval = timed(
            "train_eval",
            lambda: evaluate_stream_sliced(
                model,
                ds.bins("train"),
                train_node_targets,
                train_edge_targets,
                train_cfg,
                slices=eval_slices,
            ),
        )
        val_stats = timed(
            "val_eval",
            lambda: evaluate_stream_sliced(
                model,
                ds.bins("val"),
                val_node_targets,
                val_edge_targets,
                train_cfg,
                slices=eval_slices,
            ),
        )
        test_stats = timed(
            "test_eval",
            lambda: evaluate_stream_sliced(
                model,
                ds.bins("test"),
                test_node_targets,
                test_edge_targets,
                train_cfg,
                slices=eval_slices,
            ),
        )
        # Keep recovery quantities alongside test metrics so normal JSONL
        # summaries can select them. The same values are traced every epoch.
        recovery_stats = dict(parameter_trace)
        test_stats.update(recovery_stats)
        rollout_val_stats: dict[str, float] = {}
        rollout_test_stats: dict[str, float] = {}
        rollout_intervention_val_stats: dict[str, float] = {}
        rollout_intervention_test_stats: dict[str, float] = {}
        rollout_free_val_stats: dict[str, float] = {}
        rollout_free_test_stats: dict[str, float] = {}
        rollout_self_free_val_stats: dict[str, float] = {}
        rollout_self_free_test_stats: dict[str, float] = {}
        if (
            train_edge_targets is not None and train_cfg.edge_target_type == "regression"
        ) or (
            train_node_targets is not None and train_cfg.node_target_type == "regression"
        ):
            cutoff = getattr(train_cfg, "synthetic_drive_cutoff", None)
            val_rng_state = torch.get_rng_state()
            val_cuda_rng_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
            rollout_val_stats = evaluate_k_step_rollout(
                model,
                ds.bins("val"),
                val_node_targets,
                val_edge_targets,
                train_cfg,
                horizon=rollout_horizon,
                teacher_forced_readout=True,
            )
            if cutoff is not None:
                # Begin the intervention from the identical random latent
                # initialization, so the two trajectories match pre-cutoff.
                torch.set_rng_state(val_rng_state)
                if val_cuda_rng_state is not None:
                    torch.cuda.set_rng_state(val_cuda_rng_state, device)
                rollout_intervention_val_stats = evaluate_k_step_rollout(
                    model,
                    ds.bins("val"),
                    val_node_targets,
                    val_edge_targets,
                    train_cfg,
                    horizon=rollout_horizon,
                    drive_cutoff=int(cutoff),
                    field_dynamics=field_dynamics,
                    field_topology=field_topology,
                    teacher_forced_readout=True,
                )
                if bool(getattr(train_cfg, "synthetic_free_rollout", False)):
                    torch.set_rng_state(val_rng_state)
                    if val_cuda_rng_state is not None:
                        torch.cuda.set_rng_state(val_cuda_rng_state, device)
                    rollout_free_val_stats = evaluate_k_step_rollout(
                        model, ds.bins("val"), val_node_targets, val_edge_targets, train_cfg,
                        horizon=rollout_horizon, drive_cutoff=int(cutoff),
                        field_dynamics=field_dynamics, field_topology=field_topology,
                        teacher_forced_readout=True, post_cutoff_event_mode="oracle",
                    )
                if bool(getattr(train_cfg, "synthetic_self_free_rollout", False)):
                    rollout_self_free_val_stats = rollout_intervention_val_stats
            test_rng_state = torch.get_rng_state()
            test_cuda_rng_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
            rollout_test_stats = evaluate_k_step_rollout(
                model,
                ds.bins("test"),
                test_node_targets,
                test_edge_targets,
                train_cfg,
                horizon=rollout_horizon,
                teacher_forced_readout=True,
            )
            if cutoff is not None:
                torch.set_rng_state(test_rng_state)
                if test_cuda_rng_state is not None:
                    torch.cuda.set_rng_state(test_cuda_rng_state, device)
                rollout_intervention_test_stats = evaluate_k_step_rollout(
                    model,
                    ds.bins("test"),
                    test_node_targets,
                    test_edge_targets,
                    train_cfg,
                    horizon=rollout_horizon,
                    drive_cutoff=int(cutoff),
                    field_dynamics=field_dynamics,
                    field_topology=field_topology,
                    teacher_forced_readout=True,
                )
                if bool(getattr(train_cfg, "synthetic_free_rollout", False)):
                    torch.set_rng_state(test_rng_state)
                    if test_cuda_rng_state is not None:
                        torch.cuda.set_rng_state(test_cuda_rng_state, device)
                    rollout_free_test_stats = evaluate_k_step_rollout(
                        model, ds.bins("test"), test_node_targets, test_edge_targets, train_cfg,
                        horizon=rollout_horizon, drive_cutoff=int(cutoff),
                        field_dynamics=field_dynamics, field_topology=field_topology,
                        teacher_forced_readout=True, post_cutoff_event_mode="oracle",
                    )
                if bool(getattr(train_cfg, "synthetic_self_free_rollout", False)):
                    rollout_self_free_test_stats = rollout_intervention_test_stats
        elif (
            train_edge_targets is None
            and train_node_targets is None
            and callable(getattr(model, "predict_event_features", None))
        ):
            # Event-only physical tasks have no revealed scalar field target.
            # Their rollout is closed-loop in predicted forces while retaining
            # the future pair-query schedule as an explicit oracle condition.
            rollout_val_stats = timed(
                "rollout_val",
                lambda: evaluate_physical_force_rollout(
                    model, ds.bins("val"), train_cfg, horizon=rollout_horizon
                ),
            )
            rollout_test_stats = timed(
                "rollout_test",
                lambda: evaluate_physical_force_rollout(
                    model, ds.bins("test"), train_cfg, horizon=rollout_horizon
                ),
            )

        if timing_enabled:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timing_sec["total"] = time.perf_counter() - epoch_started

        snapshot = {
            "epoch": epoch,
            "train_step": train_stats_step,
            "train_eval": train_eval,
            "val": val_stats,
            "test": test_stats,
            "recovery": recovery_stats,
            "rollout_val": rollout_val_stats,
            "rollout_test": rollout_test_stats,
            "rollout_intervention_val": rollout_intervention_val_stats,
            "rollout_intervention_test": rollout_intervention_test_stats,
            "rollout_free_val": rollout_free_val_stats,
            "rollout_free_test": rollout_free_test_stats,
            "rollout_self_free_val": rollout_self_free_val_stats,
            "rollout_self_free_test": rollout_self_free_test_stats,
            "readout": _linear_hvf_readout_snapshot(model),
            "timing_sec": timing_sec,
            "parameter_trace": parameter_trace,
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
        if timing_sec:
            timing_parts = [f"{key}={value:.2f}s" for key, value in timing_sec.items()]
            print(f"           timing  | {' | '.join(timing_parts)}")
        if "diffusion_term_norm_mean" in train_stats_step:
            print(
                "           ift"
                f"     | kappa={train_stats_step.get('learned_kappa_mean', train_stats_step.get('kappa_mean', float('nan'))):.4f}"
                f" | gamma={train_stats_step.get('gamma_mean', float('nan')):.4f}"
                f" | dt={train_stats_step.get('dt_mean', float('nan')):.4f}"
                f" | alpha={train_stats_step.get('alpha_mean', float('nan')):.4f}"
                f"     | diff={train_stats_step.get('diffusion_term_norm_mean', float('nan')):.4f}"
                f" | force={train_stats_step.get('force_norm_mean', train_stats_step.get('injection_term_norm_mean', float('nan'))):.4f}"
                f" | rel_diff={train_stats_step.get('relative_diffusion_mean', float('nan')):.4f}"
                f" | rel_upd={train_stats_step.get('relative_update_mean', float('nan')):.4f}"
                f" | vel_used_r2={train_stats_step.get('used_velocity_r2_mean', float('nan')):.4f}"
                f" | vel_int_r2={train_stats_step.get('internal_velocity_r2_mean', train_stats_step.get('decoded_v_r2_against_finite_difference_mean', float('nan'))):.4f}"
                f" | vel_used_mse={train_stats_step.get('used_velocity_mse_mean', float('nan')):.4f}"
                f" | vel_int_mse={train_stats_step.get('internal_velocity_mse_mean', train_stats_step.get('velocity_loss_mean', float('nan'))):.4f}"
                f" | vel_f={train_stats_step.get('velocity_fraction_mean', float('nan')):.4f}"
                f" | force_f={train_stats_step.get('force_fraction_mean', float('nan')):.4f}"
                f" | Lnnz={train_stats_step.get('L_nnz_mean', float('nan')):.1f}"
                f" | Ldens={train_stats_step.get('L_density_mean', float('nan')):.4f}"
                f" | Ldiag={train_stats_step.get('L_diag_mean_mean', float('nan')):.4f}"
                f" | Loff={train_stats_step.get('L_offdiag_abs_mean_mean', float('nan')):.4f}"
            )
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
            physical_rollout_val = format_physical_rollout_bundle(rollout_val_stats)
            physical_rollout_test = format_physical_rollout_bundle(rollout_test_stats)
            if physical_rollout_val or physical_rollout_test:
                print(
                    f"           rollout@{rollout_horizon}"
                    f" | val {physical_rollout_val or 'n/a'}"
                    f" | test {physical_rollout_test or 'n/a'}"
                )
        if rollout_intervention_val_stats:
            intervention_val = format_rollout_metric_bundle(rollout_intervention_val_stats, stem="edge")
            intervention_test = format_rollout_metric_bundle(rollout_intervention_test_stats, stem="edge")
            cutoff = getattr(train_cfg, "synthetic_drive_cutoff", None)
            print(
                f"           intervention@{rollout_horizon}"
                f" | drive through step {cutoff}"
                f" | val edge {intervention_val or 'n/a'}"
                f" | test edge {intervention_test or 'n/a'}"
            )
        readout = snapshot.get("readout", {})
        if readout:
            coeff_str = (
                f"           readout | w_y={readout.get('w_y', float('nan')):.4f}"
                f" | w_v={readout.get('w_v', float('nan')):.4f}"
                f" | w_drive={readout.get('w_drive', float('nan')):.4f}"
                f" | bias={readout.get('bias', float('nan')):.4f}"
            )
            if "w_y_oracle" in readout:
                coeff_str += (
                    f" | oracle=({readout['w_y_oracle']:.4f}, {readout['w_v_oracle']:.4f}, "
                    f"{readout['w_drive_oracle']:.4f}, {readout['bias_oracle']:.4f})"
                    f" | abs_diff=({readout['abs_diff_w_y']:.4f}, {readout['abs_diff_w_v']:.4f}, "
                    f"{readout['abs_diff_w_drive']:.4f}, {readout['abs_diff_bias']:.4f})"
                )
            print(coeff_str)

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
