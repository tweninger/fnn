from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Iterable, Optional, cast

import numpy as np
import torch

from interactiondynamics.core.config import ModelConfig
from interactiondynamics.data.interfaces import EdgeTargetBatch
from interactiondynamics.data.synthetic import SYNTHETIC_TASKS
from interactiondynamics.eval.node_metrics import regression_metrics
from interactiondynamics.models.tgn_model import build_tgn_model
from interactiondynamics.training.presets import (
    build_ift_diagnostic_runs,
    build_suite,
    build_synthetic_dataset_config,
    load_dataset,
)
from interactiondynamics.training.reporting import (
    format_node_metric,
    format_node_metric_bundle,
    infer_primary_metric,
    node_metric_name,
)
from interactiondynamics.training.runner import (
    apply_model_overrides,
    run_one_experiment,
    short_run_label,
    short_run_label_from_name,
)
from interactiondynamics.training.task_metrics import (
    is_better_metric,
    metric_key_from_path,
    parse_task_metric_spec,
    snapshot_metric_value,
)
from interactiondynamics.training.types import PredictionMode, RunResult, SweepRun, TrainConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run interaction dynamics training presets.")
    subparsers = parser.add_subparsers(dest="command")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--dataset",
        choices=("toy", "jodie", "synthetic"),
        default=None,
        help="Optional dataset override when supported by the command.",
    )
    common.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap on the number of runs to execute after filtering.",
    )
    common.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for the preset epoch count.",
    )
    common.add_argument(
        "--use-node-scorer",
        action="store_true",
        help="Enable auxiliary node prediction on whether a node appears in the next bin.",
    )
    common.add_argument(
        "--node-loss-weight",
        type=float,
        default=1.0,
        help="Weight for the auxiliary node prediction loss.",
    )
    common.add_argument(
        "--node-scorer-hidden",
        type=int,
        default=128,
        help="Hidden size for the auxiliary node scorer MLP.",
    )
    common.add_argument(
        "--save-jsonl",
        type=str,
        default=None,
        help="Optional path to append per-epoch JSONL results.",
    )
    common.add_argument(
        "--num-bins",
        type=int,
        default=None,
        help="Number of simulated time bins for synthetic datasets.",
    )
    common.add_argument("--seed", type=int, default=0, help="Random seed for simulated datasets.")
    common.add_argument(
        "--synthetic-task",
        choices=tuple(SYNTHETIC_TASKS.keys()),
        default="deepsets_sum",
        help="Synthetic benchmark task to use when --dataset synthetic.",
    )
    common.add_argument(
        "--synthetic-num-nodes",
        type=int,
        default=None,
        help="Optional node count override for synthetic datasets.",
    )
    common.add_argument(
        "--synthetic-events-per-bin",
        type=int,
        default=None,
        help="Optional event-count override for set-based synthetic datasets.",
    )
    common.add_argument(
        "--node-target-mode",
        choices=("raw", "residual"),
        default="raw",
        help="Train the node head on raw next-step node values or residuals relative to the previous step.",
    )
    common.add_argument(
        "--edge-target-mode",
        choices=("raw", "residual"),
        default="raw",
        help="Train the edge head on raw next-step magnitudes or residuals relative to the previous step.",
    )
    common.add_argument(
        "--edge-target-scale",
        choices=("raw", "zscore"),
        default="raw",
        help="Use raw edge MSE or scale edge regression loss by the train-split target std.",
    )
    common.add_argument(
        "--prediction-mode",
        choices=("state", "delta", "state_plus_delta"),
        default="state",
        help="Train regression heads on next-state targets, delta targets, or reconstructed state from predicted deltas.",
    )
    common.add_argument(
        "--rollout-horizon",
        type=int,
        default=5,
        help="Evaluate k-step target rollout with this horizon for regression datasets.",
    )

    subparsers.add_parser(
        "smoke",
        parents=[common],
        help="Run a tiny smoke test preset.",
    )
    subparsers.add_parser(
        "quick",
        parents=[common],
        help="Run the focused shortlist preset.",
    )
    subparsers.add_parser(
        "sweep",
        parents=[common],
        help="Run the original broad sweep.",
    )
    diag = subparsers.add_parser(
        "ift-diagnose",
        parents=[common],
        help="Run focused IFT ablations on synthetic oscillator and diffusion tasks.",
    )
    diag.add_argument(
        "--ift-diagnostic-tasks",
        nargs="*",
        choices=("conservative_oscillator", "ift_diffusion"),
        default=("conservative_oscillator", "ift_diffusion"),
        help="Synthetic tasks to include in the focused IFT diagnostic suite.",
    )

    parser.add_argument(
        "--preset",
        choices=("smoke", "quick", "full"),
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.command is None:
        if args.preset is not None:
            args.command = "sweep" if args.preset == "full" else args.preset
        else:
            parser.error("please provide a subcommand: smoke, quick, or sweep")
    return args


def _base_diagnostic_model_config(event_dim: int) -> ModelConfig:
    return ModelConfig(
        node_dim=128,
        msg_dim=128,
        event_dim=event_dim,
        scorer="mlp",
        scorer_hidden=256,
        aggregator="ift",
        update="ift_update",
        use_time_features=False,
        dropout=0.0,
        scorer_dropout=0.0,
        encoder_hidden=256,
        ift_message_reduce="sum",
        ift_force_reduce="sum",
    )


def _diagnostic_metric(snapshot: dict[str, object], path: str) -> float:
    return float(snapshot_metric_value(snapshot, path))


def _resolve_prediction_mode(raw: str, *, default_state_plus_delta: bool = False) -> PredictionMode:
    if default_state_plus_delta and raw == "state":
        return "state_plus_delta"
    if raw not in {"state", "delta", "state_plus_delta"}:
        raise ValueError(f"Unsupported prediction_mode={raw!r}")
    return cast(PredictionMode, raw)


def _stack_split_edge_targets(edge_targets: Iterable[EdgeTargetBatch]) -> list[torch.Tensor]:
    return [batch.targets.detach().to(device="cpu", dtype=torch.float32) for batch in edge_targets]


def _stack_split_drives(
    bins: Iterable,
    *,
    num_nodes: int,
    drive_feature_idx: int = 0,
) -> list[torch.Tensor]:
    drives: list[torch.Tensor] = []
    for events in bins:
        if events.features is None or events.features.numel() == 0:
            drives.append(torch.zeros((num_nodes,), dtype=torch.float32))
            continue
        feats = events.features.detach().to(device="cpu", dtype=torch.float32)
        src = events.src.detach().to(device="cpu", dtype=torch.long)
        dst = events.dst.detach().to(device="cpu", dtype=torch.long)
        drive = torch.zeros((num_nodes,), dtype=torch.float32)
        values = feats[:, drive_feature_idx]
        if feats.size(1) > drive_feature_idx + 1:
            mask = feats[:, drive_feature_idx + 1]
            if torch.all((mask == 0) | (mask == 1)):
                values = values * mask
        drive.index_add_(0, dst, values)
        self_mask = src == dst
        if torch.any(self_mask):
            drive = torch.zeros((num_nodes,), dtype=torch.float32)
            drive.index_add_(0, dst[self_mask], values[self_mask])
        drives.append(drive)
    return drives


def _fit_ar_coefficients(
    targets: list[torch.Tensor],
    drives: list[torch.Tensor],
    *,
    order: int,
) -> Optional[torch.Tensor]:
    start = max(order, 1)
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for idx in range(start, len(targets)):
        pieces = [targets[idx - 1].reshape(-1, 1)]
        if order >= 2:
            pieces.append(targets[idx - 2].reshape(-1, 1))
        pieces.append(drives[idx - 1].reshape(-1, 1))
        xs.append(torch.cat(pieces, dim=1))
        ys.append(targets[idx].reshape(-1, 1))
    if not xs:
        return None
    design = torch.cat(xs, dim=0)
    response = torch.cat(ys, dim=0)
    solution = torch.linalg.lstsq(design, response).solution.squeeze(-1)
    return solution.to(dtype=torch.float32)


def _fit_delta_linear_coefficients(
    targets: list[torch.Tensor],
    drives: list[torch.Tensor],
) -> Optional[torch.Tensor]:
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for idx in range(1, len(targets)):
        y_t = targets[idx - 1].reshape(-1, 1)
        y_prev = targets[idx - 2].reshape(-1, 1) if idx >= 2 else targets[idx - 1].reshape(-1, 1)
        velocity = y_t - y_prev
        drive = drives[idx - 1].reshape(-1, 1)
        bias = torch.ones_like(y_t)
        xs.append(torch.cat([y_t, velocity, drive, bias], dim=1))
        ys.append((targets[idx] - targets[idx - 1]).reshape(-1, 1))
    if not xs:
        return None
    design = torch.cat(xs, dim=0)
    response = torch.cat(ys, dim=0)
    solution = torch.linalg.lstsq(design, response).solution.squeeze(-1)
    return solution.to(dtype=torch.float32)


def _predict_autoregressive(
    coeffs: torch.Tensor,
    prev_target: torch.Tensor,
    prev_prev_target: Optional[torch.Tensor],
    drive: torch.Tensor,
) -> torch.Tensor:
    out = coeffs[0] * prev_target + coeffs[-1] * drive
    if coeffs.numel() >= 3:
        prev2 = prev_target if prev_prev_target is None else prev_prev_target
        out = out + coeffs[1] * prev2
    return out


def _predict_delta_linear(
    coeffs: torch.Tensor,
    y_t: torch.Tensor,
    y_prev: torch.Tensor,
    drive: torch.Tensor,
) -> torch.Tensor:
    velocity = y_t - y_prev
    return (
        coeffs[0] * y_t
        + coeffs[1] * velocity
        + coeffs[2] * drive
        + coeffs[3]
    )


def _sequence_baseline_metrics(
    targets: list[torch.Tensor],
    drives: list[torch.Tensor],
    *,
    coeffs: torch.Tensor,
    horizon: int,
) -> dict[str, float]:
    order = 2 if coeffs.numel() >= 3 else 1
    if len(targets) <= 1:
        return {}

    step_preds: list[torch.Tensor] = []
    step_truths: list[torch.Tensor] = []
    step_prev: list[torch.Tensor] = []
    for idx in range(1, len(targets)):
        pred = _predict_autoregressive(
            coeffs,
            targets[idx - 1],
            None if idx < 2 or order < 2 else targets[idx - 2],
            drives[idx - 1],
        )
        step_preds.append(pred)
        step_truths.append(targets[idx])
        step_prev.append(targets[idx - 1])

    pred_all = torch.cat(step_preds)
    truth_all = torch.cat(step_truths)
    prev_all = torch.cat(step_prev)
    out = regression_metrics(pred_all, truth_all, prefix="edge")
    out.update(regression_metrics(pred_all - prev_all, truth_all - prev_all, prefix="edge_delta"))

    rollout_preds: list[torch.Tensor] = []
    rollout_truths: list[torch.Tensor] = []
    rollout_prev: list[torch.Tensor] = []
    if order >= 2:
        for start_idx in range(1, len(targets) - horizon):
            prev2 = targets[start_idx - 1]
            prev1 = targets[start_idx]
            curr_pred = prev1
            for idx in range(start_idx + 1, start_idx + horizon + 1):
                curr_pred = _predict_autoregressive(coeffs, prev1, prev2, drives[idx - 1])
                prev2 = prev1
                prev1 = curr_pred
            rollout_preds.append(curr_pred)
            rollout_truths.append(targets[start_idx + horizon])
            rollout_prev.append(targets[start_idx])
    else:
        for start_idx in range(1, len(targets) - horizon + 1):
            prev1 = targets[start_idx - 1]
            prev2 = None
            curr_pred = prev1
            for idx in range(start_idx, start_idx + horizon):
                curr_pred = _predict_autoregressive(coeffs, prev1, prev2, drives[idx - 1])
                prev1 = curr_pred
            rollout_preds.append(curr_pred)
            rollout_truths.append(targets[start_idx + horizon - 1])
            rollout_prev.append(targets[start_idx - 1])
    if rollout_preds:
        roll_pred = torch.cat(rollout_preds)
        roll_truth = torch.cat(rollout_truths)
        roll_prev = torch.cat(rollout_prev)
        out.update({f"rollout_{k}": v for k, v in regression_metrics(roll_pred, roll_truth, prefix="edge").items()})
        out.update(
            {
                f"rollout_{k}": v
                for k, v in regression_metrics(roll_pred - roll_prev, roll_truth - roll_prev, prefix="edge_delta").items()
            }
        )
        persistent = regression_metrics(roll_prev, roll_truth, prefix="edge")
        out.update({f"rollout_persistent_{k}": v for k, v in persistent.items()})
    return out


def _sequence_delta_baseline_metrics(
    targets: list[torch.Tensor],
    drives: list[torch.Tensor],
    *,
    coeffs: torch.Tensor,
    horizon: int,
) -> dict[str, float]:
    if len(targets) <= 1:
        return {}

    step_preds: list[torch.Tensor] = []
    step_truths: list[torch.Tensor] = []
    step_prev: list[torch.Tensor] = []
    for idx in range(1, len(targets)):
        y_prev = targets[idx - 2] if idx >= 2 else targets[idx - 1]
        y_t = targets[idx - 1]
        delta = _predict_delta_linear(coeffs, y_t, y_prev, drives[idx - 1])
        pred = y_t + delta
        step_preds.append(pred)
        step_truths.append(targets[idx])
        step_prev.append(y_t)

    pred_all = torch.cat(step_preds)
    truth_all = torch.cat(step_truths)
    prev_all = torch.cat(step_prev)
    out = regression_metrics(pred_all, truth_all, prefix="edge")
    out.update(regression_metrics(pred_all - prev_all, truth_all - prev_all, prefix="edge_delta"))

    rollout_preds: list[torch.Tensor] = []
    rollout_truths: list[torch.Tensor] = []
    rollout_prev: list[torch.Tensor] = []
    for start_idx in range(1, len(targets) - horizon):
        y_prev = targets[start_idx - 1]
        y_t = targets[start_idx]
        curr = y_t
        prev = y_prev
        for idx in range(start_idx + 1, start_idx + horizon + 1):
            delta = _predict_delta_linear(coeffs, curr, prev, drives[idx - 1])
            y_next = curr + delta
            prev = curr
            curr = y_next
        rollout_preds.append(curr)
        rollout_truths.append(targets[start_idx + horizon])
        rollout_prev.append(targets[start_idx])
    if rollout_preds:
        roll_pred = torch.cat(rollout_preds)
        roll_truth = torch.cat(rollout_truths)
        roll_prev = torch.cat(rollout_prev)
        out.update({f"rollout_{k}": v for k, v in regression_metrics(roll_pred, roll_truth, prefix="edge").items()})
        out.update(
            {
                f"rollout_{k}": v
                for k, v in regression_metrics(roll_pred - roll_prev, roll_truth - roll_prev, prefix="edge_delta").items()
            }
        )
        persistent = regression_metrics(roll_prev, roll_truth, prefix="edge")
        out.update({f"rollout_persistent_{k}": v for k, v in persistent.items()})
    return out


def _diagnostic_baseline_rows(
    ds,
    spec,
    *,
    task_name: str,
    horizon: int,
) -> dict[str, dict[str, float]]:
    if task_name != "conservative_oscillator":
        return {}
    train_targets = _stack_split_edge_targets(ds.edge_targets("train") or [])
    val_targets = _stack_split_edge_targets(ds.edge_targets("val") or [])
    test_targets = _stack_split_edge_targets(ds.edge_targets("test") or [])
    train_drives = _stack_split_drives(ds.bins("train"), num_nodes=spec.num_nodes, drive_feature_idx=0)
    val_drives = _stack_split_drives(ds.bins("val"), num_nodes=spec.num_nodes, drive_feature_idx=0)
    test_drives = _stack_split_drives(ds.bins("test"), num_nodes=spec.num_nodes, drive_feature_idx=0)

    rows: dict[str, dict[str, float]] = {}
    for name, order in (("ar1_baseline", 1), ("ar2_baseline", 2)):
        coeffs = _fit_ar_coefficients(train_targets, train_drives, order=order)
        if coeffs is None:
            continue
        val_metrics = _sequence_baseline_metrics(val_targets, val_drives, coeffs=coeffs, horizon=horizon)
        test_metrics = _sequence_baseline_metrics(test_targets, test_drives, coeffs=coeffs, horizon=horizon)
        rows[name] = {
            "val_r2": float(val_metrics.get("edge_r2", float("nan"))),
            "test_r2": float(test_metrics.get("edge_r2", float("nan"))),
            "rollout_val_rollout_edge_r2": float(val_metrics.get("rollout_edge_r2", float("nan"))),
            "rollout_test_rollout_edge_r2": float(test_metrics.get("rollout_edge_r2", float("nan"))),
            "persistent_edge_r2": float(test_metrics.get("rollout_persistent_edge_r2", float("nan"))),
            "delta_vs_persistent": float(test_metrics.get("rollout_edge_r2", float("nan")) - test_metrics.get("rollout_persistent_edge_r2", float("nan"))),
            "edge_delta_r2": float(test_metrics.get("rollout_edge_delta_r2", float("nan"))),
            "edge_delta_mae": float(test_metrics.get("rollout_edge_delta_mae", float("nan"))),
        }

    generator_params = {}
    if spec.extra is not None:
        generator_params = dict(spec.extra.get("generator_params") or {})
    if generator_params:
        coeffs = torch.tensor(
            [float(generator_params["a"]), float(generator_params["b"]), float(generator_params["c"])],
            dtype=torch.float32,
        )
        val_metrics = _sequence_baseline_metrics(val_targets, val_drives, coeffs=coeffs, horizon=horizon)
        test_metrics = _sequence_baseline_metrics(test_targets, test_drives, coeffs=coeffs, horizon=horizon)
        rows["oracle_baseline"] = {
            "val_r2": float(val_metrics.get("edge_r2", float("nan"))),
            "test_r2": float(test_metrics.get("edge_r2", float("nan"))),
            "rollout_val_rollout_edge_r2": float(val_metrics.get("rollout_edge_r2", float("nan"))),
            "rollout_test_rollout_edge_r2": float(test_metrics.get("rollout_edge_r2", float("nan"))),
            "persistent_edge_r2": float(test_metrics.get("rollout_persistent_edge_r2", float("nan"))),
            "delta_vs_persistent": float(test_metrics.get("rollout_edge_r2", float("nan")) - test_metrics.get("rollout_persistent_edge_r2", float("nan"))),
            "edge_delta_r2": float(test_metrics.get("rollout_edge_delta_r2", float("nan"))),
            "edge_delta_mae": float(test_metrics.get("rollout_edge_delta_mae", float("nan"))),
        }
    delta_coeffs = _fit_delta_linear_coefficients(train_targets, train_drives)
    if delta_coeffs is not None:
        val_metrics = _sequence_delta_baseline_metrics(val_targets, val_drives, coeffs=delta_coeffs, horizon=horizon)
        test_metrics = _sequence_delta_baseline_metrics(test_targets, test_drives, coeffs=delta_coeffs, horizon=horizon)
        rows["closed_form_delta"] = {
            "val_r2": float(val_metrics.get("edge_r2", float("nan"))),
            "test_r2": float(test_metrics.get("edge_r2", float("nan"))),
            "rollout_val_rollout_edge_r2": float(val_metrics.get("rollout_edge_r2", float("nan"))),
            "rollout_test_rollout_edge_r2": float(test_metrics.get("rollout_edge_r2", float("nan"))),
            "persistent_edge_r2": float(test_metrics.get("rollout_persistent_edge_r2", float("nan"))),
            "delta_vs_persistent": float(test_metrics.get("rollout_edge_r2", float("nan")) - test_metrics.get("rollout_persistent_edge_r2", float("nan"))),
            "edge_delta_r2": float(test_metrics.get("rollout_edge_delta_r2", float("nan"))),
            "edge_delta_mae": float(test_metrics.get("rollout_edge_delta_mae", float("nan"))),
            "w_y": float(delta_coeffs[0].item()),
            "w_v": float(delta_coeffs[1].item()),
            "w_drive": float(delta_coeffs[2].item()),
            "bias": float(delta_coeffs[3].item()),
        }
    return rows


def _print_ift_diagnostic_table(
    task_name: str,
    results: list[RunResult],
    *,
    extra_rows: Optional[dict[str, dict[str, float]]] = None,
) -> dict[str, dict[str, float]]:
    print(f"\n=== IFT diagnostic table: {task_name} ===")
    header = (
        f"{'run':<20} {'state_v':>8} {'state_t':>8} {'roll_v':>8} {'roll_t':>8} {'pers':>8} "
        f"{'d_pers':>8} {'delta_r2':>8} {'delta_mae':>9} {'vel_r2':>8} {'kappa':>7} {'gamma':>7} {'dt':>6} "
        f"{'alpha':>7} {'force':>8} {'diff':>8} {'rel_d':>8} {'rel_u':>8} {'vel_f':>7} {'for_f':>7} {'d_corr':>8}"
    )
    print(header)
    print("-" * len(header))

    rows: dict[str, dict[str, float]] = {}
    for result in results:
        snap = result.best_snapshot if result.best_snapshot else result.final_snapshot
        train_step = snap.get("train_step", {})
        readout = snap.get("readout", {})
        row = {
            "val_r2": _diagnostic_metric(snap, "val.edge_r2"),
            "test_r2": _diagnostic_metric(snap, "test.edge_r2"),
            "rollout_val_rollout_edge_r2": _diagnostic_metric(snap, "rollout_val.rollout_edge_r2"),
            "rollout_test_rollout_edge_r2": _diagnostic_metric(snap, "rollout_test.rollout_edge_r2"),
            "persistent_edge_r2": _diagnostic_metric(snap, "rollout_test.rollout_persistent_edge_r2"),
            "delta_vs_persistent": (
                _diagnostic_metric(snap, "rollout_test.rollout_edge_r2")
                - _diagnostic_metric(snap, "rollout_test.rollout_persistent_edge_r2")
            ),
            "edge_delta_r2": _diagnostic_metric(snap, "rollout_test.rollout_edge_delta_r2"),
            "edge_delta_mae": _diagnostic_metric(snap, "rollout_test.rollout_edge_delta_mae"),
            "decoded_v_r2_against_finite_difference": float(
                train_step.get("decoded_v_r2_against_finite_difference_mean", float("nan"))
            ),
            "learned_kappa": float(train_step.get("learned_kappa_mean", train_step.get("kappa_mean", float("nan")))),
            "gamma": float(train_step.get("gamma_mean", float("nan"))),
            "dt": float(train_step.get("dt_mean", float("nan"))),
            "alpha": float(train_step.get("alpha_mean", float("nan"))),
            "force_norm": float(train_step.get("force_norm_mean", train_step.get("injection_term_norm_mean", float("nan")))),
            "diffusion_term_norm": float(train_step.get("diffusion_term_norm_mean", float("nan"))),
            "injection_term_norm": float(train_step.get("injection_term_norm_mean", float("nan"))),
            "relative_diffusion": float(train_step.get("relative_diffusion_mean", float("nan"))),
            "relative_update": float(train_step.get("relative_update_mean", float("nan"))),
            "velocity_fraction": float(train_step.get("velocity_fraction_mean", float("nan"))),
            "force_fraction": float(train_step.get("force_fraction_mean", float("nan"))),
            "pred_delta_corr": float(train_step.get("pred_delta_corr_mean", float("nan"))),
        }
        rows[result.name] = row
        print(
            f"{result.name[:20]:<20} "
            f"{row['val_r2']:>8.3f} "
            f"{row['test_r2']:>8.3f} "
            f"{row['rollout_val_rollout_edge_r2']:>8.3f} "
            f"{row['rollout_test_rollout_edge_r2']:>8.3f} "
            f"{row['persistent_edge_r2']:>8.3f} "
            f"{row['delta_vs_persistent']:>8.3f} "
            f"{row['edge_delta_r2']:>8.3f} "
            f"{row['edge_delta_mae']:>9.3f} "
            f"{row['decoded_v_r2_against_finite_difference']:>8.3f} "
            f"{row['learned_kappa']:>7.3f} "
            f"{row['gamma']:>7.3f} "
            f"{row['dt']:>6.3f} "
            f"{row['alpha']:>7.3f} "
            f"{row['force_norm']:>8.3f} "
            f"{row['diffusion_term_norm']:>8.3f} "
            f"{row['relative_diffusion']:>8.3f} "
            f"{row['relative_update']:>8.3f} "
            f"{row['velocity_fraction']:>7.3f} "
            f"{row['force_fraction']:>7.3f} "
            f"{row['pred_delta_corr']:>8.3f}"
        )
        if readout:
            coeff_line = (
                "  coeffs"
                f" | w_y={float(readout.get('w_y', float('nan'))):.4f}"
                f" w_v={float(readout.get('w_v', float('nan'))):.4f}"
                f" w_drive={float(readout.get('w_drive', float('nan'))):.4f}"
                f" bias={float(readout.get('bias', float('nan'))):.4f}"
            )
            if "w_y_oracle" in readout:
                coeff_line += (
                    f" | oracle=({float(readout['w_y_oracle']):.4f}, {float(readout['w_v_oracle']):.4f}, "
                    f"{float(readout['w_drive_oracle']):.4f}, {float(readout['bias_oracle']):.4f})"
                    f" | abs_diff=({float(readout['abs_diff_w_y']):.4f}, {float(readout['abs_diff_w_v']):.4f}, "
                    f"{float(readout['abs_diff_w_drive']):.4f}, {float(readout['abs_diff_bias']):.4f})"
                )
            print(coeff_line)
    if extra_rows:
        for name, row in extra_rows.items():
            rows[name] = dict(row)
            print(
                f"{name[:20]:<20} "
                f"{row.get('val_r2', float('nan')):>8.3f} "
                f"{row.get('test_r2', float('nan')):>8.3f} "
                f"{row.get('rollout_val_rollout_edge_r2', float('nan')):>8.3f} "
                f"{row.get('rollout_test_rollout_edge_r2', float('nan')):>8.3f} "
                f"{row.get('persistent_edge_r2', float('nan')):>8.3f} "
                f"{row.get('delta_vs_persistent', float('nan')):>8.3f} "
                f"{row.get('edge_delta_r2', float('nan')):>8.3f} "
                f"{row.get('edge_delta_mae', float('nan')):>9.3f} "
                f"{float('nan'):>8.3f} "
                f"{float('nan'):>7.3f} "
                f"{float('nan'):>7.3f} "
                f"{float('nan'):>6.3f} "
                f"{float('nan'):>7.3f} "
                f"{float('nan'):>8.3f} "
                f"{float('nan'):>8.3f} "
                f"{float('nan'):>8.3f} "
                f"{float('nan'):>8.3f} "
                f"{float('nan'):>7.3f} "
                f"{float('nan'):>7.3f} "
                f"{float('nan'):>8.3f}"
            )
            if {"w_y", "w_v", "w_drive", "bias"} <= set(row):
                print(
                    "  coeffs"
                    f" | w_y={float(row['w_y']):.4f}"
                    f" w_v={float(row['w_v']):.4f}"
                    f" w_drive={float(row['w_drive']):.4f}"
                    f" bias={float(row['bias']):.4f}"
                )
    return rows


def _describe_ift_task(task_name: str, rows: dict[str, dict[str, float]]) -> list[str]:
    notes: list[str] = []
    if task_name == "ift_diffusion":
        generic = rows.get("ift1_generic")
        structured = [
            rows[name]["rollout_test_rollout_edge_r2"]
            for name in ("ift1_linear", "ift1_direct", "ift1_gated_direct")
            if name in rows
        ]
        structured = [value for value in structured if np.isfinite(value)]
        if generic is not None and structured and max(structured) > generic["rollout_test_rollout_edge_r2"] + 0.03:
            notes.append("Structured forcing improves over generic IFT.")
        linear = rows.get("ift1_linear")
        if generic is not None and linear is not None:
            if linear["rollout_test_rollout_edge_r2"] > generic["rollout_test_rollout_edge_r2"] + 0.03:
                notes.append("IFT needs a better forcing/injection pathway.")
        return notes

    first_generic = rows.get("ift1_generic")
    first_direct = rows.get("ift1_direct")
    second_generic = rows.get("ift2_generic")
    second_linear = rows.get("ift2_linear")
    second_gated_linear = rows.get("ift2_gated_linear")
    second_direct = rows.get("ift2_direct")
    second_gated = rows.get("ift2_gated_direct")
    ar1 = rows.get("ar1_baseline")
    ar2 = rows.get("ar2_baseline")
    if first_generic and second_generic:
        if second_generic["rollout_test_rollout_edge_r2"] > first_generic["rollout_test_rollout_edge_r2"] + 0.03:
            notes.append("The task mismatch is first-order diffusion vs second-order oscillator.")
    if first_direct and second_direct:
        if second_direct["rollout_test_rollout_edge_r2"] > first_direct["rollout_test_rollout_edge_r2"] + 0.03:
            notes.append("Second-order IFT helps once the forcing signal is exposed.")
    if second_linear and first_generic:
        if second_linear["rollout_test_rollout_edge_r2"] > first_generic["rollout_test_rollout_edge_r2"] + 0.03:
            notes.append("Linear-event forcing exposes the oscillator drive more directly than the generic message pathway.")
    direct_scores = [
        rows[name]["rollout_test_rollout_edge_r2"]
        for name in ("ift1_direct", "ift2_direct", "ift2_gated_direct", "ift2_linear", "ift2_gated_linear")
        if name in rows
    ]
    if first_generic and direct_scores and max(direct_scores) > first_generic["rollout_test_rollout_edge_r2"] + 0.03:
        notes.append("IFT needs a better forcing/injection pathway.")
    if second_gated and second_direct:
        if second_gated["rollout_test_rollout_edge_r2"] > second_direct["rollout_test_rollout_edge_r2"] + 0.03:
            notes.append("Gated direct forcing is better aligned with the oscillator drive signal.")
    if second_gated_linear and second_linear:
        if second_gated_linear["rollout_test_rollout_edge_r2"] > second_linear["rollout_test_rollout_edge_r2"] + 0.03:
            notes.append("Gated linear forcing is better aligned with the oscillator drive signal.")
    if ar1 and ar2:
        if ar2["rollout_test_rollout_edge_r2"] > ar1["rollout_test_rollout_edge_r2"] + 0.03:
            notes.append("AR(2) beats AR(1), so the oscillator task genuinely needs second-order memory.")
    if ar2 and second_direct:
        if ar2["rollout_test_rollout_edge_r2"] > second_direct["rollout_test_rollout_edge_r2"] + 0.03:
            notes.append("AR(2) beats second-order IFT, so the current IFT2 implementation/readout is still the bottleneck.")
    return notes


def _infer_drive_feature_idx(task_name: str, feature_schema: list[str]) -> Optional[int]:
    if "drive" in feature_schema:
        return feature_schema.index("drive")
    if task_name == "ift_diffusion" and "signal" in feature_schema:
        return feature_schema.index("signal")
    return None


def run_ift_diagnostics(args: argparse.Namespace, device: torch.device) -> None:
    task_rows: dict[str, dict[str, dict[str, float]]] = {}
    epochs = 3 if args.epochs is None else int(args.epochs)
    diag_tasks = [str(task) for task in args.ift_diagnostic_tasks]

    for task_name in diag_tasks:
        task_args = argparse.Namespace(**vars(args))
        task_args.synthetic_task = task_name
        synthetic_cfg = build_synthetic_dataset_config(task_args, device, preset="quick")
        ds = load_dataset("synthetic", asdict(synthetic_cfg))
        spec = ds.spec()
        objective_metric = parse_task_metric_spec(
            spec.extra.get("primary_metric") if spec.extra is not None else None
        )
        feature_schema = list(spec.extra.get("task_axes", {}).get("feature_schema", [])) if spec.extra is not None else []
        drive_feature_idx = _infer_drive_feature_idx(task_name, feature_schema)

        base_train_cfg = TrainConfig(
            num_nodes=spec.num_nodes,
            num_neg=10,
            lr=1e-3,
            weight_decay=1e-3,
            device=device,
            log_every=0,
            tbptt_steps=1,
            node_loss_weight=float(args.node_loss_weight),
            node_target_type="regression",
            edge_target_type="regression",
            node_target_mode=str(args.node_target_mode),
            edge_target_mode=str(args.edge_target_mode),
            edge_target_scale=str(args.edge_target_scale),
            prediction_mode=_resolve_prediction_mode(
                str(args.prediction_mode),
                default_state_plus_delta=True,
            ),
            rollout_horizon=int(args.rollout_horizon),
        )
        setattr(base_train_cfg, "ift_generator_params", spec.extra.get("generator_params", {}) if spec.extra is not None else {})
        setattr(base_train_cfg, "ift_batch_sanity_print", task_name == "conservative_oscillator")
        setattr(base_train_cfg, "ift_batch_sanity_done", False)
        base_model_cfg = _base_diagnostic_model_config(spec.event_dim)
        near_ar1_delta_coeffs: Optional[tuple[float, float, float, float]] = None
        if task_name == "conservative_oscillator":
            train_targets = _stack_split_edge_targets(ds.edge_targets("train") or [])
            train_drives = _stack_split_drives(ds.bins("train"), num_nodes=spec.num_nodes, drive_feature_idx=0)
            ar1_coeffs = _fit_ar_coefficients(train_targets, train_drives, order=1)
            if ar1_coeffs is not None:
                near_ar1_delta_coeffs = (
                    float(ar1_coeffs[0].item() - 1.0),
                    0.0,
                    float(ar1_coeffs[1].item()),
                    0.0,
                )
        runs = build_ift_diagnostic_runs(
            base_model_cfg,
            task_name=task_name,
            drive_feature_idx=drive_feature_idx,
            near_ar1_delta_coeffs=near_ar1_delta_coeffs,
            seed=int(args.seed),
        )
        if args.max_runs is not None:
            runs = runs[: args.max_runs]

        print(
            f"\n=== Running IFT diagnostics for {task_name} "
            f"(num_nodes={spec.num_nodes}, num_bins={synthetic_cfg.num_bins}, epochs={epochs}) ==="
        )
        print(f"feature_schema={feature_schema or ['none']}")
        print(f"ift_drive_feature_idx={drive_feature_idx}")
        print(f"prediction_mode={base_train_cfg.prediction_mode}")
        results: list[RunResult] = []
        for run in runs:
            line = f"run {run.name} | seed={run.seed}"
            if run.model_cfg.aggregator == "ift":
                line += (
                    f" | forcing={run.model_cfg.ift_forcing_mode}"
                    f" | order={run.model_cfg.ift_update_order}"
                    f" | drive_idx={run.model_cfg.ift_drive_feature_idx}"
                    f" | v_init={getattr(run.model_cfg, 'ift_velocity_init_mode', 'finite_difference')}"
                    f" | readout={getattr(run.model_cfg, 'ift2_readout_mode', 'default')}"
                )
            print(line)
            results.append(
                run_one_experiment(
                    ds=ds,
                    spec=spec,
                    base_train_cfg=base_train_cfg,
                    run=run,
                    build_model_fn=build_tgn_model,  # type: ignore[arg-type]
                    epochs=epochs,
                    objective_metric=objective_metric,
                    eval_slices=None,
                    save_jsonl_path=args.save_jsonl,
                    rollout_horizon=int(args.rollout_horizon),
                )
            )

        extra_rows = _diagnostic_baseline_rows(
            ds,
            spec,
            task_name=task_name,
            horizon=int(args.rollout_horizon),
        )
        rows = _print_ift_diagnostic_table(task_name, results, extra_rows=extra_rows)
        task_rows[task_name] = rows
        notes = _describe_ift_task(task_name, rows)
        if notes:
            print("Diagnosis:")
            for note in notes:
                print(f"  {note}")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.command == "ift-diagnose":
        run_ift_diagnostics(args, device)
        return

    preset = "full" if args.command == "sweep" else args.command
    suite = build_suite(preset, device, dataset_override=args.dataset, args=args)
    ds = load_dataset(suite.dataset, suite.dataset_kwargs)
    spec = ds.spec()
    objective_metric = parse_task_metric_spec(
        spec.extra.get("primary_metric") if spec.extra is not None else None
    )
    summary_metric_paths = (
        list(spec.extra.get("summary_metrics", []))
        if spec.extra is not None
        else []
    )

    base_train_cfg = TrainConfig(**asdict(suite.train_cfg))
    base_train_cfg.num_nodes = spec.num_nodes
    base_train_cfg.node_loss_weight = float(args.node_loss_weight)
    metric_family = str(spec.extra.get("metric_family")) if spec.extra is not None else ""
    base_train_cfg.node_target_type = "classification" if metric_family == "node_classification" else "regression"
    base_train_cfg.edge_target_type = "classification" if metric_family == "edge_classification" else "regression"
    base_train_cfg.node_target_mode = str(args.node_target_mode)
    base_train_cfg.edge_target_mode = str(args.edge_target_mode)
    base_train_cfg.edge_target_scale = str(args.edge_target_scale)
    base_train_cfg.prediction_mode = _resolve_prediction_mode(str(args.prediction_mode))
    base_train_cfg.rollout_horizon = int(args.rollout_horizon)
    if base_train_cfg.node_target_type == "classification" and base_train_cfg.node_target_mode != "raw":
        raise ValueError("Node classification tasks require --node-target-mode raw.")
    if base_train_cfg.edge_target_type == "classification" and base_train_cfg.edge_target_mode != "raw":
        raise ValueError("Edge classification tasks require --edge-target-mode raw.")

    runs: list[SweepRun] = []
    for run in suite.runs:
        model_cfg = apply_model_overrides(run.model_cfg, args)
        if model_cfg.event_dim is None:
            model_cfg.event_dim = spec.event_dim
        runs.append(
            SweepRun(
                name=run.name,
                model_cfg=model_cfg,
                lr=run.lr,
                weight_decay=run.weight_decay,
                num_neg=run.num_neg,
                tbptt_steps=run.tbptt_steps,
                prediction_mode=run.prediction_mode,
                seed=run.seed,
            )
        )

    if args.max_runs is not None:
        runs = runs[: args.max_runs]
    epochs = suite.epochs if args.epochs is None else args.epochs

    print(
        f"Preset={preset} dataset={spec.name} device={device.type} "
        f"runs={len(runs)} epochs={epochs}"
    )
    if spec.extra is not None and "synthetic_task" in spec.extra:
        task_axes = spec.extra.get("task_axes", {})
        task_tags = spec.extra.get("task_tags", [])
        supervision_desc = "n/a"
        if task_axes:
            supervision_desc = (
                f"{task_axes.get('supervision_level', 'n/a')}/"
                f"{task_axes.get('supervision_type', 'n/a')}"
            )
        print(
            f"Synthetic task={spec.extra['synthetic_task']} "
            f"| graph={task_axes.get('graph_type', 'n/a')}"
            f" | dynamics={task_axes.get('dynamics_type', 'n/a')}"
            f" | target={supervision_desc}"
        )
        if task_axes:
            feature_schema = task_axes.get("feature_schema", [])
            feature_desc = "+".join(str(name) for name in feature_schema) if feature_schema else "none"
            print(
                f"Task family={task_axes.get('generator_family', 'n/a')} "
                f"| events={task_axes.get('event_structure', 'n/a')}"
                f" | temporal={task_axes.get('temporal_mode', 'n/a')}"
                f" | features={feature_desc}"
            )
        if task_tags:
            print(f"Task tags={', '.join(str(tag) for tag in task_tags)}")
        print(f"Focus={spec.extra.get('focus', 'n/a')}")
        if objective_metric is not None:
            print(
                f"Primary metric={objective_metric.path} "
                f"| goal={objective_metric.goal} "
                f"| supported={', '.join(spec.extra.get('supported_metrics', []))}"
            )

    results: list[RunResult] = []
    for run in runs:
        print(f"run {short_run_label(run)} | seed={run.seed}")
        if run.model_cfg.aggregator == "ift" and spec.extra is not None:
            feature_schema = list(spec.extra.get("task_axes", {}).get("feature_schema", []))
            print(
                "  ift config"
                f" | features={feature_schema or ['none']}"
                f" | forcing={getattr(run.model_cfg, 'ift_forcing_mode', 'generic_mlp')}"
                f" | drive_idx={getattr(run.model_cfg, 'ift_drive_feature_idx', None)}"
                f" | order={getattr(run.model_cfg, 'ift_update_order', 'first')}"
            )
        results.append(
            run_one_experiment(
                ds=ds,
                spec=spec,
                base_train_cfg=base_train_cfg,
                run=run,
                build_model_fn=build_tgn_model,  # type: ignore[arg-type]
                epochs=epochs,
                objective_metric=objective_metric,
                eval_slices=suite.eval_slices,
                save_jsonl_path=args.save_jsonl or suite.save_jsonl_path,
                rollout_horizon=int(args.rollout_horizon),
            )
        )

    if objective_metric is not None:
        results.sort(
            key=lambda result: (
                (
                    float("inf")
                    if np.isnan(result.best_objective_value)
                    else (
                        result.best_objective_value
                        if objective_metric.goal == "min"
                        else -result.best_objective_value
                    )
                ),
                result.best_val_loss,
            ),
        )
    else:
        results.sort(
            key=lambda result: (
                result.best_val_loss,
                -result.best_val_mrr if not np.isnan(result.best_val_mrr) else 0.0,
            )
        )
    print(
        "\n=== Sweep summary (sorted by "
        f"{objective_metric.path if objective_metric is not None else 'best val loss'}) ==="
    )
    if objective_metric is not None:
        header_metrics: list[str] = []
        seen_metrics: set[str] = set()
        for path in [objective_metric.path, *summary_metric_paths]:
            if path in seen_metrics:
                continue
            seen_metrics.add(path)
            header_metrics.append(path)
        header = f"{'method':<24} {'seed':>4} {'objective':>12}"
        for path in header_metrics[1:]:
            header += f" {path:>24}"
        header += f" {'node':>32}"
        print(header)
        print("-" * len(header))
        for result in results[:20]:
            snapshot = result.best_snapshot
            objective_value = (
                snapshot_metric_value(snapshot, objective_metric.path)
                if result.best_snapshot
                else float("nan")
            )
            val_metrics = snapshot.get("val", {})
            test_metrics = snapshot.get("test", {})
            node_val_summary = format_node_metric(val_metrics)
            node_test_summary = format_node_metric(test_metrics)
            node_summary = "-"
            if node_val_summary or node_test_summary:
                node_summary = f"va {node_val_summary or '-'} | te {node_test_summary or '-'}"

            row = (
                f"{short_run_label_from_name(result.name)[:24]:<24} "
                f"{result.seed:>4d} "
                f"{objective_value:>12.4g}"
            )
            for path in header_metrics[1:]:
                row += f" {snapshot_metric_value(snapshot, path):>24.4g}"
            row += f" {node_summary:>32}"
            print(row)
        return

    summary_primary = (
        infer_primary_metric(
            results[0].best_snapshot.get("train_eval", {}),
            results[0].best_snapshot.get("val", {}),
            results[0].best_snapshot.get("test", {}),
        )
        if results
        else "mrr"
    )
    if summary_primary == "edge_mse":
        header = (
            f"{'method':<24} {'seed':>4} {'val_loss':>9} {'val_mse':>10} "
            f"{'val_r2':>8} {'pers_r2':>8} {'roll_r2':>8} {'val_nrmse':>10} {'test_mse':>10} {'node':>56}"
        )
    else:
        val_label = f"val_{summary_primary}"
        test_label = f"test_{summary_primary}"
        header = (
            f"{'method':<24} {'seed':>4} {'val_loss':>9} {val_label:>12} "
            f"{test_label:>12} {'node':>18}"
        )
    print(header)
    print("-" * len(header))
    for result in results[:20]:
        val_metrics = result.best_snapshot["val"]
        test_metrics = result.best_snapshot["test"]
        node_name = node_metric_name(val_metrics)
        if node_name is None:
            node_summary = "-"
        else:
            node_val_summary = format_node_metric(val_metrics)
            node_test_summary = format_node_metric(test_metrics)
            node_summary = f"va {node_val_summary} | te {node_test_summary}"

        if summary_primary == "edge_mse":
            print(
                f"{short_run_label_from_name(result.name)[:24]:<24} "
                f"{result.seed:>4d} "
                f"{result.best_val_loss:>9.4f} "
                f"{val_metrics.get('edge_mse', float('nan')):>10.4g} "
                f"{val_metrics.get('edge_r2', float('nan')):>8.3f} "
                f"{val_metrics.get('persistent_edge_r2', float('nan')):>8.3f} "
                f"{result.best_snapshot.get('rollout_val', {}).get('rollout_edge_r2', float('nan')):>8.3f} "
                f"{val_metrics.get('edge_nrmse', float('nan')):>10.3f} "
                f"{test_metrics.get('edge_mse', float('nan')):>10.4g} "
                f"{node_summary:>56}"
            )
        else:
            print(
                f"{short_run_label_from_name(result.name)[:24]:<24} "
                f"{result.seed:>4d} "
                f"{result.best_val_loss:>9.4f} "
                f"{val_metrics.get(summary_primary, float('nan')):>12.4g} "
                f"{test_metrics.get(summary_primary, float('nan')):>12.4g} "
                f"{node_summary:>18}"
            )


if __name__ == "__main__":
    main()
