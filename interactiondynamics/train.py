from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Iterable, Optional, Sequence, cast

import numpy as np
import torch

from interactiondynamics.core.config import ModelConfig
from interactiondynamics.data.interfaces import EdgeTargetBatch
from interactiondynamics.data.synthetic import SYNTHETIC_TASKS
from interactiondynamics.data.synthetic import DIFFUSION_TOPOLOGY_CHOICES
from interactiondynamics.eval.node_metrics import regression_metrics
from interactiondynamics.models.tgn_model import build_tgn_model
from interactiondynamics.training.presets import (
    IFT_HISTORY_STEP_CHOICES,
    IFT_ORDER_CHOICES,
    IFT_VARIANT_CHOICES,
    build_suite,
    load_dataset,
)
from interactiondynamics.training.reporting import (
    infer_primary_metric,
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


TARGET_MODE_CHOICES = ("raw", "residual")
EDGE_TARGET_SCALE_CHOICES = ("raw", "zscore")
PREDICTION_MODE_CHOICES = ("state", "delta", "state_plus_delta")
DATASET_CHOICES = ("toy", "jodie", "synthetic")
PRESET_CHOICES = ("smoke", "quick", "full")


def _build_common_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    data_group = common.add_argument_group("dataset")
    data_group.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default=None,
        help="Optional dataset override when supported by the command.",
    )
    data_group.add_argument(
        "--synthetic-task",
        choices=tuple(SYNTHETIC_TASKS.keys()),
        default="deepsets_sum",
        help="Synthetic benchmark task to use when --dataset synthetic.",
    )
    data_group.add_argument(
        "--synthetic-topology",
        choices=DIFFUSION_TOPOLOGY_CHOICES,
        default=None,
        help="Topology for the diffusion synthetic task; defaults to ring.",
    )
    data_group.add_argument(
        "--synthetic-num-nodes",
        type=int,
        default=None,
        help="Optional node count override for synthetic datasets.",
    )
    data_group.add_argument(
        "--synthetic-events-per-bin",
        type=int,
        default=None,
        help="Optional event-count override for set-based synthetic datasets.",
    )
    data_group.add_argument(
        "--num-bins",
        type=int,
        default=None,
        help="Number of simulated time bins for synthetic datasets.",
    )
    data_group.add_argument("--seed", type=int, default=0, help="Random seed for simulated datasets.")
    data_group.add_argument(
        "--ift-variants",
        nargs="*",
        choices=IFT_VARIANT_CHOICES,
        default=None,
        help=(
            "Replace the default quick/smoke IFT run with an IFT variant sweep over "
            f"{IFT_VARIANT_CHOICES}. Pass no variant names to sweep them all."
        ),
    )
    data_group.add_argument(
        "--ift-orders",
        nargs="*",
        type=int,
        choices=IFT_ORDER_CHOICES,
        default=None,
        help="Optional subset of IFT orders to sweep when --ift-variants is active. Defaults to 1 and 2.",
    )
    data_group.add_argument(
        "--ift-history-steps",
        nargs="*",
        type=int,
        choices=IFT_HISTORY_STEP_CHOICES,
        default=None,
        help="Optional subset of history readout steps to sweep for the IFT auto variant. Defaults to 1, 2, and 3.",
    )
    data_group.add_argument(
        "--ift-self-rollout",
        action="store_true",
        help="Add a deterministic closed-loop self-field rollout for ring- and grid-wave tasks.",
    )
    data_group.add_argument(
        "--ift-free-rollout",
        action="store_true",
        help="Add a drive-free IFT2 rollout for ring- and grid-wave tasks.",
    )

    train_group = common.add_argument_group("training")
    train_group.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap on the number of runs to execute after filtering.",
    )
    train_group.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for the preset epoch count.",
    )
    train_group.add_argument(
        "--rollout-horizon",
        type=int,
        default=5,
        help="Evaluate k-step target rollout with this horizon for regression datasets.",
    )
    train_group.add_argument(
        "--rollout-train-steps",
        type=int,
        default=1,
        help=(
            "Number of differentiable autoregressive steps per optimizer update for compatible "
            "IFT2 history-readout runs. One keeps ordinary one-step training."
        ),
    )
    train_group.add_argument(
        "--use-node-scorer",
        action="store_true",
        help="Enable auxiliary node prediction on whether a node appears in the next bin.",
    )
    train_group.add_argument(
        "--node-loss-weight",
        type=float,
        default=1.0,
        help="Weight for the auxiliary node prediction loss.",
    )
    train_group.add_argument(
        "--node-scorer-hidden",
        type=int,
        default=128,
        help="Hidden size for the auxiliary node scorer MLP.",
    )

    target_group = common.add_argument_group("targets")
    target_group.add_argument(
        "--node-target-mode",
        choices=TARGET_MODE_CHOICES,
        default="raw",
        help="Train the node head on raw next-step node values or residuals relative to the previous step.",
    )
    target_group.add_argument(
        "--edge-target-mode",
        choices=TARGET_MODE_CHOICES,
        default="raw",
        help="Train the edge head on raw next-step magnitudes or residuals relative to the previous step.",
    )
    target_group.add_argument(
        "--edge-target-scale",
        choices=EDGE_TARGET_SCALE_CHOICES,
        default="raw",
        help="Use raw edge MSE or scale edge regression loss by the train-split target std.",
    )
    target_group.add_argument(
        "--prediction-mode",
        choices=PREDICTION_MODE_CHOICES,
        default="state",
        help="Train regression heads on next-state targets, delta targets, or reconstructed state from predicted deltas.",
    )

    output_group = common.add_argument_group("output")
    output_group.add_argument(
        "--save-jsonl",
        type=str,
        default=None,
        help="Optional path to append per-epoch JSONL results.",
    )
    return common


def _add_subcommands(
    parser: argparse.ArgumentParser,
    *,
    common: argparse.ArgumentParser,
) -> None:
    subparsers = parser.add_subparsers(dest="command")
    for name, help_text in (
        ("smoke", "Run a tiny smoke test preset."),
        ("quick", "Run the focused shortlist preset."),
        ("sweep", "Run the original broad sweep."),
    ):
        subparsers.add_parser(name, parents=[common], help=help_text)


def _resolve_command_or_error(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> argparse.Namespace:
    if args.command is None:
        if args.preset is not None:
            args.command = "sweep" if args.preset == "full" else args.preset
        else:
            parser.error("please provide a subcommand: smoke, quick, or sweep")
    return args


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run interaction dynamics training presets.")
    common = _build_common_parser()
    _add_subcommands(parser, common=common)
    parser.add_argument(
        "--preset",
        choices=PRESET_CHOICES,
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    return _resolve_command_or_error(parser, args)


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
            curr_pred = prev1
            for idx in range(start_idx, start_idx + horizon):
                curr_pred = _predict_autoregressive(coeffs, prev1, None, drives[idx - 1])
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
    edge_targets_train = ds.edge_targets("train")
    edge_targets_val = ds.edge_targets("val")
    edge_targets_test = ds.edge_targets("test")
    if edge_targets_train is None or edge_targets_val is None or edge_targets_test is None:
        return {}
    train_targets = _stack_split_edge_targets(edge_targets_train)
    val_targets = _stack_split_edge_targets(edge_targets_val)
    test_targets = _stack_split_edge_targets(edge_targets_test)
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
            "delta_vs_persistent": float(
                test_metrics.get("rollout_edge_r2", float("nan"))
                - test_metrics.get("rollout_persistent_edge_r2", float("nan"))
            ),
            "edge_delta_r2": float(test_metrics.get("rollout_edge_delta_r2", float("nan"))),
            "edge_delta_mae": float(test_metrics.get("rollout_edge_delta_mae", float("nan"))),
        }

    generator_params = {}
    if spec.extra is not None:
        generator_params = dict(spec.extra.get("generator_params") or {})
    if {"a", "b", "c"} <= set(generator_params):
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
            "delta_vs_persistent": float(
                test_metrics.get("rollout_edge_r2", float("nan"))
                - test_metrics.get("rollout_persistent_edge_r2", float("nan"))
            ),
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
            "delta_vs_persistent": float(
                test_metrics.get("rollout_edge_r2", float("nan"))
                - test_metrics.get("rollout_persistent_edge_r2", float("nan"))
            ),
            "edge_delta_r2": float(test_metrics.get("rollout_edge_delta_r2", float("nan"))),
            "edge_delta_mae": float(test_metrics.get("rollout_edge_delta_mae", float("nan"))),
            "w_y": float(delta_coeffs[0].item()),
            "w_v": float(delta_coeffs[1].item()),
            "w_drive": float(delta_coeffs[2].item()),
            "bias": float(delta_coeffs[3].item()),
        }
    return rows


def _infer_regression_target_kind(snapshot: dict[str, object]) -> Optional[str]:
    val_metrics = cast(dict[str, float], snapshot.get("val", {}))
    if "edge_mse" in val_metrics or "edge_r2" in val_metrics:
        return "edge"
    if "node_mse" in val_metrics or "node_r2" in val_metrics:
        return "node"
    return None


def _infer_snapshot_kind(snapshot: dict[str, object]) -> str:
    val_metrics = cast(dict[str, float], snapshot.get("val", {}))
    if any(key.startswith("edge_") for key in val_metrics):
        return "edge"
    if any(key.startswith("node_") for key in val_metrics):
        return "node"
    if "mrr" in val_metrics or "hits@1" in val_metrics:
        return "rank"
    return "-"


def _snapshot_metric_or_none(snapshot: dict[str, object], path: str) -> Optional[float]:
    value = float(snapshot_metric_value(snapshot, path))
    return None if np.isnan(value) else value


def _format_table_float(
    value: Optional[float],
    *,
    width: int,
    precision: int = 3,
) -> str:
    if value is None or np.isnan(value):
        return f"{'-':>{width}}"
    return f"{value:>{width}.{precision}f}"


def _format_summary_float(
    value: Optional[float],
    *,
    width: int,
) -> str:
    if value is None or np.isnan(value):
        return f"{'-':>{width}}"
    return f"{value:>{width}.4g}"


def _build_ift_diagnostic_row(snapshot: dict[str, object]) -> Optional[dict[str, Optional[float] | str]]:
    target_kind = _infer_regression_target_kind(snapshot)
    if target_kind is None:
        inferred_kind = _infer_snapshot_kind(snapshot)
        return None if inferred_kind == "-" else {
            "target_kind": inferred_kind,
            "auc_val": _snapshot_metric_or_none(snapshot, f"val.{inferred_kind}_auroc") if inferred_kind in {"edge", "node"} else None,
            "auc_test": _snapshot_metric_or_none(snapshot, f"test.{inferred_kind}_auroc") if inferred_kind in {"edge", "node"} else None,
            "f1_val": _snapshot_metric_or_none(snapshot, f"val.{inferred_kind}_f1") if inferred_kind in {"edge", "node"} else None,
            "f1_test": _snapshot_metric_or_none(snapshot, f"test.{inferred_kind}_f1") if inferred_kind in {"edge", "node"} else None,
            "state_val_r2": None,
            "state_test_r2": None,
            "rollout_val_r2": None,
            "rollout_test_r2": None,
            "persistent_r2": None,
            "delta_vs_persistent": None,
            "delta_r2": None,
            "delta_mae": None,
        }
    # The diagnostic rollout columns must compare like with like: use the
    # frozen baseline evaluated at the same test rollout horizon, not the
    # unrelated one-step validation persistence score.
    persistent_r2 = _snapshot_metric_or_none(
        snapshot,
        f"rollout_test.rollout_persistent_{target_kind}_r2",
    )
    rollout_test_r2 = _snapshot_metric_or_none(snapshot, f"rollout_test.rollout_{target_kind}_r2")
    return {
        "target_kind": target_kind,
        "auc_val": _snapshot_metric_or_none(snapshot, f"val.{target_kind}_auroc"),
        "auc_test": _snapshot_metric_or_none(snapshot, f"test.{target_kind}_auroc"),
        "f1_val": _snapshot_metric_or_none(snapshot, f"val.{target_kind}_f1"),
        "f1_test": _snapshot_metric_or_none(snapshot, f"test.{target_kind}_f1"),
        "state_val_r2": _snapshot_metric_or_none(snapshot, f"val.{target_kind}_r2"),
        "state_test_r2": _snapshot_metric_or_none(snapshot, f"test.{target_kind}_r2"),
        "rollout_val_r2": _snapshot_metric_or_none(snapshot, f"rollout_val.rollout_{target_kind}_r2"),
        "rollout_test_r2": rollout_test_r2,
        "persistent_r2": persistent_r2,
        "delta_vs_persistent": (
            None
            if rollout_test_r2 is None or persistent_r2 is None
            else rollout_test_r2 - persistent_r2
        ),
        "delta_r2": _snapshot_metric_or_none(snapshot, f"rollout_test.rollout_{target_kind}_delta_r2"),
        "delta_mae": _snapshot_metric_or_none(snapshot, f"rollout_test.rollout_{target_kind}_delta_mae"),
    }


def _compact_regression_summary_kind(results: Sequence[RunResult]) -> Optional[str]:
    for result in results:
        snapshot = result.best_snapshot if result.best_snapshot else result.final_snapshot
        target_kind = _infer_regression_target_kind(snapshot)
        if target_kind is not None:
            return target_kind
    return None


def _print_ift_diagnostic_table(
    task_name: str,
    results: Sequence[RunResult],
    *,
    extra_rows: Optional[dict[str, dict[str, float]]] = None,
) -> None:
    print(f"\n=== IFT diagnostic table: {task_name} ===")
    header = (
        f"{'run':<20} {'target':>6} {'auc_v':>7} {'auc_t':>7} {'f1_v':>7} {'f1_t':>7} "
        f"{'state_v':>8} {'state_t':>8} {'roll_v':>8} {'roll_t':>8} {'roll_pers':>9} "
        f"{'d_rollpers':>10} {'delta_r2':>8} {'delta_mae':>9} {'vel_r2':>8} {'kappa':>7} {'gamma':>7} {'dt':>6} "
        f"{'alpha':>7} {'force':>8} {'diff':>8} {'rel_d':>8} {'rel_u':>8} {'vel_f':>7} {'for_f':>7} {'d_corr':>8} {'vel_mse':>8}"
    )
    print(header)
    print("-" * len(header))
    node_delta_metrics_missing = False
    non_regression_metrics_missing = False

    for result in results:
        snap = result.best_snapshot if result.best_snapshot else result.final_snapshot
        train_step = snap.get("train_step", {})
        readout = snap.get("readout", {})
        row = _build_ift_diagnostic_row(snap)
        if row is None:
            continue
        if (
            cast(Optional[float], row["state_val_r2"]) is None
            and cast(Optional[float], row["state_test_r2"]) is None
            and cast(Optional[float], row["rollout_val_r2"]) is None
            and cast(Optional[float], row["rollout_test_r2"]) is None
        ):
            non_regression_metrics_missing = True
        if (
            row["target_kind"] == "node"
            and row["delta_r2"] is None
            and row["delta_mae"] is None
        ):
            node_delta_metrics_missing = True
        row.update({
            "decoded_v_r2_against_finite_difference": float(
                train_step.get(
                    "internal_velocity_r2_mean",
                    train_step.get("decoded_v_r2_against_finite_difference_mean", float("nan")),
                )
            ),
            "internal_velocity_mse": float(
                train_step.get("internal_velocity_mse_mean", train_step.get("velocity_loss_mean", float("nan")))
            ),
            "learned_kappa": float(train_step.get("learned_kappa_mean", train_step.get("kappa_mean", float("nan")))),
            "gamma": float(train_step.get("gamma_mean", float("nan"))),
            "dt": float(train_step.get("dt_mean", float("nan"))),
            "alpha": float(train_step.get("alpha_mean", float("nan"))),
            "force_norm": float(train_step.get("force_norm_mean", train_step.get("injection_term_norm_mean", float("nan")))),
            "diffusion_term_norm": float(train_step.get("diffusion_term_norm_mean", float("nan"))),
            "relative_diffusion": float(train_step.get("relative_diffusion_mean", float("nan"))),
            "relative_update": float(train_step.get("relative_update_mean", float("nan"))),
            "velocity_fraction": float(train_step.get("velocity_fraction_mean", float("nan"))),
            "force_fraction": float(train_step.get("force_fraction_mean", float("nan"))),
            "pred_delta_corr": float(train_step.get("pred_delta_corr_mean", float("nan"))),
        })
        print(
            f"{result.name[:20]:<20} "
            f"{str(row['target_kind']):>6} "
            f"{_format_table_float(cast(Optional[float], row['auc_val']), width=7)} "
            f"{_format_table_float(cast(Optional[float], row['auc_test']), width=7)} "
            f"{_format_table_float(cast(Optional[float], row['f1_val']), width=7)} "
            f"{_format_table_float(cast(Optional[float], row['f1_test']), width=7)} "
            f"{_format_table_float(cast(Optional[float], row['state_val_r2']), width=8)} "
            f"{_format_table_float(cast(Optional[float], row['state_test_r2']), width=8)} "
            f"{_format_table_float(cast(Optional[float], row['rollout_val_r2']), width=8)} "
            f"{_format_table_float(cast(Optional[float], row['rollout_test_r2']), width=8)} "
            f"{_format_table_float(cast(Optional[float], row['persistent_r2']), width=9)} "
            f"{_format_table_float(cast(Optional[float], row['delta_vs_persistent']), width=10)} "
            f"{_format_table_float(cast(Optional[float], row['delta_r2']), width=8)} "
            f"{_format_table_float(cast(Optional[float], row['delta_mae']), width=9)} "
            f"{_format_table_float(cast(Optional[float], row['decoded_v_r2_against_finite_difference']), width=8)} "
            f"{_format_table_float(cast(Optional[float], row['learned_kappa']), width=7)} "
            f"{_format_table_float(cast(Optional[float], row['gamma']), width=7)} "
            f"{_format_table_float(cast(Optional[float], row['dt']), width=6)} "
            f"{_format_table_float(cast(Optional[float], row['alpha']), width=7)} "
            f"{_format_table_float(cast(Optional[float], row['force_norm']), width=8)} "
            f"{_format_table_float(cast(Optional[float], row['diffusion_term_norm']), width=8)} "
            f"{_format_table_float(cast(Optional[float], row['relative_diffusion']), width=8)} "
            f"{_format_table_float(cast(Optional[float], row['relative_update']), width=8)} "
            f"{_format_table_float(cast(Optional[float], row['velocity_fraction']), width=7)} "
            f"{_format_table_float(cast(Optional[float], row['force_fraction']), width=7)} "
            f"{_format_table_float(cast(Optional[float], row['pred_delta_corr']), width=8)} "
            f"{_format_table_float(cast(Optional[float], row['internal_velocity_mse']), width=8)}"
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
            print(
                f"{name[:20]:<20} "
                f"{'edge':>6} "
                f"{_format_table_float(None, width=7)} "
                f"{_format_table_float(None, width=7)} "
                f"{_format_table_float(None, width=7)} "
                f"{_format_table_float(None, width=7)} "
                f"{_format_table_float(row.get('val_r2'), width=8)} "
                f"{_format_table_float(row.get('test_r2'), width=8)} "
                f"{_format_table_float(row.get('rollout_val_rollout_edge_r2'), width=8)} "
                f"{_format_table_float(row.get('rollout_test_rollout_edge_r2'), width=8)} "
                f"{_format_table_float(row.get('persistent_edge_r2'), width=8)} "
                f"{_format_table_float(row.get('delta_vs_persistent'), width=8)} "
                f"{_format_table_float(row.get('edge_delta_r2'), width=8)} "
                f"{_format_table_float(row.get('edge_delta_mae'), width=9)} "
                f"{_format_table_float(None, width=8)} "
                f"{_format_table_float(None, width=7)} "
                f"{_format_table_float(None, width=7)} "
                f"{_format_table_float(None, width=6)} "
                f"{_format_table_float(None, width=7)} "
                f"{_format_table_float(None, width=8)} "
                f"{_format_table_float(None, width=8)} "
                f"{_format_table_float(None, width=8)} "
                f"{_format_table_float(None, width=8)} "
                f"{_format_table_float(None, width=7)} "
                f"{_format_table_float(None, width=7)} "
                f"{_format_table_float(None, width=8)} "
                f"{_format_table_float(None, width=8)}"
            )
            if {"w_y", "w_v", "w_drive", "bias"} <= set(row):
                print(
                    "  coeffs"
                    f" | w_y={float(row['w_y']):.4f}"
                    f" w_v={float(row['w_v']):.4f}"
                    f" w_drive={float(row['w_drive']):.4f}"
                    f" bias={float(row['bias']):.4f}"
                )
    if node_delta_metrics_missing:
        print("  note: node rollout delta metrics are not currently tracked; delta_r2 and delta_mae are shown as -.")
    if non_regression_metrics_missing:
        print("  note: regression-style state and rollout metrics are unavailable for this target family; those columns are shown as -.")


def _print_ift_diagnostic_footer(
    args: argparse.Namespace,
    *,
    ds,
    spec,
    runs: Sequence[SweepRun],
    results: Sequence[RunResult],
) -> None:
    if spec.extra is None or "synthetic_task" not in spec.extra:
        return
    if not _ift_variant_selector_requested(args) and str(args.synthetic_task) != "diffusion":
        return
    run_lookup = {(run.name, run.seed): run for run in runs}
    ift_results = [
        result
        for result in results
        if (result.name, result.seed) in run_lookup
        and run_lookup[(result.name, result.seed)].model_cfg.aggregator == "ift"
    ]
    if not ift_results:
        return
    task_name = str(spec.extra["synthetic_task"])
    extra_rows = _diagnostic_baseline_rows(
        ds,
        spec,
        task_name=task_name,
        horizon=int(args.rollout_horizon),
    )
    _print_ift_diagnostic_table(task_name, ift_results, extra_rows=extra_rows or None)


def _ift_variant_selector_requested(args: argparse.Namespace) -> bool:
    return any(
        getattr(args, name, None) is not None
        for name in ("ift_variants", "ift_orders", "ift_history_steps", "ift_self_rollout", "ift_free_rollout")
    )


def _normalize_ift_variant_args(args: argparse.Namespace) -> None:
    if not _ift_variant_selector_requested(args):
        return
    raw_variants = cast(Optional[Sequence[str]], getattr(args, "ift_variants", None))
    raw_orders = cast(Optional[Sequence[int]], getattr(args, "ift_orders", None))
    raw_history = cast(Optional[Sequence[int]], getattr(args, "ift_history_steps", None))
    self_rollout = bool(getattr(args, "ift_self_rollout", False))
    free_rollout = bool(getattr(args, "ift_free_rollout", False))
    if args.command not in {"smoke", "quick"}:
        raise ValueError("IFT variant selection is only supported with the smoke or quick presets.")
    if raw_history is not None and raw_variants not in (None, []) and "auto" not in raw_variants:
        raise ValueError("--ift-history-steps requires selecting the auto IFT variant.")
    if raw_orders not in (None, []) and 2 not in raw_orders:
        if (raw_variants not in (None, []) and "auto" in raw_variants) or raw_history is not None or self_rollout or free_rollout:
            raise ValueError("IFT auto/history/free/self rollout variants require including second-order IFT via --ift-orders 2.")
    autonomous_tasks = {"diffusion", "wave", "wave_grid", "wave_torus", "wave_doorway", "wave_swiss_cheese"}
    if (self_rollout or free_rollout) and args.synthetic_task not in autonomous_tasks:
        raise ValueError("--ift-free-rollout and --ift-self-rollout are implemented only for diffusion and wave tasks.")
    if args.dataset is None:
        args.dataset = "synthetic"
        return
    if args.dataset != "synthetic":
        raise ValueError("IFT variant selection requires the synthetic dataset.")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _normalize_ift_variant_args(args)
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
    if args.rollout_train_steps < 1:
        raise ValueError("--rollout-train-steps must be at least 1.")
    base_train_cfg.rollout_train_steps = int(args.rollout_train_steps)
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
                f" | rollout_train_steps={base_train_cfg.rollout_train_steps}"
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
        regression_kind = _compact_regression_summary_kind(results)
        if regression_kind is not None:
            header = (
                f"{'method':<24} {'seed':>4} {'kind':>6} {'objective':>12} "
                f"{'val_mse':>10} {'val_r2':>8} {'val_pers_r2':>11} "
                f"{'test_mse':>10} {'test_r2':>8} {'test_roll_r2':>12} {'test_roll_pers_r2':>17}"
            )
            print(header)
            print("-" * len(header))
            for result in results[:20]:
                snapshot = result.best_snapshot if result.best_snapshot else result.final_snapshot
                objective_value = _snapshot_metric_or_none(snapshot, objective_metric.path)
                row = (
                    f"{short_run_label_from_name(result.name)[:24]:<24} "
                    f"{result.seed:>4d} "
                    f"{regression_kind:>6} "
                    f"{_format_summary_float(objective_value, width=12)} "
                    f"{_format_summary_float(_snapshot_metric_or_none(snapshot, f'val.{regression_kind}_mse'), width=10)} "
                    f"{_format_summary_float(_snapshot_metric_or_none(snapshot, f'val.{regression_kind}_r2'), width=8)} "
                    f"{_format_summary_float(_snapshot_metric_or_none(snapshot, f'val.persistent_{regression_kind}_r2'), width=11)} "
                    f"{_format_summary_float(_snapshot_metric_or_none(snapshot, f'test.{regression_kind}_mse'), width=10)} "
                    f"{_format_summary_float(_snapshot_metric_or_none(snapshot, f'test.{regression_kind}_r2'), width=8)} "
                    f"{_format_summary_float(_snapshot_metric_or_none(snapshot, f'rollout_test.rollout_{regression_kind}_r2'), width=12)} "
                    f"{_format_summary_float(_snapshot_metric_or_none(snapshot, f'rollout_test.rollout_persistent_{regression_kind}_r2'), width=17)}"
                )
                print(row)
        else:
            header_metrics: list[str] = []
            seen_metrics: set[str] = set()
            for path in [objective_metric.path, *summary_metric_paths]:
                if path in seen_metrics:
                    continue
                seen_metrics.add(path)
                header_metrics.append(path)
            header = f"{'method':<24} {'seed':>4} {'kind':>6} {'objective':>12}"
            for path in header_metrics[1:]:
                header += f" {path:>24}"
            print(header)
            print("-" * len(header))
            for result in results[:20]:
                snapshot = result.best_snapshot if result.best_snapshot else result.final_snapshot
                objective_value = _snapshot_metric_or_none(snapshot, objective_metric.path)
                row = (
                    f"{short_run_label_from_name(result.name)[:24]:<24} "
                    f"{result.seed:>4d} "
                    f"{_infer_snapshot_kind(snapshot):>6} "
                    f"{_format_summary_float(objective_value, width=12)}"
                )
                for path in header_metrics[1:]:
                    row += f" {_format_summary_float(_snapshot_metric_or_none(snapshot, path), width=24)}"
                print(row)
        _print_ift_diagnostic_footer(args, ds=ds, spec=spec, runs=runs, results=results)
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
    if summary_primary in {"edge_mse", "node_mse"}:
        stem = "edge" if summary_primary == "edge_mse" else "node"
        header = (
            f"{'method':<24} {'seed':>4} {'kind':>6} {'val_loss':>9} "
            f"{'val_mse':>10} {'val_r2':>8} {'pers_r2':>8} "
            f"{'test_mse':>10} {'test_r2':>8} {'test_roll_r2':>12}"
        )
    else:
        val_label = f"val_{summary_primary}"
        test_label = f"test_{summary_primary}"
        header = (
            f"{'method':<24} {'seed':>4} {'kind':>6} {'val_loss':>9} "
            f"{val_label:>12} {test_label:>12}"
        )
    print(header)
    print("-" * len(header))
    for result in results[:20]:
        val_metrics = result.best_snapshot["val"]
        test_metrics = result.best_snapshot["test"]
        if summary_primary in {"edge_mse", "node_mse"}:
            stem = "edge" if summary_primary == "edge_mse" else "node"
            print(
                f"{short_run_label_from_name(result.name)[:24]:<24} "
                f"{result.seed:>4d} "
                f"{stem:>6} "
                f"{result.best_val_loss:>9.4f} "
                f"{_format_summary_float(val_metrics.get(f'{stem}_mse'), width=10)} "
                f"{_format_summary_float(val_metrics.get(f'{stem}_r2'), width=8)} "
                f"{_format_summary_float(val_metrics.get(f'persistent_{stem}_r2'), width=8)} "
                f"{_format_summary_float(test_metrics.get(f'{stem}_mse'), width=10)} "
                f"{_format_summary_float(test_metrics.get(f'{stem}_r2'), width=8)} "
                f"{_format_summary_float(result.best_snapshot.get('rollout_test', {}).get(f'rollout_{stem}_r2'), width=12)}"
            )
        else:
            print(
                f"{short_run_label_from_name(result.name)[:24]:<24} "
                f"{result.seed:>4d} "
                f"{_infer_snapshot_kind(result.best_snapshot):>6} "
                f"{result.best_val_loss:>9.4f} "
                f"{val_metrics.get(summary_primary, float('nan')):>12.4g} "
                f"{test_metrics.get(summary_primary, float('nan')):>12.4g}"
            )
    _print_ift_diagnostic_footer(args, ds=ds, spec=spec, runs=runs, results=results)


if __name__ == "__main__":
    main()
