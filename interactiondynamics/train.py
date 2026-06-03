from __future__ import annotations

import argparse
from dataclasses import asdict

import numpy as np
import torch

from interactiondynamics.data.synthetic import SYNTHETIC_TASKS
from interactiondynamics.models.tgn_model import build_tgn_model
from interactiondynamics.training.presets import build_suite, load_dataset
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
from interactiondynamics.training.types import RunResult, SweepRun, TrainConfig


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


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
