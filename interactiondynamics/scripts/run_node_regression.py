# scripts/run_node_regression.py

from __future__ import annotations

from pathlib import Path
import json
import torch
import shutil
import numpy as np

from core.config import ModelConfig
from models.tgn_model import build_tgn_model
from dataclasses import replace

from training.node_regression import NodeTrainConfig, run_one_node_experiment
from eval.node_regression import collect_node_predictions_over_time
from plotting.node_regression import plot_node_targets_by_feature

from experiments.dataset_stats import build_dataset_metadata_row
from experiments.node_regression_runs import make_node_runs
from experiments.physical_dataset_registry import build_spring_web_target_horizon_datasets
from experiments.defaults import build_base_model_cfg, build_base_train_cfg
from experiments.results_summary import (
    describe_run_result,
    format_node_analysis_metrics,
    format_finished_label,
    format_analysis_label,
    format_starting_run_banner,
    format_top_runs_header,
)
from utils.io import append_jsonl
from utils.metric_selection import higher_is_better, is_better_metric
from utils.output_paths import dataset_group_name, experiment_output_paths, prediction_npz_path, run_plot_path
from datasets.corrupted import CorruptedEventStreamDataset

# -----------------------------
# experiment config
# -----------------------------

CORRUPTION = dict(
    drop_real_prob=0.25,
    add_fake_ratio=0.0,
    corrupt_splits=("train",),
    seed=17,
    fake_feature_mode="zeros",
    avoid_self_loops=True,
    min_keep_per_nonempty_bin=1,
    corrupt_unit="node_block",
    block_node_select="high_degree",
)

USE_CORRUPTION = False

TARGET_TYPES = ("delta_v", "dv", "accel")
TARGET_HORIZONS = (1, 10)

# median_node_pearson: robust per-node correlation (better than mean when a few nodes fail)
SELECTION_METRIC = "median_node_pearson"

EXPERIMENT_NAME = "node_regression_corrupted_ift_sweep"
RESULTS_ROOT = Path("/home/akapociu/ift/interactiondynamics/results")
PLOTS_ROOT = Path("/home/akapociu/ift/interactiondynamics/plots")

PLOT_MODE = "best"      # "none" | "best" | "all"
EPOCHS = 15
SEEDS = (0,)

# -----------------------------
# small helpers
# -----------------------------

def unpack_prediction_pack(pack):
    if len(pack) == 4:
        times, y_true, y_pred, node_mask = pack
    else:
        times, y_true, y_pred = pack
        node_mask = None
    return times, y_true, y_pred, node_mask


def wrap_with_corruption(clean_ds, dataset_name: str):
    if not USE_CORRUPTION:
        return clean_ds, clean_ds, None, dataset_name

    corruption_tag = (
        f"drop={CORRUPTION['drop_real_prob']:.2f}|"
        f"splits={'-'.join(CORRUPTION['corrupt_splits'])}|"
        f"unit={CORRUPTION['corrupt_unit']}"
    )
    tagged_name = f"{dataset_name}_{corruption_tag}"
    ds_corruption = {k: v for k, v in CORRUPTION.items()}
    ds = CorruptedEventStreamDataset(clean_ds, **ds_corruption)
    return ds, clean_ds, CORRUPTION, tagged_name


def main():
    cleared_plot_dirs = set()
    PLOT_RESULTS = PLOT_MODE in {"best", "all"}
    PLOT_BEST_RUN_ONLY = PLOT_MODE == "best"
    PLOT_ALL_RUNS = PLOT_MODE == "all"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    paths = experiment_output_paths(RESULTS_ROOT, PLOTS_ROOT, EXPERIMENT_NAME)

    results_dir = paths["results_dir"]
    plot_dir = paths["plots_dir"]
    pred_dir = paths["preds_dir"]
    save_jsonl_path = paths["results_jsonl"]
    save_summary_path = paths["summary_jsonl"]
    metadata_jsonl_path = results_dir / "metadata.jsonl"

    results_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    clean_datasets = build_spring_web_target_horizon_datasets(
        device=device,
        target_types=TARGET_TYPES,
        target_horizons=TARGET_HORIZONS,
        standardize_node_targets=False,
    )

    all_results = []

    for dataset_name, clean_ds in clean_datasets.items():
        ds, clean_ds, corruption_cfg, dataset_name = wrap_with_corruption(clean_ds, dataset_name)

        print(f"\n=== DATASET: {dataset_name} ===")

        spec = ds.spec()
        target_names = spec.extra.get("node_target_names") if spec.extra else None

        print("Dataset spec:")
        print(json.dumps({
            "name": spec.name,
            "num_nodes": spec.num_nodes,
            "event_dim": spec.event_dim,
            "num_events": spec.num_events,
            "num_bins": spec.num_bins,
            "extra": spec.extra,
        }, indent=2, default=str))

        append_jsonl(metadata_jsonl_path, build_dataset_metadata_row(dataset_name, ds))

        if spec.extra is None or "node_target_dim" not in spec.extra:
            print(f"Skipping {dataset_name}: missing node_target_dim")
            continue

        base_model_cfg = build_base_model_cfg(spec)
        base_train_cfg = replace(
            build_base_train_cfg(spec, device),
            selection_metric=SELECTION_METRIC,
        )

        runs = make_node_runs(
            base_model_cfg,
            seeds=SEEDS,
            aggregator=("ift", "sum", "hopfield", "settransformer"),
            upd = ("ift_update", "tgn_gru", "hopfield_update"),
            dropout=(0.01, 0.1, .05),
            use_time_features=(False, ),
            ift_kappa_param=("softplus", ),
            ift_dt=(0.05, 0.01, 0.1, 0.001),
            ift_gamma=(0.05, 0.01, 0.1, 0.001),
            ift_kappa_init=(0.1, 0.5, 1.0, 0.05),
            # optional
            ift_kappa_cap=(False, True),
            ift_kappa_max= (None, 1.0, 2.0, 5.0),
        )

        allowed_pairs = {
            ("ift", "ift_update"),
            ("ift", "tgn_gru"),
        }
        runs = [
            run for run in runs
            if (run.model_cfg.aggregator, run.model_cfg.update) in allowed_pairs
        ]

        results = []
        best_result = None
        best_model = None
        best_train_cfg = None
        plot_payloads = [] if PLOT_ALL_RUNS else None

        for run in runs:
            print(format_starting_run_banner(dataset_name, run.name, run.seed))

            try:
                result, model, train_cfg = run_one_node_experiment(
                    ds=ds,
                    spec=spec,
                    base_train_cfg=base_train_cfg,
                    run=run,
                    build_model_fn=build_tgn_model,
                    epochs=EPOCHS,
                    save_jsonl_path=str(save_jsonl_path),
                    save_summary_path=str(save_summary_path),
                    dataset_name=dataset_name,
                    clean_ds=clean_ds,
                    corruption_cfg=corruption_cfg,
                )

                results.append(result)
                all_results.append((dataset_name, result))
                print(f"{format_finished_label()} {describe_run_result(result)}")
                print(f"{format_analysis_label()} {format_node_analysis_metrics(result)}")

                if best_result is None or is_better_metric(
                    result.best_val_metric,
                    best_result.best_val_metric,
                    result.selection_metric,
                ):
                    best_result = result
                    if PLOT_RESULTS:
                        best_model = model
                        best_train_cfg = train_cfg

                if plot_payloads is not None:
                    plot_payloads.append((run, result, model, train_cfg))

            except Exception as e:
                print(f"RUN FAILED: dataset={dataset_name} | run={run.name} | seed={run.seed}")
                print(f"Reason: {type(e).__name__}: {e}")

                append_jsonl(
                    save_jsonl_path,
                    {
                        "dataset": dataset_name,
                        "run": run.name,
                        "seed": run.seed,
                        "status": "failed",
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                )

        if not results:
            print(f"No successful runs for dataset={dataset_name}")
            continue

        selection_metric = base_train_cfg.selection_metric
        reverse = higher_is_better(selection_metric)

        print(format_top_runs_header(selection_metric, dataset_name))

        results_sorted = sorted(results, key=lambda r: r.best_val_metric, reverse=reverse)
        for r in results_sorted[:10]:
            print(describe_run_result(r))

        if PLOT_RESULTS:
            dataset_plot_dir = plot_dir / dataset_group_name(dataset_name)

            if dataset_plot_dir not in cleared_plot_dirs:
                shutil.rmtree(dataset_plot_dir, ignore_errors=True)
                cleared_plot_dirs.add(dataset_plot_dir)

            dataset_plot_dir.mkdir(parents=True, exist_ok=True)

            use_clean_targets = USE_CORRUPTION and clean_ds is not ds
            test_target_bins = clean_ds.bins("test") if use_clean_targets else None

            if PLOT_ALL_RUNS and plot_payloads is not None and len(plot_payloads) > 0:
                print(f"\nPlotting all successful runs for {dataset_name}...")

                for run, result, model, train_cfg in plot_payloads:
                    pack = collect_node_predictions_over_time(
                        model,
                        ds.bins("test"),
                        train_cfg,
                        target_bins=test_target_bins,
                    )
                    times, y_true, y_pred, node_mask = unpack_prediction_pack(pack)
                    save_dict = {
                        "times": times,
                        "y_true": y_true,
                        "y_pred": y_pred,
                    }
                    if node_mask is not None:
                        save_dict["node_mask"] = node_mask

                    pred_path = prediction_npz_path(
                        pred_dir,
                        dataset_name=dataset_name,
                        run_name=run.name,
                        seed=run.seed,
                    )
                    np.savez(pred_path, **save_dict)

                    plot_path = run_plot_path(
                        plot_dir,
                        dataset_name=dataset_name,
                        run_name=run.name,
                        seed=run.seed,
                        best_epoch=result.best_epoch,
                    )

                    plot_node_targets_by_feature(
                        times=times,
                        y_true=y_true,
                        y_pred=y_pred,
                        out_path=str(plot_path),
                        dataset_name=dataset_name,
                        target_names=target_names,
                        model_combo=f"{run.name} | seed={run.seed} | best_epoch={result.best_epoch}",
                        y_mode="auto",
                    )

            elif PLOT_BEST_RUN_ONLY:
                if best_model is None or best_train_cfg is None or best_result is None:
                    raise RuntimeError(f"Best model/train_cfg not available for plotting on {dataset_name}")

                print(f"\nCollecting predictions from BEST run for {dataset_name}...")

                pack = collect_node_predictions_over_time(
                    best_model,
                    ds.bins("test"),
                    best_train_cfg,
                    target_bins=test_target_bins,
                )
                times, y_true, y_pred, node_mask = unpack_prediction_pack(pack)

                plot_path = run_plot_path(
                    plot_dir,
                    dataset_name=dataset_name,
                    run_name=best_result.name,
                    seed=best_result.seed,
                )

                plot_node_targets_by_feature(
                    times=times,
                    y_true=y_true,
                    y_pred=y_pred,
                    out_path=str(plot_path),
                    dataset_name=dataset_name,
                    target_names=target_names,
                    model_combo=f"{best_result.name} | seed={best_result.seed}",
                    y_mode="auto",
                )

                print(f"Saved best-run plot: {plot_path}")


if __name__ == "__main__":
    main()
