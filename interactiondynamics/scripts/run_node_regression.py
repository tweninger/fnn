# scripts/run_train_node.py

from __future__ import annotations

from pathlib import Path
import json
import torch
import shutil
import copy
import numpy as np
from core.config import ModelConfig
from models.tgn_model import build_tgn_model
from dataclasses import asdict, replace
from collections import defaultdict
import os

from training.node_regression import NodeTrainConfig, run_one_node_experiment
from eval.node_regression import collect_node_predictions_over_time
from plotting.node_regression import plot_node_targets_by_feature

from experiments.dataset_stats import build_dataset_metadata_row
from experiments.node_regression_runs import make_node_runs
from experiments.physical_dataset_registry import build_physical_datasets
from experiments.defaults import build_base_model_cfg, build_base_train_cfg
from experiments.results_summary import (
    describe_run_result,
    format_node_analysis_metrics,
    format_finished_label,
    format_analysis_label,
    format_starting_run_banner,
    format_top_runs_header,
)
from utils.io import append_jsonl, ensure_parent, make_safe_name
from utils.metric_selection import higher_is_better, is_better_metric
from utils.output_paths import dataset_group_name, experiment_output_paths, prediction_npz_path, run_plot_path

from datasets.spring_mass import SpringMassDataset, SpringMassConfig
from datasets.spring_ring_2d import SpringRing2DDataset, SpringRing2DConfig
from datasets.wave_1d import WaveEquationBinnedDataset, WaveEquationBinnedConfig, make_wave_variants
from datasets.three_body_binned import ThreeBodyBinnedDataset, ThreeBodyBinnedConfig
from datasets.charged_particles import ChargedParticlesBinnedDataset, ChargedParticlesBinnedConfig
from datasets.md22_binned import MD22BinnedDataset, MD22BinnedConfig
from datasets.spring_web_2d import SpringWeb2DConfig, SpringWeb2DDataset, make_spring_web_variants

# -----------------------------
# small helpers
# -----------------------------

def unpack_prediction_pack(pack):
    """
    Normalizes collect_node_predictions_over_time(...) output to always be:
        times, y_true, y_pred, node_mask
    """
    if len(pack) == 4:
        times, y_true, y_pred, node_mask = pack
    else:
        times, y_true, y_pred = pack
        node_mask = None
    return times, y_true, y_pred, node_mask



def main():

    cleared_plot_dirs = set()
    PLOT_MODE = "all"      # "none" | "best" | "all"
    PLOT_RESULTS = PLOT_MODE in {"best", "all"}
    PLOT_BEST_RUN_ONLY = PLOT_MODE == "best"
    PLOT_ALL_RUNS = PLOT_MODE == "all"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    EXPERIMENT_NAME = "physics_sweep"
    RESULTS_ROOT = Path("/home/akapociu/ift/interactiondynamics/results")
    PLOTS_ROOT = Path("/home/akapociu/ift/interactiondynamics/plots")

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

    md22_npz_paths = [
        "/home/akapociu/ift/interactiondynamics/data/MD_DATA/naphthalene.npz",
        # "/home/akapociu/ift/interactiondynamics/data/MD_DATA/stachyose.npz",
        # "/home/akapociu/ift/interactiondynamics/data/MD_DATA/uracil.npz",
    ]

    datasets = build_physical_datasets(
        device=device,
        md22_npz_paths=md22_npz_paths,
        include=("wave",)
        # include=("nbody", "wave", "threebody", "spring_ring", "spring_mass", "md22"),
    )

    all_results = []

    for dataset_name, ds in datasets.items():
        print(f"\n=== DATASET: {dataset_name} ===")

        spec = ds.spec()
        target_names = None
        if spec.extra is not None:
            target_names = spec.extra.get("node_target_names", None)

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
        base_train_cfg = build_base_train_cfg(spec, device)

        runs = make_node_runs(
            base_model_cfg,
            seeds=(0,),
            aggregator=("ift", "sum", "hopfield", "settransformer"),
            upd = ("ift_update", "tgn_gru", "hopfield_update"),
            dropout=(0.0,),
            use_time_features=(False, ),
            ift_kappa_param=("exp", ),
            ift_dt=(0.05, ),
            ift_gamma=(0.05,),
            ift_kappa_init=(0.1, ),
            # optional
            ift_kappa_cap=(False, ),
            ift_kappa_max= (None,),
        )
        # ift_dt=(0.05,),
        # ift_gamma=(0.0,),
        # ift_kappa_init=(1.0,),
        # ift_dt=(0.01, 0.05, 0.1, 0.2),
        # ift_gamma=(0.0, 0.01, 0.05, 0.1),
        # ift_kappa_init=(0.1, 0.5, 1.0, 2.0),

        results = []
        best_result = None
        best_model = None
        best_train_cfg = None
        plot_payloads = [] if PLOT_ALL_RUNS else None
        
        allowed_pairs = None
        allowed_pairs = { # set to None for all pairs
            # ("ift", "hopfield_update"),
            # ("ift", "ift_update"),
            # ("ift", "tgn_gru"),
            # ("settransformer", "tgn_gru"),
            ("sum", "tgn_gru"),
        }

        if allowed_pairs is not None:
            runs = [ 
                run for run in runs
                if (run.model_cfg.aggregator, run.model_cfg.update) in allowed_pairs
            ]

        for run in runs:
            print(format_starting_run_banner(dataset_name, run.name, run.seed))

            try:
                result, model, train_cfg = run_one_node_experiment(
                    ds=ds,
                    spec=spec,
                    base_train_cfg=base_train_cfg,
                    run=run,
                    build_model_fn=build_tgn_model,
                    epochs=3,
                    save_jsonl_path=str(save_jsonl_path),
                    save_summary_path=str(save_summary_path),
                    dataset_name=spec.name,
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

        # -----------------------------
        # plotting
        # -----------------------------

        if PLOT_RESULTS:
            dataset_plot_dir = plot_dir / dataset_group_name(dataset_name)

            if dataset_plot_dir not in cleared_plot_dirs:
                shutil.rmtree(dataset_plot_dir, ignore_errors=True)
                cleared_plot_dirs.add(dataset_plot_dir)

            dataset_plot_dir.mkdir(parents=True, exist_ok=True)


            if PLOT_ALL_RUNS and plot_payloads is not None and len(plot_payloads) > 0:
                print(f"\nPlotting all successful runs for {dataset_name}...")

                for run, result, model, train_cfg in plot_payloads:
                    pack = collect_node_predictions_over_time(
                        model,
                        ds.bins("test"),
                        train_cfg,
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
                    #print(f"Saved plot: {plot_path}")

            elif PLOT_BEST_RUN_ONLY:
                if best_model is None or best_train_cfg is None or best_result is None:
                    raise RuntimeError(f"Best model/train_cfg not available for plotting on {dataset_name}")

                print(f"\nCollecting predictions from BEST run for {dataset_name}...")

                pack = collect_node_predictions_over_time(
                    best_model,
                    ds.bins("test"),
                    best_train_cfg,
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
                    y_mode="symlog",
                    symlog_linthresh=1e-3,
                )

                print(f"Saved best-run plot: {plot_path}")


if __name__ == "__main__":
    main()