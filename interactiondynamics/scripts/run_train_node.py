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


from train_node import (
    NodeTrainConfig,
    make_node_runs,
    run_one_node_experiment,
    collect_node_predictions_over_time,
)

from analysis.node_regression_plots import plot_node_targets_by_feature

from data.spring_mass import SpringMassDataset, SpringMassConfig
from data.spring_ring import SpringRing2DDataset, SpringRing2DConfig
from data.one_dimension_wave_binned import WaveEquationBinnedDataset, WaveEquationBinnedConfig, make_wave_variants
from data.three_body_binned import ThreeBodyBinnedDataset, ThreeBodyBinnedConfig
from data.nbody_continuous import ChargedParticlesBinnedDataset, ChargedParticlesBinnedConfig
from data.md22_binned import MD22BinnedDataset, MD22BinnedConfig
from data.spring_web_2d import SpringWeb2DConfig, SpringWeb2DDataset, make_spring_web_variants


# -----------------------------
# small helpers
# -----------------------------

def plot_group_name(dataset_name: str) -> str:
    if dataset_name.startswith("springweb"):
        return "spring_web_2d"
    if dataset_name.startswith("wave"):
        return "wave"
    return dataset_name

def higher_is_better(metric_name: str) -> bool:
    return metric_name in {
        "r2",
        "pearson",
        "spearman",
        "mean_node_pearson",
        "mean_node_spearman",
        "mean_node_r2",
        "mean_cosine",
    }


def is_better_metric(new_value: float, old_value: float, metric_name: str) -> bool:
    return new_value > old_value if higher_is_better(metric_name) else new_value < old_value


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


def describe_run_result(r) -> str:
    selection_metric = getattr(r, "selection_metric", "rmse")
    best_val_metric = getattr(r, "best_val_metric", float("nan"))

    analysis_test = getattr(r, "analysis_test", {}) or {}
    test_rmse = analysis_test.get(
        "rmse",
        r.best_snapshot.get("test", {}).get("rmse", float("nan"))
    )
    test_r2 = analysis_test.get("r2", float("nan"))
    test_pearson = analysis_test.get("mean_node_pearson", float("nan"))

    return (
        f"{r.name} | seed={r.seed} | "
        f"best_val_{selection_metric}={best_val_metric:.6f} @ epoch {r.best_epoch} | "
        f"test_rmse={test_rmse:.6f} | "
        f"test_r2={test_r2:.6f} | "
        f"test_mean_node_pearson={test_pearson:.6f}"
    )

def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
        
def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def make_safe_name(s: str) -> str:
    return (
        s.replace("|", "_")
         .replace("=", "-")
         .replace("/", "_")
         .replace(" ", "_")
    )


# -----------------------------
# experiment config
# -----------------------------

def build_physical_datasets(
    device,
    md22_npz_paths=(),
    include=("nbody", "wave", "spring_ring", "md22", "spring_web_2d"),
):
    datasets = {}

    if "nbody" in include:
        cfg = ChargedParticlesBinnedConfig(
            name="nbody",
            device=device,
        )
        datasets[cfg.name] = ChargedParticlesBinnedDataset(cfg)

    if "wave" in include:
        base_cfg = WaveEquationBinnedConfig(
            name="wave",
            device=device,
            event_mode="thresholded",
            interaction_threshold=19.0334,
            threshold_metric="pair_accel",
            threshold_use_absolute=True,
            standardize_node_targets=False,
            target_name="dv",
        )
        datasets[cfg.name] = WaveEquationBinnedDataset(cfg)
        # wave_variants = make_wave_variants(
        #     base_cfg,
        #     event_modes=("thresholded", "all_neighbors"),
        #     threshold_metrics=("pair_accel", "rel_q", "rel_v", "pair_grad"),
        #     interaction_thresholds=(11.0496, 19.0334, 27.6114),
        #     threshold_use_absolute_options=(True, False),
        #     standardize_node_targets_options=(False, True),
        #     target_names=("dv", "delta_v", "q_xx"),
        #     target_horizons=(1,),
        # )

        # datasets.update(wave_variants)

    if "threebody" in include:
        cfg = ThreeBodyBinnedConfig(
            name="threebody",
            device=device,
        )
        datasets[cfg.name] = ThreeBodyBinnedDataset(cfg)

    if "spring_ring" in include:
        cfg = SpringRing2DConfig(
            name="spring_ring",
            device=device,
            force_threshold=0.0422165, #{50: 0.0275852, 75: 0.0422165, 90: 0.0685315}
            standardize_node_targets=False,
        )
        datasets[cfg.name] = SpringRing2DDataset(cfg)

    if "spring_mass" in include:
        cfg = SpringMassConfig(
            name="spring_mass",
            device=device,
        )
        datasets[cfg.name] = SpringMassDataset(cfg)

    if "md22" in include:
        for npz_path in md22_npz_paths:
            stem = Path(npz_path).stem

            cfg = MD22BinnedConfig(
                name=stem,
                npz_path=str(npz_path),
                device=device,
            )
            datasets[cfg.name] = MD22BinnedDataset(cfg)

    if "spring_web_2d" in include:
        base_cfg = SpringWeb2DConfig(
            name="springweb",
            device=device,
            event_mode="thresholded",
            threshold_metric="force_mag",
            interaction_threshold=0.0275852,
            threshold_use_absolute=True,
        )
        #datasets[cfg.name] = SpringWeb2DDataset(cfg)
        metric_to_values = {
            "force_mag": (0.0273215,),
            "extension": (-0.0112979,),
            # "distance": (0.96589,),
           # "rel_speed": (0.0321316,),
        }
        spring_web_variants = {}

        for metric, values in metric_to_values.items():
            spring_web_variants.update(
                make_spring_web_variants(
                    base_cfg,
                    topologies=("knn",),
                    radius_values=("0.05",),
                    knn_values=(4,),
                    include_ring_edges_options=(False,),
                    event_modes=("thresholded",),
                    threshold_metrics=(metric,),   # one metric at a time
                    threshold_values=values,       # only that metric's value(s)
                    target_types=("dv", "accel", "delta_x"),
                    target_horizons=(1,),
                    standardize_node_targets_options=(False,),
                )
            )

        datasets.update(spring_web_variants)

    return datasets


def build_base_model_cfg(spec) -> ModelConfig:
    return ModelConfig(
        node_dim=64,
        msg_dim=64,
        event_dim=spec.event_dim,

        # baseline/default choices
        aggregator="ift",
        update="ift_update",
        scorer="mlp",

        # generic widths
        dropout=0.0,
        scorer_dropout=0.0,
        use_time_features=False,

        # task-specific flags
        task="node_regression",
        predictor="mlp_node",
    )


def build_base_train_cfg(spec, device: torch.device) -> NodeTrainConfig:
    return NodeTrainConfig(
        num_nodes=spec.num_nodes,
        lr=1e-3,
        weight_decay=1e-5,
        grad_clip=1.0,
        device=device,
        log_every=20,
        tbptt_steps=1,
        debug=False,
        loss_name="mse", # mse | mae | huber
        selection_metric="rmse",
    )


def main():

    cleared_plot_dirs = set()
    PLOT_MODE = "all"      # "none" | "best" | "all"
    PLOT_RESULTS = PLOT_MODE in {"best", "all"}
    PLOT_BEST_RUN_ONLY = PLOT_MODE == "best"
    PLOT_ALL_RUNS = PLOT_MODE == "all"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # PATHSSSSS
    out_dir = Path("/home/akapociu/ift/interactiondynamics/results/wave")
    plot_dir = Path("/home/akapociu/ift/interactiondynamics/plots/11")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # summary and results output
    save_jsonl_path = out_dir / "1.jsonl"
    save_summary_path = out_dir / "11.jsonl"
    #deletes them each run
    for p in [save_jsonl_path, save_summary_path]:
        if p.exists():
            p.unlink()

    ensure_parent(save_jsonl_path)
    ensure_parent(save_summary_path)

    #outputs for correlation stats
    pred_dir = Path("/home/akapociu/ift/interactiondynamics/results/wave/11")
    #also deletes each run
    shutil.rmtree(pred_dir, ignore_errors=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    md22_npz_paths = [
        "/home/akapociu/ift/interactiondynamics/data/MD_DATA/naphthalene.npz",
        # "/home/akapociu/ift/interactiondynamics/data/MD_DATA/stachyose.npz",
        # "/home/akapociu/ift/interactiondynamics/data/MD_DATA/uracil.npz",
    ]

    datasets = build_physical_datasets(
        device=device,
        md22_npz_paths=md22_npz_paths,
        include=("spring_web_2d",)
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

        if spec.extra is None or "node_target_dim" not in spec.extra:
            print(f"Skipping {dataset_name}: missing node_target_dim")
            continue

        base_model_cfg = build_base_model_cfg(spec)
        base_train_cfg = build_base_train_cfg(spec, device)

        runs = make_node_runs(
            base_model_cfg,
            seeds=(42,0,123 ),
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
        
        #allowed_pairs = None
        #allowed_pairs = { # set to None for all pairs
            #("ift", "hopfield_update"),
            #("ift", "ift_update"),
            #("ift", "tgn_gru"),
            #("settransformer", "tgn_gru"),
            #("sum", "tgn_gru"),
        #}

        # if allowed_pairs is not None:
        #     runs = [ 
        #         run for run in runs
        #         if (run.model_cfg.aggregator, run.model_cfg.update) in allowed_pairs
        #     ]

        for run in runs:
            print("\n" + "🧹" * 60)
            print(f"STARTING RUN: dataset={dataset_name} | run={run.name} | seed={run.seed}")
            print("🧹" * 60)

            try:
                result, model, train_cfg = run_one_node_experiment(
                    ds=ds,
                    spec=spec,
                    base_train_cfg=base_train_cfg,
                    run=run,
                    build_model_fn=build_tgn_model,
                    epochs=6,
                    save_jsonl_path=str(save_jsonl_path),
                    save_summary_path=str(save_summary_path),
                    dataset_name=spec.name,
                )

                results.append(result)
                all_results.append((dataset_name, result))
                print("FINISHED:", describe_run_result(result))
                print(
                    f"ANALYSIS: "
                    f"rmse={result.analysis_test.get('rmse', float('nan')):.6f} | "
                    f"r2={result.analysis_test.get('r2', float('nan')):.6f} | "
                    f"mean_node_pearson={result.analysis_test.get('mean_node_pearson', float('nan')):.6f} | "
                    f"mean_node_spearman={result.analysis_test.get('mean_node_spearman', float('nan')):.6f}"
                )

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

        print("\n" + "-" * 60)
        print(f"TOP RUNS BY BEST VAL {selection_metric.upper()} — {dataset_name}")
        print("-" * 60)

        results_sorted = sorted(results, key=lambda r: r.best_val_metric, reverse=reverse)
        for r in results_sorted[:10]:
            print(describe_run_result(r))

        # -----------------------------
        # plotting
        # -----------------------------

        if PLOT_RESULTS:
            plot_group = plot_group_name(dataset_name)
            dataset_plot_dir = plot_dir / plot_group

            if plot_group not in cleared_plot_dirs:
                shutil.rmtree(dataset_plot_dir, ignore_errors=True)
                cleared_plot_dirs.add(plot_group)

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
  
                    safe_dataset_name = make_safe_name(dataset_name)
                    safe_run_name = make_safe_name(run.name)

                    pred_path = pred_dir / f"{safe_dataset_name}__{safe_run_name}_seed-{run.seed}.npz"
                    np.savez(pred_path, **save_dict)

                    plot_path = dataset_plot_dir / (
                        f"{safe_dataset_name}__{safe_run_name}_seed-{run.seed}_epoch-{result.best_epoch}.png"
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

                safe_run_name = make_safe_name(best_result.name)
                plot_path = dataset_plot_dir / f"{safe_run_name}_seed-{best_result.seed}.png"

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