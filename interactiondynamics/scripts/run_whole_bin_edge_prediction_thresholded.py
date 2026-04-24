from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from eval.evaluate import EvalSlices
from experiments.dataset_stats import build_dataset_metadata_row
from experiments.defaults import build_base_interaction_model_cfg
from experiments.interaction_prediction_runs import make_runs
from experiments.results_summary import (
    format_analysis_label,
    format_failed_label,
    format_finished_label,
    format_starting_run_banner,
    format_top_runs_header,
)
from models.tgn_model import build_tgn_model
from training.whole_bin_edge_prediction import (
    WholeBinTrainConfig,
    WholeBinRunResult,
    run_one_whole_bin_experiment,
)
from utils.io import append_jsonl
from utils.output_paths import experiment_output_paths

from datasets.charged_particles import (
    ChargedParticlesBinnedConfig,
    make_charged_particle_threshold_variants,
)
# Optional later:
# from datasets.wave_1d import WaveEquationBinnedConfig, make_wave_variants
# from datasets.spring_web_2d import SpringWeb2DConfig, make_spring_web_variants


RESULTS_ROOT = Path("/home/akapociu/ift/interactiondynamics/results")
PLOTS_ROOT = Path("/home/akapociu/ift/interactiondynamics/plots")
#EXPERIMENT_NAME = "whole_bin_edge_prediction_thresholded_all_models_.05_cutoff_plus_ift_sweep"
EXPERIMENT_NAME = "test"

def build_thresholded_whole_bin_datasets(device: torch.device):
    datasets = {}

    charged_base_cfg = ChargedParticlesBinnedConfig(
        name="charged_particles",
        device=device,
        interaction_rule="all_pairs",
        obs_edge_keep_prob=1.0,
        min_edges_per_bin=1,
        threshold_splits=("train", "val"),
    )

    datasets.update(
        make_charged_particle_threshold_variants(
            charged_base_cfg,
            threshold_metric="distance_threshold",
            threshold_values=(
                #2.23916,   # keep roughly 80% of edges
                4.582872,
                # 6.782091,
                
            ),
            threshold_splits_options=(("train", "val", "test"),),
        )
    )

    # Example force-threshold sweep if you want it too:
    # datasets.update(
    #     make_charged_particle_threshold_variants(
    #         charged_base_cfg,
    #         threshold_metric="force_threshold",
    #         threshold_values=(0.395001, 0.024478),
    #         threshold_splits_options=(("train", "val"),),
    #     )
    # )

    return datasets


# Per-dataset choice: should next-bin labels treat (i,j) and (j,i) as the same pair?
def use_upper_triangle_for_dataset(dataset_name: str) -> bool:
    if dataset_name.startswith("charged_particles"):
        return True
    if dataset_name.startswith("springweb"):
        return True
    return False


def describe_whole_bin_run_result(result: WholeBinRunResult) -> str:
    best_snapshot = result.best_snapshot or {}
    best_val = float(best_snapshot.get("val", {}).get(result.selection_metric, float("nan")))
    best_test_jaccard = float(best_snapshot.get("test", {}).get("jaccard", float("nan")))
    best_test_pr_auc = float(best_snapshot.get("test", {}).get("pr_auc", float("nan")))
    return (
        f"{result.name} | seed={result.seed} | "
        f"best_val_{result.selection_metric}={best_val:.6f} @ epoch {result.best_epoch} | "
        f"best_test_jaccard={best_test_jaccard:.6f} | "
        f"best_test_pr_auc={best_test_pr_auc:.6f}"
    )


def format_whole_bin_metrics(result: WholeBinRunResult) -> str:
    best_snapshot = result.best_snapshot or {}
    train_eval = best_snapshot.get("train_eval", {}) or {}
    test = best_snapshot.get("test", {}) or {}
    return (
        f"train_jaccard={float(train_eval.get('jaccard', float('nan'))):.6f} | "
        f"val_{result.selection_metric}={result.best_val_metric:.6f} | "
        f"best_test_jaccard={float(test.get('jaccard', float('nan'))):.6f} | "
        f"best_test_f1={float(test.get('f1', float('nan'))):.6f} | "
        f"best_test_pr_auc={float(test.get('pr_auc', float('nan'))):.6f} | "
        f"best_test_roc_auc={float(test.get('roc_auc', float('nan'))):.6f}"
    )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    paths = experiment_output_paths(RESULTS_ROOT, PLOTS_ROOT, EXPERIMENT_NAME)
    results_dir = paths["results_dir"]
    results_jsonl = str(paths["results_jsonl"])
    summary_jsonl = str(paths["summary_jsonl"])
    metadata_jsonl = str(results_dir / "metadata.jsonl")

    results_dir.mkdir(parents=True, exist_ok=True)

    datasets = build_thresholded_whole_bin_datasets(device)
    all_results: list[WholeBinRunResult] = []

    for dataset_name, ds in datasets.items():
        spec = ds.spec()
        append_jsonl(metadata_jsonl, build_dataset_metadata_row(dataset_name, ds))

        upper_triangle_only = use_upper_triangle_for_dataset(dataset_name)
        log_every = 2000 if "lastfm" in str(spec.name).lower() else 50

        base_train_cfg = WholeBinTrainConfig(
            num_nodes=spec.num_nodes,
            lr=1e-3,
            weight_decay=1e-3,
            grad_clip=1.0,
            device=device,
            log_every=log_every,
            tbptt_steps=1,
            decision_threshold=0.5,
            upper_triangle_only= True, #upper_triangle_only,
            include_self_loops=False,
            pos_weight=None,
            auto_pos_weight=True,
            max_auto_pos_weight=50.0,
            selection_metric="jaccard",
        )
        base_model_cfg = build_base_interaction_model_cfg(spec)

        runs = make_runs(
            base_model_cfg,
            seeds=(0,),
            aggregator=("ift", "hopfield", "settransformer", "sum", "deepsets"),
            upd=("ift_update", "tgn_gru", "lnn", "hopfield_update", "hnn"),
            dropout=(0.0,),
            scorer_dropout=(0.0,),
            use_time_features=(False,),
            ift_kappa_param=("softplus",),
            ift_dt=(0.01, 0.05, 0.1, 0.2),
            ift_gamma=(0.0, 0.01, 0.05, 0.1),
            ift_kappa_init=(0.1, 0.5, 1.0, 2.0),
            ift_kappa_cap=(False,),
            ift_kappa_max=(None,),
        )

        allowed_pairs = {
            #("ift", "ift_update"),
            ("sum", "lnn"),
            #("sum", "tgn_gru"),
            # ("deepsets", "tgn_gru"),
        }
        runs = [
            run for run in runs
            if (run.model_cfg.aggregator, run.model_cfg.update) in allowed_pairs
        ]

        dataset_results: list[WholeBinRunResult] = []
        for run in runs:
            run_for_ds = replace(run, name=f"{dataset_name}_{run.name}")
            print(format_starting_run_banner(dataset_name, run_for_ds.name, run.seed))
            print(
                f"upper_triangle_only={upper_triangle_only} | "
                f"selection_metric={base_train_cfg.selection_metric}"
            )

            try:
                result, model, train_cfg = run_one_whole_bin_experiment(
                    ds=ds,
                    spec=spec,
                    base_train_cfg=base_train_cfg,
                    run=run_for_ds,
                    build_model_fn=build_tgn_model,
                    epochs=3,
                    eval_slices=EvalSlices(early_steps=10),
                    save_jsonl_path=results_jsonl,
                    save_summary_path=summary_jsonl,
                    dataset_name=dataset_name,
                )
                dataset_results.append(result)
                all_results.append(result)
                
                print(f"{format_finished_label()} {describe_whole_bin_run_result(result)}")
                print(f"{format_analysis_label()} {format_whole_bin_metrics(result)}")

            except Exception as e:
                print(
                    f"{format_failed_label()} dataset={dataset_name} | run={run_for_ds.name} | seed={run.seed}"
                )
                print(f"Reason: {type(e).__name__}: {e}")
                append_jsonl(
                    results_jsonl,
                    {
                        "dataset": dataset_name,
                        "run": run_for_ds.name,
                        "seed": run.seed,
                        "status": "failed",
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                )

        if not dataset_results:
            print(f"No successful runs for dataset={dataset_name}")
            continue

        print(format_top_runs_header(base_train_cfg.selection_metric, dataset_name))
        for result in sorted(dataset_results, key=lambda r: r.best_val_metric, reverse=True)[:10]:
            print(describe_whole_bin_run_result(result))


if __name__ == "__main__":
    main()