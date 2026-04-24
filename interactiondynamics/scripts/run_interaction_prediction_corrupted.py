from __future__ import annotations

import json
import shutil
from pathlib import Path
from dataclasses import replace

import torch

from core.config import ModelConfig
from eval.evaluate import EvalSlices
from experiments.dataset_stats import build_dataset_metadata_row
from experiments.defaults import (
    build_base_interaction_model_cfg,
    build_base_interaction_train_cfg,
)
from experiments.interaction_prediction_runs import make_runs
from experiments.physical_dataset_registry import build_physical_datasets
from experiments.results_summary import (
    describe_interaction_run_result,
    format_analysis_label,
    format_failed_label,
    format_finished_label,
    format_interaction_metrics,
    format_starting_run_banner,
    format_top_runs_header,
    print_interaction_seed_avg,
    format_recovery_label,
    format_recovery_metrics
)

from models.tgn_model import build_tgn_model
from plotting.interaction_prediction import plot_interaction_run_rankings
from training.interaction_prediction import run_one_experiment
from utils.io import append_jsonl
from utils.output_paths import dataset_group_name, experiment_output_paths
from datasets.jodie import JODIEBinnedDataset, JODIEConfig
from datasets.corrupted import CorruptedEventStreamDataset

def get_clean_ref_name(dataset_name: str) -> str:
    if dataset_name.startswith("wave__"):
        return "wave__clean_ref"
    if dataset_name.startswith("springweb"):
        return "springweb__clean_ref"
    if dataset_name.startswith("charged_particles__") or dataset_name.startswith("charged_particles_"):
        return "charged_particles__clean_ref"
    raise ValueError(f"No clean ref mapping for dataset {dataset_name}")

def main():
    cleared_plot_dirs = set()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    RESULTS_ROOT = Path("/home/akapociu/ift/interactiondynamics/results")
    PLOTS_ROOT = Path("/home/akapociu/ift/interactiondynamics/plots")
    EXPERIMENT_NAME = "interaction_predictions_charged_particles_thresholded_train_distance_misfire_.50_sparse"

    paths = experiment_output_paths(RESULTS_ROOT, PLOTS_ROOT, EXPERIMENT_NAME)

    results_dir = paths["results_dir"]
    plot_dir = paths["plots_dir"]
    results_jsonl = str(paths["results_jsonl"])
    summary_jsonl = str(paths["summary_jsonl"])
    metadata_jsonl = str(results_dir / "metadata.jsonl")

    results_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    md22_npz_paths = [
        "/home/akapociu/ift/interactiondynamics/data/MD_DATA/naphthalene.npz",
        "/home/akapociu/ift/interactiondynamics/data/MD_DATA/stachyose.npz",
        "/home/akapociu/ift/interactiondynamics/data/MD_DATA/uracil.npz",
    ]

    physical_datasets = build_physical_datasets(
        device=device,
        md22_npz_paths=md22_npz_paths,
        include=("charged_particles",),
        include_clean_references=True,  
    )

    # jodie_datasets = {
    #     # "wikipedia": JODIEBinnedDataset(
    #     #     JODIEConfig(root="./data/JODIE", name="Wikipedia", device=device)
    #     # ),
    #     # "reddit": JODIEBinnedDataset(
    #     #     JODIEConfig(root="./data/JODIE", name="Reddit", device=device)
    #     # ),
    #     # "mooc": JODIEBinnedDataset(
    #     #     JODIEConfig(root="./data/JODIE", name="MOOC", device=device)
    #     # ),
    #     "lastfm": JODIEBinnedDataset(
    #         JODIEConfig(root="./data/JODIE", name="LastFM", device=device)
    #     ),
    # }
    clean_refs = {name: ds for name, ds in physical_datasets.items() if name.endswith("__clean_ref")}
    datasets = {name: ds for name, ds in physical_datasets.items() if not name.endswith("__clean_ref")}
    
    # clean_datasets = {
    #         **physical_datasets,
    #         #**jodie_datasets,
    #     }
    
    # jodie_datasets = {
    #     name: ds
    #     for name, ds in jodie_datasets.items()
    #     if "__clean_ref" not in name
    # }

    

    CORRUPTION = dict(
        drop_real_prob=0.50,
        add_fake_ratio=0.0,
        corrupt_splits=("train", ),
        seed=17,
        fake_feature_mode="zeros",
        avoid_self_loops=True,
        min_keep_per_nonempty_bin=1,
    )

    all_results = []

    for dataset_name, ds in datasets.items():
        corruption_tag = (
            f"drop={CORRUPTION['drop_real_prob']:.2f}|"
            #f"fake={CORRUPTION['add_fake_ratio']:.2f}|"
            f"splits={'-'.join(CORRUPTION['corrupt_splits'])}"
        )
        dataset_name = f"{dataset_name}_{corruption_tag}"

        ds = CorruptedEventStreamDataset(ds, **CORRUPTION)
        spec = ds.spec()
        # print("Dataset spec:")
        # print(
        #     json.dumps(
        #         {
        #             "name": spec.name,
        #             "num_nodes": spec.num_nodes,
        #             "event_dim": spec.event_dim,
        #             "num_events": spec.num_events,
        #             "num_bins": spec.num_bins,
        #             "extra": spec.extra,
        #         },
        #         indent=2,
        #         default=str,
        #     )
        # )

        append_jsonl(metadata_jsonl, build_dataset_metadata_row(dataset_name, ds))

        base_train_cfg = build_base_interaction_train_cfg(spec, device)
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
            ift_dt=(0.05,),
            ift_gamma=(0.0,),
            ift_kappa_init=(1.0,),
            ift_kappa_cap=(False,),
            ift_kappa_max=(None,),
        )

        # allowed_pairs = {
        #     #("sum", "tgn_gru"),
        #     ("ift", "ift_update"),
        #     #("ift", "hopfield_update"),
        # }

        # if allowed_pairs is not None:
        #     runs = [
        #         run for run in runs
        #         if (run.model_cfg.aggregator, run.model_cfg.update) in allowed_pairs
        #     ]

        dataset_results = []

        for run in runs:
            run_for_ds = replace(run, name=f"{dataset_name}_{run.name}")
            print(format_starting_run_banner(dataset_name, run_for_ds.name, run.seed))

            try:
                clean_ref_name = get_clean_ref_name(dataset_name)
                clean_ds = clean_refs[clean_ref_name]

                result, model, train_cfg = run_one_experiment(
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
                    clean_ds=clean_ds,
                    corruption_cfg=CORRUPTION,
                )
                dataset_results.append(result)
                all_results.append(result)

                print(f"{format_finished_label()} {describe_interaction_run_result(result)}")
                recovery_to_print = result.recovery
                if recovery_to_print is not None:
                    print(f"{format_recovery_label()} {format_recovery_metrics(recovery_to_print)}")
                    
                print(f"{format_analysis_label()} {format_interaction_metrics(result)}")

            except Exception as e:
                print(
                    f"{format_failed_label()} dataset={dataset_name} | run={run.name} | seed={run.seed}"
                )
                print(f"Reason: {type(e).__name__}: {e}")

                append_jsonl(
                    results_jsonl,
                    {
                        "dataset": dataset_name,
                        "run": run.name,
                        "seed": run.seed,
                        "status": "failed",
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                )

        if not dataset_results:
            print(f"No successful runs for dataset={dataset_name}")
            continue

        print(format_top_runs_header("mrr", dataset_name))
        results_sorted = sorted(dataset_results, key=lambda r: r.best_val_mrr, reverse=True)
        for result in results_sorted[:10]:
            print(describe_interaction_run_result(result))

        dataset_plot_dir = plot_dir / dataset_group_name(dataset_name)
        if dataset_plot_dir not in cleared_plot_dirs:
            shutil.rmtree(dataset_plot_dir, ignore_errors=True)
            cleared_plot_dirs.add(dataset_plot_dir)
        dataset_plot_dir.mkdir(parents=True, exist_ok=True)

        ranking_plot_path = dataset_plot_dir / "interaction_prediction_mrr_rankings.png"
        plot_interaction_run_rankings(
            dataset_results,
            out_path=str(ranking_plot_path),
            dataset_name=dataset_name,
        )
        print(f"Saved interaction summary plot: {ranking_plot_path}")

    # print_interaction_seed_avg(all_results)


if __name__ == "__main__":
    main()