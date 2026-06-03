from __future__ import annotations

import shutil
from pathlib import Path
from dataclasses import replace
import traceback

import torch

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
)

from models.tgn_model import build_tgn_model
from plotting.interaction_prediction import plot_interaction_run_rankings
from training.interaction_prediction import run_one_experiment
from utils.io import append_jsonl
from utils.output_paths import dataset_group_name, experiment_output_paths
from datasets.jodie import JODIEBinnedDataset, JODIEConfig


def _print_count_summary(
    values: list[int],
    *,
    label: str,
    split: str,
    max_bins: int,
) -> None:
    if not values:
        print(f"[{label}] split={split}: no samples in first {max_bins} bins")
        return
    n = len(values)
    mean = sum(values) / n
    vals = sorted(values)
    median = vals[n // 2]
    print(
        f"[{label}] split={split} | src-bin samples={n} | "
        f"mean={mean:.2f} median={median} min={vals[0]} max={vals[-1]}"
    )


def print_avg_dst_per_src(ds, split: str = "train", max_bins: int = 200) -> None:
    """
    Per (src, bin) stats over the first max_bins bins.

    - edges_per_src: event **row** count (what ranking / directed_event drop use)
    - distinct_dst_per_src: unique dst ids (old stat; repeats to same dst count as 1)
    - distinct_pair_per_src: unique directed (src, dst) keys (same as distinct_dst when src fixed)
    """
    per_src_rows: list[int] = []
    per_src_distinct_dst: list[int] = []
    per_src_distinct_pair: list[int] = []

    for i, batch in enumerate(ds.bins(split)):
        if i >= max_bins:
            break
        s, d = batch.src, batch.dst
        if s.numel() == 0:
            continue
        for u in s.unique().tolist():
            mask = s == u
            dst_u = d[mask]
            per_src_rows.append(int(mask.sum().item()))
            per_src_distinct_dst.append(int(dst_u.unique().numel()))
            pair_keys = u * int(ds.spec().num_nodes) + dst_u.long()
            per_src_distinct_pair.append(int(pair_keys.unique().numel()))

    if not per_src_rows:
        print(f"[src/bin debug] split={split}: no edges in first {max_bins} bins")
        return

    _print_count_summary(
        per_src_rows,
        label="edges_per_src (rows)",
        split=split,
        max_bins=max_bins,
    )
    _print_count_summary(
        per_src_distinct_dst,
        label="distinct_dst_per_src",
        split=split,
        max_bins=max_bins,
    )
    if any(r != d for r, d in zip(per_src_rows, per_src_distinct_dst)):
        _print_count_summary(
            per_src_distinct_pair,
            label="distinct_pair_per_src",
            split=split,
            max_bins=max_bins,
        )


def main():
    cleared_plot_dirs = set()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    RESULTS_ROOT = Path("/home/akapociu/ift/interactiondynamics/results")
    PLOTS_ROOT = Path("/home/akapociu/ift/interactiondynamics/plots")
    EXPERIMENT_NAME = "test"

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
        threshold_splits_options=(("train", "val", "test"),),
    )


    jodie_datasets = {
        "wikipedia": JODIEBinnedDataset(
            JODIEConfig(root="./data/JODIE", name="Wikipedia", device=device)
        ),
        # "reddit": JODIEBinnedDataset(
        #     JODIEConfig(root="./data/JODIE", name="Reddit", device=device)
        # ),
        # "mooc": JODIEBinnedDataset(
        #     JODIEConfig(root="./data/JODIE", name="MOOC", device=device)
        # ),
        # "lastfm": JODIEBinnedDataset(
        #     JODIEConfig(root="./data/JODIE", name="LastFM", device=device)
        # ),
    }

    datasets = {
        **physical_datasets,
        #**jodie_datasets,
    }

    all_results = []

    for dataset_name, ds in datasets.items():
        spec = ds.spec()
        print(f"\n=== {dataset_name} ===")
        print_avg_dst_per_src(ds, split="train")
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
            seeds=(0, ),
            aggregator=("ift", "hopfield", "settransformer", "sum", "deepsets"),
            upd=("ift_update", "tgn_gru", "lnn", "hopfield_update", "hnn"),
            dropout=(0.0,),
            scorer_dropout=(0.0,),
            use_time_features=(False,),
            ift_kappa_param=("softplus",),
            ift_dt=(0.1,),
            ift_gamma=(0.05, ),
            ift_kappa_init=(0.1,),
            ift_kappa_cap=(False,),
            ift_kappa_max=(5.0,),
        )   

        allowed_pairs = {
            #("sum", "tgn_gru"),
            ("ift", "ift_update"),
        }

        if allowed_pairs is not None:
            runs = [
                run for run in runs
                if (run.model_cfg.aggregator, run.model_cfg.update) in allowed_pairs
            ]

        dataset_results = []

        for run in runs:
            run_for_ds = replace(run, name=f"{dataset_name}_{run.name}")
            print(format_starting_run_banner(dataset_name, run_for_ds.name, run.seed))

            try:
                result, _, _ = run_one_experiment(
                    ds=ds,
                    spec=spec,
                    base_train_cfg=base_train_cfg,
                    run=run_for_ds,
                    build_model_fn=build_tgn_model,
                    epochs=5,
                    eval_slices=EvalSlices(early_steps=10),
                    save_jsonl_path=results_jsonl,
                    save_summary_path=summary_jsonl,
                    dataset_name=dataset_name,
                )
                dataset_results.append(result)
                all_results.append(result)

                
                print(f"{format_finished_label()} {describe_interaction_run_result(result)}")
                print(f"{format_analysis_label()} {format_interaction_metrics(result)}")

            except Exception as e:
                print(
                    f"{format_failed_label()} dataset={dataset_name} | run={run.name} | seed={run.seed}"
                )
                print(f"Reason: {type(e).__name__}: {e}")
                traceback.print_exc()

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