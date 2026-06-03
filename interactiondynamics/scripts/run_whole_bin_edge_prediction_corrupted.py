from __future__ import annotations

from pathlib import Path

import torch

from datasets.corrupted import CorruptedEventStreamDataset
from eval.evaluate import EvalSlices
from experiments.dataset_stats import build_dataset_metadata_row
from experiments.defaults import build_base_interaction_model_cfg
from experiments.interaction_prediction_runs import make_runs
from experiments.whole_bin_common import (
    DEFAULT_WHOLE_BIN_CORRUPTION,
    format_whole_bin_corruption_tag,
    build_thresholded_whole_bin_datasets,
    describe_whole_bin_run_result,
    format_analysis_label,
    format_failed_label,
    format_finished_label,
    format_starting_run_banner,
    format_top_runs_header,
    format_whole_bin_metrics,
    use_upper_triangle_for_dataset,
)
from models.tgn_model import build_tgn_model
from training.whole_bin_edge_prediction import (
    WholeBinRunResult,
    WholeBinTrainConfig,
    run_one_whole_bin_experiment,
)
from utils.io import append_jsonl
from utils.output_paths import experiment_output_paths

RESULTS_ROOT = Path("/home/akapociu/ift/interactiondynamics/results")
PLOTS_ROOT = Path("/home/akapociu/ift/interactiondynamics/plots")
EXPERIMENT_NAME = "test"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    paths = experiment_output_paths(RESULTS_ROOT, PLOTS_ROOT, EXPERIMENT_NAME)
    results_dir = paths["results_dir"]
    results_jsonl = str(paths["results_jsonl"])
    summary_jsonl = str(paths["summary_jsonl"])
    metadata_jsonl = str(results_dir / "metadata.jsonl")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Same knobs as run_interaction_prediction_corrupted.py (CorruptedEventStreamDataset).
    CORRUPTION = dict(DEFAULT_WHOLE_BIN_CORRUPTION)
    # CORRUPTION.update(corrupt_unit="node_block", block_node_select="high_degree")

    ds_corruption = {k: v for k, v in CORRUPTION.items() if k != "recovery_splits"}
    corruption_tag = format_whole_bin_corruption_tag(CORRUPTION)

    for dataset_name, clean_ds in build_thresholded_whole_bin_datasets(device).items():
        context_ds = CorruptedEventStreamDataset(clean_ds, **ds_corruption)
        run_dataset_name = f"{dataset_name}_{corruption_tag}"

        spec = clean_ds.spec()
        append_jsonl(metadata_jsonl, {
            **build_dataset_metadata_row(run_dataset_name, clean_ds),
            "context_corruption": CORRUPTION,
        })

        upper_triangle_only = use_upper_triangle_for_dataset(dataset_name)

        base_train_cfg = WholeBinTrainConfig(
            num_nodes=spec.num_nodes,
            lr=1e-3,
            weight_decay=1e-3,
            device=device,
            log_every=50,
            decision_threshold=0.5,
            upper_triangle_only=upper_triangle_only,
            selection_metric="micro_pr_auc", # or pr_auc for macro
            tune_threshold=True,
            threshold_metric="f1",
        )
        base_model_cfg = build_base_interaction_model_cfg(spec)

        runs = make_runs(
            base_model_cfg,
            seeds=(0,),
            aggregator=("ift", "sum"),
            upd=("ift_update", "tgn_gru"),
            dropout=(0.0,),
            scorer_dropout=(0.0,),
            use_time_features=(False,),
            ift_kappa_param=("softplus",),
            ift_dt=(0.1,),
            ift_gamma=(0.02,),
            ift_kappa_init=(0.75,),
            ift_kappa_cap=(False,),
            ift_kappa_max=(5.0,),
        )

        allowed_pairs = {
            ("ift", "ift_update"), 
            ("sum", "tgn_gru")
        }

        runs = [r for r in runs if (r.model_cfg.aggregator, r.model_cfg.update) in allowed_pairs]

        dataset_results: list[WholeBinRunResult] = []

        for run in runs:
            print(format_starting_run_banner(run_dataset_name, run.name, run.seed))

            try:
                result, _, _ = run_one_whole_bin_experiment(
                    ds=context_ds,
                    spec=spec,
                    base_train_cfg=base_train_cfg,
                    run=run,
                    build_model_fn=build_tgn_model,
                    epochs=4,
                    eval_slices=EvalSlices(early_steps=10),
                    save_jsonl_path=results_jsonl,
                    save_summary_path=summary_jsonl,
                    dataset_name=run_dataset_name,
                    clean_ds=clean_ds,
                    corruption_cfg=CORRUPTION,
                )
                dataset_results.append(result)
                print(f"{format_finished_label()} {describe_whole_bin_run_result(result)}")
                print(f"{format_analysis_label()} {format_whole_bin_metrics(result)}")

            except Exception as e:
                print(f"{format_failed_label()} dataset={run_dataset_name} | run={run.name}")
                print(f"Reason: {type(e).__name__}: {e}")
                append_jsonl(results_jsonl, {
                    "dataset": run_dataset_name,
                    "run": run.name,
                    "seed": run.seed,
                    "status": "failed",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                })

        if dataset_results:
            print(format_top_runs_header(base_train_cfg.selection_metric, run_dataset_name))
            for result in sorted(dataset_results, key=lambda r: r.best_val_metric, reverse=True)[:10]:
                print(describe_whole_bin_run_result(result))


if __name__ == "__main__":
    main()
