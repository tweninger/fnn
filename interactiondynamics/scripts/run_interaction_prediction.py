from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import os
import torch

from core.config import ModelConfig
from eval.evaluate import EvalSlices
from experiments.dataset_stats import build_dataset_metadata_row
from experiments.interaction_prediction_runs import make_runs
from experiments.physical_dataset_registry import build_physical_datasets
from experiments.results_summary import print_seed_avg
from models.tgn_model import build_tgn_model
from training.interaction_prediction import TrainConfig, run_one_experiment
from utils.io import append_jsonl
from utils.output_paths import experiment_output_paths


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results", exist_ok=True)

    RESULTS_ROOT = Path("/home/akapociu/ift/interactiondynamics/results")
    PLOTS_ROOT = Path("/home/akapociu/ift/interactiondynamics/plots")
    EXPERIMENT_NAME = "physical_systems_sweep"

    paths = experiment_output_paths(RESULTS_ROOT, PLOTS_ROOT, EXPERIMENT_NAME)

    results_jsonl = str(paths["results_jsonl"])
    summary_jsonl = str(paths["summary_jsonl"])
    metadata_jsonl = str(paths["results_dir"] / "metadata.jsonl")

    md22_npz_paths = [
        "/home/akapociu/ift/interactiondynamics/data/MD_DATA/naphthalene.npz",
        "/home/akapociu/ift/interactiondynamics/data/MD_DATA/stachyose.npz",
        "/home/akapociu/ift/interactiondynamics/data/MD_DATA/uracil.npz",
    ]

    # or do this instead:
    # md22_npz_paths = sorted(
    #     str(p) for p in Path("/home/akapociu/ift/interactiondynamics/data/MD_DATA").glob("*.npz")
    # )

    datasets = build_physical_datasets(
        device=device,
        md22_npz_paths=md22_npz_paths,
        include=("wave",),
        #include=("spring_ring", "spring_mass"),

    )

    results = []

    for dataset_name, ds in datasets.items():
        print(f"\n=== DATASET: {dataset_name} ===")

        spec = ds.spec()

        # write one metadata row per dataset
        metadata_row = build_dataset_metadata_row(dataset_name, ds)
        append_jsonl(metadata_jsonl, metadata_row)

        base_train_cfg = TrainConfig(
            num_nodes=spec.num_nodes,
            num_neg=20,
            tbptt_steps=1,
            log_every=50,
            device=device,
            weight_decay=1e-3,
            lr=1e-3,
        )

        base_model_cfg = ModelConfig(
            node_dim=128,
            msg_dim=128,
            event_dim=spec.event_dim,
            scorer="mlp",
            scorer_hidden=256,
            aggregator="sum",
            use_time_features=False,
            dropout=0.0,
            scorer_dropout=0.0,
            encoder_hidden=256,
        )

        runs = make_runs(
            base_model_cfg,
            seeds=(0,),   # change back to (0, 42, 123) whenever
            aggregator=("ift", "hopfield","settransformer", "sum", "deepsets"),
            upd = ("ift_update", "tgn_gru", "lnn", "hopfield_update", "hnn",),
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
        #allowed_pairs = None
        allowed_pairs = { # set to None for all pairs
            ("sum", "tgn_gru"),
            #("ift", "ift_update"),
        }

        if allowed_pairs is not None:
            runs = [ 
                run for run in runs
                if (run.model_cfg.aggregator, run.model_cfg.update) in allowed_pairs
            ]

        for run in runs:
            run_for_ds = replace(run, name=f"{dataset_name}_{run.name}")

            print(f"\n🧹🧹 === Running {run_for_ds.name} === 🧹🧹")

            try:
                result = run_one_experiment(
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
                results.append(asdict(result))

            except Exception as e:
                print(
                    f"FAILED: dataset={dataset_name}, run={run.name}, "
                    f"seed={run.seed}, error={type(e).__name__}: {e}"
                )

    by_dataset = defaultdict(list)
    for r in results:
        by_dataset[r.get("dataset", "UNKNOWN")].append(r)

    print("\n=== Best runs per dataset ===")
    for dataset, rows in by_dataset.items():
        rows.sort(key=lambda r: r["best_val_mrr"], reverse=True)
        print(f"\n--- {dataset} ---")

        for r in rows[:5]:
            best_test = r["best_snapshot"]["test"]["mrr"]
            print(
                f"{r['name']} | seed={r['seed']} | "
                f"best_val={r['best_val_mrr']:.4f} "
                f"@epoch {r['best_epoch']} | best_test={best_test:.4f}"
            )

    print_seed_avg(results)


if __name__ == "__main__":
    main()