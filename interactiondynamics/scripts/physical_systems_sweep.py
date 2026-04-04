from dataclasses import asdict, replace
from pathlib import Path
from collections import defaultdict
import os
import json

import numpy as np
import torch

from train import TrainConfig, run_one_experiment, make_runs
from core.config import ModelConfig
from eval.evaluate import EvalSlices
from models.tgn_model import build_tgn_model

# physical system datasets
from data.three_body_binned import ThreeBodyBinnedConfig, ThreeBodyBinnedDataset
from data.one_dimension_wave_binned import WaveEquationBinnedConfig, WaveEquationBinnedDataset
from data.md22_binned import MD22BinnedConfig, MD22BinnedDataset
from data.nbody_continuous import ChargedParticlesBinnedConfig, ChargedParticlesBinnedDataset
from data.spring_mass import SpringMassConfig, SpringMassDataset
from data.spring_ring import SpringRing2DConfig, SpringRing2DDataset


# --------------------------------------------------
# small JSONL helper
# --------------------------------------------------

def append_jsonl(path: str, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


# --------------------------------------------------
# console summary helper
# --------------------------------------------------

def print_seed_avg(results):
    grouped = defaultdict(list)

    for r in results:
        run_name = r["name"]   # already includes dataset prefix
        best_test = r["best_snapshot"]["test"]["mrr"]
        grouped[run_name].append((r["best_val_mrr"], best_test))

    print("\n=== Mean across seeds ===")
    for run_name, vals in grouped.items():
        val_mrrs = np.array([x[0] for x in vals], dtype=float)
        test_mrrs = np.array([x[1] for x in vals], dtype=float)

        print(f"\n=== {run_name} ===")
        print(f"n_seeds:           {len(vals)}")
        print(f"avg best val mrr:  {val_mrrs.mean():.4f}")
        print(f"std best val mrr:  {val_mrrs.std(ddof=1) if len(vals) > 1 else 0.0:.4f}")
        print(f"avg best test mrr: {test_mrrs.mean():.4f}")
        print(f"std best test mrr: {test_mrrs.std(ddof=1) if len(vals) > 1 else 0.0:.4f}")


# --------------------------------------------------
# metadata helpers
# --------------------------------------------------

def edge_set(batch):
    return set(zip(batch.src.cpu().tolist(), batch.dst.cpu().tolist()))


def summarize_edge_changes(ds, split="train"):
    bins = list(ds.bins(split))
    edge_sets = [edge_set(batch) for batch in bins]

    if len(edge_sets) < 2:
        return {
            "num_transitions": 0,
            "mean_jaccard": None,
            "median_jaccard": None,
            "min_jaccard": None,
            "max_jaccard": None,
            "mean_edges_added": None,
            "mean_edges_removed": None,
        }

    jaccards = []
    added_counts = []
    removed_counts = []

    for prev, curr in zip(edge_sets[:-1], edge_sets[1:]):
        inter = len(prev & curr)
        union = len(prev | curr)
        jaccard = inter / union if union > 0 else 1.0

        added = len(curr - prev)
        removed = len(prev - curr)

        jaccards.append(jaccard)
        added_counts.append(added)
        removed_counts.append(removed)

    jaccards = np.array(jaccards, dtype=float)
    added_counts = np.array(added_counts, dtype=float)
    removed_counts = np.array(removed_counts, dtype=float)

    return {
        "num_transitions": int(len(jaccards)),
        "mean_jaccard": float(jaccards.mean()),
        "median_jaccard": float(np.median(jaccards)),
        "min_jaccard": float(jaccards.min()),
        "max_jaccard": float(jaccards.max()),
        "mean_edges_added": float(added_counts.mean()),
        "mean_edges_removed": float(removed_counts.mean()),
    }


def summarize_event_counts(ds, split="train"):
    counts = np.array([int(batch.src.numel()) for batch in ds.bins(split)], dtype=float)

    if counts.size == 0:
        return {
            "num_bins": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "percentiles": {},
        }

    return {
        "num_bins": int(len(counts)),
        "min": int(counts.min()),
        "max": int(counts.max()),
        "mean": float(counts.mean()),
        "median": float(np.median(counts)),
        "std": float(counts.std()),
        "percentiles": {
            "p0": float(np.percentile(counts, 0)),
            "p5": float(np.percentile(counts, 5)),
            "p10": float(np.percentile(counts, 10)),
            "p25": float(np.percentile(counts, 25)),
            "p50": float(np.percentile(counts, 50)),
            "p75": float(np.percentile(counts, 75)),
            "p90": float(np.percentile(counts, 90)),
            "p95": float(np.percentile(counts, 95)),
            "p100": float(np.percentile(counts, 100)),
        },
    }


def build_dataset_metadata_row(dataset_name, ds):
    spec = ds.spec()

    return {
        "dataset": dataset_name,
        "spec": {
            "name": spec.name,
            "num_nodes": int(spec.num_nodes),
            "event_dim": int(spec.event_dim),
            "num_events": None if spec.num_events is None else int(spec.num_events),
            "num_bins": None if spec.num_bins is None else int(spec.num_bins),
            "extra": spec.extra if spec.extra is not None else {},
        },
        "splits": {
            "train": {
                "event_counts": summarize_event_counts(ds, split="train"),
                "edge_changes": summarize_edge_changes(ds, split="train"),
            },
            "val": {
                "event_counts": summarize_event_counts(ds, split="val"),
                "edge_changes": summarize_edge_changes(ds, split="val"),
            },
            "test": {
                "event_counts": summarize_event_counts(ds, split="test"),
                "edge_changes": summarize_edge_changes(ds, split="test"),
            },
        },
    }


# --------------------------------------------------
# dataset registry / builder
# --------------------------------------------------

def build_physical_datasets(
    device,
    md22_npz_paths=(),
    include=("nbody", "wave", "threebody", "spring_ring", "spring_mass", "md22"),
):
    datasets = {}

    if "nbody" in include:
        cfg = ChargedParticlesBinnedConfig(
            name="nbody",
            device=device,
        )
        datasets[cfg.name] = ChargedParticlesBinnedDataset(cfg)

    if "wave" in include:
        cfg = WaveEquationBinnedConfig(
            name="wave",
            device=device,
        )
        datasets[cfg.name] = WaveEquationBinnedDataset(cfg)

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
            stem = Path(npz_path).stem  # e.g. uracil, naphthalene, stachyose

            cfg = MD22BinnedConfig(
                name=stem,               
                npz_path=str(npz_path),
                device=device,
            )
            datasets[cfg.name] = MD22BinnedDataset(cfg)

    return datasets


# --------------------------------------------------
# main
# --------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results", exist_ok=True)

    results_jsonl = "results/physics_sweep_results.jsonl"
    summary_jsonl = "results/physics_sweep_summary.jsonl"
    metadata_jsonl = "results/physics_sweep_metadata.jsonl"

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
        include=("nbody", "wave", "threebody", "spring_ring", "spring_mass", "md22"),
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
            seeds=(0, 42, 123),   # change back to (0, 42, 123) whenever
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
        allowed_pairs = None
        # allowed_pairs = { # set to None for all pairs
        #     ("sum", "tgn_gru"),
        #     ("ift", "ift_update"),
        # }

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