from dataclasses import asdict, replace
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

# add make_runs function from train.py for our sweepingggg
from train import TrainConfig, SweepRun, run_one_experiment, make_runs
from core.config import ModelConfig
from data.jodie import JODIEBinnedDataset, JODIEConfig
from eval.evaluate import EvalSlices
from models.tgn_model import build_tgn_model
from collections import defaultdict
from data.nbody_continuous import NBodyConfig, NBodyDataset
from data.three_body_binned import ThreeBodyBinnedConfig, ThreeBodyBinnedDataset
from data.one_dimension_wave_binned import WaveEquationBinnedDataset, WaveEquationBinnedConfig
from data.md22_binned import MD22BinnedConfig, MD22BinnedDataset

"""
- small sweep again, but we are checking across all JODIE datasets
- failure catching here too
- again the same JSON results and summary are outputted for each run + dataset is noted too
"""

def print_seed_avg(results):

    grouped = defaultdict(list)

    for r in results:
        run_name = r["name"]   # includes dataset prefix already
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

def edge_set(batch):
    return set(zip(batch.src.cpu().tolist(), batch.dst.cpu().tolist()))

def summarize_edge_changes(ds, split="train"):
    bins = list(ds.bins(split))
    edge_sets = [edge_set(batch) for batch in bins]

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

    jaccards = np.array(jaccards)
    added_counts = np.array(added_counts)
    removed_counts = np.array(removed_counts)

    print(f"split={split}")
    print(f"mean Jaccard(prev,curr): {jaccards.mean():.4f}")
    print(f"median Jaccard(prev,curr): {np.median(jaccards):.4f}")
    print(f"min Jaccard(prev,curr): {jaccards.min():.4f}")
    print(f"max Jaccard(prev,curr): {jaccards.max():.4f}")
    print(f"mean edges added: {added_counts.mean():.2f}")
    print(f"mean edges removed: {removed_counts.mean():.2f}")

    return jaccards, added_counts, removed_counts


def summarize_event_counts(ds, split="train"):
    counts = [int(batch.src.numel()) for batch in ds.bins(split)]
    counts = np.array(counts)

    print(f"split={split}")
    print(f"num bins: {len(counts)}")
    print(f"min:     {counts.min()}")
    print(f"max:     {counts.max()}")
    print(f"mean:    {counts.mean():.2f}")
    print(f"median:  {np.median(counts):.2f}")
    print(f"std:     {counts.std():.2f}")

    for q in [0, 5, 10, 25, 50, 75, 90, 95, 100]:
        print(f"p{q:>3}:   {np.percentile(counts, q):.2f}")

    return counts


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    # dataset_names = ("Wikipedia", "Reddit", "MOOC", "LastFM")
    dataset_names = ("molecular_dynamics",)
    #loop over all 4 datasets in JODIE
    for dataset_name in dataset_names:
        ds = MD22BinnedDataset(MD22BinnedConfig(device=device))
        # ds = WaveEquationBinnedDataset(WaveEquationBinnedConfig(device=device))
        # ds = ThreeBodyBinnedDataset(ThreeBodyBinnedConfig(device=device))
        # ds = NBodyDataset(NBodyConfig(device=device))
        # ds = JODIEBinnedDataset(
        #     JODIEConfig(root="./data/JODIE", name=dataset_name, device=device)
        # )
        
        spec = ds.spec()

        # debugging bc we might just be reusing the same dataset over and over again
        print(ds.cfg)
        print(ds.spec())
        train_batches = list(ds.bins("train"))
        print("num train bins:", len(train_batches))
        print("first batch num events:", train_batches[0].src.numel())
        print("first batch feature shape:", train_batches[0].features.shape)
        train_counts = summarize_event_counts(ds, "train")
        val_counts = summarize_event_counts(ds, "val")
        test_counts = summarize_event_counts(ds, "test")
        jaccards, added, removed = summarize_edge_changes(ds, "train")
        for i, batch in enumerate(ds.bins("train")):
            print(i, int(batch.src.numel()))
            if i >= 19:
                break
        base_train_cfg = TrainConfig(
            num_nodes=spec.num_nodes,
            num_neg=20,
            tbptt_steps=1,
            log_every=50, #if dataset_name == "LastFM" else 50, #LastFM -> way more events
            device=device,
            weight_decay=1e-3,
            lr=1e-3,
        )

        base_model_cfg = ModelConfig(
            node_dim=128,
            msg_dim=128, # size of message representation
            event_dim=spec.event_dim, # event feature dimension from ds
            scorer="mlp", #mlp scoring head/or just src/dst with dot
            scorer_hidden=256, # hidden width of scorer
            aggregator="sum",          # baseline
            use_time_features=False,
            dropout=0.0, # none by default? ok
            scorer_dropout=0.0,
            encoder_hidden=256,
        )

        #again how we're getting out sweeping sweep sweep sweep! 🧹🧹🧹
        runs = make_runs(
            base_model_cfg,
            seeds=(0, 42, 123), # freestyling here, 0, 42, 123
            aggregator=("ift", "sum"),
            upd = ("ift_update", "tgn_gru"),
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

        allowed_pairs = {
        ("sum", "tgn_gru"),
        ("ift", "ift_update"),
        }
        
        runs = [
            run for run in runs
            if (run.model_cfg.aggregator, run.model_cfg.update) in allowed_pairs
        ]

        for run in runs:
            run_for_ds = replace(run, name=f"{dataset_name}__{run.name}")
            
            print(f"\n🧹🧹 === Running {run_for_ds.name} === 🧹🧹")

            try: 
                result = run_one_experiment(
                    ds=ds,
                    spec=spec,
                    base_train_cfg=base_train_cfg,
                    run= run_for_ds,
                    build_model_fn=build_tgn_model,
                    epochs=6, # freestyling here
                    eval_slices=EvalSlices(early_steps=10),
                    save_jsonl_path="results/MD_ift_bl_debug_results.jsonl",
                    save_summary_path="results/MD_ift_bl_debug_summary.json",
                    dataset_name=dataset_name,  # added this
                )
                results.append(asdict(result))

            except Exception as e:
                print(f"FAILED: dataset={dataset_name}, run={run.name}, seed={run.seed}, error={type(e).__name__}: {e}")
            
    by_dataset = defaultdict(list)

    for r in results:
        by_dataset[r.get("dataset", "UNKNOWN")].append(r)

    # print entire best run result per dataset into terminal for a quick peek
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