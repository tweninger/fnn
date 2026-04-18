# train.py

import os
# helps make combinations of settings
import itertools
# save results to json
import json
# randomness/numerical computing
import random
import time
import numpy as np
import torch
# type hints for readability??
from typing import Callable, Dict, Iterable, Optional, Sequence
# easy to make parameter-holding classes/turns dataclass -> dictionary
from dataclasses import asdict, dataclass

from traitlets import Any

# how event data is stored
from core.events import EventBatch
# toy synthetic data
from datasets import ToyShiftConfig, ToyShiftDataset
# evaluation logic
from eval.evaluate import EvalSlices, evaluate_stream_sliced
# model config object
from core.config import ModelConfig
# real temporal datasets
from datasets import JODIEBinnedDataset, JODIEConfig # type: ignore
from models.tgn_model import build_tgn_model

from eval.ranking import ranking_loss_and_metrics


# what happens when you run python train.py lol
def main():   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use GPU if avail, if not cpu 

    # dataset sweep loop needs to be added 👋👋

    # choose one:
    # ds = ToyShiftDataset(ToyShiftConfig(device=device))
    # or:
    ds = JODIEBinnedDataset(JODIEConfig(root="./data/JODIE", name="Wikipedia", device=device))

    # get metadata/specification from the dataset
    # prob like number of nodes, event feature dimension, other structural info?
    spec = ds.spec()

    # default training settings for all runs
    base_train_cfg = TrainConfig(
        num_nodes=spec.num_nodes,
        num_neg=20,
        tbptt_steps=1,
        log_every=2000,
        device=device,
        weight_decay=1e-3,
        lr=1e-3,
    )

    # default architecture settings
    base_model_cfg = ModelConfig(
        node_dim=128,
        msg_dim=128, # size of message representation
        event_dim=spec.event_dim, # event feature dimension from ds
        scorer="mlp", #mlp scoring head
        scorer_hidden=256, # hidden width of scorer
        aggregator="sum",          # baseline
        use_time_features=False,
        dropout=0.0, # none by default? ok
        scorer_dropout=0.0,
        encoder_hidden=256, 
    )

    # giant experiment grid... LMAO sweeps across like... a lot
    runs = make_runs(
        base_model_cfg,
        seeds=(0, 42, 123),
        aggregator=("ift", "hopfield","settransformer", "sum", "deepsets"),
        upd = ("ift_update", "tgn_gru", "lnn", "hopfield_update", "hnn",),
        dropout=(0.0,0.1),
        scorer_dropout=(0.0,0.1),
        use_time_features=(False, True),
        ift_kappa_param=("softplus", "exp"),
        ift_dt=(0.01, 0.05, 0.1, 0.2),
        ift_gamma=(0.0, 0.01, 0.05, 0.1),
        ift_kappa_init=(0.1, 0.5, 1.0, 2.0),
        # optional
        ift_kappa_cap=(False, True),
        ift_kappa_max=(1.0, 2.0, 5.0, None),
    )

    # how many runs? i dunno
    print(f"Total runs to execute: {len(runs)}")

    # for each run, print it, run experiment for 6 epochs, save per epoch results, keep final result in memory
    results: list[RunResult] = []
    for run in runs:
        
        try:
            print(run)
            res = run_one_experiment(
                ds=ds,
                spec=spec,
                base_train_cfg=base_train_cfg,
                run=run,
                build_model_fn=build_tgn_model, # type: ignore
                epochs=6,  
                eval_slices=EvalSlices(early_steps=10),
                save_jsonl_path="results/sweep_results.jsonl",
                save_summary_path="results/sweep_summary.jsonl",
            )
        except Exception as e:
            print(f"FAILED: dataset={dataset_name}, run={run.name}, seed={run.seed}, error={type(e).__name__}: {e}")
        results.append(res)

    # sort runs by best val MRR and print top 20/experiment comparison
    # Print summary sorted by best val MRR
    results.sort(key=lambda r: r.best_val_mrr, reverse=True)
    print("\n=== Sweep summary (sorted by best val MRR) ===")
    for r in results[:20]:
        print(
            f"{r.name} | seed={r.seed} | best_val={r.best_val_mrr:.4f} "
            f"@epoch {r.best_epoch} | best_test={r.best_snapshot['test']['mrr']:.4f}"
        )

# if this file is run directly, call main() lol
if __name__ == "__main__":
    main()
