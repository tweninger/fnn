import torch
from dataclasses import asdict

#just importing dataclasses and functions from train.py so i don't have to copy and paste
from train import TrainConfig, SweepRun, run_one_experiment

from core.config import ModelConfig
from data.jodie import JODIEBinnedDataset, JODIEConfig
from eval.evaluate import EvalSlices
from models.tgn_model import build_tgn_model

"""
- let's get a baseline w/ TGN GRU and sum...
- records one JSON per epoch w/ run name, seed, model config, epoch number, train/val/test metrics
- we have that other JSON now with the final run summary
"""
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    ds = JODIEBinnedDataset(
        JODIEConfig(
            root="./data/JODIE",
            name="Wikipedia",
            device=device,
        )
    )
    spec = ds.spec()

    print("Loaded dataset spec:", spec)
    print("Using device:", device)

    # build training config
    base_train_cfg = TrainConfig(
        num_nodes=spec.num_nodes,
        num_neg=20,
        tbptt_steps=1,
        log_every=10,
        device=device,
        weight_decay=1e-3,
        lr=1e-3,
    )

    # build baseline model
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
        update="tgn_gru",
    )

    
    # wrap it up into one sweeprun
    # one experiment instance w this specific model, and this here seed
    run = SweepRun(
        name="baseline_one_run",
        model_cfg=base_model_cfg, #swap it
        seed=0,
    )

    # call it! 
    result = run_one_experiment(
        ds=ds,
        spec=spec,
        base_train_cfg=base_train_cfg,
        run=run,
        build_model_fn=build_tgn_model,
        epochs=6, # 1 or 6 atm
        eval_slices=EvalSlices(early_steps=10),
        save_jsonl_path="results/baseline_results_6ep.jsonl", # results go into this file/folder
        save_summary_path="results/baseline_summary_6ep.jsonl",
    )

    # yay hearts
    print("\n" + "❤️ " * 10)
    print("=== Test finished ===")
    print(result)
    print("❤️ " * 10)


if __name__ == "__main__":
    main()