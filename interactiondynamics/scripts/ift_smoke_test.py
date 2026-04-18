import torch
from dataclasses import asdict

#just importing dataclasses and functions from train.py so i don't have to copy and paste
from training.interaction_prediction import TrainConfig, run_one_experiment
from experiments.interaction_prediction_runs import SweepRun

from core.config import ModelConfig
from datasets import JODIEBinnedDataset, JODIEConfig
from eval.evaluate import EvalSlices
from models.tgn_model import build_tgn_model
from datasets import NBodyConfig, NBodyDataset
from datasets import SpringMassConfig, SpringMassDataset

"""
- just one run/one epoch with IFT x IFT to check if this works
- records one JSON per epoch w/ run name, seed, model config, epoch number, train/val/test metrics
- we have that other JSON now with the final run summary

"""
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset

    ds = NBodyDataset(NBodyConfig(device=device))
    # ds = SpringMassDataset(SpringMassConfig(device=device))
    # ds = JODIEBinnedDataset(
    #     JODIEConfig(
    #         root="./data/JODIE",
    #         name="Wikipedia",
    #         device=device,
    #     )
    # )
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

    # build ift model
    model_cfg = ModelConfig(
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
        ift_kappa_param="softplus",
        ift_dt=0.05,
        ift_gamma=0.0,
        ift_kappa=1.0,
        ift_kappa_cap=False,
        ift_kappa_max=None,
    )

    # wrap it up into one sweeprun
    # one experiment instance w this specific model, and this here seed
    run = SweepRun(
        name="ift_smoke_test",
        model_cfg=model_cfg,
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
        save_jsonl_path="results/newbl_results_nbody.jsonl", # results go into this file/folder
        save_summary_path="results/newbl_summary_nbody.json",
    )

    # yay hearts
    print("\n" + "❤️ " * 10)
    print("=== Test finished ===")
    print(result)
    print("❤️ " * 10)


if __name__ == "__main__":
    main()