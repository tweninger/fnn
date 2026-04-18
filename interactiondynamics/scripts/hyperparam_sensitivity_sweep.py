from dataclasses import asdict, replace
import argparse
import os
import torch

from training.interaction_prediction import TrainConfig, run_one_experiment
from experiments.interaction_prediction_runs import make_runs
from core.config import ModelConfig
from datasets import JODIEBinnedDataset, JODIEConfig
from eval.evaluate import EvalSlices
from models.tgn_model import build_tgn_model


def main():
    print("entered main", flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["Wikipedia", "Reddit", "MOOC", "LastFM"],
        help="Which JODIE dataset to run",
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    print(f"args = {args}", flush=True)

    os.makedirs(args.results_dir, exist_ok=True)

    dataset_name = args.dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}", flush=True)

    ds = JODIEBinnedDataset(
        JODIEConfig(root="./data/JODIE", name=dataset_name, device=device)
    )
    print("dataset constructed", flush=True)

    spec = ds.spec()
    print(f"spec = {spec}", flush=True)

    base_train_cfg = TrainConfig(
        num_nodes=spec.num_nodes,
        num_neg=20,
        tbptt_steps=1,
        log_every=1000 if dataset_name == "LastFM" else 50,
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
        seeds=(0,),
        aggregator=("ift", "hopfield", "settransformer", "sum", "deepsets"),
        upd=("ift_update", "tgn_gru", "lnn", "hopfield_update", "hnn"),
        dropout=(0.0,0.1), # changing all of these HPs below
        scorer_dropout=(0.0,0.1),
        use_time_features=(False, True),
        ift_kappa_param=("softplus", "exp"),
        ift_dt=(0.01, 0.05, 0.1, 0.2),
        ift_gamma=(0.0, 0.01, 0.05, 0.1),
        ift_kappa_init=(0.1, 0.5, 1.0, 2.0),
        # optional
        ift_kappa_cap=(False, True),
        ift_kappa_max=(1.0, 2.0, 5.0, None), # to down here ^
    )

    results_jsonl = os.path.join(
    args.results_dir, f"{dataset_name}_results_{args.epochs}ep_3seed.jsonl"
    )
    summary_jsonl = os.path.join(
    args.results_dir, f"{dataset_name}_summary_{args.epochs}ep_3seed.jsonl"
    )

    results = []
    
    for run in runs:
        run_for_ds = replace(run, name=f"{dataset_name}__{run.name}")

        print(f"\n🧹🧹 === Running {run_for_ds.name} === 🧹🧹")

        try:
            result = run_one_experiment(
                ds=ds,
                spec=spec,
                base_train_cfg=base_train_cfg,
                run=run_for_ds,
                build_model_fn=build_tgn_model,
                epochs=args.epochs,
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

    results.sort(key=lambda r: r["best_val_mrr"], reverse=True)

    print(f"\n=== Best runs for {dataset_name} ===")
    for r in results[:5]:
        best_test = r["best_snapshot"]["test"]["mrr"]
        print(
            f"{r['name']} | seed={r['seed']} | "
            f"best_val={r['best_val_mrr']:.4f} "
            f"@epoch {r['best_epoch']} | best_test={best_test:.4f}"
        )


if __name__ == "__main__":
    main()