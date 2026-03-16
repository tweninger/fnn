from dataclasses import asdict, replace
import json
from pathlib import Path
import torch

from train import TrainConfig, run_one_experiment, make_runs
from core.config import ModelConfig
from data.jodie import JODIEBinnedDataset, JODIEConfig
from eval.evaluate import EvalSlices
from models.tgn_model import build_tgn_model


def load_completed_lastfm(summary_path):
    completed = set()

    if not Path(summary_path).exists():
        return completed

    with open(summary_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)

            if r.get("dataset") == "LastFM":
                completed.add((r["name"], r["seed"]))

    return completed


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    dataset_name = "LastFM"

    results_path = "results/ds_sweep_results_6ep_3seed.jsonl"
    summary_path = "results/ds_sweep_summary_6ep_3seed.jsonl"

    completed_lastfm = load_completed_lastfm(summary_path)
    print(f"Found {len(completed_lastfm)} completed LastFM runs in summary file.")

    ds = JODIEBinnedDataset(
        JODIEConfig(root="./data/JODIE", name=dataset_name, device=device)
    )

    spec = ds.spec()

    base_train_cfg = TrainConfig(
        num_nodes=spec.num_nodes,
        num_neg=20,
        tbptt_steps=1,
        log_every=1000,
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
        seeds=(0, 42, 123),
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

    pending = []
    for run in runs:
        run_for_ds = replace(run, name=f"{dataset_name}__{run.name}")

        if (run_for_ds.name, run_for_ds.seed) in completed_lastfm:
            continue

        pending.append((run, run_for_ds))

    print(f"Need to run {len(pending)} missing LastFM runs.")

    for run, run_for_ds in pending:
        print(f"\n🧹🧹 === Running MISSING {run_for_ds.name} === 🧹🧹")

        try:
            result = run_one_experiment(
                ds=ds,
                spec=spec,
                base_train_cfg=base_train_cfg,
                run=run_for_ds,
                build_model_fn=build_tgn_model,
                epochs=6,
                eval_slices=EvalSlices(early_steps=10),
                save_jsonl_path=results_path,
                save_summary_path=summary_path,
                dataset_name=dataset_name,
            )
            results.append(asdict(result))

            # update in-memory set too
            completed_lastfm.add((run_for_ds.name, run_for_ds.seed))

        except Exception as e:
            print(
                f"FAILED: dataset={dataset_name}, run={run.name}, "
                f"seed={run.seed}, error={type(e).__name__}: {e}"
            )

    results.sort(key=lambda r: r["best_val_mrr"], reverse=True)

    print(f"\n=== Best newly completed LastFM runs ===")
    for r in results[:5]:
        best_test = r["best_snapshot"]["test"]["mrr"]
        print(
            f"{r['name']} | seed={r['seed']} | "
            f"best_val={r['best_val_mrr']:.4f} "
            f"@epoch {r['best_epoch']} | best_test={best_test:.4f}"
        )


if __name__ == "__main__":
    main()