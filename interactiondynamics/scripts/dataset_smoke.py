from dataclasses import asdict, replace, is_dataclass
import json
import torch

# add make_runs function from train.py for our sweepingggg
from train import TrainConfig, SweepRun, run_one_experiment, make_runs
from core.config import ModelConfig
from data.jodie import JODIEBinnedDataset, JODIEConfig
from eval.evaluate import EvalSlices
from models.tgn_model import build_tgn_model
from collections import defaultdict


"""
- small sweep -> do all the JODIE datasets even work? -> yeah
- does failure catch even work? -> idk, no fails yet lol
- again the same JSON results and summary are outputted for each run + dataset is noted too
- dumps dataset specs into json file
"""

# dataset spec stuff
def spec_to_dict(spec):
    if is_dataclass(spec):
        return asdict(spec)
    try:
        return dict(vars(spec))
    except TypeError:
        return {"repr": repr(spec)}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #dataset stuff - putting specs into json still
    dataset_specs = {}
    dataset_names = ("Wikipedia", "Reddit", "MOOC", "LastFM")
    results = []

    #loop over all 4 datasets in JODIE
    for dataset_name in dataset_names:
        ds = JODIEBinnedDataset(
            JODIEConfig(root="./data/JODIE", name=dataset_name, device=device)
        )

        spec = ds.spec()

        #ds specs into json..
        dataset_specs[dataset_name] = spec_to_dict(spec)

        base_train_cfg = TrainConfig(
            num_nodes=spec.num_nodes,
            num_neg=20,
            tbptt_steps=1,
            log_every=1000 if dataset_name == "LastFM" else 50,
            device=device,
            weight_decay=1e-3,
            lr=1e-3,
        )

        # for smoke we are just using sum/tgn gru
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

        #again how we're getting out sweeping sweep sweep sweep! 🧹🧹🧹
        runs = make_runs(
            base_model_cfg,
            seeds=(0,), # freestyling here, up to 3 
            aggregator=("sum",),
            upd = ("tgn_gru",),
            dropout=(0.0,),
            scorer_dropout=(0.0,),
            use_time_features=(False,),
        )

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
                    epochs=1, # freestyling here
                    eval_slices=EvalSlices(early_steps=10),
                    save_jsonl_path="results/ds_smoke_results.jsonl",
                    save_summary_path="results/ds_smoke_summary.jsonl",
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

    # still just dataset specs
    with open("results/dataset_specs.json", "w") as f:
        json.dump(dataset_specs, f, indent=2)

if __name__ == "__main__":
    main()