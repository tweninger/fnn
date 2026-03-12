from dataclasses import asdict
import json
import torch

# add make_runs function from train.py for our sweepingggg
from train import TrainConfig, SweepRun, run_one_experiment, make_runs
from core.config import ModelConfig
from data.jodie import JODIEBinnedDataset, JODIEConfig
from eval.evaluate import EvalSlices
from models.tgn_model import build_tgn_model

"""
- mini-sweep smoke again a) does it even work b) lets compare mixing and matching w miss cartesian product 
- again the same JSON results and summary are outputted for each run
- added failure catching in main
"""
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = JODIEBinnedDataset(
        JODIEConfig(root="./data/JODIE", name="Wikipedia", device=device)
    )
    spec = ds.spec()

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
        update="tgn_gru",
    )

    #again how we're getting out sweeping sweep sweep sweep! 🧹🧹🧹
    runs = make_runs(
        base_model_cfg,
        seeds=(0, 42, 123), # freestyling here, up to 3 
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

    results = []
    for run in runs:
        print(f"\n🧹 🧹 === Running {run.name} === 🧹 🧹")
        try:
            result = run_one_experiment(
                ds=ds,
                spec=spec,
                base_train_cfg=base_train_cfg,
                run=run,
                build_model_fn=build_tgn_model,
                epochs=6, # freestyling here
                eval_slices=EvalSlices(early_steps=10),
                save_jsonl_path="results/mini_sweep_results_6ep_3seeds.jsonl",
                save_summary_path="results/mini_sweep_summary_6ep_3seeds.jsonl",
            )
        # catching failures before they blow up the whole run    
        except Exception as e:
            print(f"FAILED: run={run.name}, seed={run.seed}, error={type(e).__name__}: {e}")
        
        results.append(asdict(result))


if __name__ == "__main__":
    main()