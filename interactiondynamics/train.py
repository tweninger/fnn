# train.py

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
from data.toy import ToyShiftConfig, ToyShiftDataset
# evaluation logic
from eval.evaluate import EvalSlices, evaluate_stream_sliced
# model config object
from core.config import ModelConfig
# real temporal datasets
from data.jodie import JODIEBinnedDataset, JODIEConfig # type: ignore
from models.tgn_model import build_tgn_model

from eval.ranking import ranking_loss_and_metrics


# dataclasses = containers for settings/results
# automatically make this class behave like a nice clean parameter container instead of whole __init__ situation


@dataclass(frozen=True) #once created, don't change the fields (immutable)
# one specific experiment setting/one run in the sweep
class SweepRun:
    name: str
    model_cfg: ModelConfig
    # optional train overrides; keep TrainConfig stable and override only what you need
    lr: Optional[float] = None
    weight_decay: Optional[float] = None
    num_neg: Optional[int] = None
    tbptt_steps: Optional[int] = None
    seed: int = 0


# results of one experiment run
@dataclass
class RunResult:
    dataset: Optional[str]
    name: str
    seed: int
    epochs: int
    best_val_mrr: float
    best_epoch: int
    best_snapshot: dict
    final_snapshot: dict
    wall_sec: float


# training settings container
@dataclass
class TrainConfig:
    num_nodes: int
    num_neg: int = 20 #negative samples
    lr: float = 1e-3
    weight_decay: float = 1e-5 #regularizaiton//penality on large weights
    grad_clip: float = 1.0 #clip gradients if they get too big
    device: torch.device = torch.device("cpu")
    log_every: int = 50 # print every N steps
    tbptt_steps: int = 1           # detach every k scored steps --> truncate backprop through time every k steps // cut off backprob periodically
    update_before_score: bool = True  # "prev->update, score current" 
    debug: bool = False



# VERY IMPORTANT FUNCTION !!
def train_one_epoch(
    model, #neural model
    bins: Iterable[EventBatch], #training data as event bins
    optimizer: torch.optim.Optimizer, #adam or whatever updates weights
    cfg: TrainConfig, #training settings
) -> Dict[str, float]: # returns dict 
    """
    One-step-ahead (binned) TGN training:

    For consecutive bins (prev_bin, curr_bin):
      1) Update memory using prev_bin            state <- step(state, prev_bin)
      2) Predict curr_bin using that memory      loss <- rank(state, curr_bin)
      3) Optimizer step                          update params
      4) Detach memory to truncate BPTT          state.detach_()
    """
    """
    In other words lol:
    1) use previous bin to update hidden memory/state
    2) use that updated state to predict the current bin
    3) compute loss and train
    4)detatch memory so the computational graph doesn't grow forever
    """

    model.train() # training mode
    device = torch.device(cfg.device) # choose cpu/gpu from config

    # Latent memory (per-node) lives in ModelState
    # initalize model's hidden state/memory
    state = model.init_state(batch_size=1, num_nodes=cfg.num_nodes, device=device)

# book keeping variables
    total_loss = 0.0
    total_mrr = 0.0
    n_steps = 0

    # checks for ift
    kappa_sum = 0.0
    kappa_n = 0    

# we are looping through event bins one at a time
# prev - previous time bin, curr - current time bin
    prev: Optional[EventBatch] = None
    for curr in bins:
        curr = curr.to(device)

        # Need consecutive (prev, curr) to do one-step-ahead prediction // no previous bin so prev is curr thats fine
        if prev is None:
            prev = curr
            continue

        # use previous bin to update the model state // old hidden + previous events -> new hidden state
        # aux - extra diagnostic vals
        state, _aux = model.step(state, prev)

        # if kappa returned, store for avg
        if _aux is not None and "kappa" in _aux:
            k = _aux["kappa"]
            if torch.is_tensor(k):
                kappa_sum += float(k.detach().item())
                kappa_n += 1        

        # check whether hidden states are behaving
        h = state.node
        if cfg.debug:
            print("DEBUG node variance:", float(h.std(dim=0).mean().item()),
                "max|h|:", float(h.abs().max().item()))        

        # right after model.step(state, prev):
        # bug catcher!!
        if prev.t is not None and getattr(state, "aux", None) is not None and "L_bin_t_min" in state.aux:
            assert state.aux["L_bin_t_min"] == int(prev.t.min().item()), "step() did not use prev bin for operator"


        # make sure no NaN/inf in node state 
        if state.node is not None and (not torch.isfinite(state.node).all()):
            raise RuntimeError("Non-finite state.node after model.step()")

        # ---- 2) score CURRENT bin using memory-after-prev (no leakage) ----
        # before computing new gradients, clear old ones (standard pytorch training step yay)
        optimizer.zero_grad(set_to_none=True)

        # all events in prev bin must have same timestamp, all events in curr have same, prev must be earlier than curr
        # data - discrete time bins in order
        assert prev.t is not None and int(prev.t.min().item()) == int(prev.t.max().item()), \
            "Expected all events in prev bin to have the same timestamp"
        assert curr.t is not None and int(curr.t.min().item()) == int(curr.t.max().item()), \
            "Expected all events in curr bin to have the same timestamp"
        assert int(prev.t.max().item()) < int(curr.t.min().item()), \
            "Expected prev bin to be strictly before curr bin in time"
        
        # sanity check
        if getattr(state, "aux", None) is not None:
            if "L_bin_t_min" in state.aux and "L_bin_t_max" in state.aux:
                pt = int(prev.t.min().item())
                assert state.aux["L_bin_t_min"] == pt and state.aux["L_bin_t_max"] == pt, \
                    f"L bin mismatch: L=({state.aux['L_bin_t_min']},{state.aux['L_bin_t_max']}) prev.t={pt}"

        # given current hidden state, score/predict the next events
        loss, metrics = ranking_loss_and_metrics(
            model=model,
            state=state,
            next_events=curr,
            num_nodes=cfg.num_nodes,
            num_neg=cfg.num_neg,
        )
        

        # ---- 3) optimize ----

        # classic pytorch trio hehe
        loss.backward()
        if cfg.grad_clip and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        # ---- 4) truncate backprop-through-time (TGN-style) ----
        # sequence model trick... instead of gradients flow thru entire time history forever...
        # detach periodically! again so graph cant grow too large and training is too expensive/unstable
        if cfg.tbptt_steps and ((n_steps + 1) % cfg.tbptt_steps == 0):
            if state is not None:
                state.detach_()

        # ---- bookkeeping ----
        total_loss += float(loss.item())
        total_mrr += float(metrics["mrr"])
        n_steps += 1

        # print progress every so often
        if cfg.log_every and (n_steps % cfg.log_every) == 0:
            print(
                f"[step {n_steps:6d}] "
                f"loss={loss.item():.4f} "
                f"mrr={metrics['mrr']:.4f} "
                f"hits@1={metrics.get('hits@1', 0):.4f} "
                f"hits@10={metrics.get('hits@10', 0):.4f}"
            )

        # Slide the window forward
        # current becomes previous for next iteration!
        prev = curr

    # end of epoch return/safety case!! its fine
    if n_steps == 0:
        return {"loss": 0.0, "mrr": 0.0}

    # return average training metrics for the epoch
    out = {"loss": total_loss / n_steps, "mrr": total_mrr / n_steps}
    if kappa_n > 0:
        out["kappa_mean"] = kappa_sum / kappa_n
    return out


# for reproducability! = try to make runs with the same seed behave the same
# needed for ML b/c randomness affects initailization/sampling/GPU kernels
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# BUILDS HUGE EXPERIMENT SWEEP!
def make_runs(
    base_model_cfg: ModelConfig,
    *, # everything after * must be passed by keyword
    seeds: Sequence[int] = (0,),
    aggregator: Sequence[str] = ("sum", "deepsets", "settransformer"),
    upd: Sequence[str] = ("tgn_gru",),
    dropout: Sequence[float] = (0.0,),
    scorer_dropout: Sequence[float] = (0.0,),
    use_time_features: Sequence[bool] = (False,),
    ift_kappa_param: Sequence[str] = ("softplus", "exp"),
    ift_dt: Sequence[float] = (0.05,),
    ift_gamma: Sequence[float] = (0.0,),
    ift_kappa_init: Sequence[float] = (1.0,),
    ift_kappa_cap: Sequence[bool] = (False,),
    ift_kappa_max: Sequence[float | None] = (None,),    
) -> list[SweepRun]:
    # create EVERY combination of aggregator, dropout, update type, scorer dropout, time features on/off, and seed
    # aka grid search/sweep
    runs: list[SweepRun] = []
    for (agg, do, upd, sdo, tf, seed) in itertools.product( # cartesian product hehe
        aggregator, dropout, upd, scorer_dropout, use_time_features, seeds
    ):
        cfg = ModelConfig(**asdict(base_model_cfg)) # make copy of base model config "make new modelconfig with same fields as base config"
        # customize thse copied config for current run
        cfg.update = upd  # type: ignore 
        cfg.aggregator = agg  # type: ignore
        cfg.dropout = do
        cfg.scorer_dropout = sdo
        cfg.use_time_features = tf

        # build human readable name
        base_name = f"agg={agg}|update={upd}|do={do}|sdo={sdo}|time={tf}"

        # conditional dimension: only for IFT
        # sweep over extra IFT hyperparameters too if update = IFT
        if upd == "ift_update":
            for (kp, dt, ga, k0) in itertools.product( # every combo of those IFT params
                ift_kappa_param, ift_dt, ift_gamma, ift_kappa_init
            ):
                for cap in ift_kappa_cap:
                    if cap:
                        # cap=True => sweep over explicit maxima (skip None)
                        for kmax in ift_kappa_max:
                            if kmax is None:
                                continue
                            cfg2 = ModelConfig(**asdict(cfg)) # specialized config and append a run
                            cfg2.ift_kappa_param = kp  # type: ignore
                            cfg2.ift_dt = float(dt)
                            cfg2.ift_gamma = float(ga)
                            cfg2.ift_kappa = float(k0)
                            cfg2.ift_kappa_cap = True
                            cfg2.ift_kappa_max = float(kmax)

                            name2 = f"{base_name}|{kp}|dt={dt}|gamma={ga}|k0={k0}|cap={kmax}"
                            runs.append(SweepRun(name=name2, model_cfg=cfg2, seed=int(seed)))
                    else:
                        # cap=False => exactly ONE config (don't sweep kmax)
                        cfg2 = ModelConfig(**asdict(cfg))
                        cfg2.ift_kappa_param = kp  # type: ignore
                        cfg2.ift_dt = float(dt)
                        cfg2.ift_gamma = float(ga)
                        cfg2.ift_kappa = float(k0)
                        cfg2.ift_kappa_cap = False
                        cfg2.ift_kappa_max = None

                        name2 = f"{base_name}|{kp}|dt={dt}|gamma={ga}|k0={k0}|cap=none"
                        runs.append(SweepRun(name=name2, model_cfg=cfg2, seed=int(seed)))
            continue # we already handled IFT speically so skip normal run append below
        
        # for no IFT runs, just add one run normally
        runs.append(SweepRun(name=base_name, model_cfg=cfg, seed=int(seed)))
    return runs

# function manages one experiment run from start to finish!
def run_one_experiment(
    ds,
    spec,
    base_train_cfg: TrainConfig,
    run: SweepRun,
    build_model_fn: Callable[[Any, ModelConfig], torch.nn.Module],
    epochs: int = 5,
    eval_slices: Optional[EvalSlices] = None,
    save_jsonl_path: Optional[str] = None,
    save_summary_path: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> RunResult:
    # choose device and set reproducible seed
    device = torch.device(base_train_cfg.device)
    set_seed(run.seed)

    # Apply per-run overrides without mutating base cfg
    train_cfg = TrainConfig(**asdict(base_train_cfg)) # make fresh copy of training config
    # override specific settings if this run requested them
    if run.lr is not None:
        train_cfg.lr = run.lr
    if run.weight_decay is not None:
        train_cfg.weight_decay = run.weight_decay
    if run.num_neg is not None:
        train_cfg.num_neg = run.num_neg
    if run.tbptt_steps is not None:
        train_cfg.tbptt_steps = run.tbptt_steps

    # Fresh model+opt each run (important!)
    # build model from spec + model config, move to GPU/CPU, use Adam optimizer on model params
    model = build_model_fn(spec, run.model_cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    # default eval config
    if eval_slices is None:
        eval_slices = EvalSlices(early_steps=10)

    # track best validation score seen so far
    best_val_mrr = float("-inf")
    best_epoch = -1
    best_snapshot: dict = {}

    #start the timer!!
    t0 = time.time()

    # loop thru 1, 2, x epochs
    for epoch in range(1, epochs + 1):
        # run one training epoch on training bins
        train_stats_step = train_one_epoch(model, ds.bins("train"), optimizer, train_cfg)

        # eval on train validation and test!
        train_eval = evaluate_stream_sliced(model, ds.bins("train"), train_cfg, slices=eval_slices)
        val_stats  = evaluate_stream_sliced(model, ds.bins("val"),   train_cfg, slices=eval_slices)
        test_stats = evaluate_stream_sliced(model, ds.bins("test"),  train_cfg, slices=eval_slices)

        # package stats from this epoch 
        snapshot = {
            "epoch": epoch,
            "train_step": train_stats_step,
            "train_eval": train_eval,
            "val": val_stats,
            "test": test_stats,
        }

        # print nice progress summary !
        km = train_stats_step.get("kappa_mean", None)
        kappa_str = f"{km:.4f}" if km is not None else ""
        print(
            f"[{run.name} | seed={run.seed} | epoch {epoch:03d}] "
            f"train(step) mrr={train_stats_step['mrr']:.4f} "
            f"kappa={kappa_str} | "
            f"train(eval) mrr={train_eval['mrr']:.4f} "
            f"val mrr={val_stats['mrr']:.4f} "
            f"test mrr={test_stats['mrr']:.4f}"
        )

        # if curren validation MRR is the best, remember that
        # this is how we decide the "best" version of the run
        if val_stats["mrr"] > best_val_mrr:
            best_val_mrr = float(val_stats["mrr"])
            best_epoch = epoch
            best_snapshot = snapshot

        # save to json
        if save_jsonl_path is not None:
            row = {
                "dataset": dataset_name,
                "run": run.name,
                "seed": run.seed,
                "model_cfg": asdict(run.model_cfg),
                "train_cfg_overrides": {
                    k: v for k, v in {
                        "lr": run.lr,
                        "weight_decay": run.weight_decay,
                        "num_neg": run.num_neg,
                        "tbptt_steps": run.tbptt_steps,
                    }.items() if v is not None
                },
                **snapshot,
            }
            with open(save_jsonl_path, "a") as f:
                f.write(json.dumps(row) + "\n")
    # total runtime + remember final epoch states
    wall = time.time() - t0

    final_snapshot = snapshot  # last epoch snapshot

    # save one final run summary as a JSONL row 
    summary = {
        "dataset": dataset_name,
        "name": run.name,
        "seed": run.seed,
        "epochs": epochs,
        "best_val_mrr": best_val_mrr,
        "best_epoch": best_epoch,
        "best_snapshot": best_snapshot,
        "final_snapshot": final_snapshot,
        "wall_sec": wall,
    }

    if save_summary_path is not None:
        with open(save_summary_path, "a") as f:
            f.write(json.dumps(summary) + "\n")

    # return structured result object
    return RunResult(
        dataset=dataset_name,
        name=run.name,
        seed=run.seed,
        epochs=epochs,
        best_val_mrr=best_val_mrr,
        best_epoch=best_epoch,
        best_snapshot=best_snapshot,
        final_snapshot=final_snapshot,
        wall_sec=wall,
    )

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
