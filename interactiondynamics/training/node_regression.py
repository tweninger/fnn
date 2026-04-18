import itertools
import json
import random
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterable, Optional, Sequence
import matplotlib.pyplot as plt
import numpy as np
import torch
import copy

from scipy.stats import spearmanr
from core.events import EventBatch
from core.config import ModelConfig
from data.spring_mass import SpringMassDataset, SpringMassConfig
from eval.node_regression import (
    NodeEvalSlices,
    evaluate_node_stream_sliced,
    node_regression_loss_and_metrics,
    collect_node_predictions_over_time,
    compute_prediction_analysis,
)
from models.tgn_model import build_tgn_model
from plotting.node_regression import plot_node_targets_by_feature
from utils.metric_selection import higher_is_better, is_better
from utils.repro import set_seed
from experiments.node_runs import NodeSweepRun, make_node_runs

# -----------------------------
# dataclasses
# -----------------------------
@dataclass(frozen=True)
class NodeSweepRun:
    name: str
    model_cfg: ModelConfig
    lr: Optional[float] = None
    weight_decay: Optional[float] = None
    tbptt_steps: Optional[int] = None
    seed: int = 0

@dataclass
class NodeRunResult:
    dataset: Optional[str]
    name: str
    seed: int
    epochs: int
    selection_metric: str
    best_val_metric: float
    best_epoch: int
    best_snapshot: dict
    final_snapshot: dict
    analysis_val: dict
    analysis_test: dict
    wall_sec: float


@dataclass
class NodeTrainConfig:
    num_nodes: int
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    device: torch.device = torch.device("cpu")
    log_every: int = 50
    tbptt_steps: int = 1
    debug: bool = False
    loss_name: str = "mse"   # "mse" | "mae" | "huber"
    selection_metric: str = "rmse"    # start with "rmse"


# -----------------------------
# train / eval loops
# -----------------------------

def train_one_epoch_node(
    model,
    bins: Iterable[EventBatch],
    optimizer: torch.optim.Optimizer,
    cfg: NodeTrainConfig,
) -> Dict[str, float]:
    """
    One-step-ahead node regression:

    For consecutive bins (prev_bin, curr_bin):
      1) Update memory using prev_bin            state <- step(state, prev_bin)
      2) Predict node targets for curr_bin       loss <- predict_nodes(state) vs curr.node_targets
      3) Optimizer step
      4) Detach memory periodically for truncated BPTT
    """
    model.train()
    device = torch.device(cfg.device)

    state = model.init_state(batch_size=1, num_nodes=cfg.num_nodes, device=device)

    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    n_steps = 0

    # optional diagnostics for IFT
    kappa_sum = 0.0
    kappa_n = 0

    prev: Optional[EventBatch] = None
    for curr in bins:
        curr = curr.to(device)

        if prev is None:
            prev = curr
            continue

        # update state with previous bin
        state, aux = model.step(state, prev)

        if aux is not None and "kappa" in aux:
            k = aux["kappa"]
            if torch.is_tensor(k):
                kappa_sum += float(k.detach().item())
                kappa_n += 1

        if cfg.debug and state is not None and state.node is not None:
            print(
                "DEBUG node variance:",
                float(state.node.std(dim=0).mean().item()),
                "max|h|:",
                float(state.node.abs().max().item()),
            )

        if state is not None and state.node is not None and (not torch.isfinite(state.node).all()):
            raise RuntimeError("Non-finite state.node after model.step()")

        optimizer.zero_grad(set_to_none=True)

        loss, metrics = node_regression_loss_and_metrics(
            model=model,
            state=state,
            next_events=curr,
            loss_name=cfg.loss_name,
        )

        loss.backward()

        if cfg.grad_clip and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()

        if cfg.tbptt_steps and ((n_steps + 1) % cfg.tbptt_steps == 0):
            if state is not None:
                state.detach_()

        total_loss += float(loss.item())
        total_mae += float(metrics["mae"])
        total_rmse += float(metrics["rmse"])
        n_steps += 1

        if cfg.log_every and (n_steps % cfg.log_every) == 0:
            print(
                f"[step {n_steps:6d}] "
                f"loss={loss.item():.4f} "
                f"mae={metrics['mae']:.4f} "
                f"rmse={metrics['rmse']:.4f}"
            )

        prev = curr

    if n_steps == 0:
        return {"loss": 0.0, "mae": 0.0, "rmse": 0.0}

    out = {
        "loss": total_loss / n_steps,
        "mae": total_mae / n_steps,
        "rmse": total_rmse / n_steps,
    }
    if kappa_n > 0:
        out["kappa_mean"] = kappa_sum / kappa_n
    return out


def make_node_runs(
    base_model_cfg: ModelConfig,
    *,
    seeds: Sequence[int] = (0,),
    aggregator: Sequence[str] = ("sum", "deepsets", "settransformer"),
    upd: Sequence[str] = ("tgn_gru",),
    dropout: Sequence[float] = (0.0,),
    predictor_dropout: Sequence[float] = (0.0,),
    use_time_features: Sequence[bool] = (False,),
    ift_kappa_param: Sequence[str] = ("softplus", "exp"),
    ift_dt: Sequence[float] = (0.05,),
    ift_gamma: Sequence[float] = (0.0,),
    ift_kappa_init: Sequence[float] = (1.0,),
    ift_kappa_cap: Sequence[bool] = (False,),
    ift_kappa_max: Sequence[float | None] = (None,),
) -> list[NodeSweepRun]:
    runs: list[NodeSweepRun] = []

    for (agg, do, update_name, pdo, tf, seed) in itertools.product(
        aggregator, dropout, upd, predictor_dropout, use_time_features, seeds
    ):
        cfg = ModelConfig(**asdict(base_model_cfg))
        cfg.update = update_name          # type: ignore[attr-defined]
        cfg.aggregator = agg              # type: ignore[attr-defined]
        cfg.dropout = do
        cfg.scorer_dropout = pdo          # reuse this width/dropout field for predictor head too
        cfg.use_time_features = tf

        # if your ModelConfig does not already define these fields,
        # add them there, or dynamically attach them if your class allows it.
        cfg.task = "node_regression"      # type: ignore[attr-defined]
        cfg.predictor = "mlp_node"        # type: ignore[attr-defined]

        base_name = f"agg={agg}|update={update_name}|do={do}|pdo={pdo}|time={tf}"

        if update_name == "ift_update":
            for (kp, dt, ga, k0) in itertools.product(
                ift_kappa_param, ift_dt, ift_gamma, ift_kappa_init
            ):
                for cap in ift_kappa_cap:
                    if cap:
                        for kmax in ift_kappa_max:
                            if kmax is None:
                                continue
                            cfg2 = ModelConfig(**asdict(cfg))
                            cfg2.task = "node_regression"          # type: ignore[attr-defined]
                            cfg2.predictor = "mlp_node"            # type: ignore[attr-defined]
                            cfg2.ift_kappa_param = kp             # type: ignore[attr-defined]
                            cfg2.ift_dt = float(dt)
                            cfg2.ift_gamma = float(ga)
                            cfg2.ift_kappa = float(k0)
                            cfg2.ift_kappa_cap = True
                            cfg2.ift_kappa_max = float(kmax)

                            name2 = f"{base_name}|{kp}|dt={dt}|gamma={ga}|k0={k0}|cap={kmax}"
                            runs.append(NodeSweepRun(name=name2, model_cfg=cfg2, seed=int(seed)))
                    else:
                        cfg2 = ModelConfig(**asdict(cfg))
                        cfg2.task = "node_regression"              # type: ignore[attr-defined]
                        cfg2.predictor = "mlp_node"                # type: ignore[attr-defined]
                        cfg2.ift_kappa_param = kp                 # type: ignore[attr-defined]
                        cfg2.ift_dt = float(dt)
                        cfg2.ift_gamma = float(ga)
                        cfg2.ift_kappa = float(k0)
                        cfg2.ift_kappa_cap = False
                        cfg2.ift_kappa_max = None

                        name2 = f"{base_name}|{kp}|dt={dt}|gamma={ga}|k0={k0}|cap=none"
                        runs.append(NodeSweepRun(name=name2, model_cfg=cfg2, seed=int(seed)))
            continue

        runs.append(NodeSweepRun(name=base_name, model_cfg=cfg, seed=int(seed)))

    return runs

# -----------------------------
# one full experiment
# -----------------------------

def run_one_node_experiment(
    ds,
    spec,
    base_train_cfg: NodeTrainConfig,
    run: NodeSweepRun,
    build_model_fn: Callable,
    epochs: int = 5,
    save_jsonl_path: Optional[str] = None,
    save_summary_path: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> tuple[NodeRunResult, torch.nn.Module, NodeTrainConfig]:
    device = torch.device(base_train_cfg.device)
    set_seed(run.seed)

    train_cfg = NodeTrainConfig(**asdict(base_train_cfg))
    if run.lr is not None:
        train_cfg.lr = run.lr
    if run.weight_decay is not None:
        train_cfg.weight_decay = run.weight_decay
    if run.tbptt_steps is not None:
        train_cfg.tbptt_steps = run.tbptt_steps

    model = build_model_fn(spec, run.model_cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    selection_metric = train_cfg.selection_metric
    best_val_metric = float("-inf") if higher_is_better(selection_metric) else float("inf")
    best_epoch = -1
    best_snapshot: dict = {}
    best_model_state = None
    
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_stats_step = train_one_epoch_node(model, ds.bins("train"), optimizer, train_cfg)

        train_eval = evaluate_node_stream_sliced(model, ds.bins("train"), train_cfg, slices=NodeEvalSlices(early_steps=10))
        val_stats  = evaluate_node_stream_sliced(model, ds.bins("val"),   train_cfg, slices=NodeEvalSlices(early_steps=10))
        test_stats = evaluate_node_stream_sliced(model, ds.bins("test"),  train_cfg, slices=NodeEvalSlices(early_steps=10))

        snapshot = {
            "epoch": epoch,
            "train_step": train_stats_step,
            "train_eval": train_eval,
            "val": val_stats,
            "test": test_stats,
        }

        cur_val_metric = float(val_stats[selection_metric])

        km = train_stats_step.get("kappa_mean", None)
        kappa_str = f"{km:.4f}" if km is not None else ""
        print(
            f"[{run.name} | seed={run.seed} | epoch {epoch:03d}] "
            f"train(step) loss={train_stats_step['loss']:.4f} "
            f"rmse={train_stats_step['rmse']:.4f} "
            f"kappa={kappa_str} | "
            f"val {selection_metric}={cur_val_metric:.4f} "
            f"test rmse={test_stats['rmse']:.4f}"
        )

        if is_better(cur_val_metric, best_val_metric, selection_metric):
            best_val_metric = cur_val_metric
            best_epoch = epoch
            best_snapshot = snapshot
            best_model_state = copy.deepcopy(model.state_dict())

        if save_jsonl_path is not None:
            row = {
                "dataset": dataset_name,
                "run": run.name,
                "seed": run.seed,
                "task": getattr(run.model_cfg, "task", "node_regression"),
                "predictor": getattr(run.model_cfg, "predictor", "mlp_node"),
                "model_cfg": asdict(run.model_cfg),
                "train_cfg_overrides": {
                    k: v
                    for k, v in {
                        "lr": run.lr,
                        "weight_decay": run.weight_decay,
                        "tbptt_steps": run.tbptt_steps,
                    }.items()
                    if v is not None
                },
                **snapshot,
            }
            with open(save_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

    wall = time.time() - t0
    final_snapshot = snapshot
    
    analysis_val = {}
    analysis_test = {}     

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        val_pack = collect_node_predictions_over_time(model, ds.bins("val"), train_cfg)
        test_pack = collect_node_predictions_over_time(model, ds.bins("test"), train_cfg)

        if len(val_pack) == 4:
            _, y_true_val, y_pred_val, node_mask_val = val_pack
        else:
            _, y_true_val, y_pred_val = val_pack
            node_mask_val = None

        if len(test_pack) == 4:
            _, y_true_test, y_pred_test, node_mask_test = test_pack
        else:
            _, y_true_test, y_pred_test = test_pack
            node_mask_test = None

        analysis_val = compute_prediction_analysis(y_true_val, y_pred_val, node_mask_val)
        analysis_test = compute_prediction_analysis(y_true_test, y_pred_test, node_mask_test)

    summary = {
        "dataset": dataset_name,
        "name": run.name,
        "seed": run.seed,
        "epochs": epochs,
        "selection_metric": selection_metric,
        "best_val_metric": best_val_metric,
        "best_epoch": best_epoch,
        "best_snapshot": best_snapshot,
        "final_snapshot": final_snapshot,
        "analysis_val": analysis_val,
        "analysis_test": analysis_test,
        "wall_sec": wall,
        }

    if save_summary_path is not None:
        with open(save_summary_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")

    result = NodeRunResult(
        dataset=dataset_name,
        name=run.name,
        seed=run.seed,
        epochs=epochs,
        selection_metric=selection_metric,
        best_val_metric=best_val_metric,
        best_epoch=best_epoch,
        best_snapshot=best_snapshot,
        final_snapshot=final_snapshot,
        analysis_val=analysis_val,
        analysis_test=analysis_test,
        wall_sec=wall,
    )

    return result, model, train_cfg


