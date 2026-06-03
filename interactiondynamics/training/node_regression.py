import itertools
import json
import random
import time
from dataclasses import asdict, dataclass, replace
from typing import Callable, Dict, Iterable, Optional, Sequence
import matplotlib.pyplot as plt
import numpy as np
import torch
import copy

from scipy.stats import spearmanr
from core.events import EventBatch
from core.config import ModelConfig
from eval.node_regression import (
    NodeEvalSlices,
    evaluate_node_stream_sliced,
    node_regression_loss_and_metrics,
    collect_node_predictions_over_time,
    compute_prediction_analysis,
    _iter_context_target_bins,
)
from models.tgn_model import build_tgn_model
from plotting.node_regression import plot_node_targets_by_feature
from utils.metric_selection import higher_is_better, is_better_metric
from utils.repro import set_seed
from experiments.node_regression_runs import NodeSweepRun
from training.ift_aux import accumulate_ift_aux, finalize_ift_aux

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
    selection_metric: str = "median_node_pearson"    # higher is better; robust per-node correlation


# -----------------------------
# train / eval loops
# -----------------------------

def train_one_epoch_node(
    model,
    bins: Iterable[EventBatch],
    optimizer: torch.optim.Optimizer,
    cfg: NodeTrainConfig,
    target_bins: Optional[Iterable[EventBatch]] = None,
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

    ift_sums: Dict[str, float] = {}
    ift_n = 0

    prev_obs: Optional[EventBatch] = None
    for curr_obs, curr_target in _iter_context_target_bins(bins, target_bins):
        curr_obs = curr_obs.to(device)
        curr_target = curr_target.to(device)

        if prev_obs is None:
            prev_obs = curr_obs
            continue

        # update state with previous bin
        state, _aux = model.step(state, prev_obs)

        if accumulate_ift_aux(_aux, ift_sums):
            ift_n += 1

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
            next_events=curr_target,
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

        prev_obs = curr_obs

    if n_steps == 0:
        return {"loss": 0.0, "mae": 0.0, "rmse": 0.0}

    out = {
        "loss": total_loss / n_steps,
        "mae": total_mae / n_steps,
        "rmse": total_rmse / n_steps,
    }
    out.update(finalize_ift_aux(ift_sums, ift_n))
    return out


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
    clean_ds=None,
    corruption_cfg: Optional[dict] = None,
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

    model = build_model_fn(spec, replace(run.model_cfg, ift_h_init_seed=run.seed)).to(device)
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

    use_clean_targets = clean_ds is not None and clean_ds is not ds
    
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_stats_step = train_one_epoch_node(
            model,
            ds.bins("train"),
            optimizer,
            train_cfg,
            target_bins=clean_ds.bins("train") if use_clean_targets else None,
        )

        train_eval = evaluate_node_stream_sliced(
            model,
            ds.bins("train"),
            train_cfg,
            slices=NodeEvalSlices(early_steps=10),
            target_bins=clean_ds.bins("train") if use_clean_targets else None,
        )
        val_stats  = evaluate_node_stream_sliced(
            model,
            ds.bins("val"),
            train_cfg,
            slices=NodeEvalSlices(early_steps=10),
            target_bins=clean_ds.bins("val") if use_clean_targets else None,
        )
        test_stats = evaluate_node_stream_sliced(
            model,
            ds.bins("test"),
            train_cfg,
            slices=NodeEvalSlices(early_steps=10),
            target_bins=clean_ds.bins("test") if use_clean_targets else None,
        )

        snapshot = {
            "epoch": epoch,
            "train_step": train_stats_step,
            "train_eval": train_eval,
            "val": val_stats,
            "test": test_stats,
        }

        cur_val_metric = float(val_stats.get(selection_metric, float("nan")))

        km = train_stats_step.get("kappa_mean", None)
        kappa_part = f" kappa={km:.4f}" if km is not None else ""
        corr_part = ""
        if selection_metric in val_stats:
            corr_part = f" val {selection_metric}={float(val_stats[selection_metric]):.4f}"
        print(
            f"[{run.name} | seed={run.seed} | epoch {epoch:03d}] "
            f"train(step) loss={train_stats_step['loss']:.4f} "
            f"rmse={train_stats_step['rmse']:.4f}{kappa_part} |"
            f"{corr_part} "
            f"val rmse={val_stats['rmse']:.4f} "
            f"test rmse={test_stats['rmse']:.4f}"
        )

        cur_val_metric = float(val_stats.get(selection_metric, float("nan")))

        if np.isfinite(cur_val_metric) and is_better_metric(cur_val_metric, best_val_metric, selection_metric):
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
                        "selection_metric": selection_metric,
                    }.items()
                    if v is not None
                },
                "corruption_cfg": corruption_cfg,
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
        val_target_bins = clean_ds.bins("val") if use_clean_targets else None
        test_target_bins = clean_ds.bins("test") if use_clean_targets else None
        val_pack = collect_node_predictions_over_time(
            model, ds.bins("val"), train_cfg, target_bins=val_target_bins
        )
        test_pack = collect_node_predictions_over_time(
            model, ds.bins("test"), train_cfg, target_bins=test_target_bins
        )

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


