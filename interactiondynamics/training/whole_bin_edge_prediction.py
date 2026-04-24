from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, Optional

import torch

from core.events import EventBatch
from eval.evaluate import EvalSlices
from eval.whole_bin_edges import whole_bin_edge_loss_and_metrics
from experiments.interaction_prediction_runs import SweepRun
from utils.repro import set_seed


@dataclass
class WholeBinRunResult:
    dataset: Optional[str]
    name: str
    seed: int
    epochs: int
    selection_metric: str
    best_val_metric: float
    best_epoch: int
    best_snapshot: dict
    final_snapshot: dict
    wall_sec: float


@dataclass
class WholeBinTrainConfig:
    num_nodes: int
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    device: torch.device = torch.device("cpu")
    log_every: int = 50
    tbptt_steps: int = 1
    debug: bool = False
    decision_threshold: float = 0.5
    upper_triangle_only: bool = True
    include_self_loops: bool = False
    pos_weight: float | None = None
    auto_pos_weight: bool = True
    max_auto_pos_weight: float | None = 50.0
    selection_metric: str = "jaccard"


_METRIC_KEYS = (
    "loss",
    "precision",
    "recall",
    "f1",
    "jaccard",
    "accuracy",
    "roc_auc",
    "pr_auc",
    "edge_density",
    "pred_edge_density",
    "num_positive",
    "num_candidates",
)


def _acc_init() -> Dict[str, float]:
    acc = {k: 0.0 for k in _METRIC_KEYS}
    acc["steps"] = 0.0
    return acc


def _acc_update(acc: Dict[str, float], metrics: Dict[str, float]) -> None:
    for k in _METRIC_KEYS:
        v = float(metrics.get(k, float("nan")))
        if torch.isnan(torch.tensor(v)):
            continue
        acc[k] += v
    acc["steps"] += 1.0


def _acc_finalize(acc: Dict[str, float]) -> Dict[str, float]:
    steps = int(acc.get("steps", 0.0))
    if steps == 0:
        out = {k: float("nan") for k in _METRIC_KEYS}
        out["steps"] = 0.0
        return out
    out = {k: acc[k] / steps for k in _METRIC_KEYS}
    out["steps"] = float(steps)
    return out


def train_one_epoch_whole_bin(
    model,
    bins: Iterable[EventBatch],
    optimizer: torch.optim.Optimizer,
    cfg: WholeBinTrainConfig,
) -> Dict[str, float]:
    model.train()
    device = torch.device(cfg.device)
    state = model.init_state(batch_size=1, num_nodes=cfg.num_nodes, device=device)

    acc = _acc_init()
    kappa_sum = 0.0
    kappa_n = 0

    prev: Optional[EventBatch] = None
    n_steps = 0
    for curr in bins:
        curr = curr.to(device)
        if prev is None:
            prev = curr
            continue

        state, aux = model.step(state, prev)
        if aux is not None and "kappa" in aux:
            k = aux["kappa"]
            if torch.is_tensor(k):
                kappa_sum += float(k.detach().item())
                kappa_n += 1

        optimizer.zero_grad(set_to_none=True)
        loss, metrics = whole_bin_edge_loss_and_metrics(
            model=model,
            state=state,
            next_events=curr,
            num_nodes=cfg.num_nodes,
            upper_triangle_only=cfg.upper_triangle_only,
            include_self_loops=cfg.include_self_loops,
            decision_threshold=cfg.decision_threshold,
            pos_weight=cfg.pos_weight,
            auto_pos_weight=cfg.auto_pos_weight,
            max_auto_pos_weight=cfg.max_auto_pos_weight,
        )
        loss.backward()
        if cfg.grad_clip and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if cfg.tbptt_steps and ((n_steps + 1) % cfg.tbptt_steps == 0):
            if state is not None:
                state.detach_()

        step_metrics = {"loss": float(loss.item()), **metrics}
        _acc_update(acc, step_metrics)
        n_steps += 1

        if cfg.log_every and (n_steps % cfg.log_every) == 0:
            print(
                f"[step {n_steps:6d}] "
                f"loss={step_metrics['loss']:.4f} "
                f"jaccard={step_metrics['jaccard']:.4f} "
                f"f1={step_metrics['f1']:.4f} "
                f"pr_auc={step_metrics['pr_auc']:.4f} "
                f"roc_auc={step_metrics['roc_auc']:.4f}"
            )

        prev = curr

    out = _acc_finalize(acc)
    if kappa_n > 0:
        out["kappa_mean"] = kappa_sum / kappa_n
    return out


def evaluate_whole_bin_stream_sliced(
    model,
    bins: Iterable[EventBatch],
    cfg: WholeBinTrainConfig,
    *,
    slices: EvalSlices,
) -> Dict[str, float]:
    model.eval()
    device = torch.device(cfg.device)
    state = model.init_state(batch_size=1, num_nodes=cfg.num_nodes, device=device)

    overall = _acc_init()
    early = _acc_init()
    late = _acc_init()

    prev: Optional[EventBatch] = None
    scored_step = 0

    with torch.no_grad():
        for events in bins:
            events = events.to(device)
            if prev is None:
                prev = events
                continue

            state, _ = model.step(state, prev)
            loss_t, metrics = whole_bin_edge_loss_and_metrics(
                model=model,
                state=state,
                next_events=events,
                num_nodes=cfg.num_nodes,
                upper_triangle_only=cfg.upper_triangle_only,
                include_self_loops=cfg.include_self_loops,
                decision_threshold=cfg.decision_threshold,
                pos_weight=cfg.pos_weight,
                auto_pos_weight=cfg.auto_pos_weight,
                max_auto_pos_weight=cfg.max_auto_pos_weight,
            )
            if state is not None:
                state.detach_()

            step_metrics = {"loss": float(loss_t.item()), **metrics}
            _acc_update(overall, step_metrics)
            if scored_step < slices.early_steps:
                _acc_update(early, step_metrics)
            else:
                _acc_update(late, step_metrics)

            scored_step += 1
            prev = events

    o = _acc_finalize(overall)
    e = _acc_finalize(early)
    l = _acc_finalize(late)

    out = dict(o)
    for prefix, block in (("early", e), ("late", l)):
        out[f"{prefix}_loss"] = block["loss"]
        out[f"{prefix}_jaccard"] = block["jaccard"]
        out[f"{prefix}_f1"] = block["f1"]
        out[f"{prefix}_pr_auc"] = block["pr_auc"]
        out[f"{prefix}_roc_auc"] = block["roc_auc"]
        out[f"{prefix}_steps"] = block["steps"]
    return out


def run_one_whole_bin_experiment(
    ds,
    spec,
    base_train_cfg: WholeBinTrainConfig,
    run: SweepRun,
    build_model_fn: Callable[[Any, Any], torch.nn.Module],
    epochs: int = 5,
    eval_slices: Optional[EvalSlices] = None,
    save_jsonl_path: Optional[str] = None,
    save_summary_path: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> tuple[WholeBinRunResult, torch.nn.Module, WholeBinTrainConfig]:
    device = torch.device(base_train_cfg.device)
    set_seed(run.seed)

    train_cfg = WholeBinTrainConfig(**asdict(base_train_cfg))
    if run.lr is not None:
        train_cfg.lr = run.lr
    if run.weight_decay is not None:
        train_cfg.weight_decay = run.weight_decay
    if run.tbptt_steps is not None:
        train_cfg.tbptt_steps = run.tbptt_steps

    model = build_model_fn(spec, run.model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    if eval_slices is None:
        eval_slices = EvalSlices(early_steps=10)

    selection_metric = str(train_cfg.selection_metric)
    best_val_metric = float("-inf")
    best_epoch = -1
    best_snapshot: dict = {}
    best_model_state = None

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_stats_step = train_one_epoch_whole_bin(model, ds.bins("train"), optimizer, train_cfg)
        train_eval = evaluate_whole_bin_stream_sliced(model, ds.bins("train"), train_cfg, slices=eval_slices)
        val_stats = evaluate_whole_bin_stream_sliced(model, ds.bins("val"), train_cfg, slices=eval_slices)
        test_stats = evaluate_whole_bin_stream_sliced(model, ds.bins("test"), train_cfg, slices=eval_slices)

        snapshot = {
            "epoch": epoch,
            "train_step": train_stats_step,
            "train_eval": train_eval,
            "val": val_stats,
            "test": test_stats,
        }

        km = train_stats_step.get("kappa_mean", None)
        kappa_str = f"{km:.4f}" if km is not None else ""
        print(
            f"[{run.name} | seed={run.seed} | epoch {epoch:03d}] "
            f"train(step) jaccard={train_stats_step['jaccard']:.4f} "
            f"kappa={kappa_str} | "
            f"train(eval) jaccard={train_eval['jaccard']:.4f} "
            f"val {selection_metric}={val_stats.get(selection_metric, float('nan')):.4f} "
            f"test jaccard={test_stats['jaccard']:.4f} "
            f"test pr_auc={test_stats['pr_auc']:.4f}"
        )

        val_metric = float(val_stats.get(selection_metric, float("nan")))
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_epoch = epoch
            best_snapshot = snapshot
            best_model_state = copy.deepcopy(model.state_dict())

        if save_jsonl_path is not None:
            row = {
                "dataset": dataset_name,
                "run": run.name,
                "seed": run.seed,
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
                "selection_metric": selection_metric,
                **snapshot,
            }
            with open(save_jsonl_path, "a") as f:
                f.write(json.dumps(row) + "\n")

    wall = time.time() - t0
    final_snapshot = snapshot

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    summary = {
        "dataset": dataset_name,
        "run": run.name,
        "seed": run.seed,
        "epochs": epochs,
        "selection_metric": selection_metric,
        "best_val_metric": best_val_metric,
        "best_epoch": best_epoch,
        "best_snapshot": best_snapshot,
        "final_snapshot": final_snapshot,
        "wall_sec": wall,
    }
    if save_summary_path is not None:
        with open(save_summary_path, "a") as f:
            f.write(json.dumps(summary) + "\n")

    result = WholeBinRunResult(
        dataset=dataset_name,
        name=run.name,
        seed=run.seed,
        epochs=epochs,
        selection_metric=selection_metric,
        best_val_metric=best_val_metric,
        best_epoch=best_epoch,
        best_snapshot=best_snapshot,
        final_snapshot=final_snapshot,
        wall_sec=wall,
    )
    return result, model, train_cfg