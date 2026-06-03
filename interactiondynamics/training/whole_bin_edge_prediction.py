from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass, replace
from itertools import zip_longest
from typing import Any, Callable, Dict, Iterable, Optional

import torch

from core.events import EventBatch
from eval.evaluate import EvalSlices
from eval.whole_bin_edges import (
    binary_edge_metrics_from_logits,
    select_decision_threshold,
    whole_bin_edge_loss_and_metrics,
    whole_bin_logits_and_labels,
)
from experiments.interaction_prediction_runs import SweepRun
from experiments.results_summary import TermColor, color_text
from training.ift_aux import accumulate_ift_aux, finalize_ift_aux
from utils.repro import set_seed

_SENTINEL = object()
_METRIC_KEYS = ("loss", "jaccard", "f1", "pr_auc", "roc_auc")


def _aligned_target_context_bins(
    target_bins: Iterable[EventBatch],
    context_bins: Optional[Iterable[EventBatch]] = None,
):
    if context_bins is None:
        for clean in target_bins:
            yield clean, clean
        return

    for i, pair in enumerate(zip_longest(target_bins, context_bins, fillvalue=_SENTINEL)):
        clean, observed = pair
        if clean is _SENTINEL or observed is _SENTINEL:
            raise RuntimeError(
                f"Target/context stream length mismatch at bin {i}. "
                "Use skip_empty_observed_bins=False for paired corruption."
            )
        yield clean, observed


def _acc_init() -> Dict[str, float]:
    return {k: 0.0 for k in _METRIC_KEYS} | {"steps": 0.0}


def _acc_update(acc: Dict[str, float], metrics: Dict[str, float]) -> None:
    for k in _METRIC_KEYS:
        v = float(metrics.get(k, float("nan")))
        if v == v:
            acc[k] += v
    acc["steps"] += 1.0


def _acc_finalize(acc: Dict[str, float]) -> Dict[str, float]:
    steps = int(acc.get("steps", 0.0))
    if steps == 0:
        return {k: float("nan") for k in _METRIC_KEYS} | {"steps": 0.0}
    return {k: acc[k] / steps for k in _METRIC_KEYS} | {"steps": float(steps)}


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
    decision_threshold: float = 0.5
    upper_triangle_only: bool = True
    include_self_loops: bool = False
    pos_weight: float | None = None
    auto_pos_weight: bool = True
    max_auto_pos_weight: float | None = 50.0
    selection_metric: str = "pr_auc"  # "pr_auc" (macro per-bin) or "micro_pr_auc" (pooled val)
    tune_threshold: bool = True
    threshold_metric: str = "f1"


# so we aren't repeating the same keyword arguments every time we call whole_bin_edge_loss_and_metrics
def _loss_kw(cfg: WholeBinTrainConfig) -> dict:
    return dict(
        upper_triangle_only=cfg.upper_triangle_only,
        include_self_loops=cfg.include_self_loops,
        decision_threshold=cfg.decision_threshold,
        pos_weight=cfg.pos_weight,
        auto_pos_weight=cfg.auto_pos_weight,
        max_auto_pos_weight=cfg.max_auto_pos_weight,
    )


def train_one_epoch_whole_bin(
    model,
    bins: Iterable[EventBatch],
    optimizer: torch.optim.Optimizer,
    cfg: WholeBinTrainConfig,
    context_bins: Optional[Iterable[EventBatch]] = None,
) -> Dict[str, float]:
    model.train()
    device = torch.device(cfg.device)
    state = model.init_state(batch_size=1, num_nodes=cfg.num_nodes, device=device)

    acc = _acc_init()
    ift_sums: Dict[str, float] = {}
    ift_n = 0
    prev_context: Optional[EventBatch] = None
    n_steps = 0

    for curr_clean, curr_context in _aligned_target_context_bins(bins, context_bins):
        curr_clean = curr_clean.to(device)
        curr_context = curr_context.to(device)

        if prev_context is None:
            prev_context = curr_context
            continue

        state, aux = model.step(state, prev_context)
        if accumulate_ift_aux(aux, ift_sums):
            ift_n += 1

        optimizer.zero_grad(set_to_none=True)
        loss, metrics = whole_bin_edge_loss_and_metrics(
            model=model,
            state=state,
            next_events=curr_clean,
            num_nodes=cfg.num_nodes,
            **_loss_kw(cfg),
        )
        loss.backward()

        if cfg.grad_clip and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if cfg.tbptt_steps and ((n_steps + 1) % cfg.tbptt_steps == 0) and state is not None:
            state.detach_()

        _acc_update(acc, {"loss": float(loss.item()), **metrics})
        n_steps += 1

        if cfg.log_every and (n_steps % cfg.log_every) == 0:
            print(
                f"[step {n_steps:6d}] loss={loss.item():.4f} "
                f"jaccard={metrics['jaccard']:.4f} f1={metrics['f1']:.4f} "
                f"pr_auc={metrics['pr_auc']:.4f}"
            )

        prev_context = curr_context

    out = _acc_finalize(acc)
    out.update(finalize_ift_aux(ift_sums, ift_n))
    return out


def evaluate_whole_bin_stream_sliced(
    model,
    bins: Iterable[EventBatch],
    cfg: WholeBinTrainConfig,
    *,
    slices: EvalSlices = EvalSlices(),
    context_bins: Optional[Iterable[EventBatch]] = None,
) -> Dict[str, float]:
    model.eval()
    device = torch.device(cfg.device)
    state = model.init_state(batch_size=1, num_nodes=cfg.num_nodes, device=device)

    overall = _acc_init()
    early = _acc_init()
    late = _acc_init()
    prev_context: Optional[EventBatch] = None
    scored_step = 0

    with torch.no_grad():
        for curr_clean, curr_context in _aligned_target_context_bins(bins, context_bins):
            curr_clean = curr_clean.to(device)
            curr_context = curr_context.to(device)

            if prev_context is None:
                prev_context = curr_context
                continue

            state, _ = model.step(state, prev_context)
            loss_t, metrics = whole_bin_edge_loss_and_metrics(
                model=model,
                state=state,
                next_events=curr_clean,
                num_nodes=cfg.num_nodes,
                **_loss_kw(cfg),
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
            prev_context = curr_context

    out = _acc_finalize(overall)
    for prefix, block in (("early", _acc_finalize(early)), ("late", _acc_finalize(late))):
        for k in _METRIC_KEYS + ("steps",):
            out[f"{prefix}_{k}"] = block[k]
    return out


@torch.no_grad()
def _collect_pooled_logits_labels(
    model,
    bins: Iterable[EventBatch],
    cfg: WholeBinTrainConfig,
    context_bins: Optional[Iterable[EventBatch]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate per-edge logits/labels over a split (for val threshold tuning)."""
    model.eval()
    device = torch.device(cfg.device)
    state = model.init_state(batch_size=1, num_nodes=cfg.num_nodes, device=device)
    labels_all: list[torch.Tensor] = []
    logits_all: list[torch.Tensor] = []
    prev_context: Optional[EventBatch] = None

    for curr_clean, curr_context in _aligned_target_context_bins(bins, context_bins):
        curr_clean = curr_clean.to(device)
        curr_context = curr_context.to(device)
        if prev_context is None:
            prev_context = curr_context
            continue

        state, _ = model.step(state, prev_context)
        logits, labels = whole_bin_logits_and_labels(
            model,
            state,
            curr_clean,
            cfg.num_nodes,
            upper_triangle_only=cfg.upper_triangle_only,
            include_self_loops=cfg.include_self_loops,
        )
        labels_all.append(labels.cpu())
        logits_all.append(logits.cpu())
        if state is not None:
            state.detach_()
        prev_context = curr_context

    if not labels_all:
        return torch.empty(0), torch.empty(0)
    return torch.cat(logits_all), torch.cat(labels_all)


def _tuned_pooled_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
) -> Dict[str, float]:
    m = binary_edge_metrics_from_logits(logits, labels, decision_threshold=threshold)
    return {
        "tuned_threshold": float(threshold),
        "tuned_jaccard": m["jaccard"],
        "tuned_f1": m["f1"],
        "micro_pr_auc": m["pr_auc"],
        "micro_roc_auc": m["roc_auc"],
    }


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
    clean_ds=None,
    corruption_cfg=None,
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

    model = build_model_fn(spec, replace(run.model_cfg, ift_h_init_seed=run.seed)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    if eval_slices is None:
        eval_slices = EvalSlices(early_steps=10)

    use_clean_targets = clean_ds is not None and corruption_cfg is not None
    selection_metric = str(train_cfg.selection_metric)
    best_val_metric = float("-inf")
    best_epoch = -1
    best_snapshot: dict = {}
    best_model_state = None

    def _target_bins(split: str):
        return clean_ds.bins(split) if use_clean_targets else ds.bins(split)

    def _context_bins(split: str):
        return ds.bins(split) if use_clean_targets else None

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_stats_step = train_one_epoch_whole_bin(
            model,
            _target_bins("train"),
            optimizer,
            train_cfg,
            context_bins=_context_bins("train"),
        )
        train_eval = evaluate_whole_bin_stream_sliced(
            model, _target_bins("train"), train_cfg, slices=eval_slices, context_bins=_context_bins("train"),
        )
        val_stats = evaluate_whole_bin_stream_sliced(
            model, _target_bins("val"), train_cfg, slices=eval_slices, context_bins=_context_bins("val"),
        )
        test_stats = evaluate_whole_bin_stream_sliced(
            model, _target_bins("test"), train_cfg, slices=eval_slices, context_bins=_context_bins("test"),
        )

        if train_cfg.tune_threshold:
            val_logits, val_labels = _collect_pooled_logits_labels(
                model,
                _target_bins("val"),
                train_cfg,
                context_bins=_context_bins("val"),
            )
            tuned_t = select_decision_threshold(
                val_logits,
                val_labels,
                metric=train_cfg.threshold_metric,
            )
            val_stats.update(_tuned_pooled_metrics(val_logits, val_labels, tuned_t))
            for block, bins, ctx in (
                (train_eval, _target_bins("train"), _context_bins("train")),
                (test_stats, _target_bins("test"), _context_bins("test")),
            ):
                logits, labels = _collect_pooled_logits_labels(
                    model, bins, train_cfg, context_bins=ctx,
                )
                block.update(_tuned_pooled_metrics(logits, labels, tuned_t))

        snapshot = {
            "epoch": epoch,
            "train_step": train_stats_step,
            "train_eval": train_eval,
            "val": val_stats,
            "test": test_stats,
        }

        km = train_stats_step.get("kappa_mean")
        kappa_str = f" kappa={km:.4f}" if km is not None else ""
        val_sel = float(
            val_stats.get(
                "micro_pr_auc" if selection_metric == "micro_pr_auc" else "pr_auc",
                float("nan"),
            )
        )
        epoch_line = (
            f"[{run.name} | seed={run.seed} | epoch {epoch:03d}] "
            f"train jaccard={train_stats_step['jaccard']:.4f}{kappa_str}"
        )
        if "micro_pr_auc" in val_stats:
            epoch_line += " | " + color_text(
                f"val_micro_pr_auc={float(val_stats['micro_pr_auc']):.4f}", TermColor.SOFT_PINK
            )
        elif val_sel == val_sel:
            epoch_line += f" | val {selection_metric}={val_sel:.4f}"
        if "micro_pr_auc" in test_stats:
            epoch_line += " | " + color_text(
                f"test_micro_pr_auc={float(test_stats['micro_pr_auc']):.4f}", TermColor.SOFT_PINK
            )
        if "tuned_jaccard" in val_stats:
            epoch_line += " | " + color_text(
                f"val_tuned_jaccard={float(val_stats['tuned_jaccard']):.4f}@t={float(val_stats.get('tuned_threshold', float('nan'))):.4f}",
                TermColor.SOFT_PINK,
            )
        if "tuned_jaccard" in test_stats:
            epoch_line += " | " + color_text(
                f"test_tuned_jaccard={float(test_stats['tuned_jaccard']):.4f}@t={float(test_stats.get('tuned_threshold', float('nan'))):.4f}",
                TermColor.SOFT_PINK,
            )
        elif "tuned_jaccard" not in val_stats:
            epoch_line += f" | test jaccard={test_stats['jaccard']:.4f}"
        print(epoch_line)

        val_for_compare = val_sel if val_sel == val_sel else float("-inf")
        if val_for_compare > best_val_metric:
            best_val_metric = val_for_compare
            best_epoch = epoch
            best_snapshot = snapshot
            best_model_state = copy.deepcopy(model.state_dict())

        if save_jsonl_path is not None:
            with open(save_jsonl_path, "a") as f:
                f.write(json.dumps({
                    "dataset": dataset_name,
                    "run": run.name,
                    "seed": run.seed,
                    "model_cfg": asdict(run.model_cfg),
                    "selection_metric": selection_metric,
                    **snapshot,
                }) + "\n")

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
