from __future__ import annotations

from typing import Dict

import torch

from datasets.corrupted import corrupt_batch_with_metadata
from datasets.interfaces import EventStreamDataset
from eval.ranking import ranking_loss_and_metrics
from utils.io import append_jsonl
from experiments.results_summary import (
    format_recovery_label,
    format_recovery_metrics,
)

def _acc_init() -> Dict[str, float]:
    return {
        "loss_sum": 0.0,
        "mrr_sum": 0.0,
        "hits1_sum": 0.0,
        "hits10_sum": 0.0,
        "steps": 0.0,
        "removed_events": 0.0,
    }


def _acc_update(acc: Dict[str, float], loss: float, metrics: Dict[str, float], removed_count: int) -> None:
    acc["loss_sum"] += float(loss)
    acc["mrr_sum"] += float(metrics.get("mrr", 0.0))
    acc["hits1_sum"] += float(metrics.get("hits@1", 0.0))
    acc["hits10_sum"] += float(metrics.get("hits@10", 0.0))
    acc["steps"] += 1.0
    acc["removed_events"] += float(removed_count)


def _acc_finalize(acc: Dict[str, float]) -> Dict[str, float]:
    if acc["steps"] <= 0:
        return {
            "loss": float("nan"),
            "mrr": float("nan"),
            "hits@1": float("nan"),
            "hits@10": float("nan"),
            "steps": 0,
            "removed_events": 0,
        }

    n = acc["steps"]
    return {
        "loss": acc["loss_sum"] / n,
        "mrr": acc["mrr_sum"] / n,
        "hits@1": acc["hits1_sum"] / n,
        "hits@10": acc["hits10_sum"] / n,
        "steps": int(n),
        "removed_events": int(acc["removed_events"]),
    }


@torch.no_grad()
def evaluate_hidden_positive_recovery(
    model,
    clean_ds: EventStreamDataset,
    cfg,
    *,
    split: str,
    drop_real_prob: float,
    add_fake_ratio: float,
    seed: int,
    fake_feature_mode: str = "zeros",
    avoid_self_loops: bool = True,
    min_keep_per_nonempty_bin: int = 1,
) -> Dict[str, float]:
    """
    Recovery eval:
      - state is updated using the OBSERVED corrupted previous bin
      - removed positives are taken from the CLEAN current bin
      - score only those removed true positives

    This directly answers:
      "did the model recover interactions that truly existed but were hidden?"
    """
    if drop_real_prob <= 0.0:
        return {
            "loss": float("nan"),
            "mrr": float("nan"),
            "hits@1": float("nan"),
            "hits@10": float("nan"),
            "steps": 0,
            "removed_events": 0,
        }

    model.eval()
    device = torch.device(cfg.device)
    state = model.init_state(batch_size=1, num_nodes=cfg.num_nodes, device=device)

    acc = _acc_init()
    prev_observed = None

    for batch_idx, clean_curr in enumerate(clean_ds.bins(split)):
        clean_curr = clean_curr.to(device)

        info = corrupt_batch_with_metadata(
            clean_curr,
            split=split,
            batch_idx=batch_idx,
            num_nodes=cfg.num_nodes,
            drop_real_prob=drop_real_prob,
            add_fake_ratio=add_fake_ratio,
            seed=seed,
            fake_feature_mode=fake_feature_mode,
            avoid_self_loops=avoid_self_loops,
            min_keep_per_nonempty_bin=min_keep_per_nonempty_bin,
        )
        observed_curr = info.observed.to(device)
        removed_curr = info.removed.to(device)

        if prev_observed is None:
            prev_observed = observed_curr
            continue

        # use corrupted/observed history to update memory
        state, _ = model.step(state, prev_observed)

        # now ask whether the model ranks the hidden true positives highly
        if removed_curr.num_events > 0:
            loss_t, metrics = ranking_loss_and_metrics(
                model=model,
                state=state,
                next_events=removed_curr,
                num_nodes=cfg.num_nodes,
                num_neg=cfg.num_neg,
            )
            _acc_update(
                acc,
                float(loss_t.item()),
                metrics,
                removed_count=removed_curr.num_events,
            )

        if state is not None:
            state.detach_()

        prev_observed = observed_curr

    return _acc_finalize(acc)

def evaluate_recovery_splits(
    *,
    model,
    clean_ds,
    train_cfg,
    corruption_cfg: dict,
    splits=("val", "test"),
):
    recovery = {}

    for split in splits:
        recovery[split] = evaluate_hidden_positive_recovery(
            model,
            clean_ds,
            train_cfg,
            split=split,
            drop_real_prob=corruption_cfg["drop_real_prob"],
            add_fake_ratio=corruption_cfg["add_fake_ratio"],
            seed=corruption_cfg["seed"],
            fake_feature_mode=corruption_cfg["fake_feature_mode"],
            avoid_self_loops=corruption_cfg["avoid_self_loops"],
            min_keep_per_nonempty_bin=corruption_cfg["min_keep_per_nonempty_bin"],
        )

    return recovery


def print_recovery_summary(recovery: dict) -> None:
    print(f"{format_recovery_label()} {format_recovery_metrics(recovery)}")


def append_recovery_summary_row(
    *,
    summary_jsonl: str,
    dataset_name: str,
    run_name: str,
    seed: int,
    recovery: dict,
) -> None:
    append_jsonl(
        summary_jsonl,
        {
            "dataset": dataset_name,
            "run": run_name,
            "seed": seed,
            "recovery": recovery,
        },
    )