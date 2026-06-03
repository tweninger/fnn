from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import math
import torch

from core.events import EventBatch
from eval.whole_bin_edges import (
    build_whole_bin_candidate_eventbatch,
    build_whole_bin_labels,
)

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception:  # pragma: no cover
    average_precision_score = None
    roc_auc_score = None


# Key whole-bin baselines: temporal, static popularity, and random floors.
BASELINE_NAMES = (
    "persistence",
    "edge_frequency",
    "random_uniform",
    "random_same_density",
)


@dataclass
class StaticBaselineStats:
    num_nodes: int
    num_candidate_edges: int
    upper_triangle_only: bool
    include_self_loops: bool
    num_train_bins: int
    train_edge_density: float
    pair_frequency: torch.Tensor  # [num_nodes * num_nodes]


@torch.no_grad()
def build_static_baseline_stats(
    clean_ds,
    *,
    split: str = "train",
    num_nodes: Optional[int] = None,
    upper_triangle_only: bool = True,
    include_self_loops: bool = False,
    device: torch.device | str = "cpu",
) -> StaticBaselineStats:
    """Train-split edge frequencies and density (from clean labels, not corrupted context)."""
    device = torch.device(device)
    n = int(num_nodes if num_nodes is not None else clean_ds.spec().num_nodes)

    empty = EventBatch(
        src=torch.empty(0, dtype=torch.long, device=device),
        dst=torch.empty(0, dtype=torch.long, device=device),
    )
    cand = build_whole_bin_candidate_eventbatch(
        empty, n,
        upper_triangle_only=upper_triangle_only,
        include_self_loops=include_self_loops,
    )
    num_candidate_edges = int(cand.src.numel())
    if num_candidate_edges <= 0:
        raise ValueError("No candidate edges.")

    pair_counts = torch.zeros(n * n, dtype=torch.float32, device=device)
    num_bins = 0
    density_sum = 0.0

    for batch in clean_ds.bins(split):
        batch = batch.to(device)
        keys = _unique_event_keys(
            batch, n,
            upper_triangle_only=upper_triangle_only,
            include_self_loops=include_self_loops,
            device=device,
        )
        num_bins += 1
        density_sum += float(keys.numel()) / float(num_candidate_edges)
        if keys.numel() > 0:
            pair_counts[keys] += 1.0

    if num_bins <= 0:
        raise ValueError(f"split {split!r} has no bins.")

    return StaticBaselineStats(
        num_nodes=n,
        num_candidate_edges=num_candidate_edges,
        upper_triangle_only=upper_triangle_only,
        include_self_loops=include_self_loops,
        num_train_bins=num_bins,
        train_edge_density=density_sum / num_bins,
        pair_frequency=pair_counts / float(num_bins),
    )


def _edge_keys(
    src: torch.Tensor,
    dst: torch.Tensor,
    num_nodes: int,
    *,
    upper_triangle_only: bool,
    include_self_loops: bool,
) -> torch.LongTensor:
    src, dst = src.long(), dst.long()
    if upper_triangle_only:
        lo, hi = torch.minimum(src, dst), torch.maximum(src, dst)
        keep = lo < hi
        return (lo[keep] * num_nodes + hi[keep]).long()
    if not include_self_loops:
        keep = src != dst
        src, dst = src[keep], dst[keep]
    return (src * num_nodes + dst).long()


def _unique_event_keys(
    batch: EventBatch,
    num_nodes: int,
    *,
    upper_triangle_only: bool,
    include_self_loops: bool,
    device: torch.device,
) -> torch.LongTensor:
    keys = _edge_keys(
        batch.src.to(device), batch.dst.to(device), num_nodes,
        upper_triangle_only=upper_triangle_only,
        include_self_loops=include_self_loops,
    )
    return torch.unique(keys) if keys.numel() else keys


def _candidate_keys(cand: EventBatch, stats: StaticBaselineStats) -> torch.LongTensor:
    return _edge_keys(
        cand.src, cand.dst, stats.num_nodes,
        upper_triangle_only=stats.upper_triangle_only,
        include_self_loops=stats.include_self_loops,
    )


def _score_baseline_for_bin(
    baseline: str,
    cand: EventBatch,
    context_history: Sequence[EventBatch],
    stats: StaticBaselineStats,
    *,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    device = cand.src.device
    cand_keys = _candidate_keys(cand, stats)
    m = int(cand.src.numel())

    if baseline == "random_uniform":
        if generator is None:
            return torch.rand(m, dtype=torch.float32, device=device)
        return torch.rand(m, dtype=torch.float32, generator=generator).to(device)

    if baseline == "random_same_density":
        scores = torch.zeros(m, dtype=torch.float32, device=device)
        k = max(0, min(int(round(stats.train_edge_density * m)), m))
        if k > 0:
            perm = torch.randperm(m, generator=generator) if generator is not None else torch.randperm(m)
            scores[perm[:k].to(device)] = 1.0
        return scores

    if baseline == "persistence":
        if not context_history:
            return torch.zeros(m, dtype=torch.float32, device=device)
        prev_keys = _unique_event_keys(
            context_history[-1], stats.num_nodes,
            upper_triangle_only=stats.upper_triangle_only,
            include_self_loops=stats.include_self_loops,
            device=device,
        )
        if prev_keys.numel() == 0:
            return torch.zeros(m, dtype=torch.float32, device=device)
        return torch.isin(cand_keys, prev_keys).float()

    if baseline == "edge_frequency":
        return stats.pair_frequency.to(device)[cand_keys].float()

    raise ValueError(f"Unknown baseline {baseline!r}. Expected one of {BASELINE_NAMES}.")


def _binary_metrics(labels: torch.Tensor, scores: torch.Tensor, threshold: float) -> Dict[str, float]:
    labels_f = labels.float().view(-1)
    preds = (scores.float().view(-1) >= threshold).float()

    tp = float(((preds == 1) & (labels_f == 1)).sum())
    fp = float(((preds == 1) & (labels_f == 0)).sum())
    fn = float(((preds == 0) & (labels_f == 1)).sum())
    tn = float(((preds == 0) & (labels_f == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    pr_auc, roc_auc = float("nan"), float("nan")
    if labels_f.numel() > 0 and torch.unique(labels_f).numel() >= 2:
        if average_precision_score is not None and roc_auc_score is not None:
            y, s = labels_f.cpu().numpy(), scores.float().view(-1).cpu().numpy()
            try:
                pr_auc = float(average_precision_score(y, s))
                roc_auc = float(roc_auc_score(y, s))
            except Exception:
                pass

    edge_density = float(labels_f.mean()) if labels_f.numel() else float("nan")
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "edge_density": edge_density,
    }


def _select_threshold(labels: torch.Tensor, scores: torch.Tensor, *, metric: str = "f1") -> float:
    if labels.numel() == 0:
        return 0.5
    best_t, best_v = 0.5, float("-inf")
    for t in torch.linspace(0.0, 1.0, 201).tolist():
        v = _binary_metrics(labels, scores, t).get(metric, float("nan"))
        if math.isfinite(v) and v > best_v:
            best_t, best_v = float(t), float(v)
    return best_t


@torch.no_grad()
def _evaluate_split(
    clean_ds,
    *,
    split: str,
    baseline: str,
    stats: StaticBaselineStats,
    context_ds=None,
    threshold: float = 0.5,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> Dict[str, float]:
    device = torch.device(device)
    clean_bins = list(clean_ds.bins(split))
    context_bins = list(context_ds.bins(split)) if context_ds is not None else clean_bins
    if len(clean_bins) != len(context_bins):
        raise RuntimeError(f"clean/context length mismatch on {split!r}")

    gen = torch.Generator()
    gen.manual_seed(seed + {"train": 11, "val": 17, "test": 23}.get(split, 31))

    labels_all: List[torch.Tensor] = []
    scores_all: List[torch.Tensor] = []
    per_bin: List[Dict[str, float]] = []
    context_history: List[EventBatch] = []

    for i in range(1, len(clean_bins)):
        context_history.append(context_bins[i - 1].to(device))
        curr = clean_bins[i].to(device)
        cand = build_whole_bin_candidate_eventbatch(
            curr, stats.num_nodes,
            upper_triangle_only=stats.upper_triangle_only,
            include_self_loops=stats.include_self_loops,
        )
        labels = build_whole_bin_labels(
            curr, cand, stats.num_nodes,
            upper_triangle_only=stats.upper_triangle_only,
            include_self_loops=stats.include_self_loops,
        ).float()
        scores = _score_baseline_for_bin(baseline, cand, context_history, stats, generator=gen).float()
        labels_all.append(labels.cpu())
        scores_all.append(scores.cpu())
        per_bin.append(_binary_metrics(labels.cpu(), scores.cpu(), threshold))

    if not labels_all:
        return {"steps": 0.0, "threshold": threshold}

    labels_cat = torch.cat(labels_all)
    scores_cat = torch.cat(scores_all)
    micro = _binary_metrics(labels_cat, scores_cat, threshold)
    macro = {k: _finite_mean(m[k] for m in per_bin) for k in micro}

    return {
        "threshold": float(threshold),
        "steps": float(len(per_bin)),
        **{f"micro_{k}": v for k, v in micro.items()},
        **{f"macro_{k}": v for k, v in macro.items()},
    }


def _finite_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


@torch.no_grad()
def run_whole_bin_baseline_suite(
    clean_ds,
    *,
    context_ds=None,
    num_nodes: Optional[int] = None,
    upper_triangle_only: bool = True,
    include_self_loops: bool = False,
    baselines: Sequence[str] = BASELINE_NAMES,
    threshold_metric: str = "f1",
    threshold_split: str = "val",
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> List[Dict[str, object]]:
    """Run baselines on train/val/test. Threshold picked on val, applied to all splits."""
    device = torch.device(device)
    n = int(num_nodes if num_nodes is not None else clean_ds.spec().num_nodes)
    stats = build_static_baseline_stats(
        clean_ds, split="train", num_nodes=n,
        upper_triangle_only=upper_triangle_only,
        include_self_loops=include_self_loops,
        device=device,
    )

    rows: List[Dict[str, object]] = []
    for baseline in baselines:
        if baseline not in BASELINE_NAMES:
            raise ValueError(f"Unknown baseline {baseline!r}")

        val_labels, val_scores = [], []
        context_history: List[EventBatch] = []
        clean_bins = list(clean_ds.bins(threshold_split))
        context_bins = list(context_ds.bins(threshold_split)) if context_ds is not None else clean_bins
        gen = torch.Generator()
        gen.manual_seed(seed + 17)

        for i in range(1, len(clean_bins)):
            context_history.append(context_bins[i - 1].to(device))
            curr = clean_bins[i].to(device)
            cand = build_whole_bin_candidate_eventbatch(
                curr, n,
                upper_triangle_only=upper_triangle_only,
                include_self_loops=include_self_loops,
            )
            labels = build_whole_bin_labels(
                curr, cand, n,
                upper_triangle_only=upper_triangle_only,
                include_self_loops=include_self_loops,
            ).float()
            scores = _score_baseline_for_bin(baseline, cand, context_history, stats, generator=gen).float()
            val_labels.append(labels.cpu())
            val_scores.append(scores.cpu())

        threshold = _select_threshold(
            torch.cat(val_labels) if val_labels else torch.empty(0),
            torch.cat(val_scores) if val_scores else torch.empty(0),
            metric=threshold_metric,
        )

        for split in ("train", "val", "test"):
            metrics = _evaluate_split(
                clean_ds, split=split, baseline=baseline, stats=stats,
                context_ds=context_ds, threshold=threshold, seed=seed, device=device,
            )
            rows.append({
                "dataset": clean_ds.spec().name,
                "baseline": baseline,
                "split": split,
                "seed": int(seed),
                "selected_threshold": threshold,
                "train_edge_density": stats.train_edge_density,
                **metrics,
            })

    return rows
