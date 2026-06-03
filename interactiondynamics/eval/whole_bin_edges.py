from __future__ import annotations

import math
from typing import Dict, Tuple, cast

import torch
import torch.nn.functional as F

from core.events import EventBatch

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception:  # pragma: no cover
    average_precision_score = None
    roc_auc_score = None


def _build_candidate_pairs(
    num_nodes: int,
    device: torch.device,
    upper_triangle_only: bool,
    include_self_loops: bool,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    if upper_triangle_only:
        src, dst = torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)
        return cast(torch.LongTensor, src.long()), cast(torch.LongTensor, dst.long())

    nodes = torch.arange(num_nodes, device=device, dtype=torch.long)
    src = nodes.repeat_interleave(num_nodes)
    dst = nodes.repeat(num_nodes)
    if not include_self_loops:
        keep = src != dst
        src = src[keep]
        dst = dst[keep]
    return cast(torch.LongTensor, src), cast(torch.LongTensor, dst)


def build_whole_bin_candidate_eventbatch(
    next_events: EventBatch,
    num_nodes: int,
    *,
    upper_triangle_only: bool = False,
    include_self_loops: bool = False,
) -> EventBatch:
    device = next_events.src.device
    src, dst = _build_candidate_pairs(
        num_nodes=num_nodes,
        device=device,
        upper_triangle_only=upper_triangle_only,
        include_self_loops=include_self_loops,
    )

    t_rep = None
    if next_events.t is not None:
        assert int(next_events.t.min().item()) == int(next_events.t.max().item()), (
            "Expected all events in a bin to share the same timestamp."
        )
        t0 = cast(torch.LongTensor, next_events.t[:1].to(device=device, dtype=torch.long))
        t_rep = cast(torch.LongTensor, t0.expand(src.numel()).clone())

    return EventBatch(src=src, dst=dst, t=t_rep, features=None)


def build_whole_bin_labels(
    next_events: EventBatch,
    candidate_events: EventBatch,
    num_nodes: int,
    *,
    upper_triangle_only: bool = False,
    include_self_loops: bool = False,
) -> torch.Tensor:
    pos_src = next_events.src.to(device=candidate_events.src.device, dtype=torch.long)
    pos_dst = next_events.dst.to(device=candidate_events.src.device, dtype=torch.long)

    if upper_triangle_only:
        lo = torch.minimum(pos_src, pos_dst)
        hi = torch.maximum(pos_src, pos_dst)
        valid = lo < hi
        pos_keys = lo[valid] * num_nodes + hi[valid]

        cand_lo = torch.minimum(candidate_events.src, candidate_events.dst)
        cand_hi = torch.maximum(candidate_events.src, candidate_events.dst)
        cand_keys = cand_lo * num_nodes + cand_hi
    else:
        if include_self_loops:
            valid = torch.ones_like(pos_src, dtype=torch.bool)
        else:
            valid = pos_src != pos_dst
        pos_keys = pos_src[valid] * num_nodes + pos_dst[valid]
        cand_keys = candidate_events.src * num_nodes + candidate_events.dst

    if pos_keys.numel() == 0:
        return torch.zeros(cand_keys.numel(), device=cand_keys.device, dtype=torch.float32)

    pos_keys = torch.unique(pos_keys)
    return torch.isin(cand_keys, pos_keys).to(dtype=torch.float32)


def _resolve_pos_weight(
    labels: torch.Tensor,
    *,
    pos_weight: float | None,
    auto_pos_weight: bool,
    max_auto_pos_weight: float | None,
) -> torch.Tensor | None:
    if pos_weight is not None:
        return torch.tensor(float(pos_weight), device=labels.device, dtype=torch.float32)
    if not auto_pos_weight:
        return None

    pos = float(labels.sum().item())
    neg = float(labels.numel() - labels.sum().item())
    if pos <= 0.0 or neg <= 0.0:
        return None

    weight = neg / pos
    if max_auto_pos_weight is not None:
        weight = min(weight, float(max_auto_pos_weight))
    return torch.tensor(weight, device=labels.device, dtype=torch.float32)


@torch.no_grad()
def binary_edge_metrics_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    decision_threshold: float = 0.5,
) -> Dict[str, float]:
    probs = torch.sigmoid(logits.detach()).view(-1)
    labels_f = labels.detach().float().view(-1)

    preds = (probs >= float(decision_threshold)).float()
    tp = float(((preds == 1.0) & (labels_f == 1.0)).sum().item())
    fp = float(((preds == 1.0) & (labels_f == 0.0)).sum().item())
    fn = float(((preds == 0.0) & (labels_f == 1.0)).sum().item())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    roc_auc, pr_auc = float("nan"), float("nan")
    if labels_f.numel() > 0 and len(set(labels_f.cpu().tolist())) >= 2:
        if roc_auc_score is not None and average_precision_score is not None:
            y_true = labels_f.cpu().numpy()
            y_score = probs.cpu().numpy()
            try:
                roc_auc = float(roc_auc_score(y_true, y_score))
            except Exception:
                pass
            try:
                pr_auc = float(average_precision_score(y_true, y_score))
            except Exception:
                pass

    return {
        "jaccard": jaccard,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
    }


def select_decision_threshold(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    metric: str = "f1",
) -> float:
    """Pick a probability threshold on pooled val scores (same idea as dummy baselines)."""
    if labels.numel() == 0:
        return 0.5
    best_t, best_v = 0.5, float("-inf")
    for t in torch.linspace(0.0, 1.0, 201).tolist():
        v = binary_edge_metrics_from_logits(logits, labels, decision_threshold=float(t)).get(
            metric, float("nan")
        )
        if math.isfinite(v) and v > best_v:
            best_t, best_v = float(t), float(v)
    return best_t


@torch.no_grad()
def whole_bin_logits_and_labels(
    model,
    state,
    next_events: EventBatch,
    num_nodes: int,
    *,
    upper_triangle_only: bool = False,
    include_self_loops: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cand_batch = build_whole_bin_candidate_eventbatch(
        next_events,
        num_nodes,
        upper_triangle_only=upper_triangle_only,
        include_self_loops=include_self_loops,
    )
    labels = build_whole_bin_labels(
        next_events,
        cand_batch,
        num_nodes,
        upper_triangle_only=upper_triangle_only,
        include_self_loops=include_self_loops,
    )
    state_eval = state.clone(detach=True) if state is not None else state
    logits = model.score(state_eval, cand_batch).view(-1)
    return logits, labels


def whole_bin_edge_loss_and_metrics(
    model,
    state,
    next_events: EventBatch,
    num_nodes: int,
    *,
    upper_triangle_only: bool = False,
    include_self_loops: bool = False,
    decision_threshold: float = 0.5,
    pos_weight: float | None = None,
    auto_pos_weight: bool = True,
    max_auto_pos_weight: float | None = 50.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    cand_batch = build_whole_bin_candidate_eventbatch(
        next_events,
        num_nodes,
        upper_triangle_only=upper_triangle_only,
        include_self_loops=include_self_loops,
    )
    labels = build_whole_bin_labels(
        next_events,
        cand_batch,
        num_nodes,
        upper_triangle_only=upper_triangle_only,
        include_self_loops=include_self_loops,
    )

    if state is not None and getattr(state, "node", None) is not None:
        before = state.node.detach().clone()
        state_eval = state.clone(detach=not torch.is_grad_enabled())
    else:
        before = None
        state_eval = state

    logits = model.score(state_eval, cand_batch).view(-1)
    assert logits.shape == labels.shape

    if before is not None:
        assert torch.equal(before, state.node.detach()), "score() mutated state.node"
    if not torch.isfinite(logits).all():
        raise RuntimeError("Non-finite logits from model.score()")

    pw = _resolve_pos_weight(
        labels,
        pos_weight=pos_weight,
        auto_pos_weight=auto_pos_weight,
        max_auto_pos_weight=max_auto_pos_weight,
    )
    loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pw)

    metrics = binary_edge_metrics_from_logits(
        logits,
        labels,
        decision_threshold=decision_threshold,
    )
    return loss, metrics
