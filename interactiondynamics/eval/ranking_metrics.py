from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, cast

import torch
import torch.nn.functional as F

from interactiondynamics.core.events import EventBatch
from interactiondynamics.eval.prediction_metrics import binary_metrics_from_logits


@dataclass
class RankingBatch:
    """
    Candidate set for each positive event i:
      candidates[i, 0] is the positive dst
      candidates[i, 1:] are negatives
    """
    src: torch.Tensor          # (M,)
    candidates_dst: torch.Tensor  # (M, K+1)
    t: torch.Tensor | None = None
    features: torch.Tensor | None = None


def sample_negative_dsts(
    num_nodes: int,
    pos_dst: torch.Tensor,
    num_neg: int,
    device: torch.device,
    avoid: torch.Tensor | None = None,   # (M,) optional (e.g., src)
    batch: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Uniform negative sampling over node ids, avoiding collisions with pos_dst (and optionally avoid).
    Returns (M, num_neg) LongTensor on `device`.
    """
    M = pos_dst.numel()
    if num_nodes <= 1:
        return torch.zeros((M, num_neg), device=device, dtype=torch.long)

    base = (
        torch.zeros((M, 1), device=device, dtype=torch.long)
        if batch is None
        else batch.to(device=device, dtype=torch.long).view(-1, 1) * num_nodes
    )
    local_pos = pos_dst.view(-1, 1) - base
    neg = torch.randint(0, num_nodes, (M, num_neg), device=device, dtype=torch.long)

    # Fix collisions with pos_dst (one pass is usually fine)
    collide = neg.eq(local_pos)
    if collide.any():
        neg[collide] = (neg[collide] + 1) % num_nodes

    # Optional: avoid another id (e.g., src for no self-loop negatives)
    if avoid is not None:
        collide2 = neg.eq(avoid.view(-1, 1) - base)
        if collide2.any():
            neg[collide2] = (neg[collide2] + 1) % num_nodes
            # might re-collide with pos_dst in rare cases; acceptable for now

    return cast(torch.LongTensor, neg + base)


def sample_filtered_negative_dsts(
    num_nodes: int,
    src: torch.Tensor,
    dst: torch.Tensor,
    num_neg: int,
    device: torch.device,
    batch: torch.Tensor | None = None,
) -> tuple[torch.LongTensor, torch.BoolTensor]:
    """Sample destinations that are not active positives in the same bin.

    For every positive ``src[i] -> dst[i]``, the filtered candidate set removes
    *all* destinations observed for ``src[i]`` in that bin, not only ``dst[i]``.
    This implements the standard filtered temporal-link-prediction convention:
    another true simultaneous interaction must never be treated as a negative.

    Returns sampled destinations and a row-validity mask. A row is invalid only
    when its source is connected to every node in the current bin, leaving no
    true negative destination to sample.
    """
    M = int(src.numel())
    neg = torch.empty((M, num_neg), dtype=torch.long, device=device)
    valid = torch.zeros((M,), dtype=torch.bool, device=device)
    if num_nodes <= 0 or M == 0:
        return neg, valid

    # Build the same per-source exclusion set as the former Python loop, but
    # sample every row on-device at once. The old implementation synchronized
    # the GPU once for every active source in every evaluation bin, which made
    # dense physical-force streams overwhelmingly CPU-bound.
    total_nodes = num_nodes if batch is None else num_nodes * (int(batch.max().item()) + 1)
    active = torch.zeros((total_nodes, total_nodes), dtype=torch.bool, device=device)
    active[src, dst] = True
    source_has_candidate = (~active).any(dim=1)
    valid = source_has_candidate[src]

    # Rejection sampling is exactly uniform over inactive destinations. Field
    # topologies have only a few active destinations per source, so nearly all
    # rows succeed on the first draw; the loop is typically entered zero or one
    # times rather than once per source.
    base = (
        torch.zeros((M, 1), device=device, dtype=torch.long)
        if batch is None
        else batch.to(device=device, dtype=torch.long).view(-1, 1) * num_nodes
    )
    neg = torch.randint(num_nodes, (M, num_neg), device=device, dtype=torch.long) + base
    blocked = active[src.view(-1, 1), neg] & valid.view(-1, 1)
    while bool(blocked.any()):
        redraw = torch.randint(num_nodes, (M, num_neg), device=device, dtype=torch.long) + base
        neg = torch.where(blocked, redraw, neg)
        blocked = active[src.view(-1, 1), neg] & valid.view(-1, 1)
    return cast(torch.LongTensor, neg), cast(torch.BoolTensor, valid)


def sample_balanced_inactive_pairs(
    num_nodes: int,
    observed_events: EventBatch,
    num_samples: int,
    device: torch.device,
    *,
    return_batch: bool = False,
) -> tuple[torch.LongTensor, torch.LongTensor] | tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor | None]:
    """Uniformly sample currently unobserved directed pairs.

    This is used *only* for the balanced event-detection diagnostic.  Every
    endogenous event in the current bin is a positive; the same number of
    directed pairs that are absent from that bin are sampled as negatives.
    External interventions are excluded from both classes because their random
    occurrence is not a model prediction target.

    Sampling without replacement makes the 1:1 protocol exact whenever the
    graph has enough inactive pairs.  It deliberately does not condition on a
    known active source, unlike destination-ranking/MRR evaluation.
    """
    if num_nodes <= 0 or num_samples <= 0:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return (empty, empty, None) if return_batch else (empty, empty)

    def sample_missing_pair_ids(observed_ids: torch.Tensor, count: int) -> torch.LongTensor:
        """Sample distinct ranks in the complement without building N² pairs."""
        total_pairs = num_nodes * num_nodes
        observed_ids = torch.unique(observed_ids.to(device=device, dtype=torch.long), sorted=True)
        observed_ids = observed_ids[(observed_ids >= 0) & (observed_ids < total_pairs)]
        available = total_pairs - int(observed_ids.numel())
        count = min(count, available)
        if count <= 0:
            return torch.empty((0,), dtype=torch.long, device=device)

        # Draw only the requested number of complement ranks.  ``randperm``
        # over ``total_pairs`` would be as costly as the full N x N grid.
        chosen = torch.empty((0,), dtype=torch.long, device=device)
        while chosen.numel() < count:
            remaining = count - chosen.numel()
            proposal = torch.randint(available, (max(2 * remaining, 16),), device=device)
            proposal = torch.unique(proposal)
            if chosen.numel():
                proposal = proposal[~torch.isin(proposal, chosen)]
            chosen = torch.cat([chosen, proposal[:remaining]])

        # If p is a rank among absent pairs, inserting p after every observed
        # id whose number of preceding gaps is <= p gives its true pair id.
        gap_adjusted_observed = observed_ids - torch.arange(observed_ids.numel(), device=device)
        offsets = torch.searchsorted(gap_adjusted_observed, chosen, right=True)
        return cast(torch.LongTensor, chosen + offsets)

    batch = observed_events.batch
    if batch is None:
        observed_ids = observed_events.src * num_nodes + observed_events.dst
        chosen = sample_missing_pair_ids(observed_ids, int(num_samples))
        if chosen.numel() == 0:
            empty = torch.empty((0,), dtype=torch.long, device=device)
            return (empty, empty, None) if return_batch else (empty, empty)
        result = (cast(torch.LongTensor, chosen // num_nodes), cast(torch.LongTensor, chosen % num_nodes), None)
        return result if return_batch else result[:2]

    # Packed streams have disjoint node ID ranges. Draw one negative within
    # the same physical system as each positive, never across episodes.
    target_batch = batch[:num_samples].to(device=device, dtype=torch.long)
    neg_src_parts, neg_dst_parts, neg_batch_parts = [], [], []
    for batch_id in torch.unique(target_batch).tolist():
        count = int((target_batch == batch_id).sum().item())
        observed = batch == batch_id
        observed_ids = (
            observed_events.src[observed] - batch_id * num_nodes
        ) * num_nodes + (observed_events.dst[observed] - batch_id * num_nodes)
        chosen = sample_missing_pair_ids(observed_ids, count)
        if chosen.numel():
            neg_src_parts.append(chosen // num_nodes + batch_id * num_nodes)
            neg_dst_parts.append(chosen % num_nodes + batch_id * num_nodes)
            neg_batch_parts.append(torch.full_like(chosen, batch_id))
    if not neg_src_parts:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return (empty, empty, empty) if return_batch else (empty, empty)
    result = (
        cast(torch.LongTensor, torch.cat(neg_src_parts)),
        cast(torch.LongTensor, torch.cat(neg_dst_parts)),
        cast(torch.LongTensor, torch.cat(neg_batch_parts)),
    )
    return result if return_batch else result[:2]


@torch.no_grad()
def balanced_event_detection_metrics(
    model,
    state,
    events: EventBatch,
    num_nodes: int,
) -> Dict[str, float]:
    """Score observed events against an equal number of random inactive pairs.

    This complements conditional MRR.  It asks whether a pair is active in the
    current bin, without supplying the source node as an oracle query.  The
    metric is evaluation-only and does not alter the sampled-softmax training
    objective.
    """
    query_and_labels = balanced_event_detection_query(events, num_nodes)
    if query_and_labels is None:
        return {}
    query, labels = query_and_labels
    scores = model.score(state.clone(detach=True) if model.training else state, query)
    return {
        f"event_{key}": value
        for key, value in binary_metrics_from_logits(scores, labels).items()
    }


def balanced_event_detection_query(
    events: EventBatch,
    num_nodes: int,
) -> tuple[EventBatch, torch.Tensor] | None:
    """Build a 1:1 positive/random-inactive query set for reuse in evaluation."""
    positives = internal_events(events)
    if positives.num_events == 0:
        return None
    neg_src, neg_dst, neg_batch = sample_balanced_inactive_pairs(
        num_nodes=num_nodes,
        observed_events=events,
        num_samples=positives.num_events,
        device=events.src.device,
        return_batch=True,
    )
    if neg_src.numel() == 0:
        return None
    neg_t = None
    if positives.t is not None:
        neg_t = positives.t[:1].expand(neg_src.numel())
    query = EventBatch(
        src=torch.cat([positives.src, neg_src]),
        dst=torch.cat([positives.dst, neg_dst]),
        t=None if positives.t is None else torch.cat([positives.t, neg_t]),
        batch=None if positives.batch is None else torch.cat([positives.batch, neg_batch]),
    )
    labels = torch.cat([
        torch.ones((positives.num_events,), device=events.src.device),
        torch.zeros((neg_src.numel(),), device=events.src.device),
    ])
    return query, labels


def _select_event_rows(events: EventBatch, keep: torch.Tensor) -> EventBatch:
    """Index an event batch while preserving per-event metadata."""
    return EventBatch(
        src=events.src[keep],
        dst=events.dst[keep],
        features=None if events.features is None else events.features[keep],
        t=None if events.t is None else events.t[keep],
        episode=None if events.episode is None else events.episode[keep],
        is_external=None if events.is_external is None else events.is_external[keep],
        batch=None if events.batch is None else events.batch[keep],
    )



def build_candidate_eventbatch(
    src: torch.LongTensor,
    candidates_dst: torch.LongTensor,
    t: torch.LongTensor | None = None,
    features: torch.Tensor | None = None,
    is_external: torch.BoolTensor | None = None,
    batch: torch.LongTensor | None = None,
) -> EventBatch:
    """
    Flattens (M, K+1) candidate dsts into an EventBatch of size M*(K+1).
    """
    M, K1 = candidates_dst.shape

    src_rep = cast(torch.LongTensor, src.view(M, 1).expand(M, K1).reshape(-1))
    dst_flat = cast(torch.LongTensor, candidates_dst.reshape(-1))

    if t is not None:
        t_rep = cast(torch.LongTensor, t.view(M, 1).expand(M, K1).reshape(-1))
    else:
        t_rep = None

    if features is not None:
        # If features are per-positive-event (M, d), repeat them per candidate.
        feat_rep = features.unsqueeze(1).expand(M, K1, features.size(-1)).reshape(-1, features.size(-1))
    else:
        feat_rep = None

    if is_external is not None:
        external_rep = cast(torch.BoolTensor, is_external.view(M, 1).expand(M, K1).reshape(-1))
    else:
        external_rep = None

    batch_rep = None if batch is None else cast(torch.LongTensor, batch.view(M, 1).expand(M, K1).reshape(-1))
    return EventBatch(src=src_rep, dst=dst_flat, t=t_rep, features=feat_rep, is_external=external_rep, batch=batch_rep)


def bce_ranking_loss(
    scores: torch.Tensor,
    M: int,
    K1: int,
) -> torch.Tensor:
    """
    scores: (M*K1,) flattened
    Labels: candidate 0 is positive, rest negative.
    """
    logits = scores.view(M, K1)
    labels = torch.zeros((M, K1), device=scores.device)
    labels[:, 0] = 1.0
    return F.binary_cross_entropy_with_logits(logits, labels)


def softmax_ranking_loss(scores: torch.Tensor, M: int, K1: int) -> torch.Tensor:
    logits = scores.view(M, K1)
    labels = torch.zeros((M,), device=scores.device, dtype=torch.long)  # pos at index 0
    return F.cross_entropy(logits, labels, reduction="mean")


@torch.no_grad()
def ranking_metrics(
    scores: torch.Tensor,
    M: int,
    K1: int,
    hits_ks=(1, 3, 10),
    *,
    include_binary_metrics: bool = True,
) -> Dict[str, float]:
    """
    Compute sampled ranking diagnostics.

    Candidate 0 is the observed positive; candidates 1: are sampled negatives.
    Default ranking is pessimistic with respect to ties so tied positives do not
    receive artificially inflated MRR / Hits@K.
    """
    logits = scores.view(M, K1)
    labels = torch.zeros((M, K1), device=scores.device)
    labels[:, 0] = 1.0

    pos = logits[:, 0:1]
    neg = logits[:, 1:]

    rank_optimistic = 1 + (logits > pos).sum(dim=1)  # (M,)
    rank = 1 + (neg >= pos).sum(dim=1)  # pessimistic: ties count against positive

    mrr = (1.0 / rank.float()).mean().item()
    mrr_optimistic = (1.0 / rank_optimistic.float()).mean().item()

    max_neg = neg.max(dim=1).values if neg.numel() > 0 else torch.full_like(pos.squeeze(1), float("-inf"))
    margin = pos.squeeze(1) - max_neg
    probs = torch.softmax(logits, dim=1)
    pos_prob = probs[:, 0]
    top_prob = probs.max(dim=1).values

    pairwise_gt = (pos > neg).float() if neg.numel() > 0 else torch.ones((M, 0), device=scores.device)
    pairwise_eq = (pos == neg).float() if neg.numel() > 0 else torch.zeros((M, 0), device=scores.device)
    if neg.numel() > 0:
        pairwise_acc = pairwise_gt.mean().item()
        pairwise_auc_tie_half = (pairwise_gt + 0.5 * pairwise_eq).mean().item()
        tie_rate = pairwise_eq.mean().item()
        neg_mean = neg.mean().item()
        neg_std = neg.std(unbiased=False).item()
    else:
        pairwise_acc = 1.0
        pairwise_auc_tie_half = 1.0
        tie_rate = 0.0
        neg_mean = float("nan")
        neg_std = float("nan")

    out = {
        "mrr": mrr,
        "mrr_optimistic": mrr_optimistic,
        "mean_rank": rank.float().mean().item(),
        "mean_rank_optimistic": rank_optimistic.float().mean().item(),
        "median_rank": rank.float().median().item(),
        "top1_acc": (rank == 1).float().mean().item(),
        "top1_acc_optimistic": (rank_optimistic == 1).float().mean().item(),
        "pairwise_acc": pairwise_acc,
        "pairwise_auc_tie_half": pairwise_auc_tie_half,
        "tie_rate": tie_rate,
        "margin_mean": margin.mean().item(),
        "margin_median": margin.median().item(),
        "pos_score_mean": pos.mean().item(),
        "pos_score_std": pos.std(unbiased=False).item(),
        "neg_score_mean": neg_mean,
        "neg_score_std": neg_std,
        "pos_prob_mean": pos_prob.mean().item(),
        "top_prob_mean": top_prob.mean().item(),
        "bce_loss": F.binary_cross_entropy_with_logits(logits, labels).item(),
        "softmax_loss": F.cross_entropy(
            logits,
            torch.zeros((M,), device=scores.device, dtype=torch.long),
            reduction="mean",
        ).item(),
    }
    if include_binary_metrics:
        out.update(binary_metrics_from_logits(logits=logits, labels=labels))

    for k in hits_ks:
        out[f"hits@{k}"] = (rank <= k).float().mean().item()
        out[f"hits_optimistic@{k}"] = (rank_optimistic <= k).float().mean().item()
    return out


def mrr_and_hits(
    scores: torch.Tensor,
    M: int,
    K1: int,
    hits_ks=(1, 3, 10),
) -> Dict[str, float]:
    return ranking_metrics(scores=scores, M=M, K1=K1, hits_ks=hits_ks)


def event_force_loss_and_metrics(
    model,
    predicted_force: torch.Tensor,
    target_force: torch.Tensor,
    *,
    compute_metrics: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Calibrated physical-force objective shared by one-step and rollouts."""
    target_force = target_force.to(device=predicted_force.device, dtype=predicted_force.dtype)
    raw_error = predicted_force - target_force
    raw_mse = raw_error.square().mean()
    target_std = getattr(model, "event_feature_target_std", None)
    if target_std is None:
        target_std = torch.ones(target_force.shape[-1], device=target_force.device, dtype=target_force.dtype)
    else:
        target_std = target_std.to(device=target_force.device, dtype=target_force.dtype).clamp_min(1e-8)
    normalized_per_event_mse = (raw_error / target_std).square().mean(dim=-1)
    force_magnitude = target_force.norm(dim=-1)
    q90 = torch.as_tensor(
        getattr(model, "event_feature_magnitude_q90", 1.0),
        device=predicted_force.device,
        dtype=predicted_force.dtype,
    )
    magnitude_weight = float(getattr(model, "event_feature_magnitude_weight", 2.0))
    weights = 1.0 + magnitude_weight * (force_magnitude / q90.clamp_min(1e-8)).clamp(0.0, 3.0)
    weighted_loss = (weights * normalized_per_event_mse).mean()
    if not compute_metrics:
        return weighted_loss, {}
    metrics = {
        "force_mse": float(raw_mse.detach().item()),
        "force_nrmse": float(normalized_per_event_mse.mean().sqrt().detach().item()),
        "force_weighted_nrmse": float(weighted_loss.sqrt().detach().item()),
    }
    active_threshold = getattr(model, "event_feature_active_threshold", 0.0)
    active_threshold = float(active_threshold.detach().item()) if isinstance(active_threshold, torch.Tensor) else float(active_threshold)
    active = force_magnitude >= active_threshold
    if bool(active.any()):
        metrics["active_force_mse"] = float(raw_error[active].square().mean().detach().item())
    return weighted_loss, metrics


def internal_events(events: EventBatch) -> EventBatch:
    """Return events whose targets are endogenous physical interactions.

    External raindrops are observed interventions.  They update model state,
    but their random node and amplitude are not a prediction target.
    """
    if events.is_external is None:
        return events
    keep = ~events.is_external.to(dtype=torch.bool)
    if bool(keep.all()):
        return events
    return EventBatch(
        src=events.src[keep],
        dst=events.dst[keep],
        features=None if events.features is None else events.features[keep],
        t=None if events.t is None else events.t[keep],
        episode=None if events.episode is None else events.episode[keep],
        is_external=events.is_external[keep],
        batch=None if events.batch is None else events.batch[keep],
    )


def ranking_loss_and_metrics(
    model,
    state,
    next_events: EventBatch,
    num_nodes: int,
    num_neg: int,
    *,
    include_force: bool = True,
    include_binary_metrics: bool = True,
    include_event_detection: bool = True,
    compute_metrics: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Given current state, evaluate next_events as positives with negatives.
    """
    next_events = internal_events(next_events)
    device = next_events.src.device
    M = next_events.num_events
    if M == 0:
        # Keep the tensor connected to the current graph without inventing a
        # target for a bin containing only observed interventions.
        return state.node.sum() * 0.0, {}
    if next_events.t is not None and next_events.t.numel() != M:
        raise RuntimeError(
            f"Malformed event batch after internal-event filtering: M={M}, t={next_events.t.numel()}, "
            f"features={None if next_events.features is None else next_events.features.size(0)}, "
            f"external={None if next_events.is_external is None else next_events.is_external.numel()}, "
            f"batch={None if next_events.batch is None else next_events.batch.numel()}"
        )
    K1 = num_neg + 1

    neg_dst = sample_negative_dsts(
        num_nodes, next_events.dst, num_neg, device=device, batch=next_events.batch
    )
    candidates_dst = cast(torch.LongTensor, torch.cat([next_events.dst.view(M, 1), neg_dst], dim=1))  # (M, K1)

    cand_batch = build_candidate_eventbatch(
        src=next_events.src,
        candidates_dst=candidates_dst,
        t=next_events.t,
        features=next_events.features,
        is_external=next_events.is_external,
        batch=next_events.batch,
    )

    assert cand_batch.src.numel() == M * K1
    assert cand_batch.dst.numel() == M * K1

    # check that positives are actually in column 0
    dst_mat = candidates_dst  # (M, K1)
    pos = next_events.dst.view(M, 1)
    assert torch.equal(dst_mat[:, :1], pos), "positive dst not in col 0"

    # check negatives are not identical to positives (at least sometimes)
    # same = (dst_mat[:, 1:] == dst_mat[:, :1]).any(dim=1).float().mean().item()
    # print("DEBUG frac rows with a neg equal to pos:", same)

    # check candidates vary within a row
    # k = min(M, 50)
    # uniq_counts_t = torch.tensor(
    #     [torch.unique(dst_mat[i]).numel() for i in range(k)],
    #     device=dst_mat.device,
    #     dtype=torch.float32,
    # )
    # uniq_per_row = float(uniq_counts_t.mean().item())
    # print("DEBUG mean unique candidates (first 50 rows):", uniq_per_row)    


    # The full state clone and equality checks below were invaluable while
    # wiring models, but in no-grad evaluation they caused a device-to-host
    # synchronization and copied every node state for every time bin.  Built
    # models score functionally; use the live state in the production eval
    # path and retain the defensive checks during training.
    if model.training:
        before = state.node.detach().clone()
        state_eval = state.clone(detach=False)
    else:
        before = None
        state_eval = state
    scores = model.score(state_eval, cand_batch)  # (M*K1,)

    if before is not None:
        assert torch.equal(before, state.node.detach()), "score() mutated state.node"
    assert scores.shape == (M * K1,), f"scores has shape {tuple(scores.shape)} expected {(M*K1,)}"

    if before is not None and not torch.allclose(before, state.node):
        raise RuntimeError("score() mutated state.node")

    if not torch.isfinite(scores).all():
        raise RuntimeError("Non-finite scores from model.score()")

    if getattr(state, "aux", None) is not None and "L_bin_t_min" in state.aux and "L_bin_t_max" in state.aux:
        assert state.aux["L_bin_t_min"] == state.aux["L_bin_t_max"], \
            f"Expected equal L_bin_t_min/max, got {state.aux['L_bin_t_min']} vs {state.aux['L_bin_t_max']}"
    
    
    #loss = bce_ranking_loss(scores, M=M, K1=K1)  
    loss = softmax_ranking_loss(scores, M=M, K1=K1)

    metrics = (
        ranking_metrics(scores.detach(), M=M, K1=K1, include_binary_metrics=include_binary_metrics)
        if compute_metrics
        else {}
    )
    # Training retains the original sampled objective. During evaluation, add
    # the standard filtered ranking diagnostics: destinations that are another
    # true same-bin interaction from this source are excluded from negatives.
    if not model.training and compute_metrics:
        filtered_neg_dst, valid_rows = sample_filtered_negative_dsts(
            num_nodes=num_nodes,
            src=next_events.src,
            dst=next_events.dst,
            num_neg=num_neg,
            device=device,
            batch=next_events.batch,
        )
        if bool(valid_rows.any()):
            filtered_events = _select_event_rows(next_events, valid_rows)
            filtered_candidates = cast(torch.LongTensor, torch.cat([
                filtered_events.dst.view(-1, 1), filtered_neg_dst[valid_rows],
            ], dim=1))
            filtered_batch = build_candidate_eventbatch(
                src=filtered_events.src,
                candidates_dst=filtered_candidates,
                t=filtered_events.t,
                features=filtered_events.features,
                is_external=filtered_events.is_external,
                batch=filtered_events.batch,
            )
            filtered_scores = model.score(
                state_eval.clone(detach=True) if model.training else state_eval,
                filtered_batch,
            )
            filtered = ranking_metrics(
                filtered_scores.detach(),
                M=filtered_events.num_events,
                K1=num_neg + 1,
                include_binary_metrics=include_binary_metrics,
            )
            metrics.update({f"filtered_{key}": value for key, value in filtered.items()})
        # This is deliberately separate from MRR: it does not condition on
        # the source of a known positive event.  It uses a 1:1 random-pair
        # positive/negative set to diagnose event detection.
        if include_event_detection:
            metrics.update(
                balanced_event_detection_metrics(
                    model=model,
                    state=state_eval,
                    events=next_events,
                    num_nodes=num_nodes,
                )
            )
    # Event-only physical models additionally predict the measured force on
    # the *positive* next interaction.  Generic ranking baselines remain
    # unchanged because they do not expose this optional decoder.
    force_predictor = getattr(model, "predict_event_features", None)
    if include_force and callable(force_predictor) and next_events.features is not None:
        predicted_force = force_predictor(state_eval, next_events)
        force_loss, force_metrics = event_force_loss_and_metrics(
            model, predicted_force, next_events.features, compute_metrics=compute_metrics
        )
        loss = loss + float(getattr(model, "event_feature_loss_weight", 1.0)) * force_loss
        metrics.update(force_metrics)
    return loss, metrics
