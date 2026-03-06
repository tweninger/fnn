from __future__ import annotations
from asyncio import events
from dataclasses import dataclass
from typing import Dict, Tuple, cast

import torch
import torch.nn.functional as F

from core.events import EventBatch


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
) -> torch.Tensor:
    """
    Uniform negative sampling over node ids, avoiding collisions with pos_dst (and optionally avoid).
    Returns (M, num_neg) LongTensor on `device`.
    """
    M = pos_dst.numel()
    if num_nodes <= 1:
        return torch.zeros((M, num_neg), device=device, dtype=torch.long)

    neg = torch.randint(0, num_nodes, (M, num_neg), device=device, dtype=torch.long)

    # Fix collisions with pos_dst (one pass is usually fine)
    collide = neg.eq(pos_dst.view(-1, 1))
    if collide.any():
        neg[collide] = (neg[collide] + 1) % num_nodes

    # Optional: avoid another id (e.g., src for no self-loop negatives)
    if avoid is not None:
        collide2 = neg.eq(avoid.view(-1, 1))
        if collide2.any():
            neg[collide2] = (neg[collide2] + 1) % num_nodes
            # might re-collide with pos_dst in rare cases; acceptable for now

    return neg



def build_candidate_eventbatch(
    src: torch.LongTensor,
    candidates_dst: torch.LongTensor,
    t: torch.LongTensor | None = None,
    features: torch.Tensor | None = None,
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

    return EventBatch(src=src_rep, dst=dst_flat, t=t_rep, features=feat_rep)


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
def mrr_and_hits(
    scores: torch.Tensor,
    M: int,
    K1: int,
    hits_ks=(1, 3, 10),
) -> Dict[str, float]:
    """
    Compute rank of the positive (index 0) among K1 candidates for each event.
    """
    logits = scores.view(M, K1)
    # Higher is better. Rank = 1 + number of candidates strictly greater than positive.
    pos = logits[:, 0:1]
    rank = 1 + (logits > pos).sum(dim=1)  # (M,)

    mrr = (1.0 / rank.float()).mean().item()
    out = {"mrr": mrr, "mean_rank": rank.float().mean().item()}

    # S = scores.view(M, K1)
    # how many times is argmax at column 0?
    # top0 = (S.argmax(dim=1) == 0).float().mean().item()
    # print("DEBUG top0 frac:", top0, "scores[0]:", S[0].detach().cpu().tolist()[:min(10, K1)])

    for k in hits_ks:
        out[f"hits@{k}"] = (rank <= k).float().mean().item()
    return out


def ranking_loss_and_metrics(
    model,
    state,
    next_events: EventBatch,
    num_nodes: int,
    num_neg: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Given current state, evaluate next_events as positives with negatives.
    """
    device = next_events.src.device
    M = next_events.num_events
    K1 = num_neg + 1

    neg_dst = sample_negative_dsts(num_nodes, next_events.dst, num_neg, device=device)
    candidates_dst = cast(torch.LongTensor, torch.cat([next_events.dst.view(M, 1), neg_dst], dim=1))  # (M, K1)

    cand_batch = build_candidate_eventbatch(
        src=next_events.src,
        candidates_dst=candidates_dst,
        t=next_events.t,
        features=next_events.features,
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


    before = state.node.detach().clone()
    detach_for_score = not torch.is_grad_enabled()
    state_eval = state.clone(detach=detach_for_score)  # ensure score() can't mutate original state
        

    scores = model.score(state_eval, cand_batch)  # (M*K1,)


    assert torch.equal(before, state.node.detach()), "score() mutated state.node"
    
    assert scores.shape == (M * K1,), f"scores has shape {tuple(scores.shape)} expected {(M*K1,)}"

    if not torch.allclose(before, state.node):
        raise RuntimeError("score() mutated state.node")

    if not torch.isfinite(scores).all():
        raise RuntimeError("Non-finite scores from model.score()")

    if getattr(state, "aux", None) is not None and "L_bin_t_min" in state.aux and "L_bin_t_max" in state.aux:
        assert state.aux["L_bin_t_min"] == state.aux["L_bin_t_max"], \
            f"Expected equal L_bin_t_min/max, got {state.aux['L_bin_t_min']} vs {state.aux['L_bin_t_max']}"
    
    
    #loss = bce_ranking_loss(scores, M=M, K1=K1)  
    loss = softmax_ranking_loss(scores, M=M, K1=K1)

    metrics = mrr_and_hits(scores.detach(), M=M, K1=K1)
    return loss, metrics
