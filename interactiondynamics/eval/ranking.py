from __future__ import annotations
from asyncio import events
from dataclasses import dataclass
from typing import Dict, Tuple, cast

import torch
import torch.nn.functional as F

from core.events import EventBatch

# containerrrr 
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

# samples random negative destination nodes

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

    # randomly pick num_neg node IDs, avoid picking true dest, also maybe avoid source node if you want
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


# shape-handling helper: m pos events each with k+1 candidate destinations...
# this function flatters that into one big EventBatch of size m*(k+1) so model can score them all at once
# BATCHING MECHANICSSS
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

    #do we need to fix wave to output None for no features instead of empty tensor??
    if features is not None and features.dim() == 2 and features.size(-1) > 0:
        # If features are per-positive-event (M, d), repeat them per candidate.
        d = features.size(-1)
        feat_rep = features.unsqueeze(1).expand(M, K1, d).reshape(M * K1, d)
    else:
        feat_rep = None

    return EventBatch(src=src_rep, dst=dst_flat, t=t_rep, features=feat_rep)

# defines one possible ranking loss -> sigmoid/binary-cross-entropy style on each candidate
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

# main loss actually being used.. reshapes scores to (M,K1)... m row per pos event and num_neg+1 columns = one pos+neg
#then cross entropy with label 0 so candidate at index 0 is the true destination
# aka... for each row, make column 0 score the highest
# out of this set of possible dest nodes, put the real one on top
def softmax_ranking_loss(scores: torch.Tensor, M: int, K1: int) -> torch.Tensor:
    logits = scores.view(M, K1)
    labels = torch.zeros((M,), device=scores.device, dtype=torch.long)  # pos at index 0
    return F.cross_entropy(logits, labels, reduction="mean")

# mean reciprocal rank
# higher MRR = better ranking for when pos is ranked
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
    pos = logits[:, :1]
    #fixed ranking bug -> pessimistic scoring, if negative ties the positive, count it against the positives
    rank = 1 + (logits[:, 1:] >= pos).sum(dim=1)  # (M,)

    mrr = (1.0 / rank.float()).mean().item()
    out = {"mrr": mrr, "mean_rank": rank.float().mean().item()}

    # S = scores.view(M, K1)
    # how many times is argmax at column 0?
    # top0 = (S.argmax(dim=1) == 0).float().mean().item()
    # print("DEBUG top0 frac:", top0, "scores[0]:", S[0].detach().cpu().tolist()[:min(10, K1)])

    for k in hits_ks:
        out[f"hits@{k}"] = (rank <= k).float().mean().item() # fraction of times the true dest is in the top k
    return out

# takes all those and returns loss and metrics
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
    M = next_events.num_events # figure out sizes, m = how many real pos events in this bin
    K1 = num_neg + 1 # number of candidates per event

    # first column = true destination, rest = negatives 
    neg_dst = sample_negative_dsts(num_nodes, next_events.dst, num_neg, device=device)
    candidates_dst = cast(torch.LongTensor, torch.cat([next_events.dst.view(M, 1), neg_dst], dim=1))  # (M, K1)

    # flatten candidate structure into one big EventBatch for scoring
    cand_batch = build_candidate_eventbatch(
        src=next_events.src,
        candidates_dst=candidates_dst,
        t=next_events.t,
        features=next_events.features,
        #features=None,
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

    # clone state before scoring so score can't accidentally mutate the real training set
    before = state.node.detach().clone()
    detach_for_score = not torch.is_grad_enabled()
    state_eval = state.clone(detach=detach_for_score)  # ensure score() can't mutate original state
        

    scores = model.score(state_eval, cand_batch)  # (M*K1,)

    # ---- DEBUG: inspect a few candidate rows manually ----
    # if not torch.is_grad_enabled():

    #     with torch.no_grad():
    #         scores_mat = scores.view(M, K1)   # each row = [positive, neg1, neg2, ...]
    #         ranks = (scores_mat >= scores_mat[:, :1]).sum(dim=1)  # 1 = best rank
    #         # if you want strict ranking instead:
    #         # ranks = 1 + (scores_mat[:, 1:] > scores_mat[:, :1]).sum(dim=1)

    #         n_show = min(5, M)
    #         print("\n===== DEBUG CANDIDATE ROWS =====")
    #         for i in range(n_show):
    #             src_i = int(next_events.src[i].item())
    #             pos_i = int(next_events.dst[i].item())
    #             cand_i = candidates_dst[i].tolist()
    #             score_i = [float(x) for x in scores_mat[i].detach().cpu()]
    #             rank_i = int(ranks[i].item())

    #             print(f"\nrow {i}")
    #             print(f"  src           : {src_i}")
    #             print(f"  positive dst  : {pos_i}")
    #             print(f"  candidates dst: {cand_i}")
    #             print(f"  scores        : {score_i}")
    #             print(f"  positive rank : {rank_i}")

    #             # optional: best candidate according to model
    #             best_col = int(torch.argmax(scores_mat[i]).item())
    #             best_dst = int(candidates_dst[i, best_col].item())
    #             print(f"  predicted dst : {best_dst} (col {best_col})")
    #         print("===== END DEBUG =====\n")
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
    loss = softmax_ranking_loss(scores, M=M, K1=K1) # for each source event, rank the true destination above sampled negatives

    metrics = mrr_and_hits(scores.detach(), M=M, K1=K1) # so training and eval are both centered on ranking quality
    return loss, metrics
