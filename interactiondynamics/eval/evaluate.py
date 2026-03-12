# eval/evaluate.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple
import torch

from core.events import EventBatch
from eval.ranking import ranking_loss_and_metrics


@dataclass
class EvalSlices: # is behavior different earlier in the stream or later when its seen more history... the warming up stuff
    early_steps: int = 10   # first N scored steps
    # late = everything after early_steps


def _acc_init() -> Dict[str, float]:
    return {"loss_sum": 0.0, "mrr_sum": 0.0, "n": 0.0}


def _acc_update(acc: Dict[str, float], loss: float, mrr: float) -> None:
    acc["loss_sum"] += float(loss)
    acc["mrr_sum"] += float(mrr)
    acc["n"] += 1.0


def _acc_finalize(acc: Dict[str, float]) -> Dict[str, float]:
    if acc["n"] <= 0:
        return {"loss": 0.0, "mrr": 0.0, "steps": 0}
    n = acc["n"]
    return {"loss": acc["loss_sum"] / n, "mrr": acc["mrr_sum"] / n, "steps": int(n)}


@torch.no_grad() # do eval without tracking gradients - eval is cheaper/faster
def evaluate_stream_sliced( 
    model,
    bins: Iterable[EventBatch],
    cfg,
    *,
    slices: EvalSlices = EvalSlices(),
) -> Dict[str, float]:
    """
    Evaluate ranking metrics over a binned stream, returning:
      - overall loss/mrr
      - early (first N scored steps)
      - late (remaining scored steps)

    No warmup. State is initialized fresh.
    """
    model.eval()
    device = torch.device(cfg.device)

    state = model.init_state(batch_size=1, num_nodes=cfg.num_nodes, device=device) # clean state at start of eval.. eval doesnt continue from training

    overall = _acc_init()
    early = _acc_init()
    late = _acc_init()

    prev: Optional[EventBatch] = None
    scored_step = 0  # counts only steps where we actually compute loss (prev exists)

    for events in bins:
        events = events.to(device)

        if prev is None:
            prev = events
            continue

        # predict current from state(after consuming prev)
        state, _ = model.step(state, prev)

        # this file just calls the ranking evaluater 
        loss_t, metrics = ranking_loss_and_metrics( 
            model=model,
            state=state,
            next_events=events,
            num_nodes=cfg.num_nodes,
            num_neg=cfg.num_neg,
        )

        if state is not None:
            state.detach_()

        loss_val = float(loss_t.item())
        mrr_val = float(metrics["mrr"])

        _acc_update(overall, loss_val, mrr_val)
        if scored_step < slices.early_steps:
            _acc_update(early, loss_val, mrr_val)
        else:
            _acc_update(late, loss_val, mrr_val)

        scored_step += 1
        prev = events

    out: Dict[str, float] = {}
    o = _acc_finalize(overall)
    e = _acc_finalize(early)
    l = _acc_finalize(late)

    # overall
    out["loss"] = o["loss"]
    out["mrr"] = o["mrr"]
    out["steps"] = float(o["steps"])

    # early
    out["early_loss"] = e["loss"]
    out["early_mrr"] = e["mrr"]
    out["early_steps"] = float(e["steps"])

    # late
    out["late_loss"] = l["loss"]
    out["late_mrr"] = l["mrr"]
    out["late_steps"] = float(l["steps"])

    return out
