# eval/evaluate.py

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Dict, Iterable, Optional
import torch

from interactiondynamics.core.events import EventBatch
from interactiondynamics.eval.ranking_metrics import ranking_loss_and_metrics


@dataclass
class EvalSlices:
    early_steps: int = 10   # first N scored steps
    # late = everything after early_steps


def _acc_init() -> dict:
    return {"loss_sum": 0.0, "metric_sums": {}, "metric_counts": {}, "n": 0.0}


def _acc_update(acc: dict, loss: float, metrics: Dict[str, float]) -> None:
    acc["loss_sum"] += float(loss)
    for key, value in metrics.items():
        value_f = float(value)
        if math.isnan(value_f):
            continue
        acc["metric_sums"][key] = acc["metric_sums"].get(key, 0.0) + value_f
        acc["metric_counts"][key] = acc["metric_counts"].get(key, 0.0) + 1.0
    acc["n"] += 1.0


def _acc_finalize(acc: dict) -> Dict[str, float]:
    if acc["n"] <= 0:
        return {"loss": 0.0, "steps": 0}
    n = acc["n"]
    out = {"loss": acc["loss_sum"] / n, "steps": int(n)}
    for key, value in acc["metric_sums"].items():
        out[key] = value / acc["metric_counts"][key]
    return out


@torch.no_grad()
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

    state = model.init_state(batch_size=1, num_nodes=cfg.num_nodes, device=device)

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
        _acc_update(overall, loss_val, metrics)
        if scored_step < slices.early_steps:
            _acc_update(early, loss_val, metrics)
        else:
            _acc_update(late, loss_val, metrics)

        scored_step += 1
        prev = events

    out: Dict[str, float] = {}
    o = _acc_finalize(overall)
    e = _acc_finalize(early)
    l = _acc_finalize(late)

    for key, value in o.items():
        out[key] = float(value)

    for prefix, block in (("early", e), ("late", l)):
        for key, value in block.items():
            out[f"{prefix}_{key}"] = float(value)

    return out
