from __future__ import annotations

import torch

from interactiondynamics.eval.ranking_metrics import sample_filtered_negative_dsts


def test_filtered_negative_destinations_exclude_all_same_source_positives() -> None:
    src = torch.tensor([0, 0, 1, 2])
    dst = torch.tensor([1, 3, 2, 0])

    negatives, valid = sample_filtered_negative_dsts(
        num_nodes=5,
        src=src,
        dst=dst,
        num_neg=32,
        device=torch.device("cpu"),
    )

    assert bool(valid.all())
    # Both 0 -> 1 and 0 -> 3 are true interactions. Neither may appear as a
    # negative for either source-0 positive.
    assert not bool(torch.isin(negatives[:2], torch.tensor([1, 3])).any())
    assert not bool(torch.isin(negatives[2], torch.tensor([2])).any())
    assert not bool(torch.isin(negatives[3], torch.tensor([0])).any())


def test_filtered_negative_destinations_mark_fully_active_source_invalid() -> None:
    src = torch.tensor([0, 0, 0])
    dst = torch.tensor([0, 1, 2])

    _negatives, valid = sample_filtered_negative_dsts(
        num_nodes=3,
        src=src,
        dst=dst,
        num_neg=4,
        device=torch.device("cpu"),
    )

    assert not bool(valid.any())
