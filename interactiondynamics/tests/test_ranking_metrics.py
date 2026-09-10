from __future__ import annotations

import torch

from interactiondynamics.core.events import EventBatch
from interactiondynamics.data.interfaces import DataSpec
from interactiondynamics.eval.ranking_metrics import (
    sample_balanced_inactive_pairs,
    sample_filtered_negative_dsts,
    sample_negative_dsts,
)
from interactiondynamics.models.fnn import FieldNeuralNetwork


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


def test_balanced_inactive_pairs_exclude_every_observed_pair() -> None:
    events = EventBatch(
        src=torch.tensor([0, 1, 3]),
        dst=torch.tensor([1, 2, 0]),
    )
    src, dst = sample_balanced_inactive_pairs(
        num_nodes=4,
        observed_events=events,
        num_samples=3,
        device=torch.device("cpu"),
    )

    assert src.numel() == dst.numel() == 3
    observed_ids = events.src * 4 + events.dst
    assert not bool(torch.isin(src * 4 + dst, observed_ids).any())


def test_balanced_inactive_pairs_does_not_materialize_all_pairs() -> None:
    events = EventBatch(
        src=torch.tensor([0, 99_999]),
        dst=torch.tensor([1, 99_998]),
    )
    src, dst = sample_balanced_inactive_pairs(
        num_nodes=100_000,
        observed_events=events,
        num_samples=8,
        device=torch.device("cpu"),
    )

    sampled_ids = src * 100_000 + dst
    observed_ids = events.src * 100_000 + events.dst
    assert sampled_ids.numel() == 8
    assert sampled_ids.unique().numel() == 8
    assert not bool(torch.isin(sampled_ids, observed_ids).any())


def test_homogeneous_negative_destinations_use_full_id_space() -> None:
    negatives = sample_negative_dsts(
        num_nodes=11,
        pos_dst=torch.tensor([8, 9, 10]),
        num_neg=50,
        device=torch.device("cpu"),
    )

    assert torch.all(negatives >= 0)
    assert torch.all(negatives < 11)
    assert bool((negatives < 8).any())


def test_jodie_negative_destinations_stay_in_item_partition() -> None:
    dst_start = 8
    dst_end = 11
    positives = torch.tensor([8, 9, 10])

    negatives = sample_negative_dsts(
        num_nodes=11,
        pos_dst=positives,
        num_neg=100,
        device=torch.device("cpu"),
        dst_start=dst_start,
        dst_end=dst_end,
    )

    assert torch.all(negatives >= dst_start)
    assert torch.all(negatives < dst_end)
    assert not bool((negatives == positives.view(-1, 1)).any())


def test_jodie_negative_collision_wrap_stays_in_item_partition() -> None:
    # Destination 10 is the last item ID. Homogeneous wrapping would send
    # collisions to 0, which is a user. The bipartite wrap must stay in [8, 11).
    positives = torch.tensor([10, 10, 10])
    negatives = sample_negative_dsts(
        num_nodes=11,
        pos_dst=positives,
        num_neg=64,
        device=torch.device("cpu"),
        dst_start=8,
        dst_end=11,
    )

    assert torch.all(negatives >= 8)
    assert torch.all(negatives < 11)
    assert not bool((negatives == 10).any())


def test_filtered_negative_destinations_ignore_user_ids_on_bipartite_graphs() -> None:
    src = torch.tensor([0, 0, 0])
    dst = torch.tensor([8, 9, 10])

    negatives, valid = sample_filtered_negative_dsts(
        num_nodes=11,
        src=src,
        dst=dst,
        num_neg=8,
        device=torch.device("cpu"),
        dst_start=8,
        dst_end=11,
    )

    # Every item is an active destination for source 0, so there is no legal
    # negative even though most user IDs are inactive.
    assert not bool(valid.any())
    assert torch.all(negatives >= 8)
    assert torch.all(negatives < 11)


def test_balanced_inactive_pairs_sample_src_times_dst_on_bipartite_graphs() -> None:
    events = EventBatch(
        src=torch.tensor([0, 1]),
        dst=torch.tensor([8, 9]),
    )
    src, dst = sample_balanced_inactive_pairs(
        num_nodes=11,
        observed_events=events,
        num_samples=20,
        device=torch.device("cpu"),
        src_start=0,
        src_end=8,
        dst_start=8,
        dst_end=11,
    )

    assert src.numel() == dst.numel() == 20
    assert torch.all(src >= 0) and torch.all(src < 8)
    assert torch.all(dst >= 8) and torch.all(dst < 11)
    observed_ids = (events.src - 0) * 3 + (events.dst - 8)
    sampled_ids = (src - 0) * 3 + (dst - 8)
    assert not bool(torch.isin(sampled_ids, observed_ids).any())


def test_fnn_sparse_candidates_stay_in_item_partition() -> None:
    spec = DataSpec(
        name="jodie_tiny",
        num_nodes=11,
        event_dim=2,
        extra={
            "is_bipartite": True,
            "src_id_range": (0, 8),
            "dst_id_range": (8, 11),
        },
    )
    model = FieldNeuralNetwork(
        num_nodes=spec.num_nodes,
        force_dim=2,
        state_dim=2,
        gamma_init=0.1,
        omega_init=1.0,
        dt=0.1,
        topology_mode="observed_sparse",
    )
    observed_src = torch.tensor([0, 1, 2])
    observed_dst = torch.tensor([8, 9, 10])
    dst_start, dst_end = spec.destination_id_range()
    negatives = sample_negative_dsts(
        num_nodes=spec.num_nodes,
        pos_dst=observed_dst,
        num_neg=5,
        device=torch.device("cpu"),
        dst_start=dst_start,
        dst_end=dst_end,
    )
    model.set_sparse_topology_candidates(
        torch.cat([observed_src, observed_src.repeat_interleave(5)]),
        torch.cat([observed_dst, negatives.reshape(-1)]),
    )

    dst = model.sparse_candidate_keys % spec.num_nodes
    assert torch.all(dst >= dst_start)
    assert torch.all(dst < dst_end)
