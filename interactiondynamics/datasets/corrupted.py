from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping, Optional

import torch

from core.events import EventBatch
from datasets.interfaces import DataSpec, EventStreamDataset


"""
for fake pos, add safeties for:
clean edges
self loops
duplicate fake edges
"""

@dataclass
class CorruptedBatchInfo:
    clean: EventBatch
    observed: EventBatch
    removed: EventBatch
    fake: EventBatch
    keep_mask: torch.Tensor  # shape [M] over clean events


def _empty_like(batch: EventBatch) -> EventBatch:
    src = batch.src[:0].clone()
    dst = batch.dst[:0].clone()

    if batch.t is not None:
        t = batch.t[:0].clone()
    else:
        t = None

    if batch.features is not None:
        features = batch.features[:0].clone()
    else:
        features = None

    return EventBatch(
        src=src,
        dst=dst,
        t=t,
        features=features,
        node_targets=batch.node_targets,
        node_mask=batch.node_mask,
    )

def _undirected_pair_keep_mask(
    src: torch.Tensor,
    dst: torch.Tensor,
    *,
    num_nodes: int,
    drop_real_prob: float,
    generator: torch.Generator,
    device: torch.device,
    min_keep_per_nonempty_bin: int = 1,
) -> torch.Tensor:
    """
    Return an event-level keep_mask, but sample keep/drop decisions
    at the undirected-pair level.

    So if both (i -> j) and (j -> i) exist, they share the same keep/drop decision.
    """
    M = int(src.numel())
    if M == 0:
        return torch.zeros((0,), dtype=torch.bool, device=device)

    lo = torch.minimum(src.long(), dst.long())
    hi = torch.maximum(src.long(), dst.long())

    # one unique key per undirected pair
    pair_keys = lo * int(num_nodes) + hi

    unique_pairs, inverse = torch.unique(
        pair_keys,
        sorted=False,
        return_inverse=True,
    )

    P = int(unique_pairs.numel())

    pair_keep = torch.ones((P,), dtype=torch.bool, device=device)

    if drop_real_prob > 0.0:
        pair_keep = torch.rand(P, generator=generator, device=device) >= drop_real_prob

        # Keep at least some physical pairs if the bin is nonempty.
        if min_keep_per_nonempty_bin > 0:
            min_keep = min(int(min_keep_per_nonempty_bin), P)
            if int(pair_keep.sum().item()) < min_keep:
                perm = torch.randperm(P, generator=generator, device=device)
                pair_keep[:] = False
                pair_keep[perm[:min_keep]] = True

    # Convert pair-level decisions back to event-level decisions.
    keep_mask = pair_keep[inverse]
    return keep_mask


def _active_nodes_in_batch(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Unique node ids that appear as src or dst in this bin."""
    return torch.unique(torch.cat([src.long(), dst.long()], dim=0))


def _node_block_keep_mask(
    src: torch.Tensor,
    dst: torch.Tensor,
    *,
    drop_real_prob: float,
    block_node_select: str,
    generator: torch.Generator,
    device: torch.device,
    min_keep_per_nonempty_bin: int = 1,
) -> torch.Tensor:
    """
    Drop every event incident to a chosen set of nodes (node block).

    drop_real_prob is the fraction of *active nodes in this bin* to block
    (not per-edge probability). An event is removed if src or dst is blocked.

    block_node_select:
      - "random": uniform random active nodes
      - "high_degree": active nodes with largest incident edge count in the bin
    """
    M = int(src.numel())
    if M == 0:
        return torch.zeros((0,), dtype=torch.bool, device=device)

    keep_mask = torch.ones(M, dtype=torch.bool, device=device)
    if drop_real_prob <= 0.0:
        return keep_mask

    if block_node_select not in {"random", "high_degree"}:
        raise ValueError(
            f"block_node_select must be 'random' or 'high_degree', got {block_node_select!r}"
        )

    src_l = src.long()
    dst_l = dst.long()
    active = _active_nodes_in_batch(src_l, dst_l)
    n_active = int(active.numel())
    if n_active == 0:
        return keep_mask

    # Block this many nodes; leave at least one active node unblocked when possible.
    n_block = int(round(float(drop_real_prob) * n_active))
    n_block = max(0, min(n_block, max(0, n_active - 1)))

    if n_block > 0:
        if block_node_select == "random":
            perm = torch.randperm(n_active, generator=generator, device=device)
            blocked = active[perm[:n_block]]
        else:
            # Incident edge count per active node (as src or dst).
            deg = torch.zeros((n_active,), dtype=torch.long, device=device)
            for i, node in enumerate(active):
                deg[i] = int((src_l == node).sum().item() + (dst_l == node).sum().item())
            # Highest degree first; tie_key breaks ties deterministically per bin.
            tie_key = torch.rand(n_active, generator=generator, device=device)
            scores = deg.float() + tie_key * 1e-4
            order = torch.argsort(scores, descending=True)
            blocked = active[order[:n_block]]

        blocked_set = blocked
        is_blocked_src = torch.isin(src_l, blocked_set)
        is_blocked_dst = torch.isin(dst_l, blocked_set)
        keep_mask = ~(is_blocked_src | is_blocked_dst)

    if min_keep_per_nonempty_bin > 0:
        min_keep = min(int(min_keep_per_nonempty_bin), M)
        if int(keep_mask.sum().item()) < min_keep:
            # Restore random dropped events until the bin keeps enough observations.
            dropped_idx = torch.nonzero(~keep_mask, as_tuple=False).view(-1)
            need = min_keep - int(keep_mask.sum().item())
            perm = torch.randperm(
                int(dropped_idx.numel()),
                generator=generator,
                device=device,
            )
            keep_mask[dropped_idx[perm[:need]]] = True

    return keep_mask


def _make_generator(*, split: str, batch_idx: int, seed: int, device: torch.device) -> torch.Generator:
    split_offset = {
        "train": 1_000_003,
        "val": 2_000_003,
        "test": 3_000_003,
    }.get(split, 9_000_003)

    full_seed = int(seed) + split_offset + 104_729 * int(batch_idx)

    gen_device = "cuda" if device.type == "cuda" else "cpu"
    gen = torch.Generator(device=gen_device)
    gen.manual_seed(full_seed)
    return gen


def corrupt_batch_with_metadata(
    batch: EventBatch,
    *,
    split: str,
    batch_idx: int,
    num_nodes: int,
    drop_real_prob: float = 0.0,
    add_fake_ratio: float = 0.0,
    seed: int = 0,
    fake_feature_mode: str = "zeros",   # "zeros" | "sample"
    avoid_self_loops: bool = True,
    min_keep_per_nonempty_bin: int = 1,
    corrupt_unit: str = "undirected_pair",  # directed_event | undirected_pair | node_block
    block_node_select: str = "random",  # random | high_degree (node_block only)
) -> CorruptedBatchInfo:
    """
    Deterministically corrupt one clean EventBatch and also return the hidden/removed positives.

    clean   : original batch
    observed: what the model sees
    removed : true positives removed from observation
    fake    : false positives added to observation
    """
    if not (0.0 <= drop_real_prob <= 1.0):
        raise ValueError("drop_real_prob must be in [0, 1]")
    if add_fake_ratio < 0.0:
        raise ValueError("add_fake_ratio must be >= 0")
    if fake_feature_mode not in {"zeros", "sample"}:
        raise ValueError("fake_feature_mode must be 'zeros' or 'sample'")
    _valid_units = {"directed_event", "undirected_pair", "node_block"}
    if corrupt_unit not in _valid_units:
        raise ValueError(
            f"corrupt_unit must be one of {sorted(_valid_units)}, got {corrupt_unit!r}"
        )

    device = batch.src.device
    gen = _make_generator(split=split, batch_idx=batch_idx, seed=seed, device=device)

    src = batch.src
    dst = batch.dst
    feats = batch.features
    t = batch.t

    M = int(src.numel())
    if M == 0:
        empty = _empty_like(batch)
        return CorruptedBatchInfo(
            clean=batch,
            observed=empty,
            removed=empty,
            fake=empty,
            keep_mask=torch.zeros((0,), dtype=torch.bool, device=device),
        )
    if drop_real_prob == 0.0 and add_fake_ratio == 0.0:
        empty = _empty_like(batch)
        return CorruptedBatchInfo(
            clean=batch,
            observed=batch,
            removed=empty,
            fake=empty,
            keep_mask=torch.ones((M,), dtype=torch.bool, device=device),
        )
    # -------------------------
    # 1) choose which true events remain observed
    # -------------------------

    if corrupt_unit == "directed_event":
        # Old behavior: each raw event row gets its own coin flip.
        keep_mask = torch.ones(M, dtype=torch.bool, device=device)

        if drop_real_prob > 0.0:
            keep_mask = torch.rand(M, generator=gen, device=device) >= drop_real_prob

            if (
                min_keep_per_nonempty_bin > 0
                and keep_mask.sum().item() < min(min_keep_per_nonempty_bin, M)
            ):
                perm = torch.randperm(M, generator=gen, device=device)
                keep_mask[:] = False
                keep_mask[perm[: min(min_keep_per_nonempty_bin, M)]] = True

    elif corrupt_unit == "undirected_pair":
        # (i -> j) and (j -> i) are kept/dropped together.
        keep_mask = _undirected_pair_keep_mask(
            src,
            dst,
            num_nodes=num_nodes,
            drop_real_prob=drop_real_prob,
            generator=gen,
            device=device,
            min_keep_per_nonempty_bin=min_keep_per_nonempty_bin,
        )

    elif corrupt_unit == "node_block":
        keep_mask = _node_block_keep_mask(
            src,
            dst,
            drop_real_prob=drop_real_prob,
            block_node_select=block_node_select,
            generator=gen,
            device=device,
            min_keep_per_nonempty_bin=min_keep_per_nonempty_bin,
        )

    else:
        raise ValueError(f"Unknown corrupt_unit={corrupt_unit!r}")

    removed_mask = ~keep_mask

    # These are required by the fake-edge and observed-batch code below.
    src_kept = src[keep_mask]
    dst_kept = dst[keep_mask]
    feats_kept = feats[keep_mask] if feats is not None else None
    t_kept = t[keep_mask] if t is not None else None

    src_removed = src[removed_mask]
    dst_removed = dst[removed_mask]
    feats_removed = feats[removed_mask] if feats is not None else None
    t_removed = t[removed_mask] if t is not None else None

    removed_batch = EventBatch(
        src=src_removed,
        dst=dst_removed,
        features=feats_removed,
        t=t_removed,
        node_targets=batch.node_targets,
        node_mask=batch.node_mask,
    )

    # -------------------------
    # 2) add fake positives
    # -------------------------
    M_kept = int(src_kept.numel())
    n_fake = int(round(M_kept * add_fake_ratio))

    if n_fake > 0:
        fake_src = torch.randint(
            low=0,
            high=num_nodes,
            size=(n_fake,),
            generator=gen,
            device=device,
            dtype=src.dtype,
        )
        fake_dst = torch.randint(
            low=0,
            high=num_nodes,
            size=(n_fake,),
            generator=gen,
            device=device,
            dtype=dst.dtype,
        )

        if avoid_self_loops:
            collide = fake_src == fake_dst
            if collide.any():
                fake_dst[collide] = (fake_dst[collide] + 1) % num_nodes

        if t is not None:
            if t.numel() == 0:
                fake_t = t[:0].clone()
            else:
                fake_t = torch.full((n_fake,), t[0], dtype=t.dtype, device=device)
        else:
            fake_t = None

        if feats is not None:
            d = feats.size(-1)
            if fake_feature_mode == "zeros":
                fake_feats = torch.zeros((n_fake, d), dtype=feats.dtype, device=device)
            else:  # "sample"
                if M_kept > 0:
                    idx = torch.randint(
                        low=0,
                        high=M_kept,
                        size=(n_fake,),
                        generator=gen,
                        device=device,
                        dtype=torch.long,
                    )
                    fake_feats = feats_kept[idx]
                else:
                    fake_feats = torch.zeros((n_fake, d), dtype=feats.dtype, device=device)
        else:
            fake_feats = None
    else:
        fake_src = src[:0].clone()
        fake_dst = dst[:0].clone()
        fake_t = t[:0].clone() if t is not None else None
        fake_feats = feats[:0].clone() if feats is not None else None

    fake_batch = EventBatch(
        src=fake_src,
        dst=fake_dst,
        features=fake_feats,
        t=fake_t,
        node_targets=batch.node_targets,
        node_mask=batch.node_mask,
    )

    # -------------------------
    # 3) build the observed batch = kept real + fake
    # -------------------------
    src_obs = torch.cat([src_kept, fake_src], dim=0)
    dst_obs = torch.cat([dst_kept, fake_dst], dim=0)

    if t_kept is not None and fake_t is not None:
        t_obs = torch.cat([t_kept, fake_t], dim=0)
    elif t_kept is not None:
        t_obs = t_kept
    else:
        t_obs = fake_t

    if feats_kept is not None and fake_feats is not None:
        feats_obs = torch.cat([feats_kept, fake_feats], dim=0)
    elif feats_kept is not None:
        feats_obs = feats_kept
    else:
        feats_obs = fake_feats

    if src_obs.numel() > 0:
        perm = torch.randperm(src_obs.numel(), generator=gen, device=device)
        src_obs = src_obs[perm]
        dst_obs = dst_obs[perm]
        if t_obs is not None:
            t_obs = t_obs[perm]
        if feats_obs is not None:
            feats_obs = feats_obs[perm]

    observed_batch = EventBatch(
        src=src_obs,
        dst=dst_obs,
        features=feats_obs,
        t=t_obs,
        node_targets=batch.node_targets,
        node_mask=batch.node_mask,
    )

    return CorruptedBatchInfo(
        clean=batch,
        observed=observed_batch,
        removed=removed_batch,
        fake=fake_batch,
        keep_mask=keep_mask,
    )


class CorruptedEventStreamDataset(EventStreamDataset):
    """
    Standard wrapper used for training/eval on the observed corrupted stream.
    """
    def __init__(
        self,
        base_ds: EventStreamDataset,
        *,
        drop_real_prob: float = 0.0,
        add_fake_ratio: float = 0.0,
        corrupt_splits: tuple[str, ...] = ("train", "val", "test"),
        seed: int = 0,
        fake_feature_mode: str = "zeros",
        avoid_self_loops: bool = True,
        min_keep_per_nonempty_bin: int = 1,
        skip_empty_observed_bins: bool = True,
        corrupt_unit: str = "undirected_pair", #directed_pair
        block_node_select: str = "random",
    ):
        self.base_ds = base_ds
        self.drop_real_prob = float(drop_real_prob)
        self.add_fake_ratio = float(add_fake_ratio)
        self.corrupt_splits = set(corrupt_splits)
        self.seed = int(seed)
        self.fake_feature_mode = fake_feature_mode
        self.avoid_self_loops = bool(avoid_self_loops)
        self.min_keep_per_nonempty_bin = int(min_keep_per_nonempty_bin)
        self.skip_empty_observed_bins = bool(skip_empty_observed_bins)
        self.corrupt_unit = corrupt_unit
        self.block_node_select = block_node_select

        self._base_spec = self.base_ds.spec()
        self._num_nodes = int(self._base_spec.num_nodes)

    def spec(self) -> DataSpec:
        extra = dict(self._base_spec.extra or {})
        extra["corruption"] = {
            "drop_real_prob": self.drop_real_prob,
            "add_fake_ratio": self.add_fake_ratio,
            "corrupt_splits": sorted(self.corrupt_splits),
            "seed": self.seed,
            "fake_feature_mode": self.fake_feature_mode,
            "avoid_self_loops": self.avoid_self_loops,
            "min_keep_per_nonempty_bin": self.min_keep_per_nonempty_bin,
            "skip_empty_observed_bins": self.skip_empty_observed_bins,
            "corrupt_unit": self.corrupt_unit,
            "block_node_select": self.block_node_select,
        }
        return DataSpec(
            name=self._base_spec.name,
            num_nodes=self._base_spec.num_nodes,
            event_dim=self._base_spec.event_dim,
            num_events=self._base_spec.num_events,
            num_bins=self._base_spec.num_bins,
            extra=extra,
        )

    def bins(self, split: str = "train") -> Iterable[EventBatch]:
        return _CorruptedBinnedStream(
            base_iterable=self.base_ds.bins(split),
            split=split,
            num_nodes=self._num_nodes,
            drop_real_prob=self.drop_real_prob,
            add_fake_ratio=self.add_fake_ratio,
            corrupt=split in self.corrupt_splits,
            seed=self.seed,
            fake_feature_mode=self.fake_feature_mode,
            avoid_self_loops=self.avoid_self_loops,
            min_keep_per_nonempty_bin=self.min_keep_per_nonempty_bin,
            skip_empty_observed_bins=self.skip_empty_observed_bins,
            corrupt_unit=self.corrupt_unit,
            block_node_select=self.block_node_select,
        )


class _CorruptedBinnedStream(Iterable[EventBatch]):
    def __init__(
        self,
        *,
        base_iterable: Iterable[EventBatch],
        split: str,
        num_nodes: int,
        drop_real_prob: float,
        add_fake_ratio: float,
        corrupt: bool,
        seed: int,
        fake_feature_mode: str,
        avoid_self_loops: bool,
        min_keep_per_nonempty_bin: int,
        skip_empty_observed_bins: bool,
        corrupt_unit: str,
        block_node_select: str = "random",
    ):
        self.base_iterable = base_iterable
        self.split = split
        self.num_nodes = int(num_nodes)
        self.drop_real_prob = float(drop_real_prob)
        self.add_fake_ratio = float(add_fake_ratio)
        self.corrupt = bool(corrupt)
        self.seed = int(seed)
        self.fake_feature_mode = fake_feature_mode
        self.avoid_self_loops = bool(avoid_self_loops)
        self.min_keep_per_nonempty_bin = int(min_keep_per_nonempty_bin)
        self.skip_empty_observed_bins = bool(skip_empty_observed_bins)
        self.corrupt_unit = corrupt_unit
        self.block_node_select = block_node_select

    def __iter__(self) -> Iterator[EventBatch]:
        for batch_idx, batch in enumerate(self.base_iterable):
            if not self.corrupt:
                yield batch
                continue

            info = corrupt_batch_with_metadata(
                batch,
                split=self.split,
                batch_idx=batch_idx,
                num_nodes=self.num_nodes,
                drop_real_prob=self.drop_real_prob,
                add_fake_ratio=self.add_fake_ratio,
                seed=self.seed,
                fake_feature_mode=self.fake_feature_mode,
                avoid_self_loops=self.avoid_self_loops,
                min_keep_per_nonempty_bin=self.min_keep_per_nonempty_bin,
                corrupt_unit=self.corrupt_unit,
                block_node_select=self.block_node_select,
            )

            # For normal corrupted-only streams, skipping empty bins is okay.
            # For clean-target/corrupted-context streams, skipping breaks time alignment.
            if self.skip_empty_observed_bins and info.observed.num_events == 0:
                continue

            yield info.observed

def wrap_dataset_dict_with_corruption(
    datasets: Mapping[str, EventStreamDataset],
    **kwargs,
) -> dict[str, EventStreamDataset]:
    return {
        name: CorruptedEventStreamDataset(ds, **kwargs)
        for name, ds in datasets.items()
    }