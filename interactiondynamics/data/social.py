"""Homogeneous social event streams. No downloads occur during loading."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

from interactiondynamics.core.events import EventBatch
from interactiondynamics.data.interfaces import DataSpec, EventStreamDataset
from interactiondynamics.data.download_benchmarks import SOURCES


def split_bounds(n: int, fractions: tuple[float, float, float]) -> dict[str, tuple[int, int]]:
    if len(fractions) != 3 or any(not np.isfinite(f) or f <= 0 for f in fractions) or not np.isclose(sum(fractions), 1):
        raise ValueError("split_fracs must contain three positive fractions summing to one")
    a = int(n * fractions[0])
    b = a + int(n * fractions[1])
    if not 0 < a < b < n:
        raise ValueError("Not enough time steps for three nonempty splits")
    return {"train": (0, a), "val": (a, b), "test": (b, n)}


@dataclass
class SocialConfig:
    name: str = "college_msg"
    root: str = "data"
    bin_size: float | None = None
    split_fracs: tuple[float, float, float] = (0.7, 0.15, 0.15)
    device: str = "cpu"
    split_by: str = "events"


class SocialEventDataset(EventStreamDataset):
    continuous_stream = True

    def __init__(self, cfg: SocialConfig):
        if cfg.name not in {"college_msg", "email_eu_core", "sociopatterns"}:
            raise ValueError(f"Unknown social dataset: {cfg.name}")
        self.cfg = cfg
        self.bin_size = cfg.bin_size if cfg.bin_size is not None else (20.0 if cfg.name == "sociopatterns" else 3600.0)
        if not np.isfinite(self.bin_size) or self.bin_size <= 0:
            raise ValueError("bin_size must be finite and positive")
        path = Path(cfg.root) / cfg.name / SOURCES[cfg.name][0]
        # SNAP: src dst timestamp. SocioPatterns: timestamp src dst class class.
        cols = (1, 2, 0) if cfg.name == "sociopatterns" else (0, 1, 2)
        raw = np.loadtxt(path, dtype=np.int64, usecols=cols, ndmin=2)
        if not len(raw):
            raise ValueError(f"Empty dataset: {path}")
        self.node_ids = np.unique(raw[:, :2])
        src, dst = np.searchsorted(self.node_ids, raw[:, 0]), np.searchsorted(self.node_ids, raw[:, 1])
        timestamps = raw[:, 2]
        self.num_contacts = len(raw)
        if cfg.name == "sociopatterns":
            # Undirected proximity is represented once in each direction.
            nonself = src != dst
            src, dst, timestamps = (np.concatenate([src, dst[nonself]]),
                                    np.concatenate([dst, src[nonself]]),
                                    np.concatenate([timestamps, timestamps[nonself]]))
        order = np.argsort(timestamps, kind="stable")
        self.src = torch.from_numpy(src[order].copy())
        self.dst = torch.from_numpy(dst[order].copy())
        self.timestamps = timestamps[order]
        # Subtract in integer seconds before conversion: UNIX timestamps must
        # not lose their 20-second resolution through float32 rounding.
        self.bin_ids = np.floor((self.timestamps - self.timestamps[0]) / self.bin_size).astype(np.int64)
        self.num_bins = int(self.bin_ids[-1]) + 1
        self.offsets = np.searchsorted(self.bin_ids, np.arange(self.num_bins + 1))
        self.bounds = split_bounds(self.num_bins, cfg.split_fracs)
        if cfg.split_by == "events":
            # Quantile boundaries snap to bin starts, so simultaneous events
            # cannot leak across splits and long empty periods do not consume
            # the entire validation set (notably in Email-Eu-core).
            event_bounds = split_bounds(len(self.bin_ids), cfg.split_fracs)
            a = int(self.bin_ids[event_bounds['val'][0]])
            b = int(self.bin_ids[event_bounds['test'][0]])
            if not 0 < a < b < self.num_bins:
                raise ValueError("Event quantiles do not span three distinct time-bin ranges")
            self.bounds = {'train': (0, a), 'val': (a, b), 'test': (b, self.num_bins)}
        elif cfg.split_by != "time":
            raise ValueError("split_by must be 'events' or 'time'")

    def spec(self) -> DataSpec:
        return DataSpec(self.cfg.name, len(self.node_ids), 1, len(self.src), self.num_bins,
                        extra={"is_bipartite": False, "unit_force": True,
                               "bin_size": self.bin_size, "split_fracs": self.cfg.split_fracs,
                               "split_by": self.cfg.split_by,
                               "empty_bins_preserved": True,
                               "undirected": self.cfg.name == "sociopatterns",
                               "primary_metric": {"path": "val.event_auroc", "goal": "max"},
                               "summary_metrics": ["test.event_auroc", "test.event_auprc", "test.mrr"]})

    def bins(self, split: str = "train"):
        start, end = self.bounds[split]
        return _SocialBins(self, start, end)


@dataclass
class _SocialBins:
    dataset: SocialEventDataset
    start: int
    end: int

    def __len__(self):
        return self.end - self.start

    def __iter__(self):
        ds = self.dataset
        for t in range(self.start, self.end):
            a, b = ds.offsets[t:t+2]
            yield EventBatch(src=ds.src[a:b], dst=ds.dst[a:b],
                             t=torch.full((b-a,), t, dtype=torch.long),
                             features=torch.ones((b-a, 1)),
                             is_external=torch.zeros(b-a, dtype=torch.bool)).to(ds.cfg.device)
