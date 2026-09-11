"""Traffic forecasting data, kept separate from interaction-event training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from interactiondynamics.data.download_benchmarks import SOURCES
from interactiondynamics.data.social import split_bounds


@dataclass
class TrafficConfig:
    name: str = "metr_la"
    root: str = "data"
    split_fracs: tuple[float, float, float] = (0.7, 0.1, 0.2)
    history: int = 12
    horizon: int = 12
    zero_is_missing: bool = True


class TrafficDataset:
    """Raw speeds and masks [time, sensor, 1], plus lazy forecasting windows.

    Window inputs are normalized with train-only statistics. Targets retain
    original units. Every window stays entirely inside its assigned split.
    No topology is inferred from held-out speeds or fabricated as events.
    """
    def __init__(self, cfg: TrafficConfig):
        if cfg.name not in {"metr_la", "pems_bay"}:
            raise ValueError(f"Unknown traffic dataset: {cfg.name}")
        if cfg.history < 1 or cfg.horizon < 1:
            raise ValueError("history and horizon must be positive")
        self.cfg = cfg
        path = Path(cfg.root) / cfg.name / SOURCES[cfg.name][0]
        try:
            frame = pd.read_hdf(path)
        except TypeError:
            # Original DCRNN files store Python-2 byte-string pandas metadata,
            # which newer pandas cannot decode. Read their numeric fixed block
            # directly without unpickling or mutating the downloaded artifact.
            import tables
            with tables.open_file(path) as handle:
                group = handle.get_node('/df' if cfg.name == 'metr_la' else '/speed')
                columns = group.axis0.read()
                if not np.array_equal(columns, group.block0_items.read()):
                    raise ValueError("Unsupported multi-block traffic HDF layout")
                frame = pd.DataFrame(group.block0_values.read(),
                                     index=pd.to_datetime(group.axis1.read(), unit='ns'),
                                     columns=[v.decode() if isinstance(v, bytes) else v for v in columns])
        if not isinstance(frame.index, pd.DatetimeIndex) or not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
            raise ValueError("Expected unique chronological timestamps")
        gaps = frame.index[1:] - frame.index[:-1]
        if frame.columns.has_duplicates or not np.all(gaps % pd.Timedelta(minutes=5) == pd.Timedelta(0)):
            raise ValueError("Expected unique sensors and timestamps on a five-minute grid")
        self.timestamps = frame.index
        self.gap_prefix = np.concatenate([[0], np.cumsum(gaps != pd.Timedelta(minutes=5))])
        self.sensor_ids = frame.columns.to_numpy()
        values = frame.to_numpy(dtype=np.float32)[..., None]
        self.observed = torch.from_numpy(np.isfinite(values) & ((values != 0) if cfg.zero_is_missing else True))
        self.values = torch.from_numpy(np.where(self.observed.numpy(), values, 0))
        self.bounds = split_bounds(len(frame), cfg.split_fracs)
        train_end = self.bounds['train'][1]
        valid = self.values[:train_end][self.observed[:train_end]]
        if valid.numel() == 0:
            raise ValueError("No observed training values")
        self.mean = valid.mean()
        self.std = valid.std(unbiased=False).clamp_min(1e-8)

    def windows(self, split: str = "train"):
        return _TrafficWindows(self, *self.bounds[split])


class _TrafficWindows(Dataset):
    def __init__(self, data: TrafficDataset, start: int, end: int):
        self.data, self.start = data, start
        width = data.cfg.history + data.cfg.horizon
        starts = np.arange(start, max(start, end - width + 1))
        # PEMS-BAY has a clock jump in March. Preserve original timestamps,
        # excluding windows that span a discontinuity rather than pretending
        # that those observations are adjacent five-minute samples.
        self.starts = starts[data.gap_prefix[starts + width - 1] == data.gap_prefix[starts]]
        self.count = len(self.starts)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        ds = self.data
        start = int(self.starts[index])
        middle, end = start + ds.cfg.history, start + ds.cfg.history + ds.cfg.horizon
        mask = ds.observed[start:middle]
        x = torch.where(mask, (ds.values[start:middle] - ds.mean) / ds.std, 0)
        return {"x": x, "x_mask": mask, "y": ds.values[middle:end],
                "y_mask": ds.observed[middle:end], "target_index": middle}
