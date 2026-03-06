# data/jodie.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Tuple, cast
import torch

from torch_geometric.datasets import JODIEDataset

from core.events import EventBatch
from data.interfaces import DataSpec, EventStreamDataset


@dataclass
class JODIEConfig:
    root: str
    name: str                    # "Wikipedia" | "Reddit" | "MOOC" | "LastFM"
    bin_size: float = 3600.0     # seconds per bin
    split_fracs: Tuple[float, float, float] = (0.7, 0.15, 0.15)  # train/val/test by time
    device: Optional[torch.device] = None
    cast_features_to_float32: bool = True


class JODIEBinnedDataset(EventStreamDataset):
    def __init__(self, cfg: JODIEConfig):
        self.cfg = cfg

        ds = JODIEDataset(root=cfg.root, name=cfg.name)
        data = ds[0]

        src = getattr(data, "src", None)
        dst = getattr(data, "dst", None)
        ts = getattr(data, "t", None)        
        assert src is not None and dst is not None and ts is not None, \
            "JODIEDataset data must have src, dst, t tensors."

        self._src_all = src.cpu().to(torch.long)
        self._dst_all = dst.cpu().to(torch.long)
        ts_all = ts.cpu().to(torch.float32)

        has_msg = hasattr(data, "msg")
        self._msg_all = None
        if has_msg:
            msg = getattr(data, "msg", None)  # may exist depending on dataset
            assert msg is not None, "data.msg is None despite has_msg=True"
            if cfg.cast_features_to_float32:
                msg = msg.to(torch.float32)
            self._msg_all = msg

        self._num_nodes = int(torch.max(self._src_all.max(), self._dst_all.max()).item()) + 1
        self._event_dim = int(self._msg_all.size(-1)) if self._msg_all is not None else 0

        # Bin timestamps once
        t0 = float(ts_all.min().item())
        self._bin_id_all = torch.floor((ts_all - t0) / float(cfg.bin_size)).to(torch.long)
        self._max_bin = int(self._bin_id_all.max().item())
        self._num_bins = self._max_bin + 1
        self._t0 = t0

        # Compute split ranges in bin-space
        f_tr, f_va, f_te = cfg.split_fracs
        assert abs((f_tr + f_va + f_te) - 1.0) < 1e-6, "split_fracs must sum to 1.0"
        tr_end = int(self._num_bins * f_tr)
        va_end = tr_end + int(self._num_bins * f_va)

        self._split_bins = {
            "train": (0, max(0, tr_end - 1)),
            "val":   (tr_end, max(tr_end, va_end - 1)),
            "test":  (va_end, self._max_bin),
        }

    def spec(self) -> DataSpec:
        return DataSpec(
            name=f"jodie_{self.cfg.name.lower()}",
            num_nodes=self._num_nodes,
            event_dim=self._event_dim,
            num_events=int(self._src_all.numel()),
            num_bins=int(self._num_bins),
            extra={
                "root": self.cfg.root,
                "bin_size": float(self.cfg.bin_size),
                "t0": self._t0,
                "split_fracs": self.cfg.split_fracs,
            },
        )

    def bins(self, split: str = "train") -> Iterable[EventBatch]:
        assert split in self._split_bins, f"unknown split={split}"
        b0, b1 = self._split_bins[split]
        return _JODIEBinnedStream(
            src_all=self._src_all,
            dst_all=self._dst_all,
            bin_id_all=self._bin_id_all,
            msg_all=self._msg_all,
            b0=b0,
            b1=b1,
            device=self.cfg.device,
        )


@dataclass
class _JODIEBinnedStream(Iterable[EventBatch]):
    src_all: torch.Tensor
    dst_all: torch.Tensor
    bin_id_all: torch.Tensor
    msg_all: Optional[torch.Tensor]
    b0: int
    b1: int
    device: Optional[torch.device] = None

    def __iter__(self) -> Iterator[EventBatch]:
        for b in range(self.b0, self.b1 + 1):
            mask = (self.bin_id_all == b)
            if not mask.any():
                continue

            src = self.src_all[mask]
            dst = self.dst_all[mask]
            feats = (self.msg_all[mask] if self.msg_all is not None else None)

            eb = EventBatch(
                src=cast(torch.LongTensor, src),
                dst=cast(torch.LongTensor, dst),
                t=cast(torch.LongTensor, torch.full((src.numel(),), b, dtype=torch.long)),
                features=feats,
            )
            if self.device is not None:
                eb = eb.to(self.device)
            yield eb
