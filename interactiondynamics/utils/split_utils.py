from __future__ import annotations

from typing import Dict, Sequence, Tuple

VALID_SPLITS: Tuple[str, ...] = ("train", "val", "test")


def normalize_split_names(splits: Sequence[str] | None) -> Tuple[str, ...]:
    if splits is None:
        return VALID_SPLITS

    out = []
    for split in splits:
        s = str(split).lower()
        if s not in VALID_SPLITS:
            raise ValueError(
                f"unknown split={split!r}; expected one of {VALID_SPLITS}"
            )
        if s not in out:
            out.append(s)
    return tuple(out)


def split_scope_tag(splits: Sequence[str] | None) -> str:
    norm = normalize_split_names(splits)
    if norm == VALID_SPLITS:
        return "all"
    return "-".join(norm)


def compute_split_ranges(
    num_bins: int,
    split_fracs: Tuple[float, float, float],
) -> Dict[str, Tuple[int, int]]:
    f_tr, f_va, f_te = split_fracs
    if abs((f_tr + f_va + f_te) - 1.0) >= 1e-6:
        raise ValueError("split_fracs must sum to 1.0")

    tr_end = int(num_bins * f_tr)
    va_end = tr_end + int(num_bins * f_va)

    def _range(start: int, end_exclusive: int) -> Tuple[int, int]:
        if end_exclusive <= start:
            return (start, start - 1)  # empty range
        return (start, end_exclusive - 1)

    return {
        "train": _range(0, tr_end),
        "val": _range(tr_end, va_end),
        "test": _range(va_end, num_bins),
    }


def split_for_bin_idx(
    bin_idx: int,
    split_ranges: Dict[str, Tuple[int, int]],
) -> str:
    for split, (b0, b1) in split_ranges.items():
        if b0 <= bin_idx <= b1:
            return split
    raise ValueError(f"bin_idx={bin_idx} did not fall into any split range")