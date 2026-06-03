from __future__ import annotations

from typing import Optional, Sequence, Union

from utils.split_utils import split_scope_tag

Number = Union[int, float]


def _format_threshold_value(value: Number) -> str:
    if isinstance(value, int):
        return str(value)

    s = f"{float(value):.3f}".rstrip("0").rstrip(".")
    return s.replace(".", "p").replace("-", "m")


def make_threshold_dataset_name(
    base_name: str,
    threshold_metric: Optional[str] = None,
    threshold_value: Optional[Number] = None,
    threshold_splits: Optional[Sequence[str]] = None,
) -> str:
    name = base_name

    if threshold_metric is not None and threshold_value is not None:
        name = f"{name}_{threshold_metric}_{_format_threshold_value(threshold_value)}"

    return name