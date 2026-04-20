from __future__ import annotations

from typing import Optional, Union

Number = Union[int, float]


def _format_threshold_value(value: Number) -> str:
    """
    Make numeric threshold values filesystem-friendly but still readable.
    Example:
        0.074443 -> 0p074443
        38.591662 -> 38p591662
        2 -> 2
    """
    if isinstance(value, int):
        return str(value)

    s = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return s.replace(".", "p").replace("-", "m")


def make_threshold_dataset_name(
    base_name: str,
    threshold_metric: Optional[str] = None,
    threshold_value: Optional[Number] = None,
) -> str:
    """
    Examples:
        make_threshold_dataset_name("wave")
            -> "wave"

        make_threshold_dataset_name("wave", "pair_accel", 38.591662)
            -> "wave__pair_accel__38p591662"

        make_threshold_dataset_name("charged_particles", "force_threshold", 0.074443)
            -> "charged_particles__force_threshold__0p074443"
    """
    if threshold_metric is None or threshold_value is None:
        return base_name

    return f"{base_name}__{threshold_metric}__{_format_threshold_value(threshold_value)}"