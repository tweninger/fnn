from __future__ import annotations

from typing import Any, Dict, Optional

import torch


def accumulate_ift_aux(aux: Optional[Dict[str, Any]], sums: Dict[str, float]) -> bool:
    """
    Sum per-step IFT diagnostics from model.step() aux into sums.

    IFTDiffusionUpdate returns scalar floats (kappa, h_norm, inj_norm, ...).
    Returns True if any IFT aux was recorded.
    """
    if aux is None or "kappa" not in aux:
        return False

    for key, val in aux.items():
        if torch.is_tensor(val):
            sums[key] = sums.get(key, 0.0) + float(val.detach().item())
        elif isinstance(val, (int, float)):
            sums[key] = sums.get(key, 0.0) + float(val)
    return True


def finalize_ift_aux(sums: Dict[str, float], n_steps: int) -> Dict[str, float]:
    """Epoch-average IFT aux; adds kappa_mean alias for backward compatibility."""
    if n_steps <= 0 or not sums:
        return {}

    out = {key: total / n_steps for key, total in sums.items()}
    if "kappa" in out:
        out["kappa_mean"] = out["kappa"]
    return out
