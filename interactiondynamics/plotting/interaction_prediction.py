from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np



def _extract_plot_rows(results: Iterable[object]) -> list[dict]:
    rows: list[dict] = []
    for result in results:
        best_snapshot = getattr(result, "best_snapshot", {}) or {}
        best_test = (best_snapshot.get("test", {}) or {}).get("mrr", np.nan)
        rows.append(
            {
                "name": getattr(result, "name", "<unnamed>"),
                "seed": getattr(result, "seed", 0),
                "best_epoch": getattr(result, "best_epoch", -1),
                "best_val_mrr": float(getattr(result, "best_val_mrr", np.nan)),
                "best_test_mrr": float(best_test),
            }
        )
    rows.sort(key=lambda row: row["best_val_mrr"], reverse=True)
    return rows



def plot_interaction_run_rankings(
    results: Iterable[object],
    *,
    out_path: str | None = None,
    dataset_name: str | None = None,
    title: str | None = None,
):
    rows = _extract_plot_rows(results)
    if not rows:
        raise ValueError("No interaction-prediction results available for plotting.")

    labels = [
        f"{row['name']} | seed={row['seed']} | epoch={row['best_epoch']}"
        for row in rows
    ]
    best_val = np.array([row["best_val_mrr"] for row in rows], dtype=float)
    best_test = np.array([row["best_test_mrr"] for row in rows], dtype=float)

    fig_height = max(4.0, 0.55 * len(rows) + 1.5)
    fig, ax = plt.subplots(figsize=(13, fig_height))

    y = np.arange(len(rows))
    ax.barh(y, best_val, label="Best validation MRR")
    ax.scatter(best_test, y, label="Best test MRR", zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("MRR")
    ax.set_ylabel("Run")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")

    resolved_title = title
    if resolved_title is None:
        resolved_title = "Interaction prediction run ranking"
        if dataset_name is not None:
            resolved_title = f"{dataset_name}: interaction prediction run ranking"
    ax.set_title(resolved_title)

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()