import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _default_node_ids(num_nodes: int, max_nodes_large: int = 4):
    if num_nodes <= max_nodes_large:
        return list(range(num_nodes))

    return sorted({
        0,
        num_nodes // 4,
        num_nodes // 2,
        (3 * num_nodes) // 4,
    })


def _compute_robust_ylim(values, robust_percentile=99.0, pad_frac=0.08, symmetric=False):
    """
    Compute y-limits from percentiles instead of raw min/max so a few giant
    outliers do not flatten the whole plot.

    values: 1D array of plotted values
    """
    values = np.asarray(values)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return (-1.0, 1.0)

    if symmetric:
        bound = np.percentile(np.abs(values), robust_percentile)
        if bound <= 0:
            bound = 1.0
        pad = pad_frac * bound
        return (-bound - pad, bound + pad)

    lo = np.percentile(values, 100 - robust_percentile)
    hi = np.percentile(values, robust_percentile)

    if lo == hi:
        scale = max(1.0, abs(lo))
        return (lo - 0.1 * scale, hi + 0.1 * scale)

    span = hi - lo
    pad = pad_frac * span
    return (lo - pad, hi + pad)


def plot_node_targets_by_feature(
    times,
    y_true,
    y_pred,
    out_path=None,
    dataset_name="dataset",
    target_names=None,
    node_ids=None,
    max_nodes_large=4,
    dpi=200,
    model_combo=None,
    # new knobs
    y_mode="robust",            # "auto", "robust", "manual", "symlog"
    robust_percentile=99.0,
    symmetric_y=False,
    ylim=None,                  # used if y_mode="manual"
    per_panel_ylim=True,        # compute limits separately for each target dim
    symlog_linthresh=1e-2,
):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    times = np.asarray(times)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true shape {y_true.shape} != y_pred shape {y_pred.shape}")

    if y_true.ndim != 3:
        raise ValueError(
            f"Expected y_true/y_pred to have shape [T, N, D], got {y_true.shape}"
        )

    _, num_nodes, target_dim = y_true.shape

    if target_names is None:
        target_names = [f"target_{i}" for i in range(target_dim)]

    if len(target_names) != target_dim:
        raise ValueError(
            f"len(target_names)={len(target_names)} but target_dim={target_dim}"
        )

    if node_ids is None:
        node_ids = _default_node_ids(num_nodes=num_nodes, max_nodes_large=max_nodes_large)

    for n in node_ids:
        if n < 0 or n >= num_nodes:
            raise ValueError(f"node id {n} is out of bounds for num_nodes={num_nodes}")

    if out_path is not None:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    ncols = 2 if target_dim > 1 else 1
    nrows = math.ceil(target_dim / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7 * ncols, 4.5 * nrows),
        sharex=True,
    )

    if isinstance(axes, np.ndarray):
        axes = axes.ravel().tolist()
    else:
        axes = [axes]

    for ax in axes[target_dim:]:
        ax.axis("off")

    cmap = plt.get_cmap("tab10")
    node_colors = {node_idx: cmap(i % 10) for i, node_idx in enumerate(node_ids)}

    # precompute global robust ylim if wanted
    global_panel_ylim = None
    if y_mode == "robust" and not per_panel_ylim:
        all_vals = []
        for d in range(target_dim):
            for node_idx in node_ids:
                all_vals.append(y_true[:, node_idx, d])
                all_vals.append(y_pred[:, node_idx, d])
        all_vals = np.concatenate(all_vals)
        global_panel_ylim = _compute_robust_ylim(
            all_vals,
            robust_percentile=robust_percentile,
            symmetric=symmetric_y,
        )

    for d in range(target_dim):
        ax = axes[d]
        feat_name = target_names[d]

        panel_vals = []

        for node_idx in node_ids:
            color = node_colors[node_idx]

            true_series = y_true[:, node_idx, d]
            pred_series = y_pred[:, node_idx, d]

            panel_vals.append(true_series)
            panel_vals.append(pred_series)

            ax.plot(
                times,
                true_series,
                linestyle=":",
                color=color,
                linewidth=2.0,
                alpha=0.95,
            )
            ax.plot(
                times,
                pred_series,
                linestyle="-",
                color=color,
                linewidth=2.0,
                alpha=0.95,
            )

        # y-axis handling
        if y_mode == "manual":
            if ylim is not None:
                ax.set_ylim(*ylim)

        elif y_mode == "robust":
            if per_panel_ylim:
                vals = np.concatenate(panel_vals)
                lo, hi = _compute_robust_ylim(
                    vals,
                    robust_percentile=robust_percentile,
                    symmetric=symmetric_y,
                )
                ax.set_ylim(lo, hi)
            else:
                ax.set_ylim(*global_panel_ylim)

        elif y_mode == "symlog":
            ax.set_yscale("symlog", linthresh=symlog_linthresh)

        # y_mode == "auto" does nothing

        ax.set_title(feat_name)
        ax.set_xlabel("time bin")
        ax.set_ylabel(feat_name)
        ax.grid(alpha=0.25)

    node_handles = [
        Line2D([0], [0], color=node_colors[node_idx], lw=2, label=f"node {node_idx}")
        for node_idx in node_ids
    ]
    style_handles = [
        Line2D([0], [0], color="black", lw=2, linestyle=":", label="true"),
        Line2D([0], [0], color="black", lw=2, linestyle="-", label="pred"),
    ]

    fig.legend(
        handles=node_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=min(len(node_handles), 4),
        frameon=True,
    )
    fig.legend(
        handles=style_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=2,
        frameon=True,
    )

    title = f"{dataset_name}: node targets over time"
    if model_combo is not None:
        title = f"{dataset_name} | {model_combo}: node targets over time"

    fig.suptitle(
        title,
        fontsize=14,
        y=1.04,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    if out_path is not None:
        fig.savefig(
            out_path,
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(fig)
    else:
        plt.show()