from __future__ import annotations

from datasets.charged_particles import (
    ChargedParticlesBinnedConfig,
    make_charged_particle_threshold_variants,
)
from experiments.results_summary import (
    TermColor,
    color_text,
    format_analysis_label,
    format_failed_label,
    format_finished_label,
    format_starting_run_banner,
    format_top_runs_header,
)
from training.whole_bin_edge_prediction import WholeBinRunResult

DEFAULT_WHOLE_BIN_CORRUPTION = dict(
    drop_real_prob=0.50,
    add_fake_ratio=0.0,
    corrupt_splits=("train",),
    seed=17,
    fake_feature_mode="zeros",
    avoid_self_loops=True,
    min_keep_per_nonempty_bin=1,
    skip_empty_observed_bins=False,
    corrupt_unit="node_block",  # directed_event | undirected_pair | node_block
    block_node_select="high_degree",  # random | high_degree (node_block only)
)


def format_whole_bin_corruption_tag(corruption: dict) -> str:
    """Short tag for dataset/run names (matches interaction_prediction style)."""
    parts = [
        f"drop={float(corruption['drop_real_prob']):.2f}",
        f"splits={'-'.join(corruption['corrupt_splits'])}",
    ]
    unit = corruption.get("corrupt_unit", "undirected_pair")
    if unit == "node_block":
        parts.append("unit=node_block")
        parts.append(f"block={corruption.get('block_node_select', 'random')}")
    elif unit != "undirected_pair":
        parts.append(f"unit={unit}")
    return "|".join(parts)


def build_thresholded_whole_bin_datasets(device: torch.device) -> dict:

    charged_base_cfg = ChargedParticlesBinnedConfig(
        name="charged_particles",
        device=device,
        interaction_rule="all_pairs",
        obs_edge_keep_prob=1.0,
        min_edges_per_bin=1,
        threshold_splits=("train", "val", "test"),
    )
    
    return make_charged_particle_threshold_variants(
        charged_base_cfg,
        threshold_metric="distance_threshold",
        threshold_values=(4,),
        threshold_splits_options=(("train", "val", "test"),),
    )


def use_upper_triangle_for_dataset(dataset_name: str) -> bool:
    return dataset_name.startswith(("charged_particles", "springweb"))


def describe_whole_bin_run_result(result: WholeBinRunResult) -> str:
    name = str(result.name)
    agg_val = None
    upd_val = None
    for part in name.split("|"):
        if part.startswith("agg="):
            agg_val = part.split("=", 1)[1]
        elif part.startswith("update="):
            upd_val = part.split("=", 1)[1]
    agg_part = (
        color_text(f"agg={agg_val}", TermColor.BOLD, TermColor.HOT_PINK) if agg_val is not None else "agg=?"
    )
    upd_part = (
        color_text(f"update={upd_val}", TermColor.BOLD, TermColor.HOT_PINK) if upd_val is not None else "update=?"
    )
    best_part = color_text(
        f"best_val_{result.selection_metric}={result.best_val_metric:.4f}",
        TermColor.BOLD,
        TermColor.HOT_PINK,
    )
    return f"{result.name} | seed={result.seed} | {agg_part} | {upd_part} | {best_part} @ epoch {result.best_epoch}"


def format_whole_bin_metrics(result: WholeBinRunResult) -> str:
    best = result.best_snapshot or {}
    val = best.get("val", {}) or {}
    test = best.get("test", {}) or {}
    if "tuned_jaccard" not in val:
        return "no tuned metrics"
    line = f"@t={float(val.get('tuned_threshold', float('nan'))):.6f}"
    line += " | " + color_text(
        f"val_tuned_jaccard={float(val['tuned_jaccard']):.4f}", TermColor.BOLD, TermColor.HOT_PINK
    )
    line += " | " + color_text(
        f"val_tuned_f1={float(val.get('tuned_f1', float('nan'))):.4f}", TermColor.BOLD, TermColor.HOT_PINK
    )
    line += " | " + color_text(
        f"test_tuned_jaccard={float(test.get('tuned_jaccard', float('nan'))):.4f}",
        TermColor.BOLD,
        TermColor.HOT_PINK,
    )
    line += " | " + color_text(
        f"test_tuned_f1={float(test.get('tuned_f1', float('nan'))):.4f}", TermColor.BOLD, TermColor.HOT_PINK
    )
    return line


__all__ = [
    "DEFAULT_WHOLE_BIN_CORRUPTION",
    "format_whole_bin_corruption_tag",
    "build_thresholded_whole_bin_datasets",
    "use_upper_triangle_for_dataset",
    "describe_whole_bin_run_result",
    "format_whole_bin_metrics",
    "format_analysis_label",
    "format_failed_label",
    "format_finished_label",
    "format_starting_run_banner",
    "format_top_runs_header",
]
