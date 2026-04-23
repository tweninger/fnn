from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


class TermColor:
    RESET = "\033[0m"
    BOLD = "\033[1m"

    HOT_PINK = "\033[38;5;205m"
    SOFT_PINK = "\033[38;5;218m"
    PEACH = "\033[38;5;215m"
    ORANGE = "\033[38;5;208m"
    GOLD = "\033[38;5;222m"



def color_text(text: str, *styles: str) -> str:
    return "".join(styles) + text + TermColor.RESET



def _field(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)



def _nested_metric(mapping: dict | None, *keys: str, default: float = float("nan")) -> float:
    cur: Any = mapping or {}
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    if cur is None:
        return default
    try:
        return float(cur)
    except (TypeError, ValueError):
        return default



def format_node_analysis_metrics(result) -> str:
    analysis = getattr(result, "analysis_test", {}) or {}
    return (
        f"rmse={analysis.get('rmse', float('nan')):.6f} | "
        f"mse={analysis.get('mse', float('nan')):.6f} | "
        f"r2={analysis.get('r2', float('nan')):.6f} | "
        f"mean_node_pearson={analysis.get('mean_node_pearson', float('nan')):.6f} | "
        f"mean_node_spearman={analysis.get('mean_node_spearman', float('nan')):.6f}"
    )



def format_interaction_metrics(result) -> str:
    best_snapshot = _field(result, "best_snapshot", {}) or {}
    final_snapshot = _field(result, "final_snapshot", {}) or {}
    best_val_mrr = float(_field(result, "best_val_mrr", float("nan")))

    best_train_mrr = _nested_metric(best_snapshot, "train_eval", "mrr")
    best_test_mrr = _nested_metric(best_snapshot, "test", "mrr")
    final_test_mrr = _nested_metric(final_snapshot, "test", "mrr")

    return (
        f"train_mrr={best_train_mrr:.6f} | "
        f"val_mrr={best_val_mrr:.6f} | "
        f"best_test_mrr={best_test_mrr:.6f} | "
        f"final_test_mrr={final_test_mrr:.6f}"
    )



def format_finished_label() -> str:
    return color_text("FINISHED:", TermColor.BOLD, TermColor.HOT_PINK)



def format_failed_label() -> str:
    return color_text("FAILED:", TermColor.BOLD, TermColor.ORANGE)



def format_analysis_label() -> str:
    return color_text("ANALYSIS:", TermColor.BOLD, TermColor.SOFT_PINK)



def format_starting_run_banner(base_dataset_name: str, run_name: str, seed: int) -> str:
    broom = color_text("🧹" * 50, TermColor.PEACH)
    line = (
        f"{color_text('STARTING RUN:', TermColor.BOLD, TermColor.ORANGE)} "
        f"dataset={color_text(base_dataset_name, TermColor.BOLD, TermColor.HOT_PINK)} | "
        f"run={color_text(run_name, TermColor.SOFT_PINK)} | "
        f"seed={color_text(str(seed), TermColor.GOLD)}"
    )
    return f"\n{broom}\n\n{line}\n\n{broom}"



def format_top_runs_header(selection_metric: str, base_dataset_name: str) -> str:
    title = color_text(
        f"TOP RUNS BY BEST VAL {selection_metric.upper()} — {base_dataset_name}",
        TermColor.BOLD,
        TermColor.ORANGE,
    )
    divider = color_text("-" * 60, TermColor.PEACH)
    return f"\n{divider}\n{title}\n{divider}"



def describe_interaction_run_result(result) -> str:
    best_snapshot = _field(result, "best_snapshot", {}) or {}
    best_test_mrr = _nested_metric(best_snapshot, "test", "mrr")
    run_name = _field(result, "name", "<unnamed>")
    best_val_mrr = float(_field(result, "best_val_mrr", float("nan")))
    best_epoch = _field(result, "best_epoch", "?")
    seed = _field(result, "seed", "?")

    return (
        f"{color_text(str(run_name), TermColor.BOLD)} | "
        f"seed={seed} | "
        f"{color_text('best_val_mrr', TermColor.HOT_PINK)}="
        f"{color_text(f'{best_val_mrr:.6f}', TermColor.HOT_PINK)} "
        f"@ epoch {best_epoch} | "
        f"{color_text('best_test_mrr', TermColor.SOFT_PINK)}="
        f"{color_text(f'{best_test_mrr:.6f}', TermColor.SOFT_PINK)}"
    )


def print_interaction_seed_avg(results) -> None:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for result in results:
        run_name = _field(result, "name", _field(result, "run", "UNKNOWN"))
        best_snapshot = _field(result, "best_snapshot", {}) or {}
        best_test = _nested_metric(best_snapshot, "test", "mrr")
        grouped[run_name].append((float(_field(result, "best_val_mrr", np.nan)), best_test))

    print("\n=== Mean across seeds ===")
    for run_name, vals in grouped.items():
        val_mrrs = np.array([x[0] for x in vals], dtype=float)
        test_mrrs = np.array([x[1] for x in vals], dtype=float)

        print(f"\n=== {run_name} ===")
        print(f"n_seeds:           {len(vals)}")
        print(f"avg best val mrr:  {val_mrrs.mean():.4f}")
        print(f"std best val mrr:  {val_mrrs.std(ddof=1) if len(vals) > 1 else 0.0:.4f}")
        print(f"avg best test mrr: {test_mrrs.mean():.4f}")
        print(f"std best test mrr: {test_mrrs.std(ddof=1) if len(vals) > 1 else 0.0:.4f}")


# backward-compatible alias
print_seed_avg = print_interaction_seed_avg



def describe_run_result(r) -> str:
    selection_metric = getattr(r, "selection_metric", "rmse")
    best_val_metric = getattr(r, "best_val_metric", float("nan"))

    analysis_test = getattr(r, "analysis_test", {}) or {}
    test_rmse = analysis_test.get(
        "rmse",
        r.best_snapshot.get("test", {}).get("rmse", float("nan"))
    )

    return (
        f"{color_text(f'{r.name}', TermColor.BOLD)} | "
        f"{color_text(f'best_val_{selection_metric}', TermColor.HOT_PINK)}="
        f"{color_text(f'{best_val_metric:.6f}', TermColor.HOT_PINK)} "
        f"@ epoch {r.best_epoch} | "
        f"{color_text('test_rmse', TermColor.SOFT_PINK)}="
        f"{color_text(f'{test_rmse:.6f}', TermColor.SOFT_PINK)}"
    )

def format_recovery_label() -> str:
    return color_text("RECOVERY:", TermColor.BOLD, TermColor.GOLD)


def format_recovery_metrics(recovery: dict | None) -> str:
    if not recovery:
        return color_text("no recovery metrics", TermColor.SOFT_PINK)

    def _fmt_block(split_name: str, stats: dict) -> str:
        mrr = float(stats.get("mrr", float("nan")))
        hits10 = float(stats.get("hits@10", float("nan")))
        removed = int(stats.get("removed_events", 0))

        return (
            f"{color_text(split_name, TermColor.BOLD, TermColor.SOFT_PINK)} "
            f"hidden_mrr={color_text(f'{mrr:.4f}', TermColor.HOT_PINK)} | "
            f"hidden_hits@10={color_text(f'{hits10:.4f}', TermColor.PEACH)} | "
            f"removed={color_text(str(removed), TermColor.GOLD)}"
        )

    ordered_splits = ["train", "val", "test"]
    parts = []

    for split_name in ordered_splits:
        stats = recovery.get(split_name)
        if stats:
            parts.append(_fmt_block(split_name, stats))

    # catch any weird extra split names just in case
    for split_name, stats in recovery.items():
        if split_name not in ordered_splits and stats:
            parts.append(_fmt_block(split_name, stats))

    if not parts:
        return color_text("no recovery metrics", TermColor.SOFT_PINK)

    return " || ".join(parts)