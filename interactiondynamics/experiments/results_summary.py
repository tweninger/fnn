from collections import defaultdict
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


def format_node_analysis_metrics(result) -> str:
    analysis = getattr(result, "analysis_test", {}) or {}
    return (
        f"rmse={analysis.get('rmse', float('nan')):.6f} | "
        f"mse={analysis.get('mse', float('nan')):.6f} | "
        f"r2={analysis.get('r2', float('nan')):.6f} | "
        f"mean_node_pearson={analysis.get('mean_node_pearson', float('nan')):.6f} | "
        f"mean_node_spearman={analysis.get('mean_node_spearman', float('nan')):.6f}"
    )


def format_finished_label() -> str:
    return color_text("FINISHED:", TermColor.BOLD, TermColor.HOT_PINK)


def format_analysis_label() -> str:
    return color_text("ANALYSIS:", TermColor.BOLD, TermColor.SOFT_PINK)


def format_starting_run_banner(dataset_name: str, run_name: str, seed: int) -> str:
    broom = color_text("🧹" * 50, TermColor.PEACH)
    line = (
        f"{color_text('STARTING RUN:', TermColor.BOLD, TermColor.ORANGE)} "
        f"dataset={color_text(dataset_name, TermColor.BOLD, TermColor.HOT_PINK)} | "
        f"run={color_text(run_name, TermColor.SOFT_PINK)} | "
        f"seed={color_text(str(seed), TermColor.GOLD)}"
    )
    return f"\n{broom}\n\n{line}\n\n{broom}"


def format_top_runs_header(selection_metric: str, dataset_name: str) -> str:
    title = color_text(
        f"TOP RUNS BY BEST VAL {selection_metric.upper()} — {dataset_name}",
        TermColor.BOLD,
        TermColor.ORANGE,
    )
    divider = color_text("-" * 60, TermColor.PEACH)
    return f"\n{divider}\n{title}\n{divider}"


def print_seed_avg(results):
    grouped = defaultdict(list)

    for r in results:
        run_name = r["name"]   # includes dataset prefix already
        best_test = r["best_snapshot"]["test"]["mrr"]
        grouped[run_name].append((r["best_val_mrr"], best_test))

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