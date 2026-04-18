from collections import defaultdict
import numpy as np

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
    test_r2 = analysis_test.get("r2", float("nan"))
    test_pearson = analysis_test.get("mean_node_pearson", float("nan"))

    return (
        f"{r.name} | seed={r.seed} | "
        f"best_val_{selection_metric}={best_val_metric:.6f} @ epoch {r.best_epoch} | "
        f"test_rmse={test_rmse:.6f} | "
        f"test_r2={test_r2:.6f} | "
        f"test_mean_node_pearson={test_pearson:.6f}"
    )
