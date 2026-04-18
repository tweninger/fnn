from __future__ import annotations

def higher_is_better(metric_name: str) -> bool:
    return metric_name in {
        "r2",
        "pearson",
        "spearman",
        "mean_node_pearson",
        "mean_node_spearman",
        "mean_node_r2",
        "mean_cosine",
    }

def is_better_metric(new_value: float, old_value: float, metric_name: str) -> bool:
    return new_value > old_value if higher_is_better(metric_name) else new_value < old_value

