from __future__ import annotations

import math
from typing import Dict, Optional


def node_metric_name(metrics: Dict[str, float]) -> Optional[str]:
    if "node_mse" in metrics:
        return "node_mse"
    if "node_auroc" in metrics:
        return "node_auroc"
    if "node_f1" in metrics:
        return "node_f1"
    if "node_acc" in metrics:
        return "node_acc"
    return None


def edge_metric_name(metrics: Dict[str, float]) -> Optional[str]:
    if "edge_mse" in metrics:
        return "edge_mse"
    if "edge_auroc" in metrics:
        return "edge_auroc"
    if "edge_f1" in metrics:
        return "edge_f1"
    if "edge_acc" in metrics:
        return "edge_acc"
    return None


def primary_metric_name(metrics: Dict[str, float]) -> str:
    if "edge_mse" in metrics:
        return "edge_mse"
    if "edge_auroc" in metrics:
        return "edge_auroc"
    if "edge_f1" in metrics:
        return "edge_f1"
    if "edge_acc" in metrics:
        return "edge_acc"
    if "node_mse" in metrics:
        return "node_mse"
    if "node_auroc" in metrics:
        return "node_auroc"
    if "node_f1" in metrics:
        return "node_f1"
    if "node_acc" in metrics:
        return "node_acc"
    return "mrr"


def infer_primary_metric(*metric_sets: Dict[str, float]) -> str:
    for metrics in metric_sets:
        if "edge_mse" in metrics:
            return "edge_mse"
        if "edge_auroc" in metrics:
            return "edge_auroc"
        if "edge_f1" in metrics:
            return "edge_f1"
        if "edge_acc" in metrics:
            return "edge_acc"
        if "node_mse" in metrics:
            return "node_mse"
        if "node_auroc" in metrics:
            return "node_auroc"
        if "node_f1" in metrics:
            return "node_f1"
        if "node_acc" in metrics:
            return "node_acc"
    return "mrr"


def format_edge_target_summary(name: str, stats: Optional[Dict[str, float]]) -> str:
    if stats is None:
        return f"{name} unavailable"
    return (
        f"{name} count={int(stats['count'])} "
        f"mu={stats['mean']:.4g} sigma={stats['std']:.4g} "
        f"min={stats['min']:.4g} max={stats['max']:.4g}"
    )


def format_edge_metric_bundle(metrics: Dict[str, float], *, prefix: Optional[str] = None) -> str:
    key_prefix = "" if prefix is None else f"{prefix}_"
    mse = metrics.get(f"{key_prefix}edge_mse", float("nan"))
    zmse = metrics.get(f"{key_prefix}edge_zmse", float("nan"))
    r2 = metrics.get(f"{key_prefix}edge_r2", float("nan"))
    nrmse = metrics.get(f"{key_prefix}edge_nrmse", float("nan"))
    corr = metrics.get(f"{key_prefix}edge_corr", float("nan"))
    target_mean = metrics.get(f"{key_prefix}edge_target_mean", float("nan"))
    target_std = metrics.get(f"{key_prefix}edge_target_std", float("nan"))
    persistent_r2 = metrics.get("persistent_edge_r2", float("nan")) if prefix is None else float("nan")
    if math.isnan(mse):
        return ""
    parts = [
        f"mse={mse:.4g}",
        f"zmse={zmse:.3f}" if not math.isnan(zmse) else "zmse=nan",
        f"r2={r2:.3f}" if not math.isnan(r2) else "r2=nan",
        f"nrmse={nrmse:.3f}" if not math.isnan(nrmse) else "nrmse=nan",
        f"corr={corr:.3f}" if not math.isnan(corr) else "corr=nan",
        f"mu={target_mean:.4g}" if not math.isnan(target_mean) else "mu=nan",
        f"sigma={target_std:.4g}" if not math.isnan(target_std) else "sigma=nan",
    ]
    if not math.isnan(persistent_r2):
        parts.append(f"pers_r2={persistent_r2:.3f}")
    return " ".join(parts)


def format_edge_classification_bundle(
    metrics: Dict[str, float],
    *,
    prefix: Optional[str] = None,
) -> str:
    key_prefix = "" if prefix is None else f"{prefix}_"
    auroc = metrics.get(f"{key_prefix}edge_auroc", float("nan"))
    auprc = metrics.get(f"{key_prefix}edge_auprc", float("nan"))
    f1 = metrics.get(f"{key_prefix}edge_f1", float("nan"))
    acc = metrics.get(f"{key_prefix}edge_acc", float("nan"))
    precision = metrics.get(f"{key_prefix}edge_precision", float("nan"))
    recall = metrics.get(f"{key_prefix}edge_recall", float("nan"))
    if math.isnan(acc) and math.isnan(f1) and math.isnan(auroc):
        return ""
    parts = [
        f"acc={acc:.3f}" if not math.isnan(acc) else "acc=nan",
        f"f1={f1:.3f}" if not math.isnan(f1) else "f1=nan",
        f"auroc={auroc:.3f}" if not math.isnan(auroc) else "auroc=nan",
        f"auprc={auprc:.3f}" if not math.isnan(auprc) else "auprc=nan",
        f"prec={precision:.3f}" if not math.isnan(precision) else "prec=nan",
        f"rec={recall:.3f}" if not math.isnan(recall) else "rec=nan",
    ]
    return " ".join(parts)


def format_ranking_metric_bundle(
    metrics: Dict[str, float],
    *,
    prefix: Optional[str] = None,
) -> str:
    key_prefix = "" if prefix is None else f"{prefix}_"
    mrr = metrics.get(f"{key_prefix}mrr", float("nan"))
    hits1 = metrics.get(f"{key_prefix}hits@1", float("nan"))
    hits10 = metrics.get(f"{key_prefix}hits@10", float("nan"))
    auc = metrics.get(f"{key_prefix}pairwise_auc_tie_half", float("nan"))
    if math.isnan(mrr):
        return ""
    parts = [
        f"mrr={mrr:.3f}",
        f"h@1={hits1:.3f}" if not math.isnan(hits1) else "h@1=nan",
        f"h@10={hits10:.3f}" if not math.isnan(hits10) else "h@10=nan",
        f"auc={auc:.3f}" if not math.isnan(auc) else "auc=nan",
    ]
    return " ".join(parts)


def format_regression_diagnostics(metrics: Dict[str, float], stem: str) -> str:
    pred_mean = metrics.get(f"{stem}_pred_mean", float("nan"))
    pred_std = metrics.get(f"{stem}_pred_std", float("nan"))
    target_mean = metrics.get(f"{stem}_target_mean", float("nan"))
    target_std = metrics.get(f"{stem}_target_std", float("nan"))
    corr = metrics.get(f"{stem}_corr", float("nan"))
    if math.isnan(pred_mean) or math.isnan(target_mean):
        return ""
    sigma_ratio = pred_std / max(target_std, 1e-12) if not math.isnan(pred_std) and not math.isnan(target_std) else float("nan")
    bias = pred_mean - target_mean
    return " ".join([
        f"pred_mu={pred_mean:.4g}",
        f"pred_sigma={pred_std:.4g}" if not math.isnan(pred_std) else "pred_sigma=nan",
        f"bias={bias:.4g}",
        f"sigma_ratio={sigma_ratio:.3f}" if not math.isnan(sigma_ratio) else "sigma_ratio=nan",
        f"corr={corr:.3f}" if not math.isnan(corr) else "corr=nan",
    ])


def format_rollout_metric_bundle(metrics: Dict[str, float], *, stem: str) -> str:
    mse = metrics.get(f"rollout_{stem}_mse", float("nan"))
    r2 = metrics.get(f"rollout_{stem}_r2", float("nan"))
    nrmse = metrics.get(f"rollout_{stem}_nrmse", float("nan"))
    persistent_r2 = metrics.get(f"rollout_persistent_{stem}_r2", float("nan"))
    if math.isnan(mse):
        return ""
    parts = [
        f"mse={mse:.4g}",
        f"r2={r2:.3f}" if not math.isnan(r2) else "r2=nan",
        f"pers_r2={persistent_r2:.3f}" if not math.isnan(persistent_r2) else "pers_r2=nan",
        f"nrmse={nrmse:.3f}" if not math.isnan(nrmse) else "nrmse=nan",
    ]
    return " ".join(parts)


def format_node_metric(metrics: Dict[str, float]) -> str:
    metric_name = node_metric_name(metrics)
    if metric_name is None:
        return ""
    if metric_name == "node_mse":
        return format_node_metric_bundle(metrics)
    if metric_name in {"node_auroc", "node_f1"}:
        return format_node_classification_bundle(metrics)
    value = metrics.get(metric_name, float("nan"))
    return f"{metric_name}={value:.4f}"


def format_node_metric_bundle(metrics: Dict[str, float], *, prefix: Optional[str] = None) -> str:
    key_prefix = "" if prefix is None else f"{prefix}_"
    mse = metrics.get(f"{key_prefix}node_mse", float("nan"))
    r2 = metrics.get(f"{key_prefix}node_r2", float("nan"))
    nrmse = metrics.get(f"{key_prefix}node_nrmse", float("nan"))
    target_mean = metrics.get(f"{key_prefix}node_target_mean", float("nan"))
    target_std = metrics.get(f"{key_prefix}node_target_std", float("nan"))
    persistent_r2 = metrics.get("persistent_node_r2", float("nan")) if prefix is None else float("nan")
    if math.isnan(mse):
        return ""
    parts = [
        f"mse={mse:.4g}",
        f"r2={r2:.3f}" if not math.isnan(r2) else "r2=nan",
        f"nrmse={nrmse:.3f}" if not math.isnan(nrmse) else "nrmse=nan",
        f"mu={target_mean:.4g}" if not math.isnan(target_mean) else "mu=nan",
        f"sigma={target_std:.4g}" if not math.isnan(target_std) else "sigma=nan",
    ]
    if not math.isnan(persistent_r2):
        parts.append(f"pers_r2={persistent_r2:.3f}")
    return " ".join(parts)


def format_node_classification_bundle(
    metrics: Dict[str, float],
    *,
    prefix: Optional[str] = None,
) -> str:
    key_prefix = "" if prefix is None else f"{prefix}_"
    auroc = metrics.get(f"{key_prefix}node_auroc", float("nan"))
    auprc = metrics.get(f"{key_prefix}node_auprc", float("nan"))
    f1 = metrics.get(f"{key_prefix}node_f1", float("nan"))
    acc = metrics.get(f"{key_prefix}node_acc", float("nan"))
    precision = metrics.get(f"{key_prefix}node_precision", float("nan"))
    recall = metrics.get(f"{key_prefix}node_recall", float("nan"))
    if math.isnan(acc) and math.isnan(f1) and math.isnan(auroc):
        return ""
    parts = [
        f"acc={acc:.3f}" if not math.isnan(acc) else "acc=nan",
        f"f1={f1:.3f}" if not math.isnan(f1) else "f1=nan",
        f"auroc={auroc:.3f}" if not math.isnan(auroc) else "auroc=nan",
        f"auprc={auprc:.3f}" if not math.isnan(auprc) else "auprc=nan",
        f"prec={precision:.3f}" if not math.isnan(precision) else "prec=nan",
        f"rec={recall:.3f}" if not math.isnan(recall) else "rec=nan",
    ]
    return " ".join(parts)


def format_primary_metric(
    metrics: Dict[str, float],
    *,
    prefix: Optional[str] = None,
    name: Optional[str] = None,
) -> str:
    name = primary_metric_name(metrics) if name is None else name
    full_name = f"{prefix}_{name}" if prefix is not None else name
    value = metrics.get(full_name, float("nan"))
    if name in {"edge_mse", "node_mse"}:
        return f"{full_name}={value:.4g}"
    return f"{full_name}={value:.4f}"
