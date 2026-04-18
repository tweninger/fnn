from __future__ import annotations

from collections import defaultdict

import numpy as np

def edge_set(batch):
    return set(zip(batch.src.cpu().tolist(), batch.dst.cpu().tolist()))


def summarize_edge_changes(ds, split="train"):
    bins = list(ds.bins(split))
    edge_sets = [edge_set(batch) for batch in bins]

    if len(edge_sets) < 2:
        return {
            "num_transitions": 0,
            "mean_jaccard": None,
            "median_jaccard": None,
            "min_jaccard": None,
            "max_jaccard": None,
            "mean_edges_added": None,
            "mean_edges_removed": None,
        }

    jaccards = []
    added_counts = []
    removed_counts = []

    for prev, curr in zip(edge_sets[:-1], edge_sets[1:]):
        inter = len(prev & curr)
        union = len(prev | curr)
        jaccard = inter / union if union > 0 else 1.0

        added = len(curr - prev)
        removed = len(prev - curr)

        jaccards.append(jaccard)
        added_counts.append(added)
        removed_counts.append(removed)

    jaccards = np.array(jaccards, dtype=float)
    added_counts = np.array(added_counts, dtype=float)
    removed_counts = np.array(removed_counts, dtype=float)

    return {
        "num_transitions": int(len(jaccards)),
        "mean_jaccard": float(jaccards.mean()),
        "median_jaccard": float(np.median(jaccards)),
        "min_jaccard": float(jaccards.min()),
        "max_jaccard": float(jaccards.max()),
        "mean_edges_added": float(added_counts.mean()),
        "mean_edges_removed": float(removed_counts.mean()),
    }


def summarize_event_counts(ds, split="train"):
    counts = np.array([int(batch.src.numel()) for batch in ds.bins(split)], dtype=float)

    if counts.size == 0:
        return {
            "num_bins": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "percentiles": {},
        }

    return {
        "num_bins": int(len(counts)),
        "min": int(counts.min()),
        "max": int(counts.max()),
        "mean": float(counts.mean()),
        "median": float(np.median(counts)),
        "std": float(counts.std()),
        "percentiles": {
            "p0": float(np.percentile(counts, 0)),
            "p5": float(np.percentile(counts, 5)),
            "p10": float(np.percentile(counts, 10)),
            "p25": float(np.percentile(counts, 25)),
            "p50": float(np.percentile(counts, 50)),
            "p75": float(np.percentile(counts, 75)),
            "p90": float(np.percentile(counts, 90)),
            "p95": float(np.percentile(counts, 95)),
            "p100": float(np.percentile(counts, 100)),
        },
    }


def build_dataset_metadata_row(dataset_name, ds):
    spec = ds.spec()

    return {
        "dataset": dataset_name,
        "spec": {
            "name": spec.name,
            "num_nodes": int(spec.num_nodes),
            "event_dim": int(spec.event_dim),
            "num_events": None if spec.num_events is None else int(spec.num_events),
            "num_bins": None if spec.num_bins is None else int(spec.num_bins),
            "extra": spec.extra if spec.extra is not None else {},
        },
        "splits": {
            "train": {
                "event_counts": summarize_event_counts(ds, split="train"),
                "edge_changes": summarize_edge_changes(ds, split="train"),
            },
            "val": {
                "event_counts": summarize_event_counts(ds, split="val"),
                "edge_changes": summarize_edge_changes(ds, split="val"),
            },
            "test": {
                "event_counts": summarize_event_counts(ds, split="test"),
                "edge_changes": summarize_edge_changes(ds, split="test"),
            },
        },
    }