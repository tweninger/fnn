from __future__ import annotations

import json
from pathlib import Path

import torch

from datasets.corrupted import CorruptedEventStreamDataset
from eval.whole_bin_dummy_baselines import run_whole_bin_baseline_suite
from experiments.whole_bin_common import (
    DEFAULT_WHOLE_BIN_CORRUPTION,
    format_whole_bin_corruption_tag,
    build_thresholded_whole_bin_datasets,
    use_upper_triangle_for_dataset,
)
from utils.output_paths import experiment_output_paths

RESULTS_ROOT = Path("/home/akapociu/ift/interactiondynamics/results")
PLOTS_ROOT = Path("/home/akapociu/ift/interactiondynamics/plots")
EXPERIMENT_NAME = "whole_bin_dummy_baselines"

USE_CORRUPTED_CONTEXT = True
CORRUPTION = dict(DEFAULT_WHOLE_BIN_CORRUPTION)
CORRUPTION.update(corrupt_unit="node_block", block_node_select="high_degree")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    paths = experiment_output_paths(RESULTS_ROOT, PLOTS_ROOT, EXPERIMENT_NAME)
    results_dir = paths["results_dir"]
    baselines_jsonl = results_dir / "dummy_baselines_summary.jsonl"
    results_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name, clean_ds in build_thresholded_whole_bin_datasets(device).items():
        spec = clean_ds.spec()
        upper_triangle_only = use_upper_triangle_for_dataset(dataset_name)
        context_ds = (
            CorruptedEventStreamDataset(clean_ds, **CORRUPTION)
            if USE_CORRUPTED_CONTEXT
            else None
        )
        row_name = (
            f"{dataset_name}_{format_whole_bin_corruption_tag(CORRUPTION)}"
            if USE_CORRUPTED_CONTEXT
            else f"{dataset_name}_clean"
        )

        print("=" * 80)
        print(f"DATASET: {row_name}")

        rows = run_whole_bin_baseline_suite(
            clean_ds,
            context_ds=context_ds,
            num_nodes=spec.num_nodes,
            upper_triangle_only=upper_triangle_only,
            include_self_loops=False,
            seed=0,
            device=device,
        )

        for row in rows:
            row["dataset"] = row_name
            if USE_CORRUPTED_CONTEXT:
                row["context_corruption"] = CORRUPTION
            append_jsonl(baselines_jsonl, row)
            if row["split"] == "train":
                print(
                    f"{row['baseline']:>22s} | test f1={row.get('micro_f1', row.get('f1', float('nan'))):.4f} | "
                    f"jaccard={row.get('micro_jaccard', row.get('jaccard', float('nan'))):.4f}"
                )

        print(f"Saved: {baselines_jsonl}")


if __name__ == "__main__":
    main()
