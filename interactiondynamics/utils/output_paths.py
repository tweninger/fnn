from __future__ import annotations

from pathlib import Path
from utils.io import make_safe_name


def dataset_group_name(dataset_name: str) -> str:
    if dataset_name.startswith("springweb"):
        return "spring_web_2d"
    if dataset_name.startswith("wave"):
        return "wave"
    return dataset_name


def experiment_output_paths(results_root: Path, plots_root: Path, experiment_name: str) -> dict[str, Path]:
    experiment_name = make_safe_name(experiment_name)

    results_dir = results_root / experiment_name
    plots_dir = plots_root / experiment_name
    preds_dir = results_dir / "predictions"

    return {
        "results_dir": results_dir,
        "plots_dir": plots_dir,
        "preds_dir": preds_dir,
        "results_jsonl": results_dir / "results.jsonl",
        "summary_jsonl": results_dir / "summary.jsonl",
    }


def prediction_npz_path(preds_dir: Path, dataset_name: str, run_name: str, seed: int) -> Path:
    safe_dataset = make_safe_name(dataset_name)
    safe_run = make_safe_name(run_name)
    return preds_dir / f"{safe_dataset}__{safe_run}__seed-{seed}.npz"


def run_plot_path(
    plots_dir: Path,
    dataset_name: str,
    run_name: str,
    seed: int,
    best_epoch: int | None = None,
) -> Path:
    safe_dataset = make_safe_name(dataset_name)
    safe_run = make_safe_name(run_name)

    dataset_dir = plots_dir / dataset_group_name(dataset_name)
    if best_epoch is None:
        return dataset_dir / f"{safe_dataset}__{safe_run}__seed-{seed}.png"
    return dataset_dir / f"{safe_dataset}__{safe_run}__seed-{seed}__epoch-{best_epoch}.png"