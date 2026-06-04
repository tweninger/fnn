from __future__ import annotations

import argparse

import pytest
import torch

from interactiondynamics.data.synthetic import SYNTHETIC_TASKS
from interactiondynamics.training.presets import build_suite, load_dataset


def _synthetic_args(task_name: str) -> argparse.Namespace:
    return argparse.Namespace(
        dataset="synthetic",
        synthetic_task=task_name,
        synthetic_num_nodes=20,
        synthetic_events_per_bin=30,
        num_bins=18,
        seed=3,
    )


@pytest.mark.parametrize("task_name", list(SYNTHETIC_TASKS.keys()))
def test_quick_synthetic_suite_uses_task_specific_shortlist(task_name: str):
    args = _synthetic_args(task_name)
    suite = build_suite(
        "quick",
        torch.device("cpu"),
        dataset_override="synthetic",
        args=args,
    )

    assert suite.dataset == "synthetic"
    assert suite.dataset_kwargs["task"] == task_name

    task_pairs = {(run.model_cfg.aggregator, run.model_cfg.update) for run in suite.runs}
    assert task_pairs == set(SYNTHETIC_TASKS[task_name].recommended_pairs)
    if SYNTHETIC_TASKS[task_name].requires_node_scorer:
        assert all(run.model_cfg.use_node_scorer for run in suite.runs)

    ds = load_dataset(suite.dataset, suite.dataset_kwargs)
    assert ds.spec().num_nodes == 20
