from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from interactiondynamics.train import (
    _normalize_ift_variant_args,
    _print_ift_diagnostic_footer,
    parse_args,
)
from interactiondynamics.training.runner import short_run_label_from_name
from interactiondynamics.training.types import RunResult, SweepRun
from interactiondynamics.core.config import ModelConfig


def test_parse_args_supports_subcommands_and_common_flags() -> None:
    args = parse_args(
        [
            "quick",
            "--dataset",
            "synthetic",
            "--synthetic-task",
            "diffusion",
            "--num-bins",
            "12",
            "--prediction-mode",
            "delta",
        ]
    )

    assert args.command == "quick"
    assert args.dataset == "synthetic"
    assert args.synthetic_task == "diffusion"
    assert args.num_bins == 12
    assert args.prediction_mode == "delta"


def test_parse_args_supports_ift_variant_selection() -> None:
    args = parse_args(
        [
            "quick",
            "--ift-variants",
            "linear",
            "auto",
            "--ift-orders",
            "2",
            "--ift-history-steps",
            "1",
            "3",
        ]
    )

    assert args.command == "quick"
    assert args.ift_variants == ["linear", "auto"]
    assert args.ift_orders == [2]
    assert args.ift_history_steps == [1, 3]
    _normalize_ift_variant_args(args)
    assert args.dataset == "synthetic"


def test_parse_args_expands_ift_variant_selection_without_dataset_override() -> None:
    args = parse_args(
        [
            "quick",
            "--ift-variants",
        ]
    )

    assert args.ift_variants == []
    _normalize_ift_variant_args(args)
    assert args.dataset == "synthetic"


def test_ift_variant_selection_rejects_non_synthetic_dataset_override() -> None:
    args = parse_args(
        [
            "quick",
            "--dataset",
            "toy",
            "--ift-variants",
            "generic",
        ]
    )

    with pytest.raises(ValueError, match="synthetic dataset"):
        _normalize_ift_variant_args(args)


def test_ift_variant_selection_rejects_sweep_preset() -> None:
    args = parse_args(
        [
            "sweep",
            "--ift-variants",
            "generic",
        ]
    )

    with pytest.raises(ValueError, match="smoke or quick"):
        _normalize_ift_variant_args(args)


@pytest.mark.parametrize(
    ("preset", "command"),
    [
        ("smoke", "smoke"),
        ("quick", "quick"),
        ("full", "sweep"),
    ],
)
def test_parse_args_maps_legacy_preset_aliases(preset: str, command: str) -> None:
    args = parse_args(["--preset", preset])
    assert args.command == command


def test_parse_args_rejects_removed_ift_diagnose_subcommand() -> None:
    with pytest.raises(SystemExit):
        parse_args(["ift-diagnose"])


def test_ift_history_steps_without_auto_still_parse_and_validate_later() -> None:
    args = parse_args(
        [
            "quick",
            "--ift-variants",
            "linear",
            "--ift-history-steps",
            "2",
        ]
    )

    with pytest.raises(ValueError, match="requires selecting the auto IFT variant"):
        _normalize_ift_variant_args(args)


def test_parse_args_requires_a_subcommand_or_legacy_preset() -> None:
    with pytest.raises(SystemExit):
        parse_args([])


def test_short_run_label_from_name_falls_back_to_custom_variant_name() -> None:
    assert short_run_label_from_name("ift2_hist_vel_k2") == "ift2_hist_vel_k2"


def test_short_run_label_uses_custom_variant_name_when_present() -> None:
    run = SweepRun(name="ift2_hist_vel_k2", model_cfg=ModelConfig(aggregator="ift", update="ift_update"))
    from interactiondynamics.training.runner import short_run_label

    assert short_run_label(run) == "ift2_hist_vel_k2"


def test_print_ift_diagnostic_footer_emits_table_for_variant_runs(capsys: pytest.CaptureFixture[str]) -> None:
    args = argparse.Namespace(
        ift_variants=["auto"],
        ift_orders=[2],
        ift_history_steps=[3],
        rollout_horizon=5,
    )
    spec = SimpleNamespace(extra={"synthetic_task": "wave"})
    runs = [
        SweepRun(
            name="ift2_auto",
            seed=0,
            model_cfg=ModelConfig(aggregator="ift", update="ift_update"),
        )
    ]
    results = [
        RunResult(
            name="ift2_auto",
            seed=0,
            epochs=1,
            best_val_loss=0.1,
            best_val_mrr=float("nan"),
            best_epoch=1,
            best_snapshot={
                "val": {"edge_r2": 0.5},
                "test": {"edge_r2": 0.4},
                "rollout_val": {"rollout_edge_r2": 0.3},
                "rollout_test": {
                    "rollout_edge_r2": 0.2,
                    "rollout_persistent_edge_r2": 0.1,
                    "rollout_edge_delta_r2": 0.05,
                    "rollout_edge_delta_mae": 0.02,
                },
                "train_step": {
                    "learned_kappa_mean": 1.2,
                    "gamma_mean": 0.1,
                    "dt_mean": 0.2,
                    "alpha_mean": 0.9,
                    "force_norm_mean": 0.8,
                    "diffusion_term_norm_mean": 0.7,
                    "relative_diffusion_mean": 0.6,
                    "relative_update_mean": 0.5,
                    "velocity_fraction_mean": 0.4,
                    "force_fraction_mean": 0.3,
                    "pred_delta_corr_mean": 0.25,
                    "internal_velocity_r2_mean": 0.15,
                    "internal_velocity_mse_mean": 0.12,
                },
                "readout": {"w_y": 1.0, "w_v": 2.0, "w_drive": 3.0, "bias": 4.0},
            },
            final_snapshot={},
            wall_sec=0.0,
        )
    ]

    _print_ift_diagnostic_footer(args, ds=object(), spec=spec, runs=runs, results=results)

    out = capsys.readouterr().out
    assert "IFT diagnostic table: wave" in out
    assert "ift2_auto" in out
    assert "coeffs | w_y=1.0000 w_v=2.0000 w_drive=3.0000 bias=4.0000" in out


def test_print_ift_diagnostic_footer_uses_node_metrics_for_node_regression(
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = argparse.Namespace(
        ift_variants=["generic"],
        ift_orders=[1],
        ift_history_steps=None,
        rollout_horizon=5,
    )
    spec = SimpleNamespace(extra={"synthetic_task": "node_temporal_regression"})
    runs = [
        SweepRun(
            name="ift1_generic",
            seed=0,
            model_cfg=ModelConfig(aggregator="ift", update="ift_update"),
        )
    ]
    results = [
        RunResult(
            name="ift1_generic",
            seed=0,
            epochs=1,
            best_val_loss=0.1,
            best_val_mrr=float("nan"),
            best_epoch=1,
            best_snapshot={
                "val": {"node_r2": 0.61, "persistent_node_r2": 0.42},
                "test": {"node_r2": 0.57},
                "rollout_val": {"rollout_node_r2": 0.48},
                "rollout_test": {
                    "rollout_node_r2": 0.44,
                    "rollout_persistent_node_r2": 0.42,
                },
                "train_step": {
                    "learned_kappa_mean": 1.2,
                    "gamma_mean": 0.1,
                    "dt_mean": 0.2,
                    "alpha_mean": 0.9,
                    "force_norm_mean": 0.8,
                    "diffusion_term_norm_mean": 0.7,
                    "relative_diffusion_mean": 0.6,
                    "relative_update_mean": 0.5,
                    "velocity_fraction_mean": 0.4,
                    "force_fraction_mean": 0.3,
                    "pred_delta_corr_mean": 0.25,
                    "internal_velocity_r2_mean": 0.15,
                    "internal_velocity_mse_mean": 0.12,
                },
                "readout": {},
            },
            final_snapshot={},
            wall_sec=0.0,
        )
    ]

    _print_ift_diagnostic_footer(args, ds=object(), spec=spec, runs=runs, results=results)

    out = capsys.readouterr().out
    assert "IFT diagnostic table: node_temporal_regression" in out
    assert "ift1_generic" in out
    assert "node" in out
    assert "0.610" in out
    assert "0.570" in out
    assert "0.440" in out
    assert "0.420" in out
    assert "node rollout delta metrics are not currently tracked" in out
    assert " nan" not in out.lower()


def test_print_ift_diagnostic_footer_keeps_stub_for_non_regression_targets(
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = argparse.Namespace(
        ift_variants=["generic"],
        ift_orders=[1],
        ift_history_steps=None,
        rollout_horizon=5,
    )
    spec = SimpleNamespace(extra={"synthetic_task": "node_temporal_state"})
    runs = [
        SweepRun(
            name="ift1_generic",
            seed=0,
            model_cfg=ModelConfig(aggregator="ift", update="ift_update"),
        )
    ]
    results = [
        RunResult(
            name="ift1_generic",
            seed=0,
            epochs=1,
            best_val_loss=0.1,
            best_val_mrr=float("nan"),
            best_epoch=1,
            best_snapshot={
                "val": {"node_auroc": 0.73, "node_f1": 0.64},
                "test": {"node_auroc": 0.68, "node_f1": 0.59},
                "rollout_val": {},
                "rollout_test": {},
                "train_step": {
                    "learned_kappa_mean": 1.2,
                    "gamma_mean": 0.1,
                    "dt_mean": 0.2,
                    "alpha_mean": 0.9,
                    "force_norm_mean": 0.8,
                    "diffusion_term_norm_mean": 0.7,
                    "relative_diffusion_mean": 0.6,
                    "relative_update_mean": 0.5,
                },
                "readout": {},
            },
            final_snapshot={},
            wall_sec=0.0,
        )
    ]

    _print_ift_diagnostic_footer(args, ds=object(), spec=spec, runs=runs, results=results)

    out = capsys.readouterr().out
    assert "IFT diagnostic table: node_temporal_state" in out
    assert "ift1_generic" in out
    assert "node" in out
    assert "0.730" in out
    assert "0.680" in out
    assert "0.640" in out
    assert "0.590" in out
    assert "regression-style state and rollout metrics are unavailable" in out


def test_print_ift_diagnostic_footer_includes_edge_classification_metrics(
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = argparse.Namespace(
        ift_variants=["generic"],
        ift_orders=[1],
        ift_history_steps=None,
        rollout_horizon=5,
    )
    spec = SimpleNamespace(extra={"synthetic_task": "edge_temporal_state"})
    runs = [
        SweepRun(
            name="ift1_generic",
            seed=0,
            model_cfg=ModelConfig(aggregator="ift", update="ift_update"),
        )
    ]
    results = [
        RunResult(
            name="ift1_generic",
            seed=0,
            epochs=1,
            best_val_loss=0.1,
            best_val_mrr=float("nan"),
            best_epoch=1,
            best_snapshot={
                "val": {"edge_auroc": 0.81, "edge_f1": 0.72},
                "test": {"edge_auroc": 0.77, "edge_f1": 0.66},
                "rollout_val": {},
                "rollout_test": {},
                "train_step": {
                    "learned_kappa_mean": 1.2,
                    "gamma_mean": 0.1,
                    "dt_mean": 0.2,
                    "alpha_mean": 0.9,
                    "force_norm_mean": 0.8,
                    "diffusion_term_norm_mean": 0.7,
                    "relative_diffusion_mean": 0.6,
                    "relative_update_mean": 0.5,
                },
                "readout": {},
            },
            final_snapshot={},
            wall_sec=0.0,
        )
    ]

    _print_ift_diagnostic_footer(args, ds=object(), spec=spec, runs=runs, results=results)

    out = capsys.readouterr().out
    assert "IFT diagnostic table: edge_temporal_state" in out
    assert "ift1_generic" in out
    assert "edge" in out
    assert "0.810" in out
    assert "0.770" in out
    assert "0.720" in out
    assert "0.660" in out
