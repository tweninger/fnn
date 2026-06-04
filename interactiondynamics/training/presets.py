from __future__ import annotations

import argparse
import itertools
from dataclasses import asdict
from typing import Any, Dict, Optional, Sequence, cast

import torch

from interactiondynamics.core.config import ModelConfig
from interactiondynamics.data.jodie import JODIEBinnedDataset, JODIEConfig  # type: ignore
from interactiondynamics.data.synthetic import SYNTHETIC_TASKS, SyntheticDataset, SyntheticDatasetConfig
from interactiondynamics.data.toy import ToyShiftConfig, ToyShiftDataset
from interactiondynamics.eval.evaluate import EvalSlices
from interactiondynamics.training.types import PredictionMode, RunSuite, SweepRun, TrainConfig


FOCUSED_COMBINATIONS = {
    ("ift", "ift_update"),
    ("hopfield", "hopfield_update"),
    ("settransformer", "lnn"),
    ("settransformer", "hnn"),
    ("settransformer", "tgn_gru"),
}


def build_ift_diagnostic_runs(
    base_model_cfg: ModelConfig,
    *,
    task_name: str,
    drive_feature_idx: Optional[int],
    near_ar1_delta_coeffs: Optional[tuple[float, float, float, float]] = None,
    seed: int = 0,
) -> list[SweepRun]:
    runs: list[SweepRun] = []

    def add(
        name: str,
        *,
        lr: Optional[float] = None,
        prediction_mode: Optional[PredictionMode] = None,
        **overrides: Any,
    ) -> None:
        cfg = ModelConfig(**asdict(base_model_cfg))
        for key, value in overrides.items():
            setattr(cfg, key, value)
        if near_ar1_delta_coeffs is not None:
            setattr(cfg, "ift2_near_ar1_coeffs", near_ar1_delta_coeffs)
        runs.append(
            SweepRun(
                name=name,
                model_cfg=cfg,
                lr=lr,
                prediction_mode=prediction_mode,
                seed=int(seed),
            )
        )

    shared: dict[str, Any] = dict(
        aggregator="ift",
        update="ift_update",
        ift_laplacian_mode="current_bin",
        ift_message_reduce="sum",
        ift_force_reduce="sum",
        ift_drive_feature_idx=drive_feature_idx,
    )
    if task_name == "ift_diffusion":
        add("ift1_generic", **shared, ift_update_order="first", ift_forcing_mode="generic_mlp")
        add("ift1_linear", **shared, ift_update_order="first", ift_forcing_mode="linear_event")
        add("ift1_direct", **shared, ift_update_order="first", ift_forcing_mode="direct_scalar")
        add("ift1_gated_direct", **shared, ift_update_order="first", ift_forcing_mode="gated_direct_scalar")
        add("gru_baseline", aggregator="sum", update="tgn_gru")
        return runs
    if task_name == "ift_wave":
        add("ift1_generic", **shared, ift_update_order="first", ift_forcing_mode="generic_mlp")
        add("ift1_linear", **shared, ift_update_order="first", ift_forcing_mode="linear_event")
        add("ift1_direct", **shared, ift_update_order="first", ift_forcing_mode="direct_scalar")
        add("ift2_auto", **shared, ift_update_order="second", ift_forcing_mode="linear_event")
        for history_steps in (1, 2, 3):
            add(
                f"ift2_hist_vel_k{history_steps}",
                **shared,
                ift_update_order="second",
                ift_forcing_mode="linear_event",
                ift2_readout_mode="linear_h_v_force",
                ift_velocity_teacher_forcing=False,
                ift_history_vel_steps=history_steps,
                ift2_readout_init_mode="small_random",
                ift2_readout_init_scale=0.01,
                lr=1e-2,
                prediction_mode=cast(PredictionMode, "delta"),
            )
        if drive_feature_idx is not None:
            add(
                "ift2_ar_tf",
                **shared,
                ift_update_order="second",
                ift_forcing_mode="linear_event",
                ift2_readout_mode="linear_h_v_force",
                ift_velocity_teacher_forcing=True,
                ift2_readout_init_mode="small_random",
                ift2_readout_init_scale=0.01,
                lr=1e-2,
                prediction_mode=cast(PredictionMode, "delta"),
            )
        add("gru_baseline", aggregator="sum", update="tgn_gru")
        return runs

    add("ift1_generic", **shared, ift_update_order="first", ift_forcing_mode="generic_mlp")
    add("ift1_direct", **shared, ift_update_order="first", ift_forcing_mode="direct_scalar")
    add("ift2_generic", **shared, ift_update_order="second", ift_forcing_mode="generic_mlp")
    add("ift2_linear", **shared, ift_update_order="second", ift_forcing_mode="linear_event")
    add("ift2_gated_linear", **shared, ift_update_order="second", ift_forcing_mode="gated_linear_event")
    add("ift2_direct", **shared, ift_update_order="second", ift_forcing_mode="direct_scalar")
    add("ift2_gated_direct", **shared, ift_update_order="second", ift_forcing_mode="gated_direct_scalar")
    add(
        "neural_ar2_delta_small_random_lr1e-1",
        **shared,
        ift_update_order="second",
        ift_forcing_mode="direct_scalar",
        ift2_readout_mode="linear_h_v_force",
        ift_velocity_teacher_forcing=True,
        ift2_readout_init_mode="small_random",
        ift2_readout_init_scale=0.01,
        lr=1e-1,
        prediction_mode=cast(PredictionMode, "delta"),
    )
    add(
        "neural_ar2_delta_small_random_lr3e-2",
        **shared,
        ift_update_order="second",
        ift_forcing_mode="direct_scalar",
        ift2_readout_mode="linear_h_v_force",
        ift_velocity_teacher_forcing=True,
        ift2_readout_init_mode="small_random",
        ift2_readout_init_scale=0.01,
        lr=3e-2,
        prediction_mode=cast(PredictionMode, "delta"),
    )
    add(
        "neural_ar2_delta_small_random_lr1e-2",
        **shared,
        ift_update_order="second",
        ift_forcing_mode="direct_scalar",
        ift2_readout_mode="linear_h_v_force",
        ift_velocity_teacher_forcing=True,
        ift2_readout_init_mode="small_random",
        ift2_readout_init_scale=0.01,
        lr=1e-2,
        prediction_mode=cast(PredictionMode, "delta"),
    )
    add(
        "neural_ar2_delta_small_random_lr3e-3",
        **shared,
        ift_update_order="second",
        ift_forcing_mode="direct_scalar",
        ift2_readout_mode="linear_h_v_force",
        ift_velocity_teacher_forcing=True,
        ift2_readout_init_mode="small_random",
        ift2_readout_init_scale=0.01,
        lr=3e-3,
        prediction_mode=cast(PredictionMode, "delta"),
    )
    add(
        "neural_ar2_delta_near_ar1",
        **shared,
        ift_update_order="second",
        ift_forcing_mode="direct_scalar",
        ift2_readout_mode="linear_h_v_force",
        ift_velocity_teacher_forcing=True,
        ift2_readout_init_mode="near_ar1",
        lr=1e-2,
        prediction_mode=cast(PredictionMode, "delta"),
    )
    add(
        "neural_ar2_delta_oracle_init_trainable",
        **shared,
        ift_update_order="second",
        ift_forcing_mode="direct_scalar",
        ift2_readout_mode="linear_h_v_force",
        ift_velocity_teacher_forcing=True,
        ift2_oracle_init=True,
        ift2_readout_init_mode="oracle",
        ift2_readout_trainable=True,
        lr=1e-2,
        prediction_mode=cast(PredictionMode, "delta"),
    )
    add(
        "ift2_linear_h_v_force_teacher_forced",
        **shared,
        ift_update_order="second",
        ift_forcing_mode="direct_scalar",
        ift2_readout_mode="linear_h_v_force",
        ift_velocity_teacher_forcing=True,
        ift2_readout_init_mode="small_random",
        ift2_readout_init_scale=0.01,
        lr=1e-2,
        prediction_mode=cast(PredictionMode, "delta"),
    )
    add(
        "ift2_linear_h_v_force_autonomous",
        **shared,
        ift_update_order="second",
        ift_forcing_mode="direct_scalar",
        ift2_readout_mode="linear_h_v_force",
        ift_velocity_teacher_forcing=False,
        ift2_readout_init_mode="small_random",
        ift2_readout_init_scale=0.01,
        lr=1e-2,
        prediction_mode=cast(PredictionMode, "delta"),
    )
    for lambda_v in (0.0, 0.01, 0.1, 1.0):
        add(
            f"ift2_ar_auto_lambda{str(lambda_v).replace('.', 'p')}",
            **shared,
            ift_update_order="second",
            ift_forcing_mode="direct_scalar",
            ift2_readout_mode="linear_h_v_force",
            ift_velocity_teacher_forcing=False,
            ift2_readout_init_mode="small_random",
            ift2_readout_init_scale=0.01,
            ift_internal_velocity_loss_weight=float(lambda_v),
            lr=1e-2,
            prediction_mode=cast(PredictionMode, "delta"),
        )
    add(
        "ift2_ar_tf",
        **shared,
        ift_update_order="second",
        ift_forcing_mode="direct_scalar",
        ift2_readout_mode="linear_h_v_force",
        ift_velocity_teacher_forcing=True,
        ift2_readout_init_mode="small_random",
        ift2_readout_init_scale=0.01,
        lr=1e-2,
        prediction_mode=cast(PredictionMode, "delta"),
    )
    add(
        "ift2_ar_auto",
        **shared,
        ift_update_order="second",
        ift_forcing_mode="direct_scalar",
        ift2_readout_mode="linear_h_v_force",
        ift_velocity_teacher_forcing=False,
        ift2_readout_init_mode="small_random",
        ift2_readout_init_scale=0.01,
        lr=1e-2,
        prediction_mode=cast(PredictionMode, "delta"),
    )
    for history_steps in (1, 2, 3):
        add(
            f"ift2_hist_vel_k{history_steps}",
            **shared,
            ift_update_order="second",
            ift_forcing_mode="direct_scalar",
            ift2_readout_mode="linear_h_v_force",
            ift_velocity_teacher_forcing=False,
            ift_history_vel_steps=history_steps,
            ift2_readout_init_mode="small_random",
            ift2_readout_init_scale=0.01,
            lr=1e-2,
            prediction_mode=cast(PredictionMode, "delta"),
        )
    add(
        "neural_ar2_delta",
        **shared,
        ift_update_order="second",
        ift_forcing_mode="direct_scalar",
        ift2_readout_mode="linear_h_v_force",
        ift_velocity_teacher_forcing=True,
        ift2_readout_init_mode="zero",
        lr=1e-2,
        prediction_mode=cast(PredictionMode, "delta"),
    )
    add(
        "ift2_ar2_oracle_init",
        **shared,
        ift_update_order="second",
        ift_forcing_mode="direct_scalar",
        ift2_readout_mode="linear_h_v_force",
        ift_velocity_teacher_forcing=True,
        ift2_oracle_init=True,
        ift2_readout_init_mode="oracle",
        ift2_readout_trainable=False,
        prediction_mode=cast(PredictionMode, "delta"),
    )
    add("gru_baseline", aggregator="sum", update="tgn_gru")
    add("hnn_baseline", aggregator="sum", update="hnn")
    return runs


def make_runs(
    base_model_cfg: ModelConfig,
    *,
    seeds: Sequence[int] = (0,),
    aggregator: Sequence[str] = ("sum", "deepsets", "settransformer"),
    upd: Sequence[str] = ("tgn_gru",),
    dropout: Sequence[float] = (0.0,),
    scorer_dropout: Sequence[float] = (0.0,),
    use_time_features: Sequence[bool] = (False,),
    ift_kappa_param: Sequence[str] = ("softplus", "exp"),
    ift_dt: Sequence[float] = (0.05,),
    ift_gamma: Sequence[float] = (0.0,),
    ift_kappa_init: Sequence[float] = (1.0,),
    ift_kappa_cap: Sequence[bool] = (False,),
    ift_kappa_max: Sequence[float | None] = (None,),
) -> list[SweepRun]:
    runs: list[SweepRun] = []
    for agg, do, update_name, sdo, time_features, seed in itertools.product(
        aggregator, dropout, upd, scorer_dropout, use_time_features, seeds
    ):
        cfg = ModelConfig(**asdict(base_model_cfg))
        cfg.update = update_name  # type: ignore
        cfg.aggregator = agg  # type: ignore
        cfg.dropout = do
        cfg.scorer_dropout = sdo
        cfg.use_time_features = time_features
        if agg == "ift":
            cfg.ift_message_reduce = "sum"
            cfg.ift_force_reduce = "sum"

        base_name = f"agg={agg}|update={update_name}|do={do}|sdo={sdo}|time={time_features}"

        if update_name == "ift_update":
            for kp, dt, gamma, k0 in itertools.product(
                ift_kappa_param, ift_dt, ift_gamma, ift_kappa_init
            ):
                for cap in ift_kappa_cap:
                    if cap:
                        for kmax in ift_kappa_max:
                            if kmax is None:
                                continue
                            cfg2 = ModelConfig(**asdict(cfg))
                            cfg2.ift_kappa_param = kp  # type: ignore
                            cfg2.ift_dt = float(dt)
                            cfg2.ift_gamma = float(gamma)
                            cfg2.ift_kappa = float(k0)
                            cfg2.ift_kappa_cap = True
                            cfg2.ift_kappa_max = float(kmax)
                            name2 = f"{base_name}|{kp}|dt={dt}|gamma={gamma}|k0={k0}|cap={kmax}"
                            runs.append(SweepRun(name=name2, model_cfg=cfg2, seed=int(seed)))
                    else:
                        cfg2 = ModelConfig(**asdict(cfg))
                        cfg2.ift_kappa_param = kp  # type: ignore
                        cfg2.ift_dt = float(dt)
                        cfg2.ift_gamma = float(gamma)
                        cfg2.ift_kappa = float(k0)
                        cfg2.ift_kappa_cap = False
                        cfg2.ift_kappa_max = None
                        name2 = f"{base_name}|{kp}|dt={dt}|gamma={gamma}|k0={k0}|cap=none"
                        runs.append(SweepRun(name=name2, model_cfg=cfg2, seed=int(seed)))
            continue

        runs.append(SweepRun(name=base_name, model_cfg=cfg, seed=int(seed)))
    return runs


def select_runs(runs: Sequence[SweepRun], allowed: set[tuple[str, str]]) -> list[SweepRun]:
    return [
        run
        for run in runs
        if (run.model_cfg.aggregator, run.model_cfg.update) in allowed
    ]


def _default_train_config(
    *,
    num_nodes: int,
    num_neg: int,
    log_every: int,
    device: torch.device,
) -> TrainConfig:
    return TrainConfig(
        num_nodes=num_nodes,
        num_neg=num_neg,
        tbptt_steps=1,
        log_every=log_every,
        device=device,
        weight_decay=1e-3,
        lr=1e-3,
    )


def _base_model_config(*, small: bool) -> ModelConfig:
    if small:
        return ModelConfig(
            node_dim=64,
            msg_dim=64,
            event_dim=0,
            scorer="mlp",
            scorer_hidden=128,
            aggregator="sum",
            use_time_features=False,
            dropout=0.0,
            scorer_dropout=0.0,
            encoder_hidden=128,
        )
    return ModelConfig(
        node_dim=128,
        msg_dim=128,
        event_dim=0,
        scorer="mlp",
        scorer_hidden=256,
        aggregator="sum",
        use_time_features=False,
        dropout=0.0,
        scorer_dropout=0.0,
        encoder_hidden=256,
    )


def _focused_runs(model_cfg: ModelConfig) -> list[SweepRun]:
    return select_runs(
        make_runs(
            model_cfg,
            seeds=(0,),
            aggregator=("ift", "hopfield", "settransformer"),
            upd=("ift_update", "hopfield_update", "lnn", "hnn", "tgn_gru"),
            dropout=(0.0,),
            scorer_dropout=(0.0,),
            use_time_features=(False,),
            ift_kappa_param=("softplus",),
            ift_dt=(0.05,),
            ift_gamma=(0.0,),
            ift_kappa_init=(1.0,),
            ift_kappa_cap=(False,),
            ift_kappa_max=(None,),
        ),
        FOCUSED_COMBINATIONS,
    )


def _full_runs(model_cfg: ModelConfig) -> list[SweepRun]:
    return make_runs(
        model_cfg,
        seeds=(0, 42, 123),
        aggregator=("ift", "hopfield", "settransformer", "sum", "deepsets"),
        upd=("ift_update", "tgn_gru", "lnn", "hopfield_update", "hnn"),
        dropout=(0.0, 0.1),
        scorer_dropout=(0.0, 0.1),
        use_time_features=(False, True),
        ift_kappa_param=("softplus", "exp"),
        ift_dt=(0.01, 0.05, 0.1, 0.2),
        ift_gamma=(0.0, 0.01, 0.05, 0.1),
        ift_kappa_init=(0.1, 0.5, 1.0, 2.0),
        ift_kappa_cap=(False, True),
        ift_kappa_max=(1.0, 2.0, 5.0, None),
    )


def _synthetic_default_sizes(preset: str) -> tuple[int, int, int]:
    if preset == "smoke":
        return (32, 24, 64)
    if preset == "quick":
        return (64, 72, 192)
    return (96, 160, 320)


def build_synthetic_dataset_config(
    args: argparse.Namespace,
    device: torch.device,
    *,
    preset: str,
) -> SyntheticDatasetConfig:
    num_nodes, default_bins, default_events = _synthetic_default_sizes(preset)
    dataset_num_bins = int(default_bins if args.num_bins is None else args.num_bins)
    dataset_num_nodes = int(
        num_nodes if args.synthetic_num_nodes is None else args.synthetic_num_nodes
    )
    dataset_events = int(
        default_events
        if args.synthetic_events_per_bin is None
        else args.synthetic_events_per_bin
    )
    task = str(args.synthetic_task)
    return SyntheticDatasetConfig(
        name=f"synthetic_{task}_{preset}",
        task=task,
        num_nodes=dataset_num_nodes,
        num_bins=dataset_num_bins,
        events_per_bin=dataset_events,
        seed=int(args.seed),
        device=device,
    )


def _build_synthetic_suite(
    preset: str,
    device: torch.device,
    args: argparse.Namespace,
) -> RunSuite:
    synthetic_cfg = build_synthetic_dataset_config(args, device, preset=preset)
    task_spec = SYNTHETIC_TASKS[synthetic_cfg.task]
    small = preset == "smoke"
    train_cfg = _default_train_config(
        num_nodes=0,
        num_neg=5 if small else 10,
        log_every=100 if small else 200,
        device=device,
    )
    model_cfg = _base_model_config(small=small)
    model_cfg.event_dim = task_spec.event_dim
    model_cfg.use_node_scorer = task_spec.requires_node_scorer

    if preset in {"smoke", "quick"}:
        shortlist_runs = make_runs(
            model_cfg,
            seeds=(0,),
            aggregator=("ift", "hopfield", "settransformer", "sum", "deepsets"),
            upd=("ift_update", "tgn_gru", "lnn", "hopfield_update", "hnn"),
            dropout=(0.0,),
            scorer_dropout=(0.0,),
            use_time_features=(False,),
            ift_kappa_param=("softplus",),
            ift_dt=(0.05,),
            ift_gamma=(0.0,),
            ift_kappa_init=(1.0,),
            ift_kappa_cap=(False,),
            ift_kappa_max=(None,),
        )
        runs = select_runs(shortlist_runs, set(task_spec.recommended_pairs))
        epochs = 1 if preset == "smoke" else 4
        early_steps = 5 if preset == "smoke" else 10
    else:
        runs = _full_runs(model_cfg)
        epochs = 8
        early_steps = 10

    return RunSuite(
        dataset="synthetic",
        dataset_kwargs=asdict(synthetic_cfg),
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        runs=runs,
        epochs=epochs,
        eval_slices=EvalSlices(early_steps=early_steps),
        save_jsonl_path=None,
    )


def build_suite(
    preset: str,
    device: torch.device,
    dataset_override: Optional[str] = None,
    args: Optional[argparse.Namespace] = None,
) -> RunSuite:
    if dataset_override == "synthetic" and args is not None:
        return _build_synthetic_suite(preset, device, args)

    if preset == "smoke":
        toy_cfg = ToyShiftConfig(
            name="toy_smoke",
            num_nodes=64,
            num_bins=24,
            events_per_bin=32,
            shift=7,
            device=device,
        )
        train_cfg = _default_train_config(
            num_nodes=toy_cfg.num_nodes,
            num_neg=5,
            log_every=100,
            device=device,
        )
        model_cfg = _base_model_config(small=True)
        return RunSuite(
            dataset="toy",
            dataset_kwargs=asdict(toy_cfg),
            train_cfg=train_cfg,
            model_cfg=model_cfg,
            runs=_focused_runs(model_cfg),
            epochs=1,
            eval_slices=EvalSlices(early_steps=5),
            save_jsonl_path=None,
        )

    if preset == "quick":
        if dataset_override == "toy":
            toy_cfg = ToyShiftConfig(
                name="toy_quick",
                num_nodes=128,
                num_bins=48,
                events_per_bin=64,
                shift=7,
                device=device,
            )
            train_cfg = _default_train_config(num_nodes=0, num_neg=10, log_every=250, device=device)
            model_cfg = _base_model_config(small=False)
            return RunSuite(
                dataset="toy",
                dataset_kwargs=asdict(toy_cfg),
                train_cfg=train_cfg,
                model_cfg=model_cfg,
                runs=_focused_runs(model_cfg),
                epochs=2,
                eval_slices=EvalSlices(early_steps=10),
                save_jsonl_path=None,
            )

        jodie_cfg = JODIEConfig(root="./data/JODIE", name="Wikipedia", device=device)
        train_cfg = _default_train_config(num_nodes=0, num_neg=10, log_every=500, device=device)
        model_cfg = _base_model_config(small=False)
        model_cfg.event_dim = None
        return RunSuite(
            dataset="jodie",
            dataset_kwargs=asdict(jodie_cfg),
            train_cfg=train_cfg,
            model_cfg=model_cfg,
            runs=_focused_runs(model_cfg),
            epochs=2,
            eval_slices=EvalSlices(early_steps=10),
            save_jsonl_path=None,
        )

    jodie_cfg = JODIEConfig(root="./data/JODIE", name="Wikipedia", device=device)
    train_cfg = _default_train_config(num_nodes=0, num_neg=20, log_every=2000, device=device)
    model_cfg = _base_model_config(small=False)
    model_cfg.event_dim = None
    return RunSuite(
        dataset="jodie",
        dataset_kwargs=asdict(jodie_cfg),
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        runs=_full_runs(model_cfg),
        epochs=6,
        eval_slices=EvalSlices(early_steps=10),
        save_jsonl_path=None,
    )


def load_dataset(kind: str, dataset_kwargs: Dict[str, Any]):
    if kind == "toy":
        return ToyShiftDataset(ToyShiftConfig(**dataset_kwargs))
    if kind == "jodie":
        return JODIEBinnedDataset(JODIEConfig(**dataset_kwargs))
    if kind == "synthetic":
        return SyntheticDataset(SyntheticDatasetConfig(**dataset_kwargs))
    raise ValueError(f"Unsupported dataset kind: {kind}")
