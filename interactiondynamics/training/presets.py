from __future__ import annotations

import argparse
import itertools
from dataclasses import asdict, replace
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
IFT_RUN_PAIR = ("ift", "ift_update")
IFT_VARIANT_CHOICES = ("generic", "linear", "direct", "auto")
IFT_ORDER_CHOICES = (1, 2)
IFT_HISTORY_STEP_CHOICES = (1, 2, 3)
GRID_WAVE_TASKS = {"wave_grid", "wave_torus", "wave_doorway", "wave_swisscheese"}
TOPOLOGY_SELECTABLE_FIELD_TASKS = {"diffusion", "wave", "coupled_oscillator"}
# Each baseline retains its intended architecture pairing.  In particular,
# Hopfield aggregation is evaluated with its Hopfield update, while LNN/HNN
# use the Set Transformer event encoder rather than arbitrary hybrid pairs.
FIELD_COMPARISON_PANEL = (
    ("sum", "tgn_gru"),
    ("deepsets", "tgn_gru"),
    ("settransformer", "tgn_gru"),
    ("hopfield", "hopfield_update"),
    ("settransformer", "lnn"),
    ("settransformer", "hnn"),
)

# The canonical physical-event benchmark has a single input/output contract:
# observed force events in, next pair plus next force vector out.  LNN/HNN are
# retained as energy-structured event baselines. They are considerably more
# expensive than the GRU updates because each state update differentiates an
# energy network, so routine smoke runs can use --max-runs to omit them.
PHYSICAL_EVENT_COMPARISON_PANEL = (
    ("sum", "tgn_gru"),
    ("deepsets", "tgn_gru"),
    ("settransformer", "tgn_gru"),
    ("hopfield", "hopfield_update"),
    ("settransformer", "lnn"),
    ("settransformer", "hnn"),
)
# Backward-compatible name for callers that still refer to the original
# first-order diffusion panel.  The same architecture-paired baselines are
# now used for every topology-aware field dynamic.
DIFFUSION_COMPARISON_PANEL = FIELD_COMPARISON_PANEL


def _shortlist_run_grid() -> dict[str, Any]:
    return {
        "seeds": (0,),
        "aggregator": ("ift", "hopfield", "settransformer", "sum", "deepsets"),
        "upd": ("ift_update", "tgn_gru", "lnn", "hopfield_update", "hnn"),
        "dropout": (0.0,),
        "scorer_dropout": (0.0,),
        "use_time_features": (False,),
        "ift_kappa_param": ("softplus",),
        "ift_dt": (0.05,),
        "ift_gamma": (0.0,),
        "ift_kappa_init": (1.0,),
        "ift_kappa_cap": (False,),
        "ift_kappa_max": (None,),
    }


def _full_run_grid() -> dict[str, Any]:
    return {
        "seeds": (0, 42, 123),
        "aggregator": ("ift", "hopfield", "settransformer", "sum", "deepsets"),
        "upd": ("ift_update", "tgn_gru", "lnn", "hopfield_update", "hnn"),
        "dropout": (0.0, 0.1),
        "scorer_dropout": (0.0, 0.1),
        "use_time_features": (False, True),
        "ift_kappa_param": ("softplus", "exp"),
        "ift_dt": (0.01, 0.05, 0.1, 0.2),
        "ift_gamma": (0.0, 0.01, 0.05, 0.1),
        "ift_kappa_init": (0.1, 0.5, 1.0, 2.0),
        "ift_kappa_cap": (False, True),
        "ift_kappa_max": (1.0, 2.0, 5.0, None),
    }


def _make_shortlist_runs(model_cfg: ModelConfig) -> list[SweepRun]:
    return make_runs(model_cfg, **_shortlist_run_grid())


def _infer_ift_drive_feature_idx(feature_schema: Sequence[str]) -> Optional[int]:
    if "drive" in feature_schema:
        return feature_schema.index("drive")
    if "signal" in feature_schema:
        return feature_schema.index("signal")
    if "value" in feature_schema:
        return feature_schema.index("value")
    if "shift" in feature_schema:
        return feature_schema.index("shift")
    if len(feature_schema) > 0:
        return 0
    return None


def _supported_ift_variants(*, event_dim: int) -> tuple[str, ...]:
    if event_dim <= 0:
        return ("generic",)
    return IFT_VARIANT_CHOICES


def _ift_selector_requested(args: argparse.Namespace) -> bool:
    return any(
        getattr(args, name, None) is not None
        for name in ("ift_variants", "ift_orders", "ift_history_steps", "ift_self_rollout", "ift_free_rollout")
    )


def _resolve_ift_selection(
    args: argparse.Namespace,
    *,
    task_name: str,
    event_dim: int,
) -> tuple[list[int], list[str], list[int]]:
    raw_orders = cast(Optional[Sequence[int]], getattr(args, "ift_orders", None))
    raw_variants = cast(Optional[Sequence[str]], getattr(args, "ift_variants", None))
    raw_history = cast(Optional[Sequence[int]], getattr(args, "ift_history_steps", None))
    supported_variants = _supported_ift_variants(event_dim=event_dim)

    orders = list(IFT_ORDER_CHOICES) if raw_orders is None or len(raw_orders) == 0 else [int(order) for order in raw_orders]
    variants = list(supported_variants) if raw_variants is None or len(raw_variants) == 0 else [str(variant) for variant in raw_variants]
    history_steps = (
        list(IFT_HISTORY_STEP_CHOICES)
        if raw_history is None or len(raw_history) == 0
        else [int(step) for step in raw_history]
    )
    unsupported_variants = [variant for variant in variants if variant not in supported_variants]
    if unsupported_variants:
        requested = ", ".join(unsupported_variants)
        supported = ", ".join(supported_variants)
        raise ValueError(
            f"IFT variants [{requested}] are not supported for synthetic task {task_name}. "
            f"Supported variants: {supported}."
        )
    if "auto" not in variants and raw_history is not None:
        raise ValueError("--ift-history-steps requires selecting the auto IFT variant.")
    if 2 not in orders and ("auto" in variants or raw_history is not None):
        raise ValueError("IFT auto/history variants require including second-order IFT via --ift-orders 2.")
    return orders, variants, history_steps


def _resolve_selected_ift_variant_runs(
    base_model_cfg: ModelConfig,
    *,
    task_name: str,
    feature_schema: Sequence[str],
    args: argparse.Namespace,
) -> list[SweepRun]:
    drive_feature_idx = _infer_ift_drive_feature_idx(feature_schema)
    orders, variants, history_steps = _resolve_ift_selection(
        args,
        task_name=task_name,
        event_dim=int(base_model_cfg.event_dim),
    )
    return build_ift_variant_runs(
        base_model_cfg,
        task_name=task_name,
        drive_feature_idx=drive_feature_idx,
        seed=int(args.seed),
        orders=orders,
        variants=variants,
        history_steps=history_steps,
        self_rollout=bool(getattr(args, "ift_self_rollout", False)),
        free_rollout=bool(getattr(args, "ift_free_rollout", False)),
    )


def _replace_ift_shortlist_run(
    runs: Sequence[SweepRun],
    *,
    replacement_runs: Sequence[SweepRun],
) -> list[SweepRun]:
    out: list[SweepRun] = []
    replaced = False
    for run in runs:
        pair = (run.model_cfg.aggregator, run.model_cfg.update)
        if pair == IFT_RUN_PAIR:
            if not replaced:
                out.extend(replacement_runs)
                replaced = True
            continue
        out.append(run)
    if not replaced:
        out.extend(replacement_runs)
    return out


def build_ift_variant_runs(
    base_model_cfg: ModelConfig,
    *,
    task_name: str,
    drive_feature_idx: Optional[int],
    near_ar1_delta_coeffs: Optional[tuple[float, float, float, float]] = None,
    seed: int = 0,
    orders: Optional[Sequence[int]] = None,
    variants: Optional[Sequence[str]] = None,
    history_steps: Optional[Sequence[int]] = None,
    self_rollout: bool = False,
    free_rollout: bool = False,
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

    selected_orders = [int(order) for order in (IFT_ORDER_CHOICES if orders is None else orders)]
    selected_variants = [str(variant) for variant in (IFT_VARIANT_CHOICES if variants is None else variants)]
    selected_history_steps = [
        int(step)
        for step in (IFT_HISTORY_STEP_CHOICES if history_steps is None else history_steps)
    ]

    shared: dict[str, Any] = dict(
        aggregator="ift",
        update="ift_update",
        ift_laplacian_mode="current_bin",
        ift_message_reduce="sum",
        ift_force_reduce="sum",
        ift_drive_feature_idx=drive_feature_idx,
    )
    forcing_by_variant = {
        "generic": "generic_mlp",
        "linear": "linear_event",
        "direct": "direct_scalar",
    }
    if 1 in selected_orders:
        for variant in ("generic", "linear", "direct"):
            if variant not in selected_variants:
                continue
            add(
                f"ift1_{variant}",
                **shared,
                ift_update_order="first",
                ift_forcing_mode=forcing_by_variant[variant],
            )
    if 2 in selected_orders:
        for variant in ("generic", "linear", "direct"):
            if variant not in selected_variants:
                continue
            add(
                f"ift2_{variant}",
                **shared,
                ift_update_order="second",
                ift_forcing_mode=forcing_by_variant[variant],
            )
        if "auto" in selected_variants:
            add(
                "ift2_auto",
                **shared,
                ift_update_order="second",
                ift_forcing_mode="linear_event",
                ift2_readout_mode="linear_h_v_force",
                ift_velocity_teacher_forcing=False,
                ift2_readout_init_mode="small_random",
                ift2_readout_init_scale=0.01,
                lr=1e-2,
                prediction_mode=cast(PredictionMode, "delta"),
            )
            for history_step in selected_history_steps:
                add(
                    f"ift2_hist_vel_k{history_step}",
                    **shared,
                    ift_update_order="second",
                    ift_forcing_mode="linear_event",
                    ift2_readout_mode="linear_h_v_force",
                    ift_velocity_teacher_forcing=False,
                    ift_history_vel_steps=history_step,
                    ift2_readout_init_mode="small_random",
                    ift2_readout_init_scale=0.01,
                    lr=1e-2,
                    prediction_mode=cast(PredictionMode, "delta"),
                )
    if self_rollout or free_rollout:
        if task_name not in {"diffusion", "wave", "coupled_oscillator", "wave_grid", "wave_torus", "wave_doorway", "wave_swisscheese"}:
            raise ValueError("Free and self IFT rollouts are implemented only for diffusion, wave, and coupled_oscillator tasks.")
        if 2 not in selected_orders:
            raise ValueError("Free and self IFT rollouts require including second-order IFT via --ift-orders 2.")
    if free_rollout:
        add(
            "ift2_free_hist_vel_k1",
            **shared,
            ift_update_order="second",
            ift_forcing_mode="linear_event",
            ift2_readout_mode="linear_h_v_force",
            ift_velocity_teacher_forcing=False,
            ift_history_vel_steps=1,
            ift2_readout_init_mode="small_random",
            ift2_readout_init_scale=0.01,
            ift_rollout_free_drive=True,
            lr=1e-2,
            prediction_mode=cast(PredictionMode, "delta"),
        )
    if self_rollout:
        add(
            "ift2_self_hist_vel_k1",
            **shared,
            ift_update_order="second",
            ift_forcing_mode="linear_event",
            ift2_readout_mode="linear_h_v_force",
            ift_velocity_teacher_forcing=False,
            ift_history_vel_steps=1,
            ift2_readout_init_mode="small_random",
            ift2_readout_init_scale=0.01,
            ift_rollout_self_generated=True,
            lr=1e-2,
            prediction_mode=cast(PredictionMode, "delta"),
        )
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
        _make_shortlist_runs(model_cfg),
        FOCUSED_COMBINATIONS,
    )


def _diffusion_runs(model_cfg: ModelConfig, *, seed: int) -> list[SweepRun]:
    """Return the default first-order FNN diffusion comparison panel."""
    runs = build_ift_variant_runs(
        model_cfg,
        task_name="diffusion",
        drive_feature_idx=_infer_ift_drive_feature_idx(("signal", "is_drive")),
        seed=seed,
        orders=(1,),
        variants=("generic", "linear", "direct"),
    )
    runs.extend(_field_baseline_runs(model_cfg, seed=seed))
    return runs


def _field_baseline_runs(model_cfg: ModelConfig, *, seed: int) -> list[SweepRun]:
    """Build the architecture-paired baseline panel for field dynamics."""
    runs: list[SweepRun] = []
    for aggregator, update in FIELD_COMPARISON_PANEL:
        cfg = ModelConfig(**asdict(model_cfg))
        cfg.aggregator = aggregator  # type: ignore[assignment]
        cfg.update = update  # type: ignore[assignment]
        runs.append(SweepRun(name=f"{aggregator}/{update}", model_cfg=cfg, seed=seed))
    return runs


def _physical_event_runs(model_cfg: ModelConfig, *, seed: int) -> list[SweepRun]:
    """FNN plus neural baselines under the same event/force objective."""
    runs: list[SweepRun] = []

    fnn_cfg = ModelConfig(**asdict(model_cfg))
    fnn_cfg.fnn = True
    fnn_cfg.fnn_state_dim = int(model_cfg.event_dim or 1)
    fnn_cfg.predict_event_features = True
    runs.append(SweepRun(name="fnn", model_cfg=fnn_cfg, lr=3e-3, seed=seed))

    for aggregator, update in PHYSICAL_EVENT_COMPARISON_PANEL:
        cfg = ModelConfig(**asdict(model_cfg))
        cfg.aggregator = aggregator  # type: ignore[assignment]
        cfg.update = update  # type: ignore[assignment]
        cfg.predict_event_features = True
        runs.append(SweepRun(name=f"{aggregator}/{update}", model_cfg=cfg, seed=seed))
    return runs


def _full_runs(model_cfg: ModelConfig) -> list[SweepRun]:
    return make_runs(model_cfg, **_full_run_grid())


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
    task = str(args.synthetic_task)
    topology = getattr(args, "synthetic_topology", None)
    if topology is not None and task not in TOPOLOGY_SELECTABLE_FIELD_TASKS:
        raise ValueError("--synthetic-topology is supported only with diffusion, wave, and coupled_oscillator dynamics.")
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
    if (task in GRID_WAVE_TASKS or (task in TOPOLOGY_SELECTABLE_FIELD_TASKS and topology not in {None, "ring"})) and args.synthetic_num_nodes is None:
        # Keep the default benchmark a square lattice at every preset size.
        dataset_num_nodes = {"smoke": 36, "quick": 64, "sweep": 100}[preset]
    return SyntheticDatasetConfig(
        name=f"synthetic_{task}_{topology or 'ring'}_{preset}" if task in TOPOLOGY_SELECTABLE_FIELD_TASKS else f"synthetic_{task}_{preset}",
        task=task,
        num_nodes=dataset_num_nodes,
        num_bins=dataset_num_bins,
        events_per_bin=dataset_events,
        num_episodes=int(10 if getattr(args, "synthetic_num_episodes", None) is None else args.synthetic_num_episodes),
        raindrop_interval=getattr(args, "synthetic_raindrop_interval", None),
        event_threshold=float(getattr(args, "synthetic_event_threshold", 0.0)),
        dt=getattr(args, "synthetic_dt", None),
        gamma=getattr(args, "synthetic_gamma", None),
        omega=getattr(args, "synthetic_omega", None),
        force_scale=getattr(args, "synthetic_force_scale", None),
        field_topology=topology,
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

    if synthetic_cfg.task in {"diffusion", "wave", "coupled_oscillator"}:
        model_cfg.predict_event_features = True
        model_cfg.fnn_order = 1 if synthetic_cfg.task == "diffusion" else 2
        runs = _physical_event_runs(model_cfg, seed=int(args.seed))
        epochs = 2 if preset == "smoke" else 20
        early_steps = 5 if preset == "smoke" else 10
    elif preset in {"smoke", "quick"} and synthetic_cfg.task == "diffusion":
        runs = _diffusion_runs(model_cfg, seed=int(args.seed))
        if _ift_selector_requested(args):
            selected_ift_runs = _resolve_selected_ift_variant_runs(
                model_cfg,
                task_name=synthetic_cfg.task,
                feature_schema=task_spec.feature_schema,
                args=args,
            )
            runs = _replace_ift_shortlist_run(runs, replacement_runs=selected_ift_runs)
        epochs = 1 if preset == "smoke" else 4
        early_steps = 5 if preset == "smoke" else 10
    elif preset in {"smoke", "quick"} and synthetic_cfg.task in {"wave", "coupled_oscillator"}:
        # Keep second-order field dynamics on the same robust comparison
        # panel as diffusion.  The default IFT candidate is an H/V/force
        # second-order readout; --ift-variants can replace it with the full
        # requested ablation set below.
        runs = build_ift_variant_runs(
            model_cfg,
            task_name=synthetic_cfg.task,
            drive_feature_idx=_infer_ift_drive_feature_idx(task_spec.feature_schema),
            seed=int(args.seed),
            orders=(2,),
            variants=("auto",),
            history_steps=(1,),
        )
        runs.extend(_field_baseline_runs(model_cfg, seed=int(args.seed)))
        if _ift_selector_requested(args):
            selected_ift_runs = _resolve_selected_ift_variant_runs(
                model_cfg,
                task_name=synthetic_cfg.task,
                feature_schema=task_spec.feature_schema,
                args=args,
            )
            runs = _replace_ift_shortlist_run(runs, replacement_runs=selected_ift_runs)
        epochs = 1 if preset == "smoke" else 4
        early_steps = 5 if preset == "smoke" else 10
    elif preset in {"smoke", "quick"}:
        shortlist_runs = _make_shortlist_runs(model_cfg)
        runs = select_runs(shortlist_runs, set(task_spec.recommended_pairs))
        if _ift_selector_requested(args):
            selected_ift_runs = _resolve_selected_ift_variant_runs(
                model_cfg,
                task_name=synthetic_cfg.task,
                feature_schema=task_spec.feature_schema,
                args=args,
            )
            runs = _replace_ift_shortlist_run(runs, replacement_runs=selected_ift_runs)
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
    if dataset_override in {"college_msg", "email_eu_core", "sociopatterns"}:
        social_args = argparse.Namespace(**(vars(args) if args is not None else {}))
        social_args.jodie_fnn = True
        suite = build_suite("quick" if preset == "smoke" else preset, device, "jodie", social_args)
        for cfg in [suite.model_cfg, *(run.model_cfg for run in suite.runs)]:
            cfg.fnn_dt = 1.0
            cfg.fnn_learn_dt = bool(getattr(args, "fnn_learn_dt", False))
            cfg.fnn_max_dt_omega = None
        return replace(suite, dataset="social", dataset_kwargs={
            "name": dataset_override,
            "root": getattr(args, "social_root", "data"),
            "bin_size": getattr(args, "social_bin_size", None),
            "device": str(device),
        })
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

        jodie_fnn = bool(getattr(args, "jodie_fnn", False)) if args is not None else False
        jodie_cfg = JODIEConfig(
            root="./data/JODIE",
            name=str(getattr(args, "jodie_name", "Wikipedia")),
            device=device,
            unit_force=jodie_fnn,
        )
        train_cfg = _default_train_config(num_nodes=0, num_neg=10, log_every=500, device=device)
        model_cfg = _base_model_config(small=False)
        model_cfg.event_dim = 1 if jodie_fnn else None
        if jodie_fnn:
            model_cfg.fnn_state_dim = 1
            model_cfg.fnn_order = 2
            model_cfg.fnn_topology_mode = "observed_sparse"
            model_cfg.fnn_state_score = True
            model_cfg.fnn_max_dt_omega = 1.5
            model_cfg.fnn_learn_dt = True
            model_cfg.fnn_learn_gamma = True
            model_cfg.fnn_learn_omega = True
            model_cfg.fnn_learn_input_force_scale = True
            model_cfg.predict_event_features = True
            # The FNN must complete its alternating schedule, while ordinary
            # neural baselines stop after their first failed sparse val check.
            train_cfg.early_stop_patience = 1
            # Compare the FNN with neural event models under the identical
            # unit-force stream and next-pair/force objective. The helper
            # enables FNN only for its own run, leaving the baselines intact.
            runs = _physical_event_runs(model_cfg, seed=int(getattr(args, "seed", 0)))
        else:
            runs = _focused_runs(model_cfg)
        return RunSuite(
            dataset="jodie",
            dataset_kwargs=asdict(jodie_cfg),
            train_cfg=train_cfg,
            model_cfg=model_cfg,
            runs=runs,
            epochs=2,
            eval_slices=EvalSlices(early_steps=10),
            save_jsonl_path=None,
        )

    jodie_fnn = bool(getattr(args, "jodie_fnn", False)) if args is not None else False
    jodie_cfg = JODIEConfig(
        root="./data/JODIE",
        name=str(getattr(args, "jodie_name", "Wikipedia")),
        device=device,
        unit_force=jodie_fnn,
    )
    train_cfg = _default_train_config(num_nodes=0, num_neg=20, log_every=2000, device=device)
    model_cfg = _base_model_config(small=False)
    model_cfg.event_dim = 1 if jodie_fnn else None
    if jodie_fnn:
        model_cfg.fnn_state_dim = 1
        model_cfg.fnn_order = 2
        model_cfg.fnn_topology_mode = "observed_sparse"
        model_cfg.fnn_state_score = True
        model_cfg.fnn_max_dt_omega = 1.5
        model_cfg.fnn_learn_dt = True
        model_cfg.fnn_learn_gamma = True
        model_cfg.fnn_learn_omega = True
        model_cfg.fnn_learn_input_force_scale = True
        model_cfg.predict_event_features = True
        train_cfg.early_stop_patience = 1
        runs = _physical_event_runs(model_cfg, seed=int(getattr(args, "seed", 0)))
    else:
        runs = _full_runs(model_cfg)
    return RunSuite(
        dataset="jodie",
        dataset_kwargs=asdict(jodie_cfg),
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        runs=runs,
        epochs=6,
        eval_slices=EvalSlices(early_steps=10),
        save_jsonl_path=None,
    )


def load_dataset(kind: str, dataset_kwargs: Dict[str, Any]):
    if kind == "social":
        from interactiondynamics.data.social import SocialConfig, SocialEventDataset
        return SocialEventDataset(SocialConfig(**dataset_kwargs))
    if kind == "traffic":
        from interactiondynamics.data.traffic import TrafficConfig, TrafficDataset
        return TrafficDataset(TrafficConfig(**dataset_kwargs))
    if kind == "toy":
        return ToyShiftDataset(ToyShiftConfig(**dataset_kwargs))
    if kind == "jodie":
        return JODIEBinnedDataset(JODIEConfig(**dataset_kwargs))
    if kind == "synthetic":
        return SyntheticDataset(SyntheticDatasetConfig(**dataset_kwargs))
    raise ValueError(f"Unsupported dataset kind: {kind}")
