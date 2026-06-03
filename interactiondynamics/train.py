import argparse
import itertools
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

import numpy as np
import torch

from interactiondynamics.core.config import ModelConfig
from interactiondynamics.core.events import EventBatch
from interactiondynamics.data.interfaces import EdgeTargetBatch
from interactiondynamics.data.jodie import JODIEBinnedDataset, JODIEConfig  # type: ignore
from interactiondynamics.data.physical import PhysicalDatasetConfig, PhysicalDynamicsDataset
from interactiondynamics.data.toy import ToyShiftConfig, ToyShiftDataset
from interactiondynamics.eval.evaluate import EvalSlices, evaluate_stream_sliced
from interactiondynamics.eval.node_metrics import (
    edge_regression_metrics,
    node_labels_from_events,
    node_prediction_metrics,
    node_regression_metrics,
)
from interactiondynamics.eval.ranking_metrics import ranking_loss_and_metrics
from interactiondynamics.models.tgn_model import build_tgn_model


FOCUSED_COMBINATIONS = {
    ("ift", "ift_update"),
    ("hopfield", "hopfield_update"),
    ("settransformer", "lnn"),
    ("settransformer", "hnn"),
    ("settransformer", "tgn_gru"),
}


@dataclass(frozen=True)
class SweepRun:
    name: str
    model_cfg: ModelConfig
    lr: Optional[float] = None
    weight_decay: Optional[float] = None
    num_neg: Optional[int] = None
    tbptt_steps: Optional[int] = None
    seed: int = 0


@dataclass
class RunResult:
    name: str
    seed: int
    epochs: int
    best_val_loss: float
    best_val_mrr: float
    best_epoch: int
    best_snapshot: dict
    final_snapshot: dict
    wall_sec: float


@dataclass
class TrainConfig:
    num_nodes: int
    num_neg: int = 20
    node_loss_weight: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    device: torch.device = torch.device("cpu")
    log_every: int = 50
    tbptt_steps: int = 1
    update_before_score: bool = True
    debug: bool = False


@dataclass(frozen=True)
class RunSuite:
    dataset: str
    dataset_kwargs: Dict[str, Any]
    train_cfg: TrainConfig
    model_cfg: ModelConfig
    runs: list[SweepRun]
    epochs: int
    eval_slices: EvalSlices
    save_jsonl_path: Optional[str] = None


def short_run_label(run: SweepRun) -> str:
    agg = run.model_cfg.aggregator
    if agg == "settransformer":
        agg = "settf"
    upd = run.model_cfg.update
    return f"{agg}/{upd}"


def short_run_label_from_name(name: str) -> str:
    parts = dict(piece.split("=", 1) for piece in name.split("|") if "=" in piece)
    agg = parts.get("agg", "?")
    if agg == "settransformer":
        agg = "settf"
    upd = parts.get("update", "?")
    return f"{agg}/{upd}"


def node_metric_name(metrics: Dict[str, float]) -> Optional[str]:
    if "node_mse" in metrics:
        return "node_mse"
    if "node_acc" in metrics:
        return "node_acc"
    return None


def edge_metric_name(metrics: Dict[str, float]) -> Optional[str]:
    if "edge_mse" in metrics:
        return "edge_mse"
    return None


def primary_metric_name(metrics: Dict[str, float]) -> str:
    if "edge_mse" in metrics:
        return "edge_mse"
    return "mrr"


def infer_primary_metric(*metric_sets: Dict[str, float]) -> str:
    for metrics in metric_sets:
        if "edge_mse" in metrics:
            return "edge_mse"
    return "mrr"


def format_edge_metric(metrics: Dict[str, float]) -> str:
    metric_name = edge_metric_name(metrics)
    if metric_name is None:
        return ""
    value = metrics.get(metric_name, float("nan"))
    return f"{metric_name}={value:.4g}"


def format_edge_metric_bundle(metrics: Dict[str, float], *, prefix: Optional[str] = None) -> str:
    key_prefix = "" if prefix is None else f"{prefix}_"
    mse = metrics.get(f"{key_prefix}edge_mse", float("nan"))
    r2 = metrics.get(f"{key_prefix}edge_r2", float("nan"))
    nrmse = metrics.get(f"{key_prefix}edge_nrmse", float("nan"))
    skill = metrics.get(f"{key_prefix}edge_skill_vs_persistent", float("nan"))
    if math.isnan(mse):
        return ""
    parts = [
        f"mse={mse:.4g}",
        f"r2={r2:.3f}" if not math.isnan(r2) else "r2=nan",
        f"nrmse={nrmse:.3f}" if not math.isnan(nrmse) else "nrmse=nan",
    ]
    if not math.isnan(skill):
        parts.append(f"skill={skill:.1%}")
    return " ".join(parts)


def format_node_metric(metrics: Dict[str, float]) -> str:
    metric_name = node_metric_name(metrics)
    if metric_name is None:
        return ""
    value = metrics.get(metric_name, float("nan"))
    if metric_name == "node_mse":
        return f"{metric_name}={value:.4g}"
    return f"{metric_name}={value:.4f}"


def format_primary_metric(
    metrics: Dict[str, float],
    *,
    prefix: Optional[str] = None,
    name: Optional[str] = None,
) -> str:
    name = primary_metric_name(metrics) if name is None else name
    full_name = f"{prefix}_{name}" if prefix is not None else name
    value = metrics.get(full_name, float("nan"))
    if name == "edge_mse":
        return f"{full_name}={value:.4g}"
    return f"{full_name}={value:.4f}"


def build_physical_dataset_config(
    args: argparse.Namespace,
    device: torch.device,
    *,
    preset: str,
) -> PhysicalDatasetConfig:
    num_bins = args.num_bins
    if num_bins is None:
        num_bins = 24 if preset == "smoke" else 120

    grid_m = args.graph_m
    grid_n = args.graph_n
    if preset == "smoke" and args.dataset == "physical" and args.graph_m == 24 and args.graph_n == 24:
        grid_m = 12
        grid_n = 12

    return PhysicalDatasetConfig(
        graph_kind=args.graph_kind,
        dynamics_kind=args.dynamics,
        num_bins=int(num_bins),
        device=device,
        seed=int(args.seed),
        grid_m=int(grid_m),
        grid_n=int(grid_n),
        small_world_n=args.small_world_n,
        small_world_k=int(args.small_world_k),
        small_world_beta=float(args.small_world_beta),
        tree_levels=args.tree_levels,
        tree_branching=int(args.tree_branching),
    )


NEGATIVE_ASCII_RAMP = "@%#*+=-"
POSITIVE_ASCII_RAMP = "+*#%@&$"
ANSI_RESET = "\x1b[0m"
NEGATIVE_COLOR_RAMP = (52, 88, 124, 160, 196)
POSITIVE_COLOR_RAMP = (22, 28, 34, 40, 46)


def _wave_ascii_char(value: float, scale: float) -> str:
    if scale <= 0.0:
        return "."
    norm = max(-1.0, min(1.0, value / scale))
    if abs(norm) < 0.08:
        return "."
    if norm < 0.0:
        idx = min(len(NEGATIVE_ASCII_RAMP) - 1, int(abs(norm) * len(NEGATIVE_ASCII_RAMP)))
        return NEGATIVE_ASCII_RAMP[idx]
    idx = min(len(POSITIVE_ASCII_RAMP) - 1, int(norm * len(POSITIVE_ASCII_RAMP)))
    return POSITIVE_ASCII_RAMP[idx]


def _colorize_ascii_char(char: str, value: float, scale: float, use_color: bool) -> str:
    if not use_color or char == ".":
        return char
    if scale <= 0.0:
        return char
    magnitude = min(1.0, abs(float(value)) / float(scale))
    if magnitude < 0.08:
        return char
    ramp = NEGATIVE_COLOR_RAMP if value < 0.0 else POSITIVE_COLOR_RAMP
    idx = min(len(ramp) - 1, int(magnitude * len(ramp)))
    color_code = ramp[idx]
    return f"\x1b[38;5;{color_code}m{char}{ANSI_RESET}"
    return char


def render_ascii_grid_frame(
    frame: np.ndarray,
    shape: tuple[int, int],
    scale: float,
    *,
    row_stride: int = 1,
    use_color: bool = True,
) -> str:
    grid = frame.reshape(shape)
    lines = []
    for row in grid[::max(1, row_stride)]:
        rendered = []
        for value in row:
            scalar = float(value)
            char = _wave_ascii_char(scalar, scale)
            rendered.append(_colorize_ascii_char(char, scalar, scale, use_color))
        lines.append("".join(rendered))
    return "\n".join(lines)


def run_ascii_viz(args: argparse.Namespace, device: torch.device) -> None:
    if args.dataset not in (None, "physical"):
        raise ValueError("ASCII visualization currently supports only --dataset physical.")

    cfg = build_physical_dataset_config(args, device, preset="quick")
    ds = PhysicalDynamicsDataset(cfg)
    meta = ds.graph_meta()
    shape = meta.get("shape")
    if shape is None:
        raise ValueError("ASCII visualization currently supports only grid graphs.")

    states = ds.states()
    total_steps = int(states.shape[0])
    start = max(0, int(args.start_step))
    stop = total_steps if args.steps is None else min(total_steps, start + int(args.steps))
    if start >= stop:
        raise ValueError(f"No frames to render: start_step={start}, stop={stop}, total_steps={total_steps}")

    stride = max(1, int(args.every))
    frames = states[start:stop:stride]
    scale = float(np.max(np.abs(frames)))
    delay = 1.0 / max(float(args.fps), 1e-6)

    if not args.no_clear:
        print("\x1b[2J", end="")

    for offset, frame in enumerate(frames):
        step = start + offset * stride
        header = (
            f"ascii viz | {cfg.graph_kind}/{cfg.dynamics_kind} | "
            f"step {step + 1}/{total_steps} | min={frame.min():+.3f} max={frame.max():+.3f}"
        )
        if not args.no_clear:
            print("\x1b[H", end="")
        print(header)
        print(
            render_ascii_grid_frame(
                frame,
                tuple(shape),
                scale,
                row_stride=int(args.row_stride),
                use_color=not bool(args.no_color),
            )
        )
        if offset + 1 < len(frames):
            time.sleep(delay)


def apply_model_overrides(model_cfg: ModelConfig, args: argparse.Namespace) -> ModelConfig:
    cfg = ModelConfig(**asdict(model_cfg))
    cfg.use_node_scorer = bool(args.use_node_scorer)
    cfg.node_scorer_hidden = int(args.node_scorer_hidden)
    return cfg


def train_one_epoch(
    model,
    bins: Iterable[EventBatch],
    node_targets: Optional[Iterable[torch.Tensor]],
    edge_targets: Optional[Iterable[EdgeTargetBatch]],
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
) -> Dict[str, float]:
    model.train()
    device = torch.device(cfg.device)
    state = model.init_state(batch_size=1, num_nodes=cfg.num_nodes, device=device)

    total_loss = 0.0
    total_primary = 0.0
    n_steps = 0
    kappa_sum = 0.0
    kappa_n = 0

    prev: Optional[EventBatch] = None
    target_iter = iter(node_targets) if node_targets is not None else None
    edge_target_iter = iter(edge_targets) if edge_targets is not None else None
    for curr in bins:
        curr = curr.to(device)
        curr_node_target = None if target_iter is None else next(target_iter).to(device)
        curr_edge_target = None if edge_target_iter is None else next(edge_target_iter)
        if prev is None:
            prev = curr
            continue

        state, aux = model.step(state, prev)

        if aux is not None and "kappa" in aux:
            kappa = aux["kappa"]
            if torch.is_tensor(kappa):
                kappa_sum += float(kappa.detach().item())
                kappa_n += 1

        h = state.node
        if cfg.debug:
            print(
                "DEBUG node variance:",
                float(h.std(dim=0).mean().item()),
                "max|h|:",
                float(h.abs().max().item()),
            )

        if prev.t is not None and getattr(state, "aux", None) is not None and "L_bin_t_min" in state.aux:
            assert state.aux["L_bin_t_min"] == int(prev.t.min().item()), (
                "step() did not use prev bin for operator"
            )

        if state.node is not None and (not torch.isfinite(state.node).all()):
            raise RuntimeError("Non-finite state.node after model.step()")

        optimizer.zero_grad(set_to_none=True)

        assert prev.t is not None and int(prev.t.min().item()) == int(prev.t.max().item())
        assert curr.t is not None and int(curr.t.min().item()) == int(curr.t.max().item())
        assert int(prev.t.max().item()) < int(curr.t.min().item())

        if getattr(state, "aux", None) is not None:
            if "L_bin_t_min" in state.aux and "L_bin_t_max" in state.aux:
                pt = int(prev.t.min().item())
                assert state.aux["L_bin_t_min"] == pt and state.aux["L_bin_t_max"] == pt, (
                    f"L bin mismatch: L=({state.aux['L_bin_t_min']},{state.aux['L_bin_t_max']}) prev.t={pt}"
                )

        if curr_edge_target is not None:
            edge_events = curr_edge_target.events.to(device)
            edge_target_values = curr_edge_target.targets.to(device)
            edge_preds = model.score(state, edge_events)
            loss = torch.nn.functional.mse_loss(edge_preds, edge_target_values)
            metrics = edge_regression_metrics(edge_preds.detach(), edge_target_values)
        else:
            loss, metrics = ranking_loss_and_metrics(
                model=model,
                state=state,
                next_events=curr,
                num_nodes=cfg.num_nodes,
                num_neg=cfg.num_neg,
            )
        total_step_loss = loss
        if getattr(model, "node_scorer", None) is not None:
            node_logits = model.score_nodes(state)
            if curr_node_target is not None:
                node_target = curr_node_target.to(node_logits.device, dtype=node_logits.dtype)
                node_loss = torch.nn.functional.mse_loss(node_logits, node_target)
                metrics.update(node_regression_metrics(node_logits.detach(), node_target))
            else:
                node_labels = node_labels_from_events(curr, cfg.num_nodes, device=node_logits.device)
                node_loss = torch.nn.functional.binary_cross_entropy_with_logits(node_logits, node_labels)
                metrics.update(node_prediction_metrics(node_logits.detach(), node_labels))
            metrics["node_loss"] = float(node_loss.detach().item())
            total_step_loss = total_step_loss + (cfg.node_loss_weight * node_loss)

        total_step_loss.backward()
        if cfg.grad_clip and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if cfg.tbptt_steps and ((n_steps + 1) % cfg.tbptt_steps == 0) and state is not None:
            state.detach_()

        total_loss += float(total_step_loss.item())
        total_primary += float(metrics[primary_metric_name(metrics)])
        n_steps += 1

        if cfg.log_every and (n_steps % cfg.log_every) == 0:
            if "mrr" in metrics:
                print(
                    f"[step {n_steps:6d}] "
                    f"loss={loss.item():.4f} "
                    f"mrr={metrics['mrr']:.4f} "
                    f"hits@1={metrics.get('hits@1', 0):.4f} "
                    f"hits@10={metrics.get('hits@10', 0):.4f}"
                )
            else:
                print(
                    f"[step {n_steps:6d}] "
                    f"loss={loss.item():.4f} "
                    f"edge_mse={metrics.get('edge_mse', float('nan')):.4g} "
                    f"edge_mae={metrics.get('edge_mae', float('nan')):.4g}"
                )

        prev = curr

    if n_steps == 0:
        return {"loss": 0.0}

    primary_name = "edge_mse" if edge_targets is not None else "mrr"
    out = {"loss": total_loss / n_steps, primary_name: total_primary / n_steps}
    if kappa_n > 0:
        out["kappa_mean"] = kappa_sum / kappa_n
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def run_one_experiment(
    ds,
    spec,
    base_train_cfg: TrainConfig,
    run: SweepRun,
    build_model_fn: Callable[[Any, ModelConfig], torch.nn.Module],
    epochs: int = 5,
    eval_slices: Optional[EvalSlices] = None,
    save_jsonl_path: Optional[str] = None,
) -> RunResult:
    device = torch.device(base_train_cfg.device)
    set_seed(run.seed)

    train_cfg = TrainConfig(**asdict(base_train_cfg))
    if run.lr is not None:
        train_cfg.lr = run.lr
    if run.weight_decay is not None:
        train_cfg.weight_decay = run.weight_decay
    if run.num_neg is not None:
        train_cfg.num_neg = run.num_neg
    if run.tbptt_steps is not None:
        train_cfg.tbptt_steps = run.tbptt_steps

    model = build_model_fn(spec, run.model_cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )

    if eval_slices is None:
        eval_slices = EvalSlices(early_steps=10)

    best_val_loss = float("inf")
    best_val_mrr = float("nan")
    best_epoch = -1
    best_snapshot: dict = {}
    t0 = time.time()
    train_node_targets = ds.node_targets("train") if hasattr(ds, "node_targets") else None
    val_node_targets = ds.node_targets("val") if hasattr(ds, "node_targets") else None
    test_node_targets = ds.node_targets("test") if hasattr(ds, "node_targets") else None
    train_edge_targets = ds.edge_targets("train") if hasattr(ds, "edge_targets") else None
    val_edge_targets = ds.edge_targets("val") if hasattr(ds, "edge_targets") else None
    test_edge_targets = ds.edge_targets("test") if hasattr(ds, "edge_targets") else None

    baseline_val = evaluate_stream_sliced(
        model,
        ds.bins("val"),
        val_node_targets,
        val_edge_targets,
        train_cfg,
        slices=eval_slices,
    )
    baseline_test = evaluate_stream_sliced(
        model,
        ds.bins("test"),
        test_node_targets,
        test_edge_targets,
        train_cfg,
        slices=eval_slices,
    )
    run_primary_name = infer_primary_metric(baseline_val, baseline_test)
    baseline_str = (
        f"  baseline (no-update)"
        f" | {format_primary_metric(baseline_val, prefix='persistent', name=run_primary_name)}"
        f" | {format_primary_metric(baseline_test, prefix='persistent', name=run_primary_name)}"
    )
    if run_primary_name == "edge_mse":
        baseline_val_edge = format_edge_metric_bundle(baseline_val, prefix="persistent")
        baseline_test_edge = format_edge_metric_bundle(baseline_test, prefix="persistent")
        baseline_str = (
            f"  baseline (no-update)"
            f" | val {baseline_val_edge}"
            f" | test {baseline_test_edge}"
        )
    baseline_node_val = format_node_metric(
        {
            key.replace("persistent_", "", 1): value
            for key, value in baseline_val.items()
            if key.startswith("persistent_node_")
        }
    )
    baseline_node_test = format_node_metric(
        {
            key.replace("persistent_", "", 1): value
            for key, value in baseline_test.items()
            if key.startswith("persistent_node_")
        }
    )
    if baseline_node_val or baseline_node_test:
        baseline_str += f" | val {baseline_node_val} | test {baseline_node_test}"
    print(baseline_str)

    for epoch in range(1, epochs + 1):
        train_stats_step = train_one_epoch(
            model,
            ds.bins("train"),
            train_node_targets,
            train_edge_targets,
            optimizer,
            train_cfg,
        )
        train_eval = evaluate_stream_sliced(
            model,
            ds.bins("train"),
            train_node_targets,
            train_edge_targets,
            train_cfg,
            slices=eval_slices,
        )
        val_stats = evaluate_stream_sliced(
            model,
            ds.bins("val"),
            val_node_targets,
            val_edge_targets,
            train_cfg,
            slices=eval_slices,
        )
        test_stats = evaluate_stream_sliced(
            model,
            ds.bins("test"),
            test_node_targets,
            test_edge_targets,
            train_cfg,
            slices=eval_slices,
        )

        snapshot = {
            "epoch": epoch,
            "train_step": train_stats_step,
            "train_eval": train_eval,
            "val": val_stats,
            "test": test_stats,
        }

        km = train_stats_step.get("kappa_mean", None)
        kappa_str = f" | kappa={km:.4f}" if km is not None else ""
        node_val_str = format_node_metric(val_stats)
        node_test_str = format_node_metric(test_stats)
        node_str = ""
        if node_val_str or node_test_str:
            node_str = f" | val {node_val_str} | test {node_test_str}"
        run_primary_name = infer_primary_metric(train_eval, val_stats, test_stats)
        val_primary = format_primary_metric(val_stats, name=run_primary_name)
        test_primary = format_primary_metric(test_stats, name=run_primary_name)
        train_primary = format_primary_metric(train_eval, name=run_primary_name)
        primary_str = (
            f" | train {train_primary}"
            f" | val {val_primary}"
            f" | test {test_primary}"
        )
        edge_str = ""
        if run_primary_name == "edge_mse":
            edge_str = (
                f" | val {format_edge_metric_bundle(val_stats)}"
                f" | test {format_edge_metric_bundle(test_stats)}"
            )
        print(
            f"  ep {epoch:03d}"
            f" | train loss={train_stats_step['loss']:.4f}"
            f" | val loss={val_stats['loss']:.4f}"
            f"{primary_str}"
            f"{edge_str}"
            f"{kappa_str}"
            f"{node_str}"
        )

        if val_stats["loss"] < best_val_loss:
            best_val_loss = float(val_stats["loss"])
            best_val_mrr = float(val_stats["mrr"]) if "mrr" in val_stats else float("nan")
            best_epoch = epoch
            best_snapshot = snapshot

        if save_jsonl_path is not None:
            row = {
                "run": run.name,
                "seed": run.seed,
                "model_cfg": asdict(run.model_cfg),
                "train_cfg_overrides": {
                    k: v
                    for k, v in {
                        "lr": run.lr,
                        "weight_decay": run.weight_decay,
                        "num_neg": run.num_neg,
                        "tbptt_steps": run.tbptt_steps,
                    }.items()
                    if v is not None
                },
                **snapshot,
            }
            with open(save_jsonl_path, "a") as f:
                f.write(json.dumps(row) + "\n")

    wall = time.time() - t0
    final_snapshot = snapshot
    return RunResult(
        name=run.name,
        seed=run.seed,
        epochs=epochs,
        best_val_loss=best_val_loss,
        best_val_mrr=best_val_mrr,
        best_epoch=best_epoch,
        best_snapshot=best_snapshot,
        final_snapshot=final_snapshot,
        wall_sec=wall,
    )


def build_suite(
    preset: str,
    device: torch.device,
    dataset_override: Optional[str] = None,
    args: Optional[argparse.Namespace] = None,
) -> RunSuite:
    if preset == "smoke":
        if dataset_override == "physical" and args is not None:
            physical_cfg = build_physical_dataset_config(args, device, preset=preset)
            train_cfg = TrainConfig(
                num_nodes=0,
                num_neg=5,
                tbptt_steps=1,
                log_every=100,
                device=device,
                weight_decay=1e-3,
                lr=1e-3,
            )
            model_cfg = ModelConfig(
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
            runs = select_runs(
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
            return RunSuite(
                dataset="physical",
                dataset_kwargs=asdict(physical_cfg),
                train_cfg=train_cfg,
                model_cfg=model_cfg,
                runs=runs,
                epochs=1,
                eval_slices=EvalSlices(early_steps=5),
                save_jsonl_path=None,
            )

        toy_cfg = ToyShiftConfig(
            name="toy_smoke",
            num_nodes=64,
            num_bins=24,
            events_per_bin=32,
            shift=7,
            device=device,
        )
        train_cfg = TrainConfig(
            num_nodes=toy_cfg.num_nodes,
            num_neg=5,
            tbptt_steps=1,
            log_every=100,
            device=device,
            weight_decay=1e-3,
            lr=1e-3,
        )
        model_cfg = ModelConfig(
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
        runs = select_runs(
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
        return RunSuite(
            dataset="toy",
            dataset_kwargs=asdict(toy_cfg),
            train_cfg=train_cfg,
            model_cfg=model_cfg,
            runs=runs,
            epochs=1,
            eval_slices=EvalSlices(early_steps=5),
            save_jsonl_path=None,
        )

    if preset == "quick":
        if dataset_override == "physical" and args is not None:
            physical_cfg = build_physical_dataset_config(args, device, preset=preset)
            train_cfg = TrainConfig(
                num_nodes=0,
                num_neg=10,
                tbptt_steps=1,
                log_every=250,
                device=device,
                weight_decay=1e-3,
                lr=1e-3,
            )
            model_cfg = ModelConfig(
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
            runs = select_runs(
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
            return RunSuite(
                dataset="physical",
                dataset_kwargs=asdict(physical_cfg),
                train_cfg=train_cfg,
                model_cfg=model_cfg,
                runs=runs,
                epochs=2,
                eval_slices=EvalSlices(early_steps=10),
                save_jsonl_path=None,
            )

        if dataset_override == "toy":
            toy_cfg = ToyShiftConfig(
                name="toy_quick",
                num_nodes=128,
                num_bins=48,
                events_per_bin=64,
                shift=7,
                device=device,
            )
            train_cfg = TrainConfig(
                num_nodes=0,
                num_neg=10,
                tbptt_steps=1,
                log_every=250,
                device=device,
                weight_decay=1e-3,
                lr=1e-3,
            )
            model_cfg = ModelConfig(
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
            runs = select_runs(
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
            return RunSuite(
                dataset="toy",
                dataset_kwargs=asdict(toy_cfg),
                train_cfg=train_cfg,
                model_cfg=model_cfg,
                runs=runs,
                epochs=2,
                eval_slices=EvalSlices(early_steps=10),
                save_jsonl_path=None,
            )

        jodie_cfg = JODIEConfig(root="./data/JODIE", name="Wikipedia", device=device)
        train_cfg = TrainConfig(
            num_nodes=0,
            num_neg=10,
            tbptt_steps=1,
            log_every=500,
            device=device,
            weight_decay=1e-3,
            lr=1e-3,
        )
        model_cfg = ModelConfig(
            node_dim=128,
            msg_dim=128,
            event_dim=None,
            scorer="mlp",
            scorer_hidden=256,
            aggregator="sum",
            use_time_features=False,
            dropout=0.0,
            scorer_dropout=0.0,
            encoder_hidden=256,
        )
        runs = select_runs(
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

    if dataset_override == "physical" and args is not None:
        physical_cfg = build_physical_dataset_config(args, device, preset=preset)
        train_cfg = TrainConfig(
            num_nodes=0,
            num_neg=20,
            tbptt_steps=1,
            log_every=500,
            device=device,
            weight_decay=1e-3,
            lr=1e-3,
        )
        model_cfg = ModelConfig(
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
        runs = make_runs(
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
        return RunSuite(
            dataset="physical",
            dataset_kwargs=asdict(physical_cfg),
            train_cfg=train_cfg,
            model_cfg=model_cfg,
            runs=runs,
            epochs=6,
            eval_slices=EvalSlices(early_steps=10),
            save_jsonl_path=None,
        )

    jodie_cfg = JODIEConfig(root="./data/JODIE", name="Wikipedia", device=device)
    train_cfg = TrainConfig(
        num_nodes=0,
        num_neg=20,
        tbptt_steps=1,
        log_every=2000,
        device=device,
        weight_decay=1e-3,
        lr=1e-3,
    )
    model_cfg = ModelConfig(
        node_dim=128,
        msg_dim=128,
        event_dim=None,
        scorer="mlp",
        scorer_hidden=256,
        aggregator="sum",
        use_time_features=False,
        dropout=0.0,
        scorer_dropout=0.0,
        encoder_hidden=256,
    )
    runs = make_runs(
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
    if kind == "toy":
        return ToyShiftDataset(ToyShiftConfig(**dataset_kwargs))
    if kind == "jodie":
        return JODIEBinnedDataset(JODIEConfig(**dataset_kwargs))
    if kind == "physical":
        return PhysicalDynamicsDataset(PhysicalDatasetConfig(**dataset_kwargs))
    raise ValueError(f"Unsupported dataset kind: {kind}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run interaction dynamics training presets.")
    subparsers = parser.add_subparsers(dest="command")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--dataset",
        choices=("toy", "jodie", "physical"),
        default=None,
        help="Optional dataset override when supported by the command.",
    )
    common.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap on the number of runs to execute after filtering.",
    )
    common.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for the preset epoch count.",
    )
    common.add_argument(
        "--use-node-scorer",
        action="store_true",
        help="Enable auxiliary node prediction on whether a node appears in the next bin.",
    )
    common.add_argument(
        "--node-loss-weight",
        type=float,
        default=1.0,
        help="Weight for the auxiliary node prediction loss.",
    )
    common.add_argument(
        "--node-scorer-hidden",
        type=int,
        default=128,
        help="Hidden size for the auxiliary node scorer MLP.",
    )
    common.add_argument(
        "--save-jsonl",
        type=str,
        default=None,
        help="Optional path to append per-epoch JSONL results.",
    )
    common.add_argument("--num-bins", type=int, default=None, help="Number of simulated time bins for physical datasets.")
    common.add_argument("--seed", type=int, default=0, help="Random seed for simulated physical datasets.")
    common.add_argument(
        "--graph-kind",
        choices=("grid", "small_world", "tree"),
        default="grid",
        help="Graph family for physical datasets.",
    )
    common.add_argument(
        "--dynamics",
        choices=("wave", "wave_pulse", "sis", "sirs"),
        default="wave",
        help="Dynamics family for physical datasets.",
    )
    common.add_argument("--graph-m", type=int, default=24, help="Grid height for physical grid graphs.")
    common.add_argument("--graph-n", type=int, default=24, help="Grid width for physical grid graphs.")
    common.add_argument("--small-world-n", type=int, default=None, help="Node count for physical small-world graphs.")
    common.add_argument("--small-world-k", type=int, default=8, help="Ring-lattice degree for physical small-world graphs.")
    common.add_argument("--small-world-beta", type=float, default=0.12, help="Rewiring probability for physical small-world graphs.")
    common.add_argument("--tree-levels", type=int, default=None, help="Explicit level count for physical tree graphs.")
    common.add_argument("--tree-branching", type=int, default=3, help="Branching factor for physical tree graphs.")

    subparsers.add_parser(
        "smoke",
        parents=[common],
        help="Run a tiny smoke test preset.",
    )
    subparsers.add_parser(
        "quick",
        parents=[common],
        help="Run the focused shortlist preset.",
    )
    subparsers.add_parser(
        "sweep",
        parents=[common],
        help="Run the original broad sweep.",
    )
    viz = subparsers.add_parser(
        "viz",
        parents=[common],
        help="Render an ASCII animation for physical grid dynamics.",
    )
    viz.set_defaults(dataset="physical")
    viz.add_argument("--fps", type=float, default=12.0, help="Frames per second for terminal playback.")
    viz.add_argument("--every", type=int, default=1, help="Show every n-th simulated step.")
    viz.add_argument("--steps", type=int, default=None, help="Optional number of frames to display.")
    viz.add_argument("--start-step", type=int, default=0, help="Simulation step index to start from.")
    viz.add_argument("--row-stride", type=int, default=1, help="Render every n-th grid row to shorten the display.")
    viz.add_argument("--no-color", action="store_true", help="Disable ANSI coloring in the ASCII renderer.")
    viz.add_argument(
        "--no-clear",
        action="store_true",
        help="Print frames sequentially instead of reusing the terminal screen.",
    )

    parser.add_argument(
        "--preset",
        choices=("smoke", "quick", "full"),
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.command is None:
        if args.preset is not None:
            args.command = "sweep" if args.preset == "full" else args.preset
        else:
            parser.error("please provide a subcommand: smoke, quick, or sweep")
    return args


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.command == "viz":
        run_ascii_viz(args, device)
        return
    preset = "full" if args.command == "sweep" else args.command
    suite = build_suite(preset, device, dataset_override=args.dataset, args=args)
    ds = load_dataset(suite.dataset, suite.dataset_kwargs)
    spec = ds.spec()

    base_train_cfg = TrainConfig(**asdict(suite.train_cfg))
    base_train_cfg.num_nodes = spec.num_nodes
    base_train_cfg.node_loss_weight = float(args.node_loss_weight)

    runs = []
    for run in suite.runs:
        model_cfg = apply_model_overrides(run.model_cfg, args)
        if model_cfg.event_dim is None:
            model_cfg.event_dim = spec.event_dim
        runs.append(
            SweepRun(
                name=run.name,
                model_cfg=model_cfg,
                lr=run.lr,
                weight_decay=run.weight_decay,
                num_neg=run.num_neg,
                tbptt_steps=run.tbptt_steps,
                seed=run.seed,
            )
        )

    if args.max_runs is not None:
        runs = runs[: args.max_runs]
    epochs = suite.epochs if args.epochs is None else args.epochs

    print(
        f"Preset={preset} dataset={spec.name} device={device.type} "
        f"runs={len(runs)} epochs={epochs}"
    )

    results: list[RunResult] = []
    for run in runs:
        print(f"run {short_run_label(run)} | seed={run.seed}")
        results.append(
            run_one_experiment(
                ds=ds,
                spec=spec,
                base_train_cfg=base_train_cfg,
                run=run,
                build_model_fn=build_tgn_model,  # type: ignore
                epochs=epochs,
                eval_slices=suite.eval_slices,
                save_jsonl_path=args.save_jsonl or suite.save_jsonl_path,
            )
        )

    results.sort(
        key=lambda r: (
            r.best_val_loss,
            -r.best_val_mrr if not np.isnan(r.best_val_mrr) else 0.0,
        )
    )
    print("\n=== Sweep summary (sorted by best val loss) ===")
    summary_primary = (
        infer_primary_metric(
            results[0].best_snapshot.get("train_eval", {}),
            results[0].best_snapshot.get("val", {}),
            results[0].best_snapshot.get("test", {}),
        )
        if results
        else "mrr"
    )
    if summary_primary == "edge_mse":
        header = (
            f"{'method':<24} {'seed':>4} {'val_loss':>9} {'val_mse':>10} "
            f"{'val_r2':>8} {'val_nrmse':>10} {'val_skill':>10} {'test_mse':>10} {'node':>18}"
        )
    else:
        val_label = f"val_{summary_primary}"
        test_label = f"test_{summary_primary}"
        header = (
            f"{'method':<24} {'seed':>4} {'val_loss':>9} {val_label:>12} "
            f"{test_label:>12} {'node':>18}"
        )
    print(header)
    print("-" * len(header))
    for result in results[:20]:
        val_metrics = result.best_snapshot["val"]
        test_metrics = result.best_snapshot["test"]
        node_name = node_metric_name(val_metrics)
        if node_name is None:
            node_summary = "-"
        else:
            node_val = val_metrics[node_name]
            node_test = test_metrics[node_name]
            if node_name == "node_mse":
                node_summary = f"va {node_val:.3g} te {node_test:.3g}"
            else:
                node_summary = f"va {node_val:.3f} te {node_test:.3f}"
        if summary_primary == "edge_mse":
            print(
                f"{short_run_label_from_name(result.name)[:24]:<24} "
                f"{result.seed:>4d} "
                f"{result.best_val_loss:>9.4f} "
                f"{val_metrics.get('edge_mse', float('nan')):>10.4g} "
                f"{val_metrics.get('edge_r2', float('nan')):>8.3f} "
                f"{val_metrics.get('edge_nrmse', float('nan')):>10.3f} "
                f"{val_metrics.get('edge_skill_vs_persistent', float('nan')):>9.1%} "
                f"{test_metrics.get('edge_mse', float('nan')):>10.4g} "
                f"{node_summary:>18}"
            )
        else:
            print(
                f"{short_run_label_from_name(result.name)[:24]:<24} "
                f"{result.seed:>4d} "
                f"{result.best_val_loss:>9.4f} "
                f"{val_metrics.get(summary_primary, float('nan')):>12.4g} "
                f"{test_metrics.get(summary_primary, float('nan')):>12.4g} "
                f"{node_summary:>18}"
            )


if __name__ == "__main__":
    main()
