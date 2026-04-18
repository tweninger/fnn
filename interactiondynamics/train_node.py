# train_node.py
import itertools
import json
import random
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterable, Optional, Sequence
import matplotlib.pyplot as plt
import numpy as np
import torch
import copy

from scipy.stats import spearmanr
from core.events import EventBatch
from core.config import ModelConfig
from data.spring_mass import SpringMassDataset, SpringMassConfig
from eval.node_regression import (
    NodeEvalSlices,
    evaluate_node_stream_sliced,
    node_regression_loss_and_metrics,
)
from models.tgn_model import build_tgn_model
from analysis.node_regression_plots import plot_node_targets_by_feature

def higher_is_better(metric_name: str) -> bool:
    return metric_name in {"r2", "pearson", "spearman", "mean_node_pearson", "mean_node_spearman", "mean_cosine"}

def is_better(candidate: float, best: float, metric_name: str) -> bool:
    return candidate > best if higher_is_better(metric_name) else candidate < best
# -----------------------------
# dataclasses
# -----------------------------

@dataclass(frozen=True)
class NodeSweepRun:
    name: str
    model_cfg: ModelConfig
    lr: Optional[float] = None
    weight_decay: Optional[float] = None
    tbptt_steps: Optional[int] = None
    seed: int = 0


@dataclass
class NodeRunResult:
    dataset: Optional[str]
    name: str
    seed: int
    epochs: int
    selection_metric: str
    best_val_metric: float
    best_epoch: int
    best_snapshot: dict
    final_snapshot: dict
    analysis_val: dict
    analysis_test: dict
    wall_sec: float


@dataclass
class NodeTrainConfig:
    num_nodes: int
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    device: torch.device = torch.device("cpu")
    log_every: int = 50
    tbptt_steps: int = 1
    debug: bool = False
    loss_name: str = "mse"   # "mse" | "mae" | "huber"
    selection_metric: str = "rmse"    # start with "rmse"


# -----------------------------
# reproducibility
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# train / eval loops
# -----------------------------

def train_one_epoch_node(
    model,
    bins: Iterable[EventBatch],
    optimizer: torch.optim.Optimizer,
    cfg: NodeTrainConfig,
) -> Dict[str, float]:
    """
    One-step-ahead node regression:

    For consecutive bins (prev_bin, curr_bin):
      1) Update memory using prev_bin            state <- step(state, prev_bin)
      2) Predict node targets for curr_bin       loss <- predict_nodes(state) vs curr.node_targets
      3) Optimizer step
      4) Detach memory periodically for truncated BPTT
    """
    model.train()
    device = torch.device(cfg.device)

    state = model.init_state(batch_size=1, num_nodes=cfg.num_nodes, device=device)

    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    n_steps = 0

    # optional diagnostics for IFT
    kappa_sum = 0.0
    kappa_n = 0

    prev: Optional[EventBatch] = None
    for curr in bins:
        curr = curr.to(device)

        if prev is None:
            prev = curr
            continue

        # update state with previous bin
        state, aux = model.step(state, prev)

        if aux is not None and "kappa" in aux:
            k = aux["kappa"]
            if torch.is_tensor(k):
                kappa_sum += float(k.detach().item())
                kappa_n += 1

        if cfg.debug and state is not None and state.node is not None:
            print(
                "DEBUG node variance:",
                float(state.node.std(dim=0).mean().item()),
                "max|h|:",
                float(state.node.abs().max().item()),
            )

        if state is not None and state.node is not None and (not torch.isfinite(state.node).all()):
            raise RuntimeError("Non-finite state.node after model.step()")

        optimizer.zero_grad(set_to_none=True)

        loss, metrics = node_regression_loss_and_metrics(
            model=model,
            state=state,
            next_events=curr,
            loss_name=cfg.loss_name,
        )

        loss.backward()

        if cfg.grad_clip and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()

        if cfg.tbptt_steps and ((n_steps + 1) % cfg.tbptt_steps == 0):
            if state is not None:
                state.detach_()

        total_loss += float(loss.item())
        total_mae += float(metrics["mae"])
        total_rmse += float(metrics["rmse"])
        n_steps += 1

        if cfg.log_every and (n_steps % cfg.log_every) == 0:
            print(
                f"[step {n_steps:6d}] "
                f"loss={loss.item():.4f} "
                f"mae={metrics['mae']:.4f} "
                f"rmse={metrics['rmse']:.4f}"
            )

        prev = curr

    if n_steps == 0:
        return {"loss": 0.0, "mae": 0.0, "rmse": 0.0}

    out = {
        "loss": total_loss / n_steps,
        "mae": total_mae / n_steps,
        "rmse": total_rmse / n_steps,
    }
    if kappa_n > 0:
        out["kappa_mean"] = kappa_sum / kappa_n
    return out


# -----------------------------
# run factory / sweeps
# -----------------------------

def make_node_runs(
    base_model_cfg: ModelConfig,
    *,
    seeds: Sequence[int] = (0,),
    aggregator: Sequence[str] = ("sum", "deepsets", "settransformer"),
    upd: Sequence[str] = ("tgn_gru",),
    dropout: Sequence[float] = (0.0,),
    predictor_dropout: Sequence[float] = (0.0,),
    use_time_features: Sequence[bool] = (False,),
    ift_kappa_param: Sequence[str] = ("softplus", "exp"),
    ift_dt: Sequence[float] = (0.05,),
    ift_gamma: Sequence[float] = (0.0,),
    ift_kappa_init: Sequence[float] = (1.0,),
    ift_kappa_cap: Sequence[bool] = (False,),
    ift_kappa_max: Sequence[float | None] = (None,),
) -> list[NodeSweepRun]:
    runs: list[NodeSweepRun] = []

    for (agg, do, update_name, pdo, tf, seed) in itertools.product(
        aggregator, dropout, upd, predictor_dropout, use_time_features, seeds
    ):
        cfg = ModelConfig(**asdict(base_model_cfg))
        cfg.update = update_name          # type: ignore[attr-defined]
        cfg.aggregator = agg              # type: ignore[attr-defined]
        cfg.dropout = do
        cfg.scorer_dropout = pdo          # reuse this width/dropout field for predictor head too
        cfg.use_time_features = tf

        # if your ModelConfig does not already define these fields,
        # add them there, or dynamically attach them if your class allows it.
        cfg.task = "node_regression"      # type: ignore[attr-defined]
        cfg.predictor = "mlp_node"        # type: ignore[attr-defined]

        base_name = f"agg={agg}|update={update_name}|do={do}|pdo={pdo}|time={tf}"

        if update_name == "ift_update":
            for (kp, dt, ga, k0) in itertools.product(
                ift_kappa_param, ift_dt, ift_gamma, ift_kappa_init
            ):
                for cap in ift_kappa_cap:
                    if cap:
                        for kmax in ift_kappa_max:
                            if kmax is None:
                                continue
                            cfg2 = ModelConfig(**asdict(cfg))
                            cfg2.task = "node_regression"          # type: ignore[attr-defined]
                            cfg2.predictor = "mlp_node"            # type: ignore[attr-defined]
                            cfg2.ift_kappa_param = kp             # type: ignore[attr-defined]
                            cfg2.ift_dt = float(dt)
                            cfg2.ift_gamma = float(ga)
                            cfg2.ift_kappa = float(k0)
                            cfg2.ift_kappa_cap = True
                            cfg2.ift_kappa_max = float(kmax)

                            name2 = f"{base_name}|{kp}|dt={dt}|gamma={ga}|k0={k0}|cap={kmax}"
                            runs.append(NodeSweepRun(name=name2, model_cfg=cfg2, seed=int(seed)))
                    else:
                        cfg2 = ModelConfig(**asdict(cfg))
                        cfg2.task = "node_regression"              # type: ignore[attr-defined]
                        cfg2.predictor = "mlp_node"                # type: ignore[attr-defined]
                        cfg2.ift_kappa_param = kp                 # type: ignore[attr-defined]
                        cfg2.ift_dt = float(dt)
                        cfg2.ift_gamma = float(ga)
                        cfg2.ift_kappa = float(k0)
                        cfg2.ift_kappa_cap = False
                        cfg2.ift_kappa_max = None

                        name2 = f"{base_name}|{kp}|dt={dt}|gamma={ga}|k0={k0}|cap=none"
                        runs.append(NodeSweepRun(name=name2, model_cfg=cfg2, seed=int(seed)))
            continue

        runs.append(NodeSweepRun(name=base_name, model_cfg=cfg, seed=int(seed)))

    return runs


# -----------------------------
# one full experiment
# -----------------------------

def run_one_node_experiment(
    ds,
    spec,
    base_train_cfg: NodeTrainConfig,
    run: NodeSweepRun,
    build_model_fn: Callable,
    epochs: int = 5,
    save_jsonl_path: Optional[str] = None,
    save_summary_path: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> tuple[NodeRunResult, torch.nn.Module, NodeTrainConfig]:
    device = torch.device(base_train_cfg.device)
    set_seed(run.seed)

    train_cfg = NodeTrainConfig(**asdict(base_train_cfg))
    if run.lr is not None:
        train_cfg.lr = run.lr
    if run.weight_decay is not None:
        train_cfg.weight_decay = run.weight_decay
    if run.tbptt_steps is not None:
        train_cfg.tbptt_steps = run.tbptt_steps

    model = build_model_fn(spec, run.model_cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    selection_metric = train_cfg.selection_metric
    best_val_metric = float("-inf") if higher_is_better(selection_metric) else float("inf")
    best_epoch = -1
    best_snapshot: dict = {}
    best_model_state = None
    
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_stats_step = train_one_epoch_node(model, ds.bins("train"), optimizer, train_cfg)

        train_eval = evaluate_node_stream_sliced(model, ds.bins("train"), train_cfg, slices=NodeEvalSlices(early_steps=10))
        val_stats  = evaluate_node_stream_sliced(model, ds.bins("val"),   train_cfg, slices=NodeEvalSlices(early_steps=10))
        test_stats = evaluate_node_stream_sliced(model, ds.bins("test"),  train_cfg, slices=NodeEvalSlices(early_steps=10))

        snapshot = {
            "epoch": epoch,
            "train_step": train_stats_step,
            "train_eval": train_eval,
            "val": val_stats,
            "test": test_stats,
        }

        cur_val_metric = float(val_stats[selection_metric])

        km = train_stats_step.get("kappa_mean", None)
        kappa_str = f"{km:.4f}" if km is not None else ""
        print(
            f"[{run.name} | seed={run.seed} | epoch {epoch:03d}] "
            f"train(step) loss={train_stats_step['loss']:.4f} "
            f"rmse={train_stats_step['rmse']:.4f} "
            f"kappa={kappa_str} | "
            f"val {selection_metric}={cur_val_metric:.4f} "
            f"test rmse={test_stats['rmse']:.4f}"
        )

        if is_better(cur_val_metric, best_val_metric, selection_metric):
            best_val_metric = cur_val_metric
            best_epoch = epoch
            best_snapshot = snapshot
            best_model_state = copy.deepcopy(model.state_dict())

        if save_jsonl_path is not None:
            row = {
                "dataset": dataset_name,
                "run": run.name,
                "seed": run.seed,
                "task": getattr(run.model_cfg, "task", "node_regression"),
                "predictor": getattr(run.model_cfg, "predictor", "mlp_node"),
                "model_cfg": asdict(run.model_cfg),
                "train_cfg_overrides": {
                    k: v
                    for k, v in {
                        "lr": run.lr,
                        "weight_decay": run.weight_decay,
                        "tbptt_steps": run.tbptt_steps,
                    }.items()
                    if v is not None
                },
                **snapshot,
            }
            with open(save_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

    wall = time.time() - t0
    final_snapshot = snapshot
    
    analysis_val = {}
    analysis_test = {}     

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        val_pack = collect_node_predictions_over_time(model, ds.bins("val"), train_cfg)
        test_pack = collect_node_predictions_over_time(model, ds.bins("test"), train_cfg)

        if len(val_pack) == 4:
            _, y_true_val, y_pred_val, node_mask_val = val_pack
        else:
            _, y_true_val, y_pred_val = val_pack
            node_mask_val = None

        if len(test_pack) == 4:
            _, y_true_test, y_pred_test, node_mask_test = test_pack
        else:
            _, y_true_test, y_pred_test = test_pack
            node_mask_test = None

        analysis_val = compute_prediction_analysis(y_true_val, y_pred_val, node_mask_val)
        analysis_test = compute_prediction_analysis(y_true_test, y_pred_test, node_mask_test)

    summary = {
        "dataset": dataset_name,
        "name": run.name,
        "seed": run.seed,
        "epochs": epochs,
        "selection_metric": selection_metric,
        "best_val_metric": best_val_metric,
        "best_epoch": best_epoch,
        "best_snapshot": best_snapshot,
        "final_snapshot": final_snapshot,
        "analysis_val": analysis_val,
        "analysis_test": analysis_test,
        "wall_sec": wall,
        }

    if save_summary_path is not None:
        with open(save_summary_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")

    result = NodeRunResult(
        dataset=dataset_name,
        name=run.name,
        seed=run.seed,
        epochs=epochs,
        selection_metric=selection_metric,
        best_val_metric=best_val_metric,
        best_epoch=best_epoch,
        best_snapshot=best_snapshot,
        final_snapshot=final_snapshot,
        analysis_val=analysis_val,
        analysis_test=analysis_test,
        wall_sec=wall,
    )

    return result, model, train_cfg


@torch.no_grad()
def collect_node_predictions_over_time(model, bins, cfg):
    """
    Collect one-step-ahead node predictions across a stream.

    Assumes:
      - each curr EventBatch has curr.node_targets of shape [N, d_y]
      - model.predict_nodes(state) returns [N, d_y]
      - curr.t contains the bin timestamp for events in that bin

    Returns
    -------
    times : np.ndarray [T_scored]
    y_true : np.ndarray [T_scored, N, d_y]
    y_pred : np.ndarray [T_scored, N, d_y]
    node_mask : np.ndarray [T_scored, N] or None
        Returned only if masks are present in the stream.
    """
    model.eval()
    device = torch.device(cfg.device)

    state = model.init_state(batch_size=1, num_nodes=cfg.num_nodes, device=device)

    times = []
    y_true = []
    y_pred = []
    masks = []

    prev = None
    saw_mask = False

    for curr in bins:
        curr = curr.to(device)

        if prev is None:
            prev = curr
            continue

        # update memory with previous bin
        state, _aux = model.step(state, prev)

        # optional safety: predict on a detached clone so prediction cannot
        # accidentally mutate the live recurrent state
        state_eval = state.clone(detach=True) if state is not None else None

        pred = model.predict_nodes(state_eval)  # [N, d_y]
        true = curr.node_targets

        if true is None:
            raise ValueError("curr.node_targets is None; dataset must provide node targets")

        true = true.to(pred.device, pred.dtype)

        # record one scalar time for this bin
        if curr.t is not None:
            t_min = int(curr.t.min().item())
            t_max = int(curr.t.max().item())
            if t_min != t_max:
                raise ValueError(f"Expected one timestamp per bin, got [{t_min}, {t_max}]")
            t_val = t_min
        else:
            t_val = len(times)

        times.append(t_val)
        y_true.append(true.detach().cpu().numpy())
        y_pred.append(pred.detach().cpu().numpy())

        if curr.node_mask is not None:
            saw_mask = True
            masks.append(curr.node_mask.detach().cpu().numpy().astype(bool))
        else:
            masks.append(None)

        prev = curr

    if len(times) == 0:
        raise ValueError("No scored steps were collected. Need at least two bins in the stream.")

    times_np = np.asarray(times)
    y_true_np = np.stack(y_true, axis=0)   # [T, N, d_y]
    y_pred_np = np.stack(y_pred, axis=0)   # [T, N, d_y]

    if saw_mask:
        node_mask_np = np.stack(
            [m if m is not None else np.ones((cfg.num_nodes,), dtype=bool) for m in masks],
            axis=0,
        )  # [T, N]

        # debugging
    # times = np.asarray(times)
    # y_true = np.asarray(y_true)
    # y_pred = np.asarray(y_pred)

    # print("times shape:", times.shape)
    # print("y_true shape:", y_true.shape)
    # print("y_pred shape:", y_pred.shape)
    # print("first 5 times:", times[:5])
    # print("first 5 pred node0:", y_pred[:5, 0, :])
    # print("first 5 true node0:", y_true[:5, 0, :])
        return times_np, y_true_np, y_pred_np, node_mask_np

    return times_np, y_true_np, y_pred_np

def compute_prediction_analysis(y_true, y_pred, node_mask=None):
    """
    y_true, y_pred: [T, N, D]
    node_mask: [T, N] or None
    """
    eps = 1e-12

    if node_mask is None:
        yt = y_true.reshape(-1, y_true.shape[-1])
        yp = y_pred.reshape(-1, y_pred.shape[-1])
        mask = np.ones((y_true.shape[0], y_true.shape[1]), dtype=bool)
    else:
        mask = node_mask.astype(bool)
        yt = y_true[mask]   # [M, D]
        yp = y_pred[mask]   # [M, D]

    diff = yp - yt
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))

    yt_mean = np.mean(yt, axis=0, keepdims=True)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt_mean) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > eps else float("nan")

    # per-node trajectory metrics over time
    T, N, D = y_true.shape
    node_pearsons = []
    node_spearmans = []
    node_r2s = []

    for n in range(N):
        valid_t = mask[:, n]
        if valid_t.sum() < 2:
            continue

        for d in range(D):
            yt_nd = y_true[valid_t, n, d]
            yp_nd = y_pred[valid_t, n, d]

            if yt_nd.size < 2:
                continue

            # Pearson
            if np.std(yt_nd) > eps and np.std(yp_nd) > eps:
                r = np.corrcoef(yt_nd, yp_nd)[0, 1]
                if np.isfinite(r):
                    node_pearsons.append(float(r))

                rho = spearmanr(yt_nd, yp_nd).statistic
                if np.isfinite(rho):
                    node_spearmans.append(float(rho))

            # per-node R²
            ss_res_nd = float(np.sum((yt_nd - yp_nd) ** 2))
            ss_tot_nd = float(np.sum((yt_nd - np.mean(yt_nd)) ** 2))
            if ss_tot_nd > eps:
                node_r2s.append(float(1.0 - ss_res_nd / ss_tot_nd))

    out = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mean_node_pearson": float(np.mean(node_pearsons)) if node_pearsons else float("nan"),
        "mean_node_spearman": float(np.mean(node_spearmans)) if node_spearmans else float("nan"),
        "mean_node_r2": float(np.mean(node_r2s)) if node_r2s else float("nan"),
        "median_node_pearson": float(np.median(node_pearsons)) if node_pearsons else float("nan"),
        "median_node_spearman": float(np.median(node_spearmans)) if node_spearmans else float("nan"),
        "median_node_r2": float(np.median(node_r2s)) if node_r2s else float("nan"),
    }

    # vector metrics for D > 1
    if yt.shape[1] > 1:
        yt_norm = np.linalg.norm(yt, axis=1)
        yp_norm = np.linalg.norm(yp, axis=1)
        valid = (yt_norm > eps) & (yp_norm > eps)

        if np.any(valid):
            cos = np.sum(yt[valid] * yp[valid], axis=1) / (yt_norm[valid] * yp_norm[valid])
            cos = np.clip(cos, -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos))

            out["mean_cosine"] = float(np.mean(cos))
            out["mean_angle_deg"] = float(np.mean(angle_deg))
            out["magnitude_rmse"] = float(np.sqrt(np.mean((yp_norm - yt_norm) ** 2)))
        else:
            out["mean_cosine"] = float("nan")
            out["mean_angle_deg"] = float("nan")
            out["magnitude_rmse"] = float("nan")

    return out
# -----------------------------
# example main
# -----------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = SpringMassDataset(
        SpringMassConfig(
            num_nodes=32,
            num_bins=2000,
            device=device,
        )
    )
    spec = ds.spec()

    if spec.extra is None or "node_target_dim" not in spec.extra:
        raise ValueError(
            "This dataset spec is missing spec.extra['node_target_dim']. "
            "Patch the dataset first so build_tgn_model knows the predictor output size."
        )

    base_train_cfg = NodeTrainConfig(
        num_nodes=spec.num_nodes,
        tbptt_steps=1,
        log_every=100,
        device=device,
        weight_decay=1e-3,
        lr=1e-3,
        loss_name="huber",
    )

    base_model_cfg = ModelConfig(
        node_dim=128,
        msg_dim=128,
        event_dim=spec.event_dim,
        scorer="mlp",
        scorer_hidden=256,
        aggregator="sum",
        use_time_features=False,
        dropout=0.0,
        scorer_dropout=0.0,
        encoder_hidden=256,
        update="tgn_gru",
        task="node_regression",
        predictor="mlp_node",
    )

    runs = make_node_runs(
        base_model_cfg,
        seeds=(0,),
        aggregator=("ift",),
        upd=("ift_update",),
        dropout=(0.0,),
        predictor_dropout=(0.0,),
        use_time_features=(False,),
        ift_kappa_param=("softplus",),
        ift_dt=(0.05,),
        ift_gamma=(0.0,),
        ift_kappa_init=(1.0,),
        ift_kappa_cap=(False,),
        ift_kappa_max=(None,),
    )

    print(f"Total node runs to execute: {len(runs)}")

    results: list[NodeRunResult] = []
    trained_model = None
    plot_cfg = None

    for run in runs:
        try:
            print(run)
            res, model, train_cfg = run_one_node_experiment(
                ds=ds,
                spec=spec,
                base_train_cfg=base_train_cfg,
                run=run,
                build_model_fn=build_tgn_model,
                epochs=1,
                save_jsonl_path="results/node_sweep_results.jsonl",
                save_summary_path="results/node_sweep_summary.jsonl",
                dataset_name=spec.name,
            )
            results.append(res)

            if trained_model is None:
                trained_model = model
                plot_cfg = train_cfg

        except Exception as e:
            print(
                f"FAILED: dataset={spec.name}, run={run.name}, "
                f"seed={run.seed}, error={type(e).__name__}: {e}"
            )

    if trained_model is not None and plot_cfg is not None:
        pack = collect_node_predictions_over_time(trained_model, ds.bins("test"), plot_cfg)
        if len(pack) == 4:
            times, y_true, y_pred, node_mask = pack
        else:
            times, y_true, y_pred = pack
            node_mask = None

    print("Collected shapes:", times.shape, y_true.shape, y_pred.shape)

    print(f"\n=== Top runs by best validation {base_train_cfg.selection_metric} ===")
    results_sorted = sorted(
        results,
        key=lambda r: r.best_val_metric,
        reverse=higher_is_better(base_train_cfg.selection_metric),
    )

    for r in results_sorted[:10]:
        print(
            f"{r.name} | seed={r.seed} | "
            f"best_val_{r.selection_metric}={r.best_val_metric:.6f} @epoch {r.best_epoch} | "
            f"test_rmse={r.analysis_test.get('rmse', float('nan')):.6f} | "
            f"test_r2={r.analysis_test.get('r2', float('nan')):.6f} | "
            f"test_mean_node_pearson={r.analysis_test.get('mean_node_pearson', float('nan')):.6f}"
        )

if __name__ == "__main__":
    main()