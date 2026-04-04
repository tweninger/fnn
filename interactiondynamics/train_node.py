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
    best_val_mae: float
    best_epoch: int
    best_snapshot: dict
    final_snapshot: dict
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
    loss_name: str = "huber"   # "mse" | "mae" | "huber"


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

    best_val_mae = float("inf")
    best_epoch = -1
    best_snapshot: dict = {}

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

        km = train_stats_step.get("kappa_mean", None)
        kappa_str = f"{km:.4f}" if km is not None else ""
        print(
            f"[{run.name} | seed={run.seed} | epoch {epoch:03d}] "
            f"train(step) mae={train_stats_step['mae']:.4f} "
            f"kappa={kappa_str} | "
            f"train(eval) mae={train_eval['mae']:.4f} "
            f"val mae={val_stats['mae']:.4f} "
            f"test mae={test_stats['mae']:.4f}"
        )

        # lower is better now
        if val_stats["mae"] < best_val_mae:
            best_val_mae = float(val_stats["mae"])
            best_epoch = epoch
            best_snapshot = snapshot

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

    summary = {
        "dataset": dataset_name,
        "name": run.name,
        "seed": run.seed,
        "epochs": epochs,
        "best_val_mae": best_val_mae,
        "best_epoch": best_epoch,
        "best_snapshot": best_snapshot,
        "final_snapshot": final_snapshot,
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
        best_val_mae=best_val_mae,
        best_epoch=best_epoch,
        best_snapshot=best_snapshot,
        final_snapshot=final_snapshot,
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
    """
    model.eval()
    device = torch.device(cfg.device)

    state = model.init_state(batch_size=1, num_nodes=cfg.num_nodes, device=device)

    times = []
    y_true = []
    y_pred = []

    prev = None
    for curr in bins:
        curr = curr.to(device)

        if prev is None:
            prev = curr
            continue

        # update memory with previous bin
        state, _aux = model.step(state, prev)

        # predict node targets for current bin
        pred = model.predict_nodes(state)  # [N, d_y]
        true = curr.node_targets

        if true is None:
            raise ValueError("curr.node_targets is None; dataset must provide node targets")

        # record a scalar time for this bin
        if curr.t is not None:
            t_val = int(curr.t[0].item())
        else:
            t_val = len(times)

        times.append(t_val)
        y_true.append(true.detach().cpu().numpy())
        y_pred.append(pred.detach().cpu().numpy())

        prev = curr
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

    return (
        np.asarray(times),
        np.stack(y_true, axis=0),   # [T, N, d_y]
        np.stack(y_pred, axis=0),   # [T, N, d_y]
    )

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
        times, y_true, y_pred = collect_node_predictions_over_time(
            trained_model,
            ds.bins("test"),
            plot_cfg,
        )

    print("Collected shapes:", times.shape, y_true.shape, y_pred.shape)

    plot_spring_mass_quartile_nodes(
        times,
        y_true,
        y_pred,
        out_dir="/home/akapociu/ift/interactiondynamics/plots",
    )

    print("\n=== Top runs by lowest best_val_mae ===")
    results_sorted = sorted(results, key=lambda r: r.best_val_mae)
    for r in results_sorted[:10]:
        best_test_mae = r.best_snapshot.get("test", {}).get("mae", float("nan"))
        print(
            f"{r.name} | seed={r.seed} | "
            f"best_val_mae={r.best_val_mae:.6f} @epoch {r.best_epoch} | "
            f"best_test_mae={best_test_mae:.6f}"
        )

if __name__ == "__main__":
    main()