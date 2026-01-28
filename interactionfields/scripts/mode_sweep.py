from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Iterable, List, Optional, Mapping

import numpy as np
import torch
from tqdm.auto import tqdm


from interactionfields.core import IFConfig, train_if
from interactionfields.eval import (
    eval_rollout,
    prepare_series,
    rollout_eval,
    rollout_if_nodes_driven,
    rollout_if_nodes_free,
    rollout_if_nodes_self,
    _warm_start_from_train,
)
from interactionfields.graphs import build_graph
from interactionfields.run_experiments import (
    SIZE_PROFILES,
    make_default_if_config,
    seed_everything,
    _resolve_center_idx,
)
from interactionfields.simulate import SIMULATORS, run_simulator
from interactionfields.eval import (
    csr_bins_to_uvt,
    make_specs,
    run_experiment,
)



HORIZONS = (1, 5, 10)
METHODS = ("if-driven", "if-free", "if-self")
STATIC_DT = 1.0


class _NoProgress:
    def update(self, _n: int = 1) -> None:
        return None


@dataclass
class GraphPreset:
    name: str
    kind: str
    kwargs: Dict[str, Any]


@dataclass
class SweepConfig:
    outdir: str
    seeds: List[int]
    profile_name: str
    toy: bool
    epochs: Optional[int]
    t_bins: Optional[int]
    cut: Optional[int]
    ema_alpha: float
    static_graphs: bool
    static_graphs_full: bool


def _config_to_dict(cfg: IFConfig) -> Dict[str, Any]:
    data = asdict(cfg)
    device = data.get("device")
    data["device"] = str(device)
    return data


def _append_results_csv(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _tree_levels_for_target(n_target: int, *, branching: int = 3) -> int:
    levels = 1
    while True:
        n_nodes = (branching**levels - 1) // (branching - 1)
        if n_nodes >= n_target:
            return levels
        levels += 1


def _build_graph_presets(profile, seed: int) -> List[GraphPreset]:
    grid_h, grid_w = profile.h, profile.w
    n_target = max(50, int(profile.n))
    tree_levels = _tree_levels_for_target(n_target, branching=3)
    k_sw = max(6, int(round(np.sqrt(n_target))))
    if k_sw % 2 != 0:
        k_sw += 1
    if k_sw >= n_target:
        k_sw = max(2, n_target - (1 if (n_target - 1) % 2 == 0 else 2))
    if k_sw % 2 != 0:
        k_sw = max(2, k_sw - 1)
    k_rgg = max(6, int(round(np.sqrt(n_target) / 2)))

    return [
        GraphPreset("grid", "grid", {"m": grid_h, "n": grid_w}),
        GraphPreset("gate", "gate", {"m": grid_h, "n": grid_w, "gate_axis": "vertical"}),
        GraphPreset("torus", "torus_grid", {"m": grid_h, "n": grid_w}),
        GraphPreset("rgg", "rgg_knn", {"N": n_target, "dim": 2, "k": k_rgg, "seed": seed}),
        GraphPreset("small_world", "small_world", {"n": n_target, "k": k_sw, "beta": 0.12, "seed": seed}),
        GraphPreset("tree", "tree", {"levels": tree_levels, "branching": 3}),
    ]


def _default_forcing_kwargs(center_idx: int, kind: str) -> Optional[Dict[str, Any]]:
    if kind == "impulse":
        return {"center_idx": center_idx, "t0": 10, "amp": 2.0, "ricker_f": 1.0 / 25.0}
    if kind == "chirp":
        return {"center_idx": center_idx, "amp": 1.0, "f0": 1.0 / 120.0, "f1": 1.0 / 12.0}
    if kind == "multi_source":
        return {"centers": [center_idx], "amp": 1.0, "f0": 1.0 / 50.0, "phases": [0.0]}
    if kind == "moving_source":
        return {
            "center_idx": center_idx,
            "t0": 10,
            "amp": 2.0,
            "ricker_f": 1.0 / 25.0,
            "move_mode": "random_walk",
            "stay_prob": 0.25,
        }
    return None


def _build_sim_kwargs(sim_kind: str, *, seed: int, center_idx: int) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"seed": seed}
    if sim_kind == "faucet":
        kwargs["center_idx"] = center_idx
    forcing_kwargs = _default_forcing_kwargs(center_idx, sim_kind)
    if forcing_kwargs is not None:
        kwargs["forcing_kwargs"] = forcing_kwargs
    return kwargs


def _ensure_profile(profile_name: str, cfg: SweepConfig):
    if cfg.toy:
        profile = replace(
            SIZE_PROFILES["speedy"],
            name="toy",
            h=12,
            w=12,
            t_bins=30,
            cut=20,
            n=90,
            epochs=6,
        )
    else:
        profile = SIZE_PROFILES.get(profile_name, SIZE_PROFILES["speedy"])

    if cfg.epochs is not None:
        profile = replace(profile, epochs=int(cfg.epochs))
    if cfg.t_bins is not None:
        profile = replace(profile, t_bins=int(cfg.t_bins))
    if cfg.cut is not None:
        profile = replace(profile, cut=int(cfg.cut))

    if profile.cut >= profile.t_bins:
        profile = replace(profile, cut=max(1, profile.t_bins - 1))
    return profile


def _compute_horizon_metrics(
    x_true: np.ndarray,
    preds: Dict[str, np.ndarray],
    *,
    var_ref: float,
    horizons: Optional[Iterable[int]] = None,
) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    out: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for method, x_pred in preds.items():
        T = min(len(x_true), len(x_pred))
        if horizons is None:
            horizon_list = list(range(1, T + 1))
        else:
            horizon_list = [int(h) for h in horizons if int(h) > 0]
        method_out: Dict[str, Dict[str, Optional[float]]] = {}
        for h in horizon_list:
            h_use = min(h, T)
            if h_use <= 0:
                method_out[str(h)] = {"nmse": None, "node_corr_mean": None, "node_corr_median": None}
                continue
            m = eval_rollout(x_true[:h_use], x_pred[:h_use], var_ref=var_ref)
            method_out[str(h)] = {
                "nmse": m["nmse"],
                "node_corr_mean": m["node_corr_mean"],
                "node_corr_median": m["node_corr_median"],
            }
        if T > 0:
            m = eval_rollout(x_true[:T], x_pred[:T], var_ref=var_ref)
            method_out["full"] = {
                "nmse": m["nmse"],
                "node_corr_mean": m["node_corr_mean"],
                "node_corr_median": m["node_corr_median"],
            }
        else:
            method_out["full"] = {"nmse": None, "node_corr_mean": None, "node_corr_median": None}
        out[method] = method_out
    return out


def _flatten_metrics_for_csv(metrics: Dict[str, Dict[str, Dict[str, Optional[float]]]]) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    horizon_keys = [str(h) for h in HORIZONS] + ["full"]
    for method in METHODS:
        for hkey in horizon_keys:
            m = metrics.get(method, {}).get(hkey, {})
            row[f"{method}_nmse_h{hkey}"] = m.get("nmse")
            row[f"{method}_node_corr_mean_h{hkey}"] = m.get("node_corr_mean")
            row[f"{method}_node_corr_median_h{hkey}"] = m.get("node_corr_median")
    return row


def _summarize_static_rows(rows: Iterable[Mapping[Any, Any]]) -> Dict[str, Dict[str, float]]:
    if not rows:
        return {}
    out: Dict[str, Dict[str, float]] = {}
    by_spec: Dict[str, List[Mapping[Any, Any]]] = {}
    for r in rows:
        by_spec.setdefault(r["spec"], []).append(r)
    for spec, spec_rows in by_spec.items():
        def _mean(key: str) -> float:
            vals = [float(r[key]) for r in spec_rows if r.get(key) is not None]
            return float(np.mean(vals)) if vals else float("nan")
        out[spec] = {
            "auc_mean": _mean("auc"),
            "ap_mean": _mean("ap"),
            "jacc_prev_mean": _mean("jacc_prev"),
            "degspe_prev_mean": _mean("degspe_prev"),
            "jacc_sub_mean": _mean("jacc_sub"),
            "degspe_sub_mean": _mean("degspe_sub"),
        }
    return out


def _compute_static_graph_analysis(
    A,
    event_bins,
    *,
    seed: int,
    include_rows: bool,
) -> Dict[str, Any]:
    events_uvt = csr_bins_to_uvt(event_bins, dt=STATIC_DT, t0=0.0)
    t_bins = len(event_bins)
    if t_bins <= 1 or events_uvt.size == 0:
        return {"rows": [], "summary_by_spec": {}}

    windows = sorted({max(2, t_bins // 20), max(3, t_bins // 10), max(5, t_bins // 5)})
    weight_modes = [("count", None), ("exp_decay", None)]
    thresholds = [{"mode": "density", "density": 0.01}, {"mode": "density", "density": 0.02}]
    specs = []
    for W in windows:
        specs.extend(make_specs(
            windows=[W],
            weight_modes=[(wm, None if wm == "count" else max(2, int(round(W / 3)))) for wm, _ in weight_modes],
            thresholds=thresholds,
            directed=True,
        ))

    eval_times = sorted({max(1, t_bins // 3), max(1, (2 * t_bins) // 3), t_bins})
    horizon_bins = max(1, t_bins // 20)
    neg_ratio = 1.0

    df = run_experiment(
        A0=A,
        events_uvt=events_uvt,
        dt=STATIC_DT,
        specs=specs,
        eval_times=eval_times,
        horizon_bins=horizon_bins,
        neg_ratio=neg_ratio,
        seed=seed,
        pbar=_NoProgress(),
    )
    rows = df.to_dict(orient="records")
    summary = _summarize_static_rows(rows)
    payload = {
        "dt": STATIC_DT,
        "windows": windows,
        "thresholds": thresholds,
        "eval_times": eval_times,
        "horizon_bins": horizon_bins,
        "neg_ratio": neg_ratio,
        "summary_by_spec": summary,
    }
    if include_rows:
        payload["rows"] = rows
    return payload


def _run_one(
    graph: GraphPreset,
    sim_kind: str,
    *,
    seed: int,
    profile,
    outdir: str,
    ema_alpha: float,
    static_graphs: bool,
    static_graphs_full: bool,
    d_latent: int = 2,
) -> None:
    seed_everything(seed)

    A, meta = build_graph(graph.kind, **graph.kwargs)
    shape = getattr(A, "shape", None)
    if shape is None:
        raise RuntimeError(f"build_graph returned adjacency with shape=None for kind={graph.kind}")
    N = int(shape[0])

    h = profile.h
    w = profile.w
    center_idx = _resolve_center_idx(meta, N, h=h, w=w)

    sim_kwargs = _build_sim_kwargs(sim_kind, seed=seed, center_idx=center_idx)
    event_bins, _, _ = run_simulator(sim_kind, adj=A, t_bins=int(profile.t_bins), **sim_kwargs)

    cut = int(min(profile.cut, len(event_bins)))
    edges_train = event_bins[:cut]
    edges_holdout = event_bins[cut:]

    cfg = make_default_if_config(epochs=int(profile.epochs))
    theta, train_metrics, _hist = train_if(edges_train, num_nodes=N, d=d_latent, cfg=cfg, seed=seed)
    theta.eval()

    device = cfg.device if isinstance(cfg.device, torch.device) else torch.device(cfg.device)
    h_last, mem0_sparse = _warm_start_from_train(theta, cfg, edges_train, device)

    x_if_driven, _, _, _ = rollout_if_nodes_driven(
        theta, cfg, edges_holdout, x0=h_last, mem0=mem0_sparse, reduce_mode="sym"
    )
    x_if_free, _, _, _ = rollout_if_nodes_free(
        theta, cfg, T=len(edges_holdout), h0=h_last, mem0_sparse=mem0_sparse, reduce_mode="sym"
    )
    x_if_self, _, _, _ = rollout_if_nodes_self(
        theta,
        cfg,
        T=len(edges_holdout),
        h0=h_last,
        mem0_sparse=mem0_sparse,
        hazard_tau=1.0,
        reduce_mode="sym",
    )

    _, _, _, x_hold_true, _, var_ref = prepare_series(edges_train, edges_holdout, N, ema_alpha)
    preds = {"if-driven": x_if_driven, "if-free": x_if_free, "if-self": x_if_self}
    node_metrics = _compute_horizon_metrics(x_hold_true, preds, var_ref=var_ref)

    eval_summary = rollout_eval(
        theta=theta,
        cfg=cfg,
        y_train=edges_train,
        y_holdout=edges_holdout,
        num_nodes=N,
        ema_alpha=ema_alpha,
        seed=seed,
        do_viz=False,
        frame_kwargs=None,
    )
    edge_metrics = eval_summary.get("edge_metrics", [])

    static_metrics = None
    if static_graphs:
        static_metrics = _compute_static_graph_analysis(
            A,
            event_bins,
            seed=seed,
            include_rows=static_graphs_full,
        )

    run_dir = os.path.join(outdir, "mode_sweep", graph.name, sim_kind, f"seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.json")

    payload = {
        "seed": seed,
        "profile": profile.name,
        "graph": {"name": graph.name, "kind": graph.kind, "kwargs": graph.kwargs},
        "simulator": {"kind": sim_kind, "kwargs": sim_kwargs},
        "t_bins": int(profile.t_bins),
        "cut": int(profile.cut),
        "epochs": int(profile.epochs),
        "num_nodes": N,
        "ema_alpha": float(ema_alpha),
        "if_config": _config_to_dict(cfg),
        "train_metrics": {k: float(v) for k, v in train_metrics.items()},
        "node_metrics": node_metrics,
        "edge_metrics": edge_metrics,
        "static_graph_metrics": static_metrics,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    summary_row = {
        "profile": profile.name,
        "graph_kind": graph.kind,
        "graph_kwargs": json.dumps(graph.kwargs, sort_keys=True),
        "simulator_kind": sim_kind,
        "simulator_kwargs": json.dumps(sim_kwargs, sort_keys=True),
        "seed": seed,
        "epochs": int(profile.epochs),
        "t_bins": int(profile.t_bins),
        "cut": int(profile.cut),
        "num_nodes": N,
        "ema_alpha": float(ema_alpha),
    }
    summary_row.update(_flatten_metrics_for_csv(node_metrics))
    summary_csv = os.path.join(outdir, "summary_modes.csv")
    _append_results_csv(summary_csv, summary_row)


def run_sweep(cfg: SweepConfig) -> None:
    profile = _ensure_profile(cfg.profile_name, cfg)
    seeds = list(cfg.seeds)
    total_runs = len(seeds) * len(SIMULATORS) * len(_build_graph_presets(profile, seed=seeds[0]))
    progress = tqdm(total=total_runs, desc="mode_sweep", unit="run")
    try:
        for seed in seeds:
            graph_presets = _build_graph_presets(profile, seed=seed)
            for graph in graph_presets:
                for sim_kind in sorted(SIMULATORS.keys()):
                    print(f"Running graph={graph.name} sim={sim_kind} seed={seed}")
                    progress.set_postfix_str(f"{graph.name}/{sim_kind}/seed{seed}")
                    _run_one(
                        graph,
                        sim_kind,
                        seed=seed,
                        profile=profile,
                        outdir=cfg.outdir,
                        ema_alpha=cfg.ema_alpha,
                        static_graphs=cfg.static_graphs,
                        static_graphs_full=cfg.static_graphs_full,
                    )
                    progress.update(1)
    finally:
        progress.close()


def _parse_args(argv: Optional[Iterable[str]] = None) -> SweepConfig:
    parser = argparse.ArgumentParser(description="Mode sweep over graphs and simulators.")
    parser.add_argument("--outdir", default="results", help="Output root directory.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Explicit RNG seeds.")
    parser.add_argument("--seed-start", type=int, default=123, help="First seed (used when --seeds is omitted).")
    parser.add_argument("--num-seeds", type=int, default=10, help="Number of seeds to run from seed-start.")
    parser.add_argument("--profile", default="speedy", help="Size profile key.")
    parser.add_argument("--epochs", type=int, default=None, help="Override IF training epochs.")
    parser.add_argument("--t-bins", type=int, default=None, help="Override number of time bins.")
    parser.add_argument("--cut", type=int, default=None, help="Override train/holdout split index.")
    parser.add_argument("--ema-alpha", type=float, default=0.2, help="EMA smoothing for node series.")
    parser.add_argument("--toy", action="store_true", help="Use smaller graphs and fewer bins.")
    parser.add_argument("--static-graphs", action="store_true", help="Include static graph analysis in metrics.json.")
    parser.add_argument("--static-graphs-full", action="store_true", default=False, help="Include full static rows (larger JSON).")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.seeds is None:
        seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.num_seeds)))
    else:
        seeds = args.seeds
    static_graphs = bool(args.static_graphs or args.static_graphs_full)
    return SweepConfig(
        outdir=args.outdir,
        seeds=seeds,
        profile_name=args.profile,
        toy=args.toy,
        epochs=args.epochs,
        t_bins=args.t_bins,
        cut=args.cut,
        ema_alpha=args.ema_alpha,
        static_graphs=static_graphs,
        static_graphs_full=args.static_graphs_full,
    )


def main(argv: Optional[Iterable[str]] = None) -> None:
    cfg = _parse_args(argv)
    run_sweep(cfg)


if __name__ == "__main__":
    main()
