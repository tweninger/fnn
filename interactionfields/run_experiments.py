from __future__ import annotations

import os
import csv
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Sequence, Any

import numpy as np
import torch

from interactionfields.core import IFConfig, train_if, summarize_param_sizes
from interactionfields.graphs import build_graph
from interactionfields.eval import rollout_eval, print_epoch_history
from interactionfields.plot_results import plot_rollout, make_plot_kwargs
from interactionfields.simulate import run_simulator

@dataclass
class SizeProfile:
    name: str
    # grid-ish sizes
    h: int
    w: int
    t_bins: int
    cut: int
    # generic N for non-grid graphs
    n: int
    # training knobs
    epochs: int

SIZE_PROFILES: Dict[str, SizeProfile] = {
    # quick sanity checks (fast render, fast training)
    "speedy": SizeProfile(name="speedy", h=16, w=16, t_bins=40, cut=28, n=120, epochs=8),
    # dev sized (reasonable visuals, quicker iterations)
    "dev":    SizeProfile(name="dev",    h=28, w=28, t_bins=60, cut=45, n=250, epochs=15),
    # “real-ish” evaluation (still manageable but non-trivial)
    "eval":   SizeProfile(name="eval",   h=48, w=48, t_bins=90, cut=70, n=600, epochs=25),
}

@dataclass
class ExperimentSpec:
    name: str
    graph_kind: str
    graph_kwargs: dict
    simulator_kind: str
    simulator_kwargs: dict

def seed_everything(seed: int) -> None:
    """
    Set RNG seeds for numpy, Python, and torch for reproducibility.

    :param seed: Seed value used for all RNGs.
    :type seed: int
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _resolve_center_idx(meta: dict, N: int, *, h: Optional[int], w: Optional[int]) -> int:
    """
    Prefer true grid center if shape exists; else fallback to 0.

    :param meta: Graph metadata, optionally containing a "shape" tuple (m, n).
    :type meta: dict
    :param N: Total number of nodes.
    :type N: int
    :param h: Optional grid height if not in meta.
    :type h: Optional[int]
    :param w: Optional grid width if not in meta.
    :type w: Optional[int]
    :return: Center node index for grid-like graphs; 0 if no grid shape is available.
    :rtype: int
    """
    shape = meta.get("shape", None)
    if shape is not None and len(shape) == 2:
        hh, ww = int(shape[0]), int(shape[1])
        return (hh // 2) * ww + (ww // 2)
    if h is not None and w is not None and (h * w == N):
        return (h // 2) * w + (w // 2)
    return 0

def _make_outdir(root: str, profile: str, exp_name: str, seed: int) -> str:
    """
    Create (if needed) and return the experiment output directory.

    :param root: Base output directory.
    :type root: str
    :param profile: Size profile name used in the path.
    :type profile: str
    :param exp_name: Experiment name used in the path.
    :type exp_name: str
    :param seed: RNG seed used in the path.
    :type seed: int
    :return: Full output path for this run.
    :rtype: str
    """
    path = os.path.join(root, profile, exp_name, f"seed{seed}")
    os.makedirs(path, exist_ok=True)
    return path


def _append_results_csv(path: str, row: Dict[str, Any]) -> None:
    """
    Append a single results row to a CSV file, creating it with headers if needed.

    :param path: CSV file path.
    :type path: str
    :param row: Row data to append (flat dict).
    :type row: Dict[str, Any]
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def make_default_if_config(*, epochs: int) -> IFConfig:
    return IFConfig(
        mode="diffusion",
        dt=1.0,
        epochs=int(epochs),
        alpha=0.05,
        k=12,
        s_neg=64,
        tau_mem=10.0,
        eps_kernel=1e-3,
        use_candidates_diffusion=False,
        link="softplus",
        lr=1e-2,
        weight_decay=0.0,
    )

def build_experiment_suite(profile: SizeProfile, *, seed: int) -> List[ExperimentSpec]:
    """
    Construct the default suite of (graph, simulator) experiment specifications.

    Notes:
      - Grid-like graphs use profile.h/profile.w for shape and are compatible with 3D rollout plots.
      - Non-grid graphs rely on coords2d/coords3d in graph metadata for visualization.
      - Simulators are chosen to provide diverse dynamics across graph families.
    """

    # Grid family
    grid_h, grid_w = profile.h, profile.w

    suite: List[ExperimentSpec] = [
        ExperimentSpec(
            name="grid_dripping_wave",
            graph_kind="grid",
            graph_kwargs={"m": grid_h, "n": grid_w},
            simulator_kind="dripping_wave",
            simulator_kwargs=dict(
                dt=0.05,
                faucet_period=1.0,
                faucet_amp=1.0,
                faucet_width=0.08,
                source_kind="gaussian",
                edge_activation="absdiff",
                edge_threshold=0.0,
                return_states=True,
                seed=seed,
            ),
        ),
        ExperimentSpec(
            name="gate_dripping_wave",
            graph_kind="gate",
            graph_kwargs={"m": grid_h, "n": grid_w, "gate_axis": "vertical"},
            simulator_kind="dripping_wave",
            simulator_kwargs=dict(
                dt=0.05,
                faucet_period=1.0,
                faucet_amp=1.0,
                faucet_width=0.08,
                source_kind="gaussian",
                edge_activation="absdiff",
                edge_threshold=0.0,
                return_states=True,
                seed=seed,
            ),
        ),
        ExperimentSpec(
            name="torus_surface_dripping_wave",
            graph_kind="torus_surface",
            graph_kwargs={"m": grid_h, "n": grid_w, "R": 3.0, "r0": 1.0},
            simulator_kind="dripping_wave",
            simulator_kwargs=dict(
                dt=0.05,
                faucet_period=1.0,
                faucet_amp=1.0,
                faucet_width=0.08,
                source_kind="gaussian",
                edge_activation="absdiff",
                edge_threshold=0.0,
                return_states=True,
                seed=seed,
            ),
        ),

        # Non-grid but still clean/visualizable
        ExperimentSpec(
            name="sbm_sis",
            graph_kind="sbm",
            graph_kwargs={"sizes": (max(30, profile.n // 4),) * 3, "p_in": 0.16, "p_out": 0.02, "seed": seed},
            simulator_kind="sis",
            simulator_kwargs=dict(
                beta=0.10,
                mu=0.04,
                init_infected=0.02,
                seed=seed,
                return_states=True,
            ),
        ),
        ExperimentSpec(
            name="small_world_hawkes",
            graph_kind="small_world",
            graph_kwargs={"n": max(120, profile.n), "k": 10, "beta": 0.12, "seed": seed},
            simulator_kind="hawkes_edges",
            simulator_kwargs=dict(
                base_rate=2e-4,
                alpha=0.9,
                beta=0.25,
                neighbor_coupling=0.15,
                seed=seed,
            ),
        ),
        ExperimentSpec(
            name="rgg_threshold",
            graph_kind="rgg_knn",
            graph_kwargs={"N": max(120, profile.n), "dim": 2, "k": 8, "seed": seed},
            simulator_kind="threshold",
            simulator_kwargs=dict(
                theta=0.22,
                mu_off=0.01,
                init_on=0.03,
                seed=seed,
                return_states=True,
            ),
        ),
    ]

    return suite

def run_one(
    exp: ExperimentSpec,
    profile: SizeProfile,
    *,
    seed: int,
    out_root: str = "exports_suite",
    dt_grid: Tuple[float, ...] = (0.5, 1.0, 2.0, 3.0),
    ema_alpha: float = 0.2,
    do_quick_viz: bool = True,
) -> None:
    """
    Run a single experiment end-to-end: build graph, simulate events, train IF, and evaluate.

    Uses profile settings to size the graph, choose train/holdout split, and set epochs.
    """
    seed_everything(seed)

    # 1) Build graph.
    A, meta = build_graph(exp.graph_kind, **exp.graph_kwargs)
    shape = getattr(A, "shape", None)
    if shape is None:
        raise RuntimeError(f"build_graph returned adjacency with shape=None for kind={exp.graph_kind}")
    N = int(shape[0])

    # 2) Choose plotting kwargs 
    h = profile.h
    w = profile.w
    frame_kwargs = None
    if exp.graph_kind in ("grid", "gate", "torus_surface", "torus_grid", "cylinder_x", "cylinder_y"):
        # Ensure h,w match meta shape if present.
        if "shape" in meta:
            h, w = int(meta["shape"][0]), int(meta["shape"][1])
        frame_kwargs, _anim_kwargs = make_plot_kwargs(exp.graph_kind, h, w, meta, A, z_exaggeration=0.8)

    # 3) Run simulator.
    sim_kind = exp.simulator_kind
    sim_kw = dict(exp.simulator_kwargs)  # copy
    sim_kw["adj"] = A
    sim_kw["t_bins"] = int(profile.t_bins)

    # Fill center defaults if needed.
    if sim_kind in ("dripping_wave", "impulse", "chirp", "moving_source", "multi_source"):
        center_idx = _resolve_center_idx(meta, N, h=h, w=w)
        # Dripping wave uses faucet_nodes; field presets may use forcing_kwargs[center_idx].
        if sim_kind == "dripping_wave":
            sim_kw.setdefault("faucet_nodes", [center_idx])

    event_bins, H, _sim_meta = run_simulator(sim_kind, **sim_kw)

    # 4) Split train/holdout.
    cut = int(min(profile.cut, len(event_bins)))
    edges_train = event_bins[:cut]
    edges_holdout = event_bins[cut:]

    # 5) Optional viz (only if grid-ish and we have states).
    outdir = _make_outdir(out_root, profile.name, exp.name, seed)

    if do_quick_viz and (frame_kwargs is not None) and (H.size > 0):
        plot_rollout(
            variants={"sim": {"X": H, "edges": event_bins}},
            h=h, w=w,
            t=None,
            outdir=os.path.join(outdir, "viz"),
            reference_nodes=None,
            reference_edges=None,
            do_nodes_panels=False,
            do_edges_activity=False,
            do_edges_residual=False,
            do_edges_confusion=False,
            do_combined_activity=True,
            do_nodes_residual=False,
            cm_nodes="seismic",
            cm_edges_activity="Greys",
            z_exaggeration=0.8,
            frame_kwargs=frame_kwargs,
        )

    # 6) Train IF on edges_train.
    cfg = make_default_if_config(epochs=profile.epochs)

    theta, metrics, hist = train_if(edges_train, num_nodes=N, d=2, cfg=cfg, seed=seed)
    print(f"\n=== {exp.name} [{profile.name}] ===")
    print("Trainable params:", theta.parameter_count())
    print("By group:", theta.parameter_count(by_group=True))
    summarize_param_sizes(theta)
    theta.eval()

    print_epoch_history(seed, hist, metrics)

    # 7) Evaluation / rollouts.
    eval_summary = rollout_eval(
        theta=theta,
        cfg=cfg,
        y_train=edges_train,
        y_holdout=edges_holdout,
        num_nodes=N,
        ema_alpha=ema_alpha,
        dt_grid=dt_grid,
        seed=seed,
        frame_kwargs=frame_kwargs,
        do_viz=do_quick_viz
    )

    # 8) append summary row to global results CSV
    results_path = os.path.join(out_root, "results.csv")
    row = {
        "profile": profile.name,
        "exp_name": exp.name,
        "graph_kind": exp.graph_kind,
        "simulator_kind": exp.simulator_kind,
        "seed": seed,
        "epochs": profile.epochs,
        "h": profile.h,
        "w": profile.w,
        "t_bins": profile.t_bins,
        "cut": profile.cut,
        "n": profile.n,
        "dt_grid": json.dumps(list(dt_grid)),
        "ema_alpha": ema_alpha,
        "graph_kwargs": json.dumps(exp.graph_kwargs, sort_keys=True),
        "simulator_kwargs": json.dumps(exp.simulator_kwargs, sort_keys=True),
    }
    for k, v in metrics.items():
        row[f"train_{k}"] = v
    row["eval_dt_star"] = eval_summary.get("dt_star")
    row["eval_gamma_star"] = eval_summary.get("gamma_star")
    row["eval_dt_free_star"] = eval_summary.get("dt_free_star")
    row["eval_gamma_free_star"] = eval_summary.get("gamma_free_star")
    row["eval_node_metrics"] = json.dumps(eval_summary.get("node_metrics", []))
    row["eval_edge_metrics"] = json.dumps(eval_summary.get("edge_metrics", []))
    _append_results_csv(results_path, row)


def _main() -> None:
    """
    Entry point for running the experiment suite.

    Configuration:
      - IFT_PROFILE selects a size profile (speedy/dev/eval).
      - IFT_SEED sets RNG seed.
      - IFT_OUT sets the output root directory.
    """
    # Choose "speedy" for fast debugging, "dev" for medium work, "eval" for full runs.
    profile_name = os.environ.get("IFT_PROFILE", "speedy")
    profile: SizeProfile = SIZE_PROFILES.get(profile_name, SIZE_PROFILES["dev"])

    seed = int(os.environ.get("IFT_SEED", "123"))
    out_root = os.environ.get("IFT_OUT", "exports_suite")

    # Global evaluation knobs (rollout calibration grid + EMA smoothing)
    dt_grid: Tuple[float, ...] = (0.5, 1.0, 2.0, 3.0)
    ema_alpha: float = 0.2

    # Build the suite based on the selected profile and seed.
    suite: List[ExperimentSpec] = build_experiment_suite(profile, seed=seed)

    # Run all experiments (set do_quick_viz=True for per-run plots).
    for exp in suite:
        run_one(
            exp,
            profile,
            seed=seed,
            out_root=out_root,
            dt_grid=dt_grid,
            ema_alpha=ema_alpha,
            do_quick_viz=False,
        )


if __name__ == "__main__":
    _main()
