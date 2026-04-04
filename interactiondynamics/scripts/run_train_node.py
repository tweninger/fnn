# scripts/run_train_node.py

from __future__ import annotations

from pathlib import Path
import json
import torch

from core.config import ModelConfig
from models.tgn_model import build_tgn_model

from train_node import (
    NodeTrainConfig,
    make_node_runs,
    run_one_node_experiment,
    collect_node_predictions_over_time,
)

from analysis.node_regression_plots import plot_node_targets_by_feature

from data.spring_mass import SpringMassDataset, SpringMassConfig
from data.spring_ring import SpringRing2DDataset, SpringRing2DConfig
from data.one_dimension_wave_binned import WaveEquationBinnedDataset, WaveEquationBinnedConfig
from data.three_body_binned import ThreeBodyBinnedDataset, ThreeBodyBinnedConfig
from data.nbody_continuous import ChargedParticlesBinnedDataset, ChargedParticlesBinnedConfig
from data.md22_binned import MD22BinnedDataset, MD22BinnedConfig


# -----------------------------
# small helpers
# -----------------------------

def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def describe_run_result(r) -> str:
    best_test_mae = r.best_snapshot.get("test", {}).get("mae", float("nan"))
    return (
        f"{r.name} | seed={r.seed} | "
        f"best_val_mae={r.best_val_mae:.6f} @ epoch {r.best_epoch} | "
        f"best_test_mae={best_test_mae:.6f}"
    )


# -----------------------------
# experiment config
# -----------------------------

def build_physical_datasets(
    device,
    md22_npz_paths=(),
    include=("nbody", "wave", "spring_ring", "md22"),
):
    datasets = {}

    if "nbody" in include:
        cfg = ChargedParticlesBinnedConfig(
            name="nbody",
            device=device,
        )
        datasets[cfg.name] = ChargedParticlesBinnedDataset(cfg)

    if "wave" in include:
        cfg = WaveEquationBinnedConfig(
            name="wave",
            device=device,
        )
        datasets[cfg.name] = WaveEquationBinnedDataset(cfg)

    if "threebody" in include:
        cfg = ThreeBodyBinnedConfig(
            name="threebody",
            device=device,
        )
        datasets[cfg.name] = ThreeBodyBinnedDataset(cfg)

    if "spring_ring" in include:
        cfg = SpringRing2DConfig(
            name="spring_ring",
            device=device,
        )
        datasets[cfg.name] = SpringRing2DDataset(cfg)

    if "spring_mass" in include:
        cfg = SpringMassConfig(
            name="spring_mass",
            device=device,
        )
        datasets[cfg.name] = SpringMassDataset(cfg)

    if "md22" in include:
        for npz_path in md22_npz_paths:
            stem = Path(npz_path).stem

            cfg = MD22BinnedConfig(
                name=stem,
                npz_path=str(npz_path),
                device=device,
            )
            datasets[cfg.name] = MD22BinnedDataset(cfg)

    return datasets


def build_base_model_cfg(spec) -> ModelConfig:
    """
    Adjust these fields to match your actual ModelConfig definition.
    Keep this as the one place where architecture defaults live.
    """
    return ModelConfig(
        node_dim=64,
        msg_dim=64,
        event_dim=spec.event_dim,

        # baseline/default choices
        aggregator="ift",
        update="ift_update",
        scorer="mlp",

        # generic widths
        dropout=0.0,
        scorer_dropout=0.0,
        use_time_features=False,

        # task-specific flags
        task="node_regression",
        predictor="mlp_node",
    )


def build_base_train_cfg(spec, device: torch.device) -> NodeTrainConfig:
    return NodeTrainConfig(
        num_nodes=spec.num_nodes,
        lr=1e-3,
        weight_decay=1e-5,
        grad_clip=1.0,
        device=device,
        log_every=100,
        tbptt_steps=1,
        debug=False,
        loss_name="huber",
    )


# -----------------------------
# main runner
# -----------------------------

def main():
    PLOT_MODE = "all"      # "none" | "best" | "all"
    PLOT_RESULTS = PLOT_MODE in {"best", "all"}
    PLOT_BEST_RUN_ONLY = PLOT_MODE == "best"
    PLOT_ALL_RUNS = PLOT_MODE == "all"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_dir = Path("/home/akapociu/ift/interactiondynamics/results/regression")
    plot_dir = Path("/home/akapociu/ift/interactiondynamics/plots/test2")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    save_jsonl_path = out_dir / "delta_v_node_regression_nbody_sweep_results.jsonl"
    save_summary_path = out_dir / "delta_v_node_regression_nbody_sweep_summary.jsonl"

    ensure_parent(save_jsonl_path)
    ensure_parent(save_summary_path)

    md22_npz_paths = [
        "/home/akapociu/ift/interactiondynamics/data/MD_DATA/naphthalene.npz",
        # "/home/akapociu/ift/interactiondynamics/data/MD_DATA/stachyose.npz",
        # "/home/akapociu/ift/interactiondynamics/data/MD_DATA/uracil.npz",
    ]

    datasets = build_physical_datasets(
        device=device,
        md22_npz_paths=md22_npz_paths,
        include=("wave", )
        #include=("nbody", "wave", "threebody", "spring_ring", "spring_mass", "md22"),

        # include=("spring_ring", "spring_mass"),
    )

    all_results = []

    for dataset_name, ds in datasets.items():
        print(f"\n=== DATASET: {dataset_name} ===")

        spec = ds.spec()
        target_names = None
        if spec.extra is not None:
            target_names = spec.extra.get("node_target_names", None)

        print("Dataset spec:")
        print(json.dumps({
            "name": spec.name,
            "num_nodes": spec.num_nodes,
            "event_dim": spec.event_dim,
            "num_events": spec.num_events,
            "num_bins": spec.num_bins,
            "extra": spec.extra,
        }, indent=2, default=str))

        # skip datasets not yet patched for node regression
        if spec.extra is None or "node_target_dim" not in spec.extra:
            print(f"Skipping {dataset_name}: missing node_target_dim")
            continue

        base_model_cfg = build_base_model_cfg(spec)
        base_train_cfg = build_base_train_cfg(spec, device)

        runs = make_node_runs(
            base_model_cfg,
            seeds=(0,),
            aggregator=("ift",),
            upd = ("ift_update",),
            dropout=(0.0,),
            predictor_dropout=(0.0,),
            use_time_features=(False,),
            ift_kappa_param=("softplus",),
            ift_dt=(0.01, 0.05, 0.1, 0.2), #(0.01, 0.05, 0.1, 0.2),
            ift_gamma=(0.0,), #(0.0, 0.01, 0.05, 0.1),
            ift_kappa_init=(1.0,), #(0.1, 0.5, 1.0, 2.0),
            ift_kappa_cap=(False,),
            ift_kappa_max=(None,),
        )
          
        # ift_dt=(0.05,),
        # ift_gamma=(0.0,),
        # ift_kappa_init=(1.0,),
        print(f"Planned runs for {dataset_name}: {len(runs)}")
        for run in runs:
            print("  ", run.name)

        results = []
        best_result = None
        best_model = None
        best_train_cfg = None
        plot_payloads = [] if PLOT_ALL_RUNS else None

        for run in runs:
            print("\n" + "=" * 100)
            print(f"STARTING RUN: dataset={dataset_name} | run={run.name} | seed={run.seed}")
            print("=" * 100)

            try:
                result, model, train_cfg = run_one_node_experiment(
                    ds=ds,
                    spec=spec,
                    base_train_cfg=base_train_cfg,
                    run=run,
                    build_model_fn=build_tgn_model,
                    epochs=3, #EPOCHSIFJEOSIFJEOSI EPOCHSS!!!
                    save_jsonl_path=str(save_jsonl_path),
                    save_summary_path=str(save_summary_path),
                    dataset_name=spec.name,
                )

                results.append(result)
                all_results.append((dataset_name, result))
                print("FINISHED:", describe_run_result(result))

                if best_result is None or result.best_val_mae < best_result.best_val_mae:
                    best_result = result
                    if PLOT_RESULTS:
                        best_model = model
                        best_train_cfg = train_cfg

                if plot_payloads is not None:
                    plot_payloads.append((run, result, model, train_cfg))
            
            except Exception as e:
                print(f"RUN FAILED: dataset={dataset_name} | run={run.name} | seed={run.seed}")
                print(f"Reason: {type(e).__name__}: {e}")

        if not results:
            print(f"No successful runs for dataset={dataset_name}")
            continue
        
        print("\n" + "#" * 100)
        print(f"TOP RUNS BY LOWEST BEST VAL MAE — {dataset_name}")
        print("#" * 100)

        results_sorted = sorted(results, key=lambda r: r.best_val_mae)
        for r in results_sorted[:10]:
            print(describe_run_result(r))

        # plotting
        if PLOT_RESULTS:
            dataset_plot_dir = plot_dir / dataset_name
            dataset_plot_dir.mkdir(parents=True, exist_ok=True)

            if PLOT_ALL_RUNS and plot_payloads is not None:
                print(f"\nPlotting all successful runs for {dataset_name}...")
                for run, result, model, train_cfg in plot_payloads:
                    times, y_true, y_pred = collect_node_predictions_over_time(
                        model,
                        ds.bins("test"),
                        train_cfg,
                    )

                    combo_name = f"{run.model_cfg.aggregator}_{run.model_cfg.update}"

                    run_plot_dir = dataset_plot_dir / combo_name
                    run_plot_dir.mkdir(parents=True, exist_ok=True)

                    plot_node_targets_by_feature(
                        times=times,
                        y_true=y_true,
                        y_pred=y_pred,
                        out_path=str(run_plot_dir),
                        dataset_name=dataset_name,
                        target_names=target_names,
                        model_combo=combo_name,
                        y_mode="symlog",
                        symlog_linthresh=1e-3,
                    )

            elif PLOT_BEST_RUN_ONLY:
                if best_model is None or best_train_cfg is None or best_result is None:
                    raise RuntimeError(f"Best model/train_cfg not available for plotting on {dataset_name}")

                print(f"\nCollecting predictions from BEST run for {dataset_name}...")
                times, y_true, y_pred = collect_node_predictions_over_time(
                    best_model,
                    ds.bins("test"),
                    best_train_cfg,
                )

                best_plot_dir = dataset_plot_dir / "best_run"
                best_plot_dir.mkdir(parents=True, exist_ok=True)

                plot_node_targets_by_feature(
                    times=times,
                    y_true=y_true,
                    y_pred=y_pred,
                    out_path=str(best_plot_dir),
                    dataset_name=dataset_name,
                    target_names=target_names,
                    model_combo=combo_name,
                    y_mode="symlog",
                    symlog_linthresh=1e-3,
                )

if __name__ == "__main__":
    main()