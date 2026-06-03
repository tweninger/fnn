from pathlib import Path
from dataclasses import replace

from datasets.spring_mass import SpringMassDataset, SpringMassConfig
from datasets.spring_ring_2d import SpringRing2DDataset, SpringRing2DConfig
from datasets.wave_1d import (
    WaveEquationBinnedDataset,
    WaveEquationBinnedConfig,
    make_wave_variants,
)
from datasets.three_body_binned import ThreeBodyBinnedDataset, ThreeBodyBinnedConfig
from datasets.charged_particles import (
    ChargedParticlesBinnedDataset,
    ChargedParticlesBinnedConfig,
    make_charged_particle_threshold_variants,
)
from datasets.md22_binned import MD22BinnedDataset, MD22BinnedConfig
from datasets.spring_web_2d import (
    SpringWeb2DConfig,
    SpringWeb2DDataset,
    make_spring_web_variants,
)


def build_physical_datasets(
    device,
    md22_npz_paths=(),
    include=("charged_particles", "wave", "spring_ring", "md22", "spring_web_2d"),
    threshold_splits_options=(
        ("train", "val", "test"),
    ),
    include_clean_references=True,
):
    datasets = {}

    if "charged_particles" in include:
        base_cfg = ChargedParticlesBinnedConfig(
            name="charged_particles",
            device=device,
            interaction_rule="all_pairs",
            min_edges_per_bin=1,
            #distance_threshold=2.23916,
            #threshold_splits=("train", ),
        )
        datasets[base_cfg.name] = ChargedParticlesBinnedDataset(base_cfg)

        # if include_clean_references:
        #     clean_cfg = replace(base_cfg, name=f"{base_cfg.name}__clean_ref")
        #     datasets[clean_cfg.name] = ChargedParticlesBinnedDataset(clean_cfg)

        # charged_variants = make_charged_particle_threshold_variants(
        #     base_cfg,
        #     threshold_metric="k",   # or "distance_threshold"
        #     threshold_values=(
        #         0.5, #remove lowest 50 percent
        #         0.130677,
        #     ),
        #     threshold_splits_options=threshold_splits_options,
        # )

        # datasets.update(charged_variants)

    if "wave" in include:
        base_cfg = WaveEquationBinnedConfig(
            name="wave",
            device=device,
            event_mode="thresholded",
            interaction_threshold=11.05,
            threshold_metric="pair_accel",
            threshold_use_absolute=True,
            threshold_splits=("train", "val", "test"),
            standardize_node_targets=False,
            target_name="dv",
            target_horizon=1,
        )
        datasets[base_cfg.name] = WaveEquationBinnedDataset(base_cfg)

        # if include_clean_references:
        #     clean_cfg = replace(base_cfg, name=f"{base_cfg.name}__clean_ref")
        #     datasets[clean_cfg.name] = WaveEquationBinnedDataset(clean_cfg)

        # wave_variants = make_wave_variants(
        #     base_cfg,
        #     event_modes=("thresholded",),
        #     threshold_metrics=("pair_accel",),
        #     interaction_thresholds=(
        #     38.591662,
        #     14.568860,
        # ),
        #     threshold_use_absolute_options=(True,),
        #     threshold_splits_options=threshold_splits_options,
        #     standardize_node_targets_options=(False,),
        #     target_names=("dv",),
        #     target_horizons=(1,),
        # )

        # datasets.update(wave_variants)

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
            force_threshold=0.0422165,  # {50: 0.0275852, 75: 0.0422165, 90: 0.0685315}
            standardize_node_targets=False,
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

    if "spring_web_2d" in include:
        base_cfg = SpringWeb2DConfig(
            name="springweb",
            device=device,
            event_mode="all_neighbors",
            target_type="delta_v",
            target_horizon=10,
            standardize_node_targets=False,
        )
        datasets[base_cfg.name] = SpringWeb2DDataset(base_cfg)

    return datasets


def build_spring_web_target_horizon_datasets(
    device,
    *,
    base_cfg: SpringWeb2DConfig | None = None,
    target_types=("delta_v", "dv", "delta_x"),
    target_horizons=(1, 10),
    standardize_node_targets=False,
) -> dict:
    """Minimal springweb sweep over node target type and prediction horizon."""
    if base_cfg is None:
        base_cfg = SpringWeb2DConfig(
            name="springweb",
            device=device,
            event_mode="all_neighbors",
            standardize_node_targets=standardize_node_targets,
        )

    return make_spring_web_variants(
        base_cfg,
        topologies=("knn",),
        knn_values=(base_cfg.topology_k if base_cfg.topology == "knn" else 4,),
        include_ring_edges_options=(base_cfg.include_ring_edges,),
        event_modes=("all_neighbors",),
        target_types=tuple(target_types),
        target_horizons=tuple(target_horizons),
        standardize_node_targets_options=(standardize_node_targets,),
    )