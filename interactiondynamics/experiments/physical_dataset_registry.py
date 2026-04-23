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
            obs_edge_keep_prob=1.0,
            min_edges_per_bin=1,
            threshold_splits=("train", "val", "test"),
        )
        datasets[base_cfg.name] = ChargedParticlesBinnedDataset(base_cfg)

        if include_clean_references:
            clean_cfg = replace(base_cfg, name=f"{base_cfg.name}__clean_ref")
            datasets[clean_cfg.name] = ChargedParticlesBinnedDataset(clean_cfg)

        # charged_variants = make_charged_particle_threshold_variants(
        #     base_cfg,
        #     threshold_metric="force_threshold",   # or "distance_threshold"
        #     threshold_values=(
        #         0.395001,
        #         0.024478,
        #         #0.000000,
        #     ),
        #     threshold_splits_options=threshold_splits_options,
        # )

        # datasets.update(charged_variants)

    if "wave" in include:
        base_cfg = WaveEquationBinnedConfig(
            name="wave",
            device=device,
            event_mode="all_neighbors",
            interaction_threshold=19.0334,
            threshold_metric="pair_accel",
            threshold_use_absolute=True,
            threshold_splits=("train", "val", "test"),
            standardize_node_targets=False,
            target_name="dv",
            target_horizon=1,
        )
        datasets[base_cfg.name] = WaveEquationBinnedDataset(base_cfg)

        if include_clean_references:
            clean_cfg = replace(base_cfg, name=f"{base_cfg.name}__clean_ref")
            datasets[clean_cfg.name] = WaveEquationBinnedDataset(clean_cfg)

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
            threshold_metric="force_mag",
            interaction_threshold=0.0275852,
            threshold_use_absolute=True,
            threshold_splits=("train", "val", "test"),
            target_type="dv",
            target_horizon=1,
            standardize_node_targets=False,
        )
        datasets[base_cfg.name] = SpringWeb2DDataset(base_cfg)

        if include_clean_references:
            clean_cfg = replace(base_cfg, name=f"{base_cfg.name}__clean_ref")
            datasets[clean_cfg.name] = SpringWeb2DDataset(clean_cfg)

        # spring_web_variants = {}

        # spring_web_variants.update(
        #     make_spring_web_variants(
        #         base_cfg,
        #         topologies=("knn",),
        #         radius_values=(0.05,),
        #         knn_values=(4,),
        #         include_ring_edges_options=(False,),
        #         event_modes=("thresholded",),
        #         threshold_metrics=("force_mag",),
        #         threshold_values=(
        #             0.074443,
        #             0.029201,
        #         ),
        #         threshold_splits_options=threshold_splits_options,
        #         target_types=("dv",),
        #         target_horizons=(1,),
        #         standardize_node_targets_options=(False,),
        #     )
        # )

        # datasets.update(spring_web_variants)

    return datasets