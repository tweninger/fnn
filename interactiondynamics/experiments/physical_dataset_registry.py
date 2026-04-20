from pathlib import Path

from datasets.spring_mass import SpringMassDataset, SpringMassConfig
from datasets.spring_ring_2d import SpringRing2DDataset, SpringRing2DConfig
from datasets.wave_1d import WaveEquationBinnedDataset, WaveEquationBinnedConfig, make_wave_variants
from datasets.three_body_binned import ThreeBodyBinnedDataset, ThreeBodyBinnedConfig
from datasets.charged_particles import ChargedParticlesBinnedDataset, ChargedParticlesBinnedConfig, make_charged_particle_threshold_variants
from datasets.md22_binned import MD22BinnedDataset, MD22BinnedConfig
from datasets.spring_web_2d import SpringWeb2DConfig, SpringWeb2DDataset, make_spring_web_variants


def build_physical_datasets(
    device,
    md22_npz_paths=(),
    include=("charged_particles", "wave", "spring_ring", "md22", "spring_web_2d"),
):
    datasets = {}

    if "charged_particles" in include:
        base_cfg = ChargedParticlesBinnedConfig(
            name="charged_particles",
            device=device,
            interaction_rule="all_pairs",
            obs_edge_keep_prob=1.0,
            min_edges_per_bin=1,
        )
        #datasets[base_cfg.name] = ChargedParticlesBinnedDataset(base_cfg)

        # Example thresholded variants:
        # plug in whatever threshold values you computed from your kept-edge percentages
        charged_variants = make_charged_particle_threshold_variants(
            base_cfg,
            threshold_metric="force_threshold",   # or ("distance_threshold",)
            threshold_values=(
                # replace with your actual threshold sweep
                0.046366,
                0.033596,
                0.024478,
                0.015570,
                0.000000,
            ),
        )

        datasets.update(charged_variants)

    if "wave" in include:
        base_cfg = WaveEquationBinnedConfig(
            name="wave",
            device=device,
            event_mode="all_neighbors",
            interaction_threshold=19.0334,
            threshold_metric="pair_accel",
            threshold_use_absolute=True,
            standardize_node_targets=False,
            target_name="dv",
        )
        #datasets[base_cfg.name] = WaveEquationBinnedDataset(base_cfg)
        wave_variants = make_wave_variants(
            base_cfg,
            event_modes=("thresholded", ),
            threshold_metrics=("pair_accel", ),
            interaction_thresholds=(
                14.568860,
                11.055257,
                8.003366,
                5.271941,
                2.692604,
            ),
            threshold_use_absolute_options=(True,),
            standardize_node_targets_options=(False, ),
            target_names=("dv", ),
            target_horizons=(1,),
        )

        datasets.update(wave_variants)

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
            force_threshold=0.0422165, #{50: 0.0275852, 75: 0.0422165, 90: 0.0685315}
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
        )
        #datasets[base_cfg.name] = SpringWeb2DDataset(base_cfg)
        # metric_to_values = {
        #     "force_mag": (0.0273215,),
        #     "extension": (-0.0112979,),
        #     # "distance": (0.96589,),
        #    # "rel_speed": (0.0321316,),
        # }
        spring_web_variants = {}

        # for metric, values in metric_to_values.items():
        spring_web_variants.update(
            make_spring_web_variants(
                base_cfg,
                topologies=("knn",),
                radius_values=("0.05",),
                knn_values=(4,),
                include_ring_edges_options=(False,),
                event_modes=("thresholded",),
                threshold_metrics=("force_mag",),   
                threshold_values=(
                    0.029201,
                    0.022366,
                    0.016415,
                    0.011040,
                    0.005650,
                ),      
                target_types=("dv",),
                target_horizons=(1,),
                standardize_node_targets_options=(False,),
            )
        )

        datasets.update(spring_web_variants)

    return datasets



