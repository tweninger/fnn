from pathlib import Path

from datasets import (
    ChargedParticlesBinnedConfig, ChargedParticlesBinnedDataset,
    WaveEquationBinnedConfig, WaveEquationBinnedDataset,
    ThreeBodyBinnedConfig, ThreeBodyBinnedDataset,
    SpringRing2DConfig, SpringRing2DDataset,
    SpringMassConfig, SpringMassDataset,
    MD22BinnedConfig, MD22BinnedDataset,
    SpringWeb2DConfig, make_spring_web_variants,
)


def build_physical_datasets(
    device,
    md22_npz_paths=(),
    include=("nbody", "wave", "spring_ring", "md22", "spring_web_2d"),
):
    datasets = {}

    if "nbody" in include:
        cfg = ChargedParticlesBinnedConfig(
            name="nbody",
            device=device,
        )
        datasets[cfg.name] = ChargedParticlesBinnedDataset(cfg)

    if "wave" in include:
        base_cfg = WaveEquationBinnedConfig(
            name="wave",
            device=device,
            event_mode="thresholded",
            interaction_threshold=19.0334,
            threshold_metric="pair_accel",
            threshold_use_absolute=True,
            standardize_node_targets=False,
            target_name="dv",
        )
        datasets[base_cfg.name] = WaveEquationBinnedDataset(base_cfg)
        # wave_variants = make_wave_variants(
        #     base_cfg,
        #     event_modes=("thresholded", "all_neighbors"),
        #     threshold_metrics=("pair_accel", "rel_q", "rel_v", "pair_grad"),
        #     interaction_thresholds=(11.0496, 19.0334, 27.6114),
        #     threshold_use_absolute_options=(True, False),
        #     standardize_node_targets_options=(False, True),
        #     target_names=("dv", "delta_v", "q_xx"),
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
            event_mode="thresholded",
            threshold_metric="force_mag",
            interaction_threshold=0.0275852,
            threshold_use_absolute=True,
        )
        #datasets[cfg.name] = SpringWeb2DDataset(cfg)
        metric_to_values = {
            "force_mag": (0.0273215,),
            "extension": (-0.0112979,),
            # "distance": (0.96589,),
           # "rel_speed": (0.0321316,),
        }
        spring_web_variants = {}

        for metric, values in metric_to_values.items():
            spring_web_variants.update(
                make_spring_web_variants(
                    base_cfg,
                    topologies=("knn",),
                    radius_values=("0.05",),
                    knn_values=(4,),
                    include_ring_edges_options=(False,),
                    event_modes=("thresholded",),
                    threshold_metrics=(metric,),   # one metric at a time
                    threshold_values=values,       # only that metric's value(s)
                    target_types=("dv", "accel", "delta_x"),
                    target_horizons=(1,),
                    standardize_node_targets_options=(False,),
                )
            )

        datasets.update(spring_web_variants)

    return datasets



