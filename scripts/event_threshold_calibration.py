#!/usr/bin/env python3
"""Calibrate observation thresholds to comparable retained-event fractions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from interactiondynamics.data.synthetic import SyntheticDataset, SyntheticDatasetConfig


DYNAMICS = ("diffusion", "wave", "coupled_oscillator")
TOPOLOGIES = ("ring", "grid", "torus", "doorway", "swisscheese")
REGIME_TARGETS = {"dense": 1.0, "medium": 0.50, "sparse": 0.20, "very_sparse": 0.05}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-calibration", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=(
            0.0,
            0.0001,
            0.00025,
            0.0005,
            0.001,
            0.002,
            0.003,
            0.005,
            0.01,
            0.02,
            0.03,
            0.05,
            0.075,
            0.10,
            0.15,
            0.20,
            0.30,
            0.40,
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-nodes", type=int, default=64)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--num-bins", type=int, default=72)
    parser.add_argument("--events-per-bin", type=int, default=257)
    parser.add_argument("--raindrop-interval", type=int, default=12)
    parser.add_argument("--synthetic-dt", type=float, default=0.10)
    parser.add_argument("--synthetic-gamma", type=float, default=0.15)
    parser.add_argument("--synthetic-omega", type=float, default=0.80)
    parser.add_argument("--synthetic-force-scale", type=float, default=0.80)
    return parser.parse_args()


def observed_internal_events_per_bin(dataset: SyntheticDataset) -> float:
    counts: list[int] = []
    for split in ("train", "val", "test"):
        for batch in dataset.bins(split):
            if batch.is_external is None:
                counts.append(batch.num_events)
            else:
                counts.append(int((~batch.is_external).sum().item()))
    return sum(counts) / len(counts)


def make_config(args: argparse.Namespace, dynamic: str, topology: str, threshold: float) -> SyntheticDatasetConfig:
    return SyntheticDatasetConfig(
        task=dynamic,
        num_nodes=args.num_nodes,
        num_bins=args.num_bins,
        num_episodes=args.num_episodes,
        events_per_bin=args.events_per_bin,
        raindrop_interval=args.raindrop_interval,
        event_threshold=threshold,
        dt=args.synthetic_dt,
        gamma=args.synthetic_gamma,
        omega=None if dynamic == "diffusion" else args.synthetic_omega,
        force_scale=args.synthetic_force_scale,
        field_topology=topology,
        seed=args.seed,
    )


def main() -> None:
    args = parse_args()
    thresholds = sorted(set(args.thresholds))
    if not thresholds or thresholds[0] != 0.0:
        raise ValueError("--thresholds must include 0.0 for the dense reference.")
    if any(threshold < 0.0 or threshold > 1.0 for threshold in thresholds):
        raise ValueError("Thresholds must lie in [0, 1].")

    calibration_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    for dynamic in DYNAMICS:
        for topology in TOPOLOGIES:
            per_threshold: list[dict[str, object]] = []
            for threshold in thresholds:
                dataset = SyntheticDataset(make_config(args, dynamic, topology, threshold))
                observed_per_bin = observed_internal_events_per_bin(dataset)
                per_threshold.append({"threshold": threshold, "observed_events_per_bin": observed_per_bin})
            dense_events = float(per_threshold[0]["observed_events_per_bin"])
            for row in per_threshold:
                row.update(
                    dynamic=dynamic,
                    topology=topology,
                    seed=args.seed,
                    retained_event_fraction=(float(row["observed_events_per_bin"]) / dense_events if dense_events else 0.0),
                )
                calibration_rows.append(row)

            available = list(per_threshold)
            for regime, target_fraction in REGIME_TARGETS.items():
                selected = min(
                    available,
                    key=lambda row: abs(float(row["observed_events_per_bin"]) / dense_events - target_fraction),
                )
                manifest_rows.append(
                    {
                        "dynamic": dynamic,
                        "topology": topology,
                        "regime": regime,
                        "target_retained_fraction": target_fraction,
                        "threshold": selected["threshold"],
                        "observed_events_per_bin": selected["observed_events_per_bin"],
                        "retained_event_fraction": float(selected["observed_events_per_bin"]) / dense_events,
                        "seed": args.seed,
                    }
                )
                available = [row for row in available if float(row["threshold"]) > float(selected["threshold"])]
                if not available:
                    available = [selected]

    for path, rows in ((args.output_calibration, calibration_rows), (args.output_manifest, manifest_rows)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {len(calibration_rows)} calibration rows to {args.output_calibration}")
    print(f"Wrote {len(manifest_rows)} selected regimes to {args.output_manifest}")


if __name__ == "__main__":
    main()
