#!/usr/bin/env python3
"""Verify that an FNN step matches the physical synthetic update exactly.

This is deliberately an oracle test, not a training run.  It provides the FNN
with the generator's binary adjacency, a chosen true field (and velocity for
second-order systems), and the exact pair forces plus one external raindrop.
It then compares one FNN update and field-difference readout with the same
update evaluated directly from the generator equation.

Run from the repository root:
    venv/bin/python scripts/verify_fnn_oracle_step.py
"""

from __future__ import annotations

import argparse

import torch

from interactiondynamics.core.events import EventBatch
from interactiondynamics.core.interfaces import ModelState
from interactiondynamics.data.synthetic import SyntheticDataset, SyntheticDatasetConfig
from interactiondynamics.models.fnn import FieldNeuralNetwork


def oracle_case(dynamic: str, topology: str, num_nodes: int, seed: int) -> dict[str, float]:
    """Run one exact generator/FNN comparison and return maximum errors."""
    dataset = SyntheticDataset(SyntheticDatasetConfig(
        task=dynamic,
        num_nodes=num_nodes,
        num_bins=3,
        num_episodes=3,
        events_per_bin=num_nodes * num_nodes,
        field_topology=topology,
        event_threshold=0.0,
        seed=seed,
    ))
    truth = dataset.hidden_truth()
    assert truth is not None
    adjacency = truth["adjacency"].to(dtype=torch.float32)
    params = truth["params"]
    dt = float(params["dt"])
    gamma = float(params["gamma"])
    omega = float(params.get("omega", 0.0))
    force_scale = float(params["force_scale"])

    # A non-zero arbitrary state avoids a vacuous all-zero equivalence check.
    generator = torch.Generator().manual_seed(seed)
    h = torch.randn((num_nodes, 4), generator=generator)
    v = torch.randn((num_nodes, 4), generator=generator) if dynamic != "diffusion" else torch.zeros_like(h)
    drop_node = int(torch.randint(num_nodes, (), generator=generator))
    drop_force = torch.randn((4,), generator=generator)

    src, dst = torch.nonzero(adjacency, as_tuple=True)
    pair_force = force_scale * (h[src] - h[dst])
    events = EventBatch(
        src=torch.cat((src, torch.tensor([drop_node]))),
        dst=torch.cat((dst, torch.tensor([drop_node]))),
        features=torch.cat((pair_force, drop_force.unsqueeze(0))),
        is_external=torch.cat((torch.zeros(src.numel(), dtype=torch.bool), torch.ones(1, dtype=torch.bool))),
    )

    # The FNN represents its graph as sigmoid gates.  +/-30 round the positive
    # gates to exactly 1 in float32; non-edges have no input events here.
    model = FieldNeuralNetwork(
        num_nodes=num_nodes,
        force_dim=4,
        state_dim=4,
        gamma_init=gamma,
        omega_init=omega if dynamic != "diffusion" else 0.7,
        dt=dt,
        order=1 if dynamic == "diffusion" else 2,
        force_decoder="field_difference",
        force_scale_init=force_scale,
        learn_physical_params=False,
    )
    with torch.no_grad():
        model.topology_logits.fill_(-30.0)
        model.topology_logits[adjacency.bool()] = 30.0

    state = ModelState(node=h.clone(), node_prev=v.clone(), aux={"batch_size": 1})
    fnn_state, _ = model.step(state, events)
    assert fnn_state.node is not None

    incoming = torch.zeros_like(h)
    incoming.index_add_(0, dst, pair_force)
    incoming[drop_node] += drop_force
    if dynamic == "diffusion":
        expected_h = (1.0 - gamma * dt) * h + dt * incoming
    else:
        expected_v = (1.0 - gamma * dt) * v + dt * (incoming - omega**2 * h)
        expected_h = h + dt * expected_v

    predicted_force = model.predict_event_features(fnn_state, EventBatch(src=src, dst=dst))
    expected_force = force_scale * (expected_h[src] - expected_h[dst])
    return {
        "state_max_abs_error": float((fnn_state.node - expected_h).abs().max().detach()),
        "force_max_abs_error": float((predicted_force - expected_force).abs().max().detach()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", default="ring", choices=("ring", "grid", "torus", "doorway", "swisscheese"))
    parser.add_argument("--num-nodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--dynamics", nargs="+", default=("diffusion", "wave", "coupled_oscillator"), choices=("diffusion", "wave", "coupled_oscillator"))
    args = parser.parse_args()

    failures = []
    for dynamic in args.dynamics:
        errors = oracle_case(dynamic, args.topology, args.num_nodes, args.seed)
        passed = max(errors.values()) <= args.tolerance
        print(f"{dynamic}/{args.topology} | " + " | ".join(f"{key}={value:.3e}" for key, value in errors.items()) + f" | {'PASS' if passed else 'FAIL'}")
        if not passed:
            failures.append(dynamic)
    if failures:
        raise SystemExit(f"Oracle equivalence failed for: {', '.join(failures)}")


if __name__ == "__main__":
    main()
