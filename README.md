# Field Neural Networks

Physics-inspired machine learning is compelling because it gives neural models useful structure. Instead of asking a model to learn arbitrary dynamics from data alone, we can constrain the model with known physical principles and let learning focus on the unknown parts of the system.

Hamiltonian Neural Networks (HNNs), for example, learn an energy function whose gradients define motion through position and momentum. Lagrangian Neural Networks (LNNs) take a related approach by learning a Lagrangian that induces the equations of motion. In both cases, the physical object—the Hamiltonian or Lagrangian—is usually encoded through the training objective. The loss encourages the model to recover dynamics that obey the corresponding physical law.

Motion is important, but many systems are better understood as **fields**: quantities distributed over space, graphs, or interacting entities. Fields describe diffusion, wave propagation, smoothing, external forcing, damping, and other structured processes. These systems are broader than particle trajectories and are especially natural for interaction data, where signals move across nodes, edges, communities, or events.

This project explores **Field Neural Networks (FNNs)** and related interaction-field models. The central idea is to move physics-inspired structure out of the loss function and into the **state update rule**. Rather than training only to satisfy a physics residual, the model uses field-inspired updates internally, while the outer training objective remains a standard downstream task loss such as mean squared error for regression or binary cross-entropy for classification.

In other words, we use physics-inspired dynamics as the model’s inductive bias, not necessarily as the final prediction target.

This lets us train field-structured models directly on node and edge tasks:

* node regression
* edge regression
* node classification
* edge classification
* temporal rollout prediction
* synthetic diffusion, wave, oscillator, and memory benchmarks

The update functions encode mechanisms such as diffusion, forcing, damping, and second-order velocity-like dynamics. The downstream loss then asks whether those mechanisms help solve predictive tasks.

The result is a family of models that can be compared against GRUs, HNNs, LNNs, persistence baselines, and analytic synthetic baselines. As expected, field-structured models work especially well on field-style tasks: diffusion-like systems favor first-order interaction fields, while wave-like systems benefit from second-order field updates with identifiable velocity or short-history state.


## Development

```bash
venv/bin/python -m pip install -e ".[dev]"
venv/bin/python -m pytest
venv/bin/python -m ruff check
venv/bin/python -m pylint interactiondynamics interactionfields
venv/bin/pyright
```

## Training presets

Use `smoke` to check that a task and its recommended shortlist execute. Use
`quick` for a small pilot; it is not a paper-scale result. `sweep` retains
the broader legacy JODIE grid.

```bash
venv/bin/python -m interactiondynamics.train smoke --dataset synthetic --synthetic-task deepsets_sum
venv/bin/python -m interactiondynamics.train quick --dataset synthetic --synthetic-task diffusion
venv/bin/python -m interactiondynamics.train sweep
```

### Synthetic task map

“Order” refers to the order of the **data-generating dynamics**, not to an
arbitrary IFT setting selected at the command line. “N/A” means that no
field-dynamics order is meaningful for the task.

#### N/A — aggregation, selection, classification, and routing controls

These tasks test event-set processing or supervision type. They are useful
controls, but they are not evidence for a first- or second-order field claim.

| Task family | Canonical task(s) | What it tests |
| --- | --- | --- |
| Additive aggregation | `deepsets_sum`, `node_sum_regression`, `next_dst_ranking` | Sum of incident event values. |
| Keyed selection | `settransformer_max`, `node_keyed_value`, `edge_retrieval` | Select an event value by its key. |
| Associative retrieval | `associative_retrieval` | Query–key–value retrieval in a node-local event set. |
| Threshold / trigger classification | `node_count_threshold`, `node_keyed_trigger`, `edge_threshold_classification`, `edge_keyed_trigger_classification` | Classification from an incident event set. |

#### First-order — relational diffusion

| Task | Domain and mechanism | Recommended use |
| --- | --- | --- |
| `diffusion` | Driven diffusion over a selected interaction topology: ring by default, or grid-derived domains. | Main first-order mechanism test. |

The normal `diffusion` shortlist is its first-order comparison panel: three
FNN forcing variants; Sum, Deep Sets, and Set Transformer with GRU; Hopfield
with its Hopfield update; and Set Transformer with LNN and HNN updates. For a
longer pilot, run:

```bash
venv/bin/python -m interactiondynamics.train quick \
  --dataset synthetic \
  --synthetic-task diffusion \
  --num-bins 192 \
  --epochs 20 \
  --rollout-horizon 20
```

Choose the diffusion domain independently of the mechanism:

```bash
venv/bin/python -m interactiondynamics.train quick \
  --dataset synthetic \
  --synthetic-task diffusion \
  --synthetic-topology doorway
```

Available diffusion topologies are `ring` (default), `grid`, `torus`,
`doorway`, and `swiss_cheese`.

#### Second-order — inertia, velocity, and waves

| Task family | Canonical task(s) | Domain and mechanism |
| --- | --- | --- |
| Local driven oscillator | `temporal_memory`, `node_temporal_regression`, `node_temporal_state`, `edge_temporal_state`, `next_dst_temporal_ranking` | Independent self-loop systems with AR(2)-style state, velocity carry-over, and forcing; **not** graph propagation. |
| Conservative local oscillator | `conservative_oscillator` | Lightly driven, long-memory second-order oscillator. |
| Ring wave | `wave` | Driven wave propagation with neighbor coupling on a ring. |
| Topological grid waves | `wave_grid`, `wave_torus`, `wave_doorway`, `wave_swiss_cheese` | Second-order propagation over bounded, periodic, barrier, and perforated grid topologies. |

The grid-wave tasks use sparse local drives and topology-specific neighbor
events. `wave_grid` has reflecting outer boundaries, `wave_torus` wraps
both axes, `wave_doorway` adds a wall with a three-node aperture, and
`wave_swiss_cheese` removes circular patches of nodes.

### CLI reference

The training entrypoint is:

```bash
venv/bin/python -m interactiondynamics.train <smoke|quick|sweep> [flags]
```

Available subcommands:

- `smoke`: tiny toy-dataset check over the focused shortlist.
- `quick`: focused shortlist run, defaulting to JODIE Wikipedia.
- `sweep`: broader multi-seed sweep over the larger preset grid.

Common dataset flags:

- `--dataset {toy,jodie,synthetic}`: override the dataset when supported by the subcommand.
- `--synthetic-task TASK`: choose the synthetic benchmark task when `--dataset synthetic`.
- `--synthetic-topology {ring,grid,torus,doorway,swiss_cheese}`: choose the domain for `diffusion`; the default is `ring`.
- `--synthetic-drive-cutoff T`: retain the regular diffusion drive for the first `T` steps of each evaluation rollout, then compare predictions with a simulator-generated zero-drive suffix. Training data and ordinary rollout metrics are unchanged.
- `--synthetic-num-nodes N`: override synthetic node count.
- `--synthetic-events-per-bin N`: override synthetic event count per bin for set-style tasks.
- `--num-bins N`: override the number of simulated synthetic time bins.
- `--seed N`: set the synthetic-data random seed.
- `--ift-variants [VARIANT ...]`: on `smoke` or `quick`, replace the default `ift/ift_update` run with an IFT sweep over one or more variant families from `{generic, linear, direct, auto}`. Pass no variant names to sweep them all. Using this flag auto-selects the synthetic dataset.
- `--ift-orders [1 2 ...]`: optionally restrict the IFT sweep to first-order, second-order, or both. Defaults to `1 2` when IFT variants are selected.
- `--ift-history-steps [1 2 3 ...]`: optionally restrict the history readout sweep for the `auto` family. Defaults to `1 2 3`.
- `--ift-self-rollout`: add the IFT2 history model evaluated with self-generated neighbor signals and no future external drives. Supported for diffusion and wave topologies.
- `--ift-free-rollout`: add the IFT2 history model evaluated with observed neighbor signals but no future external drives. Supported for diffusion and wave topologies. Together with `--ift-self-rollout`, this separates missing forcing from self-generated relational inputs.

Common training flags:

- `--max-runs N`: cap the number of runs executed after filtering the preset.
- `--epochs N`: override the preset epoch count.
- `--rollout-horizon K`: set rollout evaluation horizon for regression tasks.
- Rollout JSONL records include `rollout_val.rollout_by_step` and `rollout_test.rollout_by_step`, keyed by relative step (`"1"`, `"2"`, …). Each entry contains the per-step regression metrics; the existing aggregate rollout fields are retained for compatibility.
- `--rollout-train-steps K`: for compatible IFT2 history-readout runs, optimize an average loss over `K` differentiable autoregressive steps before each optimizer update. With `--ift-self-rollout`, regenerated neighbor signals and removed future drives are used during this training unroll too. The default, `1`, is the original one-step trainer.
- `--use-node-scorer`: force-enable the auxiliary node scorer.
- `--node-loss-weight W`: weight for the node prediction loss.
- `--node-scorer-hidden H`: hidden width for the node scorer MLP.

Target and loss flags:

- `--node-target-mode {raw,residual}`: train node targets directly or as deltas from the previous step.
- `--edge-target-mode {raw,residual}`: train edge targets directly or as deltas from the previous step.
- `--edge-target-scale {raw,zscore}`: use raw edge regression loss or z-score scaled loss.
- `--prediction-mode {state,delta,state_plus_delta}`: predict next state directly, predict deltas, or reconstruct state from predicted deltas.

Output flags:

- `--save-jsonl PATH`: append per-epoch metrics and summaries to a JSONL file.

Compatibility note:

- The hidden legacy form `venv/bin/python -m interactiondynamics.train --preset {smoke,quick,full}` is still supported, with `full` mapping to `sweep`.
- Classification tasks require `--node-target-mode raw` and `--edge-target-mode raw`.

## IFT variants

The `ift` path supports both first-order diffusion-style updates and second-order oscillator-style variants without leaving the event-bin architecture.

- First-order IFT:
  - aggregator builds a graph/Laplacian from event bins
  - update uses diffusion, damping, and forcing
- Second-order IFT:
  - carries explicit velocity memory
  - supports scalar `h/v/force` readout for AR(2)-style diagnostics
  - supports teacher-forced or autonomous rollout evaluation

For synthetic quick/smoke runs, the shortlist normally includes one default `ift` + `ift_update` run when the task recommends it.
Passing `--ift-variants` replaces that one default IFT run with a sweep over the selected IFT axes while leaving the other quick baseline models in place.

The selector is available on all synthetic tasks.

The sweep axes are:

- order: `1`, `2`
- variant family: `generic`, `linear`, `direct`, `auto`
- history steps: `1`, `2`, `3`

How the sweep expands:

- `generic`, `linear`, and `direct` create `ift1_*` and/or `ift2_*` runs depending on `--ift-orders`
- `auto` creates `ift2_auto`
- `--ift-history-steps` adds `ift2_hist_vel_k1`, `ift2_hist_vel_k2`, and `ift2_hist_vel_k3` style runs for the `auto` family
- tasks with event features can sweep all four variant families
- zero-event tasks currently support `generic` only
- `--ift-history-steps` requires `auto`
- `auto` and history sweeps require second-order IFT, so `--ift-orders` must include `2`

Useful commands:

```bash
venv/bin/python -m interactiondynamics.train quick --synthetic-task diffusion --ift-variants linear --ift-orders 1
venv/bin/python -m interactiondynamics.train quick --synthetic-task wave --ift-variants auto --ift-history-steps 1 2 3
venv/bin/python -m interactiondynamics.train quick --synthetic-task conservative_oscillator --ift-variants --ift-orders 1 2
venv/bin/python -m interactiondynamics.train quick --synthetic-task wave --ift-variants linear auto --ift-orders 2 --ift-history-steps 3 --num-bins 40 --rollout-horizon 5
venv/bin/python -m interactiondynamics.train quick --dataset synthetic --synthetic-task edge_temporal_state --epochs 5 --num-bins 96 --synthetic-num-nodes 50 --synthetic-events-per-bin 48 --rollout-horizon 6 --ift-variants linear auto --ift-orders 2 --ift-history-steps 2
venv/bin/python -m interactiondynamics.train quick --synthetic-task deepsets_sum --ift-variants direct --ift-orders 1
venv/bin/python -m interactiondynamics.train quick --synthetic-task node_count_threshold --ift-variants
```

These runs go through the normal training/eval loop, so the output stays the standard per-run quick preset summary instead of a separate diagnostic table.
When `--ift-variants` is active, the CLI also prints an IFT diagnostic footer at the end with rollout/state metrics, learned dynamics stats, and any active linear `h/v/force` readout coefficients.

### Worked example: `edge_temporal_state`

Command:

```bash
venv/bin/python -m interactiondynamics.train quick --dataset synthetic --synthetic-task edge_temporal_state --epochs 5 --num-bins 96 --synthetic-num-nodes 50 --synthetic-events-per-bin 48 --rollout-horizon 6 --ift-variants linear auto --ift-orders 2 --ift-history-steps 2
```

Example output:

```text
=== Sweep summary (sorted by val.edge_auroc) ===
method                   seed   kind    objective          val.edge_f1      test.edge_auroc         test.edge_f1
-----------------------------------------------------------------------------------------------------------------
ift2_auto                   0   edge       0.9993               0.9774               0.9987               0.9776
ift2_hist_vel_k2            0   edge       0.9987               0.9733               0.9978               0.9736
ift2_linear                 0   edge       0.9163               0.5106               0.8885               0.4676
sum/tgn_gru                 0   edge       0.7841               0.6427               0.7583               0.6654
sum/hnn                     0   edge       0.6818                0.622               0.6294               0.6514
sum/lnn                     0   edge       0.5862             0.009881               0.6022               0.1094

=== IFT diagnostic table: edge_temporal_state ===
run                  target   auc_v   auc_t    f1_v    f1_t  state_v  state_t   roll_v   roll_t     pers   d_pers delta_r2 delta_mae   vel_r2   kappa   gamma     dt   alpha    force     diff    rel_d    rel_u   vel_f   for_f   d_corr  vel_mse
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ift2_auto              edge   0.999   0.999   0.977   0.978        -        -        -        -        -        -        -         -   -0.029   0.693   0.000  1.409   0.781    0.914    0.000    0.000    0.265   0.716   0.284        -    0.051
  coeffs | w_y=1.2855 w_v=0.6353 w_drive=1.4387 bias=-0.5171
ift2_hist_vel_k2       edge   0.999   0.998   0.973   0.974        -        -        -        -        -        -        -         -   -0.034   0.853   0.000  0.130   0.739    0.828    0.000    0.000    0.075   0.680   0.320        -    0.051
  coeffs | w_y=0.5542 w_v=0.4326 w_drive=0.5414 bias=-0.0893
ift2_linear            edge   0.916   0.889   0.511   0.468        -        -        -        -        -        -        -         -   -0.094   0.983   0.000  0.102   0.997    0.929    0.000    0.000    0.076   0.875   0.125        -    0.054
  coeffs | w_y=0.2185 w_v=0.2558 w_drive=0.2747 bias=0.0460
```

Interpretation:

- The autonomous second-order IFT variants dominate this task. `ift2_auto` is best on both validation and test AUROC/F1, and `ift2_hist_vel_k2` is a very close second.
- `ift2_linear` is much weaker than the autonomous variants, which suggests this task benefits from the richer second-order autonomous readout rather than a simpler linear forcing path alone.
- The non-IFT baselines are clearly behind here, especially on AUROC, which is the main sign that the temporal-state classification target is aligned with the IFT inductive bias.
- The blank `state_*`, `roll_*`, and delta columns are expected for classification tasks. Those diagnostics are only defined for regression targets, while classification tasks surface `auc_*` and `f1_*` instead.
- In the footer, both strong IFT runs learn positive `w_y`, `w_v`, and `w_drive`, so the classifier is using current state, velocity, and drive together. `ift2_auto` leans harder on the drive term and ends up slightly ahead of the history-based variant.

### What the field quantities mean

The IFT models maintain a latent node state over the graph induced by each event bin.

- `h_t in R^(N x d)` is the latent node state at time bin `t`, one `d`-dimensional state per node.
- The events in the current bin induce an adjacency `A_t`. In the default IFT aggregator, that adjacency is made undirected and can be built from just the current bin or from an EMA-smoothed history of previous bins.
- The normalized graph Laplacian is

```text
L_t = I - D_t^(-1/2) A_t D_t^(-1/2)
```

- `L_t h_t` is the diffusion term. If neighboring nodes have similar state, this term is small. If a node disagrees with its neighbors, this term pushes it back toward local smoothness.
- `force_t` is the externally driven part of the dynamics, built from event embeddings or structured event features such as a direct drive scalar.

First-order IFT is a damped driven diffusion update:

```text
h_(t+1) = h_t + dt * (-gamma * h_t - kappa * L_t h_t + force_t)
```

- `dt`: integration step size.
- `gamma`: decay or damping on the current state.
- `kappa`: coupling strength on the Laplacian term. Larger `kappa` means stronger smoothing / diffusion across the graph.
- `force_t`: input-driven excitation from the current event bin.

This is the right mental model for diffusion-like tasks: the state wants to smooth over the graph, decay a bit, and respond to new input.

Second-order IFT adds an explicit velocity-like latent state `v_t`:

```text
v_(t+1) = alpha * v_t - dt * (gamma * v_t + kappa * L_t h_t) + dt * force_t
h_(t+1) = h_t + dt * v_(t+1)
```

- `v_t`: latent velocity or momentum.
- `alpha`: velocity carry-over. Larger `alpha` means more inertia from the previous step.
- `gamma`: damping on velocity.
- `kappa`: restoring / coupling strength from the Laplacian field.

This is the right mental model for wave-like and oscillator-like tasks: the model is not only smoothing a state, it is carrying momentum forward while the graph field and external drive push on that motion.

For the `auto` and `hist_vel` variants, the second-order scorer exposes a simple scalar readout:

```text
score_t = w_y * y_t + w_v * v_t + w_drive * u_t + b
```

- `y_t`: current scalar state readout for the destination node.
- `v_t`: velocity scalar. In `auto`, this is the internal second-order velocity; in `hist_vel_k*`, it is replaced by a short history-based velocity estimate from the last `k` readout deltas.
- `u_t`: scalar drive / force readout from the current event bin.
- `w_y`, `w_v`, `w_drive`, `b`: the learned linear coefficients printed in the footer.

So when the footer shows positive `w_y`, `w_v`, and `w_drive`, it means larger state, velocity, and drive all push the output upward. On classification tasks like `edge_temporal_state`, that scalar is the logit for the positive class.

The other footer columns summarize the internal dynamics:

- `kappa`, `gamma`, `dt`, `alpha`: the learned physical-style parameters above.
- `force`, `diff`: average magnitudes of the forcing and Laplacian terms.
- `rel_d`: diffusion magnitude relative to forcing magnitude.
- `rel_u`: update magnitude relative to current state magnitude.
- `vel_f`, `for_f`: how much of the second-order update comes from carried velocity versus fresh forcing.
- `vel_r2`, `vel_mse`: how well the internal velocity aligns with the observed finite-difference velocity proxy when that proxy is available.

## Synthetic benchmarks

Synthetic tasks run through the same `interactiondynamics.train` entrypoint via `--dataset synthetic`.
The dataset carries a task-specific metric configuration, so the CLI can decide which checkpoint is "best" and how the final summary should be sorted.

### Available tasks

Synthetic tasks now cover the full node/edge supervision matrix plus an explicit ranking family:

- `edge_regression`
- `node_regression`
- `edge_classification`
- `node_classification`
- `edge_ranking`

#### Edge regression

| Task | What it predicts | Primary metric | Summary metrics | Intended shortlist |
| --- | --- | --- | --- | --- |
| `deepsets_sum` | Next-step per-node sum of incident event values. | `val.edge_r2` | `val.edge_r2`, `val.edge_nrmse`, `test.edge_r2`, `test.edge_nrmse` | `sum/tgn_gru`, `deepsets/tgn_gru`, `settransformer/tgn_gru`, `ift/ift_update` |
| `settransformer_max` | Value attached to the highest-key incident event. | `val.edge_r2` | `val.edge_r2`, `val.edge_corr`, `test.edge_r2`, `test.edge_corr` | `deepsets/tgn_gru`, `settransformer/tgn_gru`, `hopfield/tgn_gru`, `ift/ift_update` |
| `associative_retrieval` | Value whose key best matches a query event in the same node-local set. | `val.edge_r2` | `val.edge_r2`, `val.edge_corr`, `test.edge_r2`, `test.edge_corr` | `hopfield/tgn_gru`, `settransformer/tgn_gru`, `deepsets/tgn_gru`, `ift/ift_update` |
| `temporal_memory` | Damped latent trajectory driven by per-node self events. | `rollout_val.rollout_edge_r2` | `val.edge_r2`, `val.persistent_edge_r2`, `rollout_val.rollout_edge_r2`, `rollout_test.rollout_edge_r2` | `sum/tgn_gru`, `sum/lnn`, `sum/hnn`, `ift/ift_update` |
| `conservative_oscillator` | Lightly driven second-order oscillator with long rollout memory. | `rollout_val.rollout_edge_r2` | `val.edge_r2`, `val.persistent_edge_r2`, `rollout_val.rollout_edge_r2`, `rollout_test.rollout_edge_r2` | `sum/hnn`, `sum/lnn`, `sum/tgn_gru`, `ift/ift_update` |
| `diffusion` | Ring-graph diffusion with per-node drives carried through edge events. | `rollout_val.rollout_edge_r2` | `val.edge_r2`, `rollout_val.rollout_edge_r2`, `rollout_val.rollout_persistent_edge_r2`, `rollout_test.rollout_edge_r2`, `rollout_test.rollout_persistent_edge_r2` | `ift/ift_update`, `sum/lnn`, `sum/tgn_gru` |
| `wave` | Ring-coupled second-order wave dynamics with per-node drives carried through edge events. | `rollout_val.rollout_edge_r2` | `val.edge_r2`, `rollout_val.rollout_edge_r2`, `rollout_val.rollout_persistent_edge_r2`, `rollout_test.rollout_edge_r2`, `rollout_test.rollout_persistent_edge_r2` | `ift/ift_update`, `sum/hnn`, `sum/lnn`, `sum/tgn_gru` |

#### Node regression

| Task | What it predicts | Primary metric | Summary metrics | Intended shortlist |
| --- | --- | --- | --- | --- |
| `node_sum_regression` | Next-step per-node sum of incident event values through the node scorer. | `val.node_r2` | `val.node_r2`, `val.node_nrmse`, `test.node_r2`, `test.node_nrmse` | `sum/tgn_gru`, `deepsets/tgn_gru`, `settransformer/tgn_gru`, `ift/ift_update` |
| `node_keyed_value` | Value attached to the highest-key incident event through the node scorer. | `val.node_r2` | `val.node_r2`, `val.node_corr`, `test.node_r2`, `test.node_corr` | `deepsets/tgn_gru`, `settransformer/tgn_gru`, `hopfield/tgn_gru`, `ift/ift_update` |
| `node_temporal_regression` | Next-step latent node state under driven dynamics. | `rollout_val.rollout_node_r2` | `val.node_r2`, `val.persistent_node_r2`, `rollout_val.rollout_node_r2`, `rollout_test.rollout_node_r2` | `sum/tgn_gru`, `sum/lnn`, `sum/hnn`, `ift/ift_update` |

#### Edge classification

| Task | What it predicts | Primary metric | Summary metrics | Intended shortlist |
| --- | --- | --- | --- | --- |
| `edge_threshold_classification` | Whether the next-step incident count crosses a node-level threshold using the edge head. | `val.edge_auroc` | `val.edge_auroc`, `val.edge_f1`, `test.edge_auroc`, `test.edge_f1` | `sum/tgn_gru`, `deepsets/tgn_gru`, `settransformer/tgn_gru`, `ift/ift_update` |
| `edge_keyed_trigger_classification` | Whether the highest-key incident event carries a positive trigger using the edge head. | `val.edge_auroc` | `val.edge_auroc`, `val.edge_auprc`, `test.edge_auroc`, `test.edge_f1` | `deepsets/tgn_gru`, `settransformer/tgn_gru`, `hopfield/tgn_gru`, `ift/ift_update` |
| `edge_temporal_state` | Whether the next-step latent state is positive under driven dynamics. | `val.edge_auroc` | `val.edge_auroc`, `val.edge_f1`, `test.edge_auroc`, `test.edge_f1` | `sum/tgn_gru`, `sum/lnn`, `sum/hnn`, `ift/ift_update` |

#### Node classification

| Task | What it predicts | Primary metric | Summary metrics | Intended shortlist |
| --- | --- | --- | --- | --- |
| `node_count_threshold` | Whether the next-step incident count crosses a node-level threshold. | `val.node_auroc` | `val.node_auroc`, `val.node_f1`, `test.node_auroc`, `test.node_f1` | `sum/tgn_gru`, `deepsets/tgn_gru`, `settransformer/tgn_gru`, `ift/ift_update` |
| `node_keyed_trigger` | Whether the highest-key incident event carries a positive trigger. | `val.node_auroc` | `val.node_auroc`, `val.node_auprc`, `test.node_auroc`, `test.node_f1` | `deepsets/tgn_gru`, `settransformer/tgn_gru`, `hopfield/tgn_gru`, `ift/ift_update` |
| `node_temporal_state` | Whether the next-step latent node state is positive under driven dynamics. | `val.node_auroc` | `val.node_auroc`, `val.node_f1`, `test.node_auroc`, `test.node_f1` | `sum/tgn_gru`, `sum/lnn`, `sum/hnn`, `ift/ift_update` |

#### Edge ranking

| Task | What it predicts | Primary metric | Summary metrics | Intended shortlist |
| --- | --- | --- | --- | --- |
| `next_dst_ranking` | Next-step destination shift induced by the signed sum of per-node event values. | `val.mrr` | `val.mrr`, `val.hits@1`, `test.mrr`, `test.hits@10` | `sum/tgn_gru`, `deepsets/tgn_gru`, `settransformer/tgn_gru`, `ift/ift_update` |
| `edge_retrieval` | Correct next destination retrieved from a keyed event set. | `val.mrr` | `val.mrr`, `val.hits@1`, `test.mrr`, `test.hits@10` | `deepsets/tgn_gru`, `settransformer/tgn_gru`, `hopfield/tgn_gru`, `ift/ift_update` |
| `next_dst_temporal_ranking` | Next-step destination bucket induced by a driven latent node state. | `val.mrr` | `val.mrr`, `val.hits@1`, `test.mrr`, `test.hits@10` | `sum/tgn_gru`, `sum/lnn`, `sum/hnn`, `ift/ift_update` |

The synthetic `edge_*` regression and classification tasks use the edge scorer on one candidate self-edge per node. They are edge-head benchmarks rather than dense all-pairs link-prediction tasks.
Legacy aliases remain supported for the older names: `edge_count_threshold`, `edge_keyed_trigger`, `edge_ranking_sum_shift`, `edge_ranking_keyed_shift`, and `edge_ranking_temporal`.

### How synthetic tasks are evaluated

- The training loop still reports ordinary loss, but the "best" checkpoint for a task is selected by the task-declared metric path such as `val.edge_r2`, `val.node_auroc`, or `rollout_val.rollout_node_r2`.
- The final sweep summary is sorted by that declared primary metric, not always by `val_loss`.
- Edge regression tasks expose edge targets, so you will see edge metrics such as:
  - `edge_mse`: raw mean squared error.
  - `edge_nrmse`: RMSE normalized by target standard deviation.
  - `edge_r2`: coefficient of determination.
  - `edge_corr`: correlation between predictions and targets.
  - `persistent_edge_r2`: the previous-target persistence baseline measured on the same slice.
- Node regression tasks expose node targets and automatically enable the node scorer, so you will see node metrics such as:
  - `node_mse`
  - `node_nrmse`
  - `node_r2`
  - `node_corr`
  - `persistent_node_r2`
- Edge classification tasks expose binary edge-head labels and use BCE loss while reporting metrics such as:
  - `edge_acc`
  - `edge_f1`
  - `edge_auroc`
  - `edge_auprc`
  - `edge_precision`
  - `edge_recall`
- Node classification tasks expose node-level binary labels and automatically enable the node scorer, so you will see metrics such as:
  - `node_acc`
  - `node_f1`
  - `node_auroc`
  - `node_auprc`
  - `node_precision`
  - `node_recall`
- Edge ranking tasks expose no explicit targets. Instead, the loss and metrics are computed by scoring the observed next destination against sampled negatives, so you will see:
  - `mrr`
  - `hits@1`
  - `hits@3`
  - `hits@10`
  - `pairwise_auc_tie_half`
- For temporal regression tasks, eval also reports rollout metrics:
  - `rollout_edge_mse`
  - `rollout_edge_r2`
  - `rollout_edge_nrmse`
  - `rollout_edge_delta_r2`
  - `rollout_edge_delta_mae`
  - `rollout_node_mse`
  - `rollout_node_r2`
  - `rollout_node_nrmse`
  - `rollout_persistent_edge_r2`
  - `rollout_persistent_node_r2`
- The eval summary also reports a `persistent_*` baseline:
  - for regression, it reuses the previous target value
  - for classification and ranking, it freezes the post-warmup model state and scores from that stale state

### Resizing and controls

- `--synthetic-num-nodes` overrides the node count.
- `--synthetic-events-per-bin` overrides event count per bin for set-based tasks.
- `--num-bins` overrides the number of simulated time bins.
- `--rollout-horizon` changes the rollout evaluation horizon for temporal regression tasks.
- `--edge-target-mode`, `--node-target-mode`, and `--edge-target-scale` still apply to synthetic datasets when that supervision type is present.
