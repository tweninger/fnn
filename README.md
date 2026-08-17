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
| `diffusion` | Episodic first-order diffusion from one observed impulse over a hidden selected topology. | Main first-order mechanism test. |

The event-only `wave` shortlist is: the proposed
FNN; Sum, Deep Sets, and Set Transformer with a TGN/GRU update; and Hopfield
with its Hopfield update. Every model sees only prior force events and predicts
the next interaction pair and its force vector. For a
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

The field topologies are `ring` (default), `grid`, `torus`, `doorway`, and
`swisscheese`. They can be paired independently with the topology-aware
`diffusion`, `wave`, and `coupled_oscillator` synthetic tasks. All three use
the event-only episodic generator: an observed impulse starts a trajectory and
the subsequent directed force events are the model's only observations.

#### Second-order — inertia, velocity, and waves

| Task family | Canonical task(s) | Domain and mechanism |
| --- | --- | --- |
| Local driven oscillator | `temporal_memory`, `node_temporal_regression`, `node_temporal_state`, `edge_temporal_state`, `next_dst_temporal_ranking` | Independent self-loop systems with AR(2)-style state, velocity carry-over, and forcing; **not** graph propagation. |
| Conservative local oscillator | `conservative_oscillator` | Lightly driven, long-memory second-order oscillator. |
| Graph wave | `wave` | A hidden fixed topology and vector-valued damped wave field emit directed force events. An observed raindrop starts each episode, with optional later observed drops into the same field. |
| Graph-coupled oscillator | `coupled_oscillator` | A hidden fixed topology with coupled oscillator dynamics emits directed force events. |

For new experiments, select the dynamic with `--synthetic-task` and the
domain independently with `--synthetic-topology`. For example:

```bash
venv/bin/python -m interactiondynamics.train quick --dataset synthetic \
  --synthetic-task wave --synthetic-topology torus

venv/bin/python -m interactiondynamics.train quick --dataset synthetic \
  --synthetic-task coupled_oscillator --synthetic-topology doorway
```

The hidden topology controls which directed force interactions are emitted.
`grid` has reflecting outer boundaries, `torus` wraps both axes, `doorway`
adds a wall with a three-node aperture, and `swisscheese` removes circular
patches of nodes.

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
- `--synthetic-topology {ring,grid,torus,doorway,swisscheese}`: choose the domain independently for `diffusion`, `wave`, or `coupled_oscillator`; the default is `ring`.
- `--synthetic-num-nodes N`: override synthetic node count.
- `--synthetic-events-per-bin N`: override observed event count per bin. For physical tasks this subsamples directed internal force measurements; observed raindrops are retained.
- `--synthetic-num-episodes N`: number of independent trajectories for episodic physical dynamics; splits preserve whole trajectories.
- `--synthetic-raindrop-interval K`: for physical dynamics, inject an observed external raindrop every `K` local steps within an episode (in addition to step 0). Drops update all model states but are excluded from prediction loss because their time, node, and amplitude are exogenous.
- `--synthetic-event-threshold TAU`: emit an endogenous physical interaction only when its force-vector magnitude exceeds `TAU`, in fixed nominal-raindrop-force units from `0` to `1`. `0` retains every nonzero physical interaction.
- `--num-bins N`: override the number of simulated synthetic time bins.
- `--seed N`: set the synthetic-data random seed.

Physical rollout JSONL records include both `rollout_by_step` (distance from
the rollout start) and `rollout_by_steps_since_external` (distance from the
most recent observed raindrop). The latter isolates the predicted response at
`+1`, `+2`, and later bins after each intervention. Matching
`rollout_persistent_*` aggregates and per-step curves freeze the model state
at the observed rollout start; they receive the same future pair queries but
no later generated forces or observed raindrops update that state.

For the event-only physical tasks, force supervision is calibrated once from
the training split: each force channel is normalized by its training standard
deviation, while events in the upper force-magnitude range receive extra loss
weight. Reports retain raw `force_mse` for the physical error, and add
`force_nrmse`, `force_weighted_nrmse`, and `active_force_mse` (the upper 25% of
training-force magnitudes). These quantities are loss/evaluation calibration
only—not inputs available to any model.

Common training flags:

- `--max-runs N`: cap the number of runs executed after filtering the preset.
- `--epochs N`: override the preset epoch count.
- `--rollout-horizon K`: set the maximum closed-loop rollout horizon where supported.
- Physical rollout JSONL records include both distance-from-start and distance-from-raindrop curves; aggregate metrics are retained as `rollout_*` fields.
- `--rollout-train-steps K`: optimize an average loss over `K` differentiable autoregressive event-prediction steps before each optimizer update. The default, `1`, is ordinary one-step training.
- `--use-node-scorer`: force-enable the auxiliary node scorer.
- `--node-loss-weight W`: weight for the node prediction loss.
- `--node-scorer-hidden H`: hidden width for the node scorer MLP.

For event-only physical benchmarks, rollouts begin from observed history,
then feed each model's predicted internal force events back into its state.
The future source/destination measurement schedule remains an explicit oracle
condition; random external raindrops are instead supplied as observed
interventions to every model.

Target and loss flags:

- These flags apply only to legacy revealed-target benchmarks: `--node-target-mode`, `--edge-target-mode`, `--edge-target-scale`, and `--prediction-mode`.

Output flags:

- `--save-jsonl PATH`: append per-epoch metrics and summaries to a JSONL file.

Compatibility note:

- The hidden legacy form `venv/bin/python -m interactiondynamics.train --preset {smoke,quick,full}` is still supported, with `full` mapping to `sweep`.
- Classification tasks require `--node-target-mode raw` and `--edge-target-mode raw`.

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
| `diffusion` | Event-only first-order response from one external impulse. | `val.force_mse` | `val.force_mse`, `test.force_mse`, `test.topology_auc`, `test.topology_f1` | `fnn`, TGN/GRU encoder panel |
| `wave` | Event-only second-order wave response from one external impulse. | `val.force_mse` | `val.force_mse`, `test.force_mse`, `test.topology_auc`, `test.topology_f1` | `fnn`, TGN/GRU encoder panel |

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
| `coupled_oscillator` | Event-only coupled-oscillator response from one external impulse. | `val.force_mse` | `val.force_mse`, `test.force_mse`, `test.topology_auc`, `test.topology_f1` | `fnn`, TGN/GRU encoder panel |

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
