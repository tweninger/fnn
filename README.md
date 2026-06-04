# IFT

Interaction field and interaction dynamics experiments.

## Development

```bash
venv/bin/python -m pip install -e ".[dev]"
venv/bin/python -m pytest
venv/bin/python -m ruff check
venv/bin/python -m pylint interactiondynamics interactionfields
venv/bin/pyright
```

## Training presets

```bash
venv/bin/python -m interactiondynamics.train smoke
venv/bin/python -m interactiondynamics.train quick
venv/bin/python -m interactiondynamics.train quick --dataset synthetic --synthetic-task deepsets_sum
venv/bin/python -m interactiondynamics.train quick --dataset synthetic --synthetic-task settransformer_max
venv/bin/python -m interactiondynamics.train quick --dataset synthetic --synthetic-task temporal_memory
venv/bin/python -m interactiondynamics.train quick --dataset synthetic --synthetic-task node_count_threshold
venv/bin/python -m interactiondynamics.train quick --dataset synthetic --synthetic-task node_keyed_trigger
venv/bin/python -m interactiondynamics.train quick --dataset synthetic --synthetic-task node_temporal_state
venv/bin/python -m interactiondynamics.train quick --dataset synthetic --synthetic-task node_sum_regression
venv/bin/python -m interactiondynamics.train quick --dataset synthetic --synthetic-task node_temporal_regression
venv/bin/python -m interactiondynamics.train quick --dataset synthetic --synthetic-task edge_threshold_classification
venv/bin/python -m interactiondynamics.train quick --dataset synthetic --synthetic-task next_dst_ranking
venv/bin/python -m interactiondynamics.train quick --dataset synthetic --synthetic-task associative_retrieval
venv/bin/python -m interactiondynamics.train quick --dataset synthetic --synthetic-task conservative_oscillator
venv/bin/python -m interactiondynamics.train quick --dataset synthetic --synthetic-task ift_diffusion
```

- `smoke` runs a tiny toy dataset check over the focused model/update shortlist.
- `quick` runs the focused shortlist on JODIE Wikipedia:
  - `ift` + `ift_update`
  - `hopfield` + `hopfield_update`
- `settransformer` + `lnn`
- `settransformer` + `hnn`
- `settransformer` + `tgn_gru`
- You can override examples like `venv/bin/python -m interactiondynamics.train quick --dataset toy --epochs 1`.
- `--edge-target-mode residual` trains the edge head to predict next-step changes relative to the previous step while continuing to report raw-space edge metrics.
- `--node-target-mode residual` does the same for optional node supervision, predicting state changes instead of raw next-step node values.
- For synthetic edge regression, `--edge-target-scale zscore` rescales the MSE loss by the train-split edge-target standard deviation while still reporting raw-space metrics.
- Ranking metrics now treat ties pessimistically by default, and eval also reports a `persistent_*` baseline that keeps the post-warmup state fixed.

## IFT diagnostics

The `ift` path now supports both first-order diffusion-style updates and second-order oscillator-style diagnostics without leaving the event-bin architecture.

- First-order IFT:
  - aggregator builds a graph/Laplacian from event bins
  - update uses diffusion, damping, and forcing
- Second-order IFT:
  - carries explicit velocity memory
  - supports scalar `h/v/force` readout for AR(2)-style diagnostics
  - supports teacher-forced or autonomous rollout evaluation

Useful commands:

```bash
venv/bin/python -m interactiondynamics.train ift-diagnose
venv/bin/python -m interactiondynamics.train ift-diagnose --ift-diagnostic-tasks ift_diffusion
venv/bin/python -m interactiondynamics.train ift-diagnose --ift-diagnostic-tasks conservative_oscillator --num-bins 40 --rollout-horizon 5
```

The diagnostic suite prints:

- one-step sanity batches for second-order oscillator runs
- rollout-vs-persistent metrics
- internal IFT diagnostics such as `kappa`, `dt`, `alpha`, `force_norm`, `diffusion_term_norm`, `relative_update`
- linear `h/v/force` readout coefficients when that scorer is active

For conservative oscillator diagnostics, the suite also reports:

- `ar1_baseline`
- `ar2_baseline`
- `oracle_baseline`
- `closed_form_delta`

Current status:

- `ift2_ar2_oracle_init` now matches the AR(2) oracle rollout exactly when enough bins are available for second-order rollout seeding.
- `closed_form_delta` also recovers the oracle solution.
- learned second-order linear readouts improve with better initialization and larger learning rates, but still trail the oracle after a short run.
- teacher-forced `h/v/force` learning is available to separate coefficient-learning issues from velocity-propagation issues.

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
| `ift_diffusion` | Ring-graph diffusion with per-node drives carried through edge events. | `rollout_val.rollout_edge_r2` | `val.edge_r2`, `rollout_val.rollout_edge_r2`, `rollout_val.rollout_persistent_edge_r2`, `rollout_test.rollout_edge_r2`, `rollout_test.rollout_persistent_edge_r2` | `ift/ift_update`, `sum/lnn`, `sum/tgn_gru` |

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
