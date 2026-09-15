# Native DyGLib experiment with an FNN backbone

This is a separate experiment, **not** another adapter in our binned trainer.
It runs DyGLib's actual `train_link_prediction.py` and evaluator from commit
`3aacc36b94b8d2d8293d70a74fdf6d39089b4163`. The checkout and its MIT license
live under `derived/dyglib`; the current training path is unchanged.

## Setup (from the project root)

```bash
venv/bin/python -m experiments.dyglib.setup
venv/bin/python -m experiments.dyglib.prepare --dataset college_msg
```

The existing project environment supplies the dependencies (PyTorch, NumPy,
pandas, scikit-learn, tqdm and tabulate). Raw data must already be downloaded.
Setup refuses to overwrite a checkout; prepare refuses to overwrite exported
data. Use `--target` for another checkout and `DYGLIB_DIR` when launching it.

## First run

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 bash scripts/7_dyglib.sh \
  --dataset_name college_msg --model_name FNN \
  --batch_size 200 --num_epochs 10 --num_runs 1 \
  --learning_rate 0.001 --weight_decay 0 --gpu 0
```

Use `--gpu -1` for CPU. `--num_runs 1` uses seed zero; larger values run seeds
0 through N-1. Start with batch size 200; compare throughput and validation
before increasing it to 600 or 1000. Batch size changes how much within-batch
history is deferred, so it is not solely a hardware setting.

Change `--model_name` to `GraphMixer`, `TGN`, `DyGFormer`, `JODIE`, `CAWN`,
`TGAT`, `TCL` or `DyRep` to use the native upstream backbone and loop.
EdgeBank is evaluation-only upstream, not part of this training launcher.
Export `email_eu_core` or `sociopatterns` using the same prepare command to
run those datasets. All models must use the same exported data and protocol.

Upstream writes progress plus artifacts under the checkout's `logs/`,
`saved_models/` and `saved_results/`, **not** our old per-epoch JSONL directory.
Do not overlap identical dataset/model/seed jobs: upstream artifact names collide.

## Evaluate saved checkpoints with different negatives

Update an existing checkout first:

```bash
venv/bin/python -m experiments.dyglib.setup --update
```

The launcher accepts `eval` before the native options. This loads existing
validation-selected checkpoints; it does not retrain or select using test scores.
For five existing eight-channel checkpoints:

```bash
for strategy in random historical inductive; do
  bash scripts/7_dyglib.sh eval \
    --dataset_name college_msg --model_name FNN \
    --fnn_state_dim 8 --batch_size 200 --num_runs 5 --gpu 0 \
    --negative_sample_strategy "$strategy"
done
```

Use the training architecture and split settings, and keep evaluation batch size
the same across samplers/models. `--num_runs 5` requires checkpoints for seeds
0--4. Native training remains random-negative training; these alternatives are
standalone evaluation protocols, not a change to the training sampler.

Historical negatives prefer previously seen pairs inactive in the current
evaluation window. Inductive negatives further exclude pairs observed up to
the cutoff (end of training for validation, end of validation for testing).
Both fill shortages with collision-checked random pairs. These are different
from the separate new-node evaluation subset. Report each protocol separately.

Results go to `derived/dyglib/saved_results/FNN/college_msg/`, with names such as
`inductive_negative_sampling_FNN_seed0_dim8.json`. Samplers and widths have
separate filenames. Repeating the same evaluation overwrites its result file.
The upstream historical/inductive samplers can be slower and memory-heavy:
their random fallback uses a Cartesian pair pool. This launcher preserves
upstream behavior rather than replacing the sampling algorithm.

## What is preserved and what changes

### Multichannel fields

To enable the new option in an existing checkout (preserving data and results):

```bash
venv/bin/python -m experiments.dyglib.setup --update
```

Add `--fnn_state_dim 16` to a run to maintain 16 field and 16 velocity values
per node. Each channel learns its own positive damping and restoring frequency;
a learned shared drive vector maps a unit interaction into these channels.
Initial damping/frequency span approximately 0.5--2 times the scalar defaults
to break channel symmetry. The drive starts with unit total norm. Topology
gates, the global input scale, and the integration step remain shared.
The readout projects all H/V channels to the same upstream embedding width.

For example:

```bash
bash scripts/7_dyglib.sh --dataset_name college_msg --model_name FNN \
  --fnn_state_dim 16 --batch_size 200 --num_epochs 10 --num_runs 1 \
  --learning_rate 0.005 --weight_decay 0.003 --gpu 0
```

The default `--fnn_state_dim 1` preserves scalar behavior and checkpoint shapes.
Multichannel artifacts have a `_dim16` (or corresponding width) suffix, so they
do not overwrite scalar runs. Different hyperparameters at the same width and
seed still share artifact names. Evaluation must use the training width.
Compare widths using validation performance, then report held-out results;
more channels do not guarantee better generalization. State and transition
memory grow with the number of channels, but transitions remain vectorized.

### Evaluation and dynamics

- Native chronological event batches, negative sampling, binary link loss,
  AP/AUROC metrics, early stopping, and memory checkpoint handling are used.
  This is not our ten-negative MRR objective or auxiliary unit-force loss.
- Native splitting includes the withheld-new-node protocol, not simply our
  old chronological split: roughly 10% of nodes appearing after the training
  boundary are withheld from training. The native loader handles this equally
  for FNN and baselines. Original timestamps are retained (shifted to start at
  zero), without hourly binning or user/item ID duplication.
- FNN reuses `FieldNeuralNetwork` with scalar H/V by default, sparse train-derived gates,
  unit interaction drives and destination updates. DyGLib's training sampler
  supplies candidate support in both directions. No test-only gates are created.
- A learned projection maps H/V to the native node embedding width, followed
  by upstream MergeLayer link scoring. Unlike our current FNN scorer, there is
  no additional direct topology-logit term in the link score.
- Observed positives are buffered. Only observations strictly before the
  earliest query time in the next batch are consumed; positive and negative
  queries see the same state. Within-batch observations are deferred, not leaked.
- Field transitions compose over observed timestamps, with fixed
  effective step 0.1, initial gamma 0.15, omega 0.8 and input scale 1. Raw elapsed
  seconds do **not** determine the integration step in this initial port; gaps
  without events do not introduce transitions. This is an event-clock model.
- Gamma, omega, input scale, gates and readout train jointly under the upstream
  optimizer. Our alternating optimizer schedule is **not** ported yet.
- FNN does not rebuild a temporal-neighbor sampler every batch. Its linear
  transitions are composed with batched matrix powers and scattered event
  contributions, rather than a Python loop over timestamps. This is equivalent
  to successive field steps up to floating-point error, including gradients.
  Higher GPU utilization or accuracy is not guaranteed: benchmark events/second
  on the target machine.

Do not pool these results with old binned results, or label results on these
custom exported datasets as reproduction of published DyGLib benchmark scores.

## Optional event-clock Laplacian

`--fnn_spectral_rank 16` enables a low-rank approximation of the symmetric
normalized Laplacian of the binary, symmetrized training candidate graph.
The basis is fixed; event gates remain learned, as does a shared positive
coupling strength initialized at 0.1. This is not learned propagation geometry.
Rank zero (default) retains the original uncoupled FNN and its checkpoint names.

```bash
venv/bin/python -m experiments.dyglib.setup --update
bash scripts/7_dyglib.sh --dataset_name college_msg --model_name FNN \
  --fnn_state_dim 8 --fnn_spectral_rank 16 --batch_size 200 \
  --num_epochs 20 --num_runs 1 --learning_rate 0.003 --weight_decay 0.001 --gpu 0
```

Both local and spatial modes use the SAME semi-implicit event update with
dt=0.1 per distinct observed timestamp and the SAME event-drive convention.
Mode k has stiffness `omega_c^2 + kappa * lambda_k`. Batched matrix powers
compose these updates without sequential graph-wide propagation at each time.
Actual timestamp gaps do not affect the transition. The experimental real-clock
options are removed by `setup --update`; no clock flags are needed.

Local H/V are preserved, including for nodes absent from the training graph.
The spectral correction replaces their retained modes, not duplicates their
input. The effective Laplacian is `U diag(lambda) U.T`: omitted modes retain
local dynamics with zero coupling. Truncation can introduce nonlocal effects
and is not the full sparse Laplacian. Zero eigenmodes provide no propagation;
a very small rank on disconnected graphs may consist only of zero modes.

Basis construction uses a CPU sparse eigensolver once per construction (dense
only for at most 256 active nodes). It is saved with modal memory in checkpoints.
Training work/memory grows with rank and channels, but does not require a dense
N-by-N operator or a Python loop over timestamps. Degree normalization does not
guarantee stability for arbitrary learned coefficients; monitor training.

Artifacts include `_spectral16`, and evaluation requires the same rank. They
are separate from both ordinary and previous real-time spectral results. Old
real-clock checkpoints are not compatible with this event-clock implementation.
Compare rank 0 versus 16 with identical settings and validation selection.

## Optional one-hop sparse input propagation

Update the local bridge once, then add `--fnn_sparse_propagation` to your
usual training command (and to checkpoint evaluation):

```bash
venv/bin/python -m experiments.dyglib.setup --update
bash scripts/7_dyglib.sh --dataset_name college_msg --model_name FNN \
  --fnn_state_dim 8 --fnn_sparse_propagation --batch_size 200 \
  --learning_rate 0.003 --weight_decay 0.001 --num_epochs 30 --num_runs 5
```

Each gate-weighted event impulse retains `1-alpha` at its destination and
spreads `alpha` to that destination's training-graph neighbors, normalized by
their learned outgoing gates. The learned sigmoid fraction starts at 0.1.
Isolated destinations retain the full impulse; self edges are excluded from
spreading. Total input is conserved. This is one hop only: propagated input
does not trigger additional propagation during the same step.

This option diffuses **event input**, not existing latent fields; it is not a
full wave Laplacian solver. It retains event-clock oscillator updates and
causal pending-event buffering. Computation scales with the recipients'
neighbor counts, with no eigenbasis or per-timestamp graph sweep. High-degree
recipients can still create large batches. It cannot be combined with spectral
propagation. Checkpoints/results use a separate `_sparseprop` suffix.

## Upstream bridge

`setup.py` adds FNN to the memory-model dispatch/checkpoint branches and adds
the three dataset names to the CLI. Native models are not replaced. It also
sorts the sampled node set for Python 3.11 compatibility, and permits loading
locally generated pending-memory checkpoints under newer PyTorch versions.
Only load checkpoints you trust. The checkpoint format is not a safe format
for arbitrary downloaded files.

The upstream diff remains visible with `git -C derived/dyglib diff`.

## Verification

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 venv/bin/python -m pytest \
  interactiondynamics/tests/test_dyglib_native.py -q
```

Checks causal buffering, scalar/multichannel state and gradient equivalence,
gate gradients and memory restoration. If the local
checkout exists, also runs complete one-epoch native FNN and GraphMixer trials
on temporary generated events, including checkpoint reload and final testing.
These checks do not establish real-data accuracy or GPU throughput.
