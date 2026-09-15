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

## Small, inspectable upstream patch

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
