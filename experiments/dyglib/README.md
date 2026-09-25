# Native DyGLib experiment with an FNN backbone

This is a separate experiment, **not** another adapter in our binned trainer.
It runs DyGLib's actual `train_link_prediction.py` and evaluator from commit
`3aacc36b94b8d2d8293d70a74fdf6d39089b4163`. The checkout and its MIT license
live under `derived/dyglib`. The native loop is retained with targeted FNN branches.

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

Results go to `derived/dyglib/saved_results/FNN/college_msg/`; filenames
also encode learning rate, weight decay, batch size, architecture, clock,
ablations, and optional run tag. Standalone evaluation must repeat these
training flags to find its checkpoint. Repeating an evaluation overwrites
that result file.
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
do not overwrite scalar runs. Learning rate, weight decay, batch size, clock,
order, propagation, ablation, and run tag also enter artifact names. Evaluation
must repeat the training configuration.
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
- Observed positives are buffered. Event and batch-min normalized clocks score
  the batch against a common pre-batch state. Exact clocks score timestamp
  groups chronologically; events at time `t` become visible only to later
  timestamps, never to queries at `t`. Positive and negative queries at one
  timestamp see the same state.
- The default `event` clock uses a semi-implicit step of 0.1 per distinct
  observed timestamp; raw timestamp gaps do not change the step. Initial
  gamma, omega, and input scale are 0.15, 0.8, and 1.0. Physical coefficients,
  channel width, and first- versus second-order dynamics can be configured.
- Gamma, omega, input scale, gates and readout train jointly under the upstream
  optimizer. Our alternating optimizer schedule is **not** ported yet. For
  nonzero `--weight_decay`, positive physical coefficients are regularized in
  transformed space: the optimizer applies no decay to their raw leaves and
  the loss adds `weight_decay / 2 * sum(physical_value ** 2)`. Thus decay
  targets physical zero rather than `softplus(0) = 0.6931`. These corrected
  nonzero-decay runs use a `_physwd` suffix. `--fnn_legacy_weight_decay` instead
  applies optimizer decay to raw FNN parameters, matching the legacy ASHA and
  overnight protocol; those runs use `_legacywd`. Never pool the two protocols.
- `--fnn_clock normalized` uses elapsed timestamps without tying the oscillator
  to a dataset's raw units. One median positive training-time gap becomes one
  model-time unit, and `--fnn_time_cap 10` clips unusually long quiet gaps.
  The default `event` clock and its checkpoint names remain unchanged;
  normalized-clock runs use an `_ntime_cap10` suffix.
- `--fnn_clock normalized_exact` applies exact matrix-exponential transitions
  over median-gap-normalized elapsed time, capped by `--fnn_time_cap` (default
  10), while scoring timestamp groups causally. It uses `_ntime_exact_cap10`.
- `--fnn_clock normalized_substep` uses the same normalized elapsed duration as
  `normalized_exact`, but composes the original semi-implicit update in `dt`
  substeps instead of using a matrix exponential. This isolates integrator
  choice while holding the modeled duration fixed.
- `--fnn_clock normalized_substep_unit` also composes the original
  semi-implicit update, but treats one normalized timestamp gap as one full
  model-time unit. At the default `dt=0.1`, it therefore takes ten times as
  many substeps as `normalized_substep` for the same normalized gap.
  Both substep modes cap normalized gaps with `--fnn_time_cap` and use the same
  causal timestamp-grouped scoring as `normalized_exact`. They currently rely
  on a descriptive `--run_tag` to keep artifacts distinct; evaluation must use
  the identical clock and run tag.
- `--fnn_clock event_exact` uses the same exact solver and causal grouping, but
  advances by the configured `dt` (default 0.1) per event timestamp, regardless
  of timestamp gaps. It uses `_etime_exact`.
- `--fnn_clock event_exact_unit` uses one full model-time unit per event
  timestamp, independent of the numerical `dt`; it uses `_etime_exact_unit`.
  The event-exact variants do not use `--fnn_time_cap`. Their durations differ
  by a factor of ten at the default `dt`, so report the variant explicitly.
- `--fnn_ablation fixed_topology` fixes every directed pair observed in the
  training graph to gate 1 and treats unknown validation/test pairs as gate 0;
  gamma, omega, and input scale remain learned. Results use `_fixtop`.
- `--fnn_ablation fixed_gates --fnn_fixed_gate_value 0.5` fixes every
  sigmoid event gate to 0.5 for both training-observed and unseen pairs.
  Gate logits are frozen while gamma, omega, input scale and readout can learn.
  This is the real-data constant-gate counterpart to the synthetic topology
  ablation; it does not assume the train-observed graph is ground truth.
  Values strictly between 0 and 1 are supported for validation tuning. The
  default 0.5 is sigmoid(0), and artifacts use `_fixgate0.5`.
- `--fnn_ablation fixed_gates_physical --fnn_fixed_gate_value 0.5` combines
  the constant 0.5 sigmoid gates with frozen physical coefficients. Results use
  both `_fixphys` and `_fixgate0.5` so they remain separate from either
  individual ablation.
- `--fnn_ablation fixed_physical` fixes gamma=0.15, omega=0.8,
  input scale=1, and dt=0.1 while leaving topology and readout learning active.
  Results use `_fixphys`.
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

Under the default `event` clock, local and spatial modes use the same
semi-implicit step with dt=0.1 per observed timestamp and the same drive.
Mode k has stiffness `omega_c^2 + kappa * lambda_k`. Batched matrix powers
compose these updates without sequential graph-wide propagation at each time.
Actual timestamp gaps do not affect the default event transition. The clock
variants above remain available after `setup --update`; use the same clock
option for checkpoint evaluation.

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

Artifacts include `_spectral16`, and evaluation requires the same rank and
clock. Compare rank 0 versus 16 with identical settings and validation
selection; older spectral checkpoints are not automatically interchangeable.

## Optional K-hop sparse input propagation

Update the local bridge once, then add `--fnn_propagate 2` to your
usual training command (and to checkpoint evaluation):

```bash
venv/bin/python -m experiments.dyglib.setup --update
bash scripts/7_dyglib.sh --dataset_name college_msg --model_name FNN \
  --fnn_state_dim 8 --fnn_propagate 2 --batch_size 200 \
  --learning_rate 0.003 --weight_decay 0.001 --num_epochs 30 --num_runs 5
```

Each gate-weighted event impulse retains `1-alpha` at its destination and
spreads `alpha` to that destination's training-graph neighbors, normalized by
their learned outgoing gates. The learned sigmoid fraction starts at 0.1.
Isolated destinations retain the full impulse; self edges are excluded from
spreading. At each hop, the arriving input is split again; at the final hop,
all remaining input is retained. Total input is conserved. Contributions are
aggregated by timestamp and node between hops rather than enumerating paths.
The default `--fnn_propagate 0` disables propagation; `1` reproduces the former
one-hop behavior. `-fnn_propagate` is also accepted. The old boolean flag is removed.

This option diffuses **event input**, not existing latent fields; it is not a
full wave Laplacian solver. It retains event-clock oscillator updates and
causal pending-event buffering. Computation scales with the recipients'
neighbor counts, with no eigenbasis or per-timestamp graph sweep. High-degree
recipients can still create large batches. It cannot be combined with spectral
propagation. Checkpoints/results include the hop count (e.g. `_propagate2`).
Existing `_sparseprop` artifacts are left untouched; they are not automatically
loaded under the new naming scheme.

## Full official dataset collection

After installing the pinned checkout, download all official benchmark data:

```bash
venv/bin/python -m experiments.dyglib.download_data
```

The downloader verifies the checksums from [Zenodo record 7213796](https://zenodo.org/records/7213796),
caches the 13 individual archives in `data/dyglib/`, and installs their published
processed CSV/node/edge arrays in `derived/dyglib/processed_data/`. Myket is
already bundled in the pinned upstream checkout. Existing files must match;
different local data will not be overwritten. Re-running skips downloads of
verified archives. The combined archive is not downloaded redundantly.
Reddit's archive requires the system `unzip` utility with Deflate64 support.
The command also checks event indices, timestamp ordering, and feature-array
dimensions, and writes `data/dyglib/validation.json`.

Dataset CLI names are case-sensitive:

- Bipartite: `wikipedia`, `reddit`, `mooc`, `lastfm`, `myket`.
- Single node type: `enron`, `SocialEvo`, `uci`, `Flights`, `CanParl`,
  `USLegis`, `UNtrade`, `UNvote`, `Contacts`.

These names remain distinct from our `college_msg`, `email_eu_core`, and
`sociopatterns` conversions; do not assume those versions have identical
preprocessing to the official benchmarks. For example:

```bash
bash scripts/7_dyglib.sh --dataset_name SocialEvo --model_name FNN \
  --fnn_state_dim 8 --fnn_propagate 0 --num_epochs 20 --num_runs 5 --gpu -1
```

## CPU ASHA search on one dataset

The asynchronous launcher follows the promotion scheme described in
[CMU's ASHA overview](https://blog.ml.cmu.edu/2018/12/12/massively-parallel-hyperparameter-optimization/).
By default it samples 81 distinct configurations, runs four CPU workers, and
uses reduction factor three with epoch rungs 3, 9, and 30. Whenever a worker
becomes free, the scheduler promotes the best eligible configuration from the
highest rung; otherwise it evaluates a new configuration at three epochs.

Search uses seed zero. The three configurations reaching 30 epochs are then
confirmed with seeds zero through four, and the final winner is selected by
mean validation average precision:

```bash
venv/bin/python -m experiments.dyglib.asha --dataset enron \
  --sampling-seed 20260918 \
  --output-dir derived/dyglib_asha/enron_min3_seed20260918
```

The default four workers each receive four CPU threads, matching a 16-core
machine. Set `--workers` and `--cpu-threads` explicitly on other hardware.
Progress is printed after every completed trial and once per minute while all
workers remain busy. State, rankings, and command logs are written below
`derived/dyglib_asha/<dataset>/`; rerunning the same command resumes completed
rungs. Use `--dry-run` to inspect the sampled configurations and compute budget.

Each promoted rung currently retrains its configuration from initialization
rather than restoring optimizer state. Promotion is asynchronous and reduces
the maximum allocation from 86,400 grid seed-epochs to 1,206 seed-epochs, but the
retraining adds some overhead compared with checkpoint-continuing ASHA.

## Upstream bridge

Training checkpoint, log, and result names include learning rate, weight decay,
and batch size for all trained models. For example:
`FNN_seed0_lr0.003_wd0.001_bs200_propagate2_dim8`.
Standalone evaluation must receive the **training** values of `--learning_rate`,
`--weight_decay`, and `--batch_size`, as well as the same FNN architecture options,
to select that checkpoint. Existing checkpoints with the older naming scheme
are left untouched and are not automatically selected by the new names.
Run `venv/bin/python -m experiments.dyglib.setup --update` on other machines
after pulling these changes.

`setup.py` adds FNN to the memory-model dispatch/checkpoint branches and adds
the three converted dataset names to the CLI. `runner.patch` replays the
reviewed live-runner delta: exact-time and semi-implicit-substep causal
scoring, first-order flags, legacy decay, validation metadata, and
zero-velocity evaluation. Native
models are not replaced. The installer also
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

## Reproducibility map for a future Git handoff (2026-09-25)

The last recorded pull into local `main` was 2026-09-17 at commit
`cd4eeda3cb8a3238d6f2fd77e8b60efaff1c57f8`. Nothing in this audit was
committed, pushed, or copied over the active `derived/dyglib` checkout.
The pinned upstream DyGLib revision remains the `COMMIT` in `setup.py`.

| Work | Reproduction source | Local evidence/output | Git handoff status |
| --- | --- | --- | --- |
| Core FNN and native runner | `fnn.py`, `setup.py`, `runner.patch`, existing native tests | Pinned checkout under `derived/dyglib` | Minimal core candidate; the runner patch replays the live training/evaluation delta on a clean install. |
| Overnight baseline, mechanism, order, time, and diagnostic arms | `overnight_matrix.py` and `selected_run_manifest.json` | Ignored state, `STATUS.md`, logs, native results | The manifest freezes 231 valid completed definitions, including hand-added tasks; it omits invalidated/excluded runs and mutable queue fields. Re-export with `python3 -m experiments.dyglib.overnight_matrix export-manifest`. Do not use `init --force` to reconstruct the historical queue. |
| Matched propagation | `matched_propagation.py` | `derived/dyglib_matched_propagation_uci_small_20260924/` and native results | Keep as a focused follow-up only if its matched protocol is in the paper; prior test inspection means it is replication, not untouched confirmation. |
| Physical coefficients and identifiability | `physical_interpretability_audit.py`, `train_selective_decay.py`, overnight diagnostic tasks | `derived/results/fnn_physical_interpretability/`, physical CSVs, native results | Candidate diagnostic evidence; retain configuration and checkpoint provenance. |
| Frequency controls and velocity readout | `frequency_probe.py`, `frequency_matched_eval.py`, `run_velocity_readout_sweep.py` | `derived/results/`, `derived/velocity_readout_sweep/` | Candidate diagnostic evidence; evaluation-only sweeps depend on saved checkpoints. |
| Selected numerical summary | `compile_selected_results.py` | `derived/results/compiled_current/` | Eight refreshable CSVs: per-seed/config, grouped means/SD/SEM, physical distributions, and diagnostics. Excludes the large ASHA matrix and invalidated artifact metrics. This directory is currently Git-ignored. |
| ASHA search | `asha.py` | `derived/dyglib_asha/` | Do not promote the huge sweep now; decide separately whether the launcher or only winning configurations belong in the eventual repo. |

Forty manifest entries reuse historical result artifacts whose filenames do not
carry the queue tag; they are marked `reused_artifact`. The recorded result path
is the provenance for those rows, while `command_for()` would use a fresh tag
if rerun.

The compiled CSVs include excluded/pending queue rows for accounting; a blank
metric is **not** a zero, and an SEM is undefined for a single completed seed.
The source scripts are not a substitute for the ignored data, checkpoints, or
queue state. A clean future reproduction needs the official processed data,
exact runner/model code, full run flags and seeds, and the selection protocol.
Do not `git add -A` this workspace: it also contains large untracked data,
plots, and unrelated tests. Before promotion, decide which summary CSVs to
snapshot, whether the 231-run manifest is the right scope for the paper, and
which focused diagnostic scripts/reports are publication-relevant. The manifest
preserves completed run definitions but is not a resumable queue or a copy of
checkpoints. No raw logs, checkpoints, state file, or analysis scripts have
been deleted.
