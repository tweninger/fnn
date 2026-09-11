# Real social and traffic datasets

Install download/HDF dependencies with `venv/bin/python -m pip install '.[datasets]'`.
Download all inputs with:

```bash
venv/bin/python -m interactiondynamics.data.download_benchmarks
```

Optional positional names select a subset; `--root` changes the parent data
directory. Each dataset lives in `data/<name>/` with a source URL and SHA256
in `download.json`. Existing files are preserved. Loaders never download data.

| Name | Source | Native representation |
|---|---|---|
| college_msg | https://snap.stanford.edu/data/CollegeMsg.html | Directed user messages, seconds |
| email_eu_core | https://snap.stanford.edu/data/email-Eu-core-temporal.html | Directed email, seconds |
| sociopatterns | https://sociopatterns.org/datasets/high-school-contact-and-friendship-networks/ | 2013 high-school proximity, 20 seconds |
| metr_la | https://github.com/liyaguang/DCRNN | Five-minute sensor speeds |
| pems_bay | https://github.com/liyaguang/DCRNN | Five-minute sensor speeds |

Use the citations and terms on the source pages when publishing. SocioPatterns
specifies CC BY-NC-SA and the Mastrandrea et al. (2015) citation. Class/gender
metadata and friendship surveys are not used as model inputs.

## Social event streams

```python
from interactiondynamics.training.presets import load_dataset

ds = load_dataset('social', {'name': 'college_msg', 'root': 'data'})
print(ds.spec())
for events in ds.bins('train'):
    pass
```

Names: `college_msg`, `email_eu_core`, `sociopatterns`. All use a shared node
namespace, preserving original IDs in `ds.node_ids`. Inputs have unit event
features, with no external-force markers. Emails/messages retain direction and
repeated events. An undirected proximity contact becomes two directed unit
events. This is observational activation, not an equal-and-opposite force pair.

Defaults: hourly message/email bins, 20-second contact bins, chronological
approximately 70/15/15 event splits snapped to bin boundaries. Override
`bin_size` and `split_fracs` explicitly. `split_by='time'` instead splits the
time span, but can leave Email-Eu-core's validation interval without events
because of its long recording gap.
Empty bins are retained to preserve elapsed time (including overnight gaps);
these defaults define our protocol, not an official benchmark protocol.

The loader implements `EventStreamDataset` and can feed the existing training
API and the regular `train --dataset college_msg`, `email_eu_core`, or
`sociopatterns` CLI options. These presets include FNN followed by the usual
six baselines; use `--max-runs 1` for FNN only. `--social-root` selects the
parent data directory; `--social-bin-size` overrides bin width in seconds.

### Quick FNN run

From the repository root:

```bash
mkdir -p derived/results/social_fnn_cli
venv/bin/python -u -m interactiondynamics.train quick \
  --dataset college_msg --max-runs 1 --seed 0 --epochs 9 \
  --fnn-alternating-recovery --fnn-alternating-topology-epochs 3 \
  --fnn-alternating-physical-epochs 1 --fnn-alternating-cycles 1 \
  --rollout-horizon 1 \
  --save-jsonl derived/results/social_fnn_cli/college_msg_seed0.jsonl
```

Replace `college_msg` with either of the other social names to run those.
This runs seed 0 on the automatically selected device, retaining the default
bins and splits. The default fixes dt to one observed bin (one hour for
messages/email, 20 seconds for proximity), with empty bins retained. The
omega-dependent dt parameterization is disabled, so omega updates cannot
change the clock. Coefficients are expressed in these dataset-specific time
units, not seconds. This does not add adaptive numerical substeps; stability
must still be checked when interpreting the smoke run.
It uses the sparse, scalar-state FNN and a nine-epoch schedule: three topology
epochs, one epoch each for omega, gamma and input force scale, then three
topology epochs. Evaluation is final-only with horizon one. `--fnn-learn-dt`
adds a dt phase (use `--epochs 10` for that schedule); unlike the old script,
it does not couple dt to omega. Use a fresh `--save-jsonl` path for each run:
the regular runner appends records to existing files. This short diagnostic
is not a converged benchmark. The previous smoke results remain preserved.

## TGB event-time experiment (FNN only)

```bash
venv/bin/python -m pip install '.[tgb]'
venv/bin/python -u -m interactiondynamics.train tgb \
  --dataset tgbl-wiki --epochs 9 --seed 0 \
  --save-jsonl derived/results/tgb/wiki_fnn_seed0.jsonl
```

This separate subcommand uses TGB's official masks, validation/test negative
candidates and evaluator (MRR for tgbl-wiki). The optional dependency is pinned
to py-tgb 2.3.0; package and dataset versions are saved in the JSONL. Data are
downloaded to `data/tgb`. It runs on CPU (`--threads 2` by default).

This is an **event-time FNN variant**, not the existing binned simulation
protocol. It ignores original message attributes and treats each observed
interaction as a unit impulse into the destination velocity, gated by the
existing sparse FNN topology. Between timestamps it uses the exact unforced
damped-oscillator transition. Actual elapsed seconds divided by `--time-unit`
(3600 by default) set elapsed model time; no dt parameter is trained. No
force-regression auxiliary loss or conditional force rollout is used.

All simultaneous events are scored before any is observed. Training uses
sampled destination softmax, excluding simultaneous positives for that source;
evaluation uses only the official negative lists and per-event MRR. Candidate
parameters come from train-observed and train-sampled pairs, never held-out
edges. The destination vocabulary is transductive. Physical updates truncate
gradients at timestamp boundaries; no full-history backpropagation is claimed.

The default schedule is 3 topology epochs, one each for omega, gamma and input
scale, then 3 topology epochs. More generally use epochs = T + R*(T+3P), with
`--topology-epochs T` and `--physical-epochs P`. Each epoch replays training
history under fixed weights before validation. The highest-validation-MRR
checkpoint is saved beside the JSONL as `.best.pt`. At test time it replays
train and validation observations, then scores and observes test events
chronologically. Existing result/checkpoint paths are refused, not overwritten.

This uses the official evaluation components, but is not a claim of matching
every reference model's batching or feature choices. No reference TGN or
EdgeBank baseline is launched yet. See the [TGB reference implementation](https://github.com/shenyangHuang/TGB/blob/main/examples/linkproppred/tgbl-wiki/tgn.py).

## Traffic windows

```python
traffic = load_dataset('traffic', {'name': 'metr_la', 'root': 'data'})
sample = traffic.windows('train')[0]
# sample: x, x_mask, y, y_mask, target_index
# x and y have shape [time, sensor, 1]. y retains original speed units.
```

Names: `metr_la`, `pems_bay`. Raw `values`, `observed`, `timestamps`, and
`sensor_ids` are accessible. Zero and nonfinite values are missing by default;
set `zero_is_missing=False` to treat zero speeds as observations. Inputs use
one global mean/std fitted only to observed training readings. Missing inputs
are filled with zero after normalization; masks must be used in loss/evaluation.

Defaults: 12 history steps, 12 future steps, chronological 70/10/20 splits.
Windows stay entirely within each split; no target crosses a split boundary.
Windows crossing timestamp gaps (including PEMS-BAY's March clock jump) are
excluded; timestamps are preserved as provided, without assuming a timezone.
This conservative window protocol is not an exact reproduction of DCRNN's
window splitting. No road graph is supplied by this loader. Traffic is a
forecasting `Dataset` API, not an `EventStreamDataset`; it requires a sensor
forecasting training adapter before use with the event-only FNN runner.
