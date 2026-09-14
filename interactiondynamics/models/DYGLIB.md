# DyGLib comparison adapters

The physical-event panel now ends with `edgebank`, `graphmixer`, `tgn`, and
`dygformer`, and `jodie`, after the original seven runs. `--run-offset 7 --max-runs 5`
selects only these additions; `--max-runs 7` retains the old panel.
The real-data panel script accepts twelve runs and uses twelve by default.

Backbones are vendored from DyGLib commit
`3aacc36b94b8d2d8293d70a74fdf6d39089b4163` with its MIT license.
See `dyglib_vendor/README.md` for the mechanical extraction details.
No extra installation or TGB dependency is required.

## Architecture versus protocol

These are upstream backbones evaluated in **our binned event protocol**, not
reproductions of published DyGLib benchmark scores. They use our splits,
sampled ranking objective, negative candidates, and validation selection.
As with the existing evaluator, each split starts with fresh history; synthetic
episodes also reset independently. Raw-timestamp chronological benchmarking
with train-history warm-up is a separate protocol change.

Only `step` observes an event. `score` ignores target features and retrieves
strictly earlier temporal neighbors. IDs are shifted by one internally because
upstream reserves zero for padding. Recent-neighbor sampling is deterministic.
TGN keeps its buffered memory in the returned state, so cloned rollout states
and independent validation states cannot contaminate one another.

GraphMixer, TGN, DyGFormer, and JODIE retain their upstream embedding computation and
MergeLayer ranking head. A separate MLP predicts event features for compatibility
with our force objective and conditional force rollouts; this head is our
extension, not part of the original link-prediction model.

JODIE uses DyGLib's RNN memory updater and time-projection embedding, not the
TGN graph-attention embedding. It works on the homogeneous social streams too.
Its time-shift normalization currently uses the upstream constructor defaults
(mean zero, standard deviation one) in observation-bin units.

EdgeBank uses upstream unlimited **directed** pair memory with binary scores.
It has no optimizer, no force decoder, and no physical rollout. Its single pass
reports ranking/event metrics; force metrics remain unavailable, not zero.

Temporal configuration fields on ModelConfig: `temporal_num_neighbors` (20),
`temporal_history_length` (64, DyGFormer), plus existing node/time dimensions
and dropout. These are adapter defaults, not dataset-tuned upstream settings.
Histories are rebuilt as bins are observed; this initial integration favors
state isolation over throughput and should be profiled before large sweeps.

Citation: Yu, Sun, Du, and Lv, *Towards Better Dynamic Graph Learning: New
Architecture and Unified Library*, NeurIPS 2023.
