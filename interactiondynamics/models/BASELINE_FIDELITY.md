# Model identities and evaluation protocol

The panel adapts architectures to binned predict-before-observe evaluation;
it is not a reproduction of published DyGLib benchmark scores.

| Run family | Accurate description |
| --- | --- |
| FNN | Persistent gates on observed force events, local field/velocity dynamics, and optional state-dependent ranking. Not an autonomous latent graph-Laplacian solver. |
| Sum–GRU | Custom binned sum aggregation plus GRU, not canonical TGN. |
| DeepSets–GRU | DeepSets aggregation plus the same custom GRU. |
| SetTransformer–GRU | Masked set attention and pooling plus the custom GRU. |
| Hopfield | Hopfield-inspired iterative attention aggregation and custom gated associative update; not a guaranteed energy-minimizing Hopfield network. |
| SetTransformer–LNN | Restricted unit-mass learned-potential dynamics, not a general learned Lagrangian. |
| SetTransformer–HNN | Message-conditioned learned Hamiltonian differentiated using Hamilton's equations; explicit Euler integration, not a symplectic or energy-conserving solver. |
| EdgeBank | DyGLib unlimited-memory EdgeBank. |
| GraphMixer | Pinned DyGLib backbone; scalar inputs zero-padded to at least 172 mixer channels. |
| TGN | Pinned DyGLib memory/attention backbone under our bin protocol. |
| DyGFormer | Pinned DyGLib backbone with our configured history length and patch settings. |
| JODIE | DyGLib JODIE variant, with train-only time-gap normalization. |

Existing CLI run identifiers are retained for compatibility. Use the descriptions
above when labeling comparisons; custom GRU controls are distinct from TGN.

Continuous social streams replay training observations with frozen model weights
before validation, and training plus validation observations before test. The
first held-out bin is scored after consuming only preceding history. Rollout
evaluation receives the same past history. Independent synthetic episodes still
reset their states; they do not replay other episodes.

History is observed online after prediction; no held-out gradients or statistics
are used for training. JODIE normalization is stored in the model state dictionary.
Feature width, binning, negative sampling, auxiliary force losses, and custom
readouts remain explicit experimental choices, not upstream benchmark parity.
Conditional force rollout still retains the future pair-query schedule: it does
not generate an autonomous future interaction schedule.

Results produced before these fidelity fixes must be rerun, not pooled with new
results. In particular, continuous-stream metrics no longer use cold-start splits.
