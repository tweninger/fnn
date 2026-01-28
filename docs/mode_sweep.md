## Experiment: Rollout Mode Comparison Across Graphs and Dynamics

### Objective
This experiment evaluates how different rollout modes of Interaction Field Theory (IFT) perform across a broad suite of synthetic graph topologies and dynamical processes. The purpose is to isolate the role of *rollout semantics*—rather than graph structure, training procedure, or model capacity—in determining predictive performance.

By holding the learned interaction field fixed and varying only how the model is unrolled at inference time, we assess when interaction structure must be externally supplied versus when it can be internally generated.

---

### Rollout Modes
All models are trained on identical event streams and differ only in how predictions are generated during the holdout period.

1. **Driven Rollout**  
   The model is conditioned on the true sequence of interaction events during the holdout interval.  
   This mode evaluates whether the learned latent field evolves correctly *given* the true interactions.

2. **Free Rollout**  
   The model evolves autonomously during holdout without observing any new events.  
   This mode tests whether the learned dynamics possess stable internal structure independent of external input.

3. **Self Rollout**  
   The model generates interaction events by sampling from its own predicted intensities and feeds those events back into the latent dynamics.  
   This mode evaluates whether the learned field can sustain realistic interaction patterns in a closed loop.

All rollouts are warm-started from the final latent state reached at the end of the training interval.

---

### Synthetic Graph Suite
Experiments are conducted across a diverse set of graph topologies designed to span common regimes studied in network science, statistical physics, and complex systems.

The graph suite includes:

- **Regular lattices and periodic variants**, including planar grids, cylindrical grids, and toroidal lattices
- **Geometric graphs**, such as random geometric graphs constructed using either fixed-radius or k-nearest-neighbor connectivity
- **Small-world networks**, generated via Watts–Strogatz rewiring
- **Hierarchical structures**, including tree lattices
- **Manifold-based graphs**, such as discretizations of spherical surfaces
- **Canonical non-lattice graphs**, including rings, wheels, chorded rings, and directed cycles
- **Block-structured graphs**, including stochastic block models
- **Compositional graphs**, formed by introducing defects or multilayer structure into a base topology

Graph parameters are chosen to maintain comparable scale and density across topologies while preserving their characteristic structural properties.

---

### Dynamical Processes
For each graph, event streams are generated using multiple classes of interaction dynamics. These processes are selected to represent qualitatively distinct mechanisms of collective behavior:

- **Field-based dynamics**, in which interactions propagate through continuous latent fields
- **Epidemic-style dynamics** (SIS), modeling contagion and recovery
- **Threshold dynamics**, where state changes occur once accumulated influence exceeds a local threshold
- **Voter dynamics**, capturing opinion alignment through pairwise interactions
- **Transport processes**, modeling directed or diffusive flow across the network
- **Self-exciting point processes**, where past interactions increase the likelihood of future events

All dynamics are observed exclusively through discrete interaction events, ensuring a consistent event-based learning setting across processes.

---

### Training and Evaluation Protocol
Event streams are divided temporally into training and holdout segments using a fixed split ratio.  
Models are trained exclusively on the training segment.

During evaluation:
- All rollout modes begin from the same latent state obtained at the end of training
- Model parameters are frozen during holdout
- Any calibration of interaction intensities is performed using training data only

This protocol ensures that differences in performance arise solely from the rollout mode and not from additional information leakage or retraining.

---

### Evaluation Metrics

#### Node-Level Metrics
Node activity is represented as an exponentially smoothed time series derived from incident interaction events. Predictions are evaluated against ground truth using:

- Normalized mean squared error (NMSE)
- Mean node-wise Pearson correlation
- Median node-wise Pearson correlation

Metrics are reported at multiple prediction horizons.

#### Edge-Level Metrics
Predicted interaction intensities are evaluated against observed events using:

- Precision–Recall AUC
- ROC AUC
- Precision, recall, and F1 at fixed sparsity thresholds
- Brier score and log loss when probabilistic predictions are available

---

### Reporting and Analysis
For each combination of graph topology, dynamical process, rollout mode, and random seed, results are recorded in a standardized table. Aggregate summaries compare performance across rollout modes to identify systematic patterns.

These comparisons reveal:
- When access to true interactions is essential for accurate prediction
- When learned latent dynamics are sufficient for autonomous forecasting
- When closed-loop generation leads to degradation or stabilization of behavior

---

### Interpretation
By comparing driven, free, and self rollouts across structurally and dynamically diverse systems, this experiment directly probes the extent to which interaction graphs are necessary modeling primitives versus emergent artifacts of latent interaction fields. Performance differences across rollout modes provide evidence for—or against—the hypothesis that collective dynamics can be inferred and sustained without explicit access to underlying graph structure.
