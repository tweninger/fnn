# core/config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional

AggregatorType = Literal["ift", "sum", "deepsets", "settransformer", "hopfield"]
ScorerType = Literal["dot", "mlp"]
UpdateType = Literal["ift_update", "tgn_gru", "lnn", "hnn", "hopfield_update"]
KappaParam = Literal["exp", "softplus"]
IFTLaplacianMode = Literal["current_bin", "ema", "fixed_ring"]
IFTMessageReduce = Literal["mean", "sum"]
IFTForcingMode = Literal[
    "generic_mlp",
    "linear_event",
    "gated_linear_event",
    "direct_scalar",
    "gated_direct_scalar",
]
IFTUpdateOrder = Literal["first", "second"]
IFTVelocityInitMode = Literal["zero", "learned", "finite_difference"]
IFT2ReadoutMode = Literal["default", "linear_h_v_force"]
IFT2ReadoutInitMode = Literal["zero", "small_random", "near_ar1", "oracle"]

@dataclass
class ModelConfig:
    # Required dims
    node_dim: int = 128
    msg_dim: int = 128

    # Optional (can be inferred from dataset spec, but keeping here is fine)
    event_dim: Optional[int] = None

    # Core module choices
    aggregator: AggregatorType = "sum"
    scorer: ScorerType = "mlp"
    update: UpdateType = "tgn_gru"

    # MLP sizes
    encoder_hidden: int = 256
    scorer_hidden: int = 256
    node_scorer_hidden: int = 128
    aggregator_hidden: int = 256  # used only for deepsets for now
    dropout: float = 0.1
    scorer_dropout: float = 0.1 
    use_node_scorer: bool = False

    # Event-only physical prediction.  When enabled, the model must predict
    # the measured force vector for the *next* positive event.  Ranking heads
    # are deliberately denied that vector as an input, avoiding target leak.
    predict_event_features: bool = False
    event_feature_loss_weight: float = 1.0
    # Force targets are calibrated from the training split before fitting.
    # This controls the extra emphasis placed on large, informative forces;
    # it never changes the observed event inputs.
    event_feature_magnitude_weight: float = 2.0

    # Time features (binned for now)
    use_time_features: bool = False
    time_emb_dim: int = 32  # used only if use_time_features=True

    # SetTransformer knobs (safe defaults)
    settf_num_heads: int = 4
    settf_hidden: int = 128          # internal FF dim
    settf_num_layers: int = 1        # MAB blocks
    settf_max_events_per_node: int = 32

    # Hopfield knobs (safe defaults)
    hopfield_heads: int = 4
    hopfield_hidden: int = 128          # internal proj dim (d_k)
    hopfield_beta: float = 1.0          # temperature / association strength
    hopfield_steps: int = 1             # retrieval iterations
    hopfield_max_events_per_node: int = 32

    # Hopfield update-specific knobs (safe defaults)
    hopupd_heads: int = 4
    hopupd_hidden: int = 256
    hopupd_beta: float = 1.0
    hopupd_steps: int = 1
    hopupd_gate: str = "sigmoid"   # or "fixed"
    hopupd_alpha: float = 0.5      # used if gate=="fixed"

    # ---- LNN knobs ----
    lnn_dt: float = 1.0
    lnn_hidden: int = 256
    lnn_layers: int = 2
    lnn_damping: float = 0.0   # optional velocity damping

    # ---- HNN knobs ----
    hnn_dt: float = 1.0
    hnn_hidden: int = 256
    hnn_layers: int = 2
    hnn_damping: float = 0.0  # optional momentum damping

    # ---- IFT knobs ----
    ift_dt: float = 0.05
    ift_gamma: float = 0.0
    ift_kappa: float = 1.0
    ift_learn_kappa: bool = True
    ift_kappa_param: KappaParam = "softplus"
    ift_kappa_max: float | None = None   # None => no clamp cap
    ift_kappa_cap: bool = False          # if True, clamp to [0, ift_kappa_max]
    ift_laplacian_mode: IFTLaplacianMode = "ema"
    ift_laplacian_beta: float = 0.9
    ift_message_reduce: IFTMessageReduce = "sum"
    ift_force_reduce: IFTMessageReduce = "sum"
    ift_forcing_mode: IFTForcingMode = "generic_mlp"
    ift_drive_feature_idx: Optional[int] = None
    ift_force_scale_init: float = 1.0
    ift_force_learn_scale: bool = True
    ift_force_target_dim: Optional[int] = 0
    ift_disable_laplacian: bool = False
    ift_randomize_laplacian: bool = False
    ift_identity_laplacian: bool = False
    ift_zero_messages: bool = False
    ift_zero_injection: bool = False
    ift_inj_clip: Optional[float] = 1.0
    ift_direct_drive: bool = False
    ift_update_order: IFTUpdateOrder = "first"
    ift_second_order_alpha: float = 1.0
    ift_second_order_dt: float = 0.1
    ift_second_order_gamma: float = 0.0
    ift_second_order_kappa: float = 1.0
    ift_second_order_learn_params: bool = True
    ift_velocity_init_mode: IFTVelocityInitMode = "finite_difference"
    ift_velocity_supervision: bool = True
    ift_velocity_loss_weight: float = 0.01
    ift_internal_velocity_loss_weight: float = 0.0
    ift2_readout_mode: IFT2ReadoutMode = "default"
    ift_velocity_teacher_forcing: bool = False
    ift_history_vel_steps: int = 0
    ift2_oracle_init: bool = False
    ift2_readout_init_mode: IFT2ReadoutInitMode = "zero"
    ift2_readout_init_scale: float = 0.01
    ift2_readout_trainable: bool = True
    ift_rollout_self_generated: bool = False
    ift_rollout_free_drive: bool = False

    # ---- Field neural network (FNN) knobs ----
    # The FNN learns a persistent operator rather than reconstructing one from
    # the observed event pairs in the current bin.
    fnn: bool = False
    fnn_state_dim: int = 4
    fnn_order: int = 2
    fnn_topology_init: float = 0.0
    fnn_gamma_init: float = 0.12
    fnn_omega_init: float = 0.70
    fnn_force_scale_init: float = 1.0
    # Scales observed event impulses before the field update. Kept separate
    # from ``fnn_force_scale_init``, which belongs to the field-difference
    # output decoder used by controlled synthetic recovery.
    fnn_input_force_scale_init: float = 1.0
    fnn_learn_input_force_scale: bool = False
    fnn_dt: float = 0.10
    # Observational streams have an unknown temporal scale. This learns one
    # positive global scale, not a full irregular-time model.
    fnn_learn_dt: bool = False
    # ``observed_sparse`` stores a parameter only for train-observed directed
    # pairs rather than allocating a dense N x N operator.
    fnn_topology_mode: str = "dense"
    # Keep the physical update law fixed unless an experiment explicitly
    # studies parameter recovery.  The topology and force readout remain
    # trainable in either mode.
    fnn_learn_physical_params: bool = False
    # Selective recovery flags are mutually exclusive with the all-parameter
    # mode at the CLI.  They make single-coefficient identifiability tests
    # possible without changing the predictive-model default.
    fnn_learn_gamma: bool = False
    fnn_learn_omega: bool = False
    fnn_learn_force_scale: bool = False
    # Applied once per epoch by the selective scalar-recovery optimizer.
    fnn_physical_recovery_lr: float = 0.1
    # Explicit recovery-only oracle: fix the persistent operator to synthetic
    # hidden truth.  It is never enabled by ordinary benchmark presets.
    fnn_oracle_topology: bool = False
    # Alternating recovery first fits the persistent topology with physical
    # scalars frozen, then freezes topology while fitting omega, gamma, and
    # force scale sequentially from the full physical trajectory. The blocks
    # repeat ``cycles`` times.
    fnn_alternating_recovery: bool = False
    fnn_alternating_topology_epochs: int = 20
    fnn_alternating_physical_epochs: int = 50
    fnn_alternating_cycles: int = 2
    # ``field_difference`` is the mechanism-constrained readout
    # f(i -> j) = c * (h_i - h_j); ``linear`` is a small generic ablation.
    fnn_force_decoder: str = "mlp"
