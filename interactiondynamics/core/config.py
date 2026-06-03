# core/config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional

AggregatorType = Literal["ift", "sum", "deepsets", "settransformer", "hopfield"]
ScorerType = Literal["dot", "mlp"]
UpdateType = Literal["ift_update", "tgn_gru", "lnn", "hnn", "hopfield_update"]
KappaParam = Literal["exp", "softplus"]

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
