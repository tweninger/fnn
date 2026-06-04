# models/tgn_model.py
from interactiondynamics.aggregators.deepsets import DeepSetsAggregator
from interactiondynamics.aggregators.hopfield import HopfieldAggregator
from interactiondynamics.aggregators.ift import IFTLaplacianAggregator
from interactiondynamics.aggregators.settransformers import SetTransformerAggregator
from interactiondynamics.aggregators.sum import SumAggregator
from interactiondynamics.core.config import ModelConfig
from interactiondynamics.core.interfaces import ComposedInteractionModel
from interactiondynamics.data.interfaces import DataSpec
from interactiondynamics.encoders.event_encoder import TGNEventEncoder
from interactiondynamics.scorers.event_scorer import (
    DotProductScorer,
    IFTSecondOrderLinearHVForceScorer,
    MLPEdgeScorer,
)
from interactiondynamics.scorers.node_scorer import MLPNodeScorer
from interactiondynamics.updates.hnn import HNNUpdate
from interactiondynamics.updates.hopfield_update import HopfieldUpdate
from interactiondynamics.updates.ift_update import IFTDiffusionUpdate, IFTSecondOrderUpdate
from interactiondynamics.updates.lnn import LNNUpdate
from interactiondynamics.updates.tgn_gru import TGNGRUUpdate

def build_tgn_model(spec: DataSpec, cfg: ModelConfig):
    num_nodes = spec.num_nodes
    event_dim = spec.event_dim if cfg.event_dim is None else cfg.event_dim

    encoder = TGNEventEncoder(
        node_dim=cfg.node_dim,
        event_dim=event_dim,
        msg_dim=cfg.msg_dim,
        hidden_dim=cfg.encoder_hidden,
        dropout=cfg.dropout,
        use_time_features=cfg.use_time_features,
        time_emb_dim=cfg.time_emb_dim,
    )


    # --- Aggregator registry ---
    AGG_BUILDERS = {
        "sum": lambda c: SumAggregator(add_to_dst=True),
        "deepsets": lambda c: DeepSetsAggregator(
            msg_dim=c.msg_dim,
            out_dim=c.msg_dim,
            hidden_dim=getattr(c, "aggregator_hidden", c.encoder_hidden),
            dropout=c.dropout,
            add_to_dst=True,
            reduce="sum",
        ),
        "settransformer": lambda c: SetTransformerAggregator(
            msg_dim=c.msg_dim,
            num_heads=getattr(c, "settf_num_heads", 4),
            ff_dim=getattr(c, "settf_hidden", 128),
            num_layers=getattr(c, "settf_num_layers", 1),
            dropout=c.dropout,
            add_to_dst=True,
            max_events_per_node=getattr(c, "settf_max_events_per_node", 32),
        ), 
        "hopfield": lambda c: HopfieldAggregator(
            node_dim=cfg.node_dim,
            msg_dim=cfg.msg_dim,
            hidden_dim=getattr(cfg, "hopfield_hidden", 128),
            num_heads=getattr(cfg, "hopfield_heads", 4),
            beta=getattr(cfg, "hopfield_beta", 1.0),
            steps=getattr(cfg, "hopfield_steps", 1),
            max_events_per_node=getattr(cfg, "hopfield_max_events_per_node", 32),
            add_to_dst=True,
            dropout=cfg.dropout,
        ),
        "ift": lambda c: IFTLaplacianAggregator(
            add_to_dst=True, 
            add_to_src=False,
            make_undirected=True,
            laplacian_mode=getattr(c, "ift_laplacian_mode", "ema"),
            ema_beta=getattr(c, "ift_laplacian_beta", 0.9),
            message_reduce=getattr(c, "ift_message_reduce", "sum"),
            force_reduce=getattr(c, "ift_force_reduce", "sum"),
            disable_laplacian=getattr(c, "ift_disable_laplacian", False),
            randomize_laplacian=getattr(c, "ift_randomize_laplacian", False),
            identity_laplacian=getattr(c, "ift_identity_laplacian", False),
            zero_messages=getattr(c, "ift_zero_messages", False),
            direct_drive=getattr(c, "ift_direct_drive", False),
        )
    }
    agg_fn = AGG_BUILDERS.get(cfg.aggregator)
    if agg_fn is None:
        raise NotImplementedError(f"aggregator={cfg.aggregator} not wired yet")
    aggregator = agg_fn(cfg)

    # --- Update registry ---
    UPDATE_BUILDERS = {
        "tgn_gru": lambda c: TGNGRUUpdate(c.node_dim, c.msg_dim),
        "lnn": lambda c: LNNUpdate(
            node_dim=c.node_dim,
            msg_dim=c.msg_dim,
            hidden_dim=getattr(c, "lnn_hidden", 256),
            num_layers=getattr(c, "lnn_layers", 2),
            dt=getattr(c, "lnn_dt", 1.0),
            damping=getattr(c, "lnn_damping", 0.0),
            dropout=c.dropout,
            drive_dim=0,  # set if you pass per-node drive vectors
        ),        
        "hnn": lambda c: HNNUpdate(
            node_dim=c.node_dim,
            msg_dim=c.msg_dim,
            hidden_dim=getattr(c, "hnn_hidden", 256),
            num_layers=getattr(c, "hnn_layers", 2),
            dt=getattr(c, "hnn_dt", 1.0),
            damping=getattr(c, "hnn_damping", 0.0),
            dropout=c.dropout,
            drive_dim=0,
        ),
        "hopfield_update": lambda c: HopfieldUpdate(
            node_dim=c.node_dim,
            msg_dim=c.msg_dim,
            hidden_dim=getattr(c, "hopupd_hidden", 256),
            num_heads=getattr(c, "hopupd_heads", 4),
            beta=getattr(c, "hopupd_beta", 1.0),
            steps=getattr(c, "hopupd_steps", 1),
            dropout=c.dropout,
            gate=getattr(c, "hopupd_gate", "sigmoid"),
            fixed_alpha=getattr(c, "hopupd_alpha", 0.5),
        ),
        "ift_update": lambda c: (
            IFTSecondOrderUpdate(
                node_dim=c.node_dim,
                msg_dim=c.msg_dim,
                event_dim=event_dim,
                alpha=getattr(c, "ift_second_order_alpha", 1.0),
                dt=getattr(c, "ift_second_order_dt", 0.1),
                gamma=getattr(c, "ift_second_order_gamma", 0.0),
                kappa=getattr(c, "ift_second_order_kappa", 1.0),
                learn_params=getattr(c, "ift_second_order_learn_params", True),
                zero_injection=getattr(c, "ift_zero_injection", False),
                inj_clip=getattr(c, "ift_inj_clip", 1.0),
                direct_drive=getattr(c, "ift_direct_drive", False),
                forcing_mode=getattr(c, "ift_forcing_mode", "generic_mlp"),
                drive_feature_idx=getattr(c, "ift_drive_feature_idx", None),
                force_scale_init=getattr(c, "ift_force_scale_init", 1.0),
                force_learn_scale=getattr(c, "ift_force_learn_scale", True),
                force_target_dim=getattr(c, "ift_force_target_dim", 0),
                velocity_init_mode=getattr(c, "ift_velocity_init_mode", "finite_difference"),
                velocity_supervision=getattr(c, "ift_velocity_supervision", True),
                velocity_loss_weight=getattr(c, "ift_velocity_loss_weight", 0.01),
                internal_velocity_loss_weight=getattr(c, "ift_internal_velocity_loss_weight", 0.0),
                readout_mode=getattr(c, "ift2_readout_mode", "default"),
                velocity_teacher_forcing=getattr(c, "ift_velocity_teacher_forcing", False),
            )
            if getattr(c, "ift_update_order", "first") == "second"
            else IFTDiffusionUpdate(
                node_dim=c.node_dim,
                msg_dim=c.msg_dim,
                event_dim=event_dim,
                dt=getattr(c, "ift_dt", 1.0),
                gamma=getattr(c, "ift_gamma", 0.0),
                kappa=getattr(c, "ift_kappa", 1.0),
                learn_kappa=getattr(c, "ift_learn_kappa", True),
                kappa_param=getattr(c, "ift_kappa_param", "softplus"),
                kappa_cap=getattr(c, "ift_kappa_cap", False),
                kappa_max=getattr(c, "ift_kappa_max", None),
                zero_injection=getattr(c, "ift_zero_injection", False),
                inj_clip=getattr(c, "ift_inj_clip", 1.0),
                direct_drive=getattr(c, "ift_direct_drive", False),
                forcing_mode=getattr(c, "ift_forcing_mode", "generic_mlp"),
                drive_feature_idx=getattr(c, "ift_drive_feature_idx", None),
                force_scale_init=getattr(c, "ift_force_scale_init", 1.0),
                force_learn_scale=getattr(c, "ift_force_learn_scale", True),
                force_target_dim=getattr(c, "ift_force_target_dim", 0),
            )
        ),
    }
    upd_fn = UPDATE_BUILDERS.get(cfg.update)
    if upd_fn is None:
        raise NotImplementedError(f"update={cfg.update} not wired yet")
    update = upd_fn(cfg)

    # --- Scorer registry ---
    SCORER_BUILDERS = {
        "dot": lambda c: DotProductScorer(),
        "mlp": lambda c: MLPEdgeScorer(
            node_dim=c.node_dim,
            event_dim=event_dim,
            hidden_dim=c.scorer_hidden,
            use_time=c.use_time_features,
            time_emb_dim=c.time_emb_dim,
            dropout=c.scorer_dropout,
        ),
    }
    if (
        cfg.update == "ift_update"
        and getattr(cfg, "ift_update_order", "first") == "second"
        and getattr(cfg, "ift2_readout_mode", "default") == "linear_h_v_force"
    ):
        oracle_coeffs = None
        if getattr(cfg, "ift2_oracle_init", False) and spec.extra is not None:
            params = dict(spec.extra.get("generator_params") or {})
            if {"a", "b", "c"} <= set(params):
                oracle_coeffs = (float(params["a"]), float(params["b"]), float(params["c"]))
        scorer = IFTSecondOrderLinearHVForceScorer(
            init_mode=getattr(
                cfg,
                "ift2_readout_init_mode",
                "oracle" if getattr(cfg, "ift2_oracle_init", False) else "zero",
            ),
            init_scale=getattr(cfg, "ift2_readout_init_scale", 0.01),
            near_ar1_coeffs=getattr(cfg, "ift2_near_ar1_coeffs", None),
            oracle_coeffs=oracle_coeffs,
            trainable=bool(getattr(cfg, "ift2_readout_trainable", not bool(getattr(cfg, "ift2_oracle_init", False)))),
        )
    else:
        sc_fn = SCORER_BUILDERS.get(cfg.scorer)
        if sc_fn is None:
            raise NotImplementedError(f"scorer={cfg.scorer} not supported")
        scorer = sc_fn(cfg)
    node_scorer = None
    if getattr(cfg, "use_node_scorer", False):
        node_scorer = MLPNodeScorer(
            node_dim=cfg.node_dim,
            hidden_dim=getattr(cfg, "node_scorer_hidden", 128),
            dropout=cfg.scorer_dropout,
        )

    return ComposedInteractionModel(
        encoder=encoder,
        aggregator=aggregator,
        update=update,
        scorer=scorer,
        num_nodes=num_nodes,
        node_scorer=node_scorer,
    )
