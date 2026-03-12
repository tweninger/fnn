# models/tgn_model.py
from core.interfaces import ComposedInteractionModel
from encoders.event_encoder import TGNEventEncoder
from aggregators.deepsets import DeepSetsAggregator
from aggregators.sum import SumAggregator
from aggregators.settransformers import SetTransformerAggregator
from aggregators.hopfield import HopfieldAggregator
from aggregators.ift import IFTLaplacianAggregator
from updates.ift_update import IFTDiffusionUpdate
from updates.hopfield_update import HopfieldUpdate
from updates.hnn import HNNUpdate
from updates.lnn import LNNUpdate
from updates.tgn_gru import TGNGRUUpdate
from scorers.event_scorer import DotProductScorer, MLPEdgeScorer
from core.config import ModelConfig
from data.interfaces import DataSpec

# main function - take dataset metadata/spec and cfg/settings for arch and return a complete model
def build_tgn_model(spec: DataSpec, cfg: ModelConfig):
    num_nodes = spec.num_nodes
    event_dim = spec.event_dim if cfg.event_dim is None else cfg.event_dim
    # raw event info in -> learned vector/message out
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
    # switchboard
    AGG_BUILDERS = {
        "sum": lambda c: SumAggregator(add_to_dst=True),
        # process elements individually, aggr in perm invar way (often sum), maybe apply learned transformation
        "deepsets": lambda c: DeepSetsAggregator(
            msg_dim=c.msg_dim,
            out_dim=c.msg_dim,
            hidden_dim=getattr(c, "aggregator_hidden", c.encoder_hidden),
            dropout=c.dropout,
            add_to_dst=True,
            reduce="sum",
        ),
        # instead of sum, let elements interact via attention
        "settransformer": lambda c: SetTransformerAggregator(
            msg_dim=c.msg_dim,
            num_heads=getattr(c, "settf_num_heads", 4),
            ff_dim=getattr(c, "settf_hidden", 128),
            num_layers=getattr(c, "settf_num_layers", 1),
            dropout=c.dropout,
            add_to_dst=True,
            max_events_per_node=getattr(c, "settf_max_events_per_node", 32),
        ), 
        # hopfield style aggre
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
        #hi barbie
        "ift": lambda c: IFTLaplacianAggregator(
            add_to_dst=True, 
            add_to_src=False,
            make_undirected=True
        )
    }
    # take the string from config, find matching builder, instantiate it... or crash w err
    agg_fn = AGG_BUILDERS.get(cfg.aggregator)
    if agg_fn is None:
        raise NotImplementedError(f"aggregator={cfg.aggregator} not wired yet")
    aggregator = agg_fn(cfg)

    # same but w updaters
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
        "ift_update": lambda c: IFTDiffusionUpdate(
            node_dim=c.node_dim,
            msg_dim=c.msg_dim,
            dt=getattr(c, "ift_dt", 1.0),
            gamma=getattr(c, "ift_gamma", 0.0),
            kappa=getattr(c, "ift_kappa", 1.0),
            learn_kappa=getattr(c, "ift_learn_kappa", True),
            kappa_param=getattr(c, "ift_kappa_param", "softplus"),
        ),
    }
    upd_fn = UPDATE_BUILDERS.get(cfg.update)
    if upd_fn is None:
        raise NotImplementedError(f"update={cfg.update} not wired yet")
    update = upd_fn(cfg)

    # given node states/event time/info etc, how do we score candidate next events/edges?
    # training uses ranking loss and metrics, so scorer prob produces those event /edge scores
    # --- Scorer registry ---
    SCORER_BUILDERS = {
        "dot": lambda c: DotProductScorer(), # hi dot product
        "mlp": lambda c: MLPEdgeScorer( # hi barbie
            node_dim=c.node_dim,
            event_dim=event_dim,
            hidden_dim=c.scorer_hidden,
            use_time=c.use_time_features,
            time_emb_dim=c.time_emb_dim,
            dropout=c.scorer_dropout,
        ),
    }
    sc_fn = SCORER_BUILDERS.get(cfg.scorer)
    if sc_fn is None:
        raise NotImplementedError(f"scorer={cfg.scorer} not supported")
    scorer = sc_fn(cfg)

    # full model composed object built from encoder, aggregater, update, scorer, num _ nodes
    # repo arcitectured as pipeline of interchangeable modules
    return ComposedInteractionModel(
        encoder=encoder,
        aggregator=aggregator,
        update=update,
        scorer=scorer,
        num_nodes=num_nodes,
    )
