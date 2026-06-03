import torch
from core.config import ModelConfig
from training.interaction_prediction import TrainConfig
from training.node_regression import NodeTrainConfig


def build_base_model_cfg(spec) -> ModelConfig:
    return ModelConfig(
        node_dim=64,
        msg_dim=64,
        event_dim=spec.event_dim,

        # baseline/default choices
        aggregator="ift",
        update="ift_update",
        scorer="mlp",

        # generic widths
        dropout=0.0,
        scorer_dropout=0.0,
        use_time_features=False,

        # task-specific flags
        task="node_regression",
        predictor="mlp_node",
    )



def build_base_train_cfg(spec, device: torch.device) -> NodeTrainConfig:
    return NodeTrainConfig(
        num_nodes=spec.num_nodes,
        lr=1e-3,
        weight_decay=1e-5,
        grad_clip=1.0,
        device=device,
        log_every=20,
        tbptt_steps=1,
        debug=False,
        loss_name="mae", # mse | mae | huber
        selection_metric="median_node_pearson",
    )



def build_base_interaction_model_cfg(spec) -> ModelConfig:
    return ModelConfig(
        node_dim=128,
        msg_dim=128,
        event_dim=spec.event_dim,
        scorer="mlp",
        scorer_hidden=256,
        aggregator="sum",
        update="tgn_gru",
        use_time_features=False,
        dropout=0.0,
        scorer_dropout=0.0,
        encoder_hidden=256,
    )



def build_base_interaction_train_cfg(spec, device: torch.device) -> TrainConfig:
    dataset_name = str(spec.name).lower()
    log_every = 2000 if "lastfm" in dataset_name else 50

    return TrainConfig(
        num_nodes=spec.num_nodes,
        num_neg=50,
        hard_neg=False,
        tbptt_steps=1,
        log_every=log_every,
        device=device,
        weight_decay=1e-3,
        lr=1e-3, # changed from 1e-3 (0.001) for jodie, or 3e-4 (0.0003)
    )