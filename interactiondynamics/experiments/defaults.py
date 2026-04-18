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
        loss_name="mse", # mse | mae | huber
        selection_metric="rmse",
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
        task="ranking",
        predictor="mlp_node",
    )



def build_base_interaction_train_cfg(spec, device: torch.device) -> TrainConfig:
    return TrainConfig(
        num_nodes=spec.num_nodes,
        num_neg=20,
        tbptt_steps=1,
        log_every=50,
        device=device,
        weight_decay=1e-3,
        lr=1e-3,
    )