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