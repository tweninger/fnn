from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import torch

from interactiondynamics.core.config import ModelConfig
from interactiondynamics.eval.evaluate import EvalSlices

PredictionMode = Literal["state", "delta", "state_plus_delta"]


@dataclass(frozen=True)
class SweepRun:
    name: str
    model_cfg: ModelConfig
    lr: Optional[float] = None
    weight_decay: Optional[float] = None
    num_neg: Optional[int] = None
    tbptt_steps: Optional[int] = None
    prediction_mode: Optional[PredictionMode] = None
    seed: int = 0


@dataclass
class RunResult:
    name: str
    seed: int
    epochs: int
    best_val_loss: float
    best_val_mrr: float
    best_epoch: int
    best_snapshot: dict[str, Any]
    final_snapshot: dict[str, Any]
    wall_sec: float
    best_objective_path: Optional[str] = None
    best_objective_goal: Optional[str] = None
    best_objective_value: float = float("nan")


@dataclass
class TrainConfig:
    num_nodes: int
    num_neg: int = 20
    node_loss_weight: float = 0.0
    node_target_type: str = "regression"
    edge_target_type: str = "regression"
    node_target_mode: str = "raw"
    edge_target_mode: str = "raw"
    edge_target_scale: str = "raw"
    prediction_mode: PredictionMode = "state"
    edge_target_mean: float = 0.0
    edge_target_std: float = 1.0
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    device: torch.device = torch.device("cpu")
    log_every: int = 50
    tbptt_steps: int = 1
    update_before_score: bool = True
    debug: bool = False
    rollout_horizon: int = 5
    # Optional test-only intervention: retain the synthetic diffusion drive for
    # this many rollout steps, then set it to zero in the counterfactual world.
    synthetic_drive_cutoff: Optional[int] = None
    # Optional field-intervention reports.  ``free`` retains oracle graph
    # signals from the counterfactual simulator, whereas ``self_free``
    # regenerates those signals from the model's own rollout prediction.
    synthetic_free_rollout: bool = False
    synthetic_self_free_rollout: bool = False
    # Percent of autoregressive training chunks converted into a zero-drive,
    # self-generated suffix.  This is meaningful only for rollout training.
    synthetic_free_train_percent: float = 0.0
    synthetic_free_train_cutoff: int = 1
    # Number of differentiable, autoregressive prediction steps per optimizer
    # update. One preserves the original one-step teacher-forced trainer.
    rollout_train_steps: int = 1
    # Evaluate only after training by default. A positive value enables
    # periodic in-loop evaluation at this epoch interval.
    eval_every: Optional[int] = None
    # Opt-in wall-clock breakdown for training and evaluation stages.
    debug_timing: bool = False


@dataclass(frozen=True)
class RunSuite:
    dataset: str
    dataset_kwargs: dict[str, Any]
    train_cfg: TrainConfig
    model_cfg: ModelConfig
    runs: list[SweepRun]
    epochs: int
    eval_slices: EvalSlices
    save_jsonl_path: Optional[str] = None
