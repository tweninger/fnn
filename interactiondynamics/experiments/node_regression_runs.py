from core.config import ModelConfig
from dataclasses import dataclass, asdict
from typing import Optional, Sequence
import itertools

@dataclass(frozen=True)
class NodeSweepRun:
    name: str
    model_cfg: ModelConfig
    lr: Optional[float] = None
    weight_decay: Optional[float] = None
    tbptt_steps: Optional[int] = None
    seed: int = 0

def make_node_runs(
    base_model_cfg: ModelConfig,
    *,
    seeds: Sequence[int] = (0,),
    aggregator: Sequence[str] = ("sum", "deepsets", "settransformer"),
    upd: Sequence[str] = ("tgn_gru",),
    dropout: Sequence[float] = (0.0,),
    predictor_dropout: Sequence[float] = (0.0,),
    use_time_features: Sequence[bool] = (False,),
    ift_kappa_param: Sequence[str] = ("softplus", "exp"),
    ift_dt: Sequence[float] = (0.05,),
    ift_gamma: Sequence[float] = (0.0,),
    ift_kappa_init: Sequence[float] = (1.0,),
    ift_kappa_cap: Sequence[bool] = (False,),
    ift_kappa_max: Sequence[float | None] = (None,),
) -> list[NodeSweepRun]:
    runs: list[NodeSweepRun] = []

    for (agg, do, update_name, pdo, tf, seed) in itertools.product(
        aggregator, dropout, upd, predictor_dropout, use_time_features, seeds
    ):
        cfg = ModelConfig(**asdict(base_model_cfg))
        cfg.update = update_name          # type: ignore[attr-defined]
        cfg.aggregator = agg              # type: ignore[attr-defined]
        cfg.dropout = do
        cfg.scorer_dropout = pdo          # reuse this width/dropout field for predictor head too
        cfg.use_time_features = tf

        # if your ModelConfig does not already define these fields,
        # add them there, or dynamically attach them if your class allows it.
        cfg.task = "node_regression"      # type: ignore[attr-defined]
        cfg.predictor = "mlp_node"        # type: ignore[attr-defined]

        base_name = f"agg={agg}|update={update_name}|do={do}|pdo={pdo}|time={tf}"

        if update_name == "ift_update":
            for (kp, dt, ga, k0) in itertools.product(
                ift_kappa_param, ift_dt, ift_gamma, ift_kappa_init
            ):
                for cap in ift_kappa_cap:
                    if cap:
                        for kmax in ift_kappa_max:
                            if kmax is None:
                                continue
                            cfg2 = ModelConfig(**asdict(cfg))
                            cfg2.task = "node_regression"          # type: ignore[attr-defined]
                            cfg2.predictor = "mlp_node"            # type: ignore[attr-defined]
                            cfg2.ift_kappa_param = kp             # type: ignore[attr-defined]
                            cfg2.ift_dt = float(dt)
                            cfg2.ift_gamma = float(ga)
                            cfg2.ift_kappa = float(k0)
                            cfg2.ift_kappa_cap = True
                            cfg2.ift_kappa_max = float(kmax)

                            name2 = f"{base_name}|{kp}|dt={dt}|gamma={ga}|k0={k0}|cap={kmax}"
                            runs.append(NodeSweepRun(name=name2, model_cfg=cfg2, seed=int(seed)))
                    else:
                        cfg2 = ModelConfig(**asdict(cfg))
                        cfg2.task = "node_regression"              # type: ignore[attr-defined]
                        cfg2.predictor = "mlp_node"                # type: ignore[attr-defined]
                        cfg2.ift_kappa_param = kp                 # type: ignore[attr-defined]
                        cfg2.ift_dt = float(dt)
                        cfg2.ift_gamma = float(ga)
                        cfg2.ift_kappa = float(k0)
                        cfg2.ift_kappa_cap = False
                        cfg2.ift_kappa_max = None

                        name2 = f"{base_name}|{kp}|dt={dt}|gamma={ga}|k0={k0}|cap=none"
                        runs.append(NodeSweepRun(name=name2, model_cfg=cfg2, seed=int(seed)))
            continue

        runs.append(NodeSweepRun(name=base_name, model_cfg=cfg, seed=int(seed)))

    return runs
