from core.config import ModelConfig
from dataclasses import dataclass, asdict
from typing import Optional, Sequence
import itertools

@dataclass(frozen=True) #once created, don't change the fields (immutable)
# one specific experiment setting/one run in the sweep
class SweepRun:
    name: str
    model_cfg: ModelConfig
    # optional train overrides; keep TrainConfig stable and override only what you need
    lr: Optional[float] = None
    weight_decay: Optional[float] = None
    num_neg: Optional[int] = None
    tbptt_steps: Optional[int] = None
    seed: int = 0

# BUILDS HUGE EXPERIMENT SWEEP!
def make_runs(
    base_model_cfg: ModelConfig,
    *, # everything after * must be passed by keyword
    seeds: Sequence[int] = (0,),
    aggregator: Sequence[str] = ("sum", "deepsets", "settransformer"),
    upd: Sequence[str] = ("tgn_gru",),
    dropout: Sequence[float] = (0.0,),
    scorer_dropout: Sequence[float] = (0.0,),
    use_time_features: Sequence[bool] = (False,),
    ift_kappa_param: Sequence[str] = ("softplus", "exp"),
    ift_dt: Sequence[float] = (0.05,),
    ift_gamma: Sequence[float] = (0.0,),
    ift_kappa_init: Sequence[float] = (1.0,),
    ift_kappa_cap: Sequence[bool] = (False,),
    ift_kappa_max: Sequence[float | None] = (None,),    
) -> list[SweepRun]:
    # create EVERY combination of aggregator, dropout, update type, scorer dropout, time features on/off, and seed
    # aka grid search/sweep
    runs: list[SweepRun] = []
    for (agg, do, upd, sdo, tf, seed) in itertools.product( # cartesian product hehe
        aggregator, dropout, upd, scorer_dropout, use_time_features, seeds
    ):
        cfg = ModelConfig(**asdict(base_model_cfg)) # make copy of base model config "make new modelconfig with same fields as base config"
        # customize thse copied config for current run
        cfg.update = upd  # type: ignore 
        cfg.aggregator = agg  # type: ignore
        cfg.dropout = do
        cfg.scorer_dropout = sdo
        cfg.use_time_features = tf

        # build human readable name
        base_name = f"agg={agg}|update={upd}|do={do}|sdo={sdo}|time={tf}"

        # conditional dimension: only for IFT
        # sweep over extra IFT hyperparameters too if update = IFT
        if upd == "ift_update":
            for (kp, dt, ga, k0) in itertools.product( # every combo of those IFT params
                ift_kappa_param, ift_dt, ift_gamma, ift_kappa_init
            ):
                for cap in ift_kappa_cap:
                    if cap:
                        # cap=True => sweep over explicit maxima (skip None)
                        for kmax in ift_kappa_max:
                            if kmax is None:
                                continue
                            cfg2 = ModelConfig(**asdict(cfg)) # specialized config and append a run
                            cfg2.ift_kappa_param = kp  # type: ignore
                            cfg2.ift_dt = float(dt)
                            cfg2.ift_gamma = float(ga)
                            cfg2.ift_kappa = float(k0)
                            cfg2.ift_kappa_cap = True
                            cfg2.ift_kappa_max = float(kmax)

                            name2 = f"{base_name}|{kp}|dt={dt}|gamma={ga}|k0={k0}|cap={kmax}"
                            runs.append(SweepRun(name=name2, model_cfg=cfg2, seed=int(seed)))
                    else:
                        # cap=False => exactly ONE config (don't sweep kmax)
                        cfg2 = ModelConfig(**asdict(cfg))
                        cfg2.ift_kappa_param = kp  # type: ignore
                        cfg2.ift_dt = float(dt)
                        cfg2.ift_gamma = float(ga)
                        cfg2.ift_kappa = float(k0)
                        cfg2.ift_kappa_cap = False
                        cfg2.ift_kappa_max = None

                        name2 = f"{base_name}|{kp}|dt={dt}|gamma={ga}|k0={k0}|cap=none"
                        runs.append(SweepRun(name=name2, model_cfg=cfg2, seed=int(seed)))
            continue # we already handled IFT speically so skip normal run append below
        
        # for no IFT runs, just add one run normally
        runs.append(SweepRun(name=base_name, model_cfg=cfg, seed=int(seed)))
    return runs