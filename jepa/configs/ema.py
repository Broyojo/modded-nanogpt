"""51M baseline + EMA target encoder ablation.

Same architecture as baseline; sets use_ema=True with momentum=0.996 (BYOL/V-JEPA
default range). Test whether EMA target reduces position-cheating ratio.
"""
import dataclasses

from jepa.configs.baseline import Config, JEPAConfig, TrainConfig


CONFIG = Config(
    model=JEPAConfig(),
    train=dataclasses.replace(TrainConfig(), use_ema=True, ema_momentum=0.996),
)
