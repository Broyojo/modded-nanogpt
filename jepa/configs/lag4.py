"""51M baseline + prediction lag k=4.

Predict z_{t+4} instead of z_{t+1}. Standard CPC trick (van den Oord et al.)
that weakens the position-shortcut: predicting further ahead requires the
model to extract content-bearing structure rather than just "what's next."
"""
import dataclasses

from jepa.configs.baseline import Config, JEPAConfig, TrainConfig


CONFIG = Config(
    model=JEPAConfig(),
    train=dataclasses.replace(TrainConfig(), pred_lag=4),
)
