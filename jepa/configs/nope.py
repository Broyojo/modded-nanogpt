"""51M baseline + NoPE (no positional encoding).

Test whether removing RoPE reduces position cheating. Theory: causal mask
alone still provides position info ("I have t tokens of context"), but
removing RoPE eliminates the *direct* low-frequency position signal.

Expected: position-cheating ratio drops noticeably but not to zero;
overall val_loss may rise slightly (RoPE provides useful structure for
content tasks too).
"""
import dataclasses

from jepa.configs.baseline import Config, JEPAConfig, TrainConfig


CONFIG = Config(
    model=dataclasses.replace(JEPAConfig(), use_rope=False),
    train=TrainConfig(),
)
