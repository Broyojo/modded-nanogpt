"""LM baseline matching the large JEPA config (12L / 768D / 12 heads).

Trains a standard next-token-prediction GPT with the same architecture and
training hyperparameters as configs/large.py — so the head-to-head
feature-quality comparison vs the JEPA encoder is apples-to-apples.

There is no encoder/predictor split here; "split_index" in the config is
ignored by jepa.gpt.GPT and only used by compare_features.py to identify
the corresponding middle-layer slice (= layer 6, matching JEPA's split).
"""
from dataclasses import dataclass, field

from jepa.configs.baseline import TrainConfig


@dataclass(frozen=True, slots=True)
class LMConfig:
    vocab_size: int = 50304
    n_layers: int = 12
    split_index: int = 6
    model_dim: int = 768
    n_heads: int = 12
    head_dim: int = 64
    seq_len: int = 1024
    rope_base: float = 10000.0
    use_rope: bool = True


@dataclass(frozen=True)
class Config:
    model: LMConfig = field(default_factory=LMConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


CONFIG = Config()
