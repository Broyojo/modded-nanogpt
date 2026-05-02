"""12L / 768D capacity-scale-up config.

~145M params (vs baseline 51M). 12 transformer layers split 6+6, 12 heads × 64
head_dim. Same training recipe as baseline (AdamW, cosine schedule, bf16+compile)
but at proportionally lower throughput (~12-15 sps vs 35 sps).
"""
from dataclasses import dataclass, field

from jepa.configs.baseline import TrainConfig


@dataclass(frozen=True, slots=True)
class JEPAConfig:
    vocab_size: int = 50304
    n_layers: int = 12
    split_index: int = 6
    model_dim: int = 768
    n_heads: int = 12
    head_dim: int = 64
    proj_dim: int = 128
    seq_len: int = 1024
    rope_base: float = 10000.0


@dataclass(frozen=True)
class Config:
    model: JEPAConfig = field(default_factory=JEPAConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


CONFIG = Config()
