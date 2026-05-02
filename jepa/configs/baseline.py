from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class JEPAConfig:
    vocab_size: int = 50304
    n_layers: int = 8
    split_index: int = 4
    model_dim: int = 512
    n_heads: int = 8
    head_dim: int = 64
    proj_dim: int = 128
    seq_len: int = 1024
    rope_base: float = 10000.0


@dataclass(frozen=True, slots=True)
class TrainConfig:
    seqs_per_step: int = 8
    seq_len: int = 1024
    total_steps: int = 50_000
    warmup_steps: int = 2_000
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    tau: float = 0.1
    neg_strategy: str = "cross_batch_all"
    neg_subsample_cap: int = 8192
    val_every: int = 250
    val_steps: int = 16
    probe_every: int = 5_000
    probe_train_steps: int = 200
    probe_lr: float = 3e-4
    log_every: int = 10
    seed: int = 42
    compile: bool = True

    train_data_glob: str = "data/fineweb10B/fineweb_train_*.bin"
    val_data_glob: str = "data/fineweb10B/fineweb_val_*.bin"


@dataclass(frozen=True)
class Config:
    model: JEPAConfig = field(default_factory=JEPAConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


CONFIG = Config()
