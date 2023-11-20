from dataclasses import dataclass, field


@dataclass
class OptimizerConfig:
    name: str = "Adam"
    args: dict = field(default_factory=dict)


@dataclass
class SchedulerConfig:
    name: str = "StepLR"
    args: dict = field(default_factory=dict)


@dataclass
class LossConfig:
    name: str = "L1Loss"
    args: dict = field(default_factory=dict)


@dataclass
class DataConfig:
    in_path_glob: str | None = None
    out_path_glob: str | None = None
    batch_size: int = 32
    num_workers: int = 0
    split: list[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])


@dataclass
class CallbackConfig:
    EarlyStopping: dict = field(default_factory=dict)
    ModelCheckpoint: dict = field(default_factory=dict)


@dataclass
class Config:
    project: str = "pytorch_template"
    version: str = "0.0.1"
    log_dir: str = "logs"
    seed: int = 42
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    trainer: dict = field(default_factory=dict)
    data: DataConfig = field(default_factory=DataConfig)
    model: dict = field(default_factory=dict)
    callbacks: CallbackConfig = field(default_factory=CallbackConfig)
