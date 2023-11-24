from dataclasses import dataclass, field


@dataclass
class PathConfig:
    wav_glob: str = "*.wav"
    lab_glob: str = "*.lab"
    output_dir: str = "data"
    strict_lab: bool = True


@dataclass
class LabelConfig:
    sec_unit: float = 1.0


@dataclass
class WaveConfig:
    sampling_rate: int = 22050
    max_wav_value: float = 32768.0


@dataclass
class STFTConfig:
    hop_length: int = 256
    win_length: int = 1024
    filter_length: int = 1024


@dataclass
class MelConfig:
    n_mel_channels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: float = 8000.0


@dataclass
class Config:
    preprocess: PathConfig = field(default_factory=PathConfig)
    label: LabelConfig = field(default_factory=LabelConfig)
    wave: WaveConfig = field(default_factory=WaveConfig)
    stft: STFTConfig = field(default_factory=STFTConfig)
    mel: MelConfig = field(default_factory=MelConfig)
