import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pyworld as pw
import torch
from librosa.util import normalize
from omegaconf import OmegaConf
from scipy.interpolate import interp1d
from scipy.io.wavfile import read as read_wav
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ss_sample1.configs.preprocess import Config
from ss_sample1.modules.tacotron2.layers import TacotronSTFT
from ss_sample1.utils.label import Label

logging.basicConfig(level=logging.INFO)


def get_mel_from_wav(wav: np.ndarray, stft: TacotronSTFT) -> np.ndarray:
    wav = torch.from_numpy(wav).unsqueeze(0)
    wav = torch.autograd.Variable(wav, requires_grad=False)
    mel = stft.mel_spectrogram(wav)
    mel = mel.squeeze(0)
    mel = mel.numpy().astype(np.float32)
    return mel


def remove_outlier(values: np.ndarray) -> np.ndarray:
    values = np.array(values)
    p25 = np.percentile(values, 25)
    p75 = np.percentile(values, 75)
    lower = p25 - 1.5 * (p75 - p25)
    upper = p75 + 1.5 * (p75 - p25)
    normal_indices = np.where((values > lower) & (values < upper))[0]
    return values[normal_indices]


def preprocess(cfg: Config) -> None:
    """Preprocess wav and lab files.

    Args:
        output_dir (Path): output directory.
        config_path (Path): config file path.
    """
    logging.info("Preprocess")

    output_dir = Path(cfg.preprocess.output_dir)
    output_dir.mkdir(exist_ok=True)

    wav_files = sorted(Path().glob(cfg.preprocess.wav_glob))
    lab_files = sorted(Path().glob(cfg.preprocess.lab_glob))

    if len(wav_files) == len(lab_files):
        msg = "Number of wav and lab files are the same."
        raise ValueError(msg)

    output_dirs = {
        "duration": output_dir / "duration",
        "pitch": output_dir / "pitch",
        "mel": output_dir / "mel",
        "accent": output_dir / "accent",
    }

    for output_dir in output_dirs.values():
        output_dir.mkdir(exist_ok=True)

    pitch_scaler = StandardScaler()
    n_frames = 0

    stft = TacotronSTFT(
        filter_length=cfg.stft.filter_length,
        hop_length=cfg.stft.hop_length,
        win_length=cfg.stft.win_length,
        n_mel_channels=cfg.mel.n_mel_channels,
        sampling_rate=cfg.wave.sampling_rate,
        mel_fmin=cfg.mel.mel_fmin,
        mel_fmax=cfg.mel.mel_fmax,
    )

    for wav_file, lab_file in tqdm(zip(wav_files, lab_files, strict=True), total=len(wav_files), desc="Preprocess"):
        wav_id = wav_file.stem
        lab_id = lab_file.stem

        if wav_id != lab_id:
            msg = f"wav_id ({wav_id}) and lab_id ({lab_id}) are different."
            raise ValueError(msg)

        label = Label.load_from_path(lab_file, sec_unit=cfg.label.sec_unit)
        _, durations, start_time, end_time = label.get_alignments(
            sampling_rate=cfg.wave.sampling_rate,
            hop_length=cfg.stft.hop_length,
        )

        sr, wav = read_wav(wav_file)

        if sr != cfg.wave.sampling_rate:
            msg = f"Sampling rate of wav file ({sr}) is different from config ({cfg.wave.sampling_rate})."
            raise ValueError(msg)

        wav = wav / cfg.wave.max_wav_value
        wav = normalize(wav)
        wav = wav[int(cfg.wave.sampling_rate * start_time) : int(cfg.wave.sampling_rate * end_time)].astype(np.float32)

        pitch, t = pw.dio(
            wav.astype(np.float64),
            cfg.wave.sampling_rate,
            frame_period=cfg.stft.hop_length / cfg.wave.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, cfg.wave.sampling_rate)
        pitch = pitch[: sum(durations)]

        if np.sum(pitch != 0) <= 1:
            msg = "Number of non-zero pitch values is less than or equal to 1."
            raise ValueError(msg)

        nonzeros = np.where(pitch != 0)[0]
        interp_fn = interp1d(
            nonzeros,
            pitch[nonzeros],
            fill_value=(pitch[nonzeros[0]], pitch[nonzeros[-1]]),
            bounds_error=False,
        )
        pitch = interp_fn(np.arange(0, len(pitch)))

        _pos = 0
        for i, d in enumerate(durations):
            if d > 0:
                pitch[i] = np.mean(pitch[_pos : _pos + d])
            else:
                pitch[i] = 0
            _pos += d
        pitch = pitch[:, len(durations)]

        mel_spec = get_mel_from_wav(wav, stft)
        mel_spec = mel_spec[:, : sum(durations)]

        accent = list(label.get_accent_ids())

        pitch_r = remove_outlier(pitch)
        n_frame = mel_spec.shape[1]
        n_frames += n_frame

        if len(pitch_r) > 0:
            pitch_scaler.partial_fit(pitch_r.reshape(-1, 1))

        duration_file = output_dirs["duration"] / f"{wav_id}.npy"
        pitch_file = output_dirs["pitch"] / f"{wav_id}.npy"
        mel_file = output_dirs["mel"] / f"{wav_id}.npy"
        accent_file = output_dirs["accent"] / f"{wav_id}.npy"

        np.save(duration_file, durations)
        np.save(pitch_file, pitch)
        np.save(mel_file, mel_spec.T)
        np.save(accent_file, accent)

    logging.info("Normalize pitch")

    pitch_mean = pitch_scaler.mean_[0]
    pitch_std = pitch_scaler.scale_[0]
    pitch_min, pitch_max = np.finfo(np.float32).max, np.finfo(np.float32).min
    for wav_file in tqdm(wav_files, total=len(wav_files), desc="Normalize pitch"):
        wav_id = wav_file.stem
        pitch_file = output_dirs["pitch"] / f"{wav_id}.npy"
        pitch = np.load(pitch_file)
        pitch = (pitch - pitch_mean) / pitch_std
        np.save(pitch_file, pitch)

        pitch_min = min(pitch_min, *pitch)
        pitch_max = max(pitch_max, *pitch)

    logging.info("Save stats")

    stats_path = output_dir / "stats.json"
    stats = {
        "pitch": {
            "mean": pitch_mean,
            "std": pitch_std,
            "min": pitch_min,
            "max": pitch_max,
        },
    }
    with Path.open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)

    logging.info(f"Number of frames: {n_frames}")
    logging.info(f"Total time: {n_frames * cfg.stft.hop_length / cfg.wave.sampling_rate / 3600:.2f} hours")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, default="conf/preprocess/default_config.yaml")
    args = parser.parse_args()

    base_cfg = OmegaConf.structured(Config)
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, cfg)

    preprocess(cfg)
