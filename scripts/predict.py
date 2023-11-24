from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from lightning import seed_everything
from omegaconf import OmegaConf
from torch import from_numpy, set_float32_matmul_precision
from tqdm import tqdm

from ss_sample1.configs.train import Config
from ss_sample1.model import LitModule


def predict(
    cfg: Config,
    input_glob: str,
    output_path: Path,
    model_path: Path,
    device: str = "cpu",
) -> None:
    seed_everything(cfg.seed)
    set_float32_matmul_precision("high")

    model = LitModule.load_from_checkpoint(model_path, cfg=cfg)
    model = model.to(device)
    model.eval()

    in_paths = sorted(Path().glob(input_glob))
    output_path.mkdir(exist_ok=True, parents=True)

    for path in tqdm(in_paths):
        in_data = from_numpy(np.array([np.load(path)])).to(device)
        out_data = model(in_data).cpu().detach().numpy()[0]
        np.save(output_path / path.name, out_data)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, default="conf/train/default_config.yaml")
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--use-gpu", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    predict(cfg, args.input, args.output, args.model_path, "cuda" if args.use_gpu else "cpu")
