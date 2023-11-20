from argparse import ArgumentParser
from pathlib import Path

from lightning import Trainer, seed_everything
from omegaconf import OmegaConf
from torch import set_float32_matmul_precision

from ss_sample1.config import Config
from ss_sample1.dataset import DataModule
from ss_sample1.model import LitModule


def test(
    cfg: Config,
    model_path: Path,
) -> None:
    seed_everything(cfg.seed)
    set_float32_matmul_precision("high")

    model = LitModule(cfg)

    dm = DataModule(cfg.data)
    dm.setup()

    trainer = Trainer(
        **cfg.trainer,
    )
    trainer.test(model, dm.test_dataloader, ckpt_path=model_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, default="conf/default_config.yaml")
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    test(cfg, args.model_path)
