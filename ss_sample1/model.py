import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor, nn, optim

from ss_sample1.config import Config
from ss_sample1.networks.predictor import Predictor


class LitModule(LightningModule):
    def __init__(self, cfg: Config) -> None:
        super().__init__()

        self.model = Predictor(**cfg.model)

        if cfg.optimizer.name == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), **cfg.optimizer.args)
        else:
            raise NotImplementedError

        if cfg.scheduler.name == "StepLR":
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, **cfg.scheduler.args)
        else:
            raise NotImplementedError

        if cfg.loss.name == "L1Loss":
            self.loss_fn = nn.L1Loss(**cfg.loss.args)
        elif cfg.loss.name == "MSELoss":
            self.loss_fn = nn.MSELoss(**cfg.loss.args)
        elif cfg.loss.name == "HuberLoss":
            self.loss_fn = nn.HuberLoss(**cfg.loss.args)
        else:
            raise NotImplementedError

        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch: tuple[Tensor, Tensor, Tensor], _: int) -> STEP_OUTPUT:
        inputs, targets, masks = batch
        outputs = self.forward(inputs)

        outputs, targets = outputs[masks], targets[masks]

        loss = self.loss_fn(outputs, targets).mean()

        result = {"loss": loss}
        self.training_step_outputs.append(result)

        return result

    def on_train_epoch_end(self) -> None:
        loss = torch.stack([x["loss"] for x in self.training_step_outputs]).mean()
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("lr", self.scheduler.get_last_lr()[0], prog_bar=False, on_epoch=True)

    def validation_step(self, batch: tuple[Tensor, Tensor, Tensor], _: int) -> STEP_OUTPUT:
        inputs, targets, masks = batch
        outputs = self.forward(inputs)

        outputs, targets = outputs[masks], targets[masks]

        loss = self.loss_fn(outputs, targets).mean()

        result = {"loss": loss}
        self.validation_step_outputs.append(result)

        return result

    def on_validation_epoch_end(self) -> None:
        loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

    def configure_optimizers(self) -> tuple[list, list]:
        return [self.optimizer], [self.scheduler]
