from torch import Tensor, nn


class Predictor(nn.Module):
    def __init__(self) -> None:
        super(__class__, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x
