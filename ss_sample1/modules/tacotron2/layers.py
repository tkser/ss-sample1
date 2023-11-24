import torch
from librosa.filters import mel as librosa_mel_fn

from .audio_processing import dynamic_range_compression, dynamic_range_decompression
from .stft import STFT


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True, w_init_gain: str = "linear") -> None:
        super(__class__, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int | None = None,
        dilation: int = 1,
        bias: bool = True,
        w_init_gain: str = "linear",
    ) -> None:
        super(__class__, self).__init__()
        if padding is None:
            if kernel_size % 2 != 1:
                msg = "kernel_size must be odd number."
                raise ValueError(msg)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        torch.nn.init.xavier_uniform_(self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        conv_signal = self.conv(signal)
        return conv_signal


class TacotronSTFT(torch.nn.Module):
    def __init__(
        self,
        filter_length: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mel_channels: int = 80,
        sampling_rate: int = 22050,
        mel_fmin: float = 0.0,
        mel_fmax: float = 8000.0,
    ) -> None:
        super(__class__, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sr=sampling_rate,
            n_fft=filter_length,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)

    def spectral_normalize(self, magnitudes: torch.Tensor) -> torch.Tensor:
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes: torch.Tensor) -> torch.Tensor:
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y: torch.Tensor) -> torch.Tensor:
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        if torch.min(y.data) < -1 or torch.max(y.data) > 1:
            msg = "The range of wave is not in [-1, 1]."
            raise ValueError(msg)

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output
