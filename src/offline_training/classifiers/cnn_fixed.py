import torch.nn as nn
from src.offline_training.classifiers.cnn_lstm_combined import _build_cnn, _build_head


_CHANNEL_SLICE = {"both": slice(0, 2), "foreground": slice(0, 1), "background": slice(1, 2)}


class CNNFixed(nn.Module):
    """CNN + MLP for IoU calibration from the fixed anchor channel"""

    n_references = 1

    def __init__(self, n_channels=48, cnn_dim=256, mlp_hidden=256, dropout=0.2, channel="both"):
        super().__init__()

        self.channel = channel
        self._channel_slice = _CHANNEL_SLICE[channel]
        in_channels = self._channel_slice.stop - self._channel_slice.start

        self.cnn = _build_cnn(n_channels=n_channels, cnn_dim=cnn_dim, in_channels=in_channels)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.head = _build_head(cnn_dim, mlp_hidden, dropout)

    def forward(self, x):
        """`x`: `(B, n_input_channels, H, W, 2)`. Returns IoU `(B, 1)`."""

        x = x[:, 0, :, :, self._channel_slice].permute(0, 3, 1, 2)
        x = self.global_pool(self.cnn(x)).flatten(start_dim=1)
        return self.head(x).float()
