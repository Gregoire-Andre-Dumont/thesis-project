import torch.nn as nn


_CHANNEL_SLICE = {
    "both":         slice(0, 2),     # legacy: anchor-FG similarities only (channels 0, 1)
    "foreground":   slice(0, 1),     # channel 0 only
    "background":   slice(1, 2),     # channel 1 only
    "all":          slice(0, 4),     # channels 0-3: similarities + FG-vs-BG diffs
    "diff":         slice(2, 4),     # channels 2-3: the FG-vs-BG diffs alone
    "channels_1_2": slice(1, 3),     # background-FG sim + foreground FG-vs-BG diff
}


def _build_cnn(n_channels, cnn_dim, in_channels=2):
    """CNN: `in_channels` → cnn_dim, spatial 24→12→6. Each reference becomes a 6×6 grid of
    `cnn_dim` tokens. `in_channels` defaults to 2 (foreground + background similarities);
    set to 1 for single-channel ablations."""

    ch1, ch2 = n_channels, n_channels * 2
    return nn.Sequential(
        # Stage 1: 32 → 16
        nn.Conv2d(in_channels, ch1, kernel_size=3, padding=1),
        nn.BatchNorm2d(ch1),
        nn.GELU(),
        nn.Conv2d(ch1, ch1, kernel_size=3, padding=1),
        nn.BatchNorm2d(ch1),
        nn.GELU(),
        nn.MaxPool2d(2),

        # Stage 2: 16 -> 8
        nn.Conv2d(ch1, ch2, kernel_size=3, padding=1),
        nn.BatchNorm2d(ch2),
        nn.GELU(),
        nn.Conv2d(ch2, ch2, kernel_size=3, padding=1),
        nn.BatchNorm2d(ch2),
        nn.GELU(),
        nn.MaxPool2d(2),

        # Stage 3: 8 → 4
        nn.Conv2d(ch2, ch2, kernel_size=3, padding=1),
        nn.BatchNorm2d(ch2),
        nn.GELU(),
        nn.Conv2d(ch2, ch2, kernel_size=3, padding=1),
        nn.BatchNorm2d(ch2),
        nn.GELU(),
        nn.MaxPool2d(2),

        # Projection to cnn_dim (no spatial pooling).
        nn.Conv2d(ch2, cnn_dim, kernel_size=3, padding=1),
        nn.BatchNorm2d(cnn_dim),
        nn.GELU())


def _build_head(cnn_dim, mlp_hidden, dropout):
    """4-layer MLP head producing the calibrated IoU scalar."""

    return nn.Sequential(
        nn.Linear(cnn_dim, mlp_hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(mlp_hidden, mlp_hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(mlp_hidden, mlp_hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(mlp_hidden, 1))


class CNNFixed(nn.Module):
    """CNN + MLP for IoU calibration from the fixed anchor channel."""

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
