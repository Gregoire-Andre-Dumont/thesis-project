import torch.nn as nn


_CHANNEL_SLICE = {
    "both":         slice(0, 2),     # anchor-FG similarities: foreground + background (channels 0, 1)
    "foreground":   slice(0, 1),     # channel 0 only
    "background":   slice(1, 2),     # channel 1 only
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


def _build_head(cnn_dim, mlp_hidden, dropout, n_outputs=1):
    """4-layer MLP head. `n_outputs` 1 is the calibrated IoU alone; 2 adds the commit logit beside it."""

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
        nn.Linear(mlp_hidden, n_outputs))


class CNNFixed(nn.Module):
    """CNN + MLP over the fixed anchor's similarity map.

    One output, one objective. The controller's two decisions get a model each, because they are different
    questions: SELECTION wants an ordering over the three proposals (`MSEIouLoss` on true IoU), GATING wants
    a calibrated yes/no on one mask (`BCEIouLoss` on IoU > threshold). They read the same features but
    overfit at different rates, so sharing a trunk would force them to stop training together."""

    def __init__(self, n_channels=48, cnn_dim=256, mlp_hidden=256, dropout=0.2, channel="both", n_outputs=1):
        super().__init__()

        self.channel = channel
        self.n_outputs = n_outputs
        self._channel_slice = _CHANNEL_SLICE[channel]
        in_channels = self._channel_slice.stop - self._channel_slice.start

        self.cnn = _build_cnn(n_channels=n_channels, cnn_dim=cnn_dim, in_channels=in_channels)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.head = _build_head(cnn_dim, mlp_hidden, dropout, n_outputs)

    def forward(self, x):
        """`x`: `(B, n_input_channels, H, W, 2)`. Returns `(B, n_outputs)`: predicted IoU, then commit logit."""

        x = x[:, 0, :, :, self._channel_slice].permute(0, 3, 1, 2)
        x = self.global_pool(self.cnn(x)).flatten(start_dim=1)
        return self.head(x).float()
