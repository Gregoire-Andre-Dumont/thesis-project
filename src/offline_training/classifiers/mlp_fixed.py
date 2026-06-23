import torch
import torch.nn as nn


_CHANNEL_SLICE = {"both": slice(0, 2), "foreground": slice(0, 1), "background": slice(1, 2)}


class MLPFixed(nn.Module):
    """Statistics-based MLP for IoU calibration from the fixed anchor channel.

    Compresses the `(H, W, channels)` patch-similarity grid into a small fixed-size
    feature vector of summary statistics (mean, max, std, fraction-high, coverage) per
    similarity channel, then runs a 3-layer MLP. Drop-in replacement for `CNNFixed` with
    much fewer parameters — better fit when the training set is small."""

    n_references = 1

    def __init__(self, mlp_hidden=64, dropout=0.2, channel="both"):
        super().__init__()
        self.channel = channel
        self._channel_slice = _CHANNEL_SLICE[channel]
        in_channels = self._channel_slice.stop - self._channel_slice.start
        self.n_stats = 5                                          # mean, max, std, frac_high, coverage
        n_features = in_channels * self.n_stats

        self.head = nn.Sequential(
            nn.Linear(n_features, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1))

    def forward(self, x):
        """`x`: `(B, n_input_channels, H, W, 2)`. Returns IoU logit `(B, 1)`."""

        # Take anchor channel and optional FG/BG slice. (B, H, W, C)
        x = x[:, 0, :, :, self._channel_slice]
        batch_size, height, width, channels = x.shape
        x = x.reshape(batch_size, height * width, channels)        # (B, n_patches, C)

        # The dataset sentinel-zeros similarity values where the patch is on the wrong
        # side of the predicted mask (FG sim is 0 at BG patches and vice versa). Use the
        # nonzero positions to compute per-channel statistics over the valid patches only.
        nonzero_mask = (x > 0).float()
        counts = nonzero_mask.sum(dim=1).clamp(min=1)              # (B, C)

        means = (x * nonzero_mask).sum(dim=1) / counts
        maxes = x.amax(dim=1)
        variances = ((x - means.unsqueeze(1)) ** 2 * nonzero_mask).sum(dim=1) / counts
        stds = variances.sqrt()
        frac_high = ((x > 0.5).float() * nonzero_mask).sum(dim=1) / counts
        coverage = nonzero_mask.sum(dim=1) / float(height * width)

        features = torch.cat([means, maxes, stds, frac_high, coverage], dim=1).float()
        return self.head(features)
