import torch
import torch.nn as nn


class HeuristicFG(nn.Module):
    """Parameter-free calibrator: commit when the average anchor-channel FG similarity over
    valid (non-sentinel) patches exceeds `threshold`.

    Returns a logit shaped `(B, 1)` such that `sigmoid(logit) > 0.5  iff  mean_fg > threshold`.
    `scale` controls the sharpness of the sigmoid transition around the threshold — large
    values make the gate near-binary."""

    n_references = 1

    def __init__(self, threshold: float = 0.5, scale: float = 10.0):
        super().__init__()
        self.threshold = float(threshold)
        self.scale = float(scale)

    def forward(self, x):
        """`x`: `(B, n_input_channels, H, W, 2)`. Returns logit `(B, 1)`."""

        anchor_fg = x[:, 0, :, :, 0].float()                     # (B, H, W)
        valid = (anchor_fg > 0).float()
        counts = valid.sum(dim=(1, 2)).clamp(min=1)
        mean_fg = (anchor_fg * valid).sum(dim=(1, 2)) / counts   # (B,)
        return (self.scale * (mean_fg - self.threshold)).unsqueeze(1)
