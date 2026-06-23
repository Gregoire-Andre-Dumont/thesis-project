import torch
import torch.nn as nn


class MSEIouLoss(nn.Module):
    """Mean-squared-error loss on the IoU column only"""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, preds, targets):
        if preds.dim() == 1:
            preds = preds.unsqueeze(-1)
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)
        return self.mse(preds[:, 0], targets[:, 0])

    def __repr__(self):
        return "MSEIouLoss()"
