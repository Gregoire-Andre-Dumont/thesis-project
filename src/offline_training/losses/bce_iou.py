import torch.nn as nn


class BCEIouLoss(nn.Module):
    """Binary cross-entropy on the mask-quality logit."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        if preds.dim() == 1:
            preds = preds.unsqueeze(-1)
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)
        return self.bce(preds[:, 0], targets[:, 0])

    def __repr__(self):
        return "BCEIouLoss()"
