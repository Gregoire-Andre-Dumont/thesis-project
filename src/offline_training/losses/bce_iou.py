import torch.nn as nn


class BCEIouLoss(nn.Module):
    """Binary cross-entropy on 'is this proposal worth committing', thresholded from the true IoU.

    For a calibrator trained SOLELY as the commit gate, when gating and selection are given one model each
    instead of two heads on one trunk. The gate is a decision, not a ranking: it asks whether ONE mask is
    good enough to write into the memory bank, and an occluded frame (IoU 0 for every proposal) is a clean
    negative. Giving it its own model lets it spend all its capacity on that boundary, and lets it stop
    training on its own schedule -- the two objectives do not overfit at the same rate.

    The model head emits a raw logit, so the sigmoid lives inside the loss."""

    def __init__(self, threshold: float = 0.2):
        super().__init__()
        self.threshold = threshold
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        if preds.dim() == 1:
            preds = preds.unsqueeze(-1)
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)
        return self.bce(preds[:, 0], (targets[:, 0] > self.threshold).float())

    def __repr__(self):
        return f"BCEIouLoss(threshold={self.threshold})"
