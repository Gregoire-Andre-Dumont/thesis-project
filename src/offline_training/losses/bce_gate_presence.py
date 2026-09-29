import torch.nn as nn


class BCEGatePresenceLoss(nn.Module):
    """Two heads on one trunk: commit at `commit_threshold` on the IoU, presence on the visibility label.

    The gate asks whether a mask is worth writing to memory; the presence head asks whether the target is
    in the scene at all. Those are different questions, so they take different labels: the gate reads the
    proposal's true IoU, the presence head the annotation's own occluded flag, carried as a second label
    column by `MainDataset(presence_label=True)`. Thresholding IoU instead would teach the head to call
    the target absent whenever the mask drifted off it, which is a tracking failure, not an absence.

    Both heads emit raw logits, so the sigmoids live inside the loss."""

    def __init__(self, commit_threshold: float = 0.2, presence_weight: float = 1.0):
        super().__init__()
        self.commit_threshold = commit_threshold
        self.presence_weight = presence_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        if preds.dim() == 1:
            preds = preds.unsqueeze(-1)
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)
        if targets.shape[1] < 2:
            raise ValueError("BCEGatePresenceLoss needs a visibility column: set presence_label on the dataset")

        commit_loss = self.bce(preds[:, 0], (targets[:, 0] > self.commit_threshold).float())
        presence_loss = self.bce(preds[:, 1], targets[:, 1])
        return commit_loss + self.presence_weight * presence_loss

    def __repr__(self):
        return f"BCEGatePresenceLoss(commit={self.commit_threshold}, presence=visibility)"
