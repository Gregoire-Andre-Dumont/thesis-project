import torch.nn as nn


class CompositeIouLoss(nn.Module):
    """One trunk, two objectives: MSE on the IoU the controller RANKS by, BCE on the commit decision it GATES
    by.

    The two decisions are not the same question and a single output cannot serve both. Ranking needs an
    ordering over the three proposals, which regression gives. Gating needs a calibrated boundary -- is this
    one mask worth writing to memory -- and a threshold on a regressed IoU is not that: the regressor is free
    to compress its range anywhere, so a fixed cut on it can sit outside the predicted distribution entirely
    and leave the gate permanently open or shut.

    `preds` is `(B, 2)`: column 0 the predicted IoU, column 1 a raw commit logit. `targets` carries the true
    IoU in column 0, thresholded here into the binary commit label. `gate_weight` trades the two off; both
    read the same features, so the regression term also regularises the classifier."""

    def __init__(self, threshold: float = 0.2, gate_weight: float = 1.0):
        super().__init__()
        self.threshold = threshold
        self.gate_weight = gate_weight
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        if preds.dim() == 1:
            preds = preds.unsqueeze(-1)
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)

        true_iou = targets[:, 0]
        regression = self.mse(preds[:, 0], true_iou)
        if preds.shape[1] < 2:
            return regression

        gate = self.bce(preds[:, 1], (true_iou > self.threshold).float())
        return regression + self.gate_weight * gate

    def __repr__(self):
        return f"CompositeIouLoss(threshold={self.threshold}, gate_weight={self.gate_weight})"
