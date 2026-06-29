import torch
import torch.nn as nn


class BCEIouLoss(nn.Module):
    """Binary cross-entropy on the mask-quality logit. Supports `pos_weight` to
    rebalance the positive class — useful when training labels are heavily
    positive-skewed (e.g., the combined_oracle dataset has ~75% positive at
    IoU > 0.5). PyTorch convention: loss = -(pos_weight·y·log(p) + (1-y)·log(1-p)).

      - `pos_weight = 1.0` (default): standard BCE.
      - `pos_weight < 1.0`: down-weights positives → forces optimizer to pay
        attention to the minority (negative) class. Set to `(1-pos_freq)/pos_freq`
        for full class balance — e.g., `0.33` for a 75% positive dataset.
      - `pos_weight > 1.0`: up-weights positives, for the opposite skew."""

    def __init__(self, pos_weight: float | None = None):
        super().__init__()
        self.pos_weight = pos_weight
        if pos_weight is None or pos_weight == 1.0:
            self.bce = nn.BCEWithLogitsLoss()
        else:
            self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight)))

    def forward(self, preds, targets):
        if preds.dim() == 1:
            preds = preds.unsqueeze(-1)
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)
        return self.bce(preds[:, 0], targets[:, 0])

    def __repr__(self):
        return f"BCEIouLoss(pos_weight={self.pos_weight})"
