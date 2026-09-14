import numpy as np
import torch

from dataclasses import dataclass
from src.modules.baselines.memory_oracle import MemoryOracle


@dataclass
class MaskOracle(MemoryOracle):
    """MemoryOracle with oracle mask SELECTION: on visible frames it keeps the proposal whose mask scores the
    highest true IoU against the ground truth -- an upper bound on per-frame selection -- instead of SAM 2's
    IoU token. On occluded frames it falls back to the baseline selection, since there is nothing to verify
    against. The GT-verified commit gate is inherited unchanged."""

    def choose(self, mask_preds, iou_scores, bboxes_norm, visible, truth=None):
        if not visible:
            return super().choose(mask_preds, iou_scores, bboxes_norm, visible, truth)

        candidates = (mask_preds[0, 1:] > 0.0).cpu().numpy()
        return 1 + int(np.argmax(self.score(candidates, bboxes_norm, truth)))

    def reported_mask(self, mask_preds, best_idx, chosen_mask):
        """Selection is this arm's intervention, so it is scored on exactly the mask it selected."""

        return chosen_mask.to(torch.float64)
