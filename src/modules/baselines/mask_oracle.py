import numpy as np

from dataclasses import dataclass
from src.modules.baselines.memory_oracle import MemoryOracle
from src.utils.compute_iou import compute_iou


@dataclass
class MaskOracle(MemoryOracle):
    """MemoryOracle with oracle mask SELECTION: on visible frames it keeps the proposal whose bounding box has the
    highest true BOX IoU vs the GT box (an upper bound on per-frame selection), instead of SAM 2's IoU token. On
    occluded frames it falls back to the baseline selection. The GT-verified commit gate is inherited unchanged."""

    def select_index(self, mask_preds, iou_scores, bboxes_norm, visible):
        if not visible:
            return super().select_index(mask_preds, iou_scores, bboxes_norm, visible)
        candidates = (mask_preds[0, 1:] > 0.0).cpu().numpy()
        ious = compute_iou(np.repeat(bboxes_norm[None, :], candidates.shape[0], axis=0), candidates)
        return 1 + int(np.argmax(ious))
