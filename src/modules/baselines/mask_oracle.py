import numpy as np
import torch

from dataclasses import dataclass
from src.modules.baselines.memory_oracle import MemoryOracle
from src.utils.compute_iou import compute_iou
from src.utils.mask_components import mask_components, component_subsets


@dataclass
class MaskOracle(MemoryOracle):
    """MemoryOracle with oracle mask SELECTION: on visible frames it keeps the proposal whose bounding box has the
    highest true BOX IoU vs the GT box (an upper bound on per-frame selection), instead of SAM 2's IoU token. On
    occluded frames it falls back to the baseline selection. The GT-verified commit gate is inherited unchanged.

    With `use_components` the candidate set widens from the 3 proposals to every connected-component subset
    within them, so a proposal that merges target + distractor can still yield the target alone. The selected
    subset is what this arm reports AND commits -- selection is the intervention here."""

    def choose(self, mask_preds, iou_scores, bboxes_norm, visible):
        if not visible:
            return super().choose(mask_preds, iou_scores, bboxes_norm, visible)

        candidates = (mask_preds[0, 1:] > 0.0).cpu().numpy()
        ious = compute_iou(np.repeat(bboxes_norm[None, :], candidates.shape[0], axis=0), candidates)
        best_idx, best_iou, best_keep = 1 + int(np.argmax(ious)), float(np.max(ious)), None

        if self.use_components:
            for offset in range(candidates.shape[0]):
                proposal = 1 + offset
                components = mask_components(mask_preds[0, proposal])
                if len(components) < 2:
                    continue
                subsets = component_subsets(components)
                subset_ious = compute_iou(np.repeat(bboxes_norm[None, :], len(subsets), axis=0), np.stack(subsets))
                top = int(np.argmax(subset_ious))
                if float(subset_ious[top]) > best_iou:
                    best_idx, best_iou, best_keep = proposal, float(subset_ious[top]), subsets[top]

        return best_idx, best_keep

    def reported_mask(self, mask_preds, best_idx, chosen_mask):
        """Selection is this arm's intervention, so it is scored on exactly the mask it selected -- the
        component subset when one was chosen, the whole proposal otherwise."""

        return chosen_mask.to(torch.float64)
