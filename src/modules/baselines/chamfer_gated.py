from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from src.modules.baselines.sam_baseline import SAMBaseline
from src.offline_training.dataset_encoders import crop_around_masks


def foreground_chamfer(reference, candidate):
    """Unidirectional foreground chamfer (candidate -> reference); NaN if either side is empty."""

    if reference.shape[0] == 0 or candidate.shape[0] == 0:
        return float("nan")
    reference = F.normalize(reference, dim=-1)
    candidate = F.normalize(candidate, dim=-1)
    return float((candidate @ reference.T).max(dim=1).values.mean())


@dataclass
class ChamferGatedBaseline(SAMBaseline):
    """SAM 2 policy that commits to memory on the mixed chamfer re-ID score instead of SAM's predicted IoU:
    commit iff  anchor_weight * chamfer(pred, anchor) + (1 - anchor_weight) * mean chamfer(pred, recent) >=
    chamfer_threshold. It records the committed frames' foreground tokens so the target-vs-distractor AUC can
    be scored against this exact memory afterwards. Everything else (rollout, frame cache) is inherited."""

    encode: object = None              # crops (n, H, W, 3) uint8 -> (n, grid*grid, dim) patch tokens
    anchor_weight: float = 0.5
    memory_size: int = 8
    chamfer_threshold: float = 0.0
    crop_size: int = 512

    def prepare(self, anchor_foreground, floor):
        """Set the anchor foreground tokens and crop floor, and reset the committed-token memory for a rollout.
        `committed_tokens` is the bounded window the gate compares against (mirrors SAM's live memory); the
        unbounded `committed_log` keeps every commit's (frame index, tokens) so the memory as-of any later frame
        can be reconstructed when the target-vs-distractor F1 is scored afterwards."""

        self.anchor_foreground = anchor_foreground
        self.floor = floor
        self.committed_tokens = deque(maxlen=self.memory_size)
        self.committed_log = []
        self.frame_index = 0
        if self.main_memory is not None:
            self.main_memory.max_memory_history = self.memory_size

    @torch.inference_mode()
    def predicted_foreground(self, chosen_mask, frame):
        """Foreground tokens of the predicted mask, cropped at the anchor scale and encoded."""

        mask = chosen_mask.numpy() if hasattr(chosen_mask, "numpy") else np.asarray(chosen_mask)
        crops, crop_masks = crop_around_masks(frame[None], mask[None].astype(np.float32), self.crop_size, 0.2, self.floor)
        tokens = self.encode(crops)

        grid_size = round(tokens.shape[1] ** 0.5)
        crop_mask = torch.from_numpy(crop_masks).unsqueeze(1).to(tokens.device).float()
        foreground = (F.interpolate(crop_mask, size=(grid_size, grid_size), mode="nearest") > 0.5).flatten(1)[0]
        return tokens[0][foreground].clone()

    def should_commit(self, object_scores, iou_scores, chosen_mask, frame):
        """Commit gate on the mixed chamfer score; on commit, records the frame's foreground tokens."""

        predicted = self.predicted_foreground(chosen_mask, frame)
        anchor_similarity = foreground_chamfer(self.anchor_foreground, predicted)

        recent = list(self.committed_tokens)
        if recent:
            recent_similarity = float(np.mean([foreground_chamfer(entry, predicted) for entry in recent]))
        else:
            recent_similarity = anchor_similarity
        score = self.anchor_weight * anchor_similarity + (1 - self.anchor_weight) * recent_similarity

        committed = score >= self.chamfer_threshold
        if committed:
            self.committed_tokens.append(predicted)
            self.committed_log.append((self.frame_index, predicted))
        self.frame_index += 1
        return committed
