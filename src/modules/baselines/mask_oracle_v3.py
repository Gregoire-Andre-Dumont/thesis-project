"""The mask oracle on SAM 3 -- the tracker that GENERATES the calibrator dataset.

Why this exists: the gate in `data/mask_oracle` was trained on rollouts produced by SAM 2. Its features are
Perception Encoder similarity maps, which are backbone-agnostic, but its LABELS and its input distribution
are not -- every mask it ever scored was a SAM 2 mask, corrupted from a SAM 2 memory bank. Deploying that
gate on SAM 3 asks it to judge masks drawn from a distribution it never saw, which is the most likely
explanation for SAMARA-3's advantage being smaller and less stable than SAMARA-2's.

Regenerating the dataset from a SAM 3 rollout removes that mismatch. Everything else about the pipeline --
the corruption sweep, the pseudo-IoU labelling, the anchor-similarity features -- stays exactly as it was,
so the resulting gate differs from the SAM 2 one in the backbone it was trained against and nothing else.

`MemoryOracle3` overrides only what the generation actually changes; `choose`, `score`, `mask_iou` and
`corruption_draw` are inherited so the oracle's DECISIONS are identical across backbones.
"""

import numpy as np
import torch

from dataclasses import dataclass

from src.typing.detection_data import DetectionData
from src.modules.baselines.memory_oracle import MemoryOracle
from src.modules.baselines.mask_oracle import MaskOracle
from src.modules.memories.main_memory_v3 import MainMemoryV3
from src.modules.samara3_model import SamaraModel3
from src.utils.load_bboxes import convert_bbox

MASK_SIDE = 256          # what every consumer downstream of the rollout assumes


def _to_device(encoding, device):
    """Move a SAM 3 image encoding to `device`, preserving its nested tuple/list structure.

    SAM 2's encoding is a flat list of tensors, so the SAM 2 pipeline moves it with a comprehension.
    SAM 3's `SAMv3ImageEncoding` is a nested tuple whose first element is the feature list, so the same
    comprehension would silently iterate the wrong level and hand the model a malformed encoding."""

    if torch.is_tensor(encoding):
        return encoding.to(device)
    if isinstance(encoding, (list, tuple)):
        moved = [_to_device(item, device) for item in encoding]
        return type(encoding)(moved) if isinstance(encoding, tuple) else moved
    return encoding


def _resize_masks(masks):
    """`(k, h, w)` mask logits -> `(k, 256, 256)` binary-thresholded logits.

    SAM 3 decodes at a quarter of its 1008px input (252x252) where SAM 2 gives 256x256. Every consumer --
    `compute_iou`, the stored `proposal_masks`, the feature crops -- assumes 256, and a 252 array would be
    broadcast or truncated rather than rejected."""

    if masks.shape[-2:] == (MASK_SIDE, MASK_SIDE):
        return masks
    return torch.nn.functional.interpolate(
        masks[None].float(), size=(MASK_SIDE, MASK_SIDE), mode="bilinear", align_corners=False)[0]


@dataclass
class MemoryOracle3(MemoryOracle):
    """`MemoryOracle` on SAM 3: same GT-verified commit gate, same corruption, different backbone."""

    model: SamaraModel3 | None = None
    main_memory: MainMemoryV3 | None = None

    @torch.inference_mode()
    def pseudo_truth(self, encoded_image, bboxes_norm):
        """SAM 3's segmentation of the GT box -- the mask a frame's proposals are scored against."""

        mask, _, _ = self.model.initialize_video_masking(
            encoded_image, convert_bbox(np.asarray(bboxes_norm, dtype=np.float32)))
        mask = _resize_masks(mask.reshape(1, *mask.shape[-2:]))[0]
        return mask.to(torch.float64).cpu().numpy() > 0.0

    def distractor_memory(self, idx, encoded_image):
        """`(pointer, encoding)` for the nearest distractor, box-prompted through SAM 3.

        Returns the pointer slot as None, which `MainMemoryV3.update_memory` ignores -- a SAM 3 memory
        encoding already carries its own."""

        _, encoding, pointer = self.model.initialize_video_masking(
            encoded_image, convert_bbox(np.asarray(self.corruption_boxes[idx], dtype=np.float32)))
        return pointer, encoding

    def predict_masks(self, detection_data: DetectionData):
        """Roll SAM 3 over the sequence, committing only on GT-verified visible frames.

        Mirrors `MemoryOracle.predict_masks` decision for decision. It is reimplemented rather than reused
        because three things in the loop are backbone-shaped: the image encoding is one object instead of a
        list, the memory encoder takes that whole object where SAM 2 took `lowres_imgenc`, and the masks
        arrive at 252px."""

        self.main_memory.reset_memory()
        self.main_memory.initialize_references(self.model, detection_data, anchor_index=0)

        n_frames = detection_data.frames.shape[0]
        self.predicted_masks = torch.zeros((n_frames, MASK_SIDE, MASK_SIDE), dtype=torch.float64)

        self.iou_scores = torch.zeros(n_frames, dtype=torch.float64)
        self.object_scores = torch.zeros(n_frames, dtype=torch.float64)
        self.commit_iou = torch.zeros(n_frames, dtype=torch.float64)

        self.proposal_iou_scores = torch.zeros((n_frames, 3), dtype=torch.float64)
        self.proposal_true_iou = torch.zeros((n_frames, 3), dtype=torch.float64)
        self.chosen_index = torch.zeros(n_frames, dtype=torch.int64)
        self.proposal_masks = torch.zeros((n_frames, 3, MASK_SIDE, MASK_SIDE), dtype=torch.float16)

        self.committed_frames = []
        self.corrupted_frames = []
        corruption_rng = np.random.default_rng(self.corruption_seed)
        cache = getattr(self, "frame_cache", None)

        for idx, current_frame in enumerate(detection_data.frames):
            reuse = _to_device(cache[idx], self.device) if cache is not None and idx in cache else None
            mask_preds, iou_scores, object_pointers, object_score, encoded_image, _ = self.model.propose_masks(
                current_frame=current_frame, main_memory=self.main_memory, encoded_image=reuse)

            if cache is not None and idx not in cache:
                cache[idx] = _to_device(encoded_image, "cpu")

            bboxes_norm = detection_data.bboxes_norm[idx]
            visible = bool(detection_data.occlusions[idx] <= 0.5) and float(bboxes_norm[2]) > 0
            truth = self.pseudo_truth(encoded_image, bboxes_norm) if (visible and self.use_mask_iou) else None

            # Resized with the 0th token still attached, so `best_idx` keeps SAM 2's 1-based meaning and
            # `reported_mask` -- shared with the SAM 2 oracle -- indexes the same proposal it would there.
            all_masks = _resize_masks(mask_preds[0])
            proposals = all_masks[1:]
            proposal_iou = (self.score((proposals > 0.0).cpu().numpy(), bboxes_norm, truth)
                            if visible else None)
            self.proposal_iou_scores[idx] = iou_scores.reshape(-1)[1:4].to(torch.float64).cpu()
            self.proposal_masks[idx] = proposals.to(torch.float16).cpu()
            if proposal_iou is not None:
                self.proposal_true_iou[idx] = torch.as_tensor(proposal_iou, dtype=torch.float64)

            best_idx = self.choose(mask_preds, iou_scores, bboxes_norm, visible, truth, proposal_iou)
            self.chosen_index[idx] = int(best_idx) - 1

            chosen_mask, pointer, encoding = self.model.commit_candidate(
                mask_preds, best_idx, object_pointers, object_score, encoded_image)
            chosen_mask = _resize_masks(chosen_mask.reshape(1, *chosen_mask.shape[-2:]))[0]

            self.predicted_masks[idx] = self.reported_mask(all_masks[None].cpu(), best_idx, chosen_mask)
            self.iou_scores[idx] = float(iou_scores.reshape(-1)[best_idx])
            self.object_scores[idx] = float(torch.as_tensor(object_score).reshape(-1)[0])

            corrupt = self.corruption_draw(idx, corruption_rng)

            gated = False
            if visible:
                self.commit_iou[idx] = float(self.score((chosen_mask[None].numpy() > 0.0), bboxes_norm, truth)[0])
                gated = bool(self.commit_iou[idx] > self.iou_threshold)

            if corrupt:
                pointer, encoding = self.distractor_memory(idx, encoded_image)
                self.corrupted_frames.append(idx)

            if corrupt or gated:
                self.main_memory.update_memory(pointer, encoding)
                self.committed_frames.append(idx)

        return self.predicted_masks


@dataclass
class MaskOracle3(MemoryOracle3, MaskOracle):
    """`MaskOracle` on SAM 3. The MRO takes SAM 3's rollout from `MemoryOracle3` and the oracle SELECTION --
    `choose` and `reported_mask` -- from `MaskOracle`, so the selection rule is shared verbatim with the
    SAM 2 generator rather than restated."""
