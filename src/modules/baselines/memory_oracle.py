import cv2
import torch
import numpy as np

from dataclasses import dataclass
from src.typing.detection_data import DetectionData
from src.modules.memories.main_memory import MainMemory
from src.modules.samara_hiera_model import SamaraHieraModel
from src.utils.compute_iou import compute_iou
from src.utils.load_bboxes import convert_bbox

@dataclass
class MemoryOracle:
    """VOS with SAM 2 where the oracle updates the memory during the whole sequence.

    The commit gate uses box IoU (mask bbox vs GT box) by default; set `use_mask_iou` to gate on
    mask IoU against a pseudo-GT mask (SAM box-prompted on the GT box) instead."""

    iou_threshold: float | None = None
    model: SamaraHieraModel | None = None
    main_memory: MainMemory | None = None
    use_mask_iou: bool = False

    def __post_init__(self):
        """Load and initialize the SAM 2 model with quantization."""

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = self.model.to(device=self.device, dtype=self.dtype)

    def predict_masks(self, detection_data: DetectionData):
        """Predict the masks of the target object with the memory oracle."""

        self.main_memory.reset_memory()
        self.main_memory.initialize_references(self.model, detection_data)

        n_frames = detection_data.frames.shape[0]
        self.predicted_masks = torch.zeros((n_frames, 256, 256), dtype=torch.float64)
        self.object_scores = torch.zeros(n_frames, dtype=torch.float64)
        self.iou_scores = torch.zeros(n_frames, dtype=torch.float64)
        self.mask_iou_scores = torch.zeros(n_frames, dtype=torch.float64)   # pseudo-GT mask IoU = label
        self.iou_tokens = []

        # Optional per-frame embedding cache: the full-frame image encoding is target- and
        # tracker-independent, so a second tracker over the same frames can reuse it instead of
        # re-encoding. Populated on the first pass, read on later ones. Keyed by loop index.
        cache = getattr(self, "frame_cache", None)

        for current_idx, current_frame in enumerate(detection_data.frames):
            reuse = [e.to(self.device) for e in cache[current_idx]] \
                if cache is not None and current_idx in cache else None
            (chosen_mask, pointer, encoding, object_score, iou_score,
             iou_token_out, _, image_features) = self.model.select_best_mask(
                main_memory=self.main_memory,
                current_frame=current_frame,
                encoded_image_features_list=reuse)
            if cache is not None and current_idx not in cache:
                cache[current_idx] = [e.detach().cpu() for e in image_features]

            self.predicted_masks[current_idx] = chosen_mask
            self.iou_scores[current_idx] = iou_score
            self.object_scores[current_idx] = object_score
            self.iou_tokens.append(iou_token_out.reshape(-1).float().cpu().numpy())

            # Label the frame with pseudo-GT mask IoU, reusing the image encoding the tracker just
            # computed instead of re-encoding the frame (occluded / boxless frames stay 0).
            bboxes_norm = detection_data.bboxes_norm[current_idx]
            if detection_data.occlusions[current_idx] <= 0.5 and float(bboxes_norm[2]) != 0.0:
                self.mask_iou_scores[current_idx] = self._mask_iou(bboxes_norm, chosen_mask, image_features)

            # Commit gate. With use_mask_iou, reuse the pseudo-GT mask IoU just computed above (no second
            # encode); otherwise fall back to box IoU. Occluded/boxless frames left it at 0 -> never commit.
            quality = (float(self.mask_iou_scores[current_idx]) if self.use_mask_iou
                       else self._commit_quality(current_frame, bboxes_norm, chosen_mask))
            if quality > self.iou_threshold and detection_data.occlusions[current_idx] < 0.05:
                self.main_memory.update_memory(pointer, encoding)

        self.iou_tokens = np.stack(self.iou_tokens)
        return self.predicted_masks

    def _commit_quality(self, frame, gt_xywh_norm, chosen_mask):
        """Mask quality used to gate a memory commit: box IoU by default, else mask IoU vs the
        pseudo-GT mask (SAM box-prompted on the GT box)."""

        if not self.use_mask_iou:
            return float(compute_iou(gt_xywh_norm[None, :], chosen_mask[None, :].numpy())[0])

        pseudo_gt = self._pseudo_gt_mask(frame, gt_xywh_norm)
        if pseudo_gt is None:
            return 0.0
        predicted, target = chosen_mask.numpy() > 0, pseudo_gt > 0
        union = (predicted | target).sum()
        return float((predicted & target).sum() / union) if union > 0 else 0.0

    def _mask_iou(self, gt_xywh_norm, chosen_mask, image_features):
        """Pseudo-GT mask IoU, reusing the image encoding from tracking (no re-encode)."""

        box_tlbr = convert_bbox(np.asarray(gt_xywh_norm, dtype=np.float32))
        mask, _, _ = self.model.initialize_video_masking(image_features, box_tlbr)
        target = (mask.squeeze() > 0.0).cpu().numpy()
        predicted = chosen_mask.numpy() > 0
        union = (predicted | target).sum()
        return float((predicted & target).sum() / union) if union > 0 else 0.0

    def _pseudo_gt_mask(self, frame, gt_xywh_norm):
        """Box-prompt SAM on the whole frame with the GT box to get a pseudo-GT mask (or None)."""

        if float(gt_xywh_norm[2]) == 0.0:
            return None
        box_tlbr = convert_bbox(np.asarray(gt_xywh_norm, dtype=np.float32))
        encoded, _, _ = self.model.encode_image(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        mask, _, _ = self.model.initialize_video_masking(encoded, box_tlbr)
        return (mask.squeeze() > 0.0).to(torch.float64).cpu().numpy()

    