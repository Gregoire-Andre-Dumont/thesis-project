import torch

from dataclasses import dataclass
from src.typing.detection_data import DetectionData
from src.modules.memories.main_memory import MainMemory
from src.modules.samara_hiera_model import SamaraHieraModel
from src.utils.compute_iou import compute_iou

OBJECT_SCORE_FLOOR = 1e-3    # committed frames are GT-verified present: keep the encoder's object-score > 0


@dataclass
class MemoryOracle:
    """SAM 2 VOS that is an upper bound on the memory-COMMIT decision.

    It tracks exactly like the SAM baseline -- the mask is picked by SAM 2's own IoU token -- but it commits a
    frame to the memory bank only when the prediction is verified good against the ground truth: BOX IoU between
    the predicted mask's bounding box and the GT box exceeds `iou_threshold`, scored on visible frames only (so
    occluded frames never commit). Only the commit gate differs from the baseline; mask selection is identical.

    Subclasses may override `select_index` to change which proposal is kept (see MaskOracle)."""

    iou_threshold: float | None = None
    model: SamaraHieraModel | None = None
    main_memory: MainMemory | None = None

    def __post_init__(self):
        """Load the SAM 2 model onto the GPU in bfloat16."""

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = self.model.to(device=self.device, dtype=self.dtype)

    def select_index(self, mask_preds, iou_scores, bboxes_norm, visible):
        """Index of the proposal to keep. Memory oracle: SAM 2's own IoU-token argmax (same as the baseline)."""

        return int(1 + torch.argmax(iou_scores[:, 1:], dim=-1))

    def predict_masks(self, detection_data: DetectionData):
        """Roll SAM 2 over the sequence, committing its chosen mask to memory only on GT-verified visible frames."""

        self.main_memory.reset_memory()
        self.main_memory.initialize_references(self.model, detection_data, anchor_index=0)  # clip starts at the anchor

        n_frames = detection_data.frames.shape[0]
        self.predicted_masks = torch.zeros((n_frames, 256, 256), dtype=torch.float64)
        self.iou_scores = torch.zeros(n_frames, dtype=torch.float64)      # SAM 2's own predicted IoU (diagnostic)
        self.commit_iou = torch.zeros(n_frames, dtype=torch.float64)      # predicted-mask-box vs GT-box IoU (visible frames)

        self.committed_frames = []                                       
        cache = getattr(self, "frame_cache", None)

        for idx, current_frame in enumerate(detection_data.frames):
            reuse = [e.to(self.device) for e in cache[idx]] if cache is not None and idx in cache else None
            mask_preds, iou_scores, object_pointers, object_score, lowres_imgenc, image_features = self.model.propose_masks(
                main_memory=self.main_memory, current_frame=current_frame, encoded_image_features_list=reuse)

            if cache is not None and idx not in cache:
                cache[idx] = [e.detach().cpu() for e in image_features]

            bboxes_norm = detection_data.bboxes_norm[idx]
            visible = detection_data.occlusions[idx] <= 0.5
            best_idx = self.select_index(mask_preds, iou_scores, bboxes_norm, visible)

            # The oracle only commits GT-verified frames, where the object is known present
            encode_object_score = object_score.clamp(min=OBJECT_SCORE_FLOOR)

            chosen_mask, pointer, encoding = self.model.commit_candidate(
                mask_preds, best_idx, object_pointers, encode_object_score, lowres_imgenc)

            self.predicted_masks[idx] = chosen_mask
            self.iou_scores[idx] = float(iou_scores.max())

            # Commit gate: the chosen mask's BOX IoU vs the GT box, on visible frames only.
            if visible:
                self.commit_iou[idx] = float(compute_iou(bboxes_norm[None, :], chosen_mask[None, :].numpy())[0])
                if self.commit_iou[idx] > self.iou_threshold:
                    self.main_memory.update_memory(pointer, encoding)
                    self.committed_frames.append(idx)

        return self.predicted_masks
