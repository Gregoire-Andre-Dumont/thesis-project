import torch
import numpy as np

from dataclasses import dataclass
from src.typing.detection_data import DetectionData
from src.modules.memories.main_memory import MainMemory
from src.modules.samara_hiera_model import SamaraHieraModel
from src.utils.load_bboxes import convert_bbox


@dataclass
class SAMBaseline:
    """Baseline module for video object segmentation with SAM 2."""

    model: SamaraHieraModel | None = None
    iou_threshold: float | None = None
    main_memory: MainMemory | None = None

    def __post_init__(self):
        """Load and initialize the SAM 2 model with quantization."""

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = self.model.to(device=self.device, dtype=self.dtype)

    def should_commit(self, object_scores, iou_scores, chosen_mask, frame):
        """Whether to write this frame into the memory bank. Baseline gate = SAM's own confidence.
        Subclasses may add an identity check (e.g. a Perception-Encoder gate)."""
        
        return object_scores > 0.5 and iou_scores > self.iou_threshold

    def predict_masks(self, detection_data: DetectionData):
        """Predict the masks of the target object with the baseline SAM 2."""

        # Reset and initialize the memory bank with the new target (the anchor, same as the oracle --
        # after warmup slicing the anchor is at index 1, so use initialize_references' default).
        self.main_memory.reset_memory()
        self.main_memory.initialize_references(self.model, detection_data)

        n_frames = detection_data.frames.shape[0]
        self.predicted_masks = torch.zeros((n_frames, 256, 256), dtype=torch.float64)

        # Storage for the predicted IoU and occlusion scores
        self.object_scores = torch.zeros(n_frames, dtype=torch.float64)
        self.iou_scores = torch.zeros(n_frames, dtype=torch.float64)
        self.mask_iou_scores = torch.zeros(n_frames, dtype=torch.float64)   # pseudo-GT mask IoU = label
        self.update_memory = torch.zeros(n_frames, dtype=torch.int)
        self.object_pointers = torch.zeros((n_frames, 256), dtype=torch.float64)

        # Optional per-frame embedding cache shared with another tracker over the same frames: the
        # full-frame image encoding is target-independent, so it is encoded once and reused.
        cache = getattr(self, "frame_cache", None)

        for idx, current_frame in enumerate(detection_data.frames):
            reuse = [e.to(self.device) for e in cache[idx]] \
                if cache is not None and idx in cache else None
            (chosen_mask, pointer, encoding, object_scores, iou_scores,
             _, _, image_features) = self.model.select_best_mask(
                main_memory = self.main_memory,
                current_frame = current_frame,
                encoded_image_features_list = reuse)
            if cache is not None and idx not in cache:
                cache[idx] = [e.detach().cpu() for e in image_features]

            # Update the memory bank with the embeddings and store the mask
            if self.should_commit(object_scores, iou_scores, chosen_mask, current_frame):
                self.main_memory.update_memory(pointer, encoding)
                self.update_memory[idx] = 1
            self.predicted_masks[idx] = chosen_mask

            self.object_scores[idx] = object_scores
            self.iou_scores[idx] = iou_scores
            self.object_pointers[idx] = pointer.squeeze().to(torch.float32).cpu()

            # Label the frame with pseudo-GT mask IoU, reusing the encoding just computed for tracking.
            gt = detection_data.bboxes_norm[idx]
            if detection_data.occlusions[idx] <= 0.5 and float(gt[2]) != 0.0:
                mask, _, _ = self.model.initialize_video_masking(
                    image_features, convert_bbox(np.asarray(gt, dtype=np.float32)))
                target = (mask.squeeze() > 0.0).cpu().numpy()
                predicted = chosen_mask.numpy() > 0
                union = (predicted | target).sum()
                self.mask_iou_scores[idx] = float((predicted & target).sum() / union) if union > 0 else 0.0

        return self.predicted_masks
    