import torch
import numpy as np

from dataclasses import dataclass
from src.typing.detection_data import DetectionData
from src.modules.memories.main_memory import MainMemory
from src.modules.models.samara_hiera_model import SamaraHieraModel
from src.utils.compute_iou import compute_iou

@dataclass
class MemoryOracle:
    """VOS with SAM 2 where the oracle updates the memory during the whole sequence."""

    iou_threshold: float | None = None
    model: SamaraHieraModel | None = None
    main_memory: MainMemory | None = None

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

        for current_idx, current_frame in enumerate(detection_data.frames):
            chosen_mask, pointer, encoding, object_score, iou_score, _, _ = self.model.select_best_mask(
                main_memory=self.main_memory,
                current_frame=current_frame)

            self.predicted_masks[current_idx] = chosen_mask
            self.iou_scores[current_idx] = iou_score
            self.object_scores[current_idx] = object_score

            bboxes_norm = detection_data.bboxes_norm[current_idx]
            chosen_true_iou = compute_iou(bboxes_norm[None, :], chosen_mask[None, :].numpy())

            if chosen_true_iou > self.iou_threshold and detection_data.occlusions[current_idx] < 0.05:
                self.main_memory.update_memory(pointer, encoding)

        return self.predicted_masks

    