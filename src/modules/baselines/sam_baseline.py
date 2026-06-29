import torch
import ipdb
import numpy as np

from dataclasses import dataclass
from src.typing.detection_data import DetectionData
from src.modules.memories.main_memory import MainMemory
from src.modules.samara_hiera_model import SamaraHieraModel


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

    def predict_masks(self, detection_data: DetectionData):
        """Predict the masks of the target object with the baseline SAM 2."""

        # Reset and initialize the memory bank with the new target
        self.main_memory.reset_memory()
        self.main_memory.initialize_references(self.model, detection_data, anchor_index=0)

        n_frames = detection_data.frames.shape[0]
        self.predicted_masks = torch.zeros((n_frames, 256, 256), dtype=torch.float64)

        # Storage for the predicted IoU and occlusion scores
        self.object_scores = torch.zeros(n_frames, dtype=torch.float64)
        self.iou_scores = torch.zeros(n_frames, dtype=torch.float64)
        self.update_memory = torch.zeros(n_frames, dtype=torch.int)
        self.object_pointers = torch.zeros((n_frames, 256), dtype=torch.float64)

        for idx, current_frame in enumerate(detection_data.frames):
            chosen_mask, pointer, encoding, object_scores, iou_scores, _, _ = self.model.select_best_mask(
                main_memory = self.main_memory,
                current_frame = current_frame)

            # Update the memory bank with the embeddings and store the mask
            if object_scores > 0.5 and iou_scores > self.iou_threshold:
                self.main_memory.update_memory(pointer, encoding)
                self.update_memory[idx] = 1
            self.predicted_masks[idx] = chosen_mask

            self.object_scores[idx] = object_scores
            self.iou_scores[idx] = iou_scores
            self.object_pointers[idx] = pointer.squeeze().to(torch.float32).cpu()

        return self.predicted_masks
    