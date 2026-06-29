import torch
import numpy as np

from dataclasses import dataclass
from src.typing.detection_data import DetectionData
from src.modules.memories.main_memory import MainMemory
from src.modules.samara_hiera_model import SamaraHieraModel
from src.utils.compute_iou import compute_iou


@dataclass
class CombinedOracle:
    """VOS with SAM 2 where, at every frame, the oracle feeds the GT bbox as a prompt
    AND picks the candidate mask whose true IoU vs that GT bbox is highest. Memory is
    committed only when the chosen mask's true IoU clears `iou_threshold` AND the frame
    is essentially unoccluded (`occlusions < 0.05`).

    Differs from `MemoryOracle`, which uses SAM 2's argmax of internal predicted-IoU
    (no GT prompt) and only filters commits by true IoU. CombinedOracle conditions
    EVERY frame's prediction on GT, producing higher-quality predicted masks for
    dataset creation."""

    iou_threshold: float | None = None
    model: SamaraHieraModel | None = None
    main_memory: MainMemory | None = None

    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = self.model.to(device=self.device, dtype=self.dtype)

    def predict_masks(self, detection_data: DetectionData):
        self.main_memory.reset_memory()
        self.main_memory.initialize_references(self.model, detection_data)

        n_frames = detection_data.frames.shape[0]
        self.predicted_masks = torch.zeros((n_frames, 256, 256), dtype=torch.float64)
        self.object_scores = torch.zeros(n_frames, dtype=torch.float64)
        self.iou_scores = torch.zeros(n_frames, dtype=torch.float64)

        for current_idx, current_frame in enumerate(detection_data.frames):
            target_box = detection_data.bboxes_norm[current_idx]
            chosen_mask, pointer, encoding, object_score, iou_score, _ = self.model.select_best_mask_oracle(
                main_memory=self.main_memory,
                current_frame=current_frame,
                bboxes_norm=target_box)

            self.predicted_masks[current_idx] = chosen_mask
            self.iou_scores[current_idx] = iou_score
            self.object_scores[current_idx] = object_score

            chosen_true_iou = compute_iou(target_box[None, :], chosen_mask[None, :].numpy())

            if chosen_true_iou > self.iou_threshold and detection_data.occlusions[current_idx] < 0.05:
                self.main_memory.update_memory(pointer, encoding)

        return self.predicted_masks
