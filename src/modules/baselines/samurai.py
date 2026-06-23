import cv2
import torch
import ipdb
import numpy as np

import torch.nn.functional as F
from dataclasses import dataclass
from src.typing.detection_data import DetectionData
from muggled_sam.samurai.build_sam import build_samurai_video_predictor

torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

@dataclass
class Samurai:
    """Baseline module for video object segmentation with SAM 2."""

    checkpoint: str | None = None
    model_config: str | None = None

    def __post_init__(self):
        """Load and initialize the SAM 2 model with quantization."""

        model_config = "../conf/SAM2/samurai_hiera_large.yaml"
        self.predictor = build_samurai_video_predictor(model_config, self.checkpoint, device="cuda")

    def predict_masks(self, detection_data: DetectionData):
        """Predict the masks of the target object with the baseline SAM 2."""

        video_path = detection_data.video_path
        # Predictor's internal frame 0 maps directly to detection_data frame 0 (the chosen
        # anchor), matching SAMARA / sam_baseline's anchor convention.
        frame_indices = detection_data.frame_indices
        inference_state = self.predictor.init_state(video_path, frame_indices)

        cap = cv2.VideoCapture(detection_data.video_path)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Bounding box of the anchor (detection_data frame 0 = predictor frame 0).
        x_min, y_min, width, height = detection_data.bboxes_norm[0]

        x_min, x_max = video_width * x_min, video_width * (x_min + width)
        y_min, y_max = video_height * y_min, video_height * (y_min + height)

        bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
        self.predictor.add_new_points_or_box(inference_state=inference_state, frame_idx=0, obj_id=1, box=bbox)

        n_frames = len(detection_data.frame_indices)
        predicted_masks = torch.zeros((n_frames, 256, 256), dtype=torch.float64)

        self.object_scores = torch.zeros(n_frames, dtype=torch.float64)
        self.iou_scores = torch.zeros((n_frames, 4), dtype=torch.float64)

        for frame_idx, object_ids, masks in self.predictor.propagate_in_video(inference_state):
            mask = F.interpolate(masks.cpu(), size=(256, 256), mode='bilinear', align_corners=False)
            predicted_masks[frame_idx] = (mask.squeeze() > 0.0).to(torch.int)

        return predicted_masks
