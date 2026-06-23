"""SAM2Long baseline: SAM 2.1 with a tree-search-based memory selection over `num_pathway`
candidate memory states (Mark12Ding/SAM2Long). Wraps the locally-subclassed
`SAM2LongVideoPredictor`. Predictor's local frame 0 maps directly to
`detection_data.frames[0]` (the chosen anchor), matching SAMARA / sam_baseline."""

import cv2
import torch
import numpy as np

import torch.nn.functional as F
from dataclasses import dataclass
from src.typing.detection_data import DetectionData
from muggled_sam.sam2long.build_sam import build_sam2long_video_predictor


torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


@dataclass
class SAM2Long:
    checkpoint: str | None = None
    model_config: str = "SAM2/sam2long_hiera_large.yaml"
    num_pathway: int = 3
    iou_thre: float = 0.1
    uncertainty: int = 2

    def __post_init__(self):
        self.predictor = build_sam2long_video_predictor(
            self.model_config,
            self.checkpoint,
            device="cuda",
            num_pathway=self.num_pathway,
            iou_thre=self.iou_thre,
            uncertainty=self.uncertainty,
        )

    def predict_masks(self, detection_data: DetectionData):
        video_path = detection_data.video_path
        # Predictor's local frame 0 = detection_data frame 0 (the chosen anchor).
        frame_indices = detection_data.frame_indices
        inference_state = self.predictor.init_state(video_path, frame_indices)

        cap = cv2.VideoCapture(detection_data.video_path)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        x_min, y_min, width, height = detection_data.bboxes_norm[0]
        x_min_pixel, x_max_pixel = video_width * x_min, video_width * (x_min + width)
        y_min_pixel, y_max_pixel = video_height * y_min, video_height * (y_min + height)
        bbox = np.array([x_min_pixel, y_min_pixel, x_max_pixel, y_max_pixel], dtype=np.float32)
        self.predictor.add_new_points_or_box(inference_state=inference_state,
                                             frame_idx=0, obj_id=1, box=bbox)

        n_frames = len(detection_data.frame_indices)
        predicted_masks = torch.zeros((n_frames, 256, 256), dtype=torch.float64)

        # SAM2Long's `propagate_in_video` is NOT a generator (unlike vanilla SAM 2 / SAMURAI):
        # it returns `(obj_ids, mask_list)` after running the whole video. `mask_list[i]` is
        # the video-resolution mask tensor for predictor-local frame `i` = detection_data frame `i`.
        _obj_ids, mask_list = self.predictor.propagate_in_video(inference_state)
        for predictor_frame_idx, masks in enumerate(mask_list):
            mask = F.interpolate(masks.cpu(), size=(256, 256), mode="bilinear", align_corners=False)
            predicted_masks[predictor_frame_idx] = (mask.squeeze() > 0.0).to(torch.int)

        return predicted_masks
