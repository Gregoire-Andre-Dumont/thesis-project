import cv2
import torch
import numpy as np

import torch.nn.functional as F
from dataclasses import dataclass
from src.typing.detection_data import DetectionData
from sam_2.samurai.build_sam import build_samurai_video_predictor

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

        model_config = self.model_config or "../conf/SAM2/samurai_hiera_large.yaml"
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

        self.update_memory = self._memory_flags(inference_state, n_frames)
        return predicted_masks

    def _memory_flags(self, inference_state, n_frames):
        """Per-frame memory-commit decisions, read back from the predictor's own stored scores.

        SAMURAI/SAMITE do not refuse to ENCODE a frame's memory; they refuse to READ it back. Memory
        conditioning walks the stored frames and keeps one only if its mask affinity, object score and
        Kalman motion score all clear the model's own thresholds (`sam2_base.py`, the `samurai_mode`
        branch). A frame that never passes is a frame whose memory is never attended to, which is the same
        decision SAMARA's commit gate makes at write time -- so the model's own predicate is replayed here
        rather than a threshold invented for this repo.

        The one asymmetry is the immediately previous frame, which the model appends unconditionally
        whatever its scores; it is marked committed to match."""

        outputs = inference_state["output_dict"]["non_cond_frame_outputs"]
        flags = torch.zeros(n_frames, dtype=torch.bool)
        scores = torch.full((n_frames, 3), float("nan"), dtype=torch.float64)

        model = self.predictor
        for index in range(n_frames):
            stored = outputs.get(index)
            if stored is None:
                continue

            iou_score = float(stored["best_iou_score"])
            object_score = float(stored["object_score_logits"])
            kalman_score = float(stored["kf_score"]) if stored.get("kf_score") is not None else float("nan")
            scores[index] = torch.tensor([iou_score, object_score, kalman_score], dtype=torch.float64)

            # bool(): the isnan term makes the expression a numpy.bool_, which a torch.BoolTensor
            # refuses to take.
            flags[index] = bool(iou_score > model.memory_bank_iou_threshold
                                and object_score > model.memory_bank_obj_score_threshold
                                and (np.isnan(kalman_score)
                                     or kalman_score > model.memory_bank_kf_score_threshold))

        # The frame immediately before the last one is always appended, pass or fail.
        if n_frames > 1:
            flags[n_frames - 2] = True

        self.memory_scores = scores
        return flags

