import cv2
import torch
import numpy as np

import torch.nn.functional as F
from dataclasses import dataclass
from src.typing.detection_data import DetectionData
from sam_3.build_sam import build_sam3_video_predictor


@dataclass
class Sam3:
    """STANDARD SAM 3, tracking one target from the anchor box -- claim_1's strongest published arm.

    SAM 3's tracker inherits SAM 2's architecture on a Perception Encoder trunk, and in single-object
    mode it inherits SAM 2's memory policy too: every frame is written, and a frame the object head calls
    occluded is written with a learned no-object embedding added rather than withheld. The gate Meta's
    code implements ships with the detector-coupled path, not this one, so `memory_selection` stays False
    here -- the arm is a stronger tracker with the same unconditional writing, which is exactly the
    comparison claim_1 is after.

    Frame 0 is the anchor, matching every other arm, so the records join on (video, person).
    """

    memory_selection: bool = False
    checkpoint_version: str = "sam3"
    device: str = "cuda"

    def __post_init__(self):
        """Build the tracker once; it is reused across every clip in the run."""

        self.predictor = build_sam3_video_predictor(memory_selection=self.memory_selection,
                                                    checkpoint_version=self.checkpoint_version,
                                                    device=self.device)

    def anchor_box(self, detection_data: DetectionData):
        """The anchor's normalised box as pixel xyxy, the prompt format the predictor expects."""

        video_capture = cv2.VideoCapture(detection_data.video_path)
        video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_capture.release()

        x_minimum, y_minimum, box_width, box_height = detection_data.bboxes_norm[0]
        left = video_width * x_minimum
        top = video_height * y_minimum
        right = video_width * (x_minimum + box_width)
        bottom = video_height * (y_minimum + box_height)

        return np.array([left, top, right, bottom], dtype=np.float32)

    @torch.inference_mode()
    def predict_masks(self, detection_data: DetectionData):
        """Roll SAM 3 over the clip from the anchor box, returning 256x256 masks like every other arm."""

        number_of_frames = len(detection_data.frame_indices)
        inference_state = self.predictor.init_state(video_path=detection_data.video_path,
                                                    num_frames=number_of_frames)
        self.predictor.add_new_points_or_box(inference_state=inference_state, frame_idx=0, obj_id=1,
                                             box=self.anchor_box(detection_data))

        predicted_masks = torch.zeros((number_of_frames, 256, 256), dtype=torch.float64)

        # This fork takes the propagation bounds positionally: start at the anchor, no frame limit, forward.
        for frame_index, _, frame_masks in self.predictor.propagate_in_video(inference_state, 0, None, False):
            resized_mask = F.interpolate(frame_masks.float().cpu(), size=(256, 256),
                                         mode="bilinear", align_corners=False)
            predicted_masks[frame_index] = (resized_mask.squeeze() > 0.0).to(torch.int)

        self.update_memory = self.memory_flags(inference_state, number_of_frames)
        return predicted_masks

    def memory_flags(self, inference_state, number_of_frames):
        """Per-frame memory decisions, read back from the predictor's own stored scores.

        With `memory_selection` off there is nothing to replay: SAM 3 writes every propagated frame, and
        that IS this arm's policy rather than a missing measurement. With it on, `eff_iou_score` records
        what Meta's `frame_filter` acts on -- a past frame conditions the current one only above
        `mf_threshold` -- which is the same decision SAMARA's commit gate makes at write time."""

        frame_outputs = inference_state["output_dict"]["non_cond_frame_outputs"]
        memory_flags = torch.zeros(number_of_frames, dtype=torch.int)
        memory_scores = torch.full((number_of_frames,), float("nan"), dtype=torch.float64)
        score_threshold = float(getattr(self.predictor, "mf_threshold", 0.0))

        for frame_index in range(number_of_frames):
            stored_output = frame_outputs.get(frame_index)
            if stored_output is None:
                memory_flags[frame_index] = 0 if self.memory_selection else 1
                continue

            frame_score = stored_output.get("eff_iou_score")
            if frame_score is None:
                memory_flags[frame_index] = 1
                continue

            memory_scores[frame_index] = float(frame_score)
            memory_flags[frame_index] = int(float(frame_score) > score_threshold)

        self.memory_scores = memory_scores
        return memory_flags
