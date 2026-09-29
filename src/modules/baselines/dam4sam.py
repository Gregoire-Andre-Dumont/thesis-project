import cv2
import torch
import numpy as np

import torch.nn.functional as F
from dataclasses import dataclass

from src.typing.detection_data import DetectionData

# The DAM4SAM fork vendors its OWN top-level `sam2` package (SAM 2.1++), which is incompatible with the
# SAMURAI, SAMITE and SAM2Long forks of the same name. Only one can be on `PYTHONPATH` per interpreter,
# which is why `baselines.py` takes exactly one arm per process. Run this arm with
#   PYTHONPATH=/workspace/SENTRY/baseline/DAM4SAM
from sam_2.dam4sam.build_sam import build_dam4sam_video_predictor

torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


@dataclass
class Dam4Sam:
    """DAM4SAM / SAM 2.1++ (Videnovic et al., CVPR'25): SAM 2.1 with a DISTRACTOR-AWARE memory.

    Where SAMURAI and SAMITE filter what the model READS back, DAM4SAM adds a second bank -- the
    distractor-resolving memory -- holding frames chosen because a distractor was present, so the model has
    negative evidence to discriminate against later. Nothing is withheld from the ordinary memory, so its
    `update_memory` is all-True like stock SAM 2; the intervention is what is ADDED, not what is refused.
    That makes it the closest published counterpart to SAMARA's gate and the reason it belongs in the
    baseline table rather than the repo's own `sam_baseline` control.

    Training-free, so it stays in the zero-shot block of a comparison. Config names resolve inside the
    fork's own package (`initialize_config_module("sam2")`), hence the bare yaml filename."""

    checkpoint: str | None = None
    model_config: str | None = None

    def __post_init__(self):
        """Build the SAM 2.1++ video predictor from the fork's own config module."""

        model_config = self.model_config or "sam21pp_hiera_l.yaml"
        self.predictor = build_dam4sam_video_predictor(model_config, self.checkpoint, device="cuda")

    def predict_masks(self, detection_data: DetectionData):
        """Roll the tracker over the clip, anchored on frame 0 exactly as the other arms are."""

        frame_indices = detection_data.frame_indices
        inference_state = self.predictor.init_state(detection_data.video_path, frame_indices)

        capture = cv2.VideoCapture(detection_data.video_path)
        video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        x_min, y_min, width, height = detection_data.bboxes_norm[0]
        x_min, x_max = video_width * x_min, video_width * (x_min + width)
        y_min, y_max = video_height * y_min, video_height * (y_min + height)
        box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
        self.predictor.add_new_points_or_box(inference_state=inference_state, frame_idx=0, obj_id=1, box=box)

        n_frames = len(frame_indices)
        predicted_masks = torch.zeros((n_frames, 256, 256), dtype=torch.float64)

        # `return_all_masks=True` is not an option here: the fork's other branch yields an unbound
        # `video_res_masks` (it computes `video_res_masks_`, with a trailing underscore), so propagation
        # raises UnboundLocalError unless all masks are requested. Their own tracker.py takes this path too.
        # The extra return value carries every proposal and its IoU for their distractor logic; the chosen
        # mask is the first element, exactly what the default branch would have yielded.
        # The generator yields a 3-tuple on frames without `all_pred_masks` and a 4-tuple on frames
        # with them, so the chosen mask is taken positionally rather than unpacked to a fixed width.
        for output in self.predictor.propagate_in_video(inference_state, return_all_masks=True):
            frame_idx, masks = output[0], output[2]
            mask = F.interpolate(masks.cpu(), size=(256, 256), mode="bilinear", align_corners=False)
            predicted_masks[frame_idx] = (mask.squeeze() > 0.0).to(torch.int)

        self.update_memory = torch.ones(n_frames, dtype=torch.bool)
        self.memory_scores = self._scores(inference_state, n_frames)
        return predicted_masks

    def _scores(self, inference_state, n_frames):
        """Per-frame confidence, read back from the predictor's stored outputs as `(n_frames, 2)`.

        Column 0 is the IoU token and stays NaN: this fork keeps only `object_score_logits` per frame, with
        no per-frame mask-quality score to recover. Column 1 is that object logit -- the presence signal,
        and the stronger of SAM's two on absence anyway. `confidence_signals` reads these two columns."""

        outputs = inference_state["output_dict"]["non_cond_frame_outputs"]
        scores = torch.full((n_frames, 2), float("nan"), dtype=torch.float64)

        for index in range(n_frames):
            stored = outputs.get(index)
            if stored is None:
                continue
            scores[index, 1] = float(torch.as_tensor(stored["object_score_logits"]).reshape(-1)[0])

        return scores
