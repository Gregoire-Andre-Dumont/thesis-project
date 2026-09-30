import numpy as np
import torch

from dataclasses import dataclass

from src.typing.detection_data import DetectionData
from src.modules.memories.main_memory_v3 import MainMemoryV3
from src.modules.samara3_model import SamaraModel3


@dataclass
class Samara3Controller:
    """SAM 3 VOS with the SAMARA calibrator deciding WHAT GOES INTO MEMORY.

    A line-for-line counterpart of `SamaraController`: the same single decision, the same gate model, the
    same pinned anchor, the same thresholds. Only the tracker underneath is SAM 3, and every difference is
    absorbed by `SamaraModel3` and `MainMemoryV3` so the loop below reads the same as the SAM 2 one.

    Keeping them as separate files rather than one parameterised controller is deliberate. The two SAM
    packages cannot be imported into the same interpreter, so a shared controller would have to defer its
    imports and would fail at rollout time rather than at construction; and a SAMARA-2 vs SAMARA-3 result
    is only worth reporting if each arm's code path is inspectable on its own.

    Nothing here reads ground truth beyond the frame-0 anchor box every arm is given, so it is deployable.
    Set `gate` False to commit every frame and recover the plain SAM 3 baseline."""

    commit_threshold: float = 0.5
    gate: bool = True
    token_threshold: float | None = None
    model: SamaraModel3 | None = None
    main_memory: MainMemoryV3 | None = None
    frame_cache: dict | None = None

    def __post_init__(self):
        """Move SAM 3 to the GPU; the gate is installed separately as `model.gate_controller`."""

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = self.model.to(device=self.device, dtype=self.dtype)

    @torch.inference_mode()
    def predict_masks(self, detection_data: DetectionData):
        """Roll SAM 3 over the sequence, letting the calibrator gate every commit."""

        if self.gate and self.model.gate_controller is None:
            raise RuntimeError("gate=True needs a gate model: set tracker.model.gate_controller before predicting")

        self.main_memory.reset_memory()
        self.main_memory.initialize_references(self.model, detection_data, anchor_index=0)

        n_frames = detection_data.frames.shape[0]
        self.predicted_masks = torch.zeros((n_frames, 256, 256), dtype=torch.float64)
        self.calibrator_scores = torch.full((n_frames, 3), float("nan"), dtype=torch.float64)      # predicted IoU
        self.commit_probabilities = torch.full((n_frames, 3), float("nan"), dtype=torch.float64)   # p(commit)
        self.chosen_index = torch.zeros(n_frames, dtype=torch.int64)
        self.committed = torch.zeros(n_frames, dtype=torch.bool)
        self.committed_frames = []
        self.token_scores = torch.full((n_frames,), float("nan"), dtype=torch.float64)

        self.object_scores = torch.full((n_frames,), float("nan"), dtype=torch.float64)
        self.mask_area = torch.zeros(n_frames, dtype=torch.int64)

        reference_foreground = reference_background = None
        for idx, current_frame in enumerate(detection_data.frames):
            mask_preds, iou_scores, object_pointers, object_score, encoded_image, _ = self.model.propose_masks(
                current_frame=current_frame, main_memory=self.main_memory)

            if reference_foreground is None:
                anchor_proposal = int(torch.argmax(iou_scores[:, 1:], dim=-1))
                anchor_mask = mask_preds[0, anchor_proposal + 1]
                self.main_memory.initialize_calibrator_anchor(self.model, detection_data, anchor_mask, anchor_index=0)
                reference_foreground, reference_background = self.main_memory.gather_calibrator_references()

            token = iou_scores[0, 1:].to(torch.float32).cpu().numpy()
            object_confidence = float(object_score.reshape(-1)[0])
            self.object_scores[idx] = object_confidence

            # SAM keeps the mask; only that one is cropped and scored, so the Perception Encoder runs once.
            proposal = int(torch.argmax(iou_scores[:, 1:], dim=-1))
            scalars = np.array([[token[proposal]]], dtype=np.float32)
            scores, probabilities, _ = self.model.score_proposals(
                current_frame, mask_preds[0, 1 + proposal][None], reference_foreground, reference_background, scalars)
            self.calibrator_scores[idx, proposal] = float(scores[0])
            self.commit_probabilities[idx, proposal] = float(probabilities[0])

            self.chosen_index[idx] = proposal
            commit_probability = float(self.commit_probabilities[idx, proposal])

            token_score = float(iou_scores[0, 1 + proposal])
            self.token_scores[idx] = token_score
            token_holds = self.token_threshold is None or token_score > self.token_threshold

            chosen_mask, pointer, encoding = self.model.commit_candidate(
                mask_preds, proposal + 1, object_pointers, object_score, encoded_image)

            self.predicted_masks[idx] = self._resize(chosen_mask)
            self.mask_area[idx] = int(self.predicted_masks[idx].sum())

            if not self.gate or (commit_probability > self.commit_threshold and token_holds):
                self.main_memory.update_memory(pointer, encoding)
                self.committed[idx] = True
                self.committed_frames.append(idx)

        return self.predicted_masks

    @staticmethod
    def _resize(mask_logits):
        """SAM 3's masks are 1/4 of its 1008px input, i.e. 252x252, where SAM 2's are 256x256.

        Every downstream consumer -- `frame_record`, the IoU against the 256x256 ground truth, the stored
        `mask_area` -- assumes 256, so the mask is resized here rather than leaving a silent 252 to be
        broadcast or truncated. Binarised before resizing and thresholded at 0.5 after, so the result is a
        clean mask rather than resampled logits."""

        mask = (mask_logits > 0.0).to(torch.float32)[None, None]
        if mask.shape[-2:] == (256, 256):
            return mask.squeeze().to(torch.float64)

        resized = torch.nn.functional.interpolate(mask, size=(256, 256), mode="bilinear", align_corners=False)
        return (resized.squeeze() > 0.5).to(torch.float64)
