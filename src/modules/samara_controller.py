import numpy as np
import torch

from dataclasses import dataclass

from src.typing.detection_data import DetectionData
from src.modules.memories.main_memory import MainMemory
from src.modules.samara_model import SamaraModel


@dataclass
class SamaraController:
    """SAM 2 VOS with the SAMARA calibrator deciding WHAT GOES INTO MEMORY.

    ONE decision: SAM picks the mask with its own IoU token, as it always does, and the calibrator decides
    whether that mask is written to the bank. The commit probability comes from `model.gate_controller`, a
    CLASSIFIER on 'worth committing' -- a binary decision, so a threshold on a regressed IoU is not a
    substitute; it is not a calibrated boundary and tends to leave the gate permanently open.

    SELECTION WAS REMOVED. Letting the calibrator also pick among the three proposals was measured
    repeatedly and never paid: it matched gate-only on small and heavily occluded targets and lost on large
    ones, where it made the gate more permissive on clips SAM was already tracking well. Scoring one
    proposal instead of three also cuts the Perception Encoder from ~40% of a rollout to ~19%.

    Nothing here reads ground truth, so unlike `MemoryOracle`/`MaskOracle` it is deployable. The anchor is
    frame 0's SAM initialization mask, pinned once; the calibrator never sees the FIFO, so a poisoned bank
    cannot drift the reference it scores against.

    Set `gate` False to commit every frame, as the baseline does, and attribute a coverage change to the
    gate alone."""

    commit_threshold: float = 0.5
    gate: bool = True       
    token_threshold: float | None = None
    model: SamaraModel | None = None
    main_memory: MainMemory | None = None
    frame_cache: dict | None = None

    def __post_init__(self):
        """Move SAM 2 to the GPU; the gate is installed separately as `model.gate_controller`."""

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = self.model.to(device=self.device, dtype=self.dtype)

    @torch.inference_mode()
    def predict_masks(self, detection_data: DetectionData):
        """Roll SAM 2 over the sequence, letting the calibrator gate every commit."""

        if self.gate and self.model.gate_controller is None:
            raise RuntimeError("gate=True needs a gate model: set tracker.model.gate_controller before predicting")

        self.main_memory.reset_memory()
        self.main_memory.initialize_references(self.model, detection_data, anchor_index=0)

        n_frames = detection_data.frames.shape[0]
        self.predicted_masks = torch.zeros((n_frames, 256, 256), dtype=torch.float64)

        self.calibrator_scores = torch.full((n_frames, 3), float("nan"), dtype=torch.float64)    # predicted IoU
        self.commit_probabilities = torch.full((n_frames, 3), float("nan"), dtype=torch.float64)  # p(commit)
        self.chosen_index = torch.zeros(n_frames, dtype=torch.int64)               # which proposal was kept, 0-2
        self.committed = torch.zeros(n_frames, dtype=torch.bool)                   # whether the frame was written
        self.committed_frames = []
        self.token_scores = torch.full((n_frames,), float("nan"), dtype=torch.float64)  # SAM's IoU token

        reference_foreground = reference_background = None
        cache = self.frame_cache
        for idx, current_frame in enumerate(detection_data.frames):
            reuse = [e.to(self.device) for e in cache[idx]] if cache is not None and idx in cache else None
            mask_preds, iou_scores, object_pointers, object_score, lowres_imgenc, image_features = self.model.propose_masks(
                main_memory=self.main_memory, current_frame=current_frame, encoded_image_features_list=reuse)

            if cache is not None and idx not in cache:
                cache[idx] = [e.detach().cpu() for e in image_features]

            if reference_foreground is None:
                anchor_proposal = int(torch.argmax(iou_scores[:, 1:], dim=-1))
                anchor_mask = mask_preds[0, anchor_proposal + 1]
                self.main_memory.initialize_calibrator_anchor(self.model, detection_data, anchor_mask, anchor_index=0)
                reference_foreground, reference_background = self.main_memory.gather_calibrator_references()

            token = iou_scores[0, 1:].to(torch.float32).cpu().numpy()
            presence = float(object_score.reshape(-1)[0])

            # SAM keeps the mask; only that one is cropped and scored, so the Perception Encoder runs
            proposal = int(torch.argmax(iou_scores[:, 1:], dim=-1))
            scalars = np.array([[token[proposal], presence]], dtype=np.float32)

            scores, probabilities = self.model.score_proposals(
                current_frame, mask_preds[0, 1 + proposal][None], reference_foreground, reference_background, scalars)
            self.calibrator_scores[idx, proposal] = float(scores[0])
            self.commit_probabilities[idx, proposal] = float(probabilities[0])

            self.chosen_index[idx] = proposal
            commit_probability = float(self.commit_probabilities[idx, proposal])

            token_score = float(iou_scores[0, 1 + proposal])
            self.token_scores[idx] = token_score
            token_holds = self.token_threshold is None or token_score > self.token_threshold

            chosen_mask, pointer, encoding = self.model.commit_candidate(
                mask_preds, proposal + 1, object_pointers, object_score, lowres_imgenc)
            self.predicted_masks[idx] = chosen_mask

            if not self.gate or (commit_probability > self.commit_threshold and token_holds):
                self.main_memory.update_memory(pointer, encoding)
                self.committed[idx] = True
                self.committed_frames.append(idx)

        return self.predicted_masks