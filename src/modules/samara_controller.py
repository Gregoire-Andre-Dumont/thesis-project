import numpy as np
import torch

from dataclasses import dataclass

from src.typing.detection_data import DetectionData
from src.modules.memories.main_memory import MainMemory
from src.modules.samara_hiera_model import SamaraHieraModel


@dataclass
class SamaraController:
    """SAM 2 VOS with the SAMARA calibrator making BOTH memory decisions.

    `SamaraFixed` scores only the mask SAM already picked, so the calibrator can veto a commit but never
    change what is tracked. Here it does both, on the same score:

        selection  -- all three proposals are scored against the anchor and the best one is kept, replacing
                      SAM's IoU-token argmax. Ranking needs an ordering, so this reads the REGRESSOR's
                      predicted IoU. This is the decision claim_4 measures offline as `agree`.
        gating     -- that proposal is committed only if its commit probability clears `commit_threshold`,
                      replacing the oracles' ground-truth gate. This is a binary decision, so it reads the
                      CLASSIFIER head when one is installed; a threshold on a regressed IoU is not a
                      calibrated decision boundary and tends to leave the gate permanently open.

    Nothing here reads ground truth, so unlike `MemoryOracle`/`MaskOracle` it is deployable. The anchor is
    frame 0's SAM initialization mask, pinned once; the calibrator never sees the FIFO, so a poisoned bank
    cannot drift the reference it scores against.

    Set `select` or `gate` False to isolate one half and attribute a coverage change to it."""

    commit_threshold: float = 0.5
    select: bool = True                # calibrator picks the proposal (else SAM's IoU token picks)
    gate: bool = True                  # calibrator gates the commit (else commit every frame, as the baseline does)

    model: SamaraHieraModel | None = None
    main_memory: MainMemory | None = None

    # {frame index: image-encoder output}, shared across arms on the same clip. The embedding depends only on
    # the frame, never on the memory bank, so two arms rolling the same clip can reuse it.
    frame_cache: dict | None = None

    def __post_init__(self):
        """Move SAM 2 to the GPU; the calibrator is installed separately as `model.controller`."""

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = self.model.to(device=self.device, dtype=self.dtype)

    @torch.inference_mode()
    def predict_masks(self, detection_data: DetectionData):
        """Roll SAM 2 over the sequence, letting the calibrator choose and gate every frame."""

        if self.model.controller is None:
            raise RuntimeError("no calibrator installed: set tracker.model.controller before predicting")

        self.main_memory.reset_memory()
        self.main_memory.initialize_references(self.model, detection_data, anchor_index=0)

        n_frames = detection_data.frames.shape[0]
        self.predicted_masks = torch.zeros((n_frames, 256, 256), dtype=torch.float64)
        self.calibrator_scores = torch.zeros((n_frames, 3), dtype=torch.float64)      # regressor: predicted IoU
        self.commit_probabilities = torch.zeros((n_frames, 3), dtype=torch.float64)   # classifier: p(commit)
        self.chosen_index = torch.zeros(n_frames, dtype=torch.int64)               # which proposal was kept, 0-2
        self.committed = torch.zeros(n_frames, dtype=torch.bool)                   # whether the frame was written
        self.committed_frames = []

        reference_foreground = reference_background = None
        cache = self.frame_cache
        for idx, current_frame in enumerate(detection_data.frames):
            reuse = [e.to(self.device) for e in cache[idx]] if cache is not None and idx in cache else None
            mask_preds, iou_scores, object_pointers, object_score, lowres_imgenc, image_features = self.model.propose_masks(
                main_memory=self.main_memory, current_frame=current_frame, encoded_image_features_list=reuse)

            if cache is not None and idx not in cache:
                cache[idx] = [e.detach().cpu() for e in image_features]

            # Frame 0 is the anchor. Its reference is the PROPOSAL this tracker keeps, not SAM's box-prompted
            # init mask -- the calibrator's dataset built its anchor the same way (`masks[0, chosen_index[0]]`
            # from the rollout), and the two masks disagree exactly where it matters: partial occlusion and
            # adjacent distractors. Selection on frame 0 therefore falls back to SAM's token, since there is
            # no reference to score against yet.
            if reference_foreground is None:
                anchor_proposal = int(torch.argmax(iou_scores[:, 1:], dim=-1))
                anchor_mask = mask_preds[0, anchor_proposal + 1]
                self.main_memory.initialize_calibrator_anchor(self.model, detection_data, anchor_mask, anchor_index=0)
                reference_foreground, reference_background = self.main_memory.gather_calibrator_references()

            scores, commit_probabilities = self.model.score_proposals(
                current_frame, mask_preds[0, 1:], reference_foreground, reference_background)
            self.calibrator_scores[idx] = torch.as_tensor(scores, dtype=torch.float64)
            self.commit_probabilities[idx] = torch.as_tensor(commit_probabilities, dtype=torch.float64)

            proposal = int(np.argmax(scores)) if self.select else int(torch.argmax(iou_scores[:, 1:], dim=-1))
            self.chosen_index[idx] = proposal

            chosen_mask, pointer, encoding = self.model.commit_candidate(
                mask_preds, proposal + 1, object_pointers, object_score, lowres_imgenc)
            self.predicted_masks[idx] = chosen_mask

            if not self.gate or float(commit_probabilities[proposal]) > self.commit_threshold:
                self.main_memory.update_memory(pointer, encoding)
                self.committed[idx] = True
                self.committed_frames.append(idx)

        return self.predicted_masks
