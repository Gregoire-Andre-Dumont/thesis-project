import numpy as np
import torch

from dataclasses import dataclass

from src.typing.detection_data import DetectionData
from src.modules.memories.main_memory import MainMemory
from src.modules.samara_model import SamaraModel


@dataclass
class SamaraController:
    """SAM 2 VOS with the SAMARA calibrator making BOTH memory decisions.

    Two decisions, from one pass over the same anchor-similarity features:

        selection  -- all three proposals are scored against the anchor and the best one is kept, replacing
                      SAM's IoU-token argmax. Ranking needs an ordering, so this reads `model.controller`,
                      a REGRESSOR on true IoU. This is the decision claim_4 measures offline as `agree`.
        gating     -- that proposal is committed only if its commit probability clears `commit_threshold`,
                      replacing the oracles' ground-truth gate. This is a binary decision, so it reads
                      `model.gate_controller`, a CLASSIFIER: a threshold on a regressed IoU is not a
                      calibrated decision boundary and tends to leave the gate permanently open.

    The two are separate models with separate objectives, so each stops training on its own schedule; only
    the one a flag turns on has to be installed.

    Nothing here reads ground truth, so unlike `MemoryOracle`/`MaskOracle` it is deployable. The anchor is
    frame 0's SAM initialization mask, pinned once; the calibrator never sees the FIFO, so a poisoned bank
    cannot drift the reference it scores against.

    Set `select` or `gate` False to isolate one half and attribute a coverage change to it."""

    commit_threshold: float = 0.5
    select: bool = True                # calibrator picks the proposal (else SAM's IoU token picks)
    gate: bool = True                  # calibrator gates the commit (else commit every frame, as the baseline does)

    # A SECOND condition on the commit, ANDed with the calibrator's: SAM's own IoU token for the kept
    # proposal must also clear this. The two scores answer different questions -- the calibrator asks "is
    # this the right person", the token asks "is this mask any good" -- so a frame that fails either is one
    # neither model vouches for. Left None, only the calibrator gates, as before.
    token_threshold: float | None = None

    model: SamaraModel | None = None
    main_memory: MainMemory | None = None
    frame_cache: dict | None = None

    def __post_init__(self):
        """Move SAM 2 to the GPU; the calibrator is installed separately as `model.controller`."""

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = self.model.to(device=self.device, dtype=self.dtype)

    @torch.inference_mode()
    def predict_masks(self, detection_data: DetectionData):
        """Roll SAM 2 over the sequence, letting the calibrator choose and gate every frame."""

        if self.select and self.model.controller is None:
            raise RuntimeError("select=True needs a selector: set tracker.model.controller before predicting")
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

            # SAM's own opinion of each proposal, for a controller trained to read it alongside the
            # anchor similarity (`dataset.scalars`). Harmless when it was not: the extra channels are
            # sliced off before the convolutions and a model with `n_scalars: 0` never looks at them.
            token = iou_scores[0, 1:].to(torch.float32).cpu().numpy()
            presence = float(object_score.reshape(-1)[0])

            if self.select:
                scalars = np.stack([token, np.full(len(token), presence, dtype=np.float32)], axis=-1)
                scores, probabilities = self.model.score_proposals(
                    current_frame, mask_preds[0, 1:], reference_foreground, reference_background, scalars)
                
                proposal = int(np.argmax(scores))
                self.calibrator_scores[idx] = torch.as_tensor(scores, dtype=torch.float64)
                self.commit_probabilities[idx] = torch.as_tensor(probabilities, dtype=torch.float64)
            else:
                proposal = int(torch.argmax(iou_scores[:, 1:], dim=-1))
                scalars = np.array([[token[proposal], presence]], dtype=np.float32)
                scores, probabilities = self.model.score_proposals(
                    current_frame, mask_preds[0, 1 + proposal][None], reference_foreground,
                    reference_background, scalars)
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
