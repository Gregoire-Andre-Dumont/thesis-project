import numpy as np
import torch

from collections import deque
from dataclasses import dataclass
from src.typing.detection_data import DetectionData
from src.modules.memories.main_memory import MainMemory
from src.modules.samara_hiera_model import SamaraHieraModel
from src.utils.compute_iou import compute_iou
from src.utils.load_bboxes import convert_bbox


@dataclass
class MemoryOracle:
    """SAM 2 VOS that commits SAM 2's own predicted mask to the memory bank whenever the prediction is verified
    good against the ground truth.

    The commit gate is box IoU between the predicted mask's bounding box and the GT box (the same metric the
    experiment grades coverage on); it is only scored on visible frames, so occluded frames never commit.

    Mask selection: by default the mask is picked by SAM 2's own IoU token (memory oracle). With
    `oracle_mask_selection` the proposed mask with the highest true box IoU vs GT is picked instead (mask oracle).
    With `memory_perturbation`, the candidate set is expanded by a one-step branch on the memory: the previous
    frame's 4 proposed masks each define a candidate bank (the then-current bank with that mask committed); the
    current frame is decoded against all 4 of those banks (4 masks each), and the oracle picks the best true-IoU
    mask over the pooled proposals.

    Corruption (optional): if `corruption_p` > 0 the caller sets a per-frame `corruption_boxes` list on the
    instance (None before the first occlusion). From then on, with probability `corruption_p`, a nearby distractor
    (box-prompted from `corruption_boxes[i]`) is committed to the bank instead of the target -- and, once
    triggered, on the next frame too -- poisoning the bank. `committed_frames` / `committed_masks` record what was
    written."""

    iou_threshold: float | None = None
    model: SamaraHieraModel | None = None
    main_memory: MainMemory | None = None
    corruption_p: float = -1.0
    oracle_mask_selection: bool = False   # False: pick mask by SAM's IoU token (memory oracle).
                                          # True: pick the proposed mask with the best true IoU vs GT (mask oracle).
    memory_perturbation: bool = False     # mask oracle only: also decode against the 4 banks formed by committing
                                          # each of the previous frame's 4 proposals, and pool all proposals.

    def __post_init__(self):
        """Load the SAM 2 model onto the GPU in bfloat16."""

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = self.model.to(device=self.device, dtype=self.dtype)

    def _fork_banks(self, mask_preds, object_pointers, object_score, lowres_imgenc):
        """Return one candidate memory bank per proposed mask: the current bank with that proposal committed.
        Each bank is a (previous_encodings, previous_pointers) pair; the fixed anchor reference is shared."""

        banks = []
        for i in range(mask_preds.shape[1]):
            _mask, pointer_i, encoding_i = self.model.commit_candidate(
                mask_preds, i, object_pointers, object_score, lowres_imgenc)
            encodings = deque(self.main_memory.previous_encodings, maxlen=self.main_memory.max_memory_history)
            pointers = deque(self.main_memory.previous_pointers, maxlen=self.main_memory.max_pointer_history)
            encodings.appendleft(encoding_i)
            pointers.appendleft(pointer_i)
            banks.append((encodings, pointers))
        return banks

    def predict_masks(self, detection_data: DetectionData):
        """Roll SAM 2 over the sequence, committing its predicted mask (or a distractor when corrupted) to memory."""

        self.main_memory.reset_memory()
        self.main_memory.initialize_references(self.model, detection_data)

        n_frames = detection_data.frames.shape[0]
        self.predicted_masks = torch.zeros((n_frames, 256, 256), dtype=torch.float64)
        self.iou_scores = torch.zeros(n_frames, dtype=torch.float64)         # SAM 2's own predicted IoU per frame
        self.commit_iou = torch.zeros(n_frames, dtype=torch.float64)         # predicted-mask-box vs GT-box IoU on visible frames
        self.committed_frames = []                                           # frame indices written to the bank
        self.committed_masks = []                                            # mask committed at each (prediction, or distractor when corrupted)

        rng = np.random.default_rng(0)
        corrupt_burst = 0
        corruption_boxes = getattr(self, "corruption_boxes", None)
        previous_banks = []                                                  # the previous frame's 4 candidate banks
        # Optional per-frame image-embedding cache (target-independent), shared across rollouts over the same
        # frames -- e.g. a threshold sweep -- so the image encoder runs once per frame instead of once per rollout.
        cache = getattr(self, "frame_cache", None)

        for current_idx, current_frame in enumerate(detection_data.frames):
            reuse = [e.to(self.device) for e in cache[current_idx]] \
                if cache is not None and current_idx in cache else None
            (mask_preds, iou_scores, object_pointers, object_score,
             lowres_imgenc, image_features) = self.model.propose_masks(
                main_memory=self.main_memory, current_frame=current_frame, encoded_image_features_list=reuse)
            if cache is not None and current_idx not in cache:
                cache[current_idx] = [e.detach().cpu() for e in image_features]

            bboxes_norm = detection_data.bboxes_norm[current_idx]
            visible = detection_data.occlusions[current_idx] <= 0.5 and float(bboxes_norm[2]) != 0.0

            # Mask selection. Default: SAM's own IoU token. With oracle_mask_selection on a visible frame: the
            # proposed mask with the highest true box IoU vs GT, pooled (with memory_perturbation) over proposals
            # decoded against the previous frame's 4 candidate banks as well.
            selected_masks, selected_pointers, selected_score = mask_preds, object_pointers, object_score
            best_idx = int(1 + torch.argmax(iou_scores[:, 1:], dim=-1))

            if self.oracle_mask_selection and visible:
                sources = [(mask_preds, object_pointers, object_score)]
                if self.memory_perturbation:
                    for bank_encodings, bank_pointers in previous_banks:
                        saved = (self.main_memory.previous_encodings, self.main_memory.previous_pointers)
                        self.main_memory.previous_encodings, self.main_memory.previous_pointers = bank_encodings, bank_pointers
                        perturbed = self.model.propose_masks(main_memory=self.main_memory, current_frame=current_frame,
                                                             encoded_image_features_list=image_features)
                        sources.append((perturbed[0], perturbed[2], perturbed[3]))
                        self.main_memory.previous_encodings, self.main_memory.previous_pointers = saved

                best_iou = -1.0
                for source_masks, source_pointers, source_score in sources:
                    candidates = (source_masks[0, 1:] > 0.0).cpu().numpy()   # skip index 0 (whole-object mask)
                    ious = compute_iou(np.repeat(bboxes_norm[None, :], candidates.shape[0], axis=0), candidates)
                    j = int(np.argmax(ious))
                    if float(ious[j]) > best_iou:
                        best_iou = float(ious[j])
                        selected_masks, selected_pointers, selected_score, best_idx = source_masks, source_pointers, source_score, 1 + j

            chosen_mask, pointer, encoding = self.model.commit_candidate(
                selected_masks, best_idx, selected_pointers, selected_score, lowres_imgenc)

            # Fork the current bank on this frame's 4 proposals, for the next frame's perturbation (uses the bank
            # as it stands before this frame's commit).
            if self.memory_perturbation:
                previous_banks = self._fork_banks(mask_preds, object_pointers, object_score, lowres_imgenc)

            self.predicted_masks[current_idx] = chosen_mask
            self.iou_scores[current_idx] = float(iou_scores.max())   # SAM 2's own confidence (diagnostic)

            # Box IoU of the chosen mask's bbox against the GT box (occluded/boxless frames stay 0).
            if visible:
                self.commit_iou[current_idx] = float(compute_iou(bboxes_norm[None, :], chosen_mask[None, :].numpy())[0])

            distractor_box = corruption_boxes[current_idx] if corruption_boxes is not None else None
            quality = float(self.commit_iou[current_idx])

            if (self.corruption_p > 0.0 and distractor_box is not None and (corrupt_burst > 0 or rng.random() < self.corruption_p)):
                corrupt_burst = corrupt_burst - 1 if corrupt_burst > 0 else 1
                distractor_logits, encoding_c, pointer_c = self.model.initialize_video_masking(
                    image_features, convert_bbox(np.asarray(distractor_box, dtype=np.float32)))

                self.main_memory.update_memory(pointer_c, encoding_c)
                self.committed_frames.append(current_idx)
                self.committed_masks.append((distractor_logits.squeeze() > 0.0).to(torch.float64).cpu().numpy())

            elif quality > self.iou_threshold and visible:
                corrupt_burst = 0
                self.main_memory.update_memory(pointer, encoding)
                self.committed_frames.append(current_idx)
                self.committed_masks.append(self.predicted_masks[current_idx].numpy().copy())

        return self.predicted_masks
