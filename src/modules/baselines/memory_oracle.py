import numpy as np
import torch

from dataclasses import dataclass
from src.typing.detection_data import DetectionData
from src.modules.memories.main_memory import MainMemory
from src.modules.samara_hiera_model import SamaraHieraModel
from src.utils.compute_iou import compute_iou
from src.utils.load_bboxes import convert_bbox
from src.utils.mask_components import mask_components, component_subsets, filter_logits

OBJECT_SCORE_FLOOR = 1e-3    # committed frames are GT-verified present: keep the encoder's object-score > 0


@dataclass
class MemoryOracle:
    """SAM 2 VOS that is an upper bound on the memory-COMMIT decision.

    It tracks exactly like the SAM baseline -- the mask is picked by SAM 2's own IoU token -- but it commits a
    frame to the memory bank only when the prediction is verified good against the ground truth: BOX IoU between
    the predicted mask's bounding box and the GT box exceeds `iou_threshold`, scored on visible frames only (so
    occluded frames never commit). Only the commit gate differs from the baseline; mask selection is identical.

    With `use_components`, the mask that gets COMMITTED is narrowed to the connected-component subset of SAM's
    chosen proposal that best matches the GT box -- so a proposal covering target + distractor writes only the
    target into memory. The REPORTED mask stays SAM's unfiltered pick, so coverage isolates the memory effect.

    Subclasses may override `choose` to change which proposal is kept (see MaskOracle)."""

    iou_threshold: float | None = None
    model: SamaraHieraModel | None = None
    main_memory: MainMemory | None = None
    use_components: bool = False

    # Memory-bank corruption (claim_2): at each commit, with probability `corruption_p`, write a CLEAN nearby
    # distractor into the bank instead of the target. `corruption_boxes` is per-frame (None = cannot corrupt).
    corruption_p: float = 0.0
    corruption_boxes: list | None = None
    corruption_seed: int = 0

    def __post_init__(self):
        """Load the SAM 2 model onto the GPU in bfloat16."""

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = self.model.to(device=self.device, dtype=self.dtype)

    def commit_corruption(self, idx, image_features, rng):
        """With probability `corruption_p`, box-prompt the nearest distractor on this frame and push THAT into the
        memory bank. Returns True when the bank was corrupted.

        This runs on EVERY frame -- occluded or visible -- and is independent of the arm's own commit gate, so all
        arms face the same corruption events rather than being poisoned wherever each happens to commit. The
        injected entry is a clean, well-formed mask of a real person, so it corrupts the bank's IDENTITY, not its
        mask quality."""

        if self.corruption_p <= 0.0 or self.corruption_boxes is None:
            return False
        box = self.corruption_boxes[idx] if idx < len(self.corruption_boxes) else None
        if box is None or float(rng.random()) >= self.corruption_p:
            return False
        _, encoding, pointer = self.model.initialize_video_masking(
            image_features, convert_bbox(np.asarray(box, dtype=np.float32)))
        self.main_memory.update_memory(pointer, encoding)
        return True

    def best_subset(self, mask_logits, bboxes_norm):
        """Component subset of one proposal with the highest box IoU vs the GT box. None when the proposal is a
        single blob (or empty) and there is nothing to gain by splitting it."""

        components = mask_components(mask_logits)
        if len(components) < 2:
            return None
        subsets = component_subsets(components)
        ious = compute_iou(np.repeat(bboxes_norm[None, :], len(subsets), axis=0), np.stack(subsets))
        return subsets[int(np.argmax(ious))]

    def choose(self, mask_preds, iou_scores, bboxes_norm, visible):
        """Return `(proposal_index, keep)`: the proposal to track with, and the component subset of it to write
        into memory (`None` = the whole proposal). Memory oracle: SAM 2's own IoU-token argmax, same as the
        baseline; only the committed region is narrowed."""

        best_idx = int(1 + torch.argmax(iou_scores[:, 1:], dim=-1))
        keep = self.best_subset(mask_preds[0, best_idx], bboxes_norm) if (self.use_components and visible) else None
        return best_idx, keep

    def predict_masks(self, detection_data: DetectionData):
        """Roll SAM 2 over the sequence, committing its chosen mask to memory only on GT-verified visible frames."""

        self.main_memory.reset_memory()
        self.main_memory.initialize_references(self.model, detection_data, anchor_index=0)

        n_frames = detection_data.frames.shape[0]
        self.predicted_masks = torch.zeros((n_frames, 256, 256), dtype=torch.float64)
        self.iou_scores = torch.zeros(n_frames, dtype=torch.float64)      # SAM 2's own predicted IoU (diagnostic)
        self.commit_iou = torch.zeros(n_frames, dtype=torch.float64)      # predicted-mask-box vs GT-box IoU (visible frames)

        self.committed_frames = []
        self.filtered_frames = []                                          # frames whose committed mask was narrowed
        self.corrupted_frames = []                                         # frames where a distractor was committed
        corruption_rng = np.random.default_rng(self.corruption_seed)
        cache = getattr(self, "frame_cache", None)

        for idx, current_frame in enumerate(detection_data.frames):
            reuse = [e.to(self.device) for e in cache[idx]] if cache is not None and idx in cache else None
            mask_preds, iou_scores, object_pointers, object_score, lowres_imgenc, image_features = self.model.propose_masks(
                main_memory=self.main_memory, current_frame=current_frame, encoded_image_features_list=reuse)

            if cache is not None and idx not in cache:
                cache[idx] = [e.detach().cpu() for e in image_features]

            bboxes_norm = detection_data.bboxes_norm[idx]
            # GT-verifiable only: not occluded AND actually annotated with a box. An unannotated gap can verify
            # nothing, and letting it score a spurious 0 IoU would jam the commit gate shut.
            visible = bool(detection_data.occlusions[idx] <= 0.5) and float(bboxes_norm[2]) > 0
            best_idx, keep = self.choose(mask_preds, iou_scores, bboxes_norm, visible)

            # The oracle only commits GT-verified frames, where the object is known present.
            if keep is None:
                chosen_mask, pointer, encoding = self.model.commit_candidate(
                    mask_preds, best_idx, object_pointers, object_score, lowres_imgenc)
            else:
                # Commit only the chosen components; the pointer still describes the parent proposal.
                pointer = object_pointers[:, best_idx:best_idx + 1, :]
                chosen_mask, pointer, encoding = self.model.commit_mask(
                    filter_logits(mask_preds[0, best_idx], keep), pointer, object_score, lowres_imgenc)
                self.filtered_frames.append(idx)

            self.predicted_masks[idx] = self.reported_mask(mask_preds, best_idx, chosen_mask)
            self.iou_scores[idx] = float(iou_scores.max())

            # Commit gate: the chosen mask's BOX IoU vs the GT box, on visible frames only.
            if visible:
                self.commit_iou[idx] = float(compute_iou(bboxes_norm[None, :], chosen_mask[None, :].numpy())[0])
                if self.commit_iou[idx] > self.iou_threshold:
                    self.main_memory.update_memory(pointer, encoding)
                    self.committed_frames.append(idx)

            # Corruption is independent of the gate above: it fires on EVERY frame, occluded or visible, so all
            # arms face the same injected errors regardless of where each one chooses to commit.
            if self.commit_corruption(idx, image_features, corruption_rng):
                self.corrupted_frames.append(idx)

        return self.predicted_masks

    def reported_mask(self, mask_preds, best_idx, chosen_mask):
        """Mask this arm is scored on. The memory oracle intervenes on MEMORY only, so it reports SAM's
        unfiltered proposal even when a narrowed one was committed."""

        return mask_preds[0, best_idx].to(torch.float64).cpu()
