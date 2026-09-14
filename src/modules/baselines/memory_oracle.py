import numpy as np
import torch

from dataclasses import dataclass
from src.typing.detection_data import DetectionData
from src.modules.memories.main_memory import MainMemory
from src.modules.samara_hiera_model import SamaraHieraModel
from src.utils.compute_iou import compute_iou
from src.utils.load_bboxes import convert_bbox

OBJECT_SCORE_FLOOR = 1e-3    # committed frames are GT-verified present: keep the encoder's object score > 0


@dataclass
class MemoryOracle:
    """SAM 2 VOS that is an upper bound on the memory-COMMIT decision.

    It tracks exactly like the SAM baseline -- the mask is picked by SAM 2's own IoU token -- but it commits a
    frame to the memory bank only when the prediction is verified good against the ground truth: IoU above
    `iou_threshold`, scored on visible frames only, so occluded frames and unannotated gaps never commit.
    Only the commit gate differs from the baseline; mask selection is identical.

    Subclasses may override `choose` to change which proposal is kept (see MaskOracle)."""

    iou_threshold: float | None = None
    model: SamaraHieraModel | None = None
    main_memory: MainMemory | None = None
    use_mask_iou: bool = True          # score against a box-prompted pseudo-GT MASK, not the GT box

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

    @torch.inference_mode()
    def pseudo_truth(self, image_features, bboxes_norm):
        """The GT box prompted through SAM 2, giving a MASK to score against instead of the box itself.

        Box IoU compares two filled rectangles, so it is blind to mask shape: a proposal covering the target
        plus an adjacent distractor can share the target's bounding box and score 1.0. This is the same
        box-prompted pseudo-GT the labelling pipeline uses -- it is SAM's own segmentation of the GT box, so
        it inherits SAM's errors, but it is the only mask-level ground truth PersonPath admits."""

        mask, _, _ = self.model.initialize_video_masking(
            image_features, convert_bbox(np.asarray(bboxes_norm, dtype=np.float32)))
        return mask.squeeze().to(torch.float64).cpu().numpy() > 0.0

    @staticmethod
    def mask_iou(truth, masks):
        """IoU of each binary mask in `masks` (k, H, W) against the binary `truth` (H, W)."""

        intersection = np.logical_and(masks, truth).sum(axis=(1, 2))
        union = np.logical_or(masks, truth).sum(axis=(1, 2))
        return intersection / np.maximum(union, 1)

    def score(self, masks, bboxes_norm, truth):
        """IoU of every candidate mask against ground truth -- mask IoU when a pseudo-GT mask is available,
        box IoU otherwise."""

        if self.use_mask_iou and truth is not None:
            return self.mask_iou(truth, masks)
        return compute_iou(np.repeat(bboxes_norm[None, :], len(masks), axis=0), masks)

    def choose(self, mask_preds, iou_scores, bboxes_norm, visible, truth=None):
        """Index of the proposal to track with. The memory oracle intervenes on MEMORY only, so this is SAM 2's
        own IoU-token argmax, identical to the baseline; the ground-truth arguments are here for subclasses
        that select on them (see MaskOracle)."""

        return int(1 + torch.argmax(iou_scores[:, 1:], dim=-1))

    def reported_mask(self, mask_preds, best_idx, chosen_mask):
        """Mask this arm is scored on -- SAM's chosen proposal. MaskOracle overrides it when selection itself
        is the intervention."""

        return mask_preds[0, best_idx].to(torch.float64).cpu()

    def predict_masks(self, detection_data: DetectionData):
        """Roll SAM 2 over the sequence, committing its chosen mask to memory only on GT-verified visible frames."""

        self.main_memory.reset_memory()
        self.main_memory.initialize_references(self.model, detection_data, anchor_index=0)

        n_frames = detection_data.frames.shape[0]
        self.predicted_masks = torch.zeros((n_frames, 256, 256), dtype=torch.float64)
        self.iou_scores = torch.zeros(n_frames, dtype=torch.float64)      # SAM 2's own predicted IoU (diagnostic)
        self.commit_iou = torch.zeros(n_frames, dtype=torch.float64)      # chosen mask vs ground truth (visible frames)

        self.committed_frames = []
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
            visible = bool(detection_data.occlusions[idx] <= 0.5) and float(bboxes_norm[2]) > 0

            truth = self.pseudo_truth(image_features, bboxes_norm) if (visible and self.use_mask_iou) else None
            best_idx = self.choose(mask_preds, iou_scores, bboxes_norm, visible, truth)

            chosen_mask, pointer, encoding = self.model.commit_candidate(
                mask_preds, best_idx, object_pointers, object_score, lowres_imgenc)

            self.predicted_masks[idx] = self.reported_mask(mask_preds, best_idx, chosen_mask)
            self.iou_scores[idx] = float(iou_scores.max())

            # Commit gate: the chosen mask's IoU vs ground truth, on visible frames only.
            if bool(detection_data.occlusions[idx] <= 0.5) and float(bboxes_norm[2]) > 0:
                self.commit_iou[idx] = float(self.score((chosen_mask[None, :].numpy() > 0.0), bboxes_norm, truth)[0])

                if self.commit_iou[idx] > self.iou_threshold:
                    self.main_memory.update_memory(pointer, encoding)
                    self.committed_frames.append(idx)

            if self.commit_corruption(idx, image_features, corruption_rng):
                self.corrupted_frames.append(idx)

        return self.predicted_masks
