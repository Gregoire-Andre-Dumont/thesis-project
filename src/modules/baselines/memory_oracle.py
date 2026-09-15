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


    # Memory-bank corruption: with probability `corruption_p`, commit a CLEAN nearby distractor instead of
    # the target. `corruption_boxes` is per-frame (None = not corruptible), non-None from the first occlusion
    # onward wherever another annotated person exists -- occluded frames included.
    corruption_p: float = 0.0
    corruption_boxes: list | None = None
    corruption_seed: int = 0

    def __post_init__(self):
        """Load the SAM 2 model onto the GPU in bfloat16."""

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = self.model.to(device=self.device, dtype=self.dtype)

    def corruption_draw(self, idx, rng):
        """True when frame `idx` is a corruption event: a commit here writes the nearest distractor, not the
        target.

        The draw happens on every frame regardless of `corruption_p` or whether a distractor exists, so the
        random stream stays aligned across a probability sweep -- with one seed per clip the events at p=0.05
        are a subset of those at p=0.20, making the sweep a nested series rather than five independent draws."""

        value = float(rng.random())
        box = self.corruption_boxes[idx] if (self.corruption_boxes is not None
                                             and idx < len(self.corruption_boxes)) else None
        return box is not None and value < self.corruption_p

    def distractor_memory(self, idx, image_features):
        """(pointer, encoding) for the nearest distractor on this frame, box-prompted through SAM.

        A clean, well-formed mask of a real person, so substituting it for the target corrupts the bank's
        IDENTITY rather than its mask quality -- the error a re-ID gate is supposed to catch."""

        _, encoding, pointer = self.model.initialize_video_masking(
            image_features, convert_bbox(np.asarray(self.corruption_boxes[idx], dtype=np.float32)))
        return pointer, encoding

    @torch.inference_mode()
    def pseudo_truth(self, image_features, bboxes_norm):
        """The GT box prompted through SAM 2, giving a MASK to score against instead of the box itself.

        Box IoU compares two filled rectangles, so it is blind to mask shape: a proposal covering the target
        plus an adjacent distractor can share the target's bounding box and score 1.0. This is the same
        box-prompted pseudo-GT the labelling pipeline uses -- it is SAM's own segmentation of the GT box, so
        it inherits SAM's errors, but it is the only mask-level ground truth PersonPath admits."""

        mask, _, _ = self.model.initialize_video_masking(image_features, convert_bbox(np.asarray(bboxes_norm, dtype=np.float32)))
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

    def choose(self, mask_preds, iou_scores, bboxes_norm, visible, truth=None, proposal_iou=None):
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

        self.iou_scores = torch.zeros(n_frames, dtype=torch.float64)   
        self.object_scores = torch.zeros(n_frames, dtype=torch.float64) 
        self.commit_iou = torch.zeros(n_frames, dtype=torch.float64)      


        self.proposal_iou_scores = torch.zeros((n_frames, 3), dtype=torch.float64)   # SAM's token per proposal
        self.proposal_true_iou = torch.zeros((n_frames, 3), dtype=torch.float64)     # vs GT; 0 when not visible
        self.chosen_index = torch.zeros(n_frames, dtype=torch.int64)                 # which proposal was kept, 0-2

        # The two masks the arm DISCARDS, kept alongside the one it keeps. Tracking only ever needs the
        # chosen mask, but a selector has to be trained and scored on the alternatives it was chosen over,
        # so the rollout is the only place they can be captured. float16: these are logits read back through
        # a `> 0` threshold, and at (n, 3, 256, 256) the full-precision copy is 8x the size for no gain.
        self.proposal_masks = torch.zeros((n_frames, 3, 256, 256), dtype=torch.float16)

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

            # Every proposal's true IoU, computed once: MaskOracle selects on it, corruption picks its argmin,
            # and it is stored for offline selection analysis.
            proposal_iou = (self.score((mask_preds[0, 1:] > 0.0).cpu().numpy(), bboxes_norm, truth)
                            if visible else None)
            self.proposal_iou_scores[idx] = iou_scores.reshape(-1)[1:4].to(torch.float64).cpu()
            self.proposal_masks[idx] = mask_preds[0, 1:].to(torch.float16).cpu()
            if proposal_iou is not None:
                self.proposal_true_iou[idx] = torch.as_tensor(proposal_iou, dtype=torch.float64)

            best_idx = self.choose(mask_preds, iou_scores, bboxes_norm, visible, truth, proposal_iou)
            self.chosen_index[idx] = int(best_idx) - 1                   # 0-2 into the proposal arrays

            chosen_mask, pointer, encoding = self.model.commit_candidate(
                mask_preds, best_idx, object_pointers, object_score, lowres_imgenc)

            self.predicted_masks[idx] = self.reported_mask(mask_preds, best_idx, chosen_mask)
            self.iou_scores[idx] = float(iou_scores.reshape(-1)[best_idx])
            self.object_scores[idx] = float(torch.as_tensor(object_score).reshape(-1)[0])

            corrupt = self.corruption_draw(idx, corruption_rng)

            # Commit gate: the chosen mask's IoU vs ground truth, on visible frames only.
            gated = False
            if visible:
                self.commit_iou[idx] = float(self.score((chosen_mask[None, :].numpy() > 0.0), bboxes_norm, truth)[0])
                gated = bool(self.commit_iou[idx] > self.iou_threshold)


            if corrupt:
                pointer, encoding = self.distractor_memory(idx, image_features)
                self.corrupted_frames.append(idx)

            if corrupt or gated:
                self.main_memory.update_memory(pointer, encoding)
                self.committed_frames.append(idx)

        return self.predicted_masks
