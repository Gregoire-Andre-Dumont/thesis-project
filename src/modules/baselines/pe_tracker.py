"""SAM 2 VOS whose mask selection AND memory control are driven by Perception-Encoder similarity.

Both decisions SAM normally makes with its own IoU token are replaced by one identity score:

    score(proposal) = chamfer(anchor_fg -> proposal_fg) - chamfer(anchor_fg -> proposal_bg)

the FG-BG margin against the ANCHOR's foreground patches (not a running prototype, so the reference
can never drift). High score = "the patches inside this mask look like the target and the patches
around it do not".

  * SELECTION: track with the proposal of highest score, instead of SAM's IoU-token argmax.
  * MEMORY CONTROL: commit that proposal only when its score clears `pe_threshold`.

Unlike MemoryOracle/MaskOracle this uses NO ground truth -- it is a deployable tracker, so it is
compared against the SAM baseline rather than used as an upper bound.
"""
import numpy as np
import torch
import torch.nn.functional as F

from collections import deque
from dataclasses import dataclass
from src.typing.detection_data import DetectionData
from src.modules.memories.main_memory import MainMemory
from src.modules.samara_hiera_model import SamaraHieraModel
from src.offline_training.dataset_encoders import crop_around_masks, anchor_size_pixels


def chamfer(reference, candidate, bidirectional=True):
    """Symmetric chamfer between two patch-token sets. NaN when either side is empty.

    The two directions ask different questions, and only their combination is a sound mask score:
      candidate -> reference  "is everything I selected part of the target?"   (precision)
      reference -> candidate  "did I select all of the target?"               (recall)
    Unidirectional candidate->reference alone is maximised by a small, highly-typical mask -- a
    torso-only proposal whose every patch matches the anchor beats the correct whole-person mask --
    which biases proposal SELECTION toward under-segmentation. The reverse direction penalises the
    anchor patches such a mask leaves unmatched, so the mean of the two removes that bias."""

    if reference.shape[0] == 0 or candidate.shape[0] == 0:
        return float("nan")
    reference = F.normalize(reference.float(), dim=-1)
    candidate = F.normalize(candidate.float(), dim=-1)
    similarity = candidate @ reference.T                          # (n_candidate, n_reference)
    candidate_to_reference = float(similarity.max(dim=1).values.mean())
    if not bidirectional:
        return candidate_to_reference
    return 0.5 * (candidate_to_reference + float(similarity.max(dim=0).values.mean()))


@dataclass
class PETracker:
    """SAM 2 with PE-driven proposal selection and PE-gated memory commits."""

    model: SamaraHieraModel | None = None
    main_memory: MainMemory | None = None
    encode: object = None              # crops (n, H, W, 3) uint8 -> (n, grid*grid, dim) patch tokens
    gate: str = "sam"                  # "sam" = SAM's own object-score + predicted-IoU gate; "pe" = FG similarity
    iou_threshold: float = 0.5         # SAM gate: predicted IoU of the chosen proposal must exceed this
    pe_threshold: float = 0.5          # PE gate: foreground similarity required to commit
    pe_select: bool = True             # False = SAM's IoU token still picks the mask, PE only gates memory
    subtract_background: bool = False  # True = score the FG-BG margin instead of foreground similarity alone
    bidirectional: bool = True         # symmetric chamfer; False = candidate->anchor only (biased to small masks)
    reference_history: int = 30        # committed foregrounds kept alongside the anchor as extra references
    crop_size: int = 512
    pad_ratio: float = 0.2
    anchor_floor: int | None = None    # crop floor in px; set by the experiment from the AMODAL anchor box

    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = self.model.to(device=self.device, dtype=self.dtype)

    @torch.inference_mode()
    def encode_proposals(self, frame, masks):
        """PE patch tokens and the per-patch foreground mask for each proposal's crop, as dense
        `(k, P, D)` / `(k, P)` tensors. All `masks` come from the SAME frame, so one encoder pass covers
        every proposal, and keeping the result dense lets the scoring stay batched.

        `masks` stay RAW LOGITS: crop_around_masks resizes then thresholds at >0 exactly once. Passing
        a pre-binarized mask instead would resample a 0/1 step function with INTER_CUBIC, whose positive
        ringing survives that threshold and dilates the foreground into background patches."""

        logits = np.asarray(masks, dtype=np.float32)
        crops, crop_masks = crop_around_masks(
            np.broadcast_to(frame[None], (len(logits),) + frame.shape), logits,
            self.crop_size, self.pad_ratio, self.floor)

        tokens = self.encode(crops)                                    # (k, P, D)
        grid_size = round(tokens.shape[1] ** 0.5)
        crop_mask = torch.from_numpy(crop_masks).unsqueeze(1).to(tokens.device).float()
        foreground = (F.interpolate(crop_mask, size=(grid_size, grid_size), mode="nearest") > 0.5).flatten(1)
        return tokens, foreground

    @staticmethod
    def _directed(similarity, selection):
        """Both chamfer directions for every proposal at once, over the patches `selection` keeps.

        `similarity` is (k, P, A) -- proposal patches vs anchor foreground patches. Deselected patches
        are pushed to a large negative value so they can never win a max, and the forward mean is taken
        over selected patches only. Returns (forward, reverse, count), each (k,)."""

        floor_value = torch.finfo(similarity.dtype).min
        masked = similarity.masked_fill(~selection.unsqueeze(-1), floor_value)
        count = selection.sum(dim=1)

        best_anchor = masked.max(dim=2).values                          # (k, P): each patch -> best anchor patch
        forward = (best_anchor * selection).sum(dim=1) / count.clamp(min=1)
        reverse = masked.max(dim=1).values.mean(dim=1)                  # (k, A): each anchor patch -> best patch
        return forward, reverse, count

    def _chamfer_batch(self, similarity, selection):
        """Chamfer score per proposal, NaN where the selection is empty."""

        forward, reverse, count = self._directed(similarity, selection)
        scores = 0.5 * (forward + reverse) if self.bidirectional else forward
        return torch.where(count > 0, scores, torch.full_like(scores, float("nan")))

    @torch.inference_mode()
    def proposal_scores(self, tokens, foreground):
        """Score every proposal against EVERY reference and average. NaN where the mask is empty.

        References are the anchor plus the last `reference_history` committed foregrounds, so the score
        asks "does this look like the target as we have actually been seeing it", not only "does it look
        like frame 0". The anchor alone cannot drift but also cannot follow pose, lighting or scale
        change; the recent entries follow those but can drift, and averaging keeps the anchor's vote in
        every score as an anchor against runaway drift.

        All proposals are scored in one matmul per reference, with a single device->host transfer at the
        end. Scoring proposals one at a time costs a GPU sync per `float()` -- several per frame -- which
        stalls the pipeline far more than the arithmetic itself.

        Foreground similarity alone by default. The FG-BG margin is available but is not
        scale-invariant across proposals -- a tight mask pushes the rest of the target into its own
        background, where it scores high against the anchor and depresses the margin -- so it
        systematically prefers loose masks and makes a poor selector."""

        candidate = F.normalize(tokens.float(), dim=-1)
        references = [self.anchor_foreground] + list(self.reference_tokens)

        total = None
        for entry in references:
            similarity = candidate @ F.normalize(entry.float(), dim=-1).T        # (k, P, A_i)
            scores = self._chamfer_batch(similarity, foreground)
            if self.subtract_background:
                scores = scores - self._chamfer_batch(similarity, ~foreground)
            total = scores if total is None else total + scores
        return (total / len(references)).double().cpu().numpy()

    @torch.inference_mode()
    def seed_anchor(self, detection_data: DetectionData, init_mask):
        """Anchor foreground tokens -- the fixed reference every later frame is scored against."""

        frame = detection_data.frames[0]
        # Crop floor at the target's TRUE scale: the amodal anchor box when the experiment supplies it,
        # since detection_data.bboxes_norm is the VISIBLE box and shrinks under partial occlusion.
        self.floor = (self.anchor_floor if self.anchor_floor is not None
                      else anchor_size_pixels(detection_data.bboxes_norm[0], frame.shape))
        mask = init_mask.to(torch.float64).cpu().numpy().squeeze()     # logits; crop_around_masks thresholds
        tokens, foreground = self.encode_proposals(frame, mask[None])
        self.anchor_foreground = tokens[0][foreground[0]]

    def should_commit(self, object_score, predicted_iou, score):
        """Whether to write this frame into the memory bank.

        "sam" reproduces the SAM baseline's gate exactly -- sigmoid(object score) > 0.5 and SAM's own
        predicted IoU for the CHOSEN proposal above `iou_threshold` -- so the only difference from the
        baseline is which proposal PE selected. `object_score` arrives raw from propose_masks (the memory
        encoder wants it pre-sigmoid), unlike select_best_mask which returns it already squashed."""

        if self.gate == "sam":
            return bool(torch.sigmoid(object_score).item() > 0.5 and predicted_iou > self.iou_threshold)
        return bool(not np.isnan(score) and score >= self.pe_threshold)

    def predict_masks(self, detection_data: DetectionData):
        """Roll SAM 2 over the sequence: PE picks the proposal, `gate` decides what reaches memory."""

        self.main_memory.reset_memory()
        init_mask = self.main_memory.initialize_references(self.model, detection_data, anchor_index=0)
        self.seed_anchor(detection_data, init_mask)

        n_frames = detection_data.frames.shape[0]
        self.predicted_masks = torch.zeros((n_frames, 256, 256), dtype=torch.float64)
        self.pe_scores = torch.zeros(n_frames, dtype=torch.float64)      # PE score of the proposal we tracked
        self.predicted_ious = torch.zeros(n_frames, dtype=torch.float64)  # SAM's IoU token for that proposal
        self.committed_frames = []
        # Foregrounds of the frames that actually reached memory -- the reference set alongside the anchor.
        self.reference_tokens = deque(maxlen=self.reference_history)

        cache = getattr(self, "frame_cache", None)

        for idx, current_frame in enumerate(detection_data.frames):
            reuse = [e.to(self.device) for e in cache[idx]] if cache is not None and idx in cache else None
            mask_preds, iou_scores, object_pointers, object_score, lowres_imgenc, image_features = self.model.propose_masks(
                main_memory=self.main_memory, current_frame=current_frame, encoded_image_features_list=reuse)

            if cache is not None and idx not in cache:
                cache[idx] = [e.detach().cpu() for e in image_features]

            # Proposal 0 is the multimask-disabled "whole" mask, skipped exactly as the oracles skip it.
            candidates = mask_preds[0, 1:].to(torch.float64).cpu().numpy()
            tokens, foreground = self.encode_proposals(current_frame, candidates)
            scores = self.proposal_scores(tokens, foreground)

            # Every proposal empty (nothing to encode) -> fall back to SAM's own IoU-token pick.
            if np.all(np.isnan(scores)) or not self.pe_select:
                best_offset = int(torch.argmax(iou_scores[0, 1:]))
                score = float(scores[best_offset]) if not np.all(np.isnan(scores)) else float("nan")
            else:
                best_offset = int(np.nanargmax(scores))
                score = float(scores[best_offset])

            chosen_mask, pointer, encoding = self.model.commit_candidate(
                mask_preds, 1 + best_offset, object_pointers, object_score, lowres_imgenc)

            self.predicted_masks[idx] = chosen_mask.to(torch.float64)
            self.pe_scores[idx] = score
            self.predicted_ious[idx] = predicted_iou = float(iou_scores[0, 1 + best_offset])

            if self.should_commit(object_score, predicted_iou, score):
                self.main_memory.update_memory(pointer, encoding)
                self.committed_frames.append(idx)
                # Advanced indexing copies, so this does not pin the whole frame's token tensor.
                self.reference_tokens.append(tokens[best_offset][foreground[best_offset]])

        return self.predicted_masks
