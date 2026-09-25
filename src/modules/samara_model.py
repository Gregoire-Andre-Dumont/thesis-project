import cv2
import torch
import numpy as np
import torch.nn.functional as F
from numpy.typing import NDArray

from sam_2.sam_v2_model import SAMV2Model
from sam_2.make_sam_v2 import make_samv2_from_state_dict
from src.offline_training.dataset_encoders import load_dataset_encoders

class SamaraModel(SAMV2Model):
    """SAM 2 whose memory decisions are scored against an EXTERNAL encoder's patch tokens.

    SAM's own Hiera tracks and proposes masks, exactly as in the baseline. What it does not do is judge them:
    the calibrator's similarity features come from `feature_encoder` (the Perception Encoder), which is what
    the calibrator's dataset was built from. Scoring with SAM's Hiera or memory encoder was possible here
    once and is not any more -- claim_3 measured both as the weakest re-ID signals of six backbones, and
    keeping them as options let a deployment silently feed the calibrator features it never trained on."""

    def __init__(
        self,
        sam_model_path: str | None = None,
        controller: torch.nn.Module | None = None,
        crop_resize: int | None = None,
        pad_ratio: float = 0.25,
        feature_encoder: str = "perception",
    ):
        if not feature_encoder:
            raise ValueError("feature_encoder is required: the calibrator scores external tokens, not SAM's")

        _, sam_model = make_samv2_from_state_dict(sam_model_path)
        super().__init__(
            image_encoder_model=sam_model.image_encoder,
            coordinate_encoder=sam_model.coordinate_encoder,
            prompt_encoder_model=sam_model.prompt_encoder,
            mask_decoder_model=sam_model.mask_decoder,
            memory_encoder_model=sam_model.memory_encoder,
            memory_fusion_model=sam_model.memory_fusion)

        self.controller = controller
        self.gate_controller = None
        self.crop_resize = crop_resize
        self.pad_ratio = pad_ratio
        self.anchor_size_pixels = 0

        self.feature_encoder = feature_encoder
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._feature_token_fn = load_dataset_encoders([feature_encoder], device, dtype)[feature_encoder]
        self.eval()

    def commit_mask(self, mask_logits, pointer, object_score, lowres_imgenc):
        """`commit_candidate` for an EDITED mask: same memory-encoder path, but the mask is handed over
        directly instead of being sliced out of the proposal stack by index. Used to commit a
        connected-component subset of a proposal; the pointer still comes from that parent proposal."""

        with torch.inference_mode():
            chosen_mask = mask_logits[None, None]                  # (H, W) -> (B, 1, H, W) the encoder expects
            chosen_encoding = self.memory_encoder(
                mask_prediction=chosen_mask, object_score=object_score, lowres_image_encoding=lowres_imgenc)
        return chosen_mask.squeeze().to(torch.float64).cpu(), pointer, chosen_encoding

    # -------------------------------------------------------------------------------
    # Calibrator-driven mask scoring
    # -------------------------------------------------------------------------------

    def _score_masks(self, current_frame, candidate_masks_raw, reference_foreground, reference_background,
                     scalars=None):
        """Crop each candidate mask and build its anchor-similarity features.
        Returns ((predicted IoU, commit probability), foreground, background, features)."""

        candidate_masks = candidate_masks_raw.to(torch.float64).cpu().numpy()   # keep logits; extract_crops thresholds
        frames = np.repeat(np.asarray(current_frame)[None], len(candidate_masks), axis=0)

        cropped_frames, cropped_masks = self.extract_crops(frames, candidate_masks)
        foreground, background = self.extract_patch_tokens(cropped_frames, cropped_masks)

        n_masks = len(candidate_masks)
        side = int(round(foreground.shape[1] ** 0.5))
        fg_fg = self.compute_patch_similarities(reference_foreground, foreground).reshape(n_masks, -1, side, side)
        bg_fg = self.compute_patch_similarities(reference_foreground, background).reshape(n_masks, -1, side, side)

        features = torch.from_numpy(np.stack([fg_fg, bg_fg], axis=-1).astype(np.float32)).to("cuda")
        features = self._append_scalars(features, scalars)
        return self._run_controller(features), foreground, background, features

    @staticmethod
    def _append_scalars(features, scalars):
        """Append per-mask scalars as constant channels, the way `MainDataset` does at training time.

        A controller trained with `dataset.scalars` reads SAM's own IoU token and object score off the
        trailing channels, so deployment has to put them in the same place. The gate is unaffected: it
        slices channels 0:2 and its `n_scalars` is 0, so extra channels pass it by."""

        if scalars is None:
            return features

        values = torch.as_tensor(np.asarray(scalars, dtype=np.float32),
                                 device=features.device).reshape(len(features), -1)
        planes = values[:, None, None, None, :].expand(-1, *features.shape[1:4], values.shape[-1])
        return torch.cat([features, planes], dim=-1)

    def _run_controller(self, features):
        """(predicted IoU, commit probability) per mask, each (n_masks,).

        Two models, one per decision, because the decisions are not the same question. `controller` REGRESSES
        the IoU that ranks the proposals -- returned as-is, since a sigmoid would squash [0, 1] into
        [0.5, 0.731] and silently disable any gate thresholded on it. `gate_controller` CLASSIFIES whether the
        mask is worth committing, and its logit becomes a probability here, so `commit_threshold` is a real
        decision boundary.

        Either may be absent when the tracker does not use that decision: the missing score comes back as NaN
        rather than as a value that looks usable.

        Autocast is disabled so a bf16 context can't corrupt the float32 calibrators."""

        blank = np.full(len(features), np.nan, dtype=np.float32)
        with torch.autocast("cuda", enabled=False):
            selected = self.controller(features.float()) if self.controller is not None else None
            gate_logits = self.gate_controller(features.float()) if self.gate_controller is not None else None

        predicted_iou = selected[:, 0].clamp(0.0, 1.0).cpu().numpy() if selected is not None else blank
        commit_probability = torch.sigmoid(gate_logits[:, 0]).cpu().numpy() if gate_logits is not None else blank
        return predicted_iou, commit_probability

    @torch.inference_mode()
    def score_proposals(self, current_frame, candidate_masks, reference_foreground, reference_background,
                        scalars=None):
        """Per candidate mask: (predicted IoU for RANKING, commit probability for the GATE), each (n_masks,).

        Both come from one forward pass over the same similarity features -- the two heads disagree on what
        they optimise, not on what they see."""

        (scores, commit_probabilities), _, _, _ = self._score_masks(
            current_frame, candidate_masks, reference_foreground, reference_background, scalars)
        return scores, commit_probabilities

    # -------------------------------------------------------------------------------
    # Cropping — per-frame adaptive, driven entirely by pad_ratio
    # -------------------------------------------------------------------------------

    def extract_crops(self, frames: NDArray[np.uint8], masks: NDArray[np.uint8]):
        """Crop each frame to a square window around its mask, padded by pad_ratio and
        resized to crop_resize. Frames whose mask is empty are returned as zeros.

        The mask arrives as continuous SAM logits: every resize is bilinear and the binary
        threshold (logit > 0) is applied AFTER rescaling, so the crop mask keeps a smooth
        boundary instead of the blocky staircase of a threshold-then-nearest-resize."""

        crop_size = self.crop_resize
        cropped_frames = np.zeros((masks.shape[0], crop_size, crop_size, 3), dtype=np.float32)
        cropped_masks = np.zeros((masks.shape[0], crop_size, crop_size), dtype=np.float32)

        for index, (frame, mask) in enumerate(zip(frames, masks)):
            mask = cv2.resize(mask.astype(np.float32), (frame.shape[1], frame.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
            coordinates = cv2.findNonZero((mask > 0.0).astype(np.uint8))
            if coordinates is None:
                continue

            x, y, width, height = cv2.boundingRect(coordinates)
            x_min, y_min, x_max, y_max = self._padding_ratio_window(x, y, width, height, frame.shape[:2])

            crop_frame = frame[y_min:y_max, x_min:x_max]
            crop_mask = mask[y_min:y_max, x_min:x_max]
            cropped_frames[index] = cv2.resize(crop_frame, (crop_size, crop_size), interpolation=cv2.INTER_CUBIC)
            crop_mask = cv2.resize(crop_mask, (crop_size, crop_size), interpolation=cv2.INTER_CUBIC)
            cropped_masks[index] = (crop_mask > 0.0).astype(np.float32)      # threshold AFTER rescaling

        return cropped_frames, cropped_masks

    def set_anchor_size_from_normalized(self, anchor_bbox_norm, frame_shape):
        """Store the anchor's (visible) box size in pixels, used as a floor for later crop
        windows so a crop doesn't collapse when the mask shrinks under partial occlusion.
        Pass a zero box to disable it and fall back to mask-derived crop sizing."""

        frame_height, frame_width = frame_shape
        width_pixels = anchor_bbox_norm[2] * frame_width
        height_pixels = anchor_bbox_norm[3] * frame_height
        self.anchor_size_pixels = int(round(max(width_pixels, height_pixels)))

    def _padding_ratio_window(self, x, y, width, height, frame_shape):
        """Compute the square crop box centered on the mask, sized by the mask and the
        anchor's (visible) size floor and clamped to fit inside the frame."""

        frame_height, frame_width = frame_shape
        base_side = max(width, height, 1, self.anchor_size_pixels)
        crop_side = min(int(round(base_side * (1 + 2 * self.pad_ratio))), frame_width, frame_height)

        half = crop_side / 2
        x_min = int(round(x + width / 2 - half))
        y_min = int(round(y + height / 2 - half))
        x_max = x_min + crop_side
        y_max = y_min + crop_side

        if x_min < 0:
            x_max -= x_min; x_min = 0
        if y_min < 0:
            y_max -= y_min; y_min = 0
        if x_max > frame_width:
            x_min -= (x_max - frame_width); x_max = frame_width
        if y_max > frame_height:
            y_min -= (y_max - frame_height); y_max = frame_height
        return x_min, y_min, x_max, y_max

    # -------------------------------------------------------------------------------
    # Patch-token extraction + foreground/background split
    # -------------------------------------------------------------------------------

    def split_foreground_background(self, patch_tokens, patch_masks):
        """Split patch tokens into foreground and background views using the patch mask.
        Masked-out patches are filled with a -5 sentinel so they can be ignored later."""

        patch_masks = patch_masks.bool().unsqueeze(-1)
        padding_value = torch.tensor(-5, device=patch_tokens.device, dtype=patch_tokens.dtype)
        foreground = torch.where(patch_masks, patch_tokens, padding_value)
        background = torch.where(~patch_masks, patch_tokens, padding_value)
        return foreground, background

    def extract_patch_tokens(self, cropped_frames, cropped_masks, encoder_chunk_size=16):
        """The calibrator's patch tokens for the crops, as foreground/background views.

        Always the external `feature_encoder` -- the same tokens the calibrator's dataset was built with.
        SAM's own Hiera and memory encoders were alternatives here once; they are gone because they made the
        deployed features silently divergeable from the trained-on ones, and because claim_3 measured them as
        the weakest re-ID signals of the six backbones tested."""

        from src.offline_training.dataset_encoders import encode_tokens, _patch_masks

        tokens = encode_tokens(self._feature_token_fn, np.asarray(cropped_frames), encoder_chunk_size)
        patch_masks = _patch_masks(np.asarray(cropped_masks, dtype=np.float32), tokens.device)
        return self.split_foreground_background(tokens, patch_masks)

    # -------------------------------------------------------------------------------
    # Patch similarity
    # -------------------------------------------------------------------------------

    def compute_patch_similarities(self, reference_tokens, target_tokens, target_chunk_size=16):
        """For each target patch, take its best cosine similarity to the reference patches.
        Tokens are L2-normalized and -5 sentinel patches are excluded from the match."""

        ref_valid = ~(reference_tokens == -5).all(dim=-1)
        ref_mask = ref_valid[:, None, :, None]
        all_ref_invalid = ~ref_valid.any(dim=-1)
        reference_normalized = F.normalize(reference_tokens, dim=-1)

        chunks = []
        for chunk_start in range(0, target_tokens.shape[0], target_chunk_size):
            target_chunk = target_tokens[chunk_start:chunk_start + target_chunk_size]
            target_valid = ~(target_chunk == -5).all(dim=-1)
            target_normalized = F.normalize(target_chunk, dim=-1)

            similarities = torch.einsum("rpd,tqd->rtpq", reference_normalized, target_normalized)
            vectors_chunk = similarities.masked_fill(~ref_mask, float("-inf")).amax(dim=2)
            if all_ref_invalid.any():
                vectors_chunk[all_ref_invalid] = 0.0
            vectors_chunk = vectors_chunk * target_valid[None, :, :].to(vectors_chunk.dtype)
            chunks.append(vectors_chunk)

        return torch.cat(chunks, dim=1).float().cpu().numpy().transpose(1, 0, 2)
