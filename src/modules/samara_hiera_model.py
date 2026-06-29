import torch
import cv2
import numpy as np
from numpy.typing import NDArray
import torch.nn.functional as F
from muggled_sam.sam_v2_model import SAMV2Model
from muggled_sam.make_sam_v2 import make_samv2_from_state_dict


class SamaraHieraModel(SAMV2Model):
    """SAM 2 with multi-reference patch-token extraction for SAMARA."""

    OBJECT_SCORE_VALUE = 100.0

    def __init__(
        self,
        sam_model_path: str | None = None,
        controller: torch.nn.Module | None = None,
        crop_resize: int | None = None,
        pad_ratio: float = 0.25,
        token_source: str = "hiera",
    ):
        if token_source not in {"hiera", "memory"}:
            raise ValueError(f"token_source must be 'hiera' or 'memory', got {token_source!r}")
        _, sam_model = make_samv2_from_state_dict(sam_model_path)
        self.stochastic_policy = True
        self.crop_resize = crop_resize
        self.pad_ratio = pad_ratio
        self.token_source = token_source
        self.anchor_amodal_pixels = 0

        super().__init__(
            image_encoder_model = sam_model.image_encoder,
            coordinate_encoder = sam_model.coordinate_encoder,
            prompt_encoder_model = sam_model.prompt_encoder,
            mask_decoder_model = sam_model.mask_decoder,
            memory_encoder_model = sam_model.memory_encoder,
            memory_fusion_model = sam_model.memory_fusion)

        self.controller = controller
        self.eval()

    # -------------------------------------------------------------------------------
    # Calibrator-driven mask scoring (unchanged from before)
    # -------------------------------------------------------------------------------

    def _score_masks(self, current_frame, candidate_masks_raw, reference_foreground, reference_background):
        """Run the calibrator on a batch of candidate masks. Returns `(samara_ious,
        foreground, background, features)` where `features` is the 4-channel patch-
        similarity tensor shaped for the controller. Channels (last axis):"""

        candidate_masks = (candidate_masks_raw > 0.0).to(torch.float64).cpu().numpy()
        frames = np.repeat(np.asarray(current_frame)[None], len(candidate_masks), axis=0)
        
        cropped_frames, cropped_masks = self.extract_crops(frames, candidate_masks)
        foreground, background = self.extract_patch_tokens(cropped_frames, cropped_masks)

        side = int(round(foreground.shape[1] ** 0.5))
        foreground_to_anchor_fg = self.compute_patch_similarities(reference_foreground, foreground).reshape(len(candidate_masks), -1, side, side)
        background_to_anchor_fg = self.compute_patch_similarities(reference_foreground, background).reshape(len(candidate_masks), -1, side, side)
        foreground_to_anchor_bg = self.compute_patch_similarities(reference_background, foreground).reshape(len(candidate_masks), -1, side, side)
        background_to_anchor_bg = self.compute_patch_similarities(reference_background, background).reshape(len(candidate_masks), -1, side, side)
        foreground_diff = foreground_to_anchor_fg - foreground_to_anchor_bg
        background_diff = background_to_anchor_fg - background_to_anchor_bg
        features = torch.from_numpy(np.stack(
            [foreground_to_anchor_fg, background_to_anchor_fg, foreground_diff, background_diff],
            axis=-1).astype(np.float32)).to("cuda")

        samara_ious = self._run_controller(self.controller, features)
        return samara_ious, foreground, background, features

    def _run_controller(self, controller, features):
        """Run the calibrator and return its predictions as a `(N,)` numpy array.

        The calibrator is trained as a binary classifier (`BCEWithLogitsLoss` on
        `IoU > t`). The model outputs raw logits; sigmoid is applied here so callers
        can threshold in `[0, 1]`. Autocast is disabled so a bf16-leaked context can't
        collapse the float32 calibrator weights. Returns NaN when no controller."""

        if controller is None:
            return np.full(features.shape[0], np.nan, dtype=np.float32)
        with torch.autocast("cuda", enabled=False):
            output = controller(features.float())
            probabilities = torch.sigmoid(output)
        return probabilities[:, 0].cpu().numpy()

    @torch.inference_mode()
    def select_best_mask_gated(self, current_frame, main_memory, reference_foreground, reference_background):
        """Pick SAM 2's highest-IoU candidate and score it with the moving calibrator
        for the memory-commit gate. Selection is pure SAM-argmax; the calibrator only
        scores it. (No SAMARA-driven re-ranking.)"""

        encoded_image_features_list, _, _ = self.encode_image(cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR))
        mask_preds, iou_scores, object_pointers, object_score, _, _ = self.step_video_masking(
            main_memory=main_memory, encoded_image_features_list=encoded_image_features_list)

        sam_ious = iou_scores.squeeze(0)[1:4].float().cpu().numpy()
        chosen = int(np.argmax(sam_ious))
        samara_ious, foreground, background, calibrator_features = self._score_masks(
            current_frame, mask_preds[:, 1 + chosen:2 + chosen, :, :].squeeze(0),
            reference_foreground, reference_background)

        best_index = 1 + chosen
        chosen_mask_raw = mask_preds[:, best_index:best_index + 1, :, :]

        chosen_encoding = self.memory_encoder(
            mask_prediction=chosen_mask_raw, object_score=object_score,
            lowres_image_encoding=encoded_image_features_list[0])

        return {
            "chosen_mask": (chosen_mask_raw > 0.0).squeeze().to(torch.float64).cpu(),
            "pointer": object_pointers[:, best_index:best_index + 1, :],
            "encoding": chosen_encoding,
            "iou_score": float(sam_ious[chosen]),
            "samara_iou": float(samara_ious[0]),
            "object_score": torch.sigmoid(object_score).squeeze().item(),
            "target_foreground": foreground[0],
            "target_background": background[0],
            "calibrator_features": calibrator_features}

    # -------------------------------------------------------------------------------
    # Cropping — per-frame adaptive, driven entirely by pad_ratio
    # -------------------------------------------------------------------------------

    def extract_crops(self, frames: NDArray[np.uint8], masks: NDArray[np.uint8]):
        """Per-frame square crops at `crop_resize` pixels:

            crop_side = min(mask_side × (1 + 2 × pad_ratio), frame_width, frame_height)
            mask_side = max(mask_bbox_width, mask_bbox_height)

        The crop is centered on the mask centroid, shifted inward at frame edges. The
        crop side is capped at the smallest frame dimension so the square always fits
        — no border padding is ever needed (the crop shrinks instead). Frames whose
        mask is empty are returned as zeros."""

        crop_size = self.crop_resize
        cropped_frames = np.zeros((masks.shape[0], crop_size, crop_size, 3), dtype=np.float32)
        cropped_masks = np.zeros((masks.shape[0], crop_size, crop_size), dtype=np.float32)

        for index, (frame, mask) in enumerate(zip(frames, masks)):
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            coordinates = cv2.findNonZero(mask.astype(np.uint8))
            if coordinates is None:
                continue

            x, y, width, height = cv2.boundingRect(coordinates)
            x_min, y_min, x_max, y_max = self._padding_ratio_window(
                x, y, width, height, frame.shape[:2])

            crop_frame = frame[y_min:y_max, x_min:x_max]
            crop_mask = mask[y_min:y_max, x_min:x_max]
            cropped_frames[index] = cv2.resize(crop_frame, (crop_size, crop_size), interpolation=cv2.INTER_CUBIC)
            cropped_masks[index] = cv2.resize(crop_mask, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)

        return cropped_frames, cropped_masks

    def set_anchor_amodal_from_normalized(self, amodal_bbox_norm, frame_shape):
        """Convenience: set `self.anchor_amodal_pixels` from a normalized `(x, y, w, h)`
        anchor amodal bbox and a frame `(height, width)`. Pass `(0, 0, 0, 0)` to disable
        (falls back to mask-derived crop sizing)."""

        frame_height, frame_width = frame_shape
        amodal_width_pixels = amodal_bbox_norm[2] * frame_width
        amodal_height_pixels = amodal_bbox_norm[3] * frame_height
        self.anchor_amodal_pixels = int(round(max(amodal_width_pixels, amodal_height_pixels)))

    def _padding_ratio_window(self, x, y, width, height, frame_shape):
        """Square `(x_min, y_min, x_max, y_max)` centered on the mask's centroid.

        Crop side:
            base = max(current_mask_side, anchor_amodal_pixels)
            crop = base × (1 + 2 × pad_ratio)
            crop = min(crop, frame_width, frame_height)

        So the crop is at LEAST as large as the anchor's amodal extent (SiamFC/STARK-
        style floor), but grows beyond it whenever the current frame's mask is larger
        (target moves closer, occlusion clears). Falls back to mask-derived-only when
        `anchor_amodal_pixels == 0`."""

        frame_height, frame_width = frame_shape
        mask_side = max(width, height, 1)
        base_side = max(mask_side, self.anchor_amodal_pixels)
        crop_side = int(round(base_side * (1 + 2 * self.pad_ratio)))
        crop_side = min(crop_side, frame_width, frame_height)

        center_x = x + width / 2
        center_y = y + height / 2
        half = crop_side / 2
        x_min = int(round(center_x - half))
        y_min = int(round(center_y - half))
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
    # Patch-token extraction (Hiera or Memory) + foreground/background split
    # -------------------------------------------------------------------------------

    def extract_raw_patch_tokens(self, cropped_frames, cropped_masks, encoder_chunk_size=16):
        """Encode cropped frames into raw patch tokens and the patch-grid foreground
        mask. Returns `(patch_tokens, patch_masks)`. The foreground/background split
        is deferred to `split_foreground_background`."""

        with torch.inference_mode():
            feature_chunks = []
            for chunk_start in range(0, len(cropped_frames), encoder_chunk_size):
                chunk_end = min(chunk_start + encoder_chunk_size, len(cropped_frames))
                prepared = []
                for crop in cropped_frames[chunk_start:chunk_end]:
                    crop_bgr = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2BGR)
                    prepared.append(self.image_encoder.prepare_image(crop_bgr, self.crop_resize, True))

                batched_input = torch.cat(prepared, dim=0)
                encoded_features_list = self.image_encoder(batched_input)
                feature_chunks.append(encoded_features_list[0])

        feats = torch.cat(feature_chunks, dim=0).float()
        B, F_dim, H, W = feats.shape
        patch_tokens = feats.permute(0, 2, 3, 1).reshape(B, H * W, F_dim)
        device = patch_tokens.device

        cropped_masks = torch.as_tensor(cropped_masks, device=device).unsqueeze(1)
        patch_masks = F.interpolate(cropped_masks, size=(H, W), mode="nearest")
        patch_masks = (patch_masks > 0).flatten(2).squeeze(1)
        return patch_tokens, patch_masks

    def extract_memory_patch_tokens(self, cropped_frames, cropped_masks, encoder_chunk_size=8):
        """Encode each crop with the image encoder, then the memory encoder against
        the crop's mask. Returns `(memory_tokens, patch_masks)` with the same shape
        semantics as `extract_raw_patch_tokens`. Memory tokens are mask-conditioned —
        the representation SAM 2 cross-attends over at deployment."""

        with torch.inference_mode():
            token_chunks = []
            patch_mask_chunks = []
            for chunk_start in range(0, len(cropped_frames), encoder_chunk_size):
                chunk_end = min(chunk_start + encoder_chunk_size, len(cropped_frames))
                prepared = []
                for crop in cropped_frames[chunk_start:chunk_end]:
                    crop_bgr = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2BGR)
                    prepared.append(self.image_encoder.prepare_image(crop_bgr, self.crop_resize, True))

                batched_input = torch.cat(prepared, dim=0)
                encoded_features_list = self.image_encoder(batched_input)
                lowres = encoded_features_list[0]
                B, _, H, W = lowres.shape

                chunk_masks = torch.as_tensor(cropped_masks[chunk_start:chunk_end], device=lowres.device, dtype=lowres.dtype).unsqueeze(1)
                mask_at_decoder = F.interpolate(chunk_masks, size=(4 * H, 4 * W), mode="bilinear", align_corners=False)
                object_score = torch.full((B, 1), self.OBJECT_SCORE_VALUE, device=lowres.device, dtype=lowres.dtype)

                memory_encoding = self.memory_encoder(
                    lowres_image_encoding=lowres,
                    mask_prediction=mask_at_decoder,
                    object_score=object_score,
                    is_prompt_encoding=True)
                tokens = memory_encoding.float().permute(0, 2, 3, 1).reshape(B, H * W, memory_encoding.shape[1])
                token_chunks.append(tokens)

                patch_mask = F.interpolate(chunk_masks, size=(H, W), mode="nearest")
                patch_mask_chunks.append((patch_mask > 0).flatten(2).squeeze(1))

        memory_tokens = torch.cat(token_chunks, dim=0)
        patch_masks = torch.cat(patch_mask_chunks, dim=0)
        return memory_tokens, patch_masks

    def split_foreground_background(self, patch_tokens, patch_masks):
        """Split raw patch tokens into foreground/background views via the -5
        padding sentinel."""

        patch_masks = patch_masks.bool().unsqueeze(-1)
        padding_value = torch.tensor(-5, device=patch_tokens.device, dtype=patch_tokens.dtype)
        foreground = torch.where(patch_masks, patch_tokens, padding_value)
        background = torch.where(~patch_masks, patch_tokens, padding_value)
        return foreground, background

    def extract_patch_tokens(self, cropped_frames, cropped_masks, encoder_chunk_size=16):
        """Dispatch to either the Hiera or Memory patch-token extractor based on
        `token_source`, then split into foreground/background views. Lets the same
        model class drive both calibrator variants at deploy time without subclassing."""

        if self.token_source == "memory":
            patch_tokens, patch_masks = self.extract_memory_patch_tokens(cropped_frames, cropped_masks)
        else:
            patch_tokens, patch_masks = self.extract_raw_patch_tokens(cropped_frames, cropped_masks, encoder_chunk_size)
        return self.split_foreground_background(patch_tokens, patch_masks)

    # -------------------------------------------------------------------------------
    # Patch similarity
    # -------------------------------------------------------------------------------

    def compute_patch_similarities(self, reference_tokens, target_tokens, target_chunk_size=16):
        """Per (target patch) best-match cosine similarity to the reference's FG patches
        (argmax over reference patches). Both token sets are L2-normalized; padding-
        sentinel patches (rows of -5) are masked to -inf so they never win the max."""

        ref_valid = ~(reference_tokens == -5).all(dim=-1)
        ref_mask = ref_valid[:, None, :, None]
        all_ref_invalid = ~ref_valid.any(dim=-1)
        reference_normalized = torch.nn.functional.normalize(reference_tokens, dim=-1)

        n_targets = target_tokens.shape[0]
        chunks = []
        for chunk_start in range(0, n_targets, target_chunk_size):
            target_chunk = target_tokens[chunk_start:chunk_start + target_chunk_size]
            target_valid = ~(target_chunk == -5).all(dim=-1)
            target_normalized = torch.nn.functional.normalize(target_chunk, dim=-1)

            similarities = torch.einsum("rpd,tqd->rtpq", reference_normalized, target_normalized)
            vectors_chunk = similarities.masked_fill(~ref_mask, float("-inf")).amax(dim=2)

            if all_ref_invalid.any():
                vectors_chunk[all_ref_invalid] = 0.0

            target_mask = target_valid[None, :, :]
            vectors_chunk = vectors_chunk * target_mask.to(vectors_chunk.dtype)

            chunks.append(vectors_chunk)

        vectors = torch.cat(chunks, dim=1)
        return vectors.float().cpu().numpy().transpose(1, 0, 2)
