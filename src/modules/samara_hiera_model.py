import cv2
import torch
import numpy as np
import torch.nn.functional as F
from numpy.typing import NDArray

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
        feature_encoder: str | None = None,
    ):
        if token_source not in {"hiera", "memory"}:
            raise ValueError(f"token_source must be 'hiera' or 'memory', got {token_source!r}")

        _, sam_model = make_samv2_from_state_dict(sam_model_path)
        super().__init__(
            image_encoder_model=sam_model.image_encoder,
            coordinate_encoder=sam_model.coordinate_encoder,
            prompt_encoder_model=sam_model.prompt_encoder,
            mask_decoder_model=sam_model.mask_decoder,
            memory_encoder_model=sam_model.memory_encoder,
            memory_fusion_model=sam_model.memory_fusion)

        self.controller = controller
        self.crop_resize = crop_resize
        self.pad_ratio = pad_ratio
        self.token_source = token_source
        self.anchor_size_pixels = 0

        # Calibrator features come from an external ~80M backbone (perception/dino/hiera_sam/hiera_mae)
        # when `feature_encoder` is set, so deployment matches the dataset the controller trained on.
        # SAM's own Hiera still drives tracking/mask selection. None -> use SAM Hiera/memory tokens.
        self.feature_encoder = feature_encoder
        self._feature_token_fn = None
        if feature_encoder is not None:
            from src.offline_training.dataset_encoders import load_dataset_encoders
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            self._feature_token_fn = load_dataset_encoders([feature_encoder], device, dtype)[feature_encoder]
        self.eval()

    # -------------------------------------------------------------------------------
    # Calibrator-driven mask scoring
    # -------------------------------------------------------------------------------

    def _score_masks(self, current_frame, candidate_masks_raw, reference_foreground, reference_background):
        """Crop each candidate mask and build its anchor-similarity features.
        Returns the calibrator's IoU predictions plus the tokens and features."""

        candidate_masks = candidate_masks_raw.to(torch.float64).cpu().numpy()   # keep logits; extract_crops thresholds
        frames = np.repeat(np.asarray(current_frame)[None], len(candidate_masks), axis=0)

        cropped_frames, cropped_masks = self.extract_crops(frames, candidate_masks)
        foreground, background = self.extract_patch_tokens(cropped_frames, cropped_masks)

        n_masks = len(candidate_masks)
        side = int(round(foreground.shape[1] ** 0.5))
        fg_fg = self.compute_patch_similarities(reference_foreground, foreground).reshape(n_masks, -1, side, side)
        bg_fg = self.compute_patch_similarities(reference_foreground, background).reshape(n_masks, -1, side, side)

        features = torch.from_numpy(np.stack([fg_fg, bg_fg], axis=-1).astype(np.float32)).to("cuda")
        return self._run_controller(features), foreground, background, features

    def _run_controller(self, features):
        """Run the calibrator on the features and return per-mask IoU probabilities.
        Autocast is disabled so a bf16 context can't corrupt the float32 calibrator."""

        with torch.autocast("cuda", enabled=False):
            logits = self.controller(features.float())
        return torch.sigmoid(logits)[:, 0].cpu().numpy()

    @torch.inference_mode()
    def select_best_mask_gated(self, current_frame, main_memory, reference_foreground, reference_background):
        """Pick SAM 2's highest-IoU candidate and score it with the calibrator for the
        memory-commit gate. Selection is plain SAM argmax, with no SAMARA re-ranking."""

        chosen_mask, chosen_pointer, chosen_encoding, object_score, iou_score, _, _, _ = self.select_best_mask(
            current_frame, main_memory)
        samara_ious, foreground, background, calibrator_features = self._score_masks(
            current_frame, chosen_mask[None], reference_foreground, reference_background)

        return {
            "chosen_mask": chosen_mask,
            "pointer": chosen_pointer,
            "encoding": chosen_encoding,
            "iou_score": float(iou_score),
            "samara_iou": float(samara_ious[0]),
            "object_score": float(object_score),
            "target_foreground": foreground[0],
            "target_background": background[0],
            "calibrator_features": calibrator_features}

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
    # Patch-token extraction (Hiera or Memory) + foreground/background split
    # -------------------------------------------------------------------------------

    def _encode_crops(self, crops):
        """Convert the RGB crops to the encoder's input format and run the image encoder.
        Returns the low-resolution feature map for the batch of crops."""

        prepared = [
            self.image_encoder.prepare_image(cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2BGR), self.crop_resize, True)
            for crop in crops]
        return self.image_encoder(torch.cat(prepared, dim=0))[0]

    def extract_raw_patch_tokens(self, cropped_frames, cropped_masks, encoder_chunk_size=16):
        """Encode the crops and flatten the image-encoder output into per-patch tokens.
        Also returns the foreground mask downsampled to the patch grid."""

        with torch.inference_mode():
            feats = torch.cat([
                self._encode_crops(cropped_frames[start:start + encoder_chunk_size])
                for start in range(0, len(cropped_frames), encoder_chunk_size)], dim=0).float()

        B, F_dim, H, W = feats.shape
        patch_tokens = feats.permute(0, 2, 3, 1).reshape(B, H * W, F_dim)
        masks = torch.as_tensor(cropped_masks, device=feats.device).unsqueeze(1)
        patch_masks = (F.interpolate(masks, size=(H, W), mode="nearest") > 0).flatten(2).squeeze(1)
        return patch_tokens, patch_masks

    def extract_memory_patch_tokens(self, cropped_frames, cropped_masks, encoder_chunk_size=8):
        """Encode the crops and pass them through the memory encoder with their masks.
        Returns the mask-conditioned memory tokens and the patch-grid foreground mask."""

        token_chunks, mask_chunks = [], []
        with torch.inference_mode():
            for start in range(0, len(cropped_frames), encoder_chunk_size):
                lowres = self._encode_crops(cropped_frames[start:start + encoder_chunk_size])
                B, _, H, W = lowres.shape
                masks = torch.as_tensor(cropped_masks[start:start + encoder_chunk_size],
                                        device=lowres.device, dtype=lowres.dtype).unsqueeze(1)

                memory_encoding = self.memory_encoder(
                    lowres_image_encoding=lowres,
                    mask_prediction=F.interpolate(masks, size=(4 * H, 4 * W), mode="bilinear", align_corners=False),
                    object_score=torch.full((B, 1), self.OBJECT_SCORE_VALUE, device=lowres.device, dtype=lowres.dtype),
                    is_prompt_encoding=True)

                token_chunks.append(memory_encoding.float().permute(0, 2, 3, 1).reshape(B, H * W, memory_encoding.shape[1]))
                mask_chunks.append((F.interpolate(masks, size=(H, W), mode="nearest") > 0).flatten(2).squeeze(1))

        return torch.cat(token_chunks, dim=0), torch.cat(mask_chunks, dim=0)

    def split_foreground_background(self, patch_tokens, patch_masks):
        """Split patch tokens into foreground and background views using the patch mask.
        Masked-out patches are filled with a -5 sentinel so they can be ignored later."""

        patch_masks = patch_masks.bool().unsqueeze(-1)
        padding_value = torch.tensor(-5, device=patch_tokens.device, dtype=patch_tokens.dtype)
        foreground = torch.where(patch_masks, patch_tokens, padding_value)
        background = torch.where(~patch_masks, patch_tokens, padding_value)
        return foreground, background

    def extract_patch_tokens(self, cropped_frames, cropped_masks, encoder_chunk_size=16):
        """Extract the calibrator's patch tokens for the crops and return their foreground/background
        views. Uses the external ~80M `feature_encoder` (32x32 grid) when set -- the same tokens the
        dataset was built with -- otherwise SAM's Hiera or memory encoder per `token_source`."""

        if self._feature_token_fn is not None:
            from src.offline_training.dataset_encoders import encode_tokens, _patch_masks
            tokens = encode_tokens(self._feature_token_fn, np.asarray(cropped_frames), encoder_chunk_size)
            patch_masks = _patch_masks(np.asarray(cropped_masks, dtype=np.float32), tokens.device)
            return self.split_foreground_background(tokens, patch_masks)

        if self.token_source == "memory":
            patch_tokens, patch_masks = self.extract_memory_patch_tokens(cropped_frames, cropped_masks)
        else:
            patch_tokens, patch_masks = self.extract_raw_patch_tokens(cropped_frames, cropped_masks, encoder_chunk_size)
        return self.split_foreground_background(patch_tokens, patch_masks)

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
