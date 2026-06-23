import torch
import cv2
import numpy as np
from numpy.typing import NDArray
import torch.nn.functional as F
from muggled_sam.sam_v2_model import SAMV2Model
from muggled_sam.make_sam_v2 import make_samv2_from_state_dict


class SamaraHieraModel(SAMV2Model):
    """SAM 2 with multi-reference feature extraction for SAMARA. Crops with a fixed
    amodal-floor square window when `set_amodal_anchor` has been called (new offline
    pipeline); otherwise falls back to a mask-bbox-with-padding crop (existing runtime
    trackers). Patch tokens are SAM 2's Hiera image-encoder lowres features."""

    def __init__(
        self,
        sam_model_path: str | None = None,
        padding_size: int | None = None,
        controller: torch.nn.Module | None = None,
        crop_resize: int | None = None,
        n_references: int = 9,
        pad_ratio: float = 0.25,
        token_source: str = "hiera",
    ):
        if token_source not in {"hiera", "memory"}:
            raise ValueError(f"token_source must be 'hiera' or 'memory', got {token_source!r}")
        _, sam_model = make_samv2_from_state_dict(sam_model_path)
        self.stochastic_policy = True
        self.padding_size = padding_size
        self.crop_resize = crop_resize
        self.n_references = n_references
        self.pad_ratio = pad_ratio
        self.token_source = token_source
        self.first_amodal_bbox_norm = None
        self.first_frame_height = None
        self.first_frame_width = None

        super().__init__(
            image_encoder_model = sam_model.image_encoder,
            coordinate_encoder = sam_model.coordinate_encoder,
            prompt_encoder_model = sam_model.prompt_encoder,
            mask_decoder_model = sam_model.mask_decoder,
            memory_encoder_model = sam_model.memory_encoder,
            memory_fusion_model = sam_model.memory_fusion)

        # Assigned AFTER super().__init__() because the controller can be a real
        # nn.Module (e.g., a heuristic instantiated directly from a tracker YAML), and
        # nn.Module's __setattr__ requires Module.__init__() to have already run.
        self.controller = controller
        self.eval()

    def set_amodal_anchor(self, amodal_bbox_norm, frame_height, frame_width):
        """Configure the fixed amodal-floor crop side for the current trajectory. Once
        set, `extract_crops` builds square crops of side
        `max(amodal_w, amodal_h) * (1 + 2*pad_ratio)`, centered on each frame's
        predicted-mask centroid. Call once per trajectory, before encoding."""

        self.first_amodal_bbox_norm = tuple(float(v) for v in amodal_bbox_norm)
        self.first_frame_height = int(frame_height)
        self.first_frame_width = int(frame_width)

    def _score_masks(self, current_frame, candidate_masks_raw, reference_foreground):
        """Run the calibrator on a batch of candidate masks. Returns `(samara_ious,
        foreground, background, features)` where `features` is the patch-similarity tensor
        already shaped for the controller — callers can use it to run an alternate
        controller without recomputing similarities. A controller that is None produces a
        NaN-filled prediction array, so a downstream gate can detect "no signal" explicitly."""

        candidate_masks = (candidate_masks_raw > 0.0).to(torch.float64).cpu().numpy()
        frames = np.repeat(np.asarray(current_frame)[None], len(candidate_masks), axis=0)
        cropped_frames, cropped_masks = self.extract_crops(frames, candidate_masks)
        foreground, background = self.extract_patch_tokens(cropped_frames, cropped_masks)

        side = int(round(foreground.shape[1] ** 0.5))
        foreground_similarities = self.compute_patch_similarities(reference_foreground, foreground).reshape(len(candidate_masks), -1, side, side)
        background_similarities = self.compute_patch_similarities(reference_foreground, background).reshape(len(candidate_masks), -1, side, side)
        features = torch.from_numpy(np.stack([foreground_similarities, background_similarities], axis=-1).astype(np.float32)).to("cuda")

        samara_ious = self._run_controller(self.controller, features)
        return samara_ious, foreground, background, features

    def _run_controller(self, controller, features):
        """Run the calibrator and return its predictions as a `(N,)` numpy array.

        The calibrator is trained as a binary classifier (`BCEWithLogitsLoss` on `IoU > t` where
        `t` is `MainDataset.iou_threshold`). The model outputs raw logits; we apply sigmoid here
        so callers can compare the result against a confidence threshold in `[0, 1]` (the
        tracker's `iou_threshold` field).

        Disables autocast so a bf16-leaked context can't collapse the float32 calibrator
        weights. Returns a NaN-filled array of the right length when no controller is attached."""

        if controller is None:
            return np.full(features.shape[0], np.nan, dtype=np.float32)
        with torch.autocast("cuda", enabled=False):
            output = controller(features.float())
            probabilities = torch.sigmoid(output)
        return probabilities[:, 0].cpu().numpy()

    @torch.inference_mode()
    def select_best_mask_gated(self, current_frame, main_memory, reference_foreground, reference_background):
        """Pick SAM 2's highest-IoU candidate mask and score that mask with the moving calibrator for
        the memory-commit gate. The calibrator does NOT influence which mask is selected — selection
        is pure SAM-argmax."""

        encoded_image_features_list, _, _ = self.encode_image(cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR))
        mask_preds, iou_scores, object_pointers, object_score, _, _ = self.step_video_masking(
            main_memory=main_memory, encoded_image_features_list=encoded_image_features_list)

        sam_ious = iou_scores.squeeze(0)[1:4].float().cpu().numpy()
        chosen = int(np.argmax(sam_ious))
        samara_ious, foreground, background, calibrator_features = self._score_masks(
            current_frame, mask_preds[:, 1 + chosen:2 + chosen, :, :].squeeze(0), reference_foreground)

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
            # Patch-similarity tensor consumed by the (current) controller — exposed so a
            # caller can run a SECOND controller on the same features without paying for
            # a recompute (e.g. SamaraMoving runs both moving + fixed calibrators).
            "calibrator_features": calibrator_features}

    def extract_crops(self, frames: NDArray[np.uint8], masks: NDArray[np.uint8]):
        """Square crops at `crop_resize`. If `set_amodal_anchor` has configured a fixed
        amodal-floor crop side, every crop reuses that side and is centered on the
        predicted-mask centroid (shifted inward at frame edges so it stays square).
        Otherwise falls back to a mask-bbox crop expanded by `padding_size` and squared."""

        crop_size = self.crop_resize
        cropped_frames = np.zeros((masks.shape[0], crop_size, crop_size, 3), dtype=np.float32)
        cropped_masks = np.zeros((masks.shape[0], crop_size, crop_size), dtype=np.float32)

        fixed_crop_side = self._fixed_crop_side()

        for idx, (frame, mask) in enumerate(zip(frames, masks)):
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            coordinates = cv2.findNonZero(mask.astype(np.uint8))
            if coordinates is None:
                continue

            x, y, width, height = cv2.boundingRect(coordinates)
            if fixed_crop_side is not None:
                x_min, y_min, x_max, y_max = self._amodal_floor_window(
                    x, y, width, height, fixed_crop_side, frame.shape[:2])
            else:
                x_min, y_min, x_max, y_max = self._padded_square_window(
                    x, y, width, height, frame.shape[:2])

            crop_frame = frame[y_min:y_max, x_min:x_max]
            crop_mask = mask[y_min:y_max, x_min:x_max]
            cropped_frames[idx] = cv2.resize(crop_frame, (crop_size, crop_size), interpolation=cv2.INTER_CUBIC)
            cropped_masks[idx] = cv2.resize(crop_mask, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)

        return cropped_frames, cropped_masks

    def _fixed_crop_side(self):
        """Crop side in pixels when an amodal anchor is configured, else `None`."""

        if self.first_amodal_bbox_norm is None:
            return None
        _, _, amodal_w_norm, amodal_h_norm = self.first_amodal_bbox_norm
        amodal_w_px = amodal_w_norm * self.first_frame_width
        amodal_h_px = amodal_h_norm * self.first_frame_height
        return int(round(max(amodal_w_px, amodal_h_px, 1.0) * (1 + 2 * self.pad_ratio)))

    @staticmethod
    def _amodal_floor_window(x, y, width, height, fixed_side, frame_shape):
        """Square `(x_min, y_min, x_max, y_max)` of side `fixed_side`, centered on the
        mask-bbox centroid, shifted inward when it would extend past the frame."""

        frame_height, frame_width = frame_shape
        side = min(fixed_side, frame_height, frame_width)
        center_x = x + width // 2
        center_y = y + height // 2
        half = side // 2
        x_min = center_x - half
        y_min = center_y - half
        x_max = x_min + side
        y_max = y_min + side
        if x_min < 0:
            x_max -= x_min; x_min = 0
        elif x_max > frame_width:
            x_min -= (x_max - frame_width); x_max = frame_width
        if y_min < 0:
            y_max -= y_min; y_min = 0
        elif y_max > frame_height:
            y_min -= (y_max - frame_height); y_max = frame_height
        return int(x_min), int(y_min), int(x_max), int(y_max)

    def _padded_square_window(self, x, y, width, height, frame_shape):
        """Square `(x_min, y_min, x_max, y_max)` derived from the mask bbox expanded by
        `padding_size`, then made square by extending the shorter side equally on both
        ends. The legacy crop used by every runtime tracker."""

        frame_height, frame_width = frame_shape
        x_min = max(0, x - self.padding_size)
        y_min = max(0, y - self.padding_size)
        x_max = min(frame_width, x + width + self.padding_size)
        y_max = min(frame_height, y + height + self.padding_size)

        bounding_box_width = x_max - x_min
        bounding_box_height = y_max - y_min
        difference = abs(bounding_box_width - bounding_box_height)
        if bounding_box_width < bounding_box_height:
            x_min = max(0, x_min - difference // 2)
            x_max = min(frame_width, x_max + (difference - difference // 2))
        elif bounding_box_height < bounding_box_width:
            y_min = max(0, y_min - difference // 2)
            y_max = min(frame_height, y_max + (difference - difference // 2))
        return x_min, y_min, x_max, y_max

    OBJECT_SCORE_VALUE = 100.0

    def extract_raw_patch_tokens(self, cropped_frames, cropped_masks, encoder_chunk_size=16):
        """Encode cropped frames into raw patch tokens and the patch-grid foreground mask.

        Returns (patch_tokens, patch_masks): patch_tokens (B, n_patches, feature_dim) are the raw
        SAM 2 image-encoder tokens; patch_masks (B, n_patches) is the predicted mask downsampled to
        the patch grid. The foreground/background split is deferred to `split_foreground_background`."""

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

    def split_foreground_background(self, patch_tokens, patch_masks):
        """Split raw patch tokens into foreground/background views via the -5 padding sentinel."""

        patch_masks = patch_masks.bool().unsqueeze(-1)
        padding_value = torch.tensor(-5, device=patch_tokens.device, dtype=patch_tokens.dtype)
        foreground = torch.where(patch_masks, patch_tokens, padding_value)
        background = torch.where(~patch_masks, patch_tokens, padding_value)
        return foreground, background

    def extract_patch_tokens(self, cropped_frames, cropped_masks, encoder_chunk_size=16):
        """Extract foreground/background patch-token views of the cropped frames. Routes
        through `extract_raw_patch_tokens` (Hiera image tokens) when
        `token_source == 'hiera'` and through `extract_memory_patch_tokens` (mask-
        conditioned memory tokens) when `token_source == 'memory'`. The dispatch lets the
        same model class drive both the Hiera-trained and memory-trained calibrators at
        deploy time without subclassing."""

        if self.token_source == "memory":
            patch_tokens, patch_masks = self.extract_memory_patch_tokens(cropped_frames, cropped_masks)
        else:
            patch_tokens, patch_masks = self.extract_raw_patch_tokens(cropped_frames, cropped_masks, encoder_chunk_size)
        return self.split_foreground_background(patch_tokens, patch_masks)

    def extract_memory_patch_tokens(self, cropped_frames, cropped_masks, encoder_chunk_size=8):
        """Encode each crop with the image encoder, then the memory encoder against the crop's mask.
        Returns (memory_tokens, patch_masks) with the same shape semantics as
        `extract_raw_patch_tokens` so `split_foreground_background` applies unchanged. Memory
        tokens have `lowres_channels / 4` features and are mask-conditioned — the representation
        SAM 2 cross-attends over at deployment."""

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

    def compute_patch_similarities(self, reference_tokens, target_tokens, target_chunk_size=16):
        """Per (reference, target patch) BEST-MATCH cosine similarity. Both token sets are
        L2-normalized so each pairwise score is bounded in [-1, 1], and we take the MAX over the
        reference's patches. No temperature, no soft attention — invariant to padding count, and
        unaffected by adding extra non-argmax valid ref patches."""

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
