import torch
import numpy as np
import cv2
from numpy.typing import NDArray

from sam_3.sam_v3_model import SAMV3TrackingModel
from sam_3.make_sam_v3 import make_samv3_from_original_state_dict
from src.modules.calibrator_scoring import CalibratorScoring
from src.offline_training.dataset_encoders import load_dataset_encoders


class SamaraModel3(CalibratorScoring, SAMV3TrackingModel):
    """SAMARA's calibrator on a SAM 3 backbone -- the SAM 3 twin of `SamaraModel`.

    Same intervention, same calibrator, different tracker underneath. The gate scores Perception Encoder
    tokens (see `CalibratorScoring`), never SAM's own features, so swapping SAM 2 for SAM 3 changes what
    is being gated without changing the gate. That is what makes SAMARA-2 vs SAMARA-3 a backbone
    comparison rather than two unrelated systems.

    THREE THINGS DIFFER FROM THE SAM 2 MODEL, all in the plumbing rather than the idea:

      * `propose_masks` comes from SAM 3's own `step_video_masking(return_best_only=False)`. SAM 2 had no
        such flag, so `sam_2/sam_v2_model.py` had to reimplement the step to get at the unselected
        proposals; here the upstream call already returns them.
      * A memory entry is ONE object. SAM 3's `encode_frame_memory` returns a `SAMv3MemoryEncoding` that
        carries its own object pointer, where SAM 2 returned an (encoding, pointer) pair. `commit_candidate`
        still returns a 3-tuple so the controller reads the same either way; the middle slot is None.
      * The image encoding is passed around whole. SAM 2's memory encoder wanted `lowres_imgenc` alone;
        SAM 3's wants the full `SAMv3ImageEncoding`, so that is what travels in that argument slot.

    The detection branch of SAM 3 is stripped from `sam_3/` -- it is a MOT/open-vocabulary head and this is
    single-object tracking -- which is also why the model is built from the tracking components directly
    rather than through `SAMV3Core.get_tracking_context()`."""

    def __init__(
        self,
        sam_model_path: str | None = None,
        controller: torch.nn.Module | None = None,
        crop_resize: int | None = None,
        pad_ratio: float = 0.25,
        feature_encoder: str = "perception",
        max_side_length: int = 1008,
        weights_only: bool = False,
    ):
        if not feature_encoder:
            raise ValueError("feature_encoder is required: the calibrator scores external tokens, not SAM's")

        # The FULL official checkpoint, not a tracker-only slice: `get_model_config_from_state_dict` reads
        # the detector tensors to infer sizings (e.g. features_per_detection_token, which the image
        # projection needs) even though the detector itself is never built.
        core = make_samv3_from_original_state_dict(sam_model_path, strict_load=True, weights_only=weights_only)
        SAMV3TrackingModel.__init__(
            self,
            image_encoder_model=core.image_encoder,
            image_projection_model=core.image_projection,
            coordinate_encoder_model=core.coordinate_encoder,
            prompt_encoder_model=core.prompt_encoder,
            mask_decoder_model=core.mask_decoder,
            memory_encoder_model=core.memory_encoder,
            memory_image_fusion_model=core.memory_image_fusion)

        self.controller = controller
        self.gate_controller = None
        self.crop_resize = crop_resize
        self.pad_ratio = pad_ratio
        self.anchor_size_pixels = 0
        self.max_side_length = max_side_length

        self.feature_encoder = feature_encoder
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._feature_token_fn = load_dataset_encoders([feature_encoder], device, dtype)[feature_encoder]
        self.eval()

    # -------------------------------------------------------------------------------
    # The two hooks the controller drives
    # -------------------------------------------------------------------------------

    def encode_image(self, image_bgr, max_side_length=None, use_square_sizing=True):
        """`SAMV3TrackingModel.encode_image` with this model's configured resolution as the default."""

        side = self.max_side_length if max_side_length is None else max_side_length
        return SAMV3TrackingModel.encode_image(self, image_bgr, side, use_square_sizing)

    def initialize_video_masking(self, encoded_image, box_tlbr_norm_list, fg_xy_norm_list=[], bg_xy_norm_list=[],
                                 mask_hint=None, mask_index_select=None):
        """SAM 2's box-prompt entry point, on SAM 3. Returns `(mask, memory_encoding, pointer)`.

        The dataset pipeline calls this in three places -- the pseudo-ground-truth used to label proposals,
        the distractor injected by the corruption sweep, and the memory seed -- all of them through SAM 2's
        signature. Keeping that signature means `MemoryOracle`, `MaskOracle` and `dataset_labels` work on
        either generation without branching, which is what makes a SAM 3 dataset comparable to the SAM 2 one
        rather than a separate pipeline that happens to produce similar files.

        The pointer comes back None: SAM 3 folds it into the memory encoding. Nothing in the labelling path
        reads it (`_pseudo_ground_truth` discards it), and `MainMemoryV3.update_memory` ignores it."""

        mask, memory_encoding = self.encode_prompt_memory(
            encoded_image, box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list,
            mask_hint=mask_hint, mask_index=mask_index_select)
        return mask, memory_encoding, None

    def propose_masks(self, current_frame: NDArray[np.uint8], main_memory, encoded_image=None):
        """Run one memory-conditioned step and return EVERY candidate mask, without SAM's argmax selection.

        Returns `(mask_preds, iou_scores, object_pointers, object_score, encoded_image, encoded_image)`,
        mirroring the SAM 2 signature so `SamaraController`-style callers need no branching. The last two
        slots are the same object: SAM 2 returned `lowres_imgenc` for the memory encoder and the feature
        list for caching, and SAM 3's memory encoder takes the whole encoding."""

        bgr_current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)
        if encoded_image is None:
            encoded_image = self.encode_image(bgr_current_frame)

        mask_preds, iou_scores, object_pointers, object_score = self.step_video_masking(
            encoded_image,
            prompt_memory_encodings=list(main_memory.prompt_memories),
            frame_memory_encodings=list(main_memory.frame_memories),
            return_best_only=False,
            is_recent_first=True)

        return mask_preds, iou_scores, object_pointers, object_score, encoded_image, encoded_image

    def commit_candidate(self, mask_preds, best_idx, object_pointers, object_score, encoded_image):
        """Build `(chosen_mask_logits_cpu, None, memory_encoding)` for one chosen candidate index.

        The pointer slot is None by design: SAM 3 folds the object pointer into the memory encoding, and
        `MainMemoryV3.update_memory` ignores the argument. Returning a 3-tuple anyway keeps the controller
        identical across backbones."""

        chosen_mask = mask_preds[:, best_idx:best_idx + 1, :, :]     # (B, 1, H, W), as the encoder expects
        memory_encoding = self.encode_frame_memory(
            encoded_image, mask_preds, object_pointers, object_score, mask_index=best_idx)

        return chosen_mask.squeeze().to(torch.float64).cpu(), None, memory_encoding
