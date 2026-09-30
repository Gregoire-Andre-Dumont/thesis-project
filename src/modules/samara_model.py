import cv2
import torch
import numpy as np
import torch.nn.functional as F
from numpy.typing import NDArray

from sam_2.sam_v2_model import SAMV2Model
from src.modules.calibrator_scoring import CalibratorScoring
from sam_2.make_sam_v2 import make_samv2_from_state_dict
from src.offline_training.dataset_encoders import load_dataset_encoders

class SamaraModel(CalibratorScoring, SAMV2Model):
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

