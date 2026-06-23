import torch
import ipdb
import torch.nn as nn
import numpy as np
import cv2

# For type hints
from torch import Tensor
from numpy import ndarray
from .image_encoder_model import SAMV2ImageEncoder
from .coordinate_encoder_model import SAMV2CoordinateEncoder
from .prompt_encoder_model import SAMV2PromptEncoder
from .mask_decoder_model import SAMV2MaskDecoder
from .memory_encoder_model import SAMV2MemoryEncoder
from .memory_fusion_model import SAMV2MemoryFusion
from src.modules.memories.main_memory import MainMemory
from numpy.typing import NDArray
from src.utils.compute_iou import compute_iou


class SAMV2Model(nn.Module):
    """
    Wrapper around separated SAMV2 model components, so that the model can be used as a singular entity
    """

    # .................................................................................................................

    def __init__(
        self,
        image_encoder_model: SAMV2ImageEncoder,
        coordinate_encoder: SAMV2CoordinateEncoder,
        prompt_encoder_model: SAMV2PromptEncoder,
        mask_decoder_model: SAMV2MaskDecoder,
        memory_encoder_model: SAMV2MemoryEncoder,
        memory_fusion_model: SAMV2MemoryFusion,
    ):

        # Inherit from parent
        super().__init__()

        # Store SAM model components
        self.image_encoder = image_encoder_model
        self.coordinate_encoder = coordinate_encoder
        self.prompt_encoder = prompt_encoder_model
        self.mask_decoder = mask_decoder_model
        self.memory_encoder = memory_encoder_model
        self.memory_fusion = memory_fusion_model

        # Default to eval mode, expecting to use inference only
        self.eval()

    # .................................................................................................................

    def forward(
        self,
        image_rgb_normalized_bchw: Tensor,
        boxes_tensor: Tensor,
        fg_tensor: Tensor,
        bg_tensor: Tensor,
        mask_hint: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        All image/mask generating code of SAMV2 model, bundled into a single function.
        Takes an image and set of prompts and produces several candidate segmentation masks.

        Note that in practice, it makes more sense to call the component pieces of the model,
        rather than using this function so that image & prompt encoding can happen independently.
        See the 'encode_prompts', 'encode_image' and 'generate_masks' functions for more info
        """

        # Encode prompts & image inputs
        box_posenc, fg_posenc, bg_posenc = self.coordinate_encoder(boxes_tensor, fg_tensor, bg_tensor)
        encoded_prompts = self.prompt_encoder(box_posenc, fg_posenc, bg_posenc)
        encoded_image = self.image_encoder(image_rgb_normalized_bchw)

        # Combine encodings to generate mask output
        patch_grid_hw = encoded_image.shape[2:]
        grid_posenc = self.coordinate_encoder.get_grid_position_encoding(patch_grid_hw)
        mask_preds, iou_preds, objscore_pred, mask_tokens_out, _, _ = self.mask_decoder(
            encoded_image, encoded_prompts, grid_posenc, mask_hint)

        return mask_preds, iou_preds, objscore_pred, mask_tokens_out


    def encode_prompts(self, box_tlbr_norm_list: list = [], fg_xy_norm_list: list = [], bg_xy_norm_list: list = []) -> Tensor:
        """Function used to encode prompt coordinates. Inputs should be given as lists
        of prompts. The length of each list does not need to match. Enter either
        None or an empty list ([]) to disable any of the prompts"""

        with torch.inference_mode():
            boxes_tensor = self.coordinate_encoder.prepare_boxes(box_tlbr_norm_list)
            fg_tensor, bg_tensor = self.coordinate_encoder.prepare_points(fg_xy_norm_list, bg_xy_norm_list)
            box_posenc, fg_posenc, bg_posenc = self.coordinate_encoder(boxes_tensor, fg_tensor, bg_tensor)
            encoded_prompts = self.prompt_encoder(box_posenc, fg_posenc, bg_posenc)

        return encoded_prompts
    
    def encode_image(
        self,
        image_bgr: ndarray,
        max_side_length=1024,
        use_square_sizing=True,
    ) -> tuple[list[Tensor], tuple[int, int], tuple[int, int]]:
        """
        Function used to compute image encodings from a bgr formatted image (e.g. loaded from opencv)
        The max_side_length setting is used to set the size at which the image is processed,
        while the use_square_sizing determines whether the image is scaled to a square resolution
        or scaled (to the max_side_length) based on it's original aspect ratio.

        Returns:
            encoded_images_list, patch_grid_hw, preencoded_image_hw
            -> Encoded images list contains 3 multi-resolution feature maps
               they have shapes: Bx256x64x64, Bx64x128x128, Bx32x256x256
               (using default settings). The first-most feature map is
               the 'low-res' map needed by several other parts of the model
            -> The patch_grid_hw contains the height & width of the low-res
               feature map (64x64 with default 1024x1024 input sizing)
            -> The preencoded_image_hw contains the height & width of the
               input image after pre-processing, just before being encoded
               by default it would be 1024x1024
        """

        with torch.inference_mode():
            image_rgb_normalized_bchw = self.image_encoder.prepare_image(image_bgr, max_side_length, use_square_sizing)
            image_preenc_hw = image_rgb_normalized_bchw.shape[2:]
            encoded_image_features_list = self.image_encoder(image_rgb_normalized_bchw)

        # Get patch sizing of lowest-res tokens (as needed by other components)
        patch_grid_hw = encoded_image_features_list[0].shape[2:]

        return encoded_image_features_list, patch_grid_hw, image_preenc_hw

    # .................................................................................................................

    def generate_masks(
        self,
        encoded_image_features_list: list[Tensor],
        encoded_prompts: Tensor,
        mask_hint: Tensor | int | None = None,
        blank_promptless_output: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """
        Function used to generate segmentation masks given an image encoding,
        as well as a prompt encoding and potentially a mask hint/prompt. These
        input encodings are expected to come from other model components.

        The mask hint can either be None (no mask input), an integer or a
        tensor. If an integer is given, this is interpreted to mean that the
        model should run twice, once to get a set of mask predictions and then
        a second time, where the mask_hint (as integer) mask is chosen to be
        used as a mask hint for a second run of the model. The idea being to
        use the models own mask output to refine itself. If a tensor is given,
        it is assumed to be a mask itself. It should be shaped to match the
        model's own output masks for the given input image size, by default
        this would be a shape of: Bx1x256x256

        Returns:
            mask_predictions, iou_predictions
            -> Masks have shape: Bx4xHxW (HxW is 256x256 using default settings)
            -> iou_predictions have shape: Bx4
        """

        # Get sizing of the lowest-resolution image encoding
        patch_grid_hw = encoded_image_features_list[0].shape[2:]

        with torch.inference_mode():
            grid_posenc = self.coordinate_encoder.get_grid_position_encoding(patch_grid_hw)
            mask_preds, iou_preds, obj_ptrs, obj_score, _, _ = self.mask_decoder(
                encoded_image_features_list, encoded_prompts, grid_posenc, mask_hint, blank_promptless_output
            )

        return mask_preds, iou_preds

    # .................................................................................................................

    def initialize_video_masking(
        self,
        encoded_image_features_list: list[Tensor],
        box_tlbr_norm_list: list,
        fg_xy_norm_list: list = [],
        bg_xy_norm_list: list = [],
        mask_hint: Tensor | int | None = None,
        mask_index_select: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Creates initial 'prompt memory' for segmenting objects through a video.
        Similar to calling the prompt encoder & mask decoder, however, this function
        outputs a single mask prediction along with a memory encoding and
        object pointer, both of which must be passed along to the per-frame video masking function.=
        If a 'mask_index_select' isn't given, then the 'best' mask will be chosen automatically
        """

        # Encode initial prompts
        encoded_prompts = self.encode_prompts(box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list)

        with torch.inference_mode():

            # For convenience
            lowres_imgenc, *hires_imgenc = encoded_image_features_list
            token_hw = lowres_imgenc.shape[2:]

            # Generate mask prediction from image/prompt encodings, as usual
            grid_posenc = self.coordinate_encoder.get_grid_position_encoding(token_hw)
            mask_preds, iou_preds, obj_ptrs, obj_score, _, _ = self.mask_decoder(
                encoded_image_features_list,
                encoded_prompts,
                grid_posenc,
                mask_hint=mask_hint,
                blank_promptless_output=False,
            )

            # Select mask to use for initial encodings (auto-select if not given an index)
            if mask_index_select is None:
                mask_index_select = self.mask_decoder.get_best_mask_index(iou_preds)
            best_mask_pred = mask_preds[:, [mask_index_select], :, :]
            best_obj_ptr = obj_ptrs[:, [mask_index_select]]

            # Encode initial memory
            memory_encoding = self.memory_encoder(lowres_imgenc, best_mask_pred, obj_score, is_prompt_encoding=True)

        return best_mask_pred, memory_encoding, best_obj_ptr

    # .................................................................................................................

    def initialize_from_mask(self, encoded_image_features_list: list[Tensor], mask_image: ndarray) -> Tensor:
        """Alternate video tracking initialization option. In this case, using a provided mask image as a 'prompt'.
        The provided image is assumed to be loaded using opencv, so that it has shape: HxW or HxWxC
        If the image has channels (e.g. RGB), only the 0th channel (e.g. red) will be used.

        Note that with this form of initializtion, there is no object pointer! The pointer normally
        comes from the mask prediction, so without a prediction, there is not pointer. The video
        masking should therefore be initialized with only the memory encoding and an empty pointer list.
        This doesn't have a substantial impact on the tracking
        """

        with torch.inference_mode():
            lowres_imgenc, *hires_imgenc = encoded_image_features_list
            token_hw = lowres_imgenc.shape[2:]
            device, dtype = lowres_imgenc.device, lowres_imgenc.dtype

            # Hard-code the object score as being 'high/confident', since we assume the given mask is accurate
            obj_score = torch.tensor(100.0, device=device, dtype=dtype)

            # Prepare mask image as if it were a prediction from the model
            if mask_image.ndim == 3:
                mask_image = mask_image[:, :, 0]
            mask_tensor = torch.tensor(mask_image > 127, device=device, dtype=dtype)
            mask_tensor = nn.functional.interpolate(
                mask_tensor.unsqueeze(0).unsqueeze(0), size=(4 * token_hw[0], 4 * token_hw[1]))

            memory_encoding = self.memory_encoder(lowres_imgenc, mask_tensor, obj_score, is_prompt_encoding=True)

        return memory_encoding

    def step_video_masking(
        self,
        encoded_image_features_list: list[Tensor],
        main_memory: MainMemory | None = None,
        mask_hint: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Function which makes segmentation predictions for consecutive frames of
        an input video. It takes in the encoded video frame data along with prior
        prompt/previous frame memory data in order to automatically continue
        segmenting some existing object (i.e. without requiring user prompts).

        Returns:
            object_score, best_mask_index, mask_predictions, memory_encoding, best_object_pointer
            -> object_score score is an indicator of whether an object is present
               values below 0 indicate lost tracking. Has shape: Bx1
            -> best_mask_index is the index of the highest iou score. Has shape: B (one index for each batch)
            -> mask_predictions are the same as with image segmentation, has shape: Bx4xHxW
            -> memory_encoding should be passed back in on future frames, has shape: BxFxH'xW'
            -> best_object_pointer should be passed back in on future frames, has shape: Bx1xF'

            The HxW of masks will be 1/4 of the input height and width (256x256 for default 1024 sizing).
            The memory encoding H'xW' is 4 times smaller than the mask sizing (64x64 by default).
            The memory & pointer features F & F' are model configs (64, 256 respectively, by default)
        """

        with torch.inference_mode():
            # Encode image features with previous memory encodings & object pointer data
            # Called '_prepare_memory_conditioned_features' in original code
            lowres_imgenc, *hires_imgenc = encoded_image_features_list

            memory_fused_image = self.memory_fusion(
                prompt_memory_encodings = main_memory.reference_encodings,
                prompt_object_pointers = main_memory.reference_pointers,
                previous_memory_encodings = main_memory.previous_encodings,
                previous_object_pointers = main_memory.previous_pointers,
                lowres_image_tokens_bchw= lowres_imgenc)

            # Run mask decoder on memory-fused features
            patch_grid_hw = memory_fused_image.shape[2:]
            grid_position = self.coordinate_encoder.get_grid_position_encoding(patch_grid_hw)

            mask_preds, iou_preds, object_ptrs, object_score, iou_token_out, obj_token_out = self.mask_decoder(
                blank_promptless_output = False,
                grid_positional_encoding = grid_position,
                encoded_image_tokens_list_bchw = [memory_fused_image, *hires_imgenc],
                encoded_prompts_bnc = self.prompt_encoder.create_video_no_prompt_encoding(),
                mask_hint = mask_hint)

        return mask_preds, iou_preds, object_ptrs, object_score, iou_token_out, obj_token_out

    # .................................................................................................................

    def refine_mask_with_hint(
        self,
        main_memory: MainMemory,
        encoded_image_features_list: list[Tensor],
        mask_hint: Tensor,
    ):
        """Re-run the mask decoder with `mask_hint` as a refinement prompt
        on the current frame — memory-conditioned, plus the hint added to
        the image tokens by the decoder's mask-hint encoder. Returns
        `(binarized mask, object pointer, memory encoding)` for the
        argmax-IoU candidate. Does NOT touch `main_memory`."""

        with torch.inference_mode():
            lowres_imgenc = encoded_image_features_list[0]
            device, dtype = lowres_imgenc.device, lowres_imgenc.dtype
            hint = mask_hint.to(device=device, dtype=dtype)
            if hint.ndim == 2:
                hint = hint.unsqueeze(0)

            mask_preds, iou_scores, object_pointers, object_score, _, _ = self.step_video_masking(
                main_memory = main_memory,
                encoded_image_features_list = encoded_image_features_list,
                mask_hint = hint)

            best_idx = 1 + torch.argmax(iou_scores[:, 1:], dim=-1)
            chosen_mask = mask_preds[:, best_idx, :, :]
            chosen_pointer = object_pointers[:, best_idx, :]
            chosen_encoding = self.memory_encoder(
                mask_prediction = chosen_mask,
                object_score = object_score,
                lowres_image_encoding = lowres_imgenc)

        chosen_mask = (chosen_mask > 0.0).squeeze().to(torch.float64).cpu()
        return chosen_mask, chosen_pointer, chosen_encoding

    def select_best_mask(self, current_frame: NDArray[np.uint8], main_memory: MainMemory,
                         encoded_image_features_list=None):
        """Select the mask, object pointer and memory encoding with the highest score.

        If `encoded_image_features_list` is provided, the expensive image-encoder
        forward pass is skipped — useful when running the same frame against
        multiple memory banks (e.g. in the ensemble tracker).
        """

        bgr_current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)

        with torch.inference_mode():
            # Extract the proposed masks and object pointers from the mask decoder
            if encoded_image_features_list is None:
                encoded_image_features_list, _, _ = self.encode_image(bgr_current_frame)
            lowres_imgenc, *hires_imgenc = encoded_image_features_list

            mask_preds, iou_scores, object_pointers, object_score, iou_token_out, obj_token_out = self.step_video_masking(
                main_memory = main_memory,
                encoded_image_features_list = encoded_image_features_list)

            best_idx = 1 + torch.argmax(iou_scores[:, 1:], dim=-1)
            chosen_mask = mask_preds[:, best_idx, :, :]
            chosen_pointer = object_pointers[:, best_idx, :]

            chosen_encoding = self.memory_encoder(
                mask_prediction = chosen_mask,
                object_score = object_score,
                lowres_image_encoding = lowres_imgenc)

        chosen_mask = (chosen_mask > 0.0).squeeze().to(torch.float64).cpu()
        iou_score = iou_scores.max(1).values.squeeze().to(torch.float64).cpu()
        object_score = torch.sigmoid(object_score.squeeze().to(torch.float64).cpu())

        return chosen_mask, chosen_pointer, chosen_encoding, object_score, iou_score, iou_token_out, obj_token_out

    def select_random_mask(self, current_frame: NDArray[np.uint8], main_memory: MainMemory,
                           encoded_image_features_list=None):
        """Same as `select_best_mask` but picks one of the 3 SAM 2 candidates uniformly
        at random, for sub-memory diversity."""

        bgr_current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)

        with torch.inference_mode():
            if encoded_image_features_list is None:
                encoded_image_features_list, _, _ = self.encode_image(bgr_current_frame)
            lowres_imgenc, *hires_imgenc = encoded_image_features_list

            mask_preds, iou_scores, object_pointers, object_score, iou_token_out, obj_token_out = self.step_video_masking(
                main_memory = main_memory,
                encoded_image_features_list = encoded_image_features_list)

            chosen_idx = torch.tensor([1 + int(np.random.randint(0, 3))], device=mask_preds.device)
            chosen_mask = mask_preds[:, chosen_idx, :, :]
            chosen_pointer = object_pointers[:, chosen_idx, :]

            chosen_encoding = self.memory_encoder(
                mask_prediction = chosen_mask,
                object_score = object_score,
                lowres_image_encoding = lowres_imgenc)

        chosen_mask = (chosen_mask > 0.0).squeeze().to(torch.float64).cpu()
        iou_score = iou_scores[:, chosen_idx].squeeze().to(torch.float64).cpu()
        object_score = torch.sigmoid(object_score.squeeze().to(torch.float64).cpu())

        return chosen_mask, chosen_pointer, chosen_encoding, object_score, iou_score, iou_token_out, obj_token_out

    def select_best_mask_oracle(self, current_frame: NDArray[np.uint8], main_memory: MainMemory, bboxes_norm):
        """Select the mask, object pointer and memory encoding with the highest score."""

        bgr_current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)

        with torch.inference_mode():
            # Extract the proposed masks and object pointers from the mask decoder
            encoded_image_features_list, _, _ = self.encode_image(bgr_current_frame)
            lowres_imgenc, *hires_imgenc = encoded_image_features_list

            mask_preds, iou_scores, object_pointers, object_score, iou_token_out, _ = self.step_video_masking(
                main_memory = main_memory,
                encoded_image_features_list = encoded_image_features_list)

            chosen_masks = (mask_preds > 0.0).squeeze().to(torch.float64).cpu()
            bboxes_norm = np.repeat(bboxes_norm[np.newaxis, :], 4, axis=0)

            true_iou_scores = compute_iou(bboxes_norm, chosen_masks.numpy())
            best_idx = [np.argmax(true_iou_scores)]

            chosen_mask = mask_preds[:, best_idx, :, :]
            chosen_pointer = object_pointers[:, best_idx, :]

            chosen_encoding = self.memory_encoder(
                mask_prediction = chosen_mask,
                object_score = object_score,
                lowres_image_encoding = lowres_imgenc)

        chosen_mask = (chosen_mask > 0.0).squeeze().to(torch.float64).cpu()
        iou_score = iou_scores.max(1).values.squeeze().to(torch.float64).cpu()
        object_score = torch.sigmoid(object_score.squeeze().to(torch.float64).cpu())

        return chosen_mask, chosen_pointer, chosen_encoding, object_score, iou_score, iou_token_out