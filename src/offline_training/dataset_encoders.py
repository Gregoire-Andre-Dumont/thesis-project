"""Encoders that turn a mask's crop into a 32x32 grid of patch tokens, and the anchor-similarity map the
calibrator reads. Every encoder lands on the same grid, so one can be swapped for another downstream.

  perception      vit_pe_spatial_large_patch14_448.fb  ~304M  contrastive VL   (448, HALF norm)
  perception_base vit_pe_spatial_base_patch16_512.fb   ~90M   contrastive VL   (512, local weights)
  dino            vit_large_patch16_dinov3.lvd1689m    ~300M  self-sup DINOv3  (512, ImageNet norm)
  hiera_sam       sam2_hiera_large.fb_r1024_2pt1       ~212M  Hiera-L SAM2     (512, stage-2)
  hiera_mae       hiera_large_224.mae                  ~213M  Hiera-L MAE      (512, stage-2, pos interp)

The calibrator dataset uses `perception`; the rest serve the encoder-comparison experiments.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import timm

GRID = 32                     # patch grid side every encoder is resampled to
SENTINEL = -5.0               # marks a token masked out of a foreground or background view
HALF = torch.tensor([.5, .5, .5]).view(1, 3, 1, 1)
IMAGENET_MEAN = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------------------------------
# encoder backbones
# ---------------------------------------------------------------------------------------

def _normalise_crops(crops, size, mean, standard_deviation, device, dtype):
    """Crops (n, H, W, 3) uint8 -> normalised (n, 3, size, size) tensor on `device`."""

    resized = np.stack([cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC) for crop in crops])
    scaled = torch.from_numpy(resized.astype(np.float32) / 255)
    channels_first = scaled.permute(0, 3, 1, 2)
    return ((channels_first - mean) / standard_deviation).to(device).to(dtype)


def _hiera_stage_two_tokens(model, crops, device, dtype):
    """Stage-2 tokens from a timm Hiera features_only backbone, (n, 1024, dim). timm returns them
    channels-first, so the grid is permuted before it is flattened."""

    normalised = _normalise_crops(crops, 512, IMAGENET_MEAN, IMAGENET_STD, device, dtype)
    stage_two = model(normalised)[2]
    grid = stage_two.permute(0, 2, 3, 1)
    return grid.reshape(grid.shape[0], GRID * GRID, grid.shape[-1]).float()


def _load_hiera_mae(device, dtype):
    """Hiera-L MAE at 512. timm will not interpolate the positional embedding on load, so the pretrained
    56x56 grid is resized to 128x128 here and loaded under the 'model.' prefix."""

    state = timm.create_model("hiera_large_224.mae", pretrained=True).state_dict()
    positional_embedding = state["pos_embed"]
    source_grid = int(positional_embedding.shape[1] ** 0.5)
    target_grid = GRID * 4
    dimension = positional_embedding.shape[2]

    as_grid = positional_embedding.reshape(1, source_grid, source_grid, dimension)
    source_map = as_grid.permute(0, 3, 1, 2).float()
    target_map = F.interpolate(source_map, size=(target_grid, target_grid), mode="bicubic", align_corners=False)
    state["pos_embed"] = target_map.permute(0, 2, 3, 1).reshape(1, target_grid * target_grid, dimension)

    model = timm.create_model("hiera_large_224.mae", pretrained=False, img_size=512, features_only=True)
    model.load_state_dict({"model." + key: value for key, value in state.items()}, strict=False)
    return model.eval().to(device).to(dtype)


def _vit_token_function(encoder, size, mean, standard_deviation, device, dtype):
    """Patch tokens from a timm ViT, with its prefix (class/register) tokens dropped."""

    def token_function(crops):
        normalised = _normalise_crops(crops, size, mean, standard_deviation, device, dtype)
        tokens = encoder.forward_features(normalised)
        return tokens[:, encoder.num_prefix_tokens:].float()
    return token_function


def _build_perception(device, dtype):
    encoder = timm.create_model("vit_pe_spatial_large_patch14_448.fb", pretrained=True, num_classes=0)
    encoder = encoder.eval().to(device).to(dtype)
    return _vit_token_function(encoder, 448, HALF, HALF, device, dtype)


def _build_perception_base(device, dtype):
    """Loaded from local weights so it never reaches HuggingFace and cannot be rate-limited mid-run."""

    candidates = [
        Path(__file__).resolve().parents[2] / "tm" / "pe_spatial_base_512.safetensors",
        Path("/workspace/thesis-project/tm/pe_spatial_base_512.safetensors"),
        Path("tm/pe_spatial_base_512.safetensors"),
    ]
    weights_path = next((candidate for candidate in candidates if candidate.exists()), None)
    if weights_path is None:
        raise FileNotFoundError(f"PE weights not found; put pe_spatial_base_512.safetensors in tm/ ({candidates})")

    from safetensors.torch import load_file
    encoder = timm.create_model("vit_pe_spatial_base_patch16_512.fb", pretrained=False, num_classes=0)
    encoder.load_state_dict(load_file(str(weights_path)), strict=False)
    encoder = encoder.eval().to(device).to(dtype)
    return _vit_token_function(encoder, 512, HALF, HALF, device, dtype)


def _build_dino(device, dtype):
    encoder = timm.create_model("vit_large_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0, img_size=512)
    encoder = encoder.eval().to(device).to(dtype)
    return _vit_token_function(encoder, 512, IMAGENET_MEAN, IMAGENET_STD, device, dtype)


def _build_hiera_sam(device, dtype):
    encoder = timm.create_model("sam2_hiera_large.fb_r1024_2pt1", pretrained=True, features_only=True)
    encoder = encoder.eval().to(device).to(dtype)
    return lambda crops: _hiera_stage_two_tokens(encoder, crops, device, dtype)


def _build_hiera_mae(device, dtype):
    encoder = _load_hiera_mae(device, dtype)
    return lambda crops: _hiera_stage_two_tokens(encoder, crops, device, dtype)


ENCODER_BUILDERS = {
    "perception": _build_perception,
    "perception_base": _build_perception_base,
    "dino": _build_dino,
    "hiera_sam": _build_hiera_sam,
    "hiera_mae": _build_hiera_mae,
}


def load_dataset_encoders(names=None, device="cuda", dtype=torch.bfloat16):
    """{name: token_function}; each maps crops (n, H, W, 3) -> (n, 1024, dim) float tokens."""

    names = list(ENCODER_BUILDERS) if names is None else names
    return {name: ENCODER_BUILDERS[name](device, dtype) for name in names}


# ---------------------------------------------------------------------------------------
# cropping a mask out of its frame
# ---------------------------------------------------------------------------------------

def anchor_size_pixels(anchor_bbox_norm, frame_shape):
    """The anchor box's longer side in pixels. Used as a floor for every crop, so the window does not
    collapse when the mask shrinks under partial occlusion."""

    frame_height, frame_width = frame_shape[:2]
    return int(round(max(anchor_bbox_norm[2] * frame_width, anchor_bbox_norm[3] * frame_height)))


def _mask_at_frame_size(mask, frame_shape):
    """One low-resolution mask resized to the frame's own resolution, float32."""

    frame_height, frame_width = frame_shape[0], frame_shape[1]
    return cv2.resize(mask.astype(np.float32), (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)


def _crop_window(mask, frame_shape, pad_ratio, size_floor):
    """(left, top, side) of the square crop around `mask`, or None when it is empty. The side is floored at
    `size_floor` so a shrinking mask keeps a consistent scale against the anchor."""

    frame_height, frame_width = frame_shape[0], frame_shape[1]
    coordinates = cv2.findNonZero((mask > 0.0).astype(np.uint8))
    if coordinates is None:
        return None

    left, top, width, height = cv2.boundingRect(coordinates)
    padded_side = int(round(max(width, height, 1, size_floor) * (1 + 2 * pad_ratio)))
    side = min(padded_side, frame_width, frame_height)
    centre_x, centre_y = left + width / 2, top + height / 2

    crop_left = min(max(int(round(centre_x - side / 2)), 0), frame_width - side)
    crop_top = min(max(int(round(centre_y - side / 2)), 0), frame_height - side)
    return crop_left, crop_top, side


def crop_around_masks(frames, masks, crop_resize=512, pad_ratio=0.25, size_floor=0):
    """Square crop around each mask -> images (n, crop_resize, crop_resize, 3) uint8 and masks
    (n, crop_resize, crop_resize) float32, binarised after the resize. An empty mask yields zeros."""

    square = (crop_resize, crop_resize)
    crop_images = np.zeros((len(masks), crop_resize, crop_resize, 3), np.uint8)
    crop_masks = np.zeros((len(masks), crop_resize, crop_resize), np.float32)

    for index, (frame, mask) in enumerate(zip(frames, masks)):
        full_size_mask = _mask_at_frame_size(mask, frame.shape)
        window = _crop_window(full_size_mask, frame.shape, pad_ratio, size_floor)
        if window is None:
            continue

        left, top, side = window
        right, bottom = left + side, top + side
        
        crop_images[index] = cv2.resize(frame[top:bottom, left:right], square, interpolation=cv2.INTER_CUBIC)
        resized_mask = cv2.resize(full_size_mask[top:bottom, left:right], square, interpolation=cv2.INTER_CUBIC)
        crop_masks[index] = (resized_mask > 0.0).astype(np.float32)
    return crop_images, crop_masks


# ---------------------------------------------------------------------------------------
# anchor-similarity feature map
# ---------------------------------------------------------------------------------------

def _patch_masks(crop_masks, device):
    """Crop masks -> per-patch foreground flags on the grid, (n, 1024) bool."""

    masks = torch.from_numpy(crop_masks).to(device).unsqueeze(1)
    patch_grid = F.interpolate(masks, size=(GRID, GRID), mode="nearest")
    return (patch_grid > 0.5).flatten(1)


def _split_foreground_background(tokens, patch_mask):
    """Two views of the same tokens, inside the mask and outside it, with the excluded patches set to
    SENTINEL so the similarity pass can skip them."""

    is_foreground = patch_mask.bool().unsqueeze(-1)
    sentinel = torch.tensor(SENTINEL, device=tokens.device, dtype=tokens.dtype)
    foreground = torch.where(is_foreground, tokens, sentinel)
    background = torch.where(~is_foreground, tokens, sentinel)
    return foreground, background


def _best_patch_similarities(reference, target, chunk=16):
    """For every target patch, its best cosine similarity to any valid reference patch: (n, references,
    patches). Sentinel patches score zero on either side rather than -inf."""

    reference_valid = ~(reference == SENTINEL).all(dim=-1)
    reference_valid_mask = reference_valid[:, None, :, None]
    reference_entirely_invalid = ~reference_valid.any(dim=-1)
    reference_normalised = F.normalize(reference, dim=-1)

    similarity_chunks = []
    for start in range(0, target.shape[0], chunk):
        target_chunk = target[start:start + chunk]
        target_valid = ~(target_chunk == SENTINEL).all(dim=-1)
        target_normalised = F.normalize(target_chunk, dim=-1)

        similarities = torch.einsum("rpd,tqd->rtpq", reference_normalised, target_normalised)
        similarities = similarities.masked_fill(~reference_valid_mask, float("-inf")).amax(dim=2)
        if reference_entirely_invalid.any():
            similarities[reference_entirely_invalid] = 0.0
        similarity_chunks.append(similarities * target_valid[None].to(similarities.dtype))

    stacked = torch.cat(similarity_chunks, dim=1)
    return stacked.float().cpu().numpy().transpose(1, 0, 2)


def encode_tokens(token_function, crop_frames, encoder_chunk=16):
    """Run one encoder over all crops in chunks -> (n, 1024, dim) tokens on the encoder's device."""

    starts = range(0, len(crop_frames), encoder_chunk)
    with torch.inference_mode():
        chunks = [token_function(crop_frames[start:start + encoder_chunk]) for start in starts]
    return torch.cat(chunks, dim=0)


def anchor_foreground(tokens, crop_masks):
    """Foreground token view of the anchor crop, the reference every similarity map is measured against."""

    patch_mask = _patch_masks(crop_masks[0:1], tokens.device)
    foreground, _ = _split_foreground_background(tokens[0:1], patch_mask)
    return foreground


def similarity_feature_map(tokens, crop_masks, reference_foreground):
    """Anchor-similarity map for one set of crops: (n, 1, 32, 32, 2) float16. Channel 0 is each FOREGROUND
    patch's best similarity to the anchor's foreground, channel 1 the same for the BACKGROUND patches.

    The channels are disjoint -- a patch appears in one and reads zero in the other -- so each has to be
    averaged over its own support rather than over the whole grid."""

    count = tokens.shape[0]
    patch_mask = _patch_masks(crop_masks, tokens.device)
    foreground, background = _split_foreground_background(tokens, patch_mask)

    foreground_similarity = _best_patch_similarities(reference_foreground, foreground)[:, 0]
    background_similarity = _best_patch_similarities(reference_foreground, background)[:, 0]
    foreground_map = foreground_similarity.reshape(count, 1, GRID, GRID)
    background_map = background_similarity.reshape(count, 1, GRID, GRID)
    return np.stack([foreground_map, background_map], axis=-1).astype(np.float16)


def proposal_feature_maps(token_function, frames, masks, crop_resize=512, pad_ratio=0.25, size_floor=0, anchor_proposal=0, encoder_chunk=16):
    """One similarity map per competing proposal: masks (n, proposals, h, w) -> (n, proposals, 32, 32, 2).

    Every proposal is cropped around its OWN mask, the framing a candidate gets at deployment. A fragment is
    therefore encoded at fragment scale and a mask that swallowed a neighbour at the swallowed scale; that
    mismatch against the anchor is the signal, and a shared window would hide it. Token sets are built one
    proposal at a time -- roughly a gigabyte each on a long clip."""

    anchor_mask = masks[0:1, anchor_proposal]
    anchor_frames, anchor_masks = crop_around_masks(frames[0:1], anchor_mask, crop_resize, pad_ratio, size_floor)
    anchor_tokens = encode_tokens(token_function, anchor_frames, encoder_chunk)
    reference = anchor_foreground(anchor_tokens, anchor_masks)

    feature_maps = []
    for proposal in range(masks.shape[1]):
        proposal_masks = masks[:, proposal]
        crop_frames, crop_masks = crop_around_masks(frames, proposal_masks, crop_resize, pad_ratio, size_floor)
        tokens = encode_tokens(token_function, crop_frames, encoder_chunk)
        feature_maps.append(similarity_feature_map(tokens, crop_masks, reference))
        del tokens
    return np.concatenate(feature_maps, axis=1)
