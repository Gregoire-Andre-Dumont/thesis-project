"""Four LARGE (~212-304M) encoders that each turn a crop into a 32x32 grid of patch tokens, for the calibrator dataset.
No hiera/memory variants -- one token set per encoder. All share the 32x32 grid so their patch-similarity
maps to the anchor are comparable.

  perception vit_pe_spatial_large_patch14_448.fb  ~304M  contrastive VL   (448 -> 32x32, HALF norm)
  dino       vit_large_patch16_dinov3.lvd1689m    ~300M  self-sup DINOv3  (512 -> 32x32, ImageNet norm)
  hiera_sam  sam2_hiera_large.fb_r1024_2pt1       ~212M  Hiera-L SAM2     (512 -> 32x32 stage-2)
  hiera_mae  hiera_large_224.mae                  ~213M  Hiera-L MAE      (512 -> 32x32 stage-2, pos interp)
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import timm

GRID = 32
HALF = torch.tensor([.5, .5, .5]).view(1, 3, 1, 1)
IMEAN = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)
ISTD = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)


def _norm(crops, size, mean, std, dev, dtype):
    resized = np.stack([cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC) for crop in crops]).astype(np.float32) / 255
    channels_first = torch.from_numpy(resized).permute(0, 3, 1, 2)
    return ((channels_first - mean) / std).to(dev).to(dtype)


def _hiera_stage2(model, crops, dev, dtype):
    """Stage-2 tokens (input/16 -> 32x32 at 512) for a timm Hiera features_only backbone: (n, 32*32, dim).
    timm hiera features_only returns channels-first (n, C, 32, 32)."""
    stage2 = model(_norm(crops, 512, IMEAN, ISTD, dev, dtype))[2]         # (n, C, 32, 32) channels-first
    grid = stage2.permute(0, 2, 3, 1)                                     # (n, 32, 32, C)
    return grid.reshape(grid.shape[0], GRID * GRID, grid.shape[-1]).float()


def _load_hiera_mae(dev, dtype):
    """Hiera-L MAE backbone at 512 (features_only). timm won't interpolate the pos_embed on load,
    so we bicubic-resize the pretrained 56x56 stage-0 grid to 128x128 and load with the 'model.' prefix."""
    
    state = timm.create_model("hiera_large_224.mae", pretrained=True).state_dict()

    pos_embed = state["pos_embed"]                                        # (1, 56*56, dim)
    source_grid = int(pos_embed.shape[1] ** 0.5)                          # 56 (pretrained stage-0 grid)
    target_grid = GRID * 4                                                # 128 (stage-0 grid at 512)
    dim = pos_embed.shape[2]

    source_map = pos_embed.reshape(1, source_grid, source_grid, dim).permute(0, 3, 1, 2).float()   # (1, dim, 56, 56)
    target_map = F.interpolate(source_map, size=(target_grid, target_grid), mode="bicubic", align_corners=False)
    state["pos_embed"] = target_map.permute(0, 2, 3, 1).reshape(1, target_grid * target_grid, dim)

    model = timm.create_model("hiera_large_224.mae", pretrained=False, img_size=512, features_only=True)
    model.load_state_dict({"model." + key: value for key, value in state.items()}, strict=False)
    return model.eval().to(dev).to(dtype)


def _build_perception(dev, dtype):
    pe = timm.create_model("vit_pe_spatial_large_patch14_448.fb", pretrained=True, num_classes=0).eval().to(dev).to(dtype)
    return lambda crops: pe.forward_features(_norm(crops, 448, HALF, HALF, dev, dtype))[:, pe.num_prefix_tokens:].float()


def _build_perception_base(dev, dtype):
    # Fully local load (pretrained=False, then load the tm/ safetensors) -- NEVER touches HuggingFace, so it can't
    # 429. Checks several absolute candidate paths for the weights and errors clearly if none is found.
    from pathlib import Path
    from safetensors.torch import load_file
    candidates = [
        Path(__file__).resolve().parents[2] / "tm" / "pe_spatial_base_512.safetensors",
        Path("/workspace/thesis-project/tm/pe_spatial_base_512.safetensors"),
        Path("tm/pe_spatial_base_512.safetensors"),
    ]
    weights = next((p for p in candidates if p.exists()), None)
    if weights is None:
        raise FileNotFoundError(f"PE weights not found; put pe_spatial_base_512.safetensors in tm/ (looked in {candidates})")
    pe = timm.create_model("vit_pe_spatial_base_patch16_512.fb", pretrained=False, num_classes=0)
    pe.load_state_dict(load_file(str(weights)), strict=False)
    pe = pe.eval().to(dev).to(dtype)
    return lambda crops: pe.forward_features(_norm(crops, 512, HALF, HALF, dev, dtype))[:, pe.num_prefix_tokens:].float()


def _build_dino(dev, dtype):
    dino = timm.create_model("vit_large_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0, img_size=512).eval().to(dev).to(dtype)
    return lambda crops: dino.forward_features(_norm(crops, 512, IMEAN, ISTD, dev, dtype))[:, dino.num_prefix_tokens:].float()


def _build_hiera_sam(dev, dtype):
    sam2 = timm.create_model("sam2_hiera_large.fb_r1024_2pt1", pretrained=True, features_only=True).eval().to(dev).to(dtype)
    return lambda crops: _hiera_stage2(sam2, crops, dev, dtype)


def _build_hiera_mae(dev, dtype):
    mae = _load_hiera_mae(dev, dtype)
    return lambda crops: _hiera_stage2(mae, crops, dev, dtype)


ENCODER_BUILDERS = {
    "perception": _build_perception,
    "perception_base": _build_perception_base,
    "dino": _build_dino,
    "hiera_sam": _build_hiera_sam,
    "hiera_mae": _build_hiera_mae,
}


def load_dataset_encoders(names=None, dev="cuda", dtype=torch.bfloat16):
    """Return {name: token_fn} for the requested encoders (all four by default). Each
    token_fn(crops:(n,H,W,3)) -> (n, 32*32, dim) float tokens. Deployment loads a single encoder;
    dataset creation loads all four."""
    names = list(ENCODER_BUILDERS) if names is None else names
    return {name: ENCODER_BUILDERS[name](dev, dtype) for name in names}


# ---------------------------------------------------------------------------------------
# crop + foreground/background similarity map (mirrors SamaraHieraModel, encoder-agnostic)
# ---------------------------------------------------------------------------------------

def anchor_size_pixels(anchor_bbox_norm, frame_shape):
    """Anchor box's larger side in pixels, used as a floor for every crop so the window doesn't
    collapse when the mask shrinks under partial occlusion. Mirrors
    SamaraHieraModel.set_anchor_size_from_normalized. A zero box disables the floor (returns 0)."""

    frame_height, frame_width = frame_shape[:2]
    return int(round(max(anchor_bbox_norm[2] * frame_width, anchor_bbox_norm[3] * frame_height)))


def crop_around_masks(frames, masks, crop_resize=512, pad_ratio=0.25, size_floor=0):
    """Square crop around each mask's bbox, padded by pad_ratio and resized to crop_resize.
    Mirrors SamaraHieraModel.extract_crops: the crop side is floored at `size_floor` pixels (the
    anchor's size), so shrinking masks keep a consistent scale against the anchor. Empty masks
    return zeros. Returns crop images (n, cr, cr, 3) uint8 and crop masks (n, cr, cr) float32
    (binary, thresholded after the resize)."""

    count = len(masks)
    crop_images = np.zeros((count, crop_resize, crop_resize, 3), np.uint8)
    crop_masks = np.zeros((count, crop_resize, crop_resize), np.float32)
    for i, (frame, mask) in enumerate(zip(frames, masks)):
        frame_height, frame_width = frame.shape[0], frame.shape[1]
        mask = cv2.resize(mask.astype(np.float32), (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)

        coords = cv2.findNonZero((mask > 0.0).astype(np.uint8))
        if coords is None:                                               # empty mask -> leave zeros
            continue

        x, y, width, height = cv2.boundingRect(coords)
        padded_side = int(round(max(width, height, 1, size_floor) * (1 + 2 * pad_ratio)))
        side = min(padded_side, frame_width, frame_height)        
        half = side / 2
        center_x, center_y = x + width / 2, y + height / 2

        x0 = min(max(int(round(center_x - half)), 0), frame_width - side)  
        y0 = min(max(int(round(center_y - half)), 0), frame_height - side)
        x1, y1 = x0 + side, y0 + side

        crop_images[i] = cv2.resize(frame[y0:y1, x0:x1], (crop_resize, crop_resize), interpolation=cv2.INTER_CUBIC)
        resized_mask = cv2.resize(mask[y0:y1, x0:x1], (crop_resize, crop_resize), interpolation=cv2.INTER_CUBIC)
        crop_masks[i] = (resized_mask > 0.0).astype(np.float32)
    return crop_images, crop_masks


def _patch_masks(crop_masks, device):
    """Crop masks (n, cr, cr) -> per-patch foreground mask on the 32x32 grid (n, 1024) bool."""
    masks = torch.from_numpy(crop_masks).to(device).unsqueeze(1)
    patch_grid = F.interpolate(masks, size=(GRID, GRID), mode="nearest")
    return (patch_grid > 0.5).flatten(1)


def _split_fg_bg(tokens, patch_mask):
    """Foreground/background token views; masked-out patches set to a -5 sentinel (SamaraHieraModel)."""
    is_foreground = patch_mask.bool().unsqueeze(-1)
    sentinel = torch.tensor(-5.0, device=tokens.device, dtype=tokens.dtype)
    foreground = torch.where(is_foreground, tokens, sentinel)
    background = torch.where(~is_foreground, tokens, sentinel)
    return foreground, background


def _patch_similarities(reference, target, chunk=16):
    """Per target patch, best cosine similarity to the valid reference patches (-5 excluded).
    reference (R, P, D), target (T, Q, D) -> (T, R, Q). Copied from SamaraHieraModel."""
    reference_valid = ~(reference == -5).all(dim=-1)
    reference_valid_mask = reference_valid[:, None, :, None]
    all_reference_invalid = ~reference_valid.any(dim=-1)
    reference_normalized = F.normalize(reference, dim=-1)

    chunks = []
    for start in range(0, target.shape[0], chunk):
        target_chunk = target[start:start + chunk]
        target_valid = ~(target_chunk == -5).all(dim=-1)
        target_normalized = F.normalize(target_chunk, dim=-1)

        similarities = torch.einsum("rpd,tqd->rtpq", reference_normalized, target_normalized)
        similarities = similarities.masked_fill(~reference_valid_mask, float("-inf")).amax(dim=2)
        if all_reference_invalid.any():
            similarities[all_reference_invalid] = 0.0
        chunks.append(similarities * target_valid[None].to(similarities.dtype))
    return torch.cat(chunks, dim=1).float().cpu().numpy().transpose(1, 0, 2)


def encode_tokens(token_fn, crop_frames, enc_chunk=16):
    """Run one encoder over all crops in chunks. Returns (n, 1024, dim) tokens on the encoder device."""
    with torch.inference_mode():
        return torch.cat([token_fn(crop_frames[s:s + enc_chunk]) for s in range(0, len(crop_frames), enc_chunk)], dim=0)


def similarity_feature_map(tokens, crop_masks, reference_foreground):
    """Anchor-similarity feature map for one encoder's tokens: (n, 1, 32, 32, 2) float16. Channel 0
    is each foreground patch's best cosine similarity to `reference_foreground`, channel 1 the
    background patches'. `reference_foreground` is (1, 1024, dim) foreground tokens (-5 elsewhere).

    This is the single shared feature the calibrator consumes at dataset-creation AND deployment."""
    count = tokens.shape[0]
    patch_mask = _patch_masks(crop_masks, tokens.device)
    foreground, background = _split_fg_bg(tokens, patch_mask)
    foreground_similarity = _patch_similarities(reference_foreground, foreground)[:, 0].reshape(count, 1, GRID, GRID)
    background_similarity = _patch_similarities(reference_foreground, background)[:, 0].reshape(count, 1, GRID, GRID)
    return np.stack([foreground_similarity, background_similarity], axis=-1).astype(np.float16)


def anchor_foreground(tokens, crop_masks):
    """Foreground token view of the anchor crop (index 0), the reference for similarity_feature_map."""
    foreground, _ = _split_fg_bg(tokens[0:1], _patch_masks(crop_masks[0:1], tokens.device))
    return foreground


def encode_trajectory(encoders, crop_frames, crop_masks, enc_chunk=16):
    """Encode one trajectory's crops with every backbone into its anchor-similarity feature map.
    The anchor is crop 0. Returns {encoder_name: (n, 1, 32, 32, 2) float16}."""

    features = {}
    for name, token_fn in encoders.items():
        tokens = encode_tokens(token_fn, crop_frames, enc_chunk)
        features[name] = similarity_feature_map(tokens, crop_masks, anchor_foreground(tokens, crop_masks))
    return features
