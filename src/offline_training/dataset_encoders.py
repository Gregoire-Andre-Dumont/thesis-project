"""Four ~80M encoders that each turn a crop into a 32x32 grid of patch tokens, for the calibrator dataset.
No hiera/memory variants -- one token set per encoder. All share the 32x32 grid so their patch-similarity
maps to the anchor are comparable.

  pe        vit_pe_spatial_base_patch16_512.fb   86.4M  contrastive VL   (512 -> 32x32, HALF norm)
  dino      vit_base_patch16_dinov3.lvd1689m     85.6M  self-sup DINOv3  (512 -> 32x32, ImageNet norm)
  hiera_sam sam2_hiera_base_plus.fb_r896_2pt1    ~70M   Hiera-B+ SAM2    (512 -> 32x32 stage-2)
  hiera_mae hiera_base_plus_224.mae              69.9M  Hiera-B+ MAE     (512 -> 32x32 stage-2, pos interp)
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
    imgs = np.stack([cv2.resize(c, (size, size), interpolation=cv2.INTER_CUBIC) for c in crops]).astype(np.float32) / 255
    return ((torch.from_numpy(imgs).permute(0, 3, 1, 2) - mean) / std).to(dev).to(dtype)


def _hiera_stage2(model, crops, dev, dtype):
    """Stage-2 tokens (input/16 -> 32x32 at 512) for a timm Hiera features_only backbone: (n, 32*32, dim).
    timm hiera features_only returns channels-first (n, C, 32, 32)."""
    feats = model(_norm(crops, 512, IMEAN, ISTD, dev, dtype))[2]          # (n, C, 32, 32)
    f = feats.permute(0, 2, 3, 1)                                         # (n, 32, 32, C)
    return f.reshape(f.shape[0], GRID * GRID, f.shape[-1]).float()


def _load_hiera_mae(dev, dtype):
    """Hiera-B+ MAE backbone at 512 (features_only). timm won't interpolate the pos_embed on load,
    so we bicubic-resize the pretrained 56x56 stage-0 grid to 128x128 and load with the 'model.' prefix."""
    src = timm.create_model("hiera_base_plus_224.mae", pretrained=True).state_dict()
    pe = src["pos_embed"]                                                 # (1, 56*56, 112)
    s, d = int(pe.shape[1] ** .5), pe.shape[2]
    src["pos_embed"] = F.interpolate(pe.reshape(1, s, s, d).permute(0, 3, 1, 2).float(),
                                     size=(GRID * 4, GRID * 4), mode="bicubic", align_corners=False
                                     ).permute(0, 2, 3, 1).reshape(1, (GRID * 4) ** 2, d)
    model = timm.create_model("hiera_base_plus_224.mae", pretrained=False, img_size=512, features_only=True)
    model.load_state_dict({"model." + k: v for k, v in src.items()}, strict=False)
    return model.eval().to(dev).to(dtype)


def _build_perception(dev, dtype):
    pe = timm.create_model("vit_pe_spatial_base_patch16_512.fb", pretrained=True, num_classes=0).eval().to(dev).to(dtype)
    return lambda crops: pe.forward_features(_norm(crops, 512, HALF, HALF, dev, dtype))[:, pe.num_prefix_tokens:].float()


def _build_dino(dev, dtype):
    dino = timm.create_model("vit_base_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0, img_size=512).eval().to(dev).to(dtype)
    return lambda crops: dino.forward_features(_norm(crops, 512, IMEAN, ISTD, dev, dtype))[:, dino.num_prefix_tokens:].float()


def _build_hiera_sam(dev, dtype):
    sam2 = timm.create_model("sam2_hiera_base_plus.fb_r896_2pt1", pretrained=True, features_only=True).eval().to(dev).to(dtype)
    return lambda crops: _hiera_stage2(sam2, crops, dev, dtype)


def _build_hiera_mae(dev, dtype):
    mae = _load_hiera_mae(dev, dtype)
    return lambda crops: _hiera_stage2(mae, crops, dev, dtype)


ENCODER_BUILDERS = {
    "perception": _build_perception,
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

    n = len(masks)
    cf = np.zeros((n, crop_resize, crop_resize, 3), np.uint8)
    cm = np.zeros((n, crop_resize, crop_resize), np.float32)
    for i, (frame, mask) in enumerate(zip(frames, masks)):
        mask = cv2.resize(mask.astype(np.float32), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        coords = cv2.findNonZero((mask > 0.0).astype(np.uint8))
        if coords is None:
            continue
        x, y, w, h = cv2.boundingRect(coords)
        side = min(int(round(max(w, h, 1, size_floor) * (1 + 2 * pad_ratio))), frame.shape[1], frame.shape[0])
        half = side / 2
        x0, y0 = int(round(x + w / 2 - half)), int(round(y + h / 2 - half))
        x1, y1 = x0 + side, y0 + side
        if x0 < 0: x1 -= x0; x0 = 0
        if y0 < 0: y1 -= y0; y0 = 0
        if x1 > frame.shape[1]: x0 -= x1 - frame.shape[1]; x1 = frame.shape[1]
        if y1 > frame.shape[0]: y0 -= y1 - frame.shape[0]; y1 = frame.shape[0]
        cf[i] = cv2.resize(frame[y0:y1, x0:x1], (crop_resize, crop_resize), interpolation=cv2.INTER_CUBIC)
        m = cv2.resize(mask[y0:y1, x0:x1], (crop_resize, crop_resize), interpolation=cv2.INTER_CUBIC)
        cm[i] = (m > 0.0).astype(np.float32)
    return cf, cm


def _patch_masks(crop_masks, device):
    """Crop masks (n, cr, cr) -> per-patch foreground mask on the 32x32 grid (n, 1024) bool."""
    m = torch.from_numpy(crop_masks).to(device).unsqueeze(1)
    return (F.interpolate(m, size=(GRID, GRID), mode="nearest") > 0.5).flatten(1)


def _split_fg_bg(tokens, patch_mask):
    """Foreground/background token views; masked-out patches set to a -5 sentinel (SamaraHieraModel)."""
    pm = patch_mask.bool().unsqueeze(-1)
    pad = torch.tensor(-5.0, device=tokens.device, dtype=tokens.dtype)
    return torch.where(pm, tokens, pad), torch.where(~pm, tokens, pad)


def _patch_similarities(reference, target, chunk=16):
    """Per target patch, best cosine similarity to the valid reference patches (-5 excluded).
    reference (R, P, D), target (T, Q, D) -> (T, R, Q). Copied from SamaraHieraModel."""
    ref_valid = ~(reference == -5).all(dim=-1)
    ref_mask = ref_valid[:, None, :, None]
    all_ref_invalid = ~ref_valid.any(dim=-1)
    refn = F.normalize(reference, dim=-1)
    out = []
    for s in range(0, target.shape[0], chunk):
        tc = target[s:s + chunk]
        tvalid = ~(tc == -5).all(dim=-1)
        tn = F.normalize(tc, dim=-1)
        sim = torch.einsum("rpd,tqd->rtpq", refn, tn).masked_fill(~ref_mask, float("-inf")).amax(dim=2)
        if all_ref_invalid.any():
            sim[all_ref_invalid] = 0.0
        out.append(sim * tvalid[None].to(sim.dtype))
    return torch.cat(out, dim=1).float().cpu().numpy().transpose(1, 0, 2)


def encode_tokens(token_fn, crop_frames, enc_chunk=16):
    """Run one encoder over all crops in chunks. Returns (n, 1024, dim) tokens on the encoder device."""
    with torch.inference_mode():
        return torch.cat([token_fn(crop_frames[s:s + enc_chunk]) for s in range(0, len(crop_frames), enc_chunk)], dim=0)


def similarity_feature_map(tokens, crop_masks, reference_foreground):
    """Anchor-similarity feature map for one encoder's tokens: (n, 1, 32, 32, 2) float16. Channel 0
    is each foreground patch's best cosine similarity to `reference_foreground`, channel 1 the
    background patches'. `reference_foreground` is (1, 1024, dim) foreground tokens (-5 elsewhere).

    This is the single shared feature the calibrator consumes at dataset-creation AND deployment."""
    n = tokens.shape[0]
    patch_mask = _patch_masks(crop_masks, tokens.device)
    fg, bg = _split_fg_bg(tokens, patch_mask)
    fg_sim = _patch_similarities(reference_foreground, fg)[:, 0].reshape(n, 1, GRID, GRID)
    bg_sim = _patch_similarities(reference_foreground, bg)[:, 0].reshape(n, 1, GRID, GRID)
    return np.stack([fg_sim, bg_sim], axis=-1).astype(np.float16)


def anchor_foreground(tokens, crop_masks):
    """Foreground token view of the anchor crop (index 0), the reference for similarity_feature_map."""
    fg, _ = _split_fg_bg(tokens[0:1], _patch_masks(crop_masks[0:1], tokens.device))
    return fg


def encode_trajectory(encoders, crop_frames, crop_masks, enc_chunk=16):
    """Encode one trajectory's crops with every backbone into its anchor-similarity feature map.
    The anchor is crop 0. Returns {encoder_name: (n, 1, 32, 32, 2) float16}."""

    features = {}
    for name, token_fn in encoders.items():
        tokens = encode_tokens(token_fn, crop_frames, enc_chunk)
        features[name] = similarity_feature_map(tokens, crop_masks, anchor_foreground(tokens, crop_masks))
    return features
