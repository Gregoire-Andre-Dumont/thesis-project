import cv2
import numpy as np
import torch

from src.utils.compute_iou import compute_iou
from src.utils.load_bboxes import convert_bbox
from src.modules.samara_hiera_model import SamaraHieraModel


# Token-source variants extracted in one encoder pass; patch similarities reduce with MAX
# over the anchor's foreground patches.
VARIANTS = ["hiera", "memory"]


# ---------------------------------------------------------------------------------------
# helpers — coverage filter + token-extraction pass
# ---------------------------------------------------------------------------------------

def _post_occlusion_coverage(iou_scores, occlusions, threshold):
    """Fraction of the visible frames after the first occlusion whose IoU beats the threshold.
    Returns zero when the trajectory is never occluded."""

    occluded = np.where(occlusions > 0.5)[0]
    visible = np.where(occlusions < 0.5)[0]
    if len(occluded) == 0:
        return 0.0
    post = visible[visible > occluded[0]]
    return (iou_scores[post] > threshold).mean() if len(post) else 0.0


@torch.inference_mode()
def mask_iou_scores(model, frames, predicted_masks, gt_bboxes, occlusions, chunk=4):
    """Per-frame mask IoU of the predicted mask vs a pseudo-GT mask (SAM box-prompted on the full
    frame with the GT box). The image encoder is run in batches; the box prompt is per frame.
    Occluded / boxless frames score zero."""

    out = np.zeros(len(frames), dtype=np.float32)
    prepared = [model.image_encoder.prepare_image(cv2.cvtColor(f, cv2.COLOR_RGB2BGR), 1024, True) for f in frames]
    for start in range(0, len(frames), chunk):
        feats = model.image_encoder(torch.cat(prepared[start:start + chunk], dim=0))
        for i in range(feats[0].shape[0]):
            t = start + i
            if occlusions[t] > 0.5 or float(gt_bboxes[t][2]) == 0.0:
                continue
            single = [f[i:i + 1] for f in feats]
            mask, _, _ = model.initialize_video_masking(single, convert_bbox(np.asarray(gt_bboxes[t], np.float32)))
            pseudo_gt = (mask.squeeze() > 0.0).cpu().numpy()
            predicted = np.asarray(predicted_masks[t]) > 0
            if predicted.shape != pseudo_gt.shape:
                predicted = cv2.resize(predicted.astype(np.uint8), pseudo_gt.shape[::-1],
                                       interpolation=cv2.INTER_NEAREST) > 0
            union = (predicted | pseudo_gt).sum()
            out[t] = float((predicted & pseudo_gt).sum() / union) if union > 0 else 0.0
    return out


def _extract_hiera_and_memory(model, cropped_frames, cropped_masks):
    """Encode the same crops with both the image encoder and the memory encoder.
    Returns each encoder's foreground and background token views."""

    hiera_tokens, hiera_patch_masks = model.extract_raw_patch_tokens(cropped_frames, cropped_masks)
    hiera_fg, hiera_bg = model.split_foreground_background(hiera_tokens, hiera_patch_masks)

    memory_tokens, memory_patch_masks = model.extract_memory_patch_tokens(cropped_frames, cropped_masks)
    memory_fg, memory_bg = model.split_foreground_background(memory_tokens, memory_patch_masks)

    return (hiera_fg, hiera_bg), (memory_fg, memory_bg)


# ---------------------------------------------------------------------------------------
# trajectory builder + per-variant feature packing
# ---------------------------------------------------------------------------------------

def compute_features(model, foreground, background, anchor_foreground, anchor_background=None):
    """Pack the per-frame anchor similarities into the feature array the dataset consumes.
    Channel zero scores the foreground patches and channel one the background patches."""

    n_frames, n_patches, _ = foreground.shape
    side = int(round(n_patches ** 0.5))

    foreground_to_anchor_fg = model.compute_patch_similarities(anchor_foreground, foreground)
    background_to_anchor_fg = model.compute_patch_similarities(anchor_foreground, background)

    return np.stack([
        foreground_to_anchor_fg.reshape(n_frames, 1, side, side),
        background_to_anchor_fg.reshape(n_frames, 1, side, side),
    ], axis=-1).astype(np.float16)


def predict_and_filter_trajectory(tracker, detection_data, warmup_count,
                                          coverage_threshold, commit_threshold):
    """Track the trajectory once, label its masks with pseudo-GT mask IoU, and apply the coverage
    filter (on box IoU). Returns the labels, frames and masks to encode, or None when the trajectory
    is too poor to keep. The stored `iou_scores` is mask IoU (the training label); box IoU is kept
    for the coverage filter and as `box_iou`."""

    predicted_masks = tracker.predict_masks(detection_data).numpy()
    box_iou = compute_iou(detection_data.bboxes_norm, predicted_masks)
    box_iou[detection_data.occlusions > 0.5] = 0.0
    # The oracle labels each frame with pseudo-GT mask IoU inline, reusing the image encoding it already
    # computed while tracking (see MemoryOracle.predict_masks); fall back to a re-encoding pass otherwise.
    if getattr(tracker, "mask_iou_scores", None) is not None:
        mask_iou = tracker.mask_iou_scores.numpy()
    else:
        mask_iou = mask_iou_scores(tracker.model, detection_data.frames, predicted_masks,
                                   detection_data.bboxes_norm, detection_data.occlusions)

    predicted_masks = predicted_masks[warmup_count:]
    box_iou = box_iou[warmup_count:]
    mask_iou = mask_iou[warmup_count:]
    occlusions = detection_data.occlusions[warmup_count:]
    bboxes_norm = detection_data.bboxes_norm[warmup_count:]
    frames = detection_data.frames[warmup_count:]
    frame_indices = detection_data.frame_indices[warmup_count:]
    tracker_iou_scores = tracker.iou_scores.numpy()[warmup_count:]

    coverage = _post_occlusion_coverage(box_iou, occlusions, commit_threshold)
    if coverage < coverage_threshold:
        return None

    metadata_kwargs = {
        "frame_indices": frame_indices.astype(np.int64),
        "iou_scores":    mask_iou.astype(np.float32),          # training label = pseudo-GT mask IoU
        "box_iou":       box_iou.astype(np.float32),           # kept for tracking-eval reference
        "occlusions":    occlusions.astype(np.float32),
        "predicted_iou": tracker_iou_scores.astype(np.float32),
        "true_bboxes":   bboxes_norm.astype(np.float32),
    }
    return metadata_kwargs, frames, predicted_masks


def extract_tokens(model, frames, predicted_masks):
    """Crop the frames around their predicted masks and encode them with both encoders.
    Returns each encoder's foreground and background token views."""

    cropped_frames, cropped_masks = model.extract_crops(frames, predicted_masks)
    return _extract_hiera_and_memory(model, cropped_frames, cropped_masks)


# ---------------------------------------------------------------------------------------
# multi-encoder feature extraction (one shared set of oracle masks -> per-encoder features)
# ---------------------------------------------------------------------------------------

def load_encoder_models(encoders, crop_resize, pad_ratio):
    """Build one tracking model per encoder checkpoint, all cropping the same way.
    Only the backbone weights differ, so their datasets stay comparable."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return {name: SamaraHieraModel(sam_model_path=checkpoint, crop_resize=crop_resize, pad_ratio=pad_ratio)
                  .to(device=device, dtype=dtype)
            for name, checkpoint in encoders.items()}


def encode_trajectory(model, frames, predicted_masks):
    """Encode one trajectory with a single backbone into its hiera and memory features.
    Returns the features keyed by variant name."""

    (hiera_fg, hiera_bg), (memory_fg, memory_bg) = extract_tokens(model, frames, predicted_masks)
    return {
        "hiera":  compute_features(model, hiera_fg, hiera_bg, hiera_fg[0:1]),
        "memory": compute_features(model, memory_fg, memory_bg, memory_fg[0:1]),
    }


def encode_with_encoders(encoder_models, frames, predicted_masks):
    """Encode one trajectory with every backbone, reusing the shared oracle masks.
    Returns the features keyed by encoder and then by variant."""

    return {name: encode_trajectory(model, frames, predicted_masks)
            for name, model in encoder_models.items()}
