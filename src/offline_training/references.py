import numpy as np
import torch

from src.utils.compute_iou import compute_iou
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
    """Track the trajectory once, score its masks against ground truth, and apply the coverage filter.
    Returns the labels, frames and masks to encode, or None when the trajectory is too poor to keep."""

    predicted_masks = tracker.predict_masks(detection_data).numpy()
    iou_scores = compute_iou(detection_data.bboxes_norm, predicted_masks)
    iou_scores[detection_data.occlusions > 0.5] = 0.0

    predicted_masks = predicted_masks[warmup_count:]
    iou_scores = iou_scores[warmup_count:]
    occlusions = detection_data.occlusions[warmup_count:]
    bboxes_norm = detection_data.bboxes_norm[warmup_count:]
    frames = detection_data.frames[warmup_count:]
    frame_indices = detection_data.frame_indices[warmup_count:]
    tracker_iou_scores = tracker.iou_scores.numpy()[warmup_count:]

    coverage = _post_occlusion_coverage(iou_scores, occlusions, commit_threshold)
    if coverage < coverage_threshold:
        return None

    metadata_kwargs = {
        "frame_indices": frame_indices.astype(np.int64),
        "iou_scores":    iou_scores.astype(np.float32),
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
