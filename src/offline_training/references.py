import numpy as np

from src.utils.compute_iou import compute_iou


# ---------------------------------------------------------------------------------------
# helpers — coverage filter + token-extraction pass
# ---------------------------------------------------------------------------------------

def _post_occlusion_coverage(iou_scores, occlusions, threshold):
    """Fraction of visible post-first-occlusion frames with IoU > `threshold`."""

    occluded = np.where(occlusions > 0.5)[0]
    visible = np.where(occlusions < 0.5)[0]
    if len(occluded) == 0:
        return 0.0
    post = visible[visible > occluded[0]]
    return (iou_scores[post] > threshold).mean() if len(post) else 0.0


def _extract_hiera_and_memory(model, cropped_frames, cropped_masks):
    """Run the Hiera image encoder AND the memory encoder over the same crops, returning
    `((hiera_fg, hiera_bg), (memory_fg, memory_bg))`."""

    hiera_tokens, hiera_patch_masks = model.extract_raw_patch_tokens(cropped_frames, cropped_masks)
    hiera_fg, hiera_bg = model.split_foreground_background(hiera_tokens, hiera_patch_masks)

    memory_tokens, memory_patch_masks = model.extract_memory_patch_tokens(cropped_frames, cropped_masks)
    memory_fg, memory_bg = model.split_foreground_background(memory_tokens, memory_patch_masks)

    return (hiera_fg, hiera_bg), (memory_fg, memory_bg)


# ---------------------------------------------------------------------------------------
# trajectory builder + per-variant feature packing
# ---------------------------------------------------------------------------------------

def compute_features(model, foreground, background, anchor_foreground, anchor_background):
    """Pack per-frame patch similarities into the `(n_frames, 1, side, side, 4)` float16
    features tensor consumed by `MainDataset`.

    Channels (last axis):
        0 — foreground patches' max cosine to anchor FG bank
        1 — background patches' max cosine to anchor FG bank
        2 — foreground patches' (max cos to anchor FG) − (max cos to anchor BG)
        3 — background patches' (max cos to anchor FG) − (max cos to anchor BG)

    Channels 0-1 are the legacy 2-channel set. Channels 2-3 are the per-patch
    FG-vs-BG discriminative margin — same geometry the deploy-time diff heuristic
    relies on. CNN opts into all four via `channel="all"`."""

    n_frames, n_patches, _ = foreground.shape
    side = int(round(n_patches ** 0.5))

    foreground_to_anchor_fg = model.compute_patch_similarities(anchor_foreground, foreground)
    background_to_anchor_fg = model.compute_patch_similarities(anchor_foreground, background)
    foreground_to_anchor_bg = model.compute_patch_similarities(anchor_background, foreground)
    background_to_anchor_bg = model.compute_patch_similarities(anchor_background, background)

    foreground_diff = foreground_to_anchor_fg - foreground_to_anchor_bg
    background_diff = background_to_anchor_fg - background_to_anchor_bg

    return np.stack([
        foreground_to_anchor_fg.reshape(n_frames, 1, side, side),
        background_to_anchor_fg.reshape(n_frames, 1, side, side),
        foreground_diff.reshape(n_frames, 1, side, side),
        background_diff.reshape(n_frames, 1, side, side),
    ], axis=-1).astype(np.float16)


def predict_and_filter_trajectory(tracker, detection_data, warmup_count,
                                          coverage_threshold, commit_threshold):
    """Run the tracker ONCE, compute IoUs vs GT, apply the post-occlusion coverage
    filter. Returns `(metadata_kwargs, frames, predicted_masks)` ready for cropping at
    any pad_ratio, or `None` if the trajectory fails the coverage filter.

    The expensive SAM 2 forward pass lives here — separated from cropping/encoding so
    the same predictions can drive multiple pad_ratio datasets in a single sweep."""

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


def extract_tokens_at_pad_ratio(model, frames, predicted_masks, pad_ratio):
    """Re-crop the frames at the given `pad_ratio` and re-encode them with BOTH the
    Hiera image encoder and the Memory encoder. Mutates `model.pad_ratio` to control
    the crop side (cheaper than threading a kwarg through `extract_crops`). Returns
    `((hiera_fg, hiera_bg), (memory_fg, memory_bg))`."""

    model.pad_ratio = pad_ratio
    cropped_frames, cropped_masks = model.extract_crops(frames, predicted_masks)
    return _extract_hiera_and_memory(model, cropped_frames, cropped_masks)
