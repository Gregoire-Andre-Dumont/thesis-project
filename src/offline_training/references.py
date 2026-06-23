import numpy as np
import torch


def build_reference_features(model, foreground, background, iou_scores, occlusions,
                             n_references, commit_threshold):
    """Build the patch-similarity feature tensor of each frame for the calibrator dataset."""

    n_frames, n_patches, feature_dim = foreground.shape
    side = int(round(n_patches ** 0.5))
    n_output_channels = 1 + n_references

    commits = np.where(iou_scores > commit_threshold)[0]

    references_foreground = torch.full(
        (n_output_channels, n_patches, feature_dim), -5.0,
        device=foreground.device, dtype=foreground.dtype)
    references_foreground[0] = foreground[0]

    features = np.zeros((n_frames, n_output_channels, side, side, 2), dtype=np.float16)
    fifo_ious = np.full((n_frames, n_references), np.nan, dtype=np.float32)

    for frame_index in range(n_frames):
        fifo_frames = commits[commits < frame_index][-n_references:]

        references_foreground[1:].fill_(-5.0)
        references_foreground[1:1 + len(fifo_frames)] = foreground[fifo_frames]
        fifo_ious[frame_index, :len(fifo_frames)] = iou_scores[fifo_frames]

        foreground_similarities = model.compute_patch_similarities(references_foreground, foreground[frame_index:frame_index + 1])
        background_similarities = model.compute_patch_similarities(references_foreground, background[frame_index:frame_index + 1])
        features[frame_index] = np.stack([
            foreground_similarities.reshape(n_output_channels, side, side),
            background_similarities.reshape(n_output_channels, side, side)], axis=-1)

    return features, fifo_ious
