"""Post-first-occlusion coverage — at a single threshold, and as AUC over a threshold sweep."""

import numpy as np


def coverage(iou_scores, occlusions, threshold):
    """Fraction of visible post-first-occlusion frames where `iou > threshold`. NaN if the
    trajectory has no occlusion or no visible frame after the first occlusion."""

    iou_scores = np.asarray(iou_scores)
    occlusions = np.asarray(occlusions)
    occluded = np.where(occlusions > 0.5)[0]
    if len(occluded) == 0:
        return np.nan
    visible_post = np.where((occlusions < 0.5) & (np.arange(len(occlusions)) > occluded[0]))[0]
    if len(visible_post) == 0:
        return np.nan
    return float((iou_scores[visible_post] > threshold).mean())


def coverage_auc(iou_scores, occlusions, thresholds=None):
    """Mean of `coverage(...)` over a sweep of IoU thresholds"""

    thresholds = np.linspace(0, 1, 21) if thresholds is None else np.asarray(thresholds)
    values = np.array([coverage(iou_scores, occlusions, float(t)) for t in thresholds])
    if np.all(np.isnan(values)):
        return np.nan
    return float(np.nanmean(values))
