
import numpy as np
import ipdb


def compute_auc_mask(bboxes_norm, mask_preds, n_thresholds=1000):
    """Compute the area under the curve between the true bbox and predicted mask."""

    iou_scores = compute_iou(bboxes_norm, mask_preds)
    thresholds = np.linspace(0, 1, n_thresholds)

    success = [(iou_scores >= t).mean() for t in thresholds]
    return np.trapz(success, thresholds)


def compute_iou(bboxes_norm, mask_preds):
    """Compute the iou score between the bounding boxes and masks."""

    iou_scores = np.zeros(mask_preds.shape[0], dtype=np.float32)
    image_resolution = mask_preds.shape[1]

    for idx, (bbox, mask_pred) in enumerate(zip(bboxes_norm, mask_preds)):
        # Extract and overlay the true bounding box
        mask_bbox = np.zeros(mask_pred.shape, np.uint8)
        x, y, w, h = (image_resolution * bbox).astype(int)
        mask_bbox[y: y + h, x: x + w] = 1

        # Extract and overlay the bounding box from the prediction
        bbox_pred = np.zeros(mask_pred.shape, np.uint8)
        coordinates = np.column_stack(np.where(mask_pred > 0))

        if coordinates.size > 0:
            y_min, x_min = coordinates.min(axis=0)
            y_max, x_max = coordinates.max(axis=0)
            bbox_pred[y_min:y_max + 1, x_min:x_max + 1] = 1

        # Check whether the true and predicted masks are empty
        if mask_bbox.sum() == 0 and bbox_pred.sum() == 0:
            iou_scores[idx] = 1.0

        # Compute the intersection over the union between the masks
        intersection = np.logical_and(bbox_pred, mask_bbox).sum()
        union = np.logical_or(bbox_pred, mask_bbox).sum()

        iou_scores[idx] = intersection / union if union > 0 else 0.0

    return iou_scores
