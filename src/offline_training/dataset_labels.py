"""Per-frame pseudo-IoU labels for the calibrator dataset: how well the tracker's proposed mask
matches the box-prompted pseudo-GT of the TARGET and of its K nearest clean DISTRACTORS.

A high target IoU means the proposal is on the right person; a high distractor IoU means it has been
captured by a confuser. Distractors are the clean (target-eligible) other persons annotated in that
frame, ranked by how close their box centre is to the proposal's centroid, box-prompted with the same
SAM model as the target so the pseudo-GT masks are comparable."""
import json

import cv2
import numpy as np
import torch

from src.utils.load_bboxes import convert_bbox

# Labels that mark an annotation as not a clean person (mirrors the PersonPath selection).
NON_TARGETS = ("crowd", "person_in_vehicle", "reflection", "person_in_background", "severly_occluded_person")


def load_clean_boxes_by_frame(visible_path, target_id, non_targets=NON_TARGETS):
    """{frame_idx: [box_norm, ...]} for every clean OTHER person, boxes normalized [x, y, w, h]."""
    data = json.load(open(visible_path))
    w = int(data["metadata"]["resolution"]["width"])
    h = int(data["metadata"]["resolution"]["height"])
    out = {}
    for e in data["entities"]:
        if e["id"] == target_id or any(lbl in non_targets for lbl in e["labels"]):
            continue
        bx, by, bw, bh = e["bb"]
        if bw <= 0 or bh <= 0:
            continue
        out.setdefault(e["blob"]["frame_idx"], []).append(
            np.array([bx / w, by / h, bw / w, bh / h], np.float32))
    return out


def _centroid_norm(mask):
    """Normalized (cx, cy) of a boolean mask, or None if empty."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return xs.mean() / mask.shape[1], ys.mean() / mask.shape[0]


def _box_center(b):
    return b[0] + b[2] / 2, b[1] + b[3] / 2


@torch.inference_mode()
def pseudo_iou_labels(model, frames, predicted_masks, target_boxes, clean_boxes_by_frame,
                      frame_indices, occlusions, k=3, chunk=4, precomputed_features=None, include_occluded=False):
    """Return target_iou (n,) and distractor_iou (n, k). Each frame's proposal mask is scored against the
    box-prompted pseudo-GT of the target and the k nearest clean distractors. Fewer than k distractors
    leaves the remaining columns zero.

    include_occluded: if False (default), occluded frames stay zero. If True, occluded frames are still
    scored for DISTRACTOR IoU (the target has no GT box so target_iou stays 0, but a distractor overlap is
    an unambiguous capture) -- used to catch captures that happen while the target is occluded.

    precomputed_features: optional {t: [lowres, hires_x2, hires_x4]} per-frame image encodings (B=1) to
    reuse instead of re-running the SAM image encoder -- the same 3-map list the trackers cache."""

    n = len(frames)
    target_iou = np.zeros(n, np.float32)
    distractor_iou = np.zeros((n, k), np.float32)

    def prompt_iou(single, predicted, box):
        mask, _, _ = model.initialize_video_masking(single, convert_bbox(np.asarray(box, np.float32)))
        pseudo_gt = (mask.squeeze() > 0.0).cpu().numpy()
        if predicted.shape != pseudo_gt.shape:
            predicted = cv2.resize(predicted.astype(np.uint8), pseudo_gt.shape[::-1],
                                   interpolation=cv2.INTER_NEAREST) > 0
        union = (predicted | pseudo_gt).sum()
        return float((predicted & pseudo_gt).sum() / union) if union > 0 else 0.0

    def score_frame(t, single):
        if occlusions[t] > 0.5 and not include_occluded:
            return
        predicted = np.asarray(predicted_masks[t]) > 0
        if float(target_boxes[t][2]) > 0.0:
            target_iou[t] = prompt_iou(single, predicted, target_boxes[t])

        center = _centroid_norm(predicted)
        if center is None:
            if float(target_boxes[t][2]) <= 0.0:
                return
            center = _box_center(target_boxes[t])
        boxes = clean_boxes_by_frame.get(int(frame_indices[t]), [])
        nearest = sorted(boxes, key=lambda b: (_box_center(b)[0] - center[0]) ** 2
                                              + (_box_center(b)[1] - center[1]) ** 2)[:k]
        for j, box in enumerate(nearest):
            distractor_iou[t, j] = prompt_iou(single, predicted, box)

    if precomputed_features is not None:                    # reuse cached encodings; no image-encoder pass
        device = next(model.image_encoder.parameters()).device
        for t in range(n):
            score_frame(t, [f.to(device) for f in precomputed_features[t]])
    else:
        prepared = [model.image_encoder.prepare_image(cv2.cvtColor(f, cv2.COLOR_RGB2BGR), 1024, True) for f in frames]
        for start in range(0, n, chunk):
            feats = model.image_encoder(torch.cat(prepared[start:start + chunk], dim=0))
            for i in range(feats[0].shape[0]):
                score_frame(start + i, [f[i:i + 1] for f in feats])

    return target_iou, distractor_iou
