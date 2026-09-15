"""Per-proposal pseudo-IoU labels: how well each mask SAM proposes matches the box-prompted pseudo-GT of the
TARGET, and of its nearest clean DISTRACTORS.

A high target IoU means the proposal is on the right person, a high distractor IoU that a confuser has taken
it. PersonPath annotates boxes rather than masks, so both ground truths come from the annotated box prompted
through the tracker's own SAM -- comparable to each other and to the proposals, at the cost of inheriting
SAM's own segmentation errors.
"""
import json

import cv2
import numpy as np
import torch

from src.utils.load_bboxes import convert_bbox

# Labels marking an annotation as not a clean person, mirroring the PersonPath selection.
NON_TARGETS = ("crowd", "person_in_vehicle", "reflection", "person_in_background", "severly_occluded_person")


# ---------------------------------------------------------------------------------------
# the clean people a proposal could have drifted onto
# ---------------------------------------------------------------------------------------

def _normalised_boxes(visible_path, target_id, non_targets):
    """(entity, normalised box) for every clean OTHER person: the target and non-target labels dropped."""

    data = json.load(open(visible_path))
    width = int(data["metadata"]["resolution"]["width"])
    height = int(data["metadata"]["resolution"]["height"])

    for entity in data["entities"]:
        if entity["id"] == target_id or any(label in non_targets for label in entity["labels"]):
            continue
        left, top, box_width, box_height = entity["bb"]
        if box_width <= 0 or box_height <= 0:
            continue
        normalised = [left / width, top / height, box_width / width, box_height / height]
        yield entity, np.array(normalised, np.float32)


def load_clean_boxes_by_frame(visible_path, target_id, non_targets=NON_TARGETS):
    """{frame_index: [box_norm, ...]} for every clean other person."""

    boxes_by_frame = {}
    for entity, box in _normalised_boxes(visible_path, target_id, non_targets):
        boxes_by_frame.setdefault(entity["blob"]["frame_idx"], []).append(box)
    return boxes_by_frame


def load_clean_boxes_by_person(visible_path, target_id, non_targets=NON_TARGETS):
    """{person_id: {frame_index: box_norm}} -- keeps identity, so one distractor can be followed."""

    boxes_by_person = {}
    for entity, box in _normalised_boxes(visible_path, target_id, non_targets):
        boxes_by_person.setdefault(entity["id"], {})[entity["blob"]["frame_idx"]] = box
    return boxes_by_person


def distractor_boxes_per_frame(clean_boxes_by_frame, frame_indices):
    """The clean boxes for each frame of a trajectory, in trajectory order.

    Resolves video frame numbering once, so the labelling below never has to know about it."""

    return [clean_boxes_by_frame.get(int(index), []) for index in frame_indices]


def _box_centre(box):
    """Normalised (x, y) centre of a [left, top, width, height] box."""

    return box[0] + box[2] / 2, box[1] + box[3] / 2


def _mask_centroid(mask):
    """Normalised (x, y) centre of a boolean mask, or None when it is empty."""

    rows, columns = np.nonzero(mask)
    if len(columns) == 0:
        return None
    return columns.mean() / mask.shape[1], rows.mean() / mask.shape[0]


def _nearest_boxes(boxes, centre, k):
    """The k boxes whose centres lie closest to `centre`, nearest first."""

    distance = lambda box: sum((a - b) ** 2 for a, b in zip(_box_centre(box), centre))
    return sorted(boxes, key=distance)[:k]


def _search_centre(proposals, target_box):
    """Where to look for distractors, or None when the frame offers nothing to look from.

    Taken from where the proposals actually are, so a proposal that drifted onto a neighbour finds that
    neighbour among its own k. Falls back to the annotated box when every proposal is empty."""

    centre = _mask_centroid(proposals.any(axis=0))
    if centre is not None:
        return centre
    return _box_centre(target_box) if float(target_box[2]) > 0.0 else None


# ---------------------------------------------------------------------------------------
# scoring the proposals against box-prompted pseudo-ground-truth
# ---------------------------------------------------------------------------------------

def _pseudo_ground_truth(model, image_features, box):
    """SAM's segmentation of one annotated box, the mask a frame's proposals are scored against."""

    prompt = convert_bbox(np.asarray(box, np.float32))
    mask, _, _ = model.initialize_video_masking(image_features, prompt)
    return (mask.squeeze() > 0.0).cpu().numpy()


def _mask_ious(proposals, truth):
    """IoU of each proposal against one pseudo-GT mask."""

    scores = np.zeros(len(proposals), np.float32)
    for index, proposal in enumerate(proposals):
        if proposal.shape != truth.shape:
            resized = cv2.resize(proposal.astype(np.uint8), truth.shape[::-1], interpolation=cv2.INTER_NEAREST)
            proposal = resized > 0
        union = (proposal | truth).sum()
        scores[index] = float((proposal & truth).sum() / union) if union > 0 else 0.0
    return scores


def _image_features(model, frames, precomputed_features, chunk):
    """One frame's SAM image encoding at a time: cached ones reused, the rest encoded in batches.

    The tracker has already encoded these frames during its rollout, so reusing its cache is what keeps
    labelling from costing a second pass over the image encoder."""

    if precomputed_features is not None:
        device = next(model.image_encoder.parameters()).device
        for frame in range(len(frames)):
            yield [features.to(device) for features in precomputed_features[frame]]
        return

    prepared = [model.image_encoder.prepare_image(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), 1024, True) for frame in frames]
    for start in range(0, len(frames), chunk):
        batch = torch.cat(prepared[start:start + chunk], dim=0)
        encoded = model.image_encoder(batch)
        for offset in range(encoded[0].shape[0]):
            yield [features[offset:offset + 1] for features in encoded]


@torch.inference_mode()
def pseudo_iou_labels(model, frames, predicted_masks, target_boxes, distractor_boxes, occlusions, k=3, chunk=4, precomputed_features=None, include_occluded=False, truth_cache=None):
    """target_iou (n, proposals) and distractor_iou (n, proposals, k).

    `predicted_masks` is (n, proposals, h, w), or (n, h, w) read as a single proposal. `distractor_boxes`
    is one list of clean boxes per frame, from `distractor_boxes_per_frame`.

    Scoring every proposal costs the same SAM decodes as scoring one: a box's pseudo-GT does not depend on
    which proposal it is measured against, so each box is prompted once per frame and intersected with all
    of them. The distractors are likewise picked once per frame, which is what makes column j the same
    person for every proposal and the labels comparable across them.

    `truth_cache` extends that same reasoning across a corruption sweep. A box's pseudo-GT depends only on
    the frame and the box, neither of which the corruption touches -- the target box is identical in every
    rollout and the nearest distractors are usually the same people -- so a dict shared between calls turns
    five identical decodes into one. These decodes are individually tiny and unbatched, so the pipeline
    spends its time in launch overhead rather than on the GPU; removing most of them is the cheapest
    speedup available, and the masks returned are bit-identical either way.

    `include_occluded` keeps occluded frames for distractor scoring only -- the target has no box there, but
    a distractor overlap while it is hidden is an unambiguous capture."""

    masks = np.asarray(predicted_masks)
    if masks.ndim == 3:
        masks = masks[:, None]
    frame_count, proposal_count = len(frames), masks.shape[1]
    target_iou = np.zeros((frame_count, proposal_count), np.float32)
    distractor_iou = np.zeros((frame_count, proposal_count, k), np.float32)

    def ground_truth(frame, image_features, box):
        """This box's pseudo-GT on this frame, taken from the cache once the sweep has decoded it."""

        if truth_cache is None:
            return _pseudo_ground_truth(model, image_features, box)

        key = (frame, np.asarray(box, np.float32).tobytes())
        if key not in truth_cache:
            truth_cache[key] = _pseudo_ground_truth(model, image_features, box)
        return truth_cache[key]

    for frame, image_features in enumerate(_image_features(model, frames, precomputed_features, chunk)):
        if occlusions[frame] > 0.5 and not include_occluded:
            continue

        proposals = masks[frame] > 0
        target_box = target_boxes[frame]
        if float(target_box[2]) > 0.0:
            target_truth = ground_truth(frame, image_features, target_box)
            target_iou[frame] = _mask_ious(proposals, target_truth)

        centre = _search_centre(proposals, target_box)
        if centre is None:
            continue

        for column, box in enumerate(_nearest_boxes(distractor_boxes[frame], centre, k)):
            distractor_truth = ground_truth(frame, image_features, box)
            distractor_iou[frame, :, column] = _mask_ious(proposals, distractor_truth)

    return target_iou, distractor_iou
