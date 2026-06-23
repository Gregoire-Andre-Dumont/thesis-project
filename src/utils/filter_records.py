"""Trajectory-level filters reused across thesis figures and experiment visualizers.

The main entry point is `filter_records_by_initial_target_size`, which drops every
record whose target's *initial* visible bbox is too small after the offline resize to a
fixed longest side (defaults to 1024 px to match `conf/offline_training.yaml`'s
`resize_resolution`). Sizes are computed at the resized resolution so the threshold
means the same thing for every video regardless of native dataset resolution."""

import json
from pathlib import Path
from functools import lru_cache


DEFAULT_VISIBLE_DIRECTORY = "data/person_path/visible"
DEFAULT_AMODAL_DIRECTORY = "data/person_path/amodal"
DEFAULT_RESIZE_RESOLUTION = 1024
DEFAULT_MIN_INITIAL_AREA = 1024  # 32 x 32 px at resize_resolution = 1024
DEFAULT_OVERLAP_IOU_THRESHOLD = 0.5


@lru_cache(maxsize=None)
def _read_visible_json(visible_json_path):
    with open(visible_json_path, "r") as handle:
        return json.load(handle)


@lru_cache(maxsize=None)
def _read_frame_to_person_bbox_table(video_json_path):
    """Return `{frame_idx: {person_id: (x, y, w, h)}}` parsed from a PersonPath22 JSON
    file (works for either `visible/` or `amodal/`). Cached so repeated calls for the
    same video (across target IDs) parse the JSON only once."""

    with open(video_json_path, "r") as handle:
        data = json.load(handle)
    frame_to_persons = {}
    for entity in data["entities"]:
        frame_index = entity["blob"]["frame_idx"]
        person_id = entity["id"]
        frame_to_persons.setdefault(frame_index, {})[person_id] = tuple(entity["bb"])
    return frame_to_persons


def _bbox_iou_xywh(bbox_a, bbox_b):
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b
    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax + aw, bx + bw)
    inter_y2 = min(ay + ah, by + bh)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area_a = max(0, aw) * max(0, ah)
    area_b = max(0, bw) * max(0, bh)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def num_persons_overlapping_target(amodal_directory, video_name, target_person_id,
                                    iou_threshold=DEFAULT_OVERLAP_IOU_THRESHOLD):
    """Count unique OTHER person IDs whose amodal bbox has IoU > `iou_threshold` with
    the target's amodal bbox at some shared frame. Proxy for the number of distractors
    that physically cross the target's spatial footprint during the trajectory. Returns
    `None` if the amodal JSON is missing."""

    amodal_path = Path(amodal_directory) / f"{video_name}.json"
    if not amodal_path.exists():
        return None
    frame_to_persons = _read_frame_to_person_bbox_table(str(amodal_path))
    overlapping_person_ids = set()
    for persons_at_frame in frame_to_persons.values():
        target_bbox = persons_at_frame.get(target_person_id)
        if target_bbox is None:
            continue
        for other_person_id, other_bbox in persons_at_frame.items():
            if other_person_id == target_person_id:
                continue
            if other_person_id in overlapping_person_ids:
                continue
            if _bbox_iou_xywh(target_bbox, other_bbox) > iou_threshold:
                overlapping_person_ids.add(other_person_id)
    return len(overlapping_person_ids)


def initial_bbox_area_at_resize(visible_directory, video_name, person_id,
                                resize_resolution=DEFAULT_RESIZE_RESOLUTION):
    """Return the area (in pixels) of the target's first visible bbox after the longest
    side is resized to `resize_resolution`. Returns `None` if the (video, person_id) pair
    has no entry in the visible JSON or the JSON is missing."""

    visible_json_path = Path(visible_directory) / f"{video_name}.json"
    if not visible_json_path.exists():
        return None
    data = _read_visible_json(str(visible_json_path))
    image_width = int(data["metadata"]["resolution"]["width"])
    image_height = int(data["metadata"]["resolution"]["height"])
    scale = resize_resolution / max(image_width, image_height)

    earliest_frame_index = None
    earliest_bbox = None
    for entity in data["entities"]:
        if entity["id"] != person_id:
            continue
        frame_index = entity["blob"]["frame_idx"]
        if earliest_frame_index is None or frame_index < earliest_frame_index:
            earliest_frame_index = frame_index
            earliest_bbox = entity["bb"]
    if earliest_bbox is None:
        return None
    _, _, bbox_width, bbox_height = earliest_bbox
    return float(bbox_width * bbox_height) * (scale ** 2)


def initial_amodal_bbox_area_at_resize(amodal_directory, video_name, person_id,
                                         resize_resolution=DEFAULT_RESIZE_RESOLUTION):
    """Same as `initial_bbox_area_at_resize` but reads from the amodal JSON. The amodal
    bbox includes occluded parts of the target, so the area reflects the target's *true*
    size at the first annotated frame even when the visible bbox is clipped or missing.

    Note: the parameter name `amodal_directory` is positional-compatible with the
    `visible_directory` slot in `large_target_keys` / `small_target_keys`, so this
    function can be passed directly as `area_fn` (with `visible_directory=` pointing at
    `data/person_path/amodal` instead)."""

    amodal_json_path = Path(amodal_directory) / f"{video_name}.json"
    if not amodal_json_path.exists():
        return None
    data = _read_visible_json(str(amodal_json_path))
    image_width = int(data["metadata"]["resolution"]["width"])
    image_height = int(data["metadata"]["resolution"]["height"])
    scale = resize_resolution / max(image_width, image_height)

    earliest_frame_index = None
    earliest_bbox = None
    for entity in data["entities"]:
        if entity["id"] != person_id:
            continue
        frame_index = entity["blob"]["frame_idx"]
        if earliest_frame_index is None or frame_index < earliest_frame_index:
            earliest_frame_index = frame_index
            earliest_bbox = entity["bb"]
    if earliest_bbox is None:
        return None
    _, _, bbox_width, bbox_height = earliest_bbox
    return float(bbox_width * bbox_height) * (scale ** 2)


def _visible_bbox_areas_at_resize(visible_directory, video_name, person_id,
                                    resize_resolution=DEFAULT_RESIZE_RESOLUTION):
    visible_json_path = Path(visible_directory) / f"{video_name}.json"
    if not visible_json_path.exists():
        return None
    data = _read_visible_json(str(visible_json_path))
    image_width = int(data["metadata"]["resolution"]["width"])
    image_height = int(data["metadata"]["resolution"]["height"])
    scale = resize_resolution / max(image_width, image_height)
    areas = []
    for entity in data["entities"]:
        if entity["id"] != person_id:
            continue
        _, _, bbox_width, bbox_height = entity["bb"]
        areas.append(float(bbox_width * bbox_height) * (scale ** 2))
    return areas if areas else None


def mean_visible_bbox_area_at_resize(visible_directory, video_name, person_id,
                                       resize_resolution=DEFAULT_RESIZE_RESOLUTION):
    """Return the *mean* area (in pixels) of the target's visible bboxes across every
    frame the target appears in, after the longest side is resized to `resize_resolution`.
    Returns `None` if the (video, person_id) pair has no entry in the visible JSON.

    Companion to `initial_bbox_area_at_resize` — using the mean smooths over noisy
    first-frame crops (e.g., the target entering the scene partially visible) and gives
    a more representative size estimate for binning."""

    areas = _visible_bbox_areas_at_resize(visible_directory, video_name, person_id,
                                            resize_resolution=resize_resolution)
    if areas is None:
        return None
    return float(sum(areas) / len(areas))


def max_visible_bbox_area_at_resize(visible_directory, video_name, person_id,
                                      resize_resolution=DEFAULT_RESIZE_RESOLUTION):
    """Return the *maximum* area (in pixels) of the target's visible bboxes across every
    frame the target appears in, after the longest side is resized to `resize_resolution`.
    Returns `None` if the (video, person_id) pair has no entry in the visible JSON.

    Useful as a 'best-case size' separator: a trajectory is considered small only if the
    target was small *even at its biggest* — robust to brief shrinkage from motion blur
    or partial entry, and isolates the truly always-small targets."""

    areas = _visible_bbox_areas_at_resize(visible_directory, video_name, person_id,
                                            resize_resolution=resize_resolution)
    if areas is None:
        return None
    return float(max(areas))


def large_target_keys(records_by_tracker, visible_directory=DEFAULT_VISIBLE_DIRECTORY,
                      min_initial_area=DEFAULT_MIN_INITIAL_AREA,
                      resize_resolution=DEFAULT_RESIZE_RESOLUTION,
                      area_fn=initial_bbox_area_at_resize):
    """Return the set of `(video_name, person_id)` keys whose target's area (under
    `area_fn`) is at least `min_initial_area` px at the offline resize resolution. Pairs
    missing from the visible JSON are dropped (treated as 'cannot verify' → small).

    `area_fn(visible_directory, video_name, person_id, resize_resolution=...)` defaults
    to the target's *initial* bbox area; pass `mean_visible_bbox_area_at_resize` instead
    to filter by mean area across all visible frames."""

    all_keys = set()
    for records in records_by_tracker.values():
        for record in records:
            all_keys.add((record["video_name"], record["person_id"]))

    keep = set()
    for video_name, person_id in all_keys:
        area = area_fn(visible_directory, video_name, person_id,
                       resize_resolution=resize_resolution)
        if area is None:
            continue
        if area >= min_initial_area:
            keep.add((video_name, person_id))
    return keep


def small_target_keys(records_by_tracker, visible_directory=DEFAULT_VISIBLE_DIRECTORY,
                      max_initial_area=DEFAULT_MIN_INITIAL_AREA,
                      resize_resolution=DEFAULT_RESIZE_RESOLUTION,
                      area_fn=initial_bbox_area_at_resize):
    """Return the set of `(video_name, person_id)` keys whose target's area (under
    `area_fn`) is *strictly less than* `max_initial_area` px at the offline resize
    resolution. Companion to `large_target_keys` — see that docstring for `area_fn`."""

    all_keys = set()
    for records in records_by_tracker.values():
        for record in records:
            all_keys.add((record["video_name"], record["person_id"]))

    keep = set()
    for video_name, person_id in all_keys:
        area = area_fn(visible_directory, video_name, person_id,
                       resize_resolution=resize_resolution)
        if area is None:
            continue
        if area < max_initial_area:
            keep.add((video_name, person_id))
    return keep


def num_persons_in_video(visible_directory, video_name):
    """Return the number of unique person IDs appearing in `video_name`'s visible JSON.
    A proxy for scene complexity / distractor count. Returns `None` if the JSON is missing."""

    visible_json_path = Path(visible_directory) / f"{video_name}.json"
    if not visible_json_path.exists():
        return None
    data = _read_visible_json(str(visible_json_path))
    return len({entity["id"] for entity in data["entities"]})


def filter_records_by_initial_target_size(records, visible_directory=DEFAULT_VISIBLE_DIRECTORY,
                                          min_initial_area=DEFAULT_MIN_INITIAL_AREA,
                                          resize_resolution=DEFAULT_RESIZE_RESOLUTION):
    """Drop records whose target's initial bbox area at the resize resolution is below
    `min_initial_area`. Convenience wrapper around `large_target_keys` for the single-list
    case."""

    keep = large_target_keys({"_": records}, visible_directory=visible_directory,
                              min_initial_area=min_initial_area,
                              resize_resolution=resize_resolution)
    return [record for record in records
            if (record["video_name"], record["person_id"]) in keep]
