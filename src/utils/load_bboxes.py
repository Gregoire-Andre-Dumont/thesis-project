import json
import numpy as np
from numpy.typing import NDArray
from src.utils.load_frame_ids import load_frame_ids


def load_bboxes(amodal_path: str, visible_path: str, person_id: int, use_amodal=False):
    """Bounding boxes at the ANNOTATED frames only — no interpolation.

    Returns shape `(n_annotated, 4)` aligned with the sorted union of amodal+visible IDs.
    Fully-occluded frames (annotated in amodal but absent from visible) get `[0, 0, 0, 0]` when
    `use_amodal=False`."""

    amodal_ids, amodal_entities, _ = load_frame_ids(amodal_path, person_id)
    visible_ids, visible_entities, n_frames = load_frame_ids(visible_path, person_id)
    start_idx = int(np.min(np.concatenate([visible_ids, amodal_ids])))

    data = json.load(open(visible_path, "r"))
    width = int(data['metadata']['resolution']['width'])
    height = int(data['metadata']['resolution']['height'])

    entities = amodal_entities if use_amodal else visible_entities
    raw_bboxes = np.zeros((n_frames, 4), dtype=np.int64)
    for entity in entities:
        raw_bboxes[entity['blob']['frame_idx'], :] = entity['bb']

    annotated_ids = np.unique(np.concatenate([amodal_ids, visible_ids]))
    return adjust_bounding_boxes(raw_bboxes[annotated_ids], width, height)


def adjust_bounding_boxes(bboxes, video_width, video_height):
    """Clip and normalize the bounding boxes to the resolution."""

    x = np.clip(bboxes[:, 0], 0, video_width - 1)
    y = np.clip(bboxes[:, 1], 0, video_height - 1)

    # Clip width/height so x+w and y+h don't exceed the frame
    w = np.clip(bboxes[:, 2], 0, video_width - x)
    h = np.clip(bboxes[:, 3], 0, video_height - y)

    # Normalize the bounding boxes to the resolution
    x_norm, w_norm = x / video_width, w / video_width
    y_norm, h_norm = y / video_height, h / video_height
    
    return np.stack([x_norm, y_norm, w_norm, h_norm], axis=1)

def convert_bbox(bbox_norm: NDArray[np.float32]):
    """Convert the bounding box to the desired format for SAM 2."""

    x_min, y_min, width, height = bbox_norm
    top_left = [x_min, y_min]

    bottom_right = [x_min + width, y_min + height]
    return np.array([[top_left, bottom_right]])

def convert_bboxes(bboxes_norm: NDArray[np.float32]):
    """Convert multiple bounding boxes to the SAM 2 format."""

    x_min = bboxes_norm[:, 0]
    y_min = bboxes_norm[:, 1]
    x_max = x_min + bboxes_norm[:, 2]
    y_max = y_min + bboxes_norm[:, 3]

    top_left = np.stack([x_min, y_min], axis=1)
    bottom_right = np.stack([x_max, y_max], axis=1)
    
    return np.stack([top_left, bottom_right], axis=1)