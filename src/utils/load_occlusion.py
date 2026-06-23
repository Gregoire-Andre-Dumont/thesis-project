import numpy as np

from src.utils.load_frame_ids import load_frame_ids


def load_occlusions(amodal_path: str, visible_path: str, person_id: int):
    """Occlusion flags at the ANNOTATED frames only — no interpolation"""

    amodal_ids, entities, _ = load_frame_ids(amodal_path, person_id)
    visible_ids, _, n_frames = load_frame_ids(visible_path, person_id)

    raw_occlusions = np.zeros(n_frames, dtype=np.float32)
    for entity in entities:
        if "fully_occluded" in entity['labels']:
            raw_occlusions[entity['blob']['frame_idx']] = 1

    annotated_ids = np.unique(np.concatenate([amodal_ids, visible_ids]))
    return raw_occlusions[annotated_ids]





