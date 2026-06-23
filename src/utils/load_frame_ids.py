import json
import numpy as np

def load_frame_ids(meta_path: str, person_id: int):
    """Extract the frame indices of the target person."""

    data = json.load(open(meta_path, "r"))
    n_frames = int(data['metadata']['number_of_frames'])

    # Filter the entities based on the person id
    filter_ids = lambda e: e['id'] == person_id
    filtered_entities = list(filter(filter_ids, data['entities']))

    extract_ids = lambda e: e['blob']['frame_idx']
    target_ids = list(map(extract_ids, filtered_entities))

    return np.array(target_ids), filtered_entities, n_frames

def load_section(amodal_path: str, visible_path: str, person_id: int):
    """"Extract the first and last frame index where the target is visible."""

    amodal_ids, entities, _ = load_frame_ids(amodal_path, person_id)
    visible_ids, _, n_frames = load_frame_ids(visible_path, person_id)

    total_ids = np.concatenate([visible_ids, amodal_ids])
    return np.min(total_ids), np.max(total_ids)