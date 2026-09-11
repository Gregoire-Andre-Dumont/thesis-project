"""Which filter relaxation actually grows the PersonPath pool?

Enumerates every anchored track ONCE, recording (occluded count, first-occlusion index, visible frames
after), then evaluates many parameter combinations against that cache instantly. One pass over the
dataset answers the whole grid, instead of re-enumerating per setting.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.typing.person_path import _anchor_video_frame, _occlusion_flags, _group_entities_by_person_id

config = OmegaConf.load(sys.argv[1] if len(sys.argv) > 1 else "conf/experiments/claim_1.yaml").person_path
main = Path(config.main_directory)

tracks = []                                     # (n_occluded, first_occlusion_index, visible_after)
for video_name in sorted(os.listdir(main / "videos")):
    amodal = json.load(open(main / "amodal" / f"{video_name}.json"))
    visible = json.load(open(main / "visible" / f"{video_name}.json"))
    resolution = visible["metadata"]["resolution"]
    scale = 1024 / max(resolution["width"], resolution["height"])
    min_area = config.min_visible_area / scale ** 2

    amodal_ids = {e["id"] for e in amodal["entities"] if not any(l in config.non_targets for l in e["labels"])}
    visible_ids = {e["id"] for e in visible["entities"] if not any(l in config.non_targets for l in e["labels"])}
    amodal_by_id = _group_entities_by_person_id(amodal["entities"])
    visible_by_id = _group_entities_by_person_id(visible["entities"])

    boxes_by_frame = {}
    for entity in visible["entities"]:
        boxes_by_frame.setdefault(entity["blob"]["frame_idx"], []).append((entity["id"], entity["bb"]))

    for person_id in amodal_ids & visible_ids:
        anchor = _anchor_video_frame(amodal_by_id.get(person_id, []), visible_by_id.get(person_id, []),
                                     boxes_by_frame, person_id, min_area, config.max_distractor_overlap,
                                     getattr(config, "min_distractor_overlap", 0.0), float("inf"))
        if anchor is None:
            continue
        flags = _occlusion_flags(amodal_by_id.get(person_id, []), visible_by_id.get(person_id, []), anchor[0])
        first = int(np.argmax(flags > 0))
        tracks.append((int(flags.sum()), first, int(np.sum(flags[first:] == 0))))

data = np.array(tracks)
n_occluded, first_occlusion, visible_after = data[:, 0], data[:, 1], data[:, 2]


def pool(low, high, first_min, after):
    return int(((n_occluded > low) & (n_occluded < high) &
                (first_occlusion > first_min) & (visible_after > after)).sum())


BASE = (5, 75, 5, 40)
print(f"anchored tracks: {len(data)}")
print(f"\nbaseline  occ({BASE[0]},{BASE[1]})  first>{BASE[2]}  after>{BASE[3]}   ->  {pool(*BASE)}\n")
print("relaxing ONE parameter at a time:")
for label, args in [
    (f"occ lower bound  5 -> 3", (3, 75, 5, 40)),
    (f"occ lower bound  5 -> 2", (2, 75, 5, 40)),
    (f"occ lower bound  5 -> 1", (1, 75, 5, 40)),
    (f"occ upper bound 75 -> 150", (5, 150, 5, 40)),
    (f"first_occ_min    5 -> 3", (5, 75, 3, 40)),
    (f"first_occ_min    5 -> 1", (5, 75, 1, 40)),
    (f"n_after_occ     40 -> 30", (5, 75, 5, 30)),
    (f"n_after_occ     40 -> 20", (5, 75, 5, 20)),
    (f"n_after_occ     40 -> 10", (5, 75, 5, 10)),
]:
    print(f"  {label:28s} ->  {pool(*args):5d}   ({pool(*args) - pool(*BASE):+d})")

print("\ncombinations:")
for label, args in [
    ("occ>3, after>30", (3, 75, 5, 30)),
    ("occ>3, after>20", (3, 75, 5, 20)),
    ("occ>2, first>3, after>20", (2, 75, 3, 20)),
    ("occ>2, first>1, after>10", (2, 75, 1, 10)),
    ("occ>1, first>1, after>10", (1, 150, 1, 10)),
]:
    print(f"  {label:28s} ->  {pool(*args):5d}")
