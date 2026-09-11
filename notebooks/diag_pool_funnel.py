"""Diagnostic: where do PersonPath trajectories get eliminated?

Replays PersonPath.select_targets with a counter at every filter so the pool size can be attributed to
a specific condition instead of guessed at. Each occlusion sub-condition is evaluated INDEPENDENTLY on
the survivors of the anchor stage, so the three occlusion numbers are marginal costs (they overlap) --
the funnel's running total applies them in the order select_targets does.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.typing.person_path import (_anchor_video_frame, _occlusion_flags, _group_entities_by_person_id)

config = OmegaConf.load(sys.argv[1] if len(sys.argv) > 1 else "conf/experiments/claim_1.yaml").person_path
main = Path(config.main_directory)
occ_low, occ_high = list(config.occlusion_ranges)[0], list(config.occlusion_ranges)[-1]

counts = dict(persons=0, targets=0, anchor=0, occ_range=0, first_occ=0, after_occ=0, selected=0)
occ_lengths, fail_reason = [], dict(too_short=0, too_long=0, never=0)

for video_name in sorted(os.listdir(main / "videos")):
    amodal = json.load(open(main / "amodal" / f"{video_name}.json"))
    visible = json.load(open(main / "visible" / f"{video_name}.json"))
    resolution = visible["metadata"]["resolution"]
    scale = config.resize_resolution if "resize_resolution" in config else 1024
    scale = scale / max(resolution["width"], resolution["height"])
    min_area = config.min_visible_area / scale ** 2

    amodal_ids = {e["id"] for e in amodal["entities"] if not any(l in config.non_targets for l in e["labels"])}
    visible_ids = {e["id"] for e in visible["entities"] if not any(l in config.non_targets for l in e["labels"])}
    amodal_by_id = _group_entities_by_person_id(amodal["entities"])
    visible_by_id = _group_entities_by_person_id(visible["entities"])

    boxes_by_frame = {}
    for entity in visible["entities"]:
        boxes_by_frame.setdefault(entity["blob"]["frame_idx"], []).append((entity["id"], entity["bb"]))

    counts["persons"] += len({e["id"] for e in visible["entities"]})
    targets = amodal_ids & visible_ids
    counts["targets"] += len(targets)

    for person_id in targets:
        anchor = _anchor_video_frame(amodal_by_id.get(person_id, []), visible_by_id.get(person_id, []),
                                     boxes_by_frame, person_id, min_area, config.max_distractor_overlap,
                                     getattr(config, "min_distractor_overlap", 0.0), float("inf"))
        if anchor is None:
            continue
        counts["anchor"] += 1

        flags = _occlusion_flags(amodal_by_id.get(person_id, []), visible_by_id.get(person_id, []), anchor[0])
        n_occluded = int(flags.sum())
        occ_lengths.append(n_occluded)

        in_range = occ_low < n_occluded < occ_high
        counts["occ_range"] += in_range
        if not in_range:
            fail_reason["never" if n_occluded == 0 else
                        ("too_short" if n_occluded <= occ_low else "too_long")] += 1

        first = int(np.argmax(flags > 0))
        run_in = first > config.first_occ_min
        counts["first_occ"] += run_in
        after = int(np.sum(flags[first:] == 0)) > config.n_after_occlusion
        counts["after_occ"] += after
        counts["selected"] += in_range and run_in and after

print(f"config: occlusion_ranges=[{occ_low}, {occ_high}]  n_after_occlusion={config.n_after_occlusion}  "
      f"first_occ_min={config.first_occ_min}  min_visible_area={config.min_visible_area}  "
      f"max_distractor_overlap={config.max_distractor_overlap}\n")
print(f"  annotated person-tracks                     {counts['persons']:6d}")
print(f"  target-labelled (in amodal AND visible)     {counts['targets']:6d}"
      f"   ({counts['targets']/max(counts['persons'],1):5.1%})")
print(f"  has a valid ANCHOR frame                    {counts['anchor']:6d}"
      f"   ({counts['anchor']/max(counts['targets'],1):5.1%} of targets)")
print(f"\n  of those {counts['anchor']} anchored tracks, each occlusion condition alone:")
print(f"    occluded count in ({occ_low}, {occ_high})            {counts['occ_range']:6d}"
      f"   ({counts['occ_range']/max(counts['anchor'],1):5.1%})")
print(f"    first occlusion > {config.first_occ_min} frames in           {counts['first_occ']:6d}"
      f"   ({counts['first_occ']/max(counts['anchor'],1):5.1%})")
print(f"    > {config.n_after_occlusion} visible frames after           {counts['after_occ']:6d}"
      f"   ({counts['after_occ']/max(counts['anchor'],1):5.1%})")
print(f"\n  ALL THREE (= pool size)                     {counts['selected']:6d}")
print(f"\n  why the occluded-count test fails: never occluded {fail_reason['never']},"
      f"  <= {occ_low} frames {fail_reason['too_short']},  >= {occ_high} frames {fail_reason['too_long']}")

lengths = np.array(occ_lengths)
if len(lengths):
    print(f"\n  occluded-frame count over anchored tracks: median {np.median(lengths):.0f}  "
          f"q75 {np.quantile(lengths,.75):.0f}  q90 {np.quantile(lengths,.90):.0f}  max {lengths.max():.0f}")
