"""Does the corrected occlusion definition rescue any of the memory oracle's zero-coverage clips?

Runs the clips where sam tracks fine but the memory oracle scored exactly 0.000, and reports for each: the
occlusion count under the new (label-only) definition, both arms' coverage, and where the commit stream stops.
If the freeze span still covers most of the clip, the failure is the commit gate, not the annotations.
"""
import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from claim_1 import load_window, first_occlusion_frame, visible_frames, frame_ious, coverage

# (video, person, gap_fraction of its occlusions under the OLD definition)
CLIPS = [("uid_vid_00157.mp4", 5, 0.29), ("uid_vid_00113.mp4", 4, 0.56), ("uid_vid_00177.mp4", 75, 0.00),
         ("uid_vid_00180.mp4", 16, 0.15), ("uid_vid_00043.mp4", 18, 0.00), ("uid_vid_00042.mp4", 109, 0.29)]
THRESHOLD = 0.3
SAM_CONFIG = "conf/trackers/baselines/sam_baseline.yaml"


@hydra.main(config_path="../conf", config_name="experiments/claim_1", version_base=None)
def run(config: DictConfig):
    detection_data = hydra.utils.instantiate(config.detection_data)
    person_path = hydra.utils.instantiate(config.person_path)
    memory = hydra.utils.instantiate(OmegaConf.load(config.oracle_config).tracker)
    sam = hydra.utils.instantiate(OmegaConf.load(SAM_CONFIG).tracker)
    sam.label_mask_iou = False
    memory.iou_threshold = THRESHOLD

    anchors = {(v, int(p)): int(a) for v, p, a in zip(
        person_path.selected_video_names.tolist(), person_path.selected_person_ids.tolist(),
        person_path.selected_anchor_video_frames.tolist())}

    print(f"{'clip':>26} {'gap':>5} {'occ':>4} {'scor':>5} {'sam':>6} {'mem':>6} {'commits':>8} {'frozen':>7}")
    for video, person, gap in CLIPS:
        anchor = anchors.get((video, person))
        if anchor is None:
            print(f"{video + ' p' + str(person):>26} -- no longer selected under the corrected occlusion definition")
            continue
        window = load_window(detection_data, (video, person, anchor), config.max_frames)
        if window is None:
            continue
        warmup, _ = window
        occlusions = detection_data.occlusions[warmup:]
        boxes = detection_data.bboxes_norm[warmup:]
        first = first_occlusion_frame(occlusions)

        cache = {}
        sam.frame_cache = memory.frame_cache = cache
        sam_cov = coverage(frame_ious(sam.predict_masks(detection_data).numpy()[warmup:], occlusions, boxes, first))
        mem_cov = coverage(frame_ious(memory.predict_masks(detection_data).numpy()[warmup:], occlusions, boxes, first))
        sam.frame_cache = memory.frame_cache = None

        committed = [f - warmup for f in memory.committed_frames if f >= warmup]
        frozen = len(occlusions) - 1 - max(committed) if committed else len(occlusions)
        print(f"{video + ' p' + str(person):>26} {gap:>5.2f} {int((occlusions >= 0.5).sum()):>4} "
              f"{len(visible_frames(occlusions, boxes, first)):>5} {sam_cov:>6.3f} {mem_cov:>6.3f} "
              f"{len(committed):>8} {frozen:>7}")


if __name__ == "__main__":
    run()
