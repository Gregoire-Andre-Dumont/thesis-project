"""Why does the memory oracle score 0.000 on clips the sam baseline tracks perfectly?

Rolls one named trajectory through sam and the memory oracle (identical mask selection -- only the commit gate
differs) and prints, per frame: the GT-verified commit IoU, whether the frame was committed, and each arm's box
IoU vs the GT box. If the FIFO-freeze hypothesis is right we should see commits stop at some frame and never
resume, with the memory arm's IoU collapsing from that point while sam's stays high.
"""
import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from claim_1 import load_window, first_occlusion_frame, visible_frames, frame_ious, coverage
from src.utils.compute_iou import compute_iou

VIDEO, PERSON = "uid_vid_00157.mp4", 5
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

    anchor = {(v, int(p)): int(a) for v, p, a in zip(
        person_path.selected_video_names.tolist(), person_path.selected_person_ids.tolist(),
        person_path.selected_anchor_video_frames.tolist())}[(VIDEO, PERSON)]

    warmup, _ = load_window(detection_data, (VIDEO, PERSON, anchor), config.max_frames)
    occlusions = detection_data.occlusions[warmup:]
    boxes = detection_data.bboxes_norm[warmup:]
    first = first_occlusion_frame(occlusions)

    cache = {}
    sam.frame_cache = memory.frame_cache = cache
    sam_masks = sam.predict_masks(detection_data).numpy()[warmup:]
    mem_masks = memory.predict_masks(detection_data).numpy()[warmup:]

    sam_iou = compute_iou(boxes, sam_masks)
    mem_iou = compute_iou(boxes, mem_masks)
    commit_iou = memory.commit_iou.numpy()[warmup:]
    committed = {f - warmup for f in memory.committed_frames if f >= warmup}
    sam_commits = set(np.flatnonzero(sam.update_memory.numpy()[warmup:]).tolist())

    print(f"\n{VIDEO} p{PERSON}  frames={len(occlusions)}  occluded={int((occlusions >= 0.5).sum())}  "
          f"first_occlusion={first}  commit_threshold={THRESHOLD}")
    print(f"sam  coverage={coverage(frame_ious(sam_masks, occlusions, boxes, first)):.3f}   "
          f"memory coverage={coverage(frame_ious(mem_masks, occlusions, boxes, first)):.3f}")
    print(f"sam committed {len(sam_commits)}/{len(occlusions)} frames, "
          f"memory committed {len(committed)}/{len(occlusions)}")
    if committed:
        print(f"memory commits: first={min(committed)}  last={max(committed)}  "
              f"-> {len(occlusions) - 1 - max(committed)} frames with a FROZEN fifo after the last commit")

    print(f"\n{'frm':>4} {'occ':>4} {'commitIoU':>10} {'mem?':>5} {'sam?':>5} {'samIoU':>7} {'memIoU':>7}")
    for idx in range(len(occlusions)):
        flag = "*" if idx in committed else "."
        sflag = "*" if idx in sam_commits else "."
        print(f"{idx:>4} {int(occlusions[idx] >= 0.5):>4} {commit_iou[idx]:>10.3f} {flag:>5} {sflag:>5} "
              f"{sam_iou[idx]:>7.3f} {mem_iou[idx]:>7.3f}")

    sam.frame_cache = memory.frame_cache = None


if __name__ == "__main__":
    run()
