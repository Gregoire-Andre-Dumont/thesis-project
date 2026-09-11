"""How many claim_1 anchors are cut off by the frame edge?

`adjust_bounding_boxes` clips BOTH the amodal and the visible box to the image, so a target half outside
the frame has amodal == visible there and visible/amodal ~= 1 -- `min_visible_ratio` cannot detect
truncation at all, however strict it is set. The only surviving signal is that the box touches a border.

A truncated anchor seeds the memory bank from a sliver of the target, so it is the one frame in the clip
where a bad reference costs the most. This counts how often it happens and how much area is involved.
"""
import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from claim_1 import test_trajectories
from create_anchor_dataset import anchor_trajectory_index

EDGE = 0.002          # a box within this fraction of a border counts as touching it (~2px at 1080p)


@hydra.main(config_path="../conf", config_name="experiments/claim_1", version_base=None)
def run(config: DictConfig):
    detection_data = hydra.utils.instantiate(config.detection_data)
    person_path = hydra.utils.instantiate(config.person_path)
    detection_data.load_frames = False                      # boxes only -- no video decoding needed

    rows = []
    for trajectory in test_trajectories(person_path, config.n_traj):
        video, person, anchor_frame = trajectory[0], int(trajectory[1]), int(trajectory[2])
        detection_data.initialize_target(video, person)
        index = anchor_trajectory_index(detection_data, anchor_frame)
        if index is None:
            continue
        x, y, w, h = [float(v) for v in detection_data.bboxes_norm[index]]
        if w <= 0:
            continue
        touches = [x <= EDGE, y <= EDGE, x + w >= 1 - EDGE, y + h >= 1 - EDGE]
        rows.append((sum(touches), touches[0] or touches[2], touches[1] or touches[3], w * h,
                     float(trajectory[3])))

    data = np.array(rows, dtype=float)
    n = len(data)
    any_edge = data[:, 0] > 0
    print(f"anchors examined: {n}\n")
    print(f"  touching ANY frame border      {int(any_edge.sum()):4d}   ({any_edge.mean():5.1%})")
    print(f"    left/right border            {int(data[:, 1].sum()):4d}   ({data[:, 1].mean():5.1%})")
    print(f"    top/bottom border            {int(data[:, 2].sum()):4d}   ({data[:, 2].mean():5.1%})")
    print(f"  touching TWO OR MORE borders   {int((data[:, 0] >= 2).sum()):4d}   ({(data[:, 0] >= 2).mean():5.1%})")

    # A tall, narrow box against a side border is the signature of a person sliced by the frame edge.
    sliver = any_edge & (data[:, 3] > 0)
    if sliver.any():
        print(f"\n  median box area, edge-touching anchors  {np.median(data[any_edge, 3]):.4f} of frame")
        print(f"  median box area, interior anchors       {np.median(data[~any_edge, 3]):.4f} of frame")
    print(f"\n  mean distractor overlap: edge {data[any_edge, 4].mean():.3f}   "
          f"interior {data[~any_edge, 4].mean():.3f}")


if __name__ == "__main__":
    run()
