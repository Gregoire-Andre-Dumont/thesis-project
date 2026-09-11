"""Dump the anchor frames claim_1 actually tracks from, each with its ground-truth box drawn.

A sanity check on trajectory selection: the anchor is the frame the memory bank is seeded from, so if it
is mis-boxed, occluded, or on the wrong person, everything downstream inherits that. Reading 50 of them
is the cheapest way to see what the selection filters are really admitting.

The anchor is resolved through claim_1's own `load_window` (with max_frames=1, so only that one frame is
decoded), NOT re-derived here -- a reimplementation could drift from the experiment and show a frame the
tracker never saw.

    python notebooks/dump_anchors.py                      # 50 anchors -> data/claim_1/anchors/
    python notebooks/dump_anchors.py n_anchors=20 out_dir=data/claim_3   # other count / other config
"""
import sys
from pathlib import Path

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from claim_1 import test_trajectories, load_window, first_occlusion_frame

BOX_COLOUR = (52, 175, 27)        # BGR: the same green sam wears in every claim figure
TEXT_COLOUR, SHADOW = (255, 255, 255), (0, 0, 0)


def draw_label(image, text, origin):
    """White text with a dark outline, so it stays readable over any frame content."""

    for colour, thickness in ((SHADOW, 3), (TEXT_COLOUR, 1)):
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, thickness, cv2.LINE_AA)


@hydra.main(config_path="../conf", config_name="experiments/claim_1", version_base=None)
def run(config: DictConfig):
    detection_data = hydra.utils.instantiate(config.detection_data)
    person_path = hydra.utils.instantiate(config.person_path)
    n_anchors = int(config.get("n_anchors", 50))

    out_dir = Path(config.out_dir) / "anchors"
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.png"):                   # a shorter run must not leave older frames behind
        stale.unlink()

    written = 0
    for trajectory in test_trajectories(person_path, config.n_traj):
        if written >= n_anchors:
            break
        video, person, anchor_frame = trajectory[0], int(trajectory[1]), int(trajectory[2])

        window = load_window(detection_data, trajectory[:3], 1)          # decode the anchor frame only
        if window is None:
            continue
        frame = detection_data.frames[0]
        box = detection_data.bboxes_norm[0]
        if float(box[2]) <= 0:                                           # no visible box: nothing to draw
            continue

        image = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2BGR)
        height, width = image.shape[:2]
        x0, y0 = int(round(box[0] * width)), int(round(box[1] * height))
        x1, y1 = int(round((box[0] + box[2]) * width)), int(round((box[1] + box[3]) * height))
        cv2.rectangle(image, (x0, y0), (x1, y1), BOX_COLOUR, 2)

        area = float(box[2] * width * box[3] * height) * (1024 / max(width, height)) ** 2
        draw_label(image, f"{video}  person {person}  frame {anchor_frame}", (12, 26))
        draw_label(image, f"anchor area {area:.0f} px2 @1024   distractor overlap {float(trajectory[3]):.2f}",
                   (12, 50))

        name = f"{written:02d}_{Path(video).stem}_p{person}_f{anchor_frame}.png"
        cv2.imwrite(str(out_dir / name), image)
        written += 1
        print(f"{written:3d}  {name}", flush=True)

    print(f"\nwrote {written} anchor frames to {out_dir}")


if __name__ == "__main__":
    run()
