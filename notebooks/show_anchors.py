"""Write the claim_1 anchor frames to a folder, one image per anchor, for eyeballing before a run.

Every selection condition is evaluated per frame and the search takes the FIRST qualifying frame, so a
condition does not filter trajectories so much as decide where each clip STARTS. These images are the only
way to see what that produced: each one is an anchor -- the frame the memory bank is seeded from.

Each file is two panels: the whole frame with the target boxed (judge crowding, scene, distance) and a 2.5x
crop around the box (judge pose, occlusion, whether it is even resolvable as a person). Files are numbered
by ASCENDING anchor area, so the most questionable anchors -- the ones sitting on the `min_visible_area`
floor -- sort to the top of the folder.

    python notebooks/show_anchors.py [n] [config.yaml] [out_dir]
"""
import sys
import json
from pathlib import Path

import torch  # noqa: F401 -- before decord, see detection_data
import hydra
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from omegaconf import OmegaConf
from decord import VideoReader, cpu, bridge

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from notebooks.claim_1 import test_trajectories

bridge.set_bridge("native")     # decord's own NDArray; `torch` here would make .asnumpy() unavailable

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
CONFIG = sys.argv[2] if len(sys.argv) > 2 else "conf/experiments/claim_1.yaml"
OUT_DIR = Path(sys.argv[3] if len(sys.argv) > 3 else "data/claim_1/anchors")
SURFACE, INK2, BOX = "#fcfcfb", "#52514e", "#eb6834"

config = OmegaConf.load(CONFIG)
person_path = hydra.utils.instantiate(config.person_path)
print(f"pool: {person_path.total_experiments} trajectories")

# The run's own draw, so these are the clips claim_1 actually rolls out -- not a fresh random sample.
trajectories = test_trajectories(person_path, config.n_traj, config.person_path.random_seed)[:N]
_annotations = {}


def anchor_box(video, person, anchor):
    """The target's visible box on its anchor frame, in native pixels; None if the frame has no box."""

    if video not in _annotations:
        _annotations[video] = json.load(
            open(Path(config.detection_data.visible_directory) / f"{video}.json"))
    for entity in _annotations[video]["entities"]:
        if entity["id"] == person and int(entity["blob"]["frame_idx"]) == int(anchor):
            return [float(v) for v in entity["bb"]]
    return None


def measured():
    """(video, person, anchor, box, area@1024) per trajectory, grouped by video so each video's reader is
    opened once, and dropping anchors whose frame carries no visible box."""

    rows = []
    for video, person, anchor, _overlap in sorted(trajectories):
        box = anchor_box(video, int(person), anchor)
        if box is None:
            print(f"  no box at anchor: {video} #{person} f{anchor}")
            continue
        resolution = _annotations[video]["metadata"]["resolution"]
        scale = 1024 / max(float(resolution["width"]), float(resolution["height"]))
        rows.append((video, int(person), int(anchor), box, box[2] * box[3] * scale ** 2))
    return rows


def write_anchor(rank, video, person, anchor, box, area, frame):
    """One two-panel PNG: whole frame with the box, and a 2.5x crop around it."""

    x, y, width, height = box
    pad_x, pad_y = width * 0.75, height * 0.75
    left, top = max(0, x - pad_x), max(0, y - pad_y)
    right, bottom = min(frame.shape[1], x + width + pad_x), min(frame.shape[0], y + height + pad_y)

    figure, (whole, crop) = plt.subplots(
        1, 2, figsize=(13, 5), facecolor=SURFACE,
        gridspec_kw={"width_ratios": [frame.shape[1] / frame.shape[0], (right - left) / (bottom - top)]})

    whole.imshow(frame)
    whole.add_patch(Rectangle((x, y), width, height, fill=False, edgecolor=BOX, linewidth=1.8))
    crop.imshow(frame[int(top):int(bottom), int(left):int(right)])
    crop.add_patch(Rectangle((x - left, y - top), width, height, fill=False, edgecolor=BOX, linewidth=2.2))
    for axis in (whole, crop):
        axis.set_facecolor(SURFACE)
        axis.axis("off")

    figure.suptitle(f"{video}   #{person}   frame {anchor}   {area:.0f} px² @1024   "
                    f"{width:.0f}×{height:.0f} native", fontsize=10, color=INK2)
    figure.tight_layout(rect=[0, 0, 1, 0.95])
    name = f"{rank:03d}_{area:06.0f}px_{video.replace('.mp4', '')}_p{person}_f{anchor}.png"
    figure.savefig(OUT_DIR / name, dpi=110, facecolor=SURFACE)
    plt.close(figure)


OUT_DIR.mkdir(parents=True, exist_ok=True)
for existing in OUT_DIR.glob("*.png"):
    existing.unlink()

rows = measured()
rank_of = {row[:3]: rank for rank, row in enumerate(sorted(rows, key=lambda row: row[4]), start=1)}

reader, open_video = None, None
for video, person, anchor, box, area in rows:
    if video != open_video:                      # rows are video-sorted, so each file opens once
        reader = VideoReader(str(Path(config.detection_data.video_directory) / video), ctx=cpu(0),
                             num_threads=8)
        open_video = video
    frame = reader.get_batch([int(anchor)]).asnumpy()[0]
    write_anchor(rank_of[(video, person, anchor)], video, person, anchor, box, area, frame)

areas = np.array([row[4] for row in rows])
print(f"\nwrote {len(rows)} anchors to {OUT_DIR}")
print(f"  anchor area @1024:  min {areas.min():.0f}   q1 {np.percentile(areas, 25):.0f}   "
      f"median {np.median(areas):.0f}   q3 {np.percentile(areas, 75):.0f}   max {areas.max():.0f}")
print(f"  under 1000 px²: {(areas < 1000).sum()}   under 2000 px²: {(areas < 2000).sum()}")
