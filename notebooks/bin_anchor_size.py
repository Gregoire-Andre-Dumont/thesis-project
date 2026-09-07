"""Post-hoc: split claim_1 coverage by ANCHOR VISIBLE AREA (GT-box area px^2 at resize_resolution=1024,
same units as person_path.min_visible_area). Recovers each clip's anchor frame from person_path (CPU only)."""
import sys
import json
from pathlib import Path
import pickle
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.typing.detection_data import DetectionData
from src.typing.person_path import PersonPath
from create_anchor_dataset import anchor_trajectory_index

RESIZE = 1024
MIN_VISIBLE_AREA = 900

s = pickle.load(open("data/claim_1/results.pkl", "rb"))
clips = s["clips"]


def cov(a, t=0.5):
    a = np.asarray(a)
    return float((a >= t).mean()) if len(a) else np.nan


pp = PersonPath(main_directory="data/person_path", occlusion_ranges=[5, 150], n_experiments=1200,
                n_after_occlusion=40, first_occ_min=2, random_seed=44, min_visible_ratio=0.05,
                min_visible_area=MIN_VISIBLE_AREA, max_distractor_overlap=0.5,
                non_targets=["crowd", "person_in_vehicle", "reflection", "person_in_background", "severly_occluded_person"])
anchor_of = {(v, int(p)): int(a) for v, p, a in zip(
    pp.selected_video_names.tolist(), pp.selected_person_ids.tolist(), pp.selected_anchor_video_frames.tolist())}

dd = DetectionData(amodal_directory="data/person_path/amodal", visible_directory="data/person_path/visible",
                   video_directory="data/person_path/videos", load_frames=False, num_threads=8)


def resolution(video):
    d = json.load(open(f"data/person_path/visible/{video}.json"))["metadata"]["resolution"]
    return float(d["width"]), float(d["height"])


area, sam, mem, mask = [], [], [], []
for c in clips:
    anchor = anchor_of.get((c["video"], int(c["person"])))
    if anchor is None:
        continue
    dd.initialize_target(c["video"], int(c["person"]))
    pos = anchor_trajectory_index(dd, anchor)
    if pos is None:
        continue
    box = dd.bboxes_norm[pos]
    W, H = resolution(c["video"])
    scale = RESIZE / max(W, H)
    area.append(float((box[2] * W) * (box[3] * H) * scale ** 2))
    sam.append(cov(c["sam"])); mem.append(cov(c["memory"])); mask.append(cov(c["mask"]))

area = np.array(area); sam = np.array(sam); mem = np.array(mem); mask = np.array(mask)
m = ~np.isnan(sam)
area, sam, mem, mask = area[m], sam[m], mem[m], mask[m]
edges = np.unique(np.quantile(area, np.linspace(0, 1, 5)))
which = np.clip(np.digitize(area, edges[1:-1]), 0, len(edges) - 2)
print(f"n={len(area)}  anchor area px^2@{RESIZE}: {area.min():.0f}-{area.max():.0f}  (min_visible_area floor={MIN_VISIBLE_AREA})")
print(f"{'area px^2 bin':>16} {'n':>4} {'sam':>6} {'mem':>6} {'mask':>6} {'mem-sam':>8} {'mask-sam':>9}")
for b in range(len(edges) - 1):
    sel = which == b
    if not sel.any():
        continue
    print(f"{int(edges[b]):>6}-{int(edges[b+1]):<8} {sel.sum():>4} {sam[sel].mean():>6.3f} {mem[sel].mean():>6.3f} "
          f"{mask[sel].mean():>6.3f} {mem[sel].mean()-sam[sel].mean():>+8.3f} {mask[sel].mean()-sam[sel].mean():>+9.3f}")
