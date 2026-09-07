"""Bin claim_1 coverage by ANCHOR AMODAL AREA (px^2 at resize_resolution=1024, the size the gate uses).
Recovers each clip's anchor frame from person_path, then the nearest amodal box's area (CPU only)."""
import sys
import json
from pathlib import Path
import pickle
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.typing.person_path import PersonPath
from src.utils.load_frame_ids import load_frame_ids

RESIZE = 1024
THR = 0.4                                                         # commit gate to score at

s = pickle.load(open("data/claim_1/results.pkl", "rb"))
clips = s["clips"]


def cov(a, t=0.5):
    a = np.asarray(a)
    return float((a >= t).mean()) if len(a) else np.nan


pp = PersonPath(main_directory="data/person_path", occlusion_ranges=[2, 50], n_experiments=1200,
                n_after_occlusion=40, first_occ_min=2, random_seed=44, min_visible_area=500,
                max_distractor_overlap=0.5,
                non_targets=["crowd", "person_in_vehicle", "reflection", "person_in_background", "severly_occluded_person"])
anchor_of = {(v, int(p)): int(a) for v, p, a in zip(
    pp.selected_video_names.tolist(), pp.selected_person_ids.tolist(), pp.selected_anchor_video_frames.tolist())}


def amodal_area_1024(video, pid, anchor):
    d = json.load(open(f"data/person_path/amodal/{video}.json"))
    res = d["metadata"]["resolution"]; W, H = float(res["width"]), float(res["height"])
    ents = [e for e in d["entities"] if e["id"] == pid]
    if not ents:
        return np.nan
    frames = np.array([e["blob"]["frame_idx"] for e in ents])
    whs = np.array([e["bb"][2:4] for e in ents], dtype=np.float64)
    nearest = int(np.argmin(np.abs(frames - anchor)))
    w, h = whs[nearest]
    scale = RESIZE / max(W, H)
    return float((w * scale) * (h * scale))                       # px^2 at RESIZE


area, sam, mem, mask = [], [], [], []
for c in clips:
    a = anchor_of.get((c["video"], int(c["person"])))
    if a is None:
        continue
    ar = amodal_area_1024(c["video"], int(c["person"]), a)
    if not np.isfinite(ar):
        continue
    area.append(ar); sam.append(cov(c["sam"])); mem.append(cov(c["memory"][THR])); mask.append(cov(c["mask"][THR]))

area = np.array(area); sam = np.array(sam); mem = np.array(mem); mask = np.array(mask)
m = ~np.isnan(sam)
area, sam, mem, mask = area[m], sam[m], mem[m], mask[m]
edges = np.unique(np.quantile(area, np.linspace(0, 1, 5)))
which = np.clip(np.digitize(area, edges[1:-1]), 0, len(edges) - 2)
print(f"n={len(area)}  amodal area px^2@{RESIZE}: {area.min():.0f}-{area.max():.0f}  (gate floor=500)  commit thr={THR}")
print(f"{'area px^2 bin':>16} {'n':>4} {'sam':>6} {'mem':>6} {'mask':>6} {'mem-sam':>8} {'mask-sam':>9}")
for b in range(len(edges) - 1):
    sel = which == b
    if not sel.any():
        continue
    print(f"{int(edges[b]):>6}-{int(edges[b+1]):<8} {sel.sum():>4} {sam[sel].mean():>6.3f} {mem[sel].mean():>6.3f} "
          f"{mask[sel].mean():>6.3f} {mem[sel].mean()-sam[sel].mean():>+8.3f} {mask[sel].mean()-sam[sel].mean():>+9.3f}")
