"""Bin claim_1 coverage by the anchor's VISIBLE box area (px^2 @1024). sam vs memory-oracle vs mask-oracle
at thr 0.2 and 0.4."""
import sys
import json
import pickle
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.typing.person_path import PersonPath, _nearest_index

RESIZE = 1024
NBINS = 4
SUCCESS_IOU = 0.5

pp = PersonPath(main_directory="data/person_path", occlusion_ranges=[5, 75], n_experiments=1200,
                n_after_occlusion=30, first_occ_min=5, random_seed=44, min_visible_ratio=0.5,
                min_visible_area=400, max_distractor_overlap=0.5,
                non_targets=["crowd", "person_in_vehicle", "reflection", "person_in_background", "severly_occluded_person"])
anchor_of = {(v, int(p)): int(a) for v, p, a in zip(
    pp.selected_video_names.tolist(), pp.selected_person_ids.tolist(), pp.selected_anchor_video_frames.tolist())}

_cache = {}


def visible_area(video, pid, anchor):
    if video not in _cache:
        _cache[video] = json.load(open(f"data/person_path/visible/{video}.json"))
    visible = _cache[video]
    scale = RESIZE / max(float(visible["metadata"]["resolution"]["width"]), float(visible["metadata"]["resolution"]["height"]))
    ents = sorted([e for e in visible["entities"] if e["id"] == pid], key=lambda e: e["blob"]["frame_idx"])
    if not ents:
        return np.nan
    frames = np.array([e["blob"]["frame_idx"] for e in ents])
    wh = np.array([e["bb"][2:4] for e in ents], dtype=np.float64)
    return float(np.prod(wh[_nearest_index(frames, anchor)]) * scale ** 2)


def cov(ious, t=SUCCESS_IOU):
    ious = np.asarray(ious, dtype=float)
    return float((ious >= t).mean()) if len(ious) else np.nan


MIN_OCC = 5                                                      # drop clips whose windowed occlusion count < this
_results = pickle.load(open("data/claim_1/results.pkl", "rb"))
THRS = [float(t) for t in _results["thresholds"]]          # swept thresholds, straight from the run
clips = [c for c in _results["clips"] if len(c["sam"]) and c["occ_count"] >= MIN_OCC]
rows = []
for c in clips:
    anchor = anchor_of.get((c["video"], int(c["person"])))
    if anchor is None:
        continue
    ar = visible_area(c["video"], int(c["person"]), anchor)
    if np.isfinite(ar):
        rows.append((ar, c))

areas = np.array([r[0] for r in rows])
edges = np.unique(np.quantile(areas, np.linspace(0, 1, NBINS + 1)))
bin_of = np.clip(np.digitize(areas, edges[1:-1]), 0, len(edges) - 2)

print(f"n={len(rows)}  visible area px^2@{RESIZE}: {areas.min():.0f}-{areas.max():.0f}")
for thr in THRS:
    sam = np.array([cov(r[1]["sam"]) for r in rows])
    mem = np.array([cov(r[1]["memory"][thr]) for r in rows])
    mask = np.array([cov(r[1]["mask"][thr]) for r in rows])
    print(f"\n=== thr={thr}  pooled sam={sam.mean():.3f} mem={mem.mean():.3f} mask={mask.mean():.3f} "
          f"(mem{mem.mean()-sam.mean():+.3f} mask{mask.mean()-sam.mean():+.3f}) ===")
    print(f"{'vis area bin':>16} {'n':>4} {'sam':>6} {'mem':>6} {'mask':>6} {'mem-sam':>8} {'mask-sam':>9}")
    for b in range(len(edges) - 1):
        sel = bin_of == b
        if not sel.any():
            continue
        s, m, k = sam[sel].mean(), mem[sel].mean(), mask[sel].mean()
        print(f"{int(edges[b]):>6}-{int(edges[b+1]):<8} {int(sel.sum()):>4} {s:>6.3f} {m:>6.3f} {k:>6.3f} "
              f"{m-s:>+8.3f} {k-s:>+9.3f}")
