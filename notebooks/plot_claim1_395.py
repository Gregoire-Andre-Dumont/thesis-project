"""The two claim_1 figures for the COMPLETED 400-trajectory run (395 scorable clips).

That run predates three changes, so it cannot be plotted with `plot_claim1.py`:
  * anchors were chosen with the visible/amodal ratio gate that has since been removed -- replicated below so
    all 395 clips resolve to the anchor they actually used;
  * occlusion was inferred from box-less frames (annotation gaps included), not from the `fully_occluded` label;
  * its threshold sweep was 0.2-0.5, with no 0.0 column.
Its numbers are therefore NOT directly comparable to the current run -- they measure a slightly different thing.
"""
import sys
import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.typing.person_path import _nearest_index, _covered_fraction, _group_entities_by_person_id

RESULTS = "data/claim_1/results_oldsemantics_395_1789030203.pkl"
MIN_VISIBLE_RATIO, MIN_VISIBLE_AREA, MAX_DISTRACTOR, RESIZE = 0.5, 400.0, 0.5, 1024
N_BINS = 4
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
MEMORY, MASK, SAM = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3, entity-stable across figures


def old_anchor(video, person_id):
    """The original `_anchor_video_frame`, ratio gate included. Returns (frame, area px² @1024) or None."""
    amodal = json.load(open(f"data/person_path/amodal/{video}.json"))
    visible = json.load(open(f"data/person_path/visible/{video}.json"))
    amodal_entities = _group_entities_by_person_id(amodal["entities"]).get(person_id, [])
    visible_entities = _group_entities_by_person_id(visible["entities"]).get(person_id, [])
    if not amodal_entities or not visible_entities:
        return None
    resolution = visible["metadata"]["resolution"]
    scale = RESIZE / max(resolution["width"], resolution["height"])
    min_area_pixels = MIN_VISIBLE_AREA / scale ** 2

    boxes_by_frame = {}
    for entity in visible["entities"]:
        boxes_by_frame.setdefault(entity["blob"]["frame_idx"], []).append((entity["id"], entity["bb"]))

    frame_of = lambda e: e["blob"]["frame_idx"]
    amodal_entities = sorted(amodal_entities, key=frame_of)
    amodal_frames = np.array([frame_of(e) for e in amodal_entities])
    amodal_wh = np.array([e["bb"][2:4] for e in amodal_entities], dtype=np.float64)
    occluded = {frame_of(e) for e in amodal_entities if "fully_occluded" in e["labels"]}

    for entity in sorted(visible_entities, key=frame_of):
        frame = int(frame_of(entity))
        wh = np.array(entity["bb"][2:4], dtype=np.float64)
        if frame in occluded or np.any(wh <= 0):
            continue
        visible_area = float(np.prod(wh))
        if visible_area < min_area_pixels:
            continue
        amodal_area = float(np.prod(amodal_wh[_nearest_index(amodal_frames, frame)]))
        if amodal_area <= 0 or visible_area / amodal_area < MIN_VISIBLE_RATIO:
            continue
        others = [b for other_id, b in boxes_by_frame.get(frame, []) if other_id != person_id]
        overlaps = [_covered_fraction(entity["bb"], b) for b in others]
        if (max(overlaps) if overlaps else 0.0) > MAX_DISTRACTOR:
            continue
        return frame, visible_area * scale ** 2
    return None


def coverage(ious, threshold=0.5):
    ious = np.asarray(ious, dtype=float)
    return float((ious >= threshold).mean()) if len(ious) else np.nan


results = pickle.load(open(RESULTS, "rb"))
thresholds = list(results["thresholds"])

rows, unresolved = [], 0
for clip in results["clips"]:
    if not len(clip["sam"]):
        continue
    resolved = old_anchor(clip["video"], int(clip["person"]))
    if resolved is None:
        unresolved += 1
        continue
    rows.append((resolved[1], int(clip["occ_count"]), coverage(clip["sam"]),
                 [coverage(clip["memory"][t]) for t in thresholds],
                 [coverage(clip["mask"][t]) for t in thresholds]))

areas = np.array([r[0] for r in rows])
occlusions = np.array([r[1] for r in rows])
sam = np.array([r[2] for r in rows])
memory = np.array([r[3] for r in rows])
mask = np.array([r[4] for r in rows])

# Featured threshold: the memory oracle's argmax on the same Q4-removed subset the occlusion figure uses.
keep = areas < np.quantile(areas, 0.75)
FEATURED = int((memory - sam[:, None])[keep].mean(0).argmax())
print(f"resolved {len(rows)}/{len(rows) + unresolved} clips; featured threshold = {thresholds[FEATURED]}")


def draw(values, xlabel, title, filename, subset=None, note=""):
    rows_in = np.ones(len(sam), dtype=bool) if subset is None else subset
    edges = np.unique(np.quantile(values, np.linspace(0, 1, N_BINS + 1)))
    index = np.clip(np.digitize(values, edges[1:-1]), 0, len(edges) - 2)
    counts = [int((index == k).sum()) for k in range(len(edges) - 1)]
    x = np.arange(len(edges) - 1)

    figure, axis = plt.subplots(figsize=(8.6, 5.4), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)

    baseline = np.array([sam[rows_in][index == k].mean() for k in x])
    axis.plot(x, baseline, color=SAM, linewidth=2, marker="o", markersize=8,
              markeredgecolor=SURFACE, markeredgewidth=2, label="sam baseline", zorder=3)
    ends = [(baseline[-1], SAM, "sam")]
    for colour, arm, label, short in ((MEMORY, memory, "memory oracle", "memory"),
                                      (MASK, mask, "mask oracle", "mask")):
        per_bin = np.array([arm[rows_in][index == k].mean(0) for k in x])
        axis.plot(x, per_bin[:, FEATURED], color=colour, linewidth=2, marker="o", markersize=8,
                  markeredgecolor=SURFACE, markeredgewidth=2, label=label, zorder=3)
        ends.append((per_bin[-1, FEATURED], colour, short))

    span = max(e[0] for e in ends) - min(e[0] for e in ends)
    gap = max(span, 0.05) * 0.16
    ends.sort()
    placed = []
    for value, colour, short in ends:
        y = value if not placed else max(value, placed[-1] + gap)
        placed.append(y)
        axis.annotate(short, (x[-1], value), xytext=(x[-1] + 0.12, y), textcoords="data",
                      color=colour, fontsize=10, va="center")

    axis.set_xticks(x)
    axis.set_xticklabels([f"{int(edges[k])}-{int(edges[k+1])}\nn={counts[k]}" for k in x], fontsize=9, color=INK2)
    axis.set_xlabel(xlabel, fontsize=10, color=INK2)
    axis.set_ylabel("post-occlusion coverage  (box IoU ≥ 0.5)", fontsize=10, color=INK2)
    axis.set_title(title, fontsize=12, color=INK, pad=12, loc="left")
    axis.grid(axis="y", color=INK2, alpha=0.13, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(INK2)
        axis.spines[side].set_alpha(0.35)
    axis.tick_params(colors=INK2, labelsize=9)
    axis.set_xlim(-0.35, len(x) - 1 + 0.55)
    axis.legend(frameon=False, fontsize=10, loc="best")
    figure.text(0.008, 0.955, f"400-trajectory run  ·  n={int(rows_in.sum())} clips  ·  oracles at commit "
                f"threshold {thresholds[FEATURED]}{note}", fontsize=9, color=INK2, ha="left")
    figure.tight_layout(rect=[0, 0, 1, 0.93])
    figure.savefig(filename, dpi=150, facecolor=SURFACE)
    print(f"saved {filename}  (n={int(rows_in.sum())})")


draw(areas, "anchor visible box area  (px² @1024)", "Coverage by target size",
     "data/claim_1/fig_area_395.png")
draw(occlusions[keep], "occluded frames", "Coverage by occlusion length",
     "data/claim_1/fig_occlusion_395.png", subset=keep,
     note="  ·  largest visible-area quartile dropped")
