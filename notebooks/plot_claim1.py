"""Two standalone claim_1 figures: post-occlusion coverage for all three arms (sam, memory oracle, mask oracle),
binned by anchor visible area and by occlusion length.

Absolute coverage, not deltas -- so the baseline's own difficulty is visible and the oracle gaps are read against
it. sam has no commit gate, so it is a single line; the oracles are drawn at one named commit threshold with a
band showing the full range across the swept thresholds (band width = threshold sensitivity).
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
from src.typing.person_path import PersonPath, _nearest_index

FEATURED = 0                                 # index into `thresholds`: the oracle lines show THIS commit threshold
N_BINS = 4
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
MEMORY, MASK, SAM = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3, fixed order, entity-stable
NON_TARGETS = ["crowd", "person_in_vehicle", "reflection", "person_in_background", "severly_occluded_person"]

person_path = PersonPath(main_directory="data/person_path", occlusion_ranges=[5, 50], n_experiments=1200,
                         n_after_occlusion=40, first_occ_min=5, random_seed=44, min_visible_area=400,
                         max_distractor_overlap=0.5, non_targets=NON_TARGETS)
anchor_of = {(v, int(p)): int(a) for v, p, a in zip(
    person_path.selected_video_names.tolist(), person_path.selected_person_ids.tolist(),
    person_path.selected_anchor_video_frames.tolist())}

_cache = {}


def anchor_area(video, person_id, anchor):
    if video not in _cache:
        _cache[video] = json.load(open(f"data/person_path/visible/{video}.json"))
    visible = _cache[video]
    scale = 1024 / max(float(visible["metadata"]["resolution"]["width"]),
                       float(visible["metadata"]["resolution"]["height"]))
    entities = sorted([e for e in visible["entities"] if e["id"] == person_id],
                      key=lambda e: e["blob"]["frame_idx"])
    if not entities:
        return np.nan
    frames = np.array([e["blob"]["frame_idx"] for e in entities])
    wh = np.array([e["bb"][2:4] for e in entities], dtype=np.float64)
    return float(np.prod(wh[_nearest_index(frames, anchor)]) * scale ** 2)


def coverage(ious, threshold=0.5):
    ious = np.asarray(ious, dtype=float)
    return float((ious >= threshold).mean()) if len(ious) else np.nan


results = pickle.load(open("data/claim_1/results.pkl", "rb"))
thresholds = list(results["thresholds"])

rows = []
for clip in results["clips"]:
    if not len(clip["sam"]):
        continue
    anchor = anchor_of.get((clip["video"], int(clip["person"])))
    if anchor is None:
        continue
    area = anchor_area(clip["video"], int(clip["person"]), anchor)
    if not np.isfinite(area):
        continue
    rows.append((area, int(clip["occ_count"]), coverage(clip["sam"]),
                 [coverage(clip["memory"][t]) for t in thresholds],
                 [coverage(clip["mask"][t]) for t in thresholds]))

areas = np.array([r[0] for r in rows])
occlusions = np.array([r[1] for r in rows])
sam = np.array([r[2] for r in rows])
memory = np.array([r[3] for r in rows])
mask = np.array([r[4] for r in rows])


def draw(values, xlabel, title, filename, tick_format, subset=None, note=""):
    """One figure: absolute coverage for the three arms across quantile bins of `values`.
    `subset` restricts which clips are used (and is applied to `values` by the caller)."""
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

    for colour, values_by_threshold, label, short in ((MEMORY, memory, "memory oracle", "memory"),
                                                      (MASK, mask, "mask oracle", "mask")):
        per_bin = np.array([values_by_threshold[rows_in][index == k].mean(0) for k in x])
        axis.plot(x, per_bin[:, FEATURED], color=colour, linewidth=2, marker="o", markersize=8,
                  markeredgecolor=SURFACE, markeredgewidth=2, label=label, zorder=3)
        ends.append((per_bin[-1, FEATURED], colour, short))

    # Direct labels at the right edge, nudged apart when two arms finish at nearly the same coverage.
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
    axis.set_xticklabels([f"{tick_format(edges[k], edges[k+1])}\nn={counts[k]}" for k in x],
                         fontsize=9, color=INK2)
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
    figure.text(0.008, 0.955, f"n={int(rows_in.sum())} clips  ·  oracles at commit threshold "
                f"{thresholds[FEATURED]}{note}", fontsize=9, color=INK2, ha="left")
    figure.tight_layout(rect=[0, 0, 1, 0.93])
    figure.savefig(filename, dpi=150, facecolor=SURFACE)
    print(f"saved {filename}  (n={int(rows_in.sum())})")


draw(areas, "anchor visible box area  (px² @1024)",
     "Coverage by target size", "data/claim_1/fig_area.png",
     lambda a, b: f"{int(a)}-{int(b)}")

draw(occlusions, "occluded frames",
     "Coverage by occlusion length", "data/claim_1/fig_occlusion_all.png",
     lambda a, b: f"{int(a)}-{int(b)}", note="  ·  all clips")

# The largest-area quartile is where sam is already near-saturated and neither oracle has room, so it flattens
# the occlusion trend; drop it for the occlusion view.
keep = areas < np.quantile(areas, 0.75)
draw(occlusions[keep], "occluded frames",
     "Coverage by occlusion length", "data/claim_1/fig_occlusion.png",
     lambda a, b: f"{int(a)}-{int(b)}", subset=keep,
     note="  ·  largest visible-area quartile dropped")
print(f"n={len(rows)} clips total")
