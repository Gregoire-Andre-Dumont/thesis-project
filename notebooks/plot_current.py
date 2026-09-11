"""Two panels for the live claim_1 run: coverage gain over the sam baseline, by anchor visible area and (with
the largest-area quartile dropped) by occlusion length. The line is a single named commit threshold and the band is the full range across all of them, so band width
reads directly as threshold sensitivity."""
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

FEATURED = 0                                  # index into `thresholds`: the line shows THIS commit threshold
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
MEMORY, MASK = "#2a78d6", "#eb6834"          # categorical slots 1 & 2 of the reference palette, fixed order
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
memory_gain = np.array([r[3] for r in rows]) - sam[:, None]
mask_gain = np.array([r[4] for r in rows]) - sam[:, None]


def binned(values, gain_a, gain_b, n_bins=4):
    """Quantile bins over `values`; each bin keeps the per-threshold mean gain for both arms."""
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    index = np.clip(np.digitize(values, edges[1:-1]), 0, len(edges) - 2)
    return [(f"{int(edges[k])}-{int(edges[k+1])}", int((index == k).sum()),
             gain_a[index == k].mean(0), gain_b[index == k].mean(0))
            for k in range(len(edges) - 1)]


cut = np.quantile(areas, 0.75)
keep = areas < cut                                    # drop the saturated largest-area quartile
panels = [("Anchor visible area  (px² @1024)", binned(areas, memory_gain, mask_gain), len(rows)),
          ("Occluded frames  (largest-area quartile dropped)",
           binned(occlusions[keep], memory_gain[keep], mask_gain[keep]), int(keep.sum()))]

figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), facecolor=SURFACE)
for axis, (title, bins, _) in zip(axes, panels):
    axis.set_facecolor(SURFACE)
    x = np.arange(len(bins))
    for colour, column, label in ((MEMORY, 2, "memory oracle"), (MASK, 3, "mask oracle")):
        low = np.array([b[column].min() for b in bins])
        high = np.array([b[column].max() for b in bins])
        middle = np.array([b[column][FEATURED] for b in bins])          # a real threshold, not a median across them
        axis.fill_between(x, low, high, color=colour, alpha=0.16, linewidth=0)
        axis.plot(x, middle, color=colour, linewidth=2, marker="o", markersize=8,
                  markeredgecolor=SURFACE, markeredgewidth=2, label=label, zorder=3)
        axis.annotate(label, (x[-1], middle[-1]), textcoords="offset points", xytext=(9, 0),
                      color=colour, fontsize=10, va="center")
    axis.axhline(0, color=INK2, linewidth=1, alpha=0.55, zorder=1)
    axis.set_xticks(x)
    axis.set_xticklabels([f"{b[0]}\nn={b[1]}" for b in bins], fontsize=9, color=INK2)
    axis.set_title(title, fontsize=11, color=INK, pad=10, loc="left")
    axis.grid(axis="y", color=INK2, alpha=0.13, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(INK2)
        axis.spines[side].set_alpha(0.35)
    axis.tick_params(colors=INK2, labelsize=9)
    axis.set_xlim(-0.35, len(bins) + 0.2)
axes[0].set_ylabel("coverage gain over sam baseline", fontsize=10, color=INK2)
axes[0].legend(frameon=False, fontsize=10, loc="upper right")
figure.suptitle(f"claim_1  occ[5,50]  n={len(rows)} clips  ·  line = commit threshold {thresholds[FEATURED]}, "
                f"shaded band = full range over thresholds {thresholds[0]}-{thresholds[-1]}", fontsize=12, color=INK, x=0.008, ha="left", y=0.985)
figure.tight_layout(rect=[0, 0, 1, 0.94])
figure.savefig("data/claim_1/current_run.png", dpi=150, facecolor=SURFACE)
print(f"saved: n={len(rows)} clips, {int(keep.sum())} after the Q4 drop")
