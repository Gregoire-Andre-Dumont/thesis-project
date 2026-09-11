"""claim_3 by target size: PE-selected masks vs the sam baseline across quantile bins of anchor visible area.

Absolute coverage for both arms, so the baseline's own difficulty per bin is visible and the PE gap is
read against it. Both arms share every trajectory, anchor and frame, and (with gate="sam") the same
commit rule -- so a gap between the lines is attributable to mask SELECTION alone.

Bins are anchor visible box area in px^2 at the 1024 working resolution, resolved from the anchor frame
the trajectory was selected on -- the same construction as plot_claim1.py.
"""
import sys
import glob
import json
import pickle
from pathlib import Path

import hydra
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.typing.person_path import _nearest_index

N_BINS = 4
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
SAM = "#1baf7a"                              # entity-stable: sam keeps its hue across every claim figure
# The thresholds are an ORDERED magnitude, so they get one hue light->dark -- never separate categorical
# hues, which would imply unrelated entities and lose the ordering.
PE_RAMP = ["#a8cbf0", "#6ba3e2", "#2a78d6", "#1d5aa5", "#123f74"]
COMPARE = None            # glob of a previous run to overlay as a comparison arm; None = no overlay
COMPARE_COLOUR, COMPARE_LABEL = "#eb6834", "pe sel + sam gate"


def ramp_colours(count):
    """`count` steps spanning the PE ramp, always including its darkest end."""
    if count == 1:
        return [PE_RAMP[2]]
    positions = np.linspace(0, len(PE_RAMP) - 1, count)
    return [PE_RAMP[int(round(p))] for p in positions]

# Build the selection from the EXPERIMENT's own config, never a hardcoded copy: min_visible_ratio (and any
# other anchor condition) changes which frame each trajectory anchors on, and a stale copy silently resolves
# the wrong anchor -- or no anchor at all, dropping every clip.
person_path = hydra.utils.instantiate(OmegaConf.load("conf/experiments/claim_3.yaml").person_path)
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


results = pickle.load(open("data/claim_3/results.pkl", "rb"))
thresholds = list(results["pe_thresholds"])
gate = results.get("gate", "pe")

# Comparison arm from a previous run. It only covers the clips THAT run reached, so it is NaN elsewhere
# and averaged with nanmean -- the main arms must never be truncated to the comparison's clip set, which
# would silently freeze the figure at that run's n no matter how far this one gets.
compare_files = sorted(glob.glob(COMPARE)) if COMPARE else []
compare = {}
if compare_files:
    other = pickle.load(open(compare_files[-1], "rb"))
    other_threshold = other["pe_thresholds"][0]
    compare = {(c["video"], int(c["person"])): coverage(c[("pe", other_threshold)])
               for c in other["clips"] if len(c["sam"])}

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
    rows.append((area, coverage(clip["sam"]), [coverage(clip[("pe", t)]) for t in thresholds],
                 compare.get((clip["video"], int(clip["person"])), np.nan)))

areas = np.array([r[0] for r in rows])
sam = np.array([r[1] for r in rows])
pe = np.array([r[2] for r in rows])                                   # (clips, thresholds)
compare_values = np.array([r[3] for r in rows]) if compare else None

edges = np.unique(np.quantile(areas, np.linspace(0, 1, N_BINS + 1)))
index = np.clip(np.digitize(areas, edges[1:-1]), 0, len(edges) - 2)
x = np.arange(len(edges) - 1)
counts = [int((index == k).sum()) for k in x]

figure, axis = plt.subplots(figsize=(8.6, 5.4), facecolor=SURFACE)
axis.set_facecolor(SURFACE)

ends = []
sam_per_bin = np.array([sam[index == k].mean() for k in x])
axis.plot(x, sam_per_bin, color=SAM, linewidth=2.4, marker="o", markersize=8,
          markeredgecolor=SURFACE, markeredgewidth=2, label="sam baseline", zorder=4)
ends.append((sam_per_bin[-1], SAM, "sam"))

for position, (threshold, colour) in enumerate(zip(thresholds, ramp_colours(len(thresholds)))):
    per_bin = np.array([pe[index == k, position].mean() for k in x])
    # With gate="sam" the swept value is SAM's predicted-IoU threshold, NOT a PE threshold -- PE has no
    # threshold there at all, it only ranks the proposals. Labelling it "pe @ x" would misread as one.
    label = "pe selects (sam gate)" if gate == "sam" else f"pe @ {threshold:g}"
    short = "pe sel" if gate == "sam" else f"{threshold:g}"
    axis.plot(x, per_bin, color=colour, linewidth=2, marker="o", markersize=7,
              markeredgecolor=SURFACE, markeredgewidth=2, label=label, zorder=3)
    ends.append((per_bin[-1], colour, short))

if compare_values is not None:
    covered = int(np.isfinite(compare_values).sum())
    per_bin = np.array([np.nanmean(compare_values[index == k]) if np.isfinite(compare_values[index == k]).any()
                        else np.nan for k in x])
    axis.plot(x, per_bin, color=COMPARE_COLOUR, linewidth=2.4, marker="o", markersize=8,
              markeredgecolor=SURFACE, markeredgewidth=2, label=f"{COMPARE_LABEL}  (n={covered})", zorder=4)
    ends.append((per_bin[-1], COMPARE_COLOUR, "fg-only"))

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
axis.set_xlabel("anchor visible box area  (px² @1024)", fontsize=10, color=INK2)
axis.set_ylabel("post-occlusion coverage  (box IoU ≥ 0.5)", fontsize=10, color=INK2)
axis.set_title("Coverage by target size", fontsize=12, color=INK, pad=12, loc="left")
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

detail = ("SAM gate (object score + predicted IoU), swept threshold" if gate == "sam"
          else "PE (base) selects AND gates, swept commit threshold")
note = "  ·  comparison arm: PE (large) selects, SAM gates — different encoder" if compare else ""
figure.text(0.008, 0.955, f"n={len(rows)} clips  ·  {detail}{note}", fontsize=9, color=INK2, ha="left")
figure.tight_layout(rect=[0, 0, 1, 0.93])
figure.savefig("data/claim_3/fig_pe_area.png", dpi=150, facecolor=SURFACE)
print(f"saved data/claim_3/fig_pe_area.png  (n={len(rows)} clips)")
