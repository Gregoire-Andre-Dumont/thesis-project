"""claim_2 figure: post-occlusion tracking quality as the memory bank is poisoned with clean distractors.

Two panels over the same rollouts -- coverage (a frame counts when box IoU >= 0.5) and mean IoU (every frame's
IoU, no success threshold). The thresholded view is the headline metric; the unthresholded one is far less
brittle at small n, so agreement between the panels is the check that a trend is real rather than a few frames
flipping across the 0.5 boundary.

p = 0.0 is the clean rollout, so each curve starts at its own uncorrupted coverage: what matters is the SLOPE,
not the height. Corruption fires per FRAME (occluded or visible) independently of any arm's commit gate, so all
three arms face the identical set of injected distractors.
"""
import sys
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS = "data/claim_2/results.pkl"
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
MEMORY, MASK, SAM = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3, same entity mapping as claim_1


def coverage(ious, threshold=0.5):
    ious = np.asarray(ious, dtype=float)
    return float((ious >= threshold).mean()) if len(ious) else np.nan


def mean_iou(ious):
    ious = np.asarray(ious, dtype=float)
    return float(ious.mean()) if len(ious) else np.nan


results = pickle.load(open(RESULTS, "rb"))
clips, probabilities = results["clips"], results["corruption_ps"]
threshold = results["commit_threshold"]

panels = [("Coverage  (box IoU ≥ 0.5)", coverage, "post-occlusion coverage"),
          ("Mean IoU  (no success threshold)", mean_iou, "mean post-occlusion box IoU")]

figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), facecolor=SURFACE)
for axis, (title, metric, ylabel) in zip(axes, panels):
    axis.set_facecolor(SURFACE)
    ends = []
    for colour, arm, label in ((SAM, "sam", "sam baseline"), (MEMORY, "memory", "memory oracle"),
                               (MASK, "mask", "mask oracle")):
        values = np.array([np.mean([metric(c[(arm, p)]) for c in clips]) for p in probabilities])
        axis.plot(probabilities, values, color=colour, linewidth=2, marker="o", markersize=8,
                  markeredgecolor=SURFACE, markeredgewidth=2, label=label, zorder=3)
        ends.append((values[-1], colour, arm))

    span = max(e[0] for e in ends) - min(e[0] for e in ends)
    gap = max(span, 0.05) * 0.16
    ends.sort()
    placed = []
    for value, colour, short in ends:
        y = value if not placed else max(value, placed[-1] + gap)
        placed.append(y)
        axis.annotate(short, (probabilities[-1], value),
                      xytext=(probabilities[-1] + 0.008, y), textcoords="data",
                      color=colour, fontsize=10, va="center")

    axis.set_xticks(probabilities)
    axis.set_xticklabels([f"{p:g}" for p in probabilities], fontsize=9, color=INK2)
    axis.set_xlabel("corruption probability  (per frame)", fontsize=10, color=INK2)
    axis.set_ylabel(ylabel, fontsize=10, color=INK2)
    axis.set_title(title, fontsize=11, color=INK, pad=10, loc="left")
    axis.grid(axis="y", color=INK2, alpha=0.13, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(INK2)
        axis.spines[side].set_alpha(0.35)
    axis.tick_params(colors=INK2, labelsize=9)
    axis.set_xlim(-0.012, probabilities[-1] + 0.045)
axes[0].legend(frameon=False, fontsize=10, loc="upper right")
figure.suptitle(f"claim_2  ·  memory-bank poisoning with clean nearby distractors  ·  n={len(clips)} clips  ·  "
                f"oracle commit threshold {threshold:g}", fontsize=12, color=INK, x=0.008, ha="left", y=0.985)
figure.tight_layout(rect=[0, 0, 1, 0.93])
figure.savefig("data/claim_2/fig_corruption.png", dpi=150, facecolor=SURFACE)
print(f"saved data/claim_2/fig_corruption.png  (n={len(clips)} clips)")
