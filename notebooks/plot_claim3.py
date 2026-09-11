"""claim_3 figure: PE foreground similarity as a replacement for SAM's mask selection and memory control.

Two panels over the same rollouts -- coverage (a frame counts when box IoU >= 0.5) and mean IoU (every
frame's IoU, no success threshold). The thresholded view is the headline; the unthresholded one is far
less brittle at small n, so agreement between the panels is the check that a trend is real rather than
a few frames flipping across the 0.5 boundary.

The x axis is the PE commit threshold, so the PE curve is a sweep and `sam` is a flat reference line --
it has no PE threshold. Read the GAP to that line, not the shape alone.

tau = 0.0 leaves the gate fully open: PE still picks every mask but commits every frame. So
`pe@0 vs sam` is what PE SELECTION costs or buys, and `pe@tau vs pe@0` is what the GATE adds on top.
"""
import sys
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS = "data/claim_3/results.pkl"
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
PE, SAM = "#2a78d6", "#1baf7a"          # same entity->hue mapping as claim_1/claim_2: sam stays green


def coverage(ious, threshold=0.5):
    ious = np.asarray(ious, dtype=float)
    return float((ious >= threshold).mean()) if len(ious) else np.nan


def mean_iou(ious):
    ious = np.asarray(ious, dtype=float)
    return float(ious.mean()) if len(ious) else np.nan


results = pickle.load(open(RESULTS, "rb"))
clips, thresholds = results["clips"], results["pe_thresholds"]

panels = [("Coverage  (box IoU ≥ 0.5)", coverage, "post-occlusion coverage"),
          ("Mean IoU  (no success threshold)", mean_iou, "mean post-occlusion box IoU")]

figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), facecolor=SURFACE)
for axis, (title, metric, ylabel) in zip(axes, panels):
    axis.set_facecolor(SURFACE)

    baseline = float(np.mean([metric(c["sam"]) for c in clips]))
    axis.axhline(baseline, color=SAM, linewidth=2, zorder=2)
    axis.annotate("sam baseline", (thresholds[-1], baseline),
                  xytext=(thresholds[-1] + 0.012, baseline), textcoords="data",
                  color=SAM, fontsize=10, va="center")

    values = np.array([np.mean([metric(c[("pe", t)]) for c in clips]) for t in thresholds])
    axis.plot(thresholds, values, color=PE, linewidth=2, marker="o", markersize=8,
              markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    gap = 0.02 if abs(values[-1] - baseline) < 0.02 else 0.0
    axis.annotate("pe tracker", (thresholds[-1], values[-1]),
                  xytext=(thresholds[-1] + 0.012, values[-1] - gap), textcoords="data",
                  color=PE, fontsize=10, va="center")

    axis.set_xticks(thresholds)
    axis.set_xticklabels([f"{t:g}" for t in thresholds], fontsize=9, color=INK2)
    axis.set_xlabel("PE commit threshold  (foreground similarity)", fontsize=10, color=INK2)
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
    axis.set_xlim(thresholds[0] - 0.03, thresholds[-1] + 0.13)

figure.suptitle(f"claim_3  ·  PE foreground similarity replacing SAM's mask selection + memory control  ·  "
                f"n={len(clips)} clips", fontsize=12, color=INK, x=0.008, ha="left", y=0.985)
figure.tight_layout(rect=[0, 0, 1, 0.93])
figure.savefig("data/claim_3/fig_pe.png", dpi=150, facecolor=SURFACE)
print(f"saved data/claim_3/fig_pe.png  (n={len(clips)} clips)")
