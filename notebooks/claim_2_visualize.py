"""claim_2 figure: post-occlusion tracking quality as the memory bank is poisoned with clean distractors.

One panel, scoring each kind of frame by what the arm can actually get right on it: a VISIBLE frame counts
when box IoU >= COVERAGE_IOU, an OCCLUDED frame counts when the arm did NOT write it to memory. While the
target is hidden there is nothing to track, so the only decision an arm makes is whether to commit -- and
committing then is exactly how a bank gets poisoned. Scoring it folds memory hygiene into the same number as
tracking quality, which is what the corruption is intervening on.

p = 0.0 is the clean rollout, so each curve starts at its own uncorrupted score: what matters is the SLOPE,
not the height. Corruption fires per FRAME (occluded or visible) independently of any arm's commit gate, so
all three arms face the identical set of injected distractors.
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


COVERAGE_IOU = 0.5


def hygiene(clip, ious, commits):
    """Visible frames held at box IoU >= COVERAGE_IOU, plus occluded frames the arm left uncommitted,
    over every annotated post-occlusion frame."""

    ious = np.asarray(ious, dtype=float)
    seen, hidden = clip["has_box"] & ~clip["occluded"], np.asarray(clip["occluded"])
    scored = int(seen.sum() + hidden.sum())
    if not scored:
        return np.nan

    held = float((ious[seen] >= COVERAGE_IOU).sum())
    clean = float((~np.asarray(commits, dtype=bool)[hidden]).sum())
    return (held + clean) / scored


results = pickle.load(open(RESULTS, "rb"))
clips, probabilities = results["clips"], results["corruption_ps"]
threshold = results["commit_threshold"]

figure, axis = plt.subplots(figsize=(8.6, 5.4), facecolor=SURFACE)
axis.set_facecolor(SURFACE)
ends = []
for colour, arm, label in ((SAM, "sam", "sam baseline"), (MEMORY, "memory", "memory oracle"),
                           (MASK, "mask", "mask oracle")):
    values = np.array([np.nanmean([hygiene(c, c[(arm, p)], c[(arm, p, "commit")]) for c in clips])
                       for p in probabilities])
    axis.plot(probabilities, values, color=colour, linewidth=2, marker="o", markersize=8,
              markeredgecolor=SURFACE, markeredgewidth=2, label=label, zorder=3)
    ends.append((values[-1], colour, arm))

gap = max(max(e[0] for e in ends) - min(e[0] for e in ends), 0.05) * 0.16
ends.sort()
placed = []
for value, colour, short in ends:
    y = value if not placed else max(value, placed[-1] + gap)
    placed.append(y)
    axis.annotate(short, (probabilities[-1], value), xytext=(probabilities[-1] + 0.008, y),
                  textcoords="data", color=colour, fontsize=10, va="center")

axis.set_xticks(probabilities)
axis.set_xticklabels([f"{p:g}" for p in probabilities], fontsize=9, color=INK2)
axis.set_xlabel("corruption probability  (per frame)", fontsize=10, color=INK2)
axis.set_ylabel(f"coverage  (visible: box IoU ≥ {COVERAGE_IOU:g}   ·   occluded: did not commit)",
                fontsize=10, color=INK2)
axis.grid(axis="y", color=INK2, alpha=0.13, linewidth=0.8)
axis.set_axisbelow(True)
for side in ("top", "right"):
    axis.spines[side].set_visible(False)
for side in ("left", "bottom"):
    axis.spines[side].set_color(INK2)
    axis.spines[side].set_alpha(0.35)
axis.tick_params(colors=INK2, labelsize=9)
axis.set_xlim(-0.012, probabilities[-1] + 0.045)
axis.legend(frameon=False, fontsize=10, loc="upper right")
figure.suptitle(f"claim_2  ·  memory-bank poisoning with clean nearby distractors  ·  n={len(clips)} clips  ·  "
                f"oracle commit threshold {threshold:g}", fontsize=12, color=INK, x=0.008, ha="left", y=0.985)
figure.tight_layout(rect=[0, 0, 1, 0.93])
figure.savefig("data/claim_2/fig_corruption.png", dpi=150, facecolor=SURFACE)
print(f"saved data/claim_2/fig_corruption.png  (n={len(clips)} clips)")
