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
sys.path.insert(0, str(Path(__file__).resolve().parent / "paper"))

import style

RESULTS = "data/claim_2/results.pkl"
FIGURES = Path("data/claim_2/paper")

# Slot order is the entity mapping claim_1 uses, and it has to be the same one: a reader who meets "sam" as
# the blue solid line in one figure and the green dotted line in the next has to relearn the legend.
ARMS = (("sam", "SAM 2 baseline"), ("memory", "memory oracle"), ("mask", "mask oracle"))


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

style.use_paper_style()
figure, axis = plt.subplots(figsize=(style.COLUMN, 3.1))

highest = 0.0
for position, (arm, label) in enumerate(ARMS):
    values = np.array([np.nanmean([hygiene(c, c[(arm, p)], c[(arm, p, "commit")]) for c in clips])
                       for p in probabilities])
    axis.plot(probabilities, values, label=label, zorder=3, **style.series_style(position))
    highest = max(highest, float(np.nanmax(values)))

axis.set_xticks(probabilities)
axis.set_xticklabels([f"{p:g}" for p in probabilities])
span = probabilities[-1] - probabilities[0]
axis.set_xlim(probabilities[0] - 0.04 * span, probabilities[-1] + 0.04 * span)
style.gridlines(axis, 0.05)
style.headroom(axis, highest)
style.style_axes(axis, "corruption probability (per frame)", f"coverage @ {COVERAGE_IOU:g}")
axis.legend(loc="best")

style.save(figure, FIGURES / "fig_corruption")
print(f"   caption: Post-occlusion coverage and memory hygiene as the memory bank is poisoned with "
      f"clean nearby distractors. n={len(clips)} clips; oracle commit threshold {threshold:g}. "
      f"p=0 is the clean rollout, so each curve starts at its own uncorrupted score and the slope is "
      f"what matters, not the height.")
