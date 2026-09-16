"""claim_4 figures: does the calibrator rank SAM's proposals better than SAM's own IoU token?

Three scores, one figure per metric, both against how heavily the memory bank is poisoned.

    sam            SAM's predicted-IoU token -- free, already computed, what the tracker selects with today
    samara p=0     the calibrator trained on CLEAN rollouts only, so every corrupted level is extrapolation
    samara p=0,0.2 the same calibrator with the worst corruption level added to its training mix

The third score is the control on the second: if training on clean rollouts alone already holds up under
poisoning, adding corrupted rollouts should buy little, and the gap between the two says how much of the
robustness is free.

AGREEMENT is the decision the controller actually makes -- on frames where the best and second-best proposal
differ by at least MARGIN, how often does the score pick the best one (chance is 1/3). POOLED R2 is
calibration against true IoU over every held-out frame. Each point is the mean over the trajectory-grouped
folds, with the fold spread as a band. A score whose predictions file is missing is skipped, so this draws
while a run is still training.

    python notebooks/claim_4_visualize.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from notebooks.claim_4 import MARGIN, ranking

OUT = Path("data/claim_4")
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
TRAINED_ON = 0.0

#         label,             predictions file,                  key,   colour
SERIES = [("sam", OUT / "predictions.pkl", "sam", "#cc4444"),
          ("samara p=0", OUT / "predictions.pkl", "cnn", "#22aa77"),
          ("samara p=0,0.2", OUT / "predictions_mixed.pkl", "cnn", "#3377cc")]

FIGURES = [("agree", "agreement", "agreement with the best proposal", 1 / 3, "chance (1/3)"),
           ("R2", "r2", "pooled $R^2$ against true IoU", 0.0, "no signal (0)")]


def series_metrics(path, key):
    """{level: {metric: [one value per fold]}} for one score, or None when its run has not finished."""

    if not path.exists():
        return None
    predictions = pickle.load(open(path, "rb"))
    collected = {}
    for level, folds in predictions.items():
        if not folds:
            continue
        scored = [ranking(fold[key], fold["truth"]) for fold in folds]
        collected[level] = {metric: [fold[metric] for fold in scored] for metric, _, _, _, _ in FIGURES}
    return collected


def draw(axis, by_series, metric, label, floor, floor_label):
    """One figure's axes: every score's fold-mean curve with the fold spread as a band."""

    for name, values, color in by_series:
        levels = sorted(values)
        folds = np.array([values[level][metric] for level in levels])          # (levels, folds)
        axis.fill_between(levels, folds.min(1), folds.max(1), color=color, alpha=0.13, linewidth=0)
        axis.plot(levels, folds.mean(1), marker="o", markersize=8, linewidth=2, color=color,
                  markeredgecolor=SURFACE, markeredgewidth=2, label=name, zorder=3)

    axis.axhline(floor, color=INK2, linestyle=":", linewidth=1, label=floor_label)
    axis.axvline(TRAINED_ON, color=INK2, linestyle="--", linewidth=1, alpha=0.4)
    axis.set_title("Proposal ranking under memory-bank poisoning", fontsize=12, color=INK, pad=12, loc="left")
    axis.set_xlabel("corruption probability  (per frame)", fontsize=10, color=INK2)
    axis.set_ylabel(label, fontsize=10, color=INK2)
    axis.grid(axis="y", color=INK2, alpha=0.13, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(INK2)
        axis.spines[side].set_alpha(0.35)
    axis.tick_params(colors=INK2, labelsize=9)
    axis.legend(frameon=False, fontsize=10, loc="lower left")


available = [(name, series_metrics(path, key), color) for name, path, key, color in SERIES]
missing = [name for name, values, _ in available if values is None]
available = [(name, values, color) for name, values, color in available if values is not None]
if not available:
    raise SystemExit(f"no predictions found under {OUT}")
if missing:
    print(f"skipped (no predictions yet): {', '.join(missing)}")

reference = pickle.load(open(SERIES[0][1], "rb"))
levels = sorted(level for level, folds in reference.items() if folds)
folds = len(reference[levels[0]])
trajectories = len({stem for fold in reference[levels[0]] for stem in fold["trajectories"]})
caption = (f"claim_4  ·  {trajectories} held-out trajectories over {folds} trajectory-grouped folds  ·  "
           f"band = fold spread")

OUT.mkdir(parents=True, exist_ok=True)
for metric, filename, label, floor, floor_label in FIGURES:
    figure, axis = plt.subplots(figsize=(8.6, 5.4), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)
    draw(axis, available, metric, label, floor, floor_label)

    note = caption + (f"  ·  frames with margin >= {MARGIN:g}" if metric == "agree" else "")
    figure.text(0.008, 0.955, note, fontsize=9, color=INK2, ha="left")
    figure.tight_layout(rect=[0, 0, 1, 0.93])
    figure.savefig(OUT / f"fig_{filename}.png", dpi=150, facecolor=SURFACE)
    plt.close(figure)
    print(f"saved {OUT / f'fig_{filename}.png'}")

print(f"\n{trajectories} trajectories   {folds} folds   fold means")
names = [name for name, _, _ in available]
print(f"{'p':7}" + "".join(f"{name + ' ' + metric:>20}" for metric, _, _, _, _ in FIGURES for name in names))
for level in levels:
    cells = [values[level][metric] for metric, _, _, _, _ in FIGURES for _, values, _ in available]
    print(f"p{level:<6.2f}" + "".join(f"{np.mean(cell):>20.3f}" for cell in cells))
