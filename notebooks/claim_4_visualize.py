"""claim_4 figure: does the calibrator judge a proposal usable better than SAM's own IoU token?

One figure, both scores against how heavily the memory bank is poisoned:

    SAM IoU token   free, already computed, what the tracker selects with today
    SAMARA gate     the calibrator, trained on the corruption mix its config names

A proposal is USABLE when its true mask IoU clears `USABLE_IOU`, and the metric is the probability a score
ranks a usable proposal above an unusable one -- an AUC over every held-out proposal. Chance is 0.50.

This is the gate's question, not the selector's. Selection needs an ordering WITHIN a frame; the gate needs
to know whether the mask in hand is worth committing at all. On the frames where every proposal is bad the
ordering is irrelevant and this is the only question left, which is exactly the regime a poisoned bank
produces.

Each point is the mean over the trajectory-grouped folds, with the fold spread as a band. A score whose
predictions file is missing is skipped, so this draws while a run is still training.

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
sys.path.insert(0, str(Path(__file__).resolve().parent / "paper"))

import style
from notebooks.claim_4 import USABLE_IOU, ranking

OUT = Path("data/claim_4")
FIGURES = Path("data/claim_4/paper")

# Slot per SCORE, fixed: the token keeps slot 0 and the calibrator slot 1 in every claim_4 figure.
SERIES = [("SAM IoU token", OUT / "predictions.pkl", "sam", 0),
          ("SAMARA gate", OUT / "predictions.pkl", "cnn", 1)]


def series_auc(path, key):
    """{level: [AUC per fold]} for one score, or None when its run has not finished."""

    if not path.exists():
        return None

    predictions = pickle.load(open(path, "rb"))
    collected = {}
    for level, folds in predictions.items():
        if not folds:
            continue
        collected[level] = [ranking(fold[key], fold["truth"])["AUC"] for fold in folds]
    return collected


def main():
    style.use_paper_style()

    drawn = []
    for name, path, key, slot in SERIES:
        values = series_auc(path, key)
        if values is None:
            print(f"skipped {name}  ({path} not found)")
            continue
        drawn.append((name, values, slot))

    if not drawn:
        raise SystemExit("no predictions to draw")

    figure, axis = plt.subplots(figsize=(style.COLUMN, 3.1))
    highest = 0.0
    for name, values, slot in drawn:
        levels = sorted(values)
        folds = np.array([values[level] for level in levels])          # (levels, folds)
        appearance = style.series_style(slot)
        axis.fill_between(levels, folds.min(1), folds.max(1), color=appearance["color"],
                          alpha=0.13, linewidth=0, zorder=2)
        axis.plot(levels, folds.mean(1), label=name, zorder=3, **appearance)
        highest = max(highest, float(folds.mean(1).max()))

    axis.axhline(0.5, color=style.INK2, linestyle=(0, (1, 2)), linewidth=0.8, zorder=2)

    levels = sorted(drawn[0][1])
    axis.set_xticks(levels)
    axis.set_xticklabels([f"{level:g}" for level in levels])
    span = levels[-1] - levels[0]
    axis.set_xlim(levels[0] - 0.04 * span, levels[-1] + 0.04 * span)
    style.gridlines(axis, 0.05)
    style.headroom(axis, highest)
    style.style_axes(axis, "corruption probability (per frame)", f"AUC (IoU $>$ {USABLE_IOU:g})")
    axis.legend(loc="lower left")

    style.save(figure, FIGURES / "fig_usable_auc")
    folds_drawn = len(next(iter(drawn[0][1].values())))
    print(f"   caption: Probability that a score ranks a usable proposal (true mask IoU > {USABLE_IOU:g}) "
          f"above an unusable one, against the rate at which the memory bank is poisoned. Mean over "
          f"{folds_drawn} trajectory-grouped folds, band is the fold spread. The dotted line is chance "
          f"(0.50).")


if __name__ == "__main__":
    main()
