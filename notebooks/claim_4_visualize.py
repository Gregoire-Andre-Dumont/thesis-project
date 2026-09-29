"""claim_4 figure: does the calibrator judge a proposal usable better than SAM's own IoU token?"""

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


SERIES = [("SAM IoU token", OUT / "predictions.pkl", "sam", 0), ("SAMARA gate", OUT / "predictions.pkl", "cnn", 1)]


def series_auc(results_path, record_key):
    """{level: [AUC per fold]} for one score, or None when its run has not finished."""

    if not results_path.exists():
        return None

    predictions = pickle.load(open(results_path, "rb"))
    collected = {}
    for corruption_level, number_of_folds in predictions.items():
        if not number_of_folds:
            continue
        collected[corruption_level] = [ranking(fold_index[record_key], fold_index["truth"])["AUC"] for fold_index in number_of_folds]
    return collected


def main():
    style.use_paper_style()

    drawn_values = []
    for arm_name, results_path, record_key, series_slot in SERIES:
        plotted_values = series_auc(results_path, record_key)
        if plotted_values is None:
            continue
        drawn_values.append((arm_name, plotted_values, series_slot))

    if not drawn_values:
        raise SystemExit("no predictions to draw")

    figure_handle, axis_handle = plt.subplots(figsize=(style.COLUMN, 3.1))
    highest = 0.0
    for arm_name, plotted_values, series_slot in drawn_values:
        corruption_levels = sorted(plotted_values)
        number_of_folds = np.array([plotted_values[corruption_level] for corruption_level in corruption_levels])
        appearance = style.series_style(series_slot)
        axis_handle.fill_between(
            corruption_levels,
            number_of_folds.min(1),
            number_of_folds.max(1),
            color=appearance["color"],
            alpha=0.13,
            linewidth=0,
            zorder=2)
        axis_handle.plot(corruption_levels, number_of_folds.mean(1), label=arm_name, zorder=3, **appearance)
        highest = max(highest, float(number_of_folds.mean(1).max()))

    axis_handle.axhline(0.5, color=style.INK2, linestyle=(0, (1, 2)), linewidth=0.8, zorder=2)

    corruption_levels = sorted(drawn_values[0][1])
    axis_handle.set_xticks(corruption_levels)
    axis_handle.set_xticklabels([f"{corruption_level:g}" for corruption_level in corruption_levels])
    scored_span = corruption_levels[-1] - corruption_levels[0]
    axis_handle.set_xlim(corruption_levels[0] - 0.04 * scored_span, corruption_levels[-1] + 0.04 * scored_span)
    style.gridlines(axis_handle, 0.05)
    style.headroom(axis_handle, highest)
    style.style_axes(axis_handle, "corruption probability (per frame)", f"AUC (IoU $>$ {USABLE_IOU:g})")
    axis_handle.legend(loc="lower left")

    style.save(figure_handle, FIGURES / "fig_usable_auc")
    folds_drawn = len(next(iter(drawn_values[0][1].values())))


if __name__ == "__main__":
    main()
