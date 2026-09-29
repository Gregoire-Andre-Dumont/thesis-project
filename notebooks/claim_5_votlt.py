"""claim_5 figure: VOT-LT F-score against how long the target is occluded.

One line per CONFIDENCE, all on the same trajectories where possible, each swept to its own best threshold
within each occlusion bin -- so no line is penalised for a default that suits a different regime.

The split is the point. Pooled over every clip the confidences look close, because most clips have short
occlusions where any sensible rule behaves the same; the differences live in the long-occlusion tail, and
a single pooled number hides them.

Two questions this answers at once:
  * does the learned commit probability beat SAM's own object score and IoU token, on identical masks;
  * is a gate trained as a CLASSIFIER (`BCEIouLoss`) a better confidence than one trained to REGRESS the
    IoU (`MSEIouLoss`)? They optimise different things and only the first is what a commit needs.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics.vot_lt import Predictions

FPS = 5.0                      # PersonPath22 is annotated at 5 fps, so bins are reported in seconds
BINS = 4
BOOTSTRAP = 2000

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3e0"

# Categorical, fixed order, never cycled. Chosen for separation under deuteranopia as well as full colour:
# blue and orange carry the two learned scores, red and grey SAM's own two.
SERIES = {
    "gate p(commit)":  "#2a78d6",
    "map-only gate":   "#e08214",
    "SAM object score": "#cc4444",
    "SAM IoU token":   "#52514e",
}

# Which stored field each line sweeps. The thresholds are derived from each score's OWN quantiles rather
# than a fixed range: a probability lives in [0, 1], SAM's object score is a logit, and the map-only gate
# emits an unbounded logit too, so any shared grid would hand some of them operating points they cannot
# reach -- the same mistake SAMURAI's shipped -1 default encodes.
CONFIDENCES = {
    "gate p(commit)":   "commit_probability",
    "map-only gate":    "map_only_score",
    "SAM object score": "object_score",
    "SAM IoU token":    "iou_token",
}
SWEEP_STEPS = 51


def per_frame(clip, field):
    """A stored confidence as one value per frame. `commit_probability` and `predicted_iou` are kept per
    PROPOSAL, with NaN in the slots SAM did not choose, so the row's only real value is its maximum --
    which is the proposal the tracker actually emitted, not an optimistic pick across three."""

    values = np.asarray(clip[field], dtype=float)
    if values.ndim == 1:
        return np.nan_to_num(values, nan=-np.inf)
    return np.nan_to_num(np.nanmax(values, axis=1), nan=-np.inf)


def predictions_for(clips, field):
    """One arm/confidence as a `Predictions`, plus the clip keys in the order they were added."""

    predictions = Predictions()
    for clip in clips:
        predictions.add(clip["samara"], per_frame(clip, field), clip["occluded"], clip["has_box"])
    return predictions


def sweep_grid(clips, field):
    """Thresholds at evenly spaced quantiles of this score, so the sweep covers its actual range."""

    values = np.concatenate([per_frame(clip, field) for clip in clips])
    values = values[np.isfinite(values)]
    if not len(values):
        return np.zeros(1)
    return np.unique(np.quantile(values, np.linspace(0.0, 1.0, SWEEP_STEPS)))


def occlusion_bins(clips):
    """`(labels, index_lists)` for quartiles of occluded DURATION in seconds."""

    seconds = np.array([clip["occ_count"] / FPS for clip in clips])
    edges = np.percentile(seconds, np.linspace(0, 100, BINS + 1)[1:-1])
    index = np.digitize(seconds, edges)

    lows = np.concatenate([[seconds.min()], edges])
    highs = np.concatenate([edges, [seconds.max()]])
    labels = [f"{low:.1f}–{high:.1f}s" for low, high in zip(lows, highs)]
    return labels, [np.flatnonzero(index == b) for b in range(BINS)]


def best_f(clips, field):
    """The protocol's statistic for one confidence on one set of clips: max F over the sweep."""

    if not len(clips):
        return np.nan
    return predictions_for(clips, field).best_f(sweep_grid(clips, field))[3]


def bootstrap_band(clips, field, threshold, seed=0):
    """A clustered 95% interval over VIDEOS, with the threshold fixed so the band is not inflated by
    re-tuning inside every resample."""

    if not len(clips):
        return np.nan, np.nan
    generator = np.random.default_rng(seed)
    videos = sorted({clip["video"] for clip in clips})
    by_video = {video: [clip for clip in clips if clip["video"] == video] for video in videos}

    draws = []
    for _ in range(BOOTSTRAP):
        picked = generator.choice(len(videos), len(videos))
        sample = [clip for index in picked for clip in by_video[videos[index]]]
        draws.append(predictions_for(sample, field).tracking_scores(threshold)[2])
    return tuple(np.percentile(draws, [2.5, 97.5]))


def figure(clips, destination):
    """F against occlusion duration, one line per confidence, with clustered bands."""

    labels, bin_indices = occlusion_bins(clips)
    positions = np.arange(BINS)

    figure, axes = plt.subplots(figsize=(6.0, 3.6), facecolor=SURFACE)
    axes.set_facecolor(SURFACE)

    for name, field in CONFIDENCES.items():
        if field not in clips[0]:
            continue

        values, lows, highs = [], [], []
        for indices in bin_indices:
            subset = [clips[i] for i in indices]
            threshold, _, _, f_score = predictions_for(subset, field).best_f(sweep_grid(subset, field))
            low, high = bootstrap_band(subset, field, threshold)
            values.append(f_score)
            lows.append(low)
            highs.append(high)

        colour = SERIES[name]
        axes.fill_between(positions, lows, highs, color=colour, alpha=0.12, linewidth=0)
        axes.plot(positions, values, color=colour, linewidth=2.0, marker="o", markersize=5,
                  markeredgecolor=SURFACE, markeredgewidth=1.2, label=name, zorder=3)
        axes.annotate(f"{values[-1]:.3f}", (positions[-1], values[-1]), textcoords="offset points",
                      xytext=(8, 0), color=colour, fontsize=8, va="center")

    axes.set_xticks(positions)
    axes.set_xticklabels(labels)
    axes.set_xlabel("time the target is occluded", color=INK2, fontsize=9)
    axes.set_ylabel("VOT-LT F-score", color=INK2, fontsize=9)
    axes.set_xlim(-0.25, BINS - 0.55)
    axes.grid(axis="y", color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axes.spines[spine].set_color(GRID)
    axes.tick_params(colors=INK2, labelsize=8, length=0)
    axes.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left")

    figure.tight_layout()
    figure.savefig(destination, dpi=220, facecolor=SURFACE)
    print(f"wrote {destination}")


def table(clips):
    """The same numbers as text, because a figure is not a result on its own."""

    labels, bin_indices = occlusion_bins(clips)
    print(f"{len(clips)} clips\n")
    print(f"{'confidence':18s}" + "".join(f"{label:>14s}" for label in labels) + f"{'ALL':>14s}")
    for name, field in CONFIDENCES.items():
        if field not in clips[0]:
            print(f"{name:18s}  not recorded in this run")
            continue
        cells = []
        for indices in list(bin_indices) + [np.arange(len(clips))]:
            subset = [clips[i] for i in indices]
            cells.append(predictions_for(subset, field).best_f(sweep_grid(subset, field))[3])
        print(f"{name:18s}" + "".join(f"{value:14.3f}" for value in cells))


def main(results_path="data/claim_5/p0.00/results.pkl"):
    clips = pickle.load(open(results_path, "rb"))["clips"]
    table(clips)

    # Beside the results it describes, so two arms' figures cannot overwrite each other.
    destination = Path(results_path).parent / "fig_votlt_by_occlusion.png"
    figure(clips, destination)


if __name__ == "__main__":
    main(*sys.argv[1:])
