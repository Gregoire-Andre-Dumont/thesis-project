"""claim_5 figure: VOT-LT F against occlusion duration, one line per corruption prior.

The prior is an ORDERED quantity, so the arms take a single-hue sequential ramp rather than categorical
colours -- light for a clean bank, dark for the heaviest corruption. That makes a monotone effect visible
as an ordered fan and its absence visible as tangled lines, which is the whole question here.

Each arm is swept to its own best threshold WITHIN each bin: a gate trained on more corruption emits
systematically lower probabilities (its pooled optimum runs 0.52 down to 0.40 as p rises), so comparing the
arms at a shared threshold would penalise the corrupted ones for their calibration rather than their
decisions.
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

FPS = 5.0
BINS = 4
BOOTSTRAP = 400
GRID = np.arange(0.0, 1.001, 0.02)

SURFACE, INK, INK2, GRIDLINE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3e0"
RAMP = ["#a1d99b", "#74c476", "#41ab5d", "#238b45", "#005a32"]     # light -> dark as the prior rises


def load(path):
    return {(clip["video"], clip["person"]): clip
            for clip in pickle.load(open(path, "rb"))["clips"]}


def confidence(clip):
    """The gate's p(commit) per frame -- the confidence the tracker actually deploys."""

    values = np.asarray(clip["commit_probability"], dtype=float)
    return np.nan_to_num(np.nanmax(values, axis=1) if values.ndim == 2 else values, nan=0.0)


def predictions_for(records, keys):
    predictions = Predictions()
    for key in keys:
        clip = records[key]
        predictions.add(clip["samara"], confidence(clip), clip["occluded"], clip["has_box"])
    return predictions


def band(records, keys, threshold, seed=0):
    """Clustered 95% interval over videos, threshold held fixed so it is not inflated by re-tuning."""

    generator = np.random.default_rng(seed)
    videos = sorted({records[key]["video"] for key in keys})
    by_video = {video: [key for key in keys if records[key]["video"] == video] for video in videos}

    draws = []
    for _ in range(BOOTSTRAP):
        picked = generator.choice(len(videos), len(videos))
        sample = [key for index in picked for key in by_video[videos[index]]]
        draws.append(predictions_for(records, sample).tracking_scores(threshold)[2])
    return np.percentile(draws, [2.5, 97.5])


def figure(arms, shared, seconds, destination):
    edges = np.percentile(seconds, np.linspace(0, 100, BINS + 1)[1:-1])
    index = np.digitize(seconds, edges)
    lows = np.concatenate([[seconds.min()], edges])
    highs = np.concatenate([edges, [seconds.max()]])
    labels = [f"{low:.1f}–{high:.1f}s" for low, high in zip(lows, highs)]
    positions = np.arange(BINS)

    plot, axes = plt.subplots(figsize=(6.2, 3.7), facecolor=SURFACE)
    axes.set_facecolor(SURFACE)

    for order, (name, records) in enumerate(arms.items()):
        colour = RAMP[order % len(RAMP)]
        values, lows_band, highs_band = [], [], []
        for bucket in range(BINS):
            keys = [shared[i] for i in np.flatnonzero(index == bucket)]
            threshold, _, _, f_score = predictions_for(records, keys).best_f(GRID)
            low, high = band(records, keys, threshold)
            values.append(f_score)
            lows_band.append(low)
            highs_band.append(high)

        axes.fill_between(positions, lows_band, highs_band, color=colour, alpha=0.10, linewidth=0)
        axes.plot(positions, values, color=colour, linewidth=2.0, marker="o", markersize=5,
                  markeredgecolor=SURFACE, markeredgewidth=1.2, label=name, zorder=3)

    axes.set_xticks(positions)
    axes.set_xticklabels(labels)
    axes.set_xlabel("time the target is occluded", color=INK2, fontsize=9)
    axes.set_ylabel("VOT-LT F-score", color=INK2, fontsize=9)
    axes.grid(axis="y", color=GRIDLINE, linewidth=0.8)
    axes.set_axisbelow(True)
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axes.spines[spine].set_color(GRIDLINE)
    axes.tick_params(colors=INK2, labelsize=8, length=0)
    axes.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left", ncol=2)

    plot.tight_layout()
    plot.savefig(destination, dpi=220, facecolor=SURFACE)
    print(f"wrote {destination}")
    return labels, index


def main(root="data/claim_5", destination="data/claim_5/fig_votlt_by_prior.png"):
    paths = sorted(Path(root).glob("*/results.pkl"))
    arms = {path.parent.name: load(path) for path in paths}
    shared = sorted(set.intersection(*(set(records) for records in arms.values())))
    seconds = np.array([arms[next(iter(arms))][key]["occ_count"] / FPS for key in shared])

    labels, index = figure(arms, shared, seconds, destination)

    print(f"\n{len(shared)} clips")
    print(f"{'arm':10s}" + "".join(f"{label:>13s}" for label in labels) + f"{'ALL':>13s}")
    for name, records in arms.items():
        cells = [predictions_for(records, [shared[i] for i in np.flatnonzero(index == bucket)]).best_f(GRID)[3]
                 for bucket in range(BINS)]
        cells.append(predictions_for(records, shared).best_f(GRID)[3])
        print(f"{name:10s}" + "".join(f"{value:13.3f}" for value in cells))


if __name__ == "__main__":
    main(*sys.argv[1:])
